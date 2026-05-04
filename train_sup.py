from torch.optim import SGD
import torch
import torch.nn as nn
from model import DGGINet
from loss import Loss
from pseudo_label import CAPLAPseudoLabeler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from test_sup import test_DGGI_Net
import copy
from measure import *


def train_DGGI_Net(mul_X, mul_X_test, WE, WE_test, label, yt_label, ind_00, obrT, device, args):
    yt_label = np.delete(yt_label, ind_00, axis=0)

    model = DGGINet(
        n_stacks=4,
        n_input=args.n_input,
        n_z=args.n_z,
        Nlabel=args.Nlabel
    ).to(device)

    loss_model = Loss(args.t, device)
    pseudo_label_model = CAPLAPseudoLabeler(device)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Module):
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight)
                    nn.init.constant_(mm.bias, 0.0)

    num_X = mul_X[0].shape[0]
    num_X_test = mul_X_test[0].shape[0]
    print(num_X, num_X_test)

    optimizer = SGD(model.parameters(), lr=args.lrkl, momentum=args.momentumkl)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)

    total_loss = 0.0
    ap_loss = []
    best_value_result = [0] * 10
    best_value_epoch = 0
    best_train_model = copy.deepcopy(model)

    pseudo_label_model.init_global_stats(label.shape[-1])
    pseudo_label_model.update_global_stats(label, obrT)

    global_pos_ratio = pseudo_label_model.get_global_pos_ratio()
    if not torch.is_tensor(global_pos_ratio):
        global_pos_ratio = torch.tensor(
            global_pos_ratio,
            dtype=torch.float32,
            device=device
        )
    else:
        global_pos_ratio = global_pos_ratio.float().to(device)

    for epoch in range(int(args.maxiter)):
        model.train()

        total_loss_last = total_loss
        total_loss = 0.0

        index_array = np.arange(num_X)
        if args.AE_shuffle:
            np.random.shuffle(index_array)

        progress = float(epoch) / float(max(1, args.maxiter - 1))
        progress = min(max(progress, 0.0), 1.0)

        eta1 = min(max(progress * float(args.alpha), 0.0), 1.0)
        eta0 = min(max(progress * float(args.beta), 0.0), 1.0)

        num_batches = int(np.ceil(num_X / args.batch_size))

        for batch_idx in range(num_batches):
            idx = index_array[
                batch_idx * args.batch_size:
                min((batch_idx + 1) * args.batch_size, num_X)
            ]

            mul_X_batch = [X[idx].to(device) for X in mul_X]
            we = WE[idx].to(device)

            sub_target = (label[idx] * obrT[idx]).to(device)
            sub_obrT = obrT[idx].to(device)

            optimizer.zero_grad()

            x_bar_list, target_pre, fusion_z, individual_zs, x_processed, S, new_x, global_adj, y_p = model(
                mul_X_batch,
                we,
                mode='train',
                sigma=0
            )

            with torch.no_grad():
                pseudo_labels = pseudo_label_model.generate_pseudo_labels(
                    y_p.detach(),
                    global_pos_ratio,
                    eta1,
                    eta0
                )

                if not torch.is_tensor(pseudo_labels):
                    pseudo_labels = torch.tensor(
                        pseudo_labels,
                        dtype=torch.float32,
                        device=device
                    )
                else:
                    pseudo_labels = pseudo_labels.float().to(device)

                pseudo_mask = (pseudo_labels != -1).float()
                pseudo_labels_clean = torch.where(
                    pseudo_labels == -1,
                    torch.zeros_like(pseudo_labels),
                    pseudo_labels
                )

                combined_labels = sub_target * sub_obrT + pseudo_labels_clean * (1 - sub_obrT)
                combined_mask = sub_obrT + pseudo_mask * (1 - sub_obrT)
                combined_mask = torch.clamp(combined_mask, 0.0, 1.0)

            is_completed = (we == 0).float()
            loss_Cont = torch.tensor(0.0, device=device)

            for i in range(len(individual_zs)):
                for j in range(i + 1, len(individual_zs)):
                    loss_Cont = loss_Cont + loss_model.contrast_loss(
                        individual_zs[i],
                        individual_zs[j],
                        we[:, i],
                        we[:, j],
                        is_completed[:, i],
                        is_completed[:, j]
                    )

            loss_CL = loss_model.weighted_CL_loss(
                combined_labels,
                target_pre,
                combined_mask
            )

            loss_CL1 = loss_model.weighted_CL_loss(
                sub_target,
                y_p,
                sub_obrT
            )

            ncr_loss = torch.tensor(0.0, device=device)
            for v in range(len(individual_zs)):
                view_features = individual_zs[v]
                view_adj = S[v]
                view_ncr = loss_model.cosine_loss(
                    view_features,
                    view_adj
                )
                ncr_loss = ncr_loss + view_ncr

            fusion_loss = (
                loss_CL
                + loss_CL1
                + args.lambda1 * loss_Cont
                + args.lambda2 * ncr_loss
            )

            total_loss += fusion_loss.item()

            fusion_loss.backward()
            optimizer.step()

        yp_prob = test_DGGI_Net(model, mul_X_test, WE_test, args, device)
        yp_prob = np.delete(yp_prob, ind_00, axis=0)

        value_result = do_metric(yp_prob, yt_label)
        ap_loss.append([value_result[4], total_loss])

        avg_loss = total_loss / max(1, num_batches)

        print(
            "DGGI-Net epoch {} loss={:.4f} hamming_loss={:.4f} AP={:.4f} "
            "AUC={:.4f} auc_me={:.4f} progress={:.4f} eta1={:.4f} eta0={:.4f} "
            "alpha={} beta={} lambda1={} lambda2={} temp={}"
            .format(
                epoch,
                avg_loss,
                value_result[0],
                value_result[4],
                value_result[5],
                value_result[6],
                progress,
                eta1,
                eta0,
                args.alpha,
                args.beta,
                args.lambda1,
                args.lambda2,
                args.t
            )
        )

        new_score = value_result[4] * 0.2 + value_result[3] * 0.4 + value_result[6] * 0.4
        best_score = best_value_result[4] * 0.2 + best_value_result[3] * 0.4 + best_value_result[6] * 0.4

        if new_score >= best_score:
            best_value_result = value_result
            best_train_model = copy.deepcopy(model)
            best_value_epoch = epoch

        del yp_prob

        if epoch > 100 and (
            (best_value_result[4] - value_result[4] > 0.03)
            or best_value_result[4] < args.min_AP
            or (abs(total_loss_last - avg_loss) < 1e-7)
        ):
            print(
                'DGGI-Net training stopped: epoch=%d, best_epoch=%d, best_AP=%.7f, min_AP=%.7f, total_loss=%.7f'
                % (
                    epoch,
                    best_value_epoch,
                    best_value_result[4],
                    args.min_AP,
                    avg_loss
                )
            )
            break

        total_loss = avg_loss

    return best_train_model, best_value_result, ap_loss
