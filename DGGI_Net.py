# -*- coding: utf-8 -*-
from __future__ import print_function, division

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, scale
import scipy.io
import math
import copy
from loss import Loss
from train_sup import train_DGGI_Net
from test_sup import test_DGGI_Net
from measure import *
import time


def setup_seed(seed=0):
    import torch
    import os
    import numpy as np
    import random

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def filterparam(file_path):
    """
    Load completed hyperparameter combinations to avoid repeated runs.

    The last seven fields in the result file are:
    lrkl momentumKL alpha beta lambda1 lambda2 temp
    """
    params = set()

    if os.path.exists(file_path):
        with open(file_path, mode='r', encoding='utf-8') as file_handle:
            lines = file_handle.readlines()

        lines = lines[1:] if len(lines) > 1 else []

        for line in lines:
            parts = line.strip().split()

            if len(parts) < 7:
                continue

            try:
                param_key = tuple(float(x) for x in parts[-7:])
                params.add(param_key)
            except Exception:
                continue

    return params


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Train DGGI-Net',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--Nlabel', default=7, type=int)
    parser.add_argument('--maxiter', default=100, type=int)
    parser.add_argument('--batch_size', default=128, type=int)

    parser.add_argument('--dataset', type=str, default='corel5k' + '_six_view')
    parser.add_argument('--dataPath', type=str, default='./data/corel5k')
    parser.add_argument('--n_z', default=256, type=int)

    parser.add_argument('--MaskRatios', type=float, default=0.5)
    parser.add_argument('--LabelMaskRatio', type=float, default=0.5)
    parser.add_argument('--TraindataRatio', type=float, default=0.7)

    parser.add_argument('--AE_shuffle', type=bool, default=True)
    parser.add_argument('--min_AP', default=0.20, type=float)
    parser.add_argument('--tol', default=1e-7, type=float)

    # Relaxation coefficients for CAPLA pseudo-label generation.
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--beta', default=0.1, type=float)

    # Temperature coefficient tau for contrastive learning.
    parser.add_argument('--t', default=0.1, type=float)

    # Loss weights: lambda1 controls Lcon, and lambda2 controls Lner.
    parser.add_argument('--lambda1', default=0.005, type=float)
    parser.add_argument('--lambda2', default=0.0005, type=float)

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))

    file_path = './DGGI-Net-' + args.dataset + '_nz_' + str(args.n_z) + '_VMR_' + str(
        args.MaskRatios) + '_LMR_' + str(args.LabelMaskRatio) + '_TR_' + str(
        args.TraindataRatio) + '-best_AP_param_search' + '.txt'

    existed_params = filterparam(file_path)

    device = torch.device("cuda" if args.cuda else "cpu")

    Pre_fnum = 10

    # SGD momentum.
    pre_momae = [0.9]

    # Learning rate.
    pre_lrkl = [1]

    # CAPLA relaxation coefficients.
    pre_alpha = [0.2]
    pre_beta = [0.3]

    # Temperature coefficient tau for contrastive learning.
    pre_t = [0.1]

    # Loss weights.
    pre_lambda1 = [0.005]
    pre_lambda2 = [0.001]

    # pre_alpha = [0.1, 0.2, 0.3, 0.4, 0.5]
    # pre_beta = [0.1, 0.2, 0.3, 0.4, 0.5]
    # pre_t = [0.01, 0.05, 0.1, 0.2, 0.3]
    # pre_lambda1 = [0.001, 0.005, 0.01]
    # pre_lambda2 = [0.0005, 0.001, 0.005]

    """
    Example dataset-specific loss-weight settings used during hyperparameter tuning.
    """

    best_AUC_me = 0
    best_AUC_mac = 0
    best_AP = 0

    data = scipy.io.loadmat(args.dataPath + '/' + args.dataset + '.mat')
    X = data['X'][0]
    view_num = X.shape[0]

    label = data['label']
    label = np.array(label, 'float32')

    for momae in pre_momae:
        args.momentumkl = momae

        for lrkl in pre_lrkl:
            args.lrkl = lrkl

            for alpha in pre_alpha:
                args.alpha = alpha

                for beta in pre_beta:
                    args.beta = beta

                    for t in pre_t:
                        args.t = t

                        for lambda1 in pre_lambda1:
                            args.lambda1 = lambda1

                            for lambda2 in pre_lambda2:
                                args.lambda2 = lambda2

                                if args.lrkl >= 0.01:
                                    args.momentumkl = 0.90

                                param_key = (
                                    float(args.lrkl),
                                    float(args.momentumkl),
                                    float(args.alpha),
                                    float(args.beta),
                                    float(args.lambda1),
                                    float(args.lambda2),
                                    float(args.t)
                                )

                                if param_key in existed_params:
                                    print(
                                        'existed param! lr:{} momentum:{} alpha:{} beta:{} lambda1:{} lambda2:{} temp:{}'
                                        .format(
                                            args.lrkl,
                                            args.momentumkl,
                                            args.alpha,
                                            args.beta,
                                            args.lambda1,
                                            args.lambda2,
                                            args.t
                                        )
                                    )
                                    continue

                                print(args)

                                hm_loss = np.zeros(Pre_fnum)
                                one_error = np.zeros(Pre_fnum)
                                coverage = np.zeros(Pre_fnum)
                                rk_loss = np.zeros(Pre_fnum)
                                AP_score = np.zeros(Pre_fnum)

                                mac_auc = np.zeros(Pre_fnum)
                                auc_me = np.zeros(Pre_fnum)
                                mac_f1 = np.zeros(Pre_fnum)
                                mic_f1 = np.zeros(Pre_fnum)

                                for fnum in range(Pre_fnum):
                                    setup_seed(43)

                                    mul_X = [None] * view_num

                                    datafold = scipy.io.loadmat(
                                        args.dataPath + '/' + args.dataset +
                                        '_MaskRatios_' + str(args.MaskRatios) +
                                        '_LabelMaskRatio_' + str(args.LabelMaskRatio) +
                                        '_TraindataRatio_' + str(args.TraindataRatio) +
                                        '.mat'
                                    )

                                    folds_data = datafold['folds_data']
                                    folds_label = datafold['folds_label']
                                    folds_sample_index = datafold['folds_sample_index']

                                    del datafold

                                    Ndata, args.Nlabel = label.shape

                                    indexperm = np.array(folds_sample_index[0, fnum], 'int32')

                                    train_num = math.ceil(Ndata * args.TraindataRatio)
                                    train_index = indexperm[0, 0:train_num] - 1

                                    remain_num = Ndata - train_num
                                    val_num = math.ceil(remain_num * 0.5)

                                    print('val_num', val_num)
                                    print('train_num', train_num)

                                    val_index = indexperm[0, train_num:train_num + val_num] - 1
                                    rtest_index = indexperm[0, train_num + val_num:indexperm.shape[1]] - 1

                                    WE = np.array(folds_data[0, fnum], 'int32')
                                    obrT = np.array(folds_label[0, fnum], 'int32')

                                    if label.min() == -1:
                                        label = (label + 1) * 0.5

                                    Inc_label = label * obrT
                                    fan_Inc_label = 1 - Inc_label

                                    for iv in range(view_num):
                                        mul_X[iv] = np.copy(X[iv])
                                        mul_X[iv] = mul_X[iv].astype(np.float32)

                                        WEiv = WE[:, iv]

                                        ind_1 = np.where(WEiv == 1)
                                        ind_1 = np.array(ind_1).reshape(-1)

                                        ind_0 = np.where(WEiv == 0)
                                        ind_0 = np.array(ind_0).reshape(-1)

                                        mul_X[iv][ind_1, :] = StandardScaler().fit_transform(
                                            mul_X[iv][ind_1, :]
                                        )

                                        mul_X[iv][ind_0, :] = 0

                                        clum = abs(mul_X[iv]).sum(0)
                                        ind_11 = np.array(np.where(clum != 0)).reshape(-1)

                                        new_X = np.copy(mul_X[iv][:, ind_11])
                                        mul_X[iv] = torch.Tensor(np.nan_to_num(np.copy(new_X)))

                                        del new_X, ind_0, ind_1, ind_11, clum

                                    WE = torch.Tensor(WE)
                                    obrT = torch.Tensor(obrT)

                                    mul_X_val = [xiv[val_index] for xiv in mul_X]
                                    mul_X_rtest = [xiv[rtest_index] for xiv in mul_X]
                                    mul_X_train = [xiv[train_index] for xiv in mul_X]

                                    WE_val = WE[val_index]
                                    WE_rtest = WE[rtest_index]
                                    WE_train = WE[train_index]

                                    args.n_input = [xiv.shape[1] for xiv in mul_X]

                                    yv_label = np.copy(label[val_index])
                                    yrt_label = np.copy(label[rtest_index])

                                    train_label = torch.Tensor(label[train_index])
                                    train_obrT = obrT[train_index].clone().float()

                                    ind_00_val = np.array(
                                        np.where(abs(yv_label).sum(1) == 0)
                                    ).reshape(-1)

                                    ind_00_test = np.array(
                                        np.where(abs(yrt_label).sum(1) == 0)
                                    ).reshape(-1)

                                    model, value_result, all_results = train_DGGI_Net(
                                        mul_X_train,
                                        mul_X_val,
                                        WE_train,
                                        WE_val,
                                        train_label,
                                        yv_label,
                                        ind_00_val,
                                        train_obrT,
                                        device,
                                        args
                                    )

                                    yp_prob = test_DGGI_Net(
                                        model,
                                        mul_X_rtest,
                                        WE_rtest,
                                        args,
                                        device
                                    )

                                    value_result = do_metric(yp_prob, yrt_label)

                                    print(
                                        "final:hamming-loss one-error coverage ranking-loss "
                                        "average-precision macro-auc auc_me macro_f1 micro_f1"
                                    )
                                    print(value_result)

                                    hm_loss[fnum] = value_result[0]
                                    one_error[fnum] = value_result[1]
                                    coverage[fnum] = value_result[2]
                                    rk_loss[fnum] = value_result[3]
                                    AP_score[fnum] = value_result[4]
                                    mac_auc[fnum] = value_result[5]
                                    auc_me[fnum] = value_result[6]
                                    mac_f1[fnum] = value_result[7]
                                    mic_f1[fnum] = value_result[8]

                                if AP_score.mean() > best_AP:
                                    best_AP = AP_score.mean()

                                with open(file_path, mode='a', encoding='utf-8') as file_handle:
                                    if os.path.getsize(file_path) == 0:
                                        file_handle.write(
                                            'mean_AP std_AP '
                                            'mean_hamming_loss std_hamming_loss '
                                            'mean_ranking_loss std_ranking_loss '
                                            'mean_AUCme std_AUCme '
                                            'mean_one_error std_one_error '
                                            'mean_coverage std_coverage '
                                            'mean_macAUC std_macAUC '
                                            'mean_macro_f1 std_macro_f1 '
                                            'mean_micro_f1 std_micro_f1 '
                                            'lrkl momentumKL alpha beta lambda1 lambda2 temp\n'
                                        )

                                    file_handle.write(
                                        str(round(AP_score.mean(), 4)) + ' ' +
                                        str(round(AP_score.std(), 4)) + ' ' +

                                        str(round(hm_loss.mean(), 4)) + ' ' +
                                        str(round(hm_loss.std(), 4)) + ' ' +

                                        str(round(rk_loss.mean(), 4)) + ' ' +
                                        str(round(rk_loss.std(), 4)) + ' ' +

                                        str(round(auc_me.mean(), 4)) + ' ' +
                                        str(round(auc_me.std(), 4)) + ' ' +

                                        str(round(one_error.mean(), 4)) + ' ' +
                                        str(round(one_error.std(), 4)) + ' ' +

                                        str(round(coverage.mean(), 4)) + ' ' +
                                        str(round(coverage.std(), 4)) + ' ' +

                                        str(round(mac_auc.mean(), 4)) + ' ' +
                                        str(round(mac_auc.std(), 4)) + ' ' +

                                        str(round(mac_f1.mean(), 4)) + ' ' +
                                        str(round(mac_f1.std(), 4)) + ' ' +

                                        str(round(mic_f1.mean(), 4)) + ' ' +
                                        str(round(mic_f1.std(), 4)) + ' ' +

                                        str(args.lrkl) + ' ' +
                                        str(args.momentumkl) + ' ' +
                                        str(args.alpha) + ' ' +
                                        str(args.beta) + ' ' +
                                        str(args.lambda1) + ' ' +
                                        str(args.lambda2) + ' ' +
                                        str(args.t)
                                    )

                                    file_handle.write('\n')