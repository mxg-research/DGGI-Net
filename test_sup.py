import torch
import numpy as np


def test_DGGI_Net(model, mul_X_test, WE_test, args, device):
    model.eval()
    num_X_test = mul_X_test[0].shape[0]
    predictions = torch.zeros([num_X_test, args.Nlabel], device=device)
    index_array_test = np.arange(num_X_test)

    with torch.no_grad():
        for batch_idx in range(int(np.ceil(num_X_test / args.batch_size))):
            idx = index_array_test[
                batch_idx * args.batch_size:
                min((batch_idx + 1) * args.batch_size, num_X_test)
            ]

            mul_X_test_batch = [X[idx].to(device) for X in mul_X_test]
            we = WE_test[idx].to(device)

            _, target_pre, _, _, _, _, _, _, _ = model(
                mul_X_test_batch,
                we,
                mode='test',
                sigma=0
            )
            predictions[idx] = target_pre

    yy_pred = predictions.data.cpu().numpy()
    yy_pred = np.nan_to_num(yy_pred)
    return yy_pred
