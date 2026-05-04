import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, t, device):
        super(Loss, self).__init__()
        self.t = t
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def contrast_loss(self, v1, v2, we1, we2, is_completed1, is_completed2):
        mask = (
            ((is_completed1 == 1) & (we2 == 1)) |
            ((is_completed2 == 1) & (we1 == 1)) |
            ((we1 == 1) & (we2 == 1))
        )
        mask = mask.bool()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)

        v1 = v1[mask]
        v2 = v2[mask]
        n = v1.size(0)
        N = 2 * n
        if n == 0:
            return torch.tensor(0.0, device=self.device)

        v1 = F.normalize(v1, p=2, dim=1)
        v2 = F.normalize(v2, p=2, dim=1)
        z = torch.cat((v1, v2), dim=0)
        similarity_mat = torch.matmul(z, z.T) / self.t
        similarity_mat = similarity_mat.fill_diagonal_(0)
        label = torch.cat((torch.arange(n, N), torch.arange(0, n))).to(self.device)
        loss = self.criterion(similarity_mat, label)
        return loss / N

    def cosine_loss(self, h, adj):
        N = h.size(0)
        q = F.normalize(h, p=2, dim=1)
        similarity_mat = torch.matmul(q, q.T)
        similarity_mat = similarity_mat.fill_diagonal_(0)
        adj = (adj + adj.T) / 2
        similarity_flat = similarity_mat.view(-1)
        adj_flat = adj.view(-1)
        loss = 1 - F.cosine_similarity(similarity_flat.unsqueeze(0), adj_flat.unsqueeze(0))
        return loss * N

    def wmse_loss(self, input, target, reduction='mean'):
        ret = (target - input) ** 2
        ret = torch.mean(ret)
        return ret

    def weighted_CL_loss(self, sub_target, target_pre, sub_obrT):
        loss = torch.abs(
            (
                sub_target.mul(torch.log(target_pre + 1e-10)) +
                (1 - sub_target).mul(torch.log(1 - target_pre + 1e-10))
            ).mul(sub_obrT)
        )
        return torch.mean(loss)
