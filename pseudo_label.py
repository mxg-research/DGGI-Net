import torch
import torch.nn as nn


class CAPLAPseudoLabeler(nn.Module):
    def __init__(self, device):
        super(CAPLAPseudoLabeler, self).__init__()
        self.device = device
        self.global_pos_count = None
        self.global_total_count = None

    def init_global_stats(self, num_classes):
        self.global_pos_count = torch.zeros(num_classes, device=self.device)
        self.global_total_count = torch.zeros(num_classes, device=self.device) + 1e-8

    def update_global_stats(self, labels, mask):
        labels = labels.to(self.device).float()
        mask = mask.to(self.device).float()

        valid_labels = labels * mask
        batch_pos = valid_labels.sum(dim=0)
        batch_total = mask.sum(dim=0)

        self.global_pos_count += batch_pos
        self.global_total_count += batch_total

    def get_global_pos_ratio(self):
        return self.global_pos_count / self.global_total_count

    def generate_pseudo_labels(self, pred_v, pos_ratio, eta1=0.4, eta0=0.4):
        """
        Generate category-adaptive pseudo-labels.

        Args:
            pred_v: Low-quality path predictions p^(l) with shape [B, C].
            pos_ratio: Global positive ratio gamma_k with shape [C].
            eta1: Relaxation coefficient for positive pseudo-label selection.
            eta0: Relaxation coefficient for negative pseudo-label selection.

        Returns:
            pseudo_labels: Tensor with shape [B, C], where values are 1, 0, or -1.
        """
        pred_v = pred_v.detach().to(self.device)
        batch_size, num_classes = pred_v.shape

        pos_ratio = pos_ratio.to(self.device).float()
        neg_ratio = 1.0 - pos_ratio

        pseudo_labels = torch.full_like(pred_v, -1.0)

        for k in range(num_classes):
            gamma_k = pos_ratio[k].clamp(0.0, 1.0)
            rho_k = neg_ratio[k].clamp(0.0, 1.0)

            num_pos = int(torch.floor(eta1 * gamma_k * batch_size).item())
            num_neg = int(torch.floor(eta0 * rho_k * batch_size).item())

            num_pos = max(0, min(num_pos, batch_size))
            num_neg = max(0, min(num_neg, batch_size - num_pos))

            scores = pred_v[:, k]

            if num_pos > 0:
                _, pos_indices = torch.topk(scores, k=num_pos, largest=True)
                pseudo_labels[pos_indices, k] = 1.0

            if num_neg > 0:
                _, neg_indices = torch.topk(scores, k=num_neg, largest=False)
                unassigned = pseudo_labels[neg_indices, k] == -1
                neg_indices = neg_indices[unassigned]
                pseudo_labels[neg_indices, k] = 0.0

        return pseudo_labels


# Backward-compatible alias. This can be removed if all scripts import CAPLAPseudoLabeler directly.
Pseudo_label = CAPLAPseudoLabeler
