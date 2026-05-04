import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class encoder(nn.Module):
    def __init__(self, n_dim, dims, n_z, dropout_prob=0.2):
        super(encoder, self).__init__()
        self.enc_layers = nn.Sequential(
            nn.Linear(n_dim, dims[0]),
            nn.BatchNorm1d(dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(dims[0], dims[1]),
            nn.BatchNorm1d(dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(dims[1], dims[2]),
            nn.BatchNorm1d(dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        self.z_layer = nn.Linear(dims[2], n_z)
        self.z_b0 = nn.BatchNorm1d(n_z)

    def forward(self, x):
        x = self.enc_layers(x)
        z = self.z_b0(self.z_layer(x))
        return z

class decoder(nn.Module):
    def __init__(self, n_dim, dims, n_z, dropout_prob=0.2):
        super(decoder, self).__init__()
        self.dec_layers = nn.Sequential(
            nn.Linear(n_z, n_z),
            nn.BatchNorm1d(n_z),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(n_z, dims[2]),
            nn.BatchNorm1d(dims[2]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(dims[2], dims[1]),
            nn.BatchNorm1d(dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),

            nn.Linear(dims[1], dims[0]),
            nn.BatchNorm1d(dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        self.x_bar_layer = nn.Linear(dims[0], n_dim)

    def forward(self, z):
        z = self.dec_layers(z)
        x_bar = self.x_bar_layer(z)
        return x_bar


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, x, adj):

        support = torch.mm(x, self.weight)  # [batch_size, out_features]

        # Convert sparse adjacency to dense format when needed.
        adj = adj.to_dense() if adj.is_sparse else adj

        # Row-normalize the adjacency matrix before graph propagation.
        adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-9)
        out = torch.mm(adj, support)
        if self.bias is not None:
            out += self.bias
        return out



class GCNCompletion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gc1 = GraphConvolution(hidden_dim, 1500)
        self.l1 = nn.Linear(1500, hidden_dim)
    def forward(self, x, adj_v, mask_v):
        device = x.device
        adj_v = adj_v.to(device)
        mask_v = mask_v.to(device)

        # Apply graph convolution for view completion.
        x_v = self.gc1(x, adj_v)
        x_v = self.l1(x_v)
        diag_mask_v = torch.diag(mask_v).to(device)
        diag_inv_mask_v = torch.diag(1 - mask_v).to(device)
        x_v = diag_mask_v.mm(x) + diag_inv_mask_v.mm(x_v)
        return x_v

class AE(nn.Module):

    def __init__(self, n_stacks, n_input, n_z, nLabel):
        super(AE, self).__init__()

        dims = []
        for n_dim in n_input:

            linshidims = []
            for idim in range(n_stacks - 2):
                linshidim = round(n_dim * 0.8)
                linshidim = int(linshidim)
                linshidims.append(linshidim)
            linshidims.append(1500)
            dims.append(linshidims)

            # View-completion modules are defined independently for each view.
            self.view_completion = nn.ModuleList([
                GCNCompletion(dim)
                for dim in n_input
            ])

        self.encoder_list = nn.ModuleList([encoder(n_input[i], dims[i], n_z) for i in range(len(n_input))])
        self.k = 7
        self.regression = nn.Sequential(
            nn.Linear(n_z, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(512, nLabel),
        )
        self.regression2 = nn.Sequential(
            nn.Linear(n_z, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(512, nLabel),
        )
        self.act = nn.Sigmoid()

    def build_adj(self, mul_x, mask, k):
        """
        Build graph structures corresponding to Eqs. (1)-(3) in the manuscript.

        Args:
            mul_x: A list of multi-view feature tensors. Each tensor has shape
                [batch_size, d_v].
            mask: View-availability mask with shape [batch_size, num_views],
                where M_{i,v}=1 indicates that the v-th view of the i-th sample
                is observed.
            k: Number of nearest neighbors used for sparse graph-guided completion.

        Returns:
            G: kNN adjacency matrices for GCN completion with shape
                [num_views, B, B]. For view v, missing samples are connected only
                to available samples in the same view.
            consensus: Cross-view consensus similarity matrix with shape [B, B].
            S_tilde: Extended similarity matrices with shape [num_views, B, B].
        """
        num_views = len(mul_x)
        batch_size = mask.size(0)
        device = mask.device
        dtype = mul_x[0].dtype

        mask = mask.float()

        # S_raw[v] corresponds to S^{(v)} in Eq. (1).
        S_raw = torch.zeros(num_views, batch_size, batch_size, device=device, dtype=dtype)

        # Accumulators for the cross-view consensus similarity.
        # consensus_ij = sum_u M_iu M_ju S_ij^(u) / sum_u M_iu M_ju
        sum_sim = torch.zeros(batch_size, batch_size, device=device, dtype=dtype)
        shared_count = torch.zeros(batch_size, batch_size, device=device, dtype=dtype)

        # Step 1: Construct the raw intra-view similarity matrix S^{(v)}.
        for v in range(num_views):
            x_v = mul_x[v]

            # Cosine similarity.
            x_v_norm = F.normalize(x_v, p=2, dim=1, eps=1e-12)
            sim_v = torch.mm(x_v_norm, x_v_norm.t())

            # Keep S^{(v)}_{ij} only when both samples are observed in view v.
            visible_pair_v = mask[:, v].unsqueeze(1) * mask[:, v].unsqueeze(0)

            sim_v = sim_v * visible_pair_v
            sim_v.fill_diagonal_(0.0)

            S_raw[v] = sim_v

            sum_sim += sim_v
            shared_count += visible_pair_v

        # Step 2: Construct the cross-view consensus similarity.
        # This term provides the missing-position completion source in Eq. (2).
        eps = 1e-8
        consensus = sum_sim / (shared_count + eps)
        consensus = torch.where(shared_count > 0, consensus, torch.zeros_like(consensus))
        consensus.fill_diagonal_(0.0)

        # Step 3: Build the extended similarity matrix \\tilde{S}^{(v)}.
        S_tilde = torch.zeros_like(S_raw)

        for v in range(num_views):
            visible_pair_v = mask[:, v].unsqueeze(1) * mask[:, v].unsqueeze(0)

            # Use S^{(v)}_{ij} when both samples are observed in view v;
            # otherwise use the cross-view consensus similarity.
            S_tilde[v] = visible_pair_v * S_raw[v] + (1.0 - visible_pair_v) * consensus
            S_tilde[v].fill_diagonal_(0.0)

        # Step 4: Build the kNN graph for GCN completion from \\tilde{S}^{(v)}.
        # Eq. (3): each missing sample aggregates only available neighbors in view v.
        G = torch.zeros_like(S_tilde)

        k_eff = min(k, batch_size)

        for v in range(num_views):
            # Rows indicate samples missing in the current view.
            row_missing_v = (1.0 - mask[:, v]).unsqueeze(1)  # [B, 1]

            # Columns indicate candidate neighbors observed in the current view.
            col_available_v = mask[:, v].unsqueeze(0)  # [1, B]

            # Allow only missing-to-available edges.
            candidate_mask = (row_missing_v * col_available_v).bool()

            # Use the extended similarity matrix of the current view.
            scores = S_tilde[v].clone()

            # Exclude non-candidate positions from top-k selection.
            scores = scores.masked_fill(~candidate_mask, float("-inf"))

            values, indices = torch.topk(scores, k=k_eff, dim=-1)

            # Replace -inf values with 0 when a row has no available candidate.
            values = torch.where(torch.isfinite(values), values, torch.zeros_like(values))

            v_adj = torch.zeros_like(scores)
            v_adj.scatter_(dim=-1, index=indices, src=values)

            # Keep only valid missing-to-available edges.
            v_adj = v_adj * candidate_mask.float()

            # Row normalization is performed in GraphConvolution.forward.
            G[v] = v_adj

        return G, consensus, S_tilde

    def forward(self, mul_X, we, mode, sigma):
        # Build the top-k missing-to-available graph and the extended similarity matrices.
        gcn_adj, consensus_adj, S_tilde = self.build_adj(mul_X, we, self.k)

        x_processed = []
        for v, gcn in enumerate(self.view_completion):
            # Use the sparse GCN adjacency for view completion.
            x_v = gcn(mul_X[v], gcn_adj[v], we[:, v])
            x_processed.append(x_v)

        individual_zs = []
        summ = 0
        summ_p = 0
        we_p = 1 - we

        for enc_i, enc in enumerate(self.encoder_list):
            z_i = enc(x_processed[enc_i])
            individual_zs.append(z_i)

            summ += torch.diag(we[:, enc_i]).mm(z_i)
            summ_p += torch.diag(we_p[:, enc_i]).mm(z_i)

        wei = 1 / (torch.sum(we, 1) + 1e-9)
        wei_p = 1 / (torch.sum(we_p, 1) + 1e-9)

        z = torch.diag(wei).mm(summ)
        z_p = torch.diag(wei_p).mm(summ_p)

        new_x = torch.stack(individual_zs, dim=0)
        x_bar_list = []

        yLable = self.act(self.regression(z))
        y_p = self.act(self.regression2(z_p))

        return x_bar_list, yLable, z, individual_zs, x_processed, S_tilde, new_x, consensus_adj, y_p

class DGGINet(nn.Module):
    def __init__(self,
                 n_stacks,
                 n_input,
                 n_z,
                 Nlabel):
        super(DGGINet, self).__init__()

        self.ae = AE(
            n_stacks=n_stacks,
            n_input=n_input,
            n_z=n_z,
            nLabel=Nlabel)

    def forward(self, mul_X, we, mode, sigma):
        x_bar_list, target_pre, fusion_z, individual_zs, x_processed, adj, new_x, global_adj, y_p = self.ae(
            mul_X, we, mode, sigma
        )
        return x_bar_list, target_pre, fusion_z, individual_zs, x_processed, adj, new_x, global_adj, y_p