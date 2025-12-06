# -*- coding: utf-8 -*-
# scene_embedding/spatial_ego_rgat.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ensure_long(x: torch.Tensor) -> torch.Tensor:
    return x if (torch.is_tensor(x) and x.dtype == torch.long) else x.to(torch.long)


class EgoRelGATLayer(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 num_heads: int,
                 num_edge_types: int,
                 edge_emb_dim: int = 32,
                 use_geo_bias: bool = True,
                 geo_dim: int = 3):
        super().__init__()
        assert out_dim % num_heads == 0
        self.H = int(num_heads)
        self.D = out_dim // num_heads
        self.use_geo = bool(use_geo_bias)
        self.geo_dim = int(geo_dim)
        self.num_edge_types = int(num_edge_types)  # 复合关系类别总数（例如 11*16）

        # 多头投影
        self.W_q = nn.Linear(in_dim,  out_dim, bias=False)
        self.W_k = nn.Linear(in_dim,  out_dim, bias=False)
        self.W_v = nn.Linear(in_dim,  out_dim, bias=False)
        self.W_o = nn.Linear(out_dim, out_dim, bias=False)

        # 关系类型嵌入
        self.edge_emb = nn.Embedding(self.num_edge_types, edge_emb_dim)
        self.W_e = nn.Linear(edge_emb_dim, out_dim, bias=False)

        # 几何偏置：投到 H 个注意力头
        if self.use_geo:
            self.geo_proj = nn.Linear(self.geo_dim, self.H, bias=False)

        # 归一化
        self.ln = nn.LayerNorm(in_dim)

    # ------------------------------------------------------------------
    def forward(self,
                X: torch.Tensor,
                A: torch.Tensor,
                E: torch.Tensor,
                geo_bias: torch.Tensor | None = None):
        """
        参数
        ----
        X: (N, Fin)
        A: (N, N) 邻接（0 行是 ego → 邻居），或 (N,N) 的 0/1 浮点均可
        E: (N, N) 或 (N,) 或与邻居一一对应的 (M,) / (M, C_onehot)
           - 值域：-1 表示无标签（将被剔除），其它将 clamp 到 [0, C-1]
           - 若是 one-hot，则先 argmax
        geo_bias: None 或 (M, *)，与“有效邻居”一一对应；
                  若最后维 != H，会投影到 (M, H)

        返回
        ----
        out    : (out_dim,)    ego 的聚合表示
        c_type : (C, out_dim)  各关系类型通道的聚合（没有该类型邻居则为 0）
        """
        device = X.device
        X = self.ln(X)
        ego = X[0]                              # (Fin,)
        N = X.size(0)
        C = self.num_edge_types
        out_dim = self.H * self.D

        # ---------- 取 ego 的邻居 ----------
        # 支持 A 为 float/bool/int；>0 视为连边
        if A.dim() != 2 or A.size(0) != A.size(1) or A.size(0) != N:
            raise ValueError("A must be a square (N, N) adjacency matrix.")
        nbr_mask = (A[0] > 0)
        if nbr_mask.dim() == 0:
            nbr_mask = nbr_mask.unsqueeze(0)
        # 去自环
        if nbr_mask.numel() >= 1:
            nbr_mask[0] = False
        nbr_idx = torch.nonzero(nbr_mask, as_tuple=False).squeeze(1)  # (M0,)
        M0 = int(nbr_idx.numel())

        # 无邻居：直接返回
        if M0 == 0:
            return ego, X.new_zeros(C, out_dim)

        # ---------- 归一化/对齐边类型到邻居 ----------
        # 统一得到邻接一行的类型向量 e_row (N,)
        # E 可能是 (N,N) / (N,) / (M,) / one-hot；统一成邻居行再索引
        if E is None:
            e_row = torch.full((N,), 0, dtype=torch.long, device=device)  # 无标签时默认 0 类
        else:
            # to device
            E = E.to(device)
            # one-hot -> index
            if E.dim() == 2 and E.size(0) == N and E.size(1) == C:
                e_row = E.argmax(dim=-1).to(torch.long)                   # (N,)
            elif E.dim() == 2 and E.size(0) == N and E.size(1) == N:
                e_row = E[0]                                             # (N,)
            elif E.dim() == 1 and E.size(0) == N:
                e_row = _ensure_long(E)
            else:
                # 其它情况（例如直接给了邻居上的 (M,)），优先尝试匹配邻居长度
                if E.dim() == 1 and E.size(0) == M0:
                    # 先构造 e_row，全 -1（表示无效），再把邻居位置填上
                    e_row = torch.full((N,), -1, dtype=torch.long, device=device)
                    e_row[nbr_idx] = _ensure_long(E)
                else:
                    # 兜底：按 (N,) 处理
                    e_row = torch.full((N,), -1, dtype=torch.long, device=device)

            # 清洗：-1 为无标签（剔除），其它 clamp 到 [0, C-1]
            e_row = torch.nan_to_num(e_row.float(), nan=-1.0, posinf=-1.0, neginf=-1.0)
            # 仅对非负项做 round+clamp
            pos_mask = e_row >= 0
            e_row[pos_mask] = e_row[pos_mask].round().clamp_(0, C - 1)
            e_row = e_row.to(torch.long)  # (N,)

        # 仅保留同时满足“是邻居且有合法类型”的节点
        valid_mask_on_N = nbr_mask & (e_row >= 0)
        nbr_idx = torch.nonzero(valid_mask_on_N, as_tuple=False).squeeze(1)  # (M,)
        M = int(nbr_idx.numel())

        if M == 0:
            # 邻居都被 -1 筛掉了
            return ego, X.new_zeros(C, out_dim)

        # 邻居特征 & 边类型（已对齐）
        x_nbr = X.index_select(0, nbr_idx)             # (M, Fin)
        e_type = e_row.index_select(0, nbr_idx)        # (M,)
        # 再次保险：clamp 到合法范围
        e_type = torch.nan_to_num(e_type.float(), nan=0.0).round().clamp_(0, C - 1).to(torch.long)

        # 关系嵌入
        e_emb = self.edge_emb(e_type)                  # (M, edge_emb_dim)

        # ---------- 多头注意力 ----------
        q = self.W_q(ego).view(self.H, self.D)                     # (H, D)
        k = (self.W_k(x_nbr) + self.W_e(e_emb)).view(M, self.H, self.D)  # (M, H, D)
        v = self.W_v(x_nbr).view(M, self.H, self.D)                # (M, H, D)

        logits = torch.einsum("hd,mhd->mh", q, k) / (self.D ** 0.5)  # (M, H)

        # ---------- 几何偏置 ----------
        if self.use_geo and (geo_bias is not None):
            gb = geo_bias
            # 可能是 list/tensor；需要与“有效邻居”对齐长度 M
            if isinstance(gb, (list, tuple)):
                gb = gb[0] if len(gb) > 0 else None
            if torch.is_tensor(gb):
                gb = gb.to(device)
                # 若原始几何是基于“ego→邻居”的顺序构造的（与 nbr_idx 同顺序），此处只需检查长度：
                if gb.dim() == 1:
                    gb = gb.unsqueeze(-1)                 # (M, 1)
                # 若长度不匹配，做容错裁剪/补零
                if gb.size(0) != M:
                    if gb.size(0) > M:
                        gb = gb[:M]
                    else:
                        pad = torch.zeros(M - gb.size(0), gb.size(1), device=device, dtype=gb.dtype)
                        gb = torch.cat([gb, pad], dim=0)
                # 投到 (M, H)
                if gb.size(-1) != self.H:
                    gb = self.geo_proj(gb)                # (M, H)
                logits = logits + gb

        # 数值保护
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)
        alpha = torch.softmax(logits, dim=0)                              # (M, H)

        # ---------- 聚合 ----------
        m = torch.einsum("mh,mhd->hd", alpha, v).reshape(-1)              # (out_dim,)
        out = ego + self.W_o(m)                                           # 残差

        # ---------- 各类型通道聚合 ----------
        c_list = []
        for t in range(C):
            idx_t = torch.nonzero(e_type == t, as_tuple=False).squeeze(1)  # (Mt,)
            if idx_t.numel() == 0:
                c_list.append(out.new_zeros(out_dim))
            else:
                a_t = alpha.index_select(0, idx_t)                        # (Mt, H)
                v_t = v.index_select(0, idx_t)                            # (Mt, H, D)
                m_t = torch.einsum("mh,mhd->hd", a_t, v_t).reshape(-1)    # (out_dim,)
                c_list.append(m_t)
        c_type = torch.stack(c_list, dim=0)                                # (C, out_dim)

        return out, c_type


class EgoRelGAT(nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden: int,
                 num_layers: int,
                 num_heads: int,
                 num_edge_types: int,
                 use_geo_bias: bool = True,
                 geo_dim: int = 3):
        super().__init__()
        self.num_edge_types = int(num_edge_types)

        self.in_proj = nn.Linear(in_dim, hidden)
        self.layers = nn.ModuleList([
            EgoRelGATLayer(
                in_dim=hidden,
                out_dim=hidden,
                num_heads=num_heads,
                num_edge_types=self.num_edge_types,
                use_geo_bias=use_geo_bias,
                geo_dim=geo_dim
            )
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(hidden, hidden)

    def forward(self,
                X: torch.Tensor,
                A: torch.Tensor,
                E: torch.Tensor,
                geo_bias: torch.Tensor | None = None):
        """
        X: (N, Fin)
        A: (N, N)
        E: 见上层注释（支持融合后的复合类别）
        geo_bias: None 或 (M, *)，与有效邻居对应
        """
        h = F.relu(self.in_proj(X))
        ego = h[0]
        c_all = h.new_zeros(self.num_edge_types, h.shape[1])

        for layer in self.layers:
            ego, c_type = layer(h, A, E, geo_bias)
            # 把更新后的 ego 回写到 h[0]
            h = h.clone()
            h[0] = ego
            # 各层类型通道累加
            c_all = c_all + c_type

        return self.out_proj(ego), c_all
