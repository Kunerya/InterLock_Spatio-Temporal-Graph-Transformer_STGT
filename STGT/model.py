# -*- coding: utf-8 -*-
# scene_embedding/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from spatial_ego_rgat import EgoRelGAT


def _finite_warn(name: str, x: torch.Tensor, enabled: bool):
    """
    调试用：张量中如存在 NaN/Inf，打印告警（不抛异常，避免训练中断）。
    """
    if enabled and torch.is_tensor(x) and (not torch.isfinite(x).all()):
        bad = (~torch.isfinite(x))
        n_bad = int(bad.sum().item())
        print(f"[NaNGuard][WARN] {name} has {n_bad} non-finite values.")


def _sanitize_in_forward(t: torch.Tensor) -> torch.Tensor:
    """
    前向“输入消毒”：仅处理浮点张量，替换 NaN/Inf 并做幅度裁剪，避免后续算子产生 NaN。
    """
    if t is None or (not torch.is_tensor(t)) or (not t.dtype.is_floating_point):
        return t
    # 将 NaN→0，+Inf→1e6，-Inf→-1e6
    t = torch.nan_to_num(t, nan=0.0, posinf=1e6, neginf=-1e6)
    # 再次裁剪幅度，抑制极端值在注意力/softmax中放大
    t = torch.clamp(t, -1e6, 1e6)
    return t


class SceneEncoder(nn.Module):
    def __init__(self,
                 node_dim: int = 8,
                 hidden: int = 128,
                 emb_dim: int = 128,
                 n_heads: int = 4,
                 n_layers_spatial: int = 3,
                 n_layers_temporal: int = 2,
                 num_edge_types: int = 11,
                 use_geo_bias: bool = True,
                 geo_dim: int = 3,
                 renorm_embeddings: bool = False,
                 renorm_eps: float = 1e-8,
                 nan_guard: bool = False):
        """
        参数
        ----
        node_dim:      节点特征维度
        hidden:        中间隐层维度（时空编码的通道数）
        emb_dim:       输出嵌入维度
        n_heads:       Multi-head 注意力头数（空间&时间编码器）
        n_layers_spatial:   空间图编码的层数（传给 EgoRelGAT）
        n_layers_temporal:  时间 TransformerEncoder 的层数
        num_edge_types:     关系类型数量（EgoRelGAT 用）
                            ★ 方式A下应为：base_edge_types * K_turn
                            例如 base_edge_types=9, K_turn=16 → num_edge_types=144
        use_geo_bias:       是否在空间聚合里使用几何偏置
        geo_dim:            几何偏置维度
        renorm_embeddings:  是否对输出嵌入做 L2 归一化
        renorm_eps:         归一化时的 eps，避免除零
        nan_guard:          是否打印 NaN/Inf 告警（默认 False，不中断）
        """
        super().__init__()
        self.C = int(num_edge_types)
        self.renorm_embeddings = bool(renorm_embeddings)
        self.renorm_eps = float(renorm_eps)
        self.nan_guard = bool(nan_guard)

        # 空间编码器：返回 (z_ego, c_type)
        # EgoRelGAT(in_dim, hidden, num_layers, num_heads, num_edge_types, use_geo_bias=True, geo_dim=3)
        self.spatial = EgoRelGAT(
            node_dim,           # in_dim
            hidden,             # hidden
            n_layers_spatial,   # num_layers
            n_heads,            # num_heads
            self.C,             # num_edge_types
            use_geo_bias=use_geo_bias,
            geo_dim=geo_dim
        )

        # 把 ego 表征 + 各关系通道拼接后压缩回 hidden
        # 维度为 hidden * (1 + C) -> hidden
        self.compress = nn.Linear(hidden * (1 + self.C), hidden)

        # 时间编码器
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            batch_first=True
        )
        self.temporal = nn.TransformerEncoder(enc_layer, num_layers=n_layers_temporal)

        # [CLS] token
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))

        # 投影到最终嵌入维度
        self.proj = nn.Linear(hidden, emb_dim)

        # 参数初始化（可选）
        nn.init.xavier_uniform_(self.compress.weight)
        nn.init.zeros_(self.compress.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    @staticmethod
    def _pad(seq_tensors, pad_val: float = 0.0) -> torch.Tensor:
        """
        把不等长的时间序列 (T_i, hidden) 补到同一长度，返回 (B, T_max, hidden)
        """
        T_max = max(x.shape[0] for x in seq_tensors)
        out = []
        for x in seq_tensors:
            if x.shape[0] < T_max:
                pad = x.new_full((T_max - x.shape[0], x.shape[1]), pad_val)
                x = torch.cat([x, pad], dim=0)
            out.append(x)
        return torch.stack(out, dim=0)

    def _check_edge_index_range(self, E_t: torch.Tensor):
        """
        CPU 端的越界/异常检查：在进入 GAT 之前尽早失败，避免 CUDA 设备端断言。
        允许 E_t 中存在 -1（通常表示“无标签”）；但若参与索引运算，仍需由上游保证对应边不被使用。
        """
        if not torch.is_tensor(E_t):
            return
        if E_t.numel() == 0:
            return
        if E_t.dtype.is_floating_point:
            return  # 浮点情况在后续规整逻辑处理
        e_min = int(E_t.min().item())
        e_max = int(E_t.max().item())
        if e_min < -1:
            raise ValueError(f"[SceneEncoder] edge type contains value < -1 (min={e_min}). "
                             f"Expected -1 or [0, {self.C-1}].")
        if e_max >= self.C:
            # 给出友好的提示：方式A下通常是 C = base_edge_types * K_turn
            raise ValueError(
                f"[SceneEncoder] edge type out of range: max={e_max} >= num_edge_types={self.C}. "
                f"Tip: under fusion (edge_label * K_turn + turn_label), set num_edge_types "
                f"= base_edge_types * K_turn (e.g., 9*16=144)."
            )

    def forward(self, batch):
        """
        输入
        ----
        batch: List[dict]，每个 dict 至少包含：
            - "X"[t]: 节点特征 (N_t, node_dim)
            - "A"[t]: 邻接 / 边索引（依赖 EgoRelGAT 的接口约定）
            - "E"[t]: 边类型/特征（依赖 EgoRelGAT 的接口约定）
            - "geo_seq"[t]: 几何偏置 (N_t, geo_dim) 若 use_geo_bias=True

        输出
        ----
        embeddings: Tensor，形状 (B, emb_dim)，B = len(batch)
        """
        device = self.cls.device  # 保证 [CLS] 与后续在同一设备
        seq = []  # 每个元素是该样本的时间序列特征 (T, hidden)

        for item in batch:
            # 时间步数
            T = item["A"].shape[0]
            vec_t = []
            for t in range(T):
                # 取出单步图数据
                X_t  = item["X"][t]
                A_t  = item["A"][t]
                E_t  = item["E"][t]
                geo  = item.get("geo_seq", None)
                geo_t = geo[t] if (geo is not None) else None

                # ===== 空图/空边守卫：该帧直接占位，不进入 GAT =====
                if X_t.numel() == 0 or A_t.numel() == 0:
                    # compress.in_features = hidden * (1 + C)，所以占位维度就是 hidden
                    hid = self.compress.in_features // (1 + self.C)
                    vec_t.append(torch.zeros(hid, device=X_t.device))
                    continue

                # ===== 仅对“浮点特征”做清洗；索引保持为整型 =====
                X_t  = _sanitize_in_forward(X_t)
                geo_t = _sanitize_in_forward(geo_t)

                # —— A_t / E_t 统一为整型索引（关键）——
                # A_t（通常为边索引 shape=(2,M) 或 (M,2)）
                if torch.is_tensor(A_t) and A_t.dtype != torch.long:
                    A_t = A_t.long()

                # E_t（边类型）：支持三种常见输入
                #  1) int/long 已是类别索引 -> 保持
                #  2) float 的类别 id（如 0.0/1.0/...）-> round + clamp 到 [0, C-1] 再 long
                #  3) one-hot（M, C）-> argmax 离散化为 (M,)
                if torch.is_tensor(E_t):
                    if E_t.dtype.is_floating_point:
                        if E_t.dim() == 2 and E_t.size(-1) == self.C:
                            E_t = E_t.argmax(dim=-1).to(torch.long)
                        else:
                            E_t = torch.nan_to_num(E_t, nan=0.0, posinf=0.0, neginf=0.0)
                            E_t = E_t.round().clamp_(0, self.C - 1).to(torch.long)
                    elif E_t.dtype != torch.long:
                        E_t = E_t.to(torch.long)

                # ===== 新增：在进入 GAT 前做边类型范围检查（CPU 端，避免 CUDA 设备断言）=====
                self._check_edge_index_range(E_t)

                # 有限性告警（不抛异常）
                _finite_warn("X_t", X_t, self.nan_guard)
                if geo_t is not None:
                    _finite_warn("geo_t", geo_t, self.nan_guard)

                # 空间编码
                z_ego, c_type = self.spatial(X_t, A_t, E_t, geo_bias=geo_t)

                # 输出再次消毒与检查（双保险）
                z_ego = _sanitize_in_forward(z_ego)
                _finite_warn("z_ego", z_ego, self.nan_guard)
                c_type = [_sanitize_in_forward(c_type[k]) for k in range(self.C)]
                for k in range(self.C):
                    _finite_warn(f"c_type[{k}]", c_type[k], self.nan_guard)

                # 拼接 ego + C 个关系通道
                cat_list = [z_ego] + [c_type[k] for k in range(self.C)]
                z_cat = torch.cat(cat_list, dim=-1)            # (hidden*(1+C),)
                _finite_warn("z_cat", z_cat, self.nan_guard)

                # 压缩到 hidden
                z_hid = self.compress(z_cat)                   # (hidden,)
                _finite_warn("z_hid", z_hid, self.nan_guard)

                vec_t.append(z_hid)

            # 若整条序列全是空帧，给一个零帧占位，避免后续堆栈为空
            if len(vec_t) == 0:
                hid = self.compress.in_features // (1 + self.C)
                vec_t = [torch.zeros(hid, device=device)]

            # (T, hidden)
            seq.append(torch.stack(vec_t, dim=0))

        # (B, T_max, hidden)
        Z = self._pad(seq, pad_val=0.0).to(device)
        _finite_warn("Z_padded", Z, self.nan_guard)

        # 前置 [CLS]，变成 (B, 1 + T_max, hidden)
        cls_tok = self.cls.expand(Z.size(0), -1, -1)          # (B,1,hidden)
        Z_in = torch.cat([cls_tok, Z], dim=1)                 # (B, 1+T, hidden)
        _finite_warn("Z_in", Z_in, self.nan_guard)

        # 时间 Transformer 编码
        H = self.temporal(Z_in)                                # (B, 1+T, hidden)
        _finite_warn("H", H, self.nan_guard)

        # 取 [CLS] 向量
        h_cls = H[:, 0, :]                                     # (B, hidden)

        # 投影到最终嵌入
        z = self.proj(h_cls)                                   # (B, emb_dim)
        _finite_warn("z_proj", z, self.nan_guard)

        # 按需做 L2 归一化（建议在用“余弦三元组”时关闭；exp 相似度无需归一化）
        if self.renorm_embeddings:
            z = F.normalize(z, p=2, dim=-1, eps=self.renorm_eps)
            _finite_warn("z_normed", z, self.nan_guard)

        return z
