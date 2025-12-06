# dataset.py
# -*- coding: utf-8 -*-
import dill, torch, pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from math import atan2, pi, hypot
import json

def _norm_ang(rad):
    """角度归一化到 [-pi, pi]"""
    return (rad + pi) % (2 * pi) - pi


class NeutralSceneDataset(Dataset):
    """
    读取 convert_trajpp_pkl.py 导出的中立格式：
      neutral_scenes.pkl: List[Dict]
        - name: str
        - node_features: (T, N, F) float32
          假定列序为: [x, y, vx, vy, ax, ay, yaw, a_yaw]  (F=8)
        - adj:           (T, N, N)  (0/1)
        - edge_label:    (T, N, N)  （-1 表示无标签）
        - turn_label:    (T, N, N)  （-1 表示无标签）

    采用“方式A”融合复合边类型：
        当且仅当 edge_label>=0 且 turn_label>=0 时：
            E = edge_label * K_turn + turn_label
        否则：
            E = edge_label  （保持 -1 或 0..BASE_EDGE_TYPES-1）

    这样不会产生负的复合索引。
    """
    def __init__(self, pkl_path, index_csv, use_geo_bias=False, K_turn=19, turn_map_json=None):
        self.scenes = dill.load(open(Path(pkl_path), "rb"))
        self.meta   = pd.read_csv(index_csv)
        self.use_geo_bias = use_geo_bias

        if turn_map_json is not None:
            m = json.load(open(turn_map_json, "r", encoding="utf-8"))
            self.K_turn = max(int(v) for v in m.values()) + 1
        else:
            self.K_turn = int(K_turn)  # 与你的 turn 类别一致（如 19）

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx: int):
        sc = self.scenes[idx]

        # === 基础读取 ===
        X = torch.as_tensor(sc["node_features"], dtype=torch.float32)  # (T,N,F)
        A = torch.as_tensor(sc["adj"], dtype=torch.float32)            # (T,N,N)
        E_raw = torch.as_tensor(sc.get("edge_label", -1), dtype=torch.long)
        T_raw = torch.as_tensor(sc.get("turn_label", -1), dtype=torch.long)
        mask = torch.ones(A.shape[0], dtype=torch.bool)

        # 维度对齐（若 turn 缺失或形状不匹配，直接填 -1）
        if T_raw.ndim == 0 or T_raw.shape != E_raw.shape:
            T_raw = torch.full_like(E_raw, -1)

        # 合法范围裁剪
        E_raw = torch.clamp(E_raw, min=-1)                               # 允许 -1 或 非负
        T_raw = torch.clamp(T_raw, min=-1, max=self.K_turn - 1)

        # 仅当 edge 与 turn 同时有效才做复合编码
        combine_mask = (E_raw >= 0) & (T_raw >= 0)
        E = torch.where(
            combine_mask,
            E_raw * self.K_turn + T_raw,
            E_raw
        ).long()  # (T,N,N)

        # === 输出项 ===
        item = dict(X=X, A=A, E=E, mask=mask, name=sc["name"])

        # === 若启用几何偏置 ===
        if self.use_geo_bias:
            x_idx, y_idx, vx_idx, vy_idx, yaw_idx = 0, 1, 2, 3, 6
            geo_seq = []
            Tlen, N = X.shape[0], X.shape[1]
            for t in range(Tlen):
                # ego -> 有效邻居（排除 ego 自身）
                nbr_mask = (A[t, 0, 1:] > 0.5)  # (N-1,)
                idxs = torch.nonzero(nbr_mask, as_tuple=False).flatten()

                ego_x, ego_y = float(X[t, 0, x_idx]), float(X[t, 0, y_idx])
                ego_vx, ego_vy = float(X[t, 0, vx_idx]), float(X[t, 0, vy_idx])
                ego_yaw = float(X[t, 0, yaw_idx]) if X.shape[-1] > yaw_idx else 0.0
                ego_spd = hypot(ego_vx, ego_vy)

                feats = []
                for rel_j in idxs.tolist():
                    j = rel_j + 1  # 映射真实节点索引
                    nbr_x, nbr_y = float(X[t, j, x_idx]), float(X[t, j, y_idx])
                    nbr_vx, nbr_vy = float(X[t, j, vx_idx]), float(X[t, j, vy_idx])
                    dx, dy = nbr_x - ego_x, nbr_y - ego_y
                    d = hypot(dx, dy)
                    th = _norm_ang(atan2(dy, dx) - ego_yaw)
                    dv = hypot(nbr_vx, nbr_vy) - ego_spd
                    feats.append([d, th, dv])

                if not feats:
                    feats = [[0.0, 0.0, 0.0]]
                geo_seq.append(torch.tensor(feats, dtype=torch.float32))
            item["geo_seq"] = geo_seq

        return item


# === 调试入口 ===
if __name__ == "__main__":
    PKL = r"***.pkl"
    IDX = r"***.csv"
    ds = NeutralSceneDataset(pkl_path=PKL, index_csv=IDX, use_geo_bias=True, K_turn=19)
    X0 = ds[0]["X"]
    print("节点特征形状 (T,N,F):", tuple(X0.shape))
    print("node_dim =", X0.shape[-1])
    E0 = ds[0]["E"]
    print("E 值域: min=", E0.min().item(), " max=", E0.max().item())
    if "geo_seq" in ds[0]:
        g0 = ds[0]["geo_seq"][0]
        print("geo_seq 单步形状:", tuple(g0.shape))
