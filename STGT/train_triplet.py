# scene_embedding/train_triplet.py
# -*- coding: utf-8 -*-
# python train_triplet.py --cfg "configs/config.yaml"


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse, random, yaml, torch, numpy as np, pandas as pd, csv, math
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import time
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime

from dataset import NeutralSceneDataset
from model import SceneEncoder
from loss import HybridTripletSimilarityLoss  # ← 使用与0-1相似度一致的损失
# --- 高吞吐设置（Amp/TF32）---
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")  # A100/RTX40系收益明显

# ============ 通用IO工具 ============
def _read_csv_robust(path):
    """更鲁棒的 CSV 读取：优先 utf-8-sig 其次 utf-8；自动分隔符嗅探。"""
    try_enc = ["utf-8-sig", "utf-8"]
    for enc in try_enc:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                dialect = csv.Sniffer().sniff(f.read(4096), delimiters=",;|\t")
                f.seek(0)
                return pd.read_csv(f, encoding=enc, delimiter=dialect.delimiter)
        except Exception:
            continue
    # 兜底
    return pd.read_csv(path, encoding="utf-8-sig")

def _clean_fp_tensor(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or not x.dtype.is_floating_point:
        return x
    x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    x = torch.clamp(x, min=-1e6, max=1e6)
    return x


# ============ 名称/索引映射 ============
def build_name_maps(index_csv: str):
    df = _read_csv_robust(index_csv)
    # 容错列名
    cols = {c.strip().lower(): c for c in df.columns}
    need_idx = next((cols[k] for k in cols if k in ("scene_idx", "idx", "id")), None)
    need_name = next((cols[k] for k in cols if k in ("scene_name", "name", "filename")), None)
    if not need_idx or not need_name:
        raise ValueError(f"index_csv 缺少 scene_idx/scene_name 列: {index_csv}")

    name2idx, idx2name = {}, {}
    for _, r in df.iterrows():
        idx = int(r[need_idx])
        name = str(r[need_name])
        stem = Path(name).stem
        for k in (name, Path(name).name, stem, str(idx)):
            name2idx.setdefault(k, idx)
        idx2name[idx] = stem
    return name2idx, idx2name

def infer_num_edge_types(dataset, sample_scenes: int = 200) -> int:
    """
    扫描数据集（最多 sample_scenes 个样本），推断复合边类型的取值空间：
      num_edge_types = max(E) + 1（忽略 <0 的无效标签）
    说明：你的 dataset.py 已将 edge_label 与 turn_label 融合(E_raw*K_turn+T)，
         因此这里只需要看 E 的最大值即可。
    """
    import numpy as np
    mx = -1
    n = min(len(dataset.scenes), int(sample_scenes))
    for i in range(n):
        sc = dataset.scenes[i]
        E = sc.get("edge_label", None)
        T = sc.get("turn_label", None)
        if E is None:
            continue
        E = np.asarray(E)
        # 若 turn_label 也有，则复合编码规则与 dataset.py 一致
        if T is not None:
            T = np.asarray(T)
            valid = (E >= 0) & (T is not None) & (T >= 0)
            K_turn = getattr(dataset, "K_turn", 19)
            E_comb = np.where(valid, E * K_turn + T, E)
        else:
            E_comb = E
        vals = E_comb[E_comb >= 0]
        if vals.size:
            mx = max(mx, int(vals.max()))
    return (mx + 1) if mx >= 0 else 1


def build_all_supervised_triplets(id_list, pos_dict, neg_dict, rng, neg_per_pos=1):
    """覆盖所有已标注的正对；每个正对配 neg_per_pos 个负样本。"""
    out = []
    id_set = set(id_list)
    for a, plist in pos_dict.items():
        if a not in id_set: 
            continue
        for (p, s_ap) in plist:
            # 每个正对配多个负样本
            for _ in range(max(1, int(neg_per_pos))):
                if a in neg_dict and neg_dict[a]:
                    n, s_an_inv = rng.choices(neg_dict[a], weights=[s for (_, s) in neg_dict[a]], k=1)[0]
                    s_an = 1.0 - s_an_inv
                else:
                    # 随机负样本（不与 a/p 相同；且不在正对列表里）
                    pos_set = {pp for (pp, _) in plist}
                    cand_pool = [x for x in id_list if x != a and x != p and x not in pos_set]
                    if not cand_pool:
                        continue
                    n = rng.choice(cand_pool); s_an = 0.0
                w = max(0.0, min(1.0, s_ap - s_an))
                out.append((a, p, n, w))
    rng.shuffle(out)
    return out

def to_idx_any(x, name2idx):
    xs = str(x).strip()
    if xs.isdigit():
        return int(xs)
    if xs in name2idx:
        return int(name2idx[xs])
    try:
        stem = Path(xs).stem
        if stem in name2idx:
            return int(name2idx[stem])
        base = Path(xs).name
        if base in name2idx:
            return int(name2idx[base])
    except Exception:
        pass
    raise KeyError(f"无法在 index 映射中找到: {xs}")

# ============ 读取成对分数（弱监督） ============
def _auto_score_column(df):
    """在 pair_scores_min.csv 中自动找一个 0~1 分数列。"""
    priors = [
        "final_score(0-1)_to_fill", "final_score", "score", "sim", "similarity",
        "label", "prob", "p"
    ]
    cols = {c.strip().lower(): c for c in df.columns}
    for k in priors:
        if k in cols:
            return cols[k]
    # 再尝试找范围在 0~1 的数值列
    numeric_cols = []
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce")
        ratio_01 = float(((v >= 0) & (v <= 1)).mean())
        if ratio_01 > 0.8:  # 大多数在[0,1]
            numeric_cols.append((c, ratio_01))
    if numeric_cols:
        numeric_cols.sort(key=lambda x: -x[1])
        return numeric_cols[0][0]
    raise ValueError("找不到可用的分数字段，请检查 pair_scores_min.csv")


def _col(df, candidates):
    """根据多种写法取列名，例如 anchor / Anchor / \ufeffanchor"""
    low = {c.strip().lower(): c for c in df.columns}
    for name in candidates:
        if name in low:
            return low[name]
    return None


def load_pair_scores(pair_csv: str, index_csv: str, pos_thresh=0.6, neg_thresh=0.4):
    name2idx, _ = build_name_maps(index_csv)

    df = _read_csv_robust(pair_csv)
    # 规范列名（去 BOM、去首尾空白）
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    # 大小写无关取列
    col_anchor = _col(df, ["anchor"])
    col_cand   = _col(df, ["candidate", "cand", "candidate_id"])
    if not col_anchor or not col_cand:
        raise ValueError(f"{pair_csv} 缺少 anchor/candidate 列，实际列: {list(df.columns)}")

    # 分数字段：优先 final_score(0-1)_to_fill > final_score > score > 自动探测
    score_candidates = ["final_score(0-1)_to_fill", "final_score", "score"]
    score_col = next(( _col(df, [c]) for c in score_candidates if _col(df, [c]) ), None)
    if score_col is None:
        score_col = _auto_score_column(df)  # ← 最后兜底自动探测 0~1 列

    df = df[[col_anchor, col_cand, score_col]].copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").clip(0, 1)
    df = df.dropna(subset=[col_anchor, col_cand, score_col])

    pos_dict, neg_dict = defaultdict(list), defaultdict(list)
    for _, r in df.iterrows():
        a = to_idx_any(r[col_anchor], name2idx)
        c = to_idx_any(r[col_cand],   name2idx)
        s = float(r[score_col])
        if s >= pos_thresh:
            pos_dict[a].append((c, s))
        if s <= neg_thresh:
            neg_dict[a].append((c, 1.0 - s))

    # 去重&排序
    for a in list(pos_dict.keys()):
        pos_dict[a] = sorted({(c, s) for (c, s) in pos_dict[a]}, key=lambda x: -x[1])
    for a in list(neg_dict.keys()):
        neg_dict[a] = sorted({(c, s) for (c, s) in neg_dict[a]}, key=lambda x: -x[1])
    print(f"[pair_scores] 使用分数字段: {score_col} | anchors={len(set(list(pos_dict.keys())+list(neg_dict.keys())))}")
    return pos_dict, neg_dict


def build_score_map(pair_csv: str, index_csv: str):
    name2idx, _ = build_name_maps(index_csv)
    df = _read_csv_robust(pair_csv)
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]

    col_anchor = _col(df, ["anchor"])
    col_cand   = _col(df, ["candidate", "cand", "candidate_id"])
    if not col_anchor or not col_cand:
        raise ValueError(f"{pair_csv} 缺少 anchor/candidate 列，实际列: {list(df.columns)}")

    score_candidates = ["final_score(0-1)_to_fill", "final_score", "score"]
    score_col = next(( _col(df, [c]) for c in score_candidates if _col(df, [c]) ), None)
    if score_col is None:
        score_col = _auto_score_column(df)

    df = df[[col_anchor, col_cand, score_col]].copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce").clip(0, 1)
    df = df.dropna(subset=[col_anchor, col_cand, score_col])

    mp = {}
    for _, r in df.iterrows():
        a = to_idx_any(r[col_anchor], name2idx)
        c = to_idx_any(r[col_cand],   name2idx)
        mp[(a, c)] = float(r[score_col])
    print(f"[pair_scores] build_score_map 使用分数字段: {score_col} | pairs={len(mp)}")
    return mp

@torch.no_grad()
def scan_dataset_for_nonfinite(ds, ids, device=None, use_clean=False, max_print=50):
    printed = 0
    for idx in ids:
        raw = ds[idx]
        item = raw
        if use_clean and device is not None:
            # 复用你已修改的 to_device_item（其内部默认 clean=True）
            item = to_device_item(raw, device)

        # 扫描时间序列
        for t, X_t in enumerate(item["X"]):
            bad = (~torch.isfinite(X_t)).sum().item()
            if bad > 0:
                print(f"[DataCheck] idx={idx} t={t} field=X nonfinite={bad}")
                printed += 1
                if printed >= max_print:
                    return

# ============ 划分/设备搬运 ============
def split_indices(num_scenes: int, train_ratio=0.8, seed=42):
    idxs = list(range(num_scenes))
    rng = random.Random(seed); rng.shuffle(idxs)
    k = int(len(idxs) * train_ratio)
    return sorted(idxs[:k]), sorted(idxs[k:])

def to_device_item(item: dict, device: torch.device, clean: bool = True, clip: float = 1e6,
                   non_blocking: bool = True):
    out = {}
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            t = v
            if clean:
                t = torch.nan_to_num(t, nan=0.0, posinf=clip, neginf=-clip)
            out[k] = t.to(device, non_blocking=non_blocking)

        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            lst = []
            for t in v:
                if clean:
                    t = torch.nan_to_num(t, nan=0.0, posinf=clip, neginf=-clip)
                lst.append(t.to(device, non_blocking=non_blocking))
            out[k] = lst
        else:
            out[k] = v
    return out

@torch.no_grad()
def ndcg_at_k_from_E_same(E, ids, score_map, k=3, gain="exp", unjudged_policy="ignore"):
    import math
    if E.size(0) < 2: return 0.0, 0, 0.0
    d = torch.cdist(E, E, p=2); d.fill_diagonal_(float("inf"))
    N = d.size(0); k_eff = int(min(k, max(1, N-1)))
    row2id = {i: ids[i] for i in range(N)}; id2row = {v:k for k,v in row2id.items()}
    anchors_eval = sorted({a for (a,_) in score_map.keys() if a in id2row})
    rows_eval = [id2row[a] for a in anchors_eval]
    if not rows_eval: return 0.0, 0, 0.0
    def _g(y): return (2.0**float(y)-1.0) if gain=="exp" else float(y)
    ndcgs, covs = [], []
    for r in rows_eval:
        a_id = row2id[r]
        labels_all = [y for ((aa,cc),y) in score_map.items() if aa==a_id]
        if not labels_all: continue
        nn_idx = torch.topk(d[r], k=k_eff, largest=False).indices.tolist()
        dcg, judged = 0.0, 0
        for rank, col in enumerate(nn_idx, 1):
            c_id = row2id[col]
            y = score_map.get((a_id, c_id))
            if y is None:
                if unjudged_policy=="zero": pass
                else: continue
            else:
                judged += 1; dcg += _g(y)/math.log2(rank+1)
        covs.append(judged/float(k_eff))
        L = judged if unjudged_policy=="ignore" else k_eff
        if L==0: continue
        labels_sorted = sorted(labels_all, reverse=True)[:L]
        idcg = sum(_g(y)/math.log2(r+1) for r,y in enumerate(labels_sorted,1))
        if idcg>0: ndcgs.append(dcg/idcg)
    cov_mean = float(sum(covs)/len(covs)) if covs else 0.0
    return (float(sum(ndcgs)/len(ndcgs)) if ndcgs else 0.0), len(ndcgs), cov_mean

@torch.no_grad()
def recall_at_k_from_E_same(E, ids, pos_map, k=10):
    if E.size(0) < 2: return 0.0, 0
    d = torch.cdist(E, E, p=2); d.fill_diagonal_(float("inf"))
    N = d.size(0); k_eff = int(min(k, max(1, N-1)))
    nn_idx = torch.topk(d, k=k_eff, dim=1, largest=False).indices
    row2id = {i: ids[i] for i in range(N)}
    rows_eval = [i for i in range(N) if row2id[i] in pos_map]
    if not rows_eval: return 0.0, 0
    hit = 0
    for i in rows_eval:
        a_id = row2id[i]
        neigh_ids = [row2id[j] for j in nn_idx[i].tolist()]
        pos_set = {c for (c,_) in pos_map.get(a_id, [])}
        if pos_set and any(nb in pos_set for nb in neigh_ids): hit += 1
    return hit/len(rows_eval), len(rows_eval)

@torch.no_grad()
def pair_reg_from_E(E, ids, score_map, alpha, pair_loss="mse"):
    import torch.nn.functional as F
    id2row = {sid:i for i,sid in enumerate(ids)}
    preds, gts = [], []
    for (a,c), y in score_map.items():
        ia, ic = id2row.get(a), id2row.get(c)
        if ia is None or ic is None: continue
        d = torch.norm(E[ia]-E[ic], p=2)
        s = torch.exp(torch.clamp(-float(alpha)*d, min=-50.0, max=0.0))
        preds.append(s)
        gts.append(torch.tensor(y, dtype=torch.float32, device=s.device))
    if not preds: return float("nan"), 0
    P, Y = torch.stack(preds), torch.stack(gts)
    L = F.mse_loss(P,Y) if pair_loss=="mse" else F.smooth_l1_loss(P,Y)
    return float(L.item()), len(preds)

@torch.no_grad()
def spearman_from_E(E, ids, score_map, alpha=1.0):
    import numpy as np, math
    id2row = {sid:i for i,sid in enumerate(ids)}
    sims, hum = [], []
    for (a,c), y in score_map.items():
        ia, ic = id2row.get(a), id2row.get(c)
        if ia is None or ic is None: continue
        d = torch.norm(E[ia]-E[ic], p=2).item()
        sims.append(math.exp(max(-50.0, min(0.0, -alpha*d))))
        hum.append(float(y))
    if len(sims) < 2: return 0.0, len(sims)
    sims = np.asarray(sims); hum = np.asarray(hum)
    r1 = sims.argsort().argsort().astype(float)
    r2 = hum.argsort().argsort().astype(float)
    if r1.std()<1e-9 or r2.std()<1e-9: return 0.0, len(sims)
    return float(np.corrcoef(r1,r2)[0,1]), len(sims)

@torch.no_grad()
def triplet_eval_from_E(E, ids, triplets, margin, alpha, mode="euclid"):
    """在已编码 E 上评估三元组满足率/损失"""
    if not triplets: return 0.0, 0.0
    id2row = {sid:i for i,sid in enumerate(ids)}
    ok = 0; total_loss = 0.0; cnt = 0
    for a,p,n,_w in triplets:
        ia, ip, in_ = id2row.get(a), id2row.get(p), id2row.get(n)
        if ia is None or ip is None or in_ is None: continue
        za, zp, zn = E[ia], E[ip], E[in_]
        if mode=="euclid":
            d_ap = torch.norm(za-zp, p=2).item()
            d_an = torch.norm(za-zn, p=2).item()
            ok += int((d_ap + margin) < d_an)
            total_loss += max(0.0, margin + d_ap - d_an)
        elif mode=="exp":
            d_ap = torch.norm(za-zp, p=2).item()
            d_an = torch.norm(za-zn, p=2).item()
            s_ap = math.exp(max(-50.0, min(0.0, -alpha*d_ap)))
            s_an = math.exp(max(-50.0, min(0.0, -alpha*d_an)))
            ok += int((s_ap - s_an) > margin)
            total_loss += max(0.0, margin - (s_ap - s_an))
        else:  # cosine
            import torch.nn.functional as F
            a_n = F.normalize(za[None], dim=-1); p_n = F.normalize(zp[None], dim=-1); n_n = F.normalize(zn[None], dim=-1)
            s_ap = float((a_n*p_n).sum())
            s_an = float((a_n*n_n).sum())
            ok += int((s_ap - s_an) > margin)
            total_loss += max(0.0, margin - (s_ap - s_an))
        cnt += 1
    return (total_loss/max(1,cnt)), (ok/max(1,cnt))

# ============ 评估：Triplet 满足率 ============
@torch.no_grad()
def eval_on_triplets(net, ds, triplets, device, loss_fn,
                     mode: str = "exp",          # "exp" | "cosine" | "euclid"
                     alpha: float = 2.0,         # 仅 "exp" 模式有效
                     margin: float = 0.1,        # 相似度域建议 0.05~0.2；欧氏域是距离差
                     ):
    """
    返回: (avg_triplet_loss, ok_ratio)
      - avg_triplet_loss 仍通过传入的 loss_fn 计算（不改你训练口径）
      - ok_ratio 的判定按 mode：
          "exp":   s_ap = exp(-alpha * d_ap), s_an = exp(-alpha * d_an)，ok 若 s_ap - s_an > margin
          "cosine":cos_ap = <a,p>，cos_an = <a,n>（均在 L2 归一化后），ok 若 cos_ap - cos_an > margin
          "euclid":沿用旧规则，ok 若 d_ap + margin < d_an
    """
    import torch
    import torch.nn.functional as F

    net.eval()
    if len(triplets) == 0:
        return 0.0, 0.0

    total_loss, ok, tot = 0.0, 0, 0

    for a, p, n, w in triplets:
        aitem = to_device_item(ds[a], device)
        pitem = to_device_item(ds[p], device)
        nitem = to_device_item(ds[n], device)

        # 前向
        emb = net([aitem, pitem, nitem])   # 期望 (3, D)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        elif emb.dim() > 2:
            emb = emb.view(emb.size(0), -1)

        # —— 计算 loss（用你现有的 loss_fn，不改变训练口径）——
        weight = torch.tensor([w], device=device, dtype=torch.float32)
        loss = loss_fn(emb[0:1], emb[1:2], emb[2:3], y_ap=None, y_an=None, weight=weight)
        if torch.isfinite(loss):
            total_loss += float(loss.item())

        # —— 计算 ok 判定（按相似度或距离）——
        a_z, p_z, n_z = emb[0:1], emb[1:2], emb[2:3]

        if mode == "cosine":
            # 归一化后做余弦相似度
            a_n = F.normalize(a_z, p=2, dim=-1)
            p_n = F.normalize(p_z, p=2, dim=-1)
            n_n = F.normalize(n_z, p=2, dim=-1)
            s_ap = (a_n * p_n).sum(dim=-1)         # in [-1, 1]
            s_an = (a_n * n_n).sum(dim=-1)
            is_ok = (s_ap - s_an) > margin

        elif mode == "exp":
            # 把欧氏距离映射到相似度域 s = exp(-alpha * d)，再比较相似度间隔
            d_ap = torch.norm(a_z - p_z, p=2, dim=-1)
            d_an = torch.norm(a_z - n_z, p=2, dim=-1)
            s_ap = torch.exp(torch.clamp(-alpha * d_ap, min=-50.0, max=0.0))
            s_an = torch.exp(torch.clamp(-alpha * d_an, min=-50.0, max=0.0))
            is_ok = (s_ap - s_an) > margin

        else:  # "euclid"：保持原判定
            d_ap = torch.norm(a_z - p_z, p=2, dim=-1)
            d_an = torch.norm(a_z - n_z, p=2, dim=-1)
            is_ok = (d_ap + margin) < d_an

        if torch.isfinite(is_ok.float()).all():
            ok += int(bool(is_ok.item()))
            tot += 1

    avg_loss = total_loss / max(1, tot)
    ok_ratio = ok / max(1, tot)
    return avg_loss, ok_ratio

# ============ 采样：只产“有监督”三元组 ============
def sample_epoch_triplets(id_list, pos_dict, neg_dict, steps, rng):
    out = []
    if steps <= 0:
        return out
    anchors = id_list[:]
    rng.shuffle(anchors)
    i = 0
    while len(out) < steps and anchors:
        a = anchors[i % len(anchors)]
        plist = pos_dict.get(a, [])
        if not plist:
            i += 1
            if i > len(anchors) * 2:
                break
            continue
        p, s_ap = rng.choices(plist, weights=[s for (_, s) in plist], k=1)[0]
        nlist = neg_dict.get(a, [])
        if nlist:
            n, s_an_inv = rng.choices(nlist, weights=[s for (_, s) in nlist], k=1)[0]
            s_an = 1.0 - s_an_inv
        else:
            cand_pool = [x for x in id_list if x != a and x not in [pp for pp, _ in plist]]
            if not cand_pool:
                i += 1
                continue
            n = rng.choice(cand_pool); s_an = 0.0
        w = max(0.0, min(1.0, s_ap - s_an))
        out.append((a, p, n, w))
        i += 1
    return out

# ============ 防数据泄漏：按 split 过滤 ============
def restrict_pairs(pos_dict, neg_dict, allowed_ids):
    allow = set(allowed_ids)
    pos_new, neg_new = {}, {}
    for a, lst in pos_dict.items():
        if a not in allow: continue
        kept = [(c, s) for (c, s) in lst if c in allow]
        if kept: pos_new[a] = kept
    for a, lst in neg_dict.items():
        if a not in allow: continue
        kept = [(c, s) for (c, s) in lst if c in allow]
        if kept: neg_new[a] = kept
    return pos_new, neg_new


@torch.no_grad()
def eval_pair_reg_loss(net_model, dataset, val_score_map, device, alpha, pair_loss="mse"):
    """
    验证阶段的 pair 回归损失：
      - 使用 s_pred = exp(-alpha * ||za - zc||_2)
      - 统一 pred/target 形状为 [1]
      - 跳过任何非有限样本，避免 NaN 污染
    """
    import torch
    import torch.nn.functional as F

    net_model.eval()
    if not val_score_map:
        return 0.0, 0

    def _loss(pred, target):
        if pair_loss == "huber":
            return F.smooth_l1_loss(pred, target, reduction="mean")
        else:
            return F.mse_loss(pred, target, reduction="mean")

    total, cnt = 0.0, 0

    for (a, c), y in val_score_map.items():
        # 准备输入
        aitem = to_device_item(dataset[a], device)
        citem = to_device_item(dataset[c], device)

        # 前向：期望得到 (2, D)
        z = net_model([aitem, citem])
        if not isinstance(z, torch.Tensor):
            continue
        if z.dim() == 1:
            z = z.unsqueeze(0)                 # -> (1, D)
        elif z.dim() > 2:
            z = z.view(z.size(0), -1)          # 形状容错
        if z.size(0) < 2:
            # 无法形成配对，跳过
            continue

        # 距离与相似度（形状统一为 [1]）
        d = torch.norm(z[0:1] - z[1:2], p=2)
        if not torch.isfinite(d):
            continue

        s_pred = torch.exp(torch.clamp(-float(alpha) * d, min=-50.0, max=0.0)).view(1)
        y_t = torch.as_tensor([y], device=device, dtype=torch.float32)  # [1]

        # 计算损失（防止非有限）
        loss_val = _loss(s_pred, y_t)
        if not torch.isfinite(loss_val):
            continue

        total += float(loss_val.item())
        cnt += 1

    return (total / max(1, cnt)), cnt

@torch.no_grad()
def suggest_margin_from_dist(net_model, dataset, train_score_map, device, 
                             pos_thresh=0.6, neg_thresh=0.4, rho=0.4, 
                             clip=None):
    import numpy as np, torch
    pos_d, neg_d = [], []
    for (a, c), y in train_score_map.items():
        if (y >= pos_thresh) or (y <= neg_thresh):
            aitem = to_device_item(dataset[a], device)
            citem = to_device_item(dataset[c], device)
            z = net_model([aitem, citem])
            if z.dim() == 1: z = z.unsqueeze(0)
            d = torch.norm(z[0]-z[1], p=2).item()
            (pos_d if y >= pos_thresh else neg_d).append(d)

    if len(pos_d) < 5 or len(neg_d) < 5:
        return None

    pos_med = float(np.median(pos_d))
    neg_med = float(np.median(neg_d))
    gap = max(neg_med - pos_med, 0.0)

    m_adapt = rho * gap                 # ρ∈[0.2,0.6] 经验值
    if clip is not None:
        m_adapt = max(clip[0], min(clip[1], m_adapt))
    print(f"[MarginAdapt] pos_med={pos_med:.3f} neg_med={neg_med:.3f} gap={gap:.3f} -> m_adapt={m_adapt:.3f}")
    return m_adapt

# ============ 评估：Recall@K（只评有正对的 anchors） ============
@torch.no_grad()
def eval_recall_at_k(net_model, dataset, ids_val, pos_va, device, k=10):
    net_model.eval()
    if len(ids_val) == 0:
        return 0.0, 0

    Z = []
    for idx in ids_val:
        item = to_device_item(dataset[idx], device)
        z = net_model([item])          # 可能是 (D,) 或 (1,D) 或 (B,D)
        if z.dim() == 1:
            z = z.unsqueeze(0)         # → (1, D)
        elif z.dim() > 2:
            z = z.view(z.size(0), -1)  # 容错
        Z.append(z[0].detach().cpu())

    if not Z or len(Z) < 2:
        return 0.0, 0

    E = torch.stack(Z, dim=0)          # (N, D)
    # 距离矩阵（兼容 N=1）
    dists = torch.cdist(E, E, p=2)
    dists.fill_diagonal_(float("inf"))

    N = dists.size(0)
    k_eff = int(min(k, max(1, N-1)))
    nn_idx = torch.topk(dists, k=k_eff, dim=1, largest=False).indices

    row2id = {i: ids_val[i] for i in range(N)}
    rows_eval = [i for i in range(N) if row2id[i] in pos_va]
    if not rows_eval:
        return 0.0, 0

    hit = 0
    for i in rows_eval:
        a_id = row2id[i]
        neigh_ids = [row2id[j] for j in nn_idx[i].tolist()]
        pos_set = {c for (c, _) in pos_va.get(a_id, [])}
        if any(nb in pos_set for nb in neigh_ids):
            hit += 1
    return hit / len(rows_eval), len(rows_eval)

# ============ 评估：Spearman（人工分 vs 嵌入相似） ============
@torch.no_grad()
def eval_spearman(net_model, dataset, ids_val, score_map, device, alpha=1.0):
    net_model.eval()
    if len(ids_val) == 0:
        return 0.0, 0

    idset = set(ids_val)
    pairs = [(a, c, s) for (a, c), s in score_map.items()
             if a in idset and c in idset]
    if not pairs:
        return 0.0, 0

    uniq_ids = sorted(set([p[0] for p in pairs] + [p[1] for p in pairs]))
    id2row = {idx: i for i, idx in enumerate(uniq_ids)}

    Z = []
    for idx in uniq_ids:
        item = to_device_item(dataset[idx], device)
        z = net_model([item])              # (1, D) or (B, D)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        elif z.dim() > 2:
            z = z.view(z.size(0), -1)
        Z.append(z[0].detach().cpu())
    E = torch.stack(Z, dim=0)              # (M, D)

    # 将欧氏距离映射为相似度 s = exp(-alpha * d)
    sim_list, human_list = [], []
    for a, c, s in pairs:
        va = E[id2row[a]]; vc = E[id2row[c]]
        d = torch.norm(va - vc, p=2).item()
        sim_list.append(math.exp(max(-50.0, min(0.0, -alpha * d))))
        human_list.append(float(s))

    if len(sim_list) < 2:
        return 0.0, len(sim_list)

    sim_arr = np.asarray(sim_list)
    hum_arr = np.asarray(human_list)

    # Spearman（稠密秩）
    def rank_dense(x):
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x), dtype=float)
        return ranks

    r1 = rank_dense(sim_arr)
    r2 = rank_dense(hum_arr)
    if np.std(r1) < 1e-9 or np.std(r2) < 1e-9:
        return 0.0, len(sim_list)
    sp = np.corrcoef(r1, r2)[0, 1]
    return float(sp), len(sim_list)

@torch.no_grad()
def check_embed_norms(net, ds, ids, device, sample_k=64):
    norms = []
    for i in ids[:min(sample_k, len(ids))]:
        item = to_device_item(ds[i], device)
        z = net([item])
        if z.dim() == 1:
            z = z.unsqueeze(0)
        norms.append(torch.norm(z[0], p=2).item())
    import numpy as np
    arr = np.array(norms, float)
    print(f"[L2 norm] mean={arr.mean():.4f} std={arr.std():.4f} "
          f"min={arr.min():.4f} p95={np.percentile(arr,95):.4f} max={arr.max():.4f}")
    
@torch.no_grad()
def eval_ndcg_at_k(
    net_model,
    dataset,
    ids_val,                # 验证集样本 ID 列表
    val_score_map,          # {(a_id, c_id): y∈[0,1]}
    device,
    k=3,
    gain="exp",             # "exp" 用 2^y-1；"linear" 用 y
    unjudged_policy="ignore"  # "ignore": 未标注项不计入DCG，并用已标注条数L来算IDCG
                              # "zero":   未标注项当作 y=0 计入DCG，IDCG仍按k计算（更保守）
):
    """
    返回: (ndcg_mean, anchors_count, cov_mean)
      - ndcg_mean: 有效 anchors 的 nDCG@k 平均值
      - anchors_count: 被计入统计的 anchor 数
      - cov_mean: 平均覆盖率 Cov@k = (Top-k中被标注的条目数)/k
    """
    def _gain(y):
        return (2.0 ** float(y) - 1.0) if gain == "exp" else float(y)

    net_model.eval()
    if len(ids_val) == 0:
        return 0.0, 0, 0.0

    # 1) 预编码验证集嵌入
    Z = []
    for idx in ids_val:
        item = to_device_item(dataset[idx], device)
        z = net_model([item])                   # (1,D) or (B,D) or (D,)
        if z.dim() == 1: z = z.unsqueeze(0)
        elif z.dim() > 2: z = z.view(z.size(0), -1)
        Z.append(z[0].detach().cpu())
    if not Z or len(Z) < 2:
        return 0.0, 0, 0.0

    E = torch.stack(Z, dim=0)                   # (N, D)
    dists = torch.cdist(E, E, p=2)
    dists.fill_diagonal_(float("inf"))
    N = dists.size(0)
    k_eff = int(min(k, max(1, N - 1)))

    row2id = {i: ids_val[i] for i in range(N)}
    id2row = {v: k for k, v in row2id.items()}

    # 只评估“在 val_score_map 里出现过”的 anchors（至少有一条人工分）
    anchors_eval = sorted({a for (a, c) in val_score_map.keys() if a in id2row})
    rows_eval = [id2row[a] for a in anchors_eval]

    ndcgs, covs = [], []
    for r in rows_eval:
        a_id = row2id[r]
        # 该 anchor 的所有人工分（用于IDCG）
        labels_all = [y for ((aa, cc), y) in val_score_map.items() if aa == a_id]
        if not labels_all:
            continue

        # Top-k 邻居（按距离升序）
        nn_idx = torch.topk(dists[r], k=k_eff, largest=False).indices.tolist()

        # DCG 与覆盖率
        dcg, judged = 0.0, 0
        for rank, col in enumerate(nn_idx, start=1):
            c_id = row2id[col]
            y = val_score_map.get((a_id, c_id), None)
            if y is None:
                if unjudged_policy == "zero":
                    # 当作0分计入（更保守）
                    # dcg += 0 / log2(rank+1) 等价于不加
                    pass
                else:
                    # "ignore": 未标注不计入DCG
                    continue
            else:
                judged += 1
                dcg += _gain(y) / math.log2(rank + 1)

        # 覆盖率
        cov = judged / float(k_eff)
        covs.append(cov)

        # IDCG：与DCG的“有效项数量”对齐
        if unjudged_policy == "ignore":
            L = judged
        else:  # "zero"：按k计算
            L = k_eff

        if L == 0:
            # 完全无已标注项，跳过该 anchor
            continue

        labels_sorted = sorted(labels_all, reverse=True)[:L]
        idcg = 0.0
        for rank, y in enumerate(labels_sorted, start=1):
            idcg += _gain(y) / math.log2(rank + 1)

        if idcg > 0.0:
            ndcgs.append(dcg / idcg)

    if not ndcgs:
        return 0.0, 0, (sum(covs) / len(covs) if covs else 0.0)
    return float(sum(ndcgs) / len(ndcgs)), len(ndcgs), float(sum(covs) / len(covs))

# ============ 主训练（排序三元组 H–M / M–L + 渐进日程 + 按 anchor 打包 + 统一编码评估） ============
def main(cfg_path: str):
    """
    关键说明：
    - 采样：不再按阈值分层，而是基于同一 anchor 的标注分数排序构造 (H–M) 与 (M–L) 三元组；
            若仅有两档，退化为 (H–L)。
    - 日程：前期稳定（w=1、pair-only 关闭、unsup 线性从 0→0.2、neg_per_pos=1），
            中后期逐步提高 λ_pair/alpha/neg_per_pos，择机开启轻量 pair-only。
    - 训练批：优先把同一 anchor 的多条三元组打到同一小批，降低 unique-ID，减少碎片/交叉传输。
    - 评估：统一编码 + 矩阵化计算 PairReg（保留 encode_ids_fast_safe）；Triplet/Spearman 在 final-best 再补评。
    - 仍保留：预检(preflight)、OOM 自降级（拆批/降 uniq-ID）、最优轮保存、metrics.csv 逐轮写入。
    """
    import os, time, math, random, platform
    from pathlib import Path
    import yaml, torch, numpy as np, pandas as pd
    import torch.nn.functional as F
    from tqdm import tqdm

    # ---- TF32（和 fp16 无关）----
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # ================= 基础小工具 =================
    def _tree_apply(obj, fn):
        if torch.is_tensor(obj):
            return fn(obj)
        if isinstance(obj, dict):
            return {k: _tree_apply(v, fn) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_tree_apply(x, fn) for x in obj)
        return obj

    def _gpu_mem_stats(device):
        if device.type != "cuda": return (0, 0, 0)
        return (torch.cuda.memory_allocated(device),
                torch.cuda.memory_reserved(device),
                torch.cuda.max_memory_allocated(device))

    def _pretty_mb(b):  # 字节→MB
        return f"{b/1024/1024:.1f}MB"

    def _device_total_mem(device):
        if device.type != "cuda": return 0
        return torch.cuda.get_device_properties(device).total_memory
    
    def _bin_label(score, hi, lo):
        if score >= hi:
            return "hi"
        elif score <= lo:
            return "lo"
        else:
            return "mid"

    def print_score_bin_stats(score_map, hi, lo, name="[unknown]"):
        from collections import Counter
        bins = [_bin_label(s, hi, lo) for s in score_map.values()]
        cnt = Counter(bins)
        total = max(1, sum(cnt.values()))
        hi_n = cnt.get("hi", 0)
        mid_n = cnt.get("mid", 0)
        lo_n = cnt.get("lo", 0)

        print(f"[ScoreDist-{name}] total={total} | "
            f"hi={hi_n}({hi_n/total:.1%}), mid={mid_n}({mid_n/total:.1%}), lo={lo_n}({lo_n/total:.1%})")
    
    def build_balanced_val_score_map(
        train_score_map: dict,
        val_score_map: dict,
        hi_thresh: float,
        lo_thresh: float,
        max_pairs: int = None,
        seed: int = 2025,
    ):
        """
        根据【train 的 hi/mid/lo 比例】，
        从 val_score_map 中抽样，构造一个“分布匹配”的子集，用于评估。

        - max_pairs: 限制平衡子集的总 pair 数；为 None 则不限制。
        """
        import random
        from collections import defaultdict

        if not val_score_map or not train_score_map:
            return val_score_map

        rnd = random.Random(seed)

        def _bin(score):
            if score >= hi_thresh:
                return "hi"
            elif score <= lo_thresh:
                return "lo"
            else:
                return "mid"

        # 1) 统计 train 中各 bin 的比例
        train_bins = [_bin(s) for s in train_score_map.values()]
        tot_tr = float(len(train_bins))
        p_hi = train_bins.count("hi") / tot_tr
        p_mid = train_bins.count("mid") / tot_tr
        p_lo = train_bins.count("lo") / tot_tr
        # 防止某一类为 0 的极端情况
        eps = 1e-6
        p_sum = max(p_hi + p_mid + p_lo, eps)
        p_hi, p_mid, p_lo = p_hi / p_sum, p_mid / p_sum, p_lo / p_sum

        print(f"[BalancedVal] target ratios from train: hi={p_hi:.2f}, mid={p_mid:.2f}, lo={p_lo:.2f}")

        # 2) 将 val 的 pair 按 bin 分桶
        bucket = defaultdict(list)  # {"hi": [(a,c,y),...], ...}
        for (a, c), y in val_score_map.items():
            b = _bin(y)
            bucket[b].append((a, c, y))

        n_hi = len(bucket["hi"])
        n_mid = len(bucket["mid"])
        n_lo = len(bucket["lo"])
        n_total_val = n_hi + n_mid + n_lo
        if n_total_val == 0:
            return val_score_map

        print(f"[BalancedVal] val raw counts: hi={n_hi}, mid={n_mid}, lo={n_lo}, total={n_total_val}")

        # 3) 决定 balanced 子集的总规模
        if max_pairs is None:
            # 不限制的话，就按 val 中数量最少的那一类来缩放
            max_pairs = n_total_val

        # 每一类的目标数量
        target_hi = int(max_pairs * p_hi)
        target_mid = int(max_pairs * p_mid)
        target_lo = int(max_pairs * p_lo)

        # 不能超过该 bin 在 val 中的实际数量
        target_hi = min(target_hi, n_hi)
        target_mid = min(target_mid, n_mid)
        target_lo = min(target_lo, n_lo)

        # 如果某一类 val 中数量严重不足，就按剩余类再均匀调整一下
        # 确保总数不超过 max_pairs
        total_target = target_hi + target_mid + target_lo
        if total_target < max_pairs:
            # 把多出来的份额，按有余量的 bin 再分一些
            deficit = max_pairs - total_target
            # 简单一点：轮流加到数量没饱和的 bin 上
            for _ in range(deficit):
                can_hi = (target_hi < n_hi)
                can_mid = (target_mid < n_mid)
                can_lo = (target_lo < n_lo)
                choices = []
                if can_hi: choices.append("hi")
                if can_mid: choices.append("mid")
                if can_lo: choices.append("lo")
                if not choices:
                    break
                b = rnd.choice(choices)
                if b == "hi":
                    target_hi += 1
                elif b == "mid":
                    target_mid += 1
                else:
                    target_lo += 1

        print(f"[BalancedVal] target counts: hi={target_hi}, mid={target_mid}, lo={target_lo}, total={target_hi+target_mid+target_lo}")

        # 4) 按目标数量从各桶里随机抽样
        def _sample(lst, k):
            if k >= len(lst):
                return lst[:]
            return rnd.sample(lst, k)

        chosen = []
        chosen += _sample(bucket["hi"],  target_hi)
        chosen += _sample(bucket["mid"], target_mid)
        chosen += _sample(bucket["lo"],  target_lo)

        # 5) 重新构造一个 dict
        balanced_map = {}
        for (a, c, y) in chosen:
            balanced_map[(a, c)] = y

        print(f"[BalancedVal] final balanced val pairs={len(balanced_map)}")
        return balanced_map


    # ================= 读配置与随机种子 =================
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))

    # ---- 路径与模型参数 ----
    pkl_path       = cfg["data"]["pkl"]
    index_csv      = cfg["data"]["index"]
    pair_csv       = cfg["data"]["pair_scores_csv"]
    turn_map_json  = cfg["data"].get("turn_label_mapping", None)

    node_dim   = int(cfg["model"]["node_dim"])
    hidden     = int(cfg["model"]["hidden"])
    emb_dim    = int(cfg["model"]["emb_dim"])
    n_heads    = int(cfg["model"]["n_heads"])

    # ---- 训练参数（含默认渐进日程）----
    lr             = float(cfg["train"].get("lr", 3e-4))
    epochs         = int(cfg["train"].get("epochs", 30))
    steps_per_ep   = int(cfg["train"].get("steps_per_epoch", 200))
    train_ratio    = float(cfg["train"].get("train_ratio", 0.8))

    # 旧阈值参数保留以向后兼容（不用于新采样）
    pos_high_thresh = float(cfg["train"].get("pos_high_thresh", 0.8))
    neg_low_thresh  = float(cfg["train"].get("neg_low_thresh", 0.3))

    # 渐进日程（如未在 YAML 指定则采用此默认）
    sched = cfg["train"].get("schedule", {}) or {}
    warmup_epochs       = int(sched.get("warmup_epochs", 5))
    neg_per_pos_after   = int(sched.get("neg_per_pos", {}).get("after", 6))
    neg_per_pos_value   = int(sched.get("neg_per_pos", {}).get("value", 2))
    unsup_start         = float(sched.get("unsup_ratio", {}).get("start", 0.0))
    unsup_end           = float(sched.get("unsup_ratio", {}).get("end", 0.2))
    unsup_until         = int(sched.get("unsup_ratio", {}).get("until_epoch", 12))
    unsup_freeze_epochs = int(sched.get("unsup_freeze_epochs", 8))  # 新增：前几轮不引入无监督
    alpha_fixed       = float(cfg["train"].get("alpha", 0.5))
    lambda_pair_fixed = float(cfg["train"].get("lambda_pair", 0.5))
    weight_from_epoch = int(cfg["train"].get("weight_from_epoch", 3))
    # 评估/批处理/显存
    bs_triplet_cfg   = int(cfg["train"].get("bs_triplet", 96))
    max_uniq_ids_cfg = int(cfg["train"].get("max_uniq_ids", 96))
    grad_accum       = int(cfg["train"].get("grad_accum_steps", 1))
    save_every       = int(cfg["train"].get("save_every", 5))
    eval_every       = int(cfg["train"].get("eval_every", 2))  # 建议每2轮评一次

    eval_bs        = int(cfg.get("eval", {}).get("batch_size", 64))
    eval_train_cap = int(cfg.get("eval", {}).get("train_cap", 200))
    eval_val_cap   = int(cfg.get("eval", {}).get("val_cap",   200))

    _mem_guard_cfg    = cfg["train"].get("mem_guard", {}) or {}
    empty_cache_every = int(_mem_guard_cfg.get("empty_cache_every", 20))
    warn_r            = float(_mem_guard_cfg.get("warn_ratio", 0.88))
    hard_r            = float(_mem_guard_cfg.get("hard_ratio", 0.92))

    # Pair 回归损失类型
    pair_loss_name = cfg["train"].get("pair_loss", "huber")

    # margin 与 alpha
    use_adapt   = bool(cfg["train"].get("use_adaptive_margin", True))
    m_cfg       = float(cfg["train"].get("margin", 0.1))
    clip_min, clip_max = (cfg["train"].get("adaptive_margin_clip", [0.1, 10.0]) or [0.1, 10.0])

    ckpt_dir = Path(cfg["train"]["ckpt_dir"]); ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ================= 标注装载与“排序桶” =================
    def _build_name_id_map(index_csv: str):
        df = pd.read_csv(index_csv, dtype={"scene_idx": int, "scene_name": str})
        name2id = dict(zip(df["scene_name"].astype(str), df["scene_idx"].astype(int)))
        id2name = dict(zip(df["scene_idx"].astype(int), df["scene_name"].astype(str)))
        return name2id, id2name

    def _load_labeled_pairs_as_ids(pair_csv: str, index_csv: str):
        df = pd.read_csv(pair_csv)
        cols = {c.lower(): c for c in df.columns}
        a = cols.get("anchor"); b = cols.get("candidate"); s = cols.get("final_score")
        if not (a and b and s):
            raise ValueError(f"{pair_csv} 需包含列 ['anchor','candidate','final_score']，当前列={list(df.columns)}")
        df = df[[a, b, s]].copy(); df.columns = ["anchor", "candidate", "final_score"]
        df["final_score"] = pd.to_numeric(df["final_score"], errors="coerce").clip(0.0, 1.0)
        df = df.dropna(subset=["final_score"]).reset_index(drop=True)
        name2id, _ = _build_name_id_map(index_csv)
        df["a_id"] = df["anchor"].map(lambda x: name2id.get(str(x)))
        df["c_id"] = df["candidate"].map(lambda x: name2id.get(str(x)))
        df = df.dropna(subset=["a_id", "c_id"]).copy()
        df["a_id"] = df["a_id"].astype(int); df["c_id"] = df["c_id"].astype(int)
        score_map = {(int(r.a_id), int(r.c_id)): float(r.final_score) for _, r in df.iterrows()}
        return df, score_map

    # ====== 从标注对生成“排序桶”（每个 anchor 一个降序列表）======
    def build_rank_buckets_by_anchor(df_pairs: pd.DataFrame):
        by_anchor = {}
        for a, g in df_pairs.groupby("a_id"):
            lst = [(int(r.c_id), float(r.final_score)) for _, r in g.iterrows()]
            if len(lst) >= 2:
                by_anchor[int(a)] = sorted(lst, key=lambda x: -x[1])
        return by_anchor

    def restrict_rank_buckets(by_anchor: dict, kept_ids):
        kept = set(kept_ids)
        out = {}
        for a, lst in by_anchor.items():
            if a not in kept: 
                continue
            sub = [(c, y) for (c, y) in lst if c in kept]
            if len(sub) >= 2:
                out[a] = sub
        return out

    # ====== 基于排序桶构造 (H–M) 与 (M–L)；两档时退化为 (H–L) ======
    def sample_epoch_triplets_ranked(
        by_anchor: dict,
        steps: int,
        rng: random.Random,
        use_weight: bool,
        weight_eps: float = 1e-3,
        hi_thresh: float = 0.7,   # 对应 pos_high_thresh
        low_thresh: float = 0.3,  # 对应 neg_low_thresh
    ):
        if steps <= 0 or not by_anchor:
            return []

        anchors = list(by_anchor.keys())
        rng.shuffle(anchors)
        out = []
        i = 0

        while len(out) < steps and i < 4 * len(anchors):
            a = anchors[i % len(anchors)]
            lst = by_anchor[a]  # [(cand_id, score), ...]
            if len(lst) < 2:
                i += 1
                continue

            # === 按分数切 hi / mid / lo ===
            hi, mid, lo = [], [], []
            for c, s in lst:
                if s >= hi_thresh:
                    hi.append((c, s))
                elif s <= low_thresh:
                    lo.append((c, s))
                else:
                    mid.append((c, s))

            # 1) 优先采 H–M
            if hi and mid and len(out) < steps:
                c_h, s_h = rng.choice(hi)
                c_m, s_m = rng.choice(mid)
                w = (max(s_h - s_m, 0.0) + weight_eps) ** 0.5 if use_weight else 1.0
                out.append((a, c_h, c_m, float(w)))

            # 2) 再采 M–L
            if mid and lo and len(out) < steps:
                c_m, s_m = rng.choice(mid)
                c_l, s_l = rng.choice(lo)
                w = (max(s_m - s_l, 0.0) + weight_eps) ** 0.5 if use_weight else 1.0
                out.append((a, c_m, c_l, float(w)))

            # 3) 如果没有 mid，但 hi+lo 都有，就兜底 H–L
            if (not mid) and hi and lo and len(out) < steps:
                c_h, s_h = rng.choice(hi)
                c_l, s_l = rng.choice(lo)
                w = (max(s_h - s_l, 0.0) + weight_eps) ** 0.5 if use_weight else 1.0
                out.append((a, c_h, c_l, float(w)))

            i += 1

        rng.shuffle(out)
        return out[:steps]

    def make_unsup_triplets(id_list, want_n, weight_unsup, rng_obj):
        out = []
        if not id_list: return out
        for _ in range(want_n):
            a = rng_obj.choice(id_list)
            neg_pool = [x for x in id_list if x != a]
            if not neg_pool: continue
            n = rng_obj.choice(neg_pool)
            out.append((a, a, n, float(weight_unsup)))  # 无监督用 (a,a,n)
        return out

    # ================= 数据集与分割 =================
    ds = NeutralSceneDataset(
        pkl_path=pkl_path, index_csv=index_csv,
        use_geo_bias=True, turn_map_json=turn_map_json,
        K_turn=int(cfg["data"].get("K_turn", 19))
    )
    num_scenes = len(ds)

    df_pairs, score_map_all = _load_labeled_pairs_as_ids(pair_csv, index_csv)
    # —— 从标注两端估计 R_label，并裁剪 margin —— 
    def _trim_mean(xs, p=0.05):
        if not xs: return None
        xs = np.sort(np.asarray(xs, dtype=float))
        lo = int(len(xs) * p); hi = int(len(xs) * (1 - p))
        xs = xs[lo:max(lo + 1, hi)]
        return float(xs.mean())

    pos_scores = df_pairs.loc[df_pairs["final_score"] >= pos_high_thresh, "final_score"].tolist()
    neg_scores = df_pairs.loc[df_pairs["final_score"] <= neg_low_thresh,  "final_score"].tolist()
    s_pos = _trim_mean(pos_scores, 0.05); s_neg = _trim_mean(neg_scores, 0.05)
    eps = 1e-6
    R_label = 1.5 if (s_pos is None or s_neg is None) else max(s_pos / max(s_neg, eps), 1.01)

    alpha_now = alpha_fixed  # 初始 α（后续随日程变）
    m_from_labels = math.log(R_label) / max(alpha_now, 1e-6)
    m_use = float(np.clip(m_from_labels, clip_min, clip_max)) if use_adapt else m_cfg
    print(f"[MarginUse] m={m_use:.3f} (α={alpha_now}, R≈{R_label:.3f}, clip=[{clip_min},{clip_max}])")

    # 分层划分（复用你原项目逻辑）
    def stratified_split_with_labels(num_scenes, df_pairs, base_train_ratio=0.8, seed=42):
        rng = random.Random(seed)
        all_ids = list(range(num_scenes))
        labeled_anchors = sorted(set(df_pairs["a_id"].tolist()))
        rng.shuffle(labeled_anchors)
        keep_val = int(max(40, len(labeled_anchors) * 0.2))
        val_anchors = set(labeled_anchors[:keep_val])
        val_ids = set(val_anchors)
        target_val = int(round(num_scenes * (1 - base_train_ratio)))
        rest = [x for x in all_ids if x not in val_ids]; rng.shuffle(rest)
        val_ids = set(list(val_ids) + rest[:max(0, target_val - len(val_ids))])
        train_ids = [x for x in all_ids if x not in val_ids]
        return sorted(train_ids), sorted(list(val_ids))

    train_ids, val_ids = stratified_split_with_labels(num_scenes, df_pairs, base_train_ratio=train_ratio)
    by_anchor_all = build_rank_buckets_by_anchor(df_pairs)
    by_anchor_tr  = restrict_rank_buckets(by_anchor_all, train_ids)
    by_anchor_va  = restrict_rank_buckets(by_anchor_all, val_ids)

    # 评估所需的 map（同你原逻辑）
    train_idset, val_idset = set(train_ids), set(val_ids)
    train_score_map = {k: v for k, v in score_map_all.items() if (k[0] in train_idset and k[1] in train_idset)}
    val_score_map   = {k: v for k, v in score_map_all.items() if (k[0] in val_idset   and k[1] in val_idset)}
    print(f"[Split] train={len(train_ids)} val={len(val_ids)} | train_pairs={len(train_score_map)} val_pairs={len(val_score_map)}")
    print_score_bin_stats(train_score_map, pos_high_thresh, neg_low_thresh, name="train")
    print_score_bin_stats(val_score_map,   pos_high_thresh, neg_low_thresh, name="val")
    # 保留一份“原始 val 标注”，以便需要时对比
    val_score_map_raw = dict(val_score_map)

    # 构造一个“分布匹配”的平衡版 val 标注子集，用于评估
    val_score_map_balanced = build_balanced_val_score_map(
        train_score_map, val_score_map,
        hi_thresh=pos_high_thresh,
        lo_thresh=neg_low_thresh,
        max_pairs=int(cfg.get("eval", {}).get("val_pairs_cap", 4000)),  # 比如最多 4000 对
        seed=2025,
    )

    # 再打印一下平衡后的分布看一眼
    print_score_bin_stats(val_score_map_balanced, pos_high_thresh, neg_low_thresh, name="val_bal")
    # ================= 模型/损失/优化器 =================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_edge_types = infer_num_edge_types(ds)
    net = SceneEncoder(
        node_dim=node_dim, hidden=hidden, emb_dim=emb_dim, n_heads=n_heads,
        n_layers_spatial=int(cfg["model"].get("n_layers_spatial", 3)),
        n_layers_temporal=int(cfg["model"].get("n_layers_temporal", 2)),
        num_edge_types=int(num_edge_types),
        use_geo_bias=True, geo_dim=int(cfg["model"].get("geo_dim", 3)),
        renorm_embeddings=bool(cfg["model"].get("renorm_embeddings", False)), renorm_eps=1e-8,
        nan_guard=bool(cfg["model"].get("nan_guard", True))
    ).to(device)

    if bool(cfg["train"].get("compile", False)) and hasattr(torch, "compile"):
        if platform.system() != "Windows":
            try:
                net = torch.compile(net, mode=cfg["train"].get("compile_mode", "max-autotune"),
                                    backend=cfg["train"].get("compile_backend", None))
                print("[Compile] torch.compile enabled.")
            except Exception as e:
                print("[Compile] fallback:", repr(e))

    loss_fn = HybridTripletSimilarityLoss(
        alpha=float(alpha_fixed), margin=float(m_use),
        lambda_pair=float(lambda_pair_fixed), pair_loss=pair_loss_name,
        triplet_space="euclid", pair_space="exp"
    )

    optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    # ================= 数据清洁 & 预检 =================
    print("[DataCheck] scan non-finite ...")
    scan_dataset_for_nonfinite(ds, train_ids, device=device, use_clean=True, max_print=50)
    scan_dataset_for_nonfinite(ds, val_ids,   device=device, use_clean=True, max_print=50)

    def run_preflight(net, ds, train_ids, device, cfg):
        pf = cfg["train"].get("preflight", {})
        if not bool(pf.get("enabled", False)): return True, None
        steps    = int(pf.get("steps", 100))
        bs_try   = int(pf.get("start_bs_triplet", cfg["train"].get("bs_triplet", 96)))
        maxuniq  = int(pf.get("start_max_uniq_ids", cfg["train"].get("max_uniq_ids", 96)))
        fwd_only = bool(pf.get("forward_only", True))
        print(f"[Preflight] steps={steps} bs={bs_try} max_uniq_ids={maxuniq} fwd_only={fwd_only}")

        rng = random.Random(2025)
        # 随机从排序桶构造少量三元组用于显存压测
        trips = sample_epoch_triplets_ranked(by_anchor_tr, steps*max(1, bs_try//4), rng, use_weight=False)

        def pack_minibatches_group_by_anchor(trips, max_trips, max_uniq_ids):
            # trips: [(a, p, n, w), ...]
            from collections import defaultdict
            buckets = defaultdict(list)
            for a,p,n,w in trips:
                buckets[a].append((a,p,n,w))
            batches, cur, cur_ids = [], [], set()
            for a, lst in buckets.items():
                for t in lst:
                    add = {t[0], t[1], t[2]}
                    if (len(cur) < max_trips) and (len(cur_ids | add) <= max_uniq_ids):
                        cur.append(t); cur_ids |= add
                    else:
                        if cur: 
                            batches.append(cur)
                        cur, cur_ids = [t], set(add)
            if cur: 
                batches.append(cur)
            return batches

        batches = pack_minibatches_group_by_anchor(trips, bs_try, maxuniq)[:steps]
        ok, note = True, None
        for bi, batch in enumerate(batches, 1):
            try:
                uniq_ids = sorted({sid for (a,p,n,_) in batch for sid in (a,p,n)})
                items = [ds[j] for j in uniq_ids]
                items = _tree_apply(items, lambda t: t.to(device) if torch.is_tensor(t) else t)
                Z = net(items)
                if Z.dim() > 2: Z = Z.view(Z.size(0), -1)
                # 只做前向/可选反传
                if not fwd_only:
                    Z.sum().backward()
                    for p in net.parameters():
                        if p.grad is not None:
                            p.grad.detach_(); p.grad.zero_()
                if (device.type == "cuda") and (bi % max(1, int(empty_cache_every)) == 0):
                    alloc, resv, peak = _gpu_mem_stats(device)
                    print(f"[Preflight] batch {bi}/{len(batches)} alloc={_pretty_mb(alloc)} resv={_pretty_mb(resv)} peak={_pretty_mb(peak)} uniq={len(uniq_ids)}")
            except RuntimeError as e:
                msg = str(e).lower()
                ok = False; note = f"RuntimeError at preflight batch {bi}: {repr(e)}"
                if "out of memory" in msg and device.type == "cuda":
                    torch.cuda.empty_cache()
                break
        if ok and device.type == "cuda":
            _, _, peak = _gpu_mem_stats(device)
            tot = _device_total_mem(device)
            print(f"[Preflight-PASS] peak={_pretty_mb(peak)} / total={_pretty_mb(tot)} ({peak/max(1,tot):.1%})")
        return ok, note

    cfg.setdefault("train", {})["margin"] = float(m_use)
    ok, note = run_preflight(net, ds, train_ids, device, cfg)
    if not ok:
        print("[Abort before training]", note); return

    # ================= 评估相关（统一编码版本） =================
    def encode_ids_fast_safe(
        net, ds, ids, device,
        bs=32, fp16=True, pin=True, non_blocking=True, autotune=True,
        mem_guard=(0.88,0.92), min_bs=1
    ):
        net.eval()
        outs = []
        if len(ids) == 0:
            return torch.empty(0, getattr(net, "emb_dim", 128)).cpu()
        bs_cur = max(min_bs, int(bs))
        i, oom_cnt = 0, 0
        total_mem = float(torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else 0.0

        def _maybe_pin(itm):
            if not (pin and device.type == "cuda"): return itm
            return _tree_apply(itm, lambda t: t.pin_memory() if torch.is_tensor(t) and t.device.type == "cpu" else t)
        def _to_dev(itm):
            return _tree_apply(itm, lambda t: t.to(device, non_blocking=non_blocking) if torch.is_tensor(t) else t)
        if device.type == "cuda": torch.cuda.empty_cache()

        with torch.inference_mode():
            while i < len(ids):
                take = min(bs_cur, len(ids)-i)
                chunk = ids[i:i+take]
                try:
                    items = []
                    for j in chunk:
                        x = ds[j]
                        x = _maybe_pin(x); x = _to_dev(x); items.append(x)
                    with torch.autocast("cuda", torch.float16, enabled=(fp16 and device.type == "cuda")):
                        Z = net(items)
                        if Z.dim() > 2: Z = Z.view(Z.size(0), -1)
                    outs.append(Z.float().cpu())
                    i += take
                    if autotune and device.type == "cuda":
                        reserved = float(torch.cuda.memory_reserved(device))
                        ratio = reserved / max(1.0, total_mem)
                        if ratio > mem_guard[0]:
                            bs_cur = max(min_bs, int(bs_cur * 0.8))
                        elif ratio < 0.50 and bs_cur < bs:
                            bs_cur = min(bs, bs_cur + 1)
                except RuntimeError as e:
                    msg = str(e).lower()
                    if ("out of memory" in msg) or ("cuda error" in msg and "out of memory" in msg):
                        if device.type == "cuda": torch.cuda.empty_cache()
                        bs_cur = max(min_bs, bs_cur // 2); oom_cnt += 1
                        if oom_cnt >= 2: fp16 = False
                        if bs_cur < 1: raise
                        continue
                    else:
                        raise
        return torch.cat(outs, dim=0) if outs else torch.empty(0, getattr(net, "emb_dim", 128)).cpu()

    # ================= 训练循环 =================
    best_val = float("inf")
    best_epoch = -1
    rng = random.Random(123)
    metrics_rows = []
    run_t0 = time.time()

    def pack_minibatches_group_by_anchor(trips, max_trips, max_uniq_ids):
        """优先把同 anchor 的三元组装进同一批，减少 unique-ID。"""
        from collections import defaultdict
        buckets = defaultdict(list)
        for a,p,n,w in trips:
            buckets[a].append((a,p,n,w))
        batches, cur, cur_ids = [], [], set()
        for a, lst in buckets.items():
            for t in lst:
                add = {t[0], t[1], t[2]}
                if (len(cur) < max_trips) and (len(cur_ids | add) <= max_uniq_ids):
                    cur.append(t); cur_ids |= add
                else:
                    if cur: batches.append(cur)
                    cur, cur_ids = [t], set(add)
        if cur: batches.append(cur)
        return batches

    def schedule_update(ep):
        """
        只对 unsup_ratio / neg_per_pos 做日程；
        alpha / lambda_pair 固定，pair_reg_steps 从第 1 轮就生效。
        无监督策略：前 unsup_freeze_epochs 轮强制为 0，此后再线性拉升到 unsup_end。
        """
        def lin(start, end, n_steps, t_idx):
            """
            简单线性插值：
            - n_steps: 总步数（>=1）
            - t_idx: 当前是第几步（从 1 开始）
            """
            if n_steps <= 1:
                return end
            t_idx = max(1, min(t_idx, n_steps))
            return start + (end - start) * (t_idx - 1) / (n_steps - 1)

        # alpha / lambda_pair 全程使用固定值
        alpha_ep       = alpha_fixed
        lambda_pair_ep = lambda_pair_fixed

        # -------- 无监督比例：先冻结，再线性拉升 --------
        if ep <= unsup_freeze_epochs:
            unsup_ratio_ep = 0.0
        else:
            # 从 (unsup_freeze_epochs+1) 到 unsup_until 这段线性从 0 → unsup_end
            if ep >= unsup_until:
                unsup_ratio_ep = unsup_end
            else:
                # 映射到 1..n_steps 的插值步
                n_steps = max(1, unsup_until - unsup_freeze_epochs)
                t_idx   = ep - unsup_freeze_epochs
                unsup_ratio_ep = lin(0.0, unsup_end, n_steps, t_idx)

        # neg_per_pos 仍然可以按 epoch 切换
        neg_per_pos_ep = 1 if ep < neg_per_pos_after else neg_per_pos_value

        # 👉 pair-only：直接使用配置中的 pair_reg_steps，从第 1 轮就开始
        pair_steps_cfg = int(cfg["train"].get("pair_reg_steps", 0))
        pair_steps_ep  = pair_steps_cfg

        return float(alpha_ep), float(lambda_pair_ep), float(unsup_ratio_ep), int(neg_per_pos_ep), int(pair_steps_ep)



    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
        net.train()

        # —— 日程参数 —— 
        alpha_now, lambda_pair_now, unsup_ratio_now, neg_per_pos_now, pair_reg_steps_ep = schedule_update(ep)
        loss_fn.alpha = float(alpha_now)     # 更新 alpha
        loss_fn.lambda_pair = float(lambda_pair_now)

        # —— 训练三元组：排序采样 + 无监督 —— 
        sup_budget   = int(steps_per_ep * (1.0 - float(unsup_ratio_now)))
        unsup_budget = int(steps_per_ep - sup_budget)
        use_weight   = (ep >= weight_from_epoch) # 热身期权重=1，之后启用 sqrt(y_pos - y_neg)

        triplets_sup = sample_epoch_triplets_ranked(
            by_anchor_tr,
            steps=sup_budget,
            rng=rng,
            use_weight=use_weight,
            hi_thresh=pos_high_thresh,
            low_thresh=neg_low_thresh,
        )
        triplets_uns = make_unsup_triplets(train_ids, unsup_budget, weight_unsup=0.2, rng_obj=rng) if unsup_budget > 0 else []
        triplets     = triplets_sup + triplets_uns
        rng.shuffle(triplets)

        print(f"[Epoch {ep}] sup={len(triplets_sup)} unsup={len(triplets_uns)} total={len(triplets)} "
              f"| α={alpha_now:.2f} λ_pair={lambda_pair_now:.2f} unsup_ratio={unsup_ratio_now:.2f} neg_per_pos={neg_per_pos_now}")

        # —— 打包并训练（按 anchor 分组） —— 
        bs_triplet   = bs_triplet_cfg
        max_uniq_ids = max_uniq_ids_cfg
        batches = pack_minibatches_group_by_anchor(triplets, max_trips=bs_triplet, max_uniq_ids=max_uniq_ids)

        total_loss = 0.0; valid = 0; skipped = 0
        bi = 0; accum_cnt = 0; processed = 0
        pbar = tqdm(total=len(batches), ncols=120, ascii=True, desc=f"Epoch {ep}/{epochs}", leave=False)
        tot_mem = _device_total_mem(device) if device.type == "cuda" else 0

        def forward_triplet_minibatch(trip_batch):
            uniq_ids = sorted({sid for a,p,n,_ in trip_batch for sid in (a,p,n)})
            id2row = {sid: i for i, sid in enumerate(uniq_ids)}
            items = [ds[j] for j in uniq_ids]
            items = _tree_apply(items, lambda t: t.to(device) if torch.is_tensor(t) else t)
            Z = net(items)
            if Z.dim() > 2: Z = Z.view(Z.size(0), -1)
            ai = torch.tensor([id2row[a] for a,_,_,_ in trip_batch], device=device, dtype=torch.long)
            pi = torch.tensor([id2row[p] for _,p,_,_ in trip_batch], device=device, dtype=torch.long)
            ni = torch.tensor([id2row[n] for _,_,n,_ in trip_batch], device=device, dtype=torch.long)
            W  = torch.tensor([w for *_, w in trip_batch], device=device, dtype=torch.float32)
            za, zp, zn = Z[ai], Z[pi], Z[ni]
            d_ap = torch.norm(za - zp, p=2, dim=1)
            d_an = torch.norm(za - zn, p=2, dim=1)
            # 标注分数（若存在）
            y_ap = [score_map_all.get((a,p)) for a,p,_,_ in trip_batch]
            y_an = [score_map_all.get((a,n)) for a,_,n,_ in trip_batch]
            m_ap = torch.tensor([y is not None for y in y_ap], device=device)
            m_an = torch.tensor([y is not None for y in y_an], device=device)
            Y_ap = torch.tensor([y if y is not None else 0.0 for y in y_ap], device=device, dtype=torch.float32) if bool(m_ap.any().item()) else None
            Y_an = torch.tensor([y if y is not None else 0.0 for y in y_an], device=device, dtype=torch.float32) if bool(m_an.any().item()) else None
            return d_ap, d_an, W, (Y_ap, m_ap), (Y_an, m_an), len(uniq_ids)

        while bi < len(batches):
            trip_batch = batches[bi]
            retry_splits = 0
            while True:
                try:
                    if device.type == "cuda":
                        alloc, resv, peak = _gpu_mem_stats(device)
                        ratio = resv / max(1, tot_mem)
                        if ratio >= hard_r and len(trip_batch) >= 2:
                            max_uniq_ids = max(16, max_uniq_ids // 2)
                            mid = len(trip_batch) // 2
                            sub1, sub2 = trip_batch[:mid], trip_batch[mid:]
                            batches[bi:bi+1] = [sub1, sub2]
                            torch.cuda.empty_cache()
                            if pbar.total != len(batches):
                                pbar.total = len(batches); pbar.refresh()
                            trip_batch = batches[bi]; continue
                        elif ratio >= warn_r:
                            max_uniq_ids = max(16, int(max_uniq_ids * 0.8))

                    d_ap, d_an, W, (Y_ap, m_ap), (Y_an, m_an), uniq_cnt = forward_triplet_minibatch(trip_batch)

                    # —— 损失：Triplet + Pair（仅在有标注时）——
                    triplet_term = torch.clamp(loss_fn.margin + d_ap - d_an, min=0.0)
                    loss_triplet = (triplet_term * W).mean()
                    pair_term = 0.0
                    if (Y_ap is not None) and bool(m_ap.any().item()):
                        s_ap = torch.exp(torch.clamp(-float(alpha_now) * d_ap, min=-50.0, max=0.0))
                        pair_term = pair_term + (F.mse_loss(s_ap[m_ap], Y_ap[m_ap]) if pair_loss_name == "mse"
                                                else F.smooth_l1_loss(s_ap[m_ap], Y_ap[m_ap]))
                    if (Y_an is not None) and bool(m_an.any().item()):
                        s_an = torch.exp(torch.clamp(-float(alpha_now) * d_an, min=-50.0, max=0.0))
                        pair_term = pair_term + (F.mse_loss(s_an[m_an], Y_an[m_an]) if pair_loss_name == "mse"
                                                else F.smooth_l1_loss(s_an[m_an], Y_an[m_an]))

                    loss = loss_triplet + float(lambda_pair_now) * pair_term

                    # —— 优化 & 累积 —— 
                    if (accum_cnt == 0):
                        optim.zero_grad(set_to_none=True)
                    loss.backward()
                    if ((accum_cnt + 1) == grad_accum):
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                        optim.step(); optim.zero_grad(set_to_none=True)
                        accum_cnt = 0
                    else:
                        accum_cnt += 1

                    total_loss += float(loss.detach()); valid += 1
                    if (processed % 20 == 0) and device.type == "cuda":
                        torch.cuda.empty_cache()
                    processed += 1
                    break
                except RuntimeError as e:
                    msg = str(e).lower()
                    if ("out of memory" in msg) and (retry_splits < 3) and (len(trip_batch) >= 2):
                        retry_splits += 1
                        mid = len(trip_batch) // 2
                        sub1, sub2 = trip_batch[:mid], trip_batch[mid:]
                        batches[bi:bi+1] = [sub1, sub2]
                        max_uniq_ids = max(16, max_uniq_ids // 2)
                        if device.type == "cuda": torch.cuda.empty_cache()
                        print(f"[Train OOM] split batch -> {len(sub1)}+{len(sub2)} | new max_uniq_ids={max_uniq_ids}")
                        if pbar.total != len(batches):
                            pbar.total = len(batches); pbar.refresh()
                        trip_batch = batches[bi]; continue
                    else:
                        skipped += 1
                        print("[Train RuntimeError]", repr(e))
                        if device.type == "cuda": torch.cuda.empty_cache()
                        break
            bi += 1; pbar.update(1)
        pbar.close()

        train_avg = total_loss / max(1, valid)
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated(device)
            print(f"[Mem] epoch_peak_alloc={_pretty_mb(peak)}")
        print(f"[Train Epoch {ep:03d}] avg_loss={train_avg:.4f} skip={skipped}")

        # ====== 轻量 Pair-only（按日程开启；默认 12 轮后） ======
        pair_batch = int(cfg["train"].get("pair_batch", 64))
        pair_reg_train_val = float("nan")
        if pair_reg_steps_ep > 0 and len(train_score_map) > 0:
            labeled_pairs = list(train_score_map.items()); rng.shuffle(labeled_pairs)
            used = 0; total_pair_loss = 0.0
            while (used < pair_reg_steps_ep and used < len(labeled_pairs)):
                batch = labeled_pairs[used: used + pair_batch]; used += len(batch)
                batch_loss = 0.0
                for ((a, c), y) in batch:
                    aitem = ds[a]; citem = ds[c]
                    aitem = _tree_apply(aitem, lambda t: t.to(device) if torch.is_tensor(t) else t)
                    citem = _tree_apply(citem, lambda t: t.to(device) if torch.is_tensor(t) else t)
                    emb_ac = net([aitem, citem])
                    if emb_ac.dim() == 1: emb_ac = emb_ac.unsqueeze(0)
                    elif emb_ac.dim() > 2: emb_ac = emb_ac.view(emb_ac.size(0), -1)
                    za, zc = emb_ac[0], emb_ac[1]
                    d = torch.norm(za - zc, p=2)
                    s_pred = torch.exp(torch.clamp(-float(alpha_now) * d, min=-50.0, max=0.0)).view(1)
                    y_t = torch.as_tensor([y], device=device, dtype=torch.float32)
                    loss_p = (F.smooth_l1_loss(s_pred, y_t) if pair_loss_name == "huber" else F.mse_loss(s_pred, y_t))
                    optim.zero_grad(set_to_none=True)
                    loss_p.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optim.step()
                    batch_loss += float(loss_p.item())
                total_pair_loss += batch_loss
            pair_reg_train_val = total_pair_loss / max(1, used)
            print(f"[Pair-only] steps={min(pair_reg_steps_ep, len(labeled_pairs))} avg_pair_loss={pair_reg_train_val:.4f}")

        # ====== 验证（统一编码；每 eval_every 轮执行） ======
        pair_reg_val = float("nan")
        val_triplet_loss = float("nan")
        val_triplet_ok   = float("nan")
        sp_val           = float("nan")

        if (ep % max(1, eval_every)) == 0:
            try:
                if device.type == "cuda": torch.cuda.empty_cache()
                # ✅ val 用平衡后的标注 anchor，train 用自己的标注 anchor
                labeled_anchors_val   = sorted({a for (a, _) in val_score_map_balanced.keys()})
                labeled_anchors_train = sorted({a for (a, _) in train_score_map.keys()})

                def take_with_label_first(pool_ids, labeled_ids, cap):
                    labeled = [i for i in pool_ids if i in labeled_ids]
                    unlabeled = [i for i in pool_ids if i not in labeled_ids]
                    head = labeled[:cap]
                    if len(head) < cap:
                        need = cap - len(head)
                        tail = unlabeled[:need] if len(unlabeled) >= need else unlabeled
                        return head + tail
                    return head

                net.eval()
                va_ids_eval = take_with_label_first(val_ids,   set(labeled_anchors_val),   eval_val_cap)
                tr_ids_eval = take_with_label_first(train_ids, set(labeled_anchors_train), eval_train_cap)

                # —— 编码 —— 
                E_va = encode_ids_fast_safe(
                    net, ds, va_ids_eval, device, bs=eval_bs,
                    fp16=bool(cfg["train"].get("amp", True)),
                    pin=True, non_blocking=True, autotune=True, mem_guard=(0.88, 0.92)
                )
                pair_reg_val, _ = pair_reg_from_E(
                    E_va, va_ids_eval, val_score_map_balanced,
                    alpha=float(alpha_now), pair_loss=pair_loss_name
                )

                # 训练侧 PairReg（监控用）
                if len(train_score_map) > 0:
                    E_tr = encode_ids_fast_safe(
                        net, ds, tr_ids_eval, device, bs=eval_bs,
                        fp16=bool(cfg["train"].get("amp", True)),
                        pin=True, non_blocking=True, autotune=True, mem_guard=(0.88, 0.92)
                    )
                    pair_reg_train_val, _ = pair_reg_from_E(
                        E_tr, tr_ids_eval, train_score_map,
                        alpha=float(alpha_now), pair_loss=pair_loss_name
                    )

                # 最优更新
                if np.isfinite(pair_reg_val) and (pair_reg_val < best_val):
                    best_val   = pair_reg_val
                    best_epoch = ep
                    torch.save(net.state_dict(), ckpt_dir / "scene_encoder_best.pth")

                if device.type == "cuda":
                    if 'E_tr' in locals(): del E_tr
                    del E_va
                    torch.cuda.empty_cache()
            except Exception as e:
                print("[Eval Warning]", repr(e))

        # —— 写入 metrics.csv（每轮一行）——
        epoch_seconds  = time.time() - ep_t0
        total_time_min = (time.time() - run_t0) / 60.0
        row = {
            "epoch": ep,
            "train_loss": float(train_avg),
            "pair_reg_train_eval": float(pair_reg_train_val),
            "pair_reg_val": float(pair_reg_val),
            "val_triplet_loss": float(val_triplet_loss),
            "val_triplet_ok": float(val_triplet_ok),
            "spearman_val": float(sp_val),
            "sup": int(len(triplets_sup)),
            "unsup": int(len(triplets_uns)),
            "alpha": float(alpha_now),
            "lambda_pair": float(lambda_pair_now),
            "unsup_ratio": float(unsup_ratio_now),
            "neg_per_pos": int(neg_per_pos_now),
            "time_per_epoch_sec": float(epoch_seconds),
            "total_time_min": float(total_time_min),
        }
        metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_csv(ckpt_dir / "metrics.csv", index=False, encoding="utf-8-sig")
        print(
            f"[Epoch {ep:03d}] train_loss={row['train_loss']:.4f} | "
            f"PairReg(train)={row['pair_reg_train_eval']:.4f} | PairReg(val)={row['pair_reg_val']:.4f} | "
            f"time/ep={epoch_seconds:.2f}s total={total_time_min:.2f}min"
        )

        # —— 持久化 —— 
        if (ep % save_every) == 0:
            torch.save(net.state_dict(), ckpt_dir / f"scene_encoder_e{ep:03d}.pth")

        if device.type == "cuda" and empty_cache_every > 0:
            torch.cuda.empty_cache()

    # ====== 训练完成：用“最优轮”补评 Triplet/Spearman 并回写 ======
    try:
        if best_epoch <= 0:
            print("[FinalEval] 未产生最优轮，跳过补充评估。")
        else:
            print(f"[FinalEval] 对最优轮 epoch={best_epoch} 进行补充评估 ...")
            state = torch.load(ckpt_dir / "scene_encoder_best.pth", map_location=device)
            net.load_state_dict(state); net.eval()
            if device.type == "cuda": torch.cuda.empty_cache()

            labeled_anchors_val = sorted({a for (a, _) in val_score_map_balanced.keys()})
            va_ids_eval = take_with_label_first(val_ids, set(labeled_anchors_val), eval_val_cap)
            E_va = encode_ids_fast_safe(
                net, ds, va_ids_eval, device, bs=eval_bs,
                fp16=bool(cfg["train"].get("amp", True)),
                pin=True, non_blocking=True, autotune=True, mem_guard=(0.88, 0.92)
            )

            # 用你项目中的工具评估 Triplet / Spearman
            # 注意：此处需要 restrict_pairs / sample_epoch_triplets / triplet_eval_from_E / spearman_from_E 已在项目中实现
            # 这里用最基础的正负采样评估（与原项目保持一致）
            pos_all = {}
            neg_all = {}
            for (a,c), y in val_score_map_balanced.items():
                if y >= pos_high_thresh:
                    pos_all.setdefault(a, []).append((c, y))
                elif y <= neg_low_thresh:
                    neg_all.setdefault(a, []).append((c, 1.0 - y))
            pos_va_sub, neg_va_sub = restrict_pairs(pos_all, neg_all, va_ids_eval)
            val_triplets = sample_epoch_triplets(
                va_ids_eval, pos_va_sub, neg_va_sub,
                steps=min(200, max(20, len(va_ids_eval) * 2)), rng=random.Random(2025)
            )
            final_val_loss, final_val_ok = triplet_eval_from_E(
                E_va, va_ids_eval, val_triplets,
                margin=float(loss_fn.margin), alpha=float(alpha_fixed), mode="euclid"
            )
            final_sp_val, _ = spearman_from_E(E_va, va_ids_eval, val_score_map_balanced, alpha=float(alpha_fixed))

            for r in metrics_rows:
                if int(r.get("epoch", -1)) == int(best_epoch):
                    r["val_triplet_loss"] = float(final_val_loss)
                    r["val_triplet_ok"]   = float(final_val_ok)
                    r["spearman_val"]     = float(final_sp_val)
                    break
            pd.DataFrame(metrics_rows).to_csv(ckpt_dir / "metrics.csv", index=False, encoding="utf-8-sig")
            if device.type == "cuda":
                del E_va
                torch.cuda.empty_cache()

    except Exception as e:
        print("[FinalEval Warning]", repr(e))

    print("[Training completed ✅]")

# ============ CLI ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="YAML 配置文件路径")
    args = parser.parse_args()
    main(args.cfg)