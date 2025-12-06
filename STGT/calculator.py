# -*- coding: utf-8 -*-
"""
通用：SceneEncoder 嵌入 + exp(-alpha*L2) 相似度（支持双数据集；支持从 YAML 读取结构与 alpha）

功能：
- 从“查询数据集（PKL+index）”中指定一个 query（id 或 name）
- 从“候选数据集（PKL+index）”中取全库或子集作为候选
- 用训练好的 SceneEncoder 生成嵌入，计算 sim = exp(-alpha * ||zq - zi||_2)
- 导出排序 CSV，并打印 Top-K

依赖：
- 你的项目中的 dataset.NeutralSceneDataset
- 你的项目中的 model.SceneEncoder
- PyTorch / pandas / numpy / tqdm / pyyaml

示例（同库匹配）：
python calculator.py \
  --cfg "/configs/scene_triplet.yaml" 
  --model "checkpoints_v2/scene_encoder_best.pth" \
  --query_name "***" \
  --out_csv "results/rank_same.csv" \
  --exclude_self \
  --cache_dir "emb_cache"

示例（双数据集 A→B）：
python calculator.py \
  --cfg "/configs/scene_triplet.yaml" \
  --qry_pkl "E:/A.pkl" --qry_index_csv "E:/A_index.csv" \
  --cand_pkl "E:/B.pkl" --cand_index_csv "E:/B_index.csv" \
  --model "checkpoints_v2/scene_encoder_best.pth" \
  --query_name "***" \
  --out_csv "results/rank_same.csv" \
  --batch_size 32 --cache_dir "emb_cache"
"""
import os, sys, csv, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import torch

# 你的工程内模块
from dataset import NeutralSceneDataset
from model import SceneEncoder

# ---------- 小工具 ----------
def _read_csv_robust(path: str) -> pd.DataFrame:
    try_enc = ["utf-8-sig", "utf-8"]
    for enc in try_enc:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                dialect = csv.Sniffer().sniff(f.read(4096), delimiters=",;|\t")
                f.seek(0)
                return pd.read_csv(f, encoding=enc, delimiter=dialect.delimiter)
        except Exception:
            continue
    return pd.read_csv(path, encoding="utf-8-sig")

def build_name_maps(index_csv: str):
    df = _read_csv_robust(index_csv)
    cols = {c.strip().lower(): c for c in df.columns}
    col_idx  = next((cols[k] for k in cols if k in ("scene_idx","idx","id")), None)
    col_name = next((cols[k] for k in cols if k in ("scene_name","name","filename","file","stem")), None)
    if not col_idx or not col_name:
        raise ValueError(f"index_csv 缺少 scene_idx/scene_name 列，实际列: {list(df.columns)}")
    name2idx, idx2name = {}, {}
    for _, r in df.iterrows():
        try:
            idx = int(r[col_idx])
        except Exception:
            continue
        name = str(r[col_name])
        stem = Path(name).stem
        for k in (name, Path(name).name, stem, str(idx)):
            name2idx.setdefault(k, idx)
        idx2name[idx] = stem
    return name2idx, idx2name

def resolve_query_id(query_id: int | None, query_name: str | None, name2idx: dict[str,int]) -> int:
    if query_id is not None:
        return int(query_id)
    if query_name:
        q = str(query_name).strip()
        if q in name2idx:            return int(name2idx[q])
        stem = Path(q).stem
        if stem in name2idx:         return int(name2idx[stem])
        base = Path(q).name
        if base in name2idx:         return int(name2idx[base])
    raise KeyError(f"无法解析 query：id={query_id}, name={query_name}")

def to_device_item(item: dict, device: torch.device, clean: bool = True, clip: float = 1e6):
    out = {}
    for k, v in item.items():
        if isinstance(v, torch.Tensor):
            t = v
            if clean and t.dtype.is_floating_point:
                t = torch.nan_to_num(t, nan=0.0, posinf=clip, neginf=-1*clip)
                t = torch.clamp(t, -clip, clip)
            out[k] = t.to(device, non_blocking=True)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            lst = []
            for t in v:
                if clean and t.dtype.is_floating_point:
                    t = torch.nan_to_num(t, nan=0.0, posinf=clip, neginf=-1*clip)
                    t = torch.clamp(t, -clip, clip)
                lst.append(t.to(device, non_blocking=True))
            out[k] = lst
        else:
            out[k] = v
    return out

@torch.no_grad()
def embed_ids(net: SceneEncoder,
              ds: NeutralSceneDataset,
              ids: list[int],
              device: torch.device,
              batch_size: int = 32,
              cache_dir: str | None = None,
              cache_tag: str = "") -> dict[int, np.ndarray]:
    net.eval()
    if cache_dir:
        cache_dir = Path(cache_dir) / cache_tag
        cache_dir.mkdir(parents=True, exist_ok=True)

    out = {}
    for st in tqdm(range(0, len(ids), batch_size), desc=f"Encoding[{cache_tag}]", ncols=120, ascii=True):
        ed = min(st + batch_size, len(ids))
        chunk = ids[st:ed]

        need_ids, cached = [], {}
        if cache_dir:
            for idx in chunk:
                cpath = cache_dir / f"{int(idx)}.npy"
                if cpath.exists():
                    try:
                        v = np.load(cpath)
                        if v.ndim == 1:
                            cached[idx] = v.astype(np.float32, copy=False)
                            continue
                    except Exception:
                        pass
                need_ids.append(idx)
        else:
            need_ids = chunk

        batch_list, order = [], []
        for idx in need_ids:
            try:
                item = ds[idx]
            except Exception:
                continue
            batch_list.append(to_device_item(item, device, clean=True))
            order.append(idx)

        if batch_list:
            Z = net(batch_list)  # (B,D)
            if Z.dim() == 1:
                Z = Z.unsqueeze(0)
            Z = Z.detach().cpu().float().numpy()
            for i, idx in enumerate(order):
                vec = Z[i]
                out[idx] = vec
                if cache_dir:
                    np.save(cache_dir / f"{int(idx)}.npy", vec.astype(np.float32))

        out.update(cached)
    return out

def build_candidate_ids(ds_len: int, cand_ids_csv: str | None) -> list[int]:
    if cand_ids_csv:
        df = _read_csv_robust(cand_ids_csv)
        cols = {c.strip().lower(): c for c in df.columns}
        col_idx = next((cols[k] for k in cols if k in ("scene_idx","idx","id")), None)
        if not col_idx:
            raise ValueError(f"{cand_ids_csv} 未找到 id 列（scene_idx/idx/id）")
        ids = pd.to_numeric(df[col_idx], errors="coerce").dropna().astype(int).tolist()
        ids = sorted(int(x) for x in ids if 0 <= int(x) < ds_len)
        return ids
    else:
        return list(range(ds_len))

# ---------- 主入口 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="训练 YAML（含结构与 alpha）")

    # 可选覆盖（若不提供则从 cfg['data'] 读取）
    ap.add_argument("--qry_pkl", default=None)
    ap.add_argument("--qry_index_csv", default=None)
    ap.add_argument("--cand_pkl", default=None)
    ap.add_argument("--cand_index_csv", default=None)

    ap.add_argument("--model", required=True, help="SceneEncoder 的 state_dict .pth")
    ap.add_argument("--alpha", type=float, default=None, help="覆盖 cfg 中的 alpha")

    ap.add_argument("--query_id", type=int, default=None)
    ap.add_argument("--query_name", type=str, default=None)
    ap.add_argument("--candidate_ids_csv", type=str, default=None)

    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--exclude_self", action="store_true")
    ap.add_argument("--cache_dir", type=str, default=None)
    ap.add_argument("--top_k", type=int, default=10)

    args = ap.parse_args()

    # 读取 YAML
    cfg = yaml.safe_load(open(args.cfg, "r", encoding="utf-8"))
    # 数据路径默认从 cfg 读取，允许命令行覆盖
    qry_pkl  = args.qry_pkl  or cfg["data"]["pkl"]
    qry_idx  = args.qry_index_csv or cfg["data"]["index"]
    same_dataset = (args.cand_pkl is None and args.cand_index_csv is None)
    if same_dataset:
        cand_pkl, cand_idx = qry_pkl, qry_idx
    else:
        if not args.cand_pkl or not args.cand_index_csv:
            raise ValueError("当指定候选数据集时，--cand_pkl 与 --cand_index_csv 需同时提供。")
        cand_pkl, cand_idx = args.cand_pkl, args.cand_index_csv

    # 模型结构：从 cfg['model'] 读取（若不存在的键用 SceneEncoder 默认）
    mcfg = cfg.get("model", {})
    node_dim = int(mcfg.get("node_dim", 8))
    hidden   = int(mcfg.get("hidden", 128))
    emb_dim  = int(mcfg.get("emb_dim", 128))
    n_heads  = int(mcfg.get("n_heads", 4))
    n_layers_spatial  = int(mcfg.get("n_layers_spatial", 3))
    n_layers_temporal = int(mcfg.get("n_layers_temporal", 2))
    num_edge_types    = int(mcfg.get("num_edge_types", 11))
    use_geo_bias      = bool(mcfg.get("use_geo_bias", True))
    geo_dim           = int(mcfg.get("geo_dim", 3))
    renorm_embeddings = bool(mcfg.get("renorm_embeddings", False))
    nan_guard         = bool(mcfg.get("nan_guard", False))

    # alpha：优先命令行；否则 cfg['train']['alpha']
    if args.alpha is not None:
        alpha = float(args.alpha)
    else:
        alpha = float(cfg.get("train", {}).get("alpha", 0.5))

    # 设备
    device = torch.device("cuda" if (args.device=="auto" and torch.cuda.is_available()) else
                          "cuda" if args.device=="cuda" else "cpu")
    print(f"[Device] {device}")

    # 映射
    qry_name2idx, qry_idx2name = build_name_maps(qry_idx)
    cand_name2idx, cand_idx2name = (qry_name2idx, qry_idx2name) if same_dataset else build_name_maps(cand_idx)

    # 解析 query
    qid = resolve_query_id(args.query_id, args.query_name, qry_name2idx)
    print(f"[Query] id={qid} | name={qry_idx2name.get(qid, 'N/A')}")

    # 数据集
    ds_q = NeutralSceneDataset(pkl_path=qry_pkl,  index_csv=qry_idx,  use_geo_bias=use_geo_bias)
    ds_c = ds_q if same_dataset else NeutralSceneDataset(pkl_path=cand_pkl, index_csv=cand_idx, use_geo_bias=use_geo_bias)
    print(f"[Dataset] query_ds={len(ds_q)} | cand_ds={len(ds_c)} (same={same_dataset})")

    # 候选 id
    cand_ids = build_candidate_ids(len(ds_c), args.candidate_ids_csv)
    if same_dataset and args.exclude_self and qid in cand_ids:
        cand_ids = [x for x in cand_ids if x != qid]
    print(f"[Candidates] {len(cand_ids)} items")

    # 模型
    net = SceneEncoder(
        node_dim=node_dim,
        hidden=hidden,
        emb_dim=emb_dim,
        n_heads=n_heads,
        n_layers_spatial=n_layers_spatial,
        n_layers_temporal=n_layers_temporal,
        num_edge_types=num_edge_types,
        use_geo_bias=use_geo_bias,
        geo_dim=geo_dim,
        renorm_embeddings=renorm_embeddings,
        nan_guard=nan_guard,
    ).to(device)
    state = torch.load(args.model, map_location=device)
    net.load_state_dict(state, strict=True)
    net.eval()
    print(f"[Model] loaded. (node_dim={node_dim}, hidden={hidden}, emb_dim={emb_dim}, n_heads={n_heads})")
    print(f"[Alpha] {alpha}")

    # 嵌入（缓存分目录）
    if args.cache_dir:
        (Path(args.cache_dir) / "query").mkdir(parents=True, exist_ok=True)
        (Path(args.cache_dir) / "cand").mkdir(parents=True, exist_ok=True)

    # 生成 query embedding
    from math import isfinite
    q_map = embed_ids(net, ds_q, [qid], device=device, batch_size=1, cache_dir=args.cache_dir, cache_tag="query")
    E_q = q_map[qid].astype(np.float32)

    # 生成候选库 embedding
    cand_map = embed_ids(net, ds_c, cand_ids, device=device, batch_size=args.batch_size, cache_dir=args.cache_dir, cache_tag="cand")
    E_mat = np.stack([cand_map[i] for i in cand_ids], axis=0).astype(np.float32)  # (M,D)

    # L2 距离（含开根号）与相似度
    dists = np.linalg.norm(E_mat - E_q[None, :], ord=2, axis=1)
    sims  = np.exp(-alpha * dists)

    order = np.argsort(-sims)  # 相似度降序
    ids_sorted   = [cand_ids[i] for i in order]
    d_sorted     = [float(dists[i]) for i in order]
    s_sorted     = [float(sims[i])  for i in order]
    names_sorted = [cand_idx2name.get(cid, str(cid)) for cid in ids_sorted]

    # 输出
    out_df = pd.DataFrame({
        "query_id":   [qid]*len(ids_sorted),
        "query_name": [qry_idx2name.get(qid, "N/A")]*len(ids_sorted),
        "cand_id":    ids_sorted,
        "cand_name":  names_sorted,
        "euclid_dist": d_sorted,
        "similarity_exp": s_sorted,
        "alpha": [float(alpha)]*len(ids_sorted),
        "same_dataset": [bool(same_dataset)]*len(ids_sorted),
    })
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[Done] Saved to {args.out_csv}")

    # Top-K
    K = max(1, min(args.top_k, len(ids_sorted)))
    print(f"\n[Top-{K}]")
    for i in range(K):
        print(f"{i+1:02d}. id={ids_sorted[i]:>6} | name={names_sorted[i]} | d={d_sorted[i]:.4f} | sim={s_sorted[i]:.6f}")

if __name__ == "__main__":
    main()
