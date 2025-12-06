# -*- coding: utf-8 -*-
"""
从 answers（行格式: pair_id|0或1）+ pair_tasks.csv（含 pair_id,gif_left,gif_right）
生成负样本CSV：anchor,candidate,score=0.0,final_score(0-1)_to_fill=""

用法（0 表示不相似）：
  python build_neg_from_answers.py --tasks pair_tasks.csv --answers pair_answers.csv --out pair_scores_neg_1.csv --neg-values 0
"""

import argparse, csv
from pathlib import Path
import pandas as pd

def read_csv_auto(path, header="infer"):
    # 先让 pandas 自动探测；失败再换常见分隔符
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", on_bad_lines="skip", header=header)
    except Exception:
        pass
    for sep in [",", "\t", ";", "|"]:
        try:
            return pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig", on_bad_lines="skip", header=header)
        except Exception:
            continue
    raise RuntimeError(f"无法解析 CSV: {path}")

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (pd.Index(df.columns).astype(str)
        .str.replace("\ufeff","", regex=False)   # BOM
        .str.replace("\u200b","", regex=False)   # 零宽
        .str.replace("\xa0"," ", regex=False)    # NBSP
        .str.strip().str.lower()
    )
    return df

def pick(df: pd.DataFrame, wants):
    s = set(df.columns)
    for w in wants:
        if w in s: return w
    for w in wants:
        for c in s:
            if w in c: return c
    raise KeyError(f"在 tasks 里找不到列: {wants} in {list(df.columns)}")

def load_answers_pipe(path: str, sep: str = "|"):
    pairs = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):  # 空行/注释
                continue
            if sep not in line:
                raise ValueError(f"第{ln}行不含分隔符 '{sep}': {line}")
            pid, label = line.rsplit(sep, 1)  # 从右切，防止 pair_id 含分隔符
            pairs.append((pid.strip(), label.strip()))
    return pd.DataFrame(pairs, columns=["pair_id_raw", "ans_raw"])

def main(args):
    # 1) 读取 tasks（带表头）
    dt = read_csv_auto(args.tasks, header="infer")
    dt = norm_cols(dt)
    col_pid  = pick(dt, ["pair_id","pairid","id"])
    col_left = pick(dt, ["gif_left","left","src","from","anchor"])
    col_right= pick(dt, ["gif_right","right","dst","to","candidate"])
    dt["_pid_key"] = dt[col_pid].astype(str).str.strip()

    # 2) 读取 answers（每行 'pair_id|0或1'）
    da = load_answers_pipe(args.answers, sep=args.sep)
    da["_pid_key"] = da["pair_id_raw"].astype(str).str.strip()

    # 3) 只保留“不相似”的 pair_id
    neg_vals = {s.strip().lower() for s in args.neg_values.split(",") if s.strip() != ""}
    if not neg_vals:
        neg_vals = {"0"}  # 默认 0 为不相似
    labels_norm = da["ans_raw"].astype(str).str.strip().str.lower()
    da_neg = da[labels_norm.isin(neg_vals)]
    if da_neg.empty:
        raise SystemExit(f"未筛到任何负样本，请检查 --neg-values（当前使用: {sorted(neg_vals)}）")

    # 4) 合并得到 anchor/candidate
    df = da_neg.merge(dt[["_pid_key", col_left, col_right]], on="_pid_key", how="inner")
    if df.empty:
        raise SystemExit("根据 pair_id 未能在 tasks 中匹配到任何条目，请检查两边的 pair_id 是否一致。")

    # 5) 每个 anchor 限制条数（默认每个 1 条）
    out_rows = []
    per_anchor = int(args.per_anchor)
    for a, sub in df.groupby(col_left):
        for _, r in sub.head(per_anchor).iterrows():
            out_rows.append({
                "anchor":   str(r[col_left]),
                "candidate":str(r[col_right]),
                "score":    0.0,           # 不用机器分，占位
                "final_score(0-1)_to_fill": ""  # 第二轮人工再填
            })

    out = pd.DataFrame(out_rows).drop_duplicates(["anchor","candidate"]).reset_index(drop=True)
    out = out.sort_values(["anchor","candidate"]).reset_index(drop=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] 写出：{args.out} ；anchors={out['anchor'].nunique()}，pairs={out.shape[0]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks",   required=True, help="pair_tasks.csv（含 pair_id,gif_left,gif_right）")
    ap.add_argument("--answers", required=True, help="pair_answers.csv（每行 'pair_id|0或1'）")
    ap.add_argument("--out",     default="pair_scores_neg.csv", help="输出 CSV 路径")
    ap.add_argument("--sep",     default="|", help="answers 的分隔符（默认 '|'）")
    ap.add_argument("--neg-values", default="0", help="不相似标签取值集合（逗号分隔），默认 '0'")
    ap.add_argument("--per-anchor", type=int, default=1, help="每个 anchor 最多取几条负样本")
    args = ap.parse_args()
    main(args)
