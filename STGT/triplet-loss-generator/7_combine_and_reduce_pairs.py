# -*- coding: utf-8 -*-
"""
合并精简：每个 Anchor 最多 2 个正样本 + 1 个负样本（无阈值、无兜底）
- 正样本 = pair_scores.csv 全部行
- 负样本 = pair_scores_neg.csv 全部行（视为已人工确认）
- 仅在【规范化后】正表与负表的 anchor 交集里产出
- 仅输出列：anchor, candidate, score, final_score(0-1)_to_fill
- 支持名称规范化：--strip-prefix 多前缀（| 分隔） + --name-mode {stem,basename,keep}

用法示例：
  python combine_and_reduce_pairs.py --pos "./pair_scores_out/pair_scores.csv" --neg "./pair_scores_neg.csv" --out "./pair_scores_min.csv" --name-mode stem --strip-prefix "***"
"""

import argparse, csv, random
from pathlib import Path
import pandas as pd

SEED = 42
MAX_POS_PER_ANCHOR  = 2
KEEP_NEG_PER_ANCHOR = 1

# ---------- 读表/列名工具 ----------
def read_csv_flex(path, header="infer"):
    """尽量鲁棒地读取 CSV（自动识别分隔符；UTF-8-SIG；跳过坏行）。"""
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig",
                           on_bad_lines="skip", header=header)
    except Exception:
        pass
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            dialect = csv.Sniffer().sniff(sample, delimiters=[",","\t",";","|"])
            sep = dialect.delimiter
        return pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig",
                           on_bad_lines="skip", header=header)
    except Exception:
        pass
    for sep in [",","\t",";","|"]:
        try:
            return pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig",
                               on_bad_lines="skip", header=header)
        except Exception:
            continue
    raise RuntimeError(f"无法解析 CSV: {path}")

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """标准化列名：去BOM/零宽/NBSP、小写、去空格。"""
    df = df.copy()
    df.columns = (pd.Index(df.columns)
                  .astype(str)
                  .str.replace("\ufeff", "", regex=False)   # BOM
                  .str.replace("\u200b","", regex=False)    # 零宽
                  .str.replace("\xa0"," ", regex=False)     # NBSP
                  .str.strip().str.lower())
    return df

def find_col(df: pd.DataFrame, candidates):
    """在列名中查找候选（支持包含式匹配）。"""
    s = set(df.columns)
    for c in candidates:
        if c in s: return c
    for c in candidates:
        for col in s:
            if c in col: return col
    raise KeyError(f"找不到列：{candidates} in {list(df.columns)}")

# ---------- 名称规范化 ----------
def _strip_invisible(s: str) -> str:
    return (s.replace("\ufeff","")
             .replace("\u200b","")
             .replace("\xa0"," ")
             .strip().strip('"').strip("'"))

def _strip_prefixes(s: str, prefixes):
    # 兼容 Windows 路径：统一成 /，不区分大小写，前缀末尾 / 可有可无
    ss = s
    for p in prefixes:
        if not p: continue
        p_norm = p.replace("\\","/").rstrip("/")
        if not p_norm: continue
        ss_norm = ss.replace("\\","/")
        if ss_norm.lower().startswith(p_norm.lower() + "/"):
            ss = ss_norm[len(p_norm)+1:]; break
        if ss_norm.lower().startswith(p_norm.lower()):
            ss = ss_norm[len(p_norm):]; break
    return ss

def normalize_name(x: str, mode: str = "stem", strip_prefixes=()):
    s = _strip_invisible(str(x))
    s = _strip_prefixes(s, strip_prefixes)
    s = s.replace("\\","/").replace("//","/").strip("/")
    p = Path(s)
    if mode == "keep":       return s
    elif mode == "basename": return p.name
    else:                    return p.stem  # 默认 stem

def normalize_df_names(df: pd.DataFrame, A: str, C: str, name_mode: str, strip_prefix: str,
                       out_anchor_col: str, out_cand_col: str):
    """对 anchor/candidate 做规范化，去 self-pair 与重复。"""
    prefixes = tuple([_strip_invisible(p) for p in (strip_prefix or "").split("|") if _strip_invisible(p)])
    g = df.copy()
    g[out_anchor_col] = g[A].map(lambda x: normalize_name(x, mode=name_mode, strip_prefixes=prefixes))
    g[out_cand_col]   = g[C].map(lambda x: normalize_name(x, mode=name_mode, strip_prefixes=prefixes))
    g = g[g[out_anchor_col] != g[out_cand_col]].copy()
    g = g.drop_duplicates([out_anchor_col, out_cand_col]).reset_index(drop=True)
    return g

# ---------- 人工列解析（仅用于输出，非判定条件） ----------
def attach_final_use(df: pd.DataFrame):
    """尝试解析人工列，映射常见文本到 0/1；仅用于输出列 final_score(0-1)_to_fill。"""
    cand_cols = ["final_score(0-1)_to_fill","final","human","label","answer","score_final","manual","gt"]
    F = None
    for cand in cand_cols:
        if cand in df.columns or any(cand in col for col in df.columns):
            F = cand if cand in df.columns else [col for col in df.columns if cand in col][0]
            break
    if F is None:
        df["final_use"] = float("nan")
    else:
        f_num = pd.to_numeric(df[F], errors="coerce")

        def _map_human_label(x):
            if isinstance(x, str):
                xs = x.strip().lower()
                m1 = {"1","true","yes","y","是","对","正","pos","positive","同意","ok","✓","✔"}
                m0 = {"0","false","no","n","否","错","负","neg","negative","不同意","✗","✘"}
                if xs in m1: return 1.0
                if xs in m0: return 0.0
            if isinstance(x, (int, float)):
                if x in (0, 0.0): return 0.0
                if x in (1, 1.0): return 1.0
            return float("nan")

        need_map = f_num.isna()
        if need_map.any():
            f_map = df.loc[need_map, F].map(_map_human_label)
            f_num.loc[need_map] = f_map
        df["final_use"] = f_num
    return df

# ---------- 主流程 ----------
def main(args):
    random.seed(SEED)

    # 读取并标准化列名
    pos_df = norm_cols(read_csv_flex(args.pos))
    neg_df = norm_cols(read_csv_flex(args.neg))

    # 定位列
    A = find_col(pos_df, ["anchor"]);            C = find_col(pos_df, ["candidate","cand","sample"])
    S = find_col(pos_df, ["score","sim","similarity","machine_score"])
    A2 = find_col(neg_df, ["anchor"]);           C2 = find_col(neg_df, ["candidate","cand","sample"])
    S2 = find_col(neg_df, ["score","sim","similarity","machine_score"])

    # 类型统一
    for df, a,c,s in [(pos_df,A,C,S),(neg_df,A2,C2,S2)]:
        df[a] = df[a].astype(str); df[c] = df[c].astype(str)
        df[s] = pd.to_numeric(df[s], errors="coerce").fillna(0.0).clip(0,1)

    # 附带 final_use（仅用于输出；不决定正/负）
    pos_df = attach_final_use(pos_df)
    neg_df = attach_final_use(neg_df)

    # 规范化（新增列：anchor_n / candidate_n）
    pos_n = normalize_df_names(pos_df, A, C, args.name_mode, args.strip_prefix, "anchor_n", "candidate_n")
    neg_n = normalize_df_names(neg_df, A2, C2, args.name_mode, args.strip_prefix, "anchor_n", "candidate_n")

    # 仅保留交集 anchor，避免“anchors(total)=0”
    pos_anchors = set(pos_n["anchor_n"])
    neg_anchors = set(neg_n["anchor_n"])
    common_anchors = sorted(pos_anchors & neg_anchors)

    print(f"[INFO] pos_rows={len(pos_df)} neg_rows={len(neg_df)}")
    print(f"[INFO] anchors(pos_norm)={len(pos_anchors)} anchors(neg_norm)={len(neg_anchors)}")
    print(f"[INFO] anchors_intersection(norm)={len(common_anchors)}")
    if not common_anchors:
        print("[WARN] 规范化后无 anchor 交集，请检查 --strip-prefix 与 --name-mode 是否正确。")
        out = pd.DataFrame(columns=["anchor","candidate","score","final_score(0-1)_to_fill"])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"[OK] 写出：{args.out}")
        print("anchors(total): 0  pairs(out): 0")
        return

    # 为 join 回原表拿 final_use 做准备
    pos_cols_keep = [A, C, S, "final_use"]
    neg_cols_keep = [A2, C2, S2, "final_use"]
    pos_join = pos_df[pos_cols_keep].copy()
    neg_join = neg_df[neg_cols_keep].copy()

    out_rows = []
    for a in common_anchors:
        # 正样本：最多 2（按 score 降序）
        sub_pos = pos_n[pos_n["anchor_n"] == a].copy()
        if sub_pos.empty:
            continue
        sub_pos = sub_pos.sort_values("score", ascending=False)
        chosen_pos = sub_pos.head(MAX_POS_PER_ANCHOR)[["anchor_n","candidate_n","score"]].copy()
        # 回填 final_use（如有）
        chosen_pos = chosen_pos.merge(
            pos_join,
            left_on=["anchor_n","candidate_n","score"],
            right_on=[A, C, S],
            how="left"
        )
        chosen_pos = chosen_pos[["anchor_n","candidate_n","score","final_use"]]

        # 负样本：最多 1（按 score 升序；全量负表，已人工确认）
        sub_neg = neg_n[neg_n["anchor_n"] == a].copy()
        if sub_neg.empty:
            continue
        sub_neg = sub_neg.sort_values("score", ascending=True)
        chosen_neg = sub_neg.head(KEEP_NEG_PER_ANCHOR)[["anchor_n","candidate_n","score"]].copy()
        chosen_neg = chosen_neg.merge(
            neg_join,
            left_on=["anchor_n","candidate_n","score"],
            right_on=[A2, C2, S2],
            how="left"
        )
        chosen_neg = chosen_neg[["anchor_n","candidate_n","score","final_use"]]

        # 安全拼接（避免 FutureWarning）
        parts = []
        if not chosen_pos.empty: parts.append(chosen_pos)
        if not chosen_neg.empty: parts.append(chosen_neg)
        merged = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["anchor_n","candidate_n","score","final_use"]
        )

        if not merged.empty:
            merged = merged.drop_duplicates(["anchor_n","candidate_n"]).reset_index(drop=True)
            out_rows.append(merged)

    # 汇总
    if out_rows:
        out = pd.concat(out_rows, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["anchor_n","candidate_n","score","final_use"])

    # 输出列名与排序
    out = out.rename(columns={"anchor_n":"anchor","candidate_n":"candidate"})
    out["final_score(0-1)_to_fill"] = out["final_use"]
    out = out[["anchor","candidate","score","final_score(0-1)_to_fill"]]
    out = out.sort_values(["anchor","score"], ascending=[True, False]).reset_index(drop=True)

    # 写文件
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"[OK] 写出：{args.out}")
    print(f"anchors(total): {out['anchor'].nunique()}  pairs(out): {out.shape[0]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos",   required=True, help="正样本总表 CSV（如 pair_scores.csv）")
    ap.add_argument("--neg",   required=True, help="负样本表 CSV（人工确认，如 pair_scores_neg.csv）")
    ap.add_argument("--out",   default="pair_scores_min.csv", help="输出精简 CSV")
    # 名称规范化选项
    ap.add_argument("--name-mode", choices=["stem","basename","keep"], default="stem",
                    help="anchor/candidate 统一形式：stem=不带扩展名(默认)、basename=仅文件名、keep=原样")
    ap.add_argument("--strip-prefix", default="",
                    help="剥掉的路径前缀（可多个，用 | 分隔），再按 name-mode 处理。示例：E:\\OnSite\\replay_visualizer\\outputs\\WJ20250116")
    args = ap.parse_args()
    main(args)
