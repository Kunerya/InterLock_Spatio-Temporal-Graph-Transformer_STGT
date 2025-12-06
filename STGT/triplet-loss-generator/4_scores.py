# -*- coding: utf-8 -*-
"""
从 pos_candidate_{tasks,answers}.csv 生成：
1) pair_scores.csv（anchor,candidate,score,wins,losses,total）
2) survey/anchor_*.csv（每个 anchor 的 Top-K 清单，含 final_score(0-1)_to_fill 空列）
用法：
python tools/build_surveys_from_poscandidate.py \
  --pos_tasks pos_candidate_tasks.csv \
  --pos_answers pos_candidate_answers.csv \
  --out_dir pair_scores_out \
  --alpha 1 --beta 1 --topk 10
"""
import argparse, pandas as pd
from pathlib import Path
from collections import defaultdict

def read_csv_robust(p):
    p = Path(p)
    try:
        return pd.read_csv(p, sep=None, engine="python")
    except Exception:
        for sep in [",", "|", "\t", ";"]:
            try:
                return pd.read_csv(p, sep=sep)
            except Exception:
                pass
        raise

def fname(x): return Path(str(x)).name

def build_scores(pos_tasks, pos_answers, alpha=1.0, beta=1.0):
    df_t = read_csv_robust(pos_tasks)
    need = {"task_id","gif_anchor","gif_cand1","gif_cand2","gif_cand3"}
    if not need.issubset(df_t.columns):
        raise ValueError(f"[PosTasks] 缺列：{need - set(df_t.columns)}")

    # answers 可能是“|”分隔且无表头
    try:
        df_a = pd.read_csv(pos_answers, sep="|", header=None, names=["task_id","label"])
        assert {"task_id","label"}.issubset(df_a.columns)
    except Exception:
        df_a = read_csv_robust(pos_answers)
        if not {"task_id","label"}.issubset(df_a.columns):
            raise ValueError("[PosAnswers] 需要 task_id,label 两列")

    df = df_t.merge(df_a, on="task_id", how="inner")
    if df.empty:
        raise ValueError("任务与答案无法匹配")

    wins, losses = defaultdict(float), defaultdict(float)
    for _, r in df.iterrows():
        a = fname(r["gif_anchor"])
        cands = [fname(r["gif_cand1"]), fname(r["gif_cand2"]), fname(r["gif_cand3"])]
        lab = str(r["label"]).strip()
        if lab in {"1","2","3"}:
            k = int(lab) - 1
            winner = cands[k]
            # 胜者赢两场；另外两位各输一场
            wins[(a, winner)] += 2.0
            for j, c in enumerate(cands):
                if j != k: losses[(a, c)] += 1.0
        else:
            # 都不相似：三位各输一场
            for c in cands: losses[(a, c)] += 1.0

    rows = []
    keys = set(list(wins.keys()) + list(losses.keys()))
    for (a,b) in keys:
        w = wins[(a,b)]; l = losses[(a,b)]
        score = (w + alpha) / (w + l + alpha + beta)
        rows.append(dict(anchor=a, candidate=b, score=score, wins=w, losses=l, total=w+l))
    df_scores = pd.DataFrame(rows).sort_values(["anchor","score"], ascending=[True,False]).reset_index(drop=True)
    return df_scores

def build_surveys(df_scores, out_dir, topk=10):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    survey_dir = out_dir / "survey"; survey_dir.mkdir(exist_ok=True)

    # 1) 总表
    df_scores.to_csv(out_dir/"pair_scores.csv", index=False, encoding="utf-8-sig")

    # 2) 每个 anchor 的 Top-K 问卷
    for a, g in df_scores.groupby("anchor"):
        g_top = g.sort_values("score", ascending=False).head(topk).copy()
        g_top["final_score(0-1)_to_fill"] = ""   # 预留人工填写
        (survey_dir/f"anchor_{a}.csv").write_text(
            g_top.to_csv(index=False, encoding="utf-8-sig"), encoding="utf-8"
        )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos_tasks", required=True)
    ap.add_argument("--pos_answers", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    scores = build_scores(args.pos_tasks, args.pos_answers, args.alpha, args.beta)
    build_surveys(scores, args.out_dir, args.topk)
    print(f"[OK] 写出 {Path(args.out_dir)/'pair_scores.csv'} 与 {Path(args.out_dir)/'survey/anchor_*.csv'}")
