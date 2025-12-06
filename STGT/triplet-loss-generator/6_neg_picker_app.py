# -*- coding: utf-8 -*-
# """
# 负样本甄别工具（对每个 Anchor 的每个候选做 三态标注：未标/否/是 + 显式完成标记）
# 运行：
#   streamlit run neg_picker_app.py
# """

import os, time, math, shutil
from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="负样本甄别（三态 + 完成标记）", layout="wide")

# ---------- 侧边栏 ----------
DEFAULT_NEG_CSV    = r"pair_scores_neg.csv"
DEFAULT_MEDIA_ROOT = r"scenes"

NEG_CSV    = st.sidebar.text_input("负样本候选表（pair_scores_neg.csv）", DEFAULT_NEG_CSV)
MEDIA_ROOT = st.sidebar.text_input("GIF 根目录", DEFAULT_MEDIA_ROOT)
TOPN_EACH  = st.sidebar.number_input("每个 Anchor 最多展示多少个候选", min_value=1, value=24, step=1)
N_COLS     = st.sidebar.number_input("每行列数", min_value=2, max_value=6, value=3, step=1)
BACKUP     = st.sidebar.checkbox("保存前备份(.bak)", value=True)
AUTO_NEXT  = st.sidebar.checkbox("保存后自动跳到下一个 Anchor", value=True)
ONLY_TODO  = st.sidebar.checkbox("只看未完成 Anchor（按完成标记）", value=False)

# ---------- 小工具 ----------
def _rerun():
    if hasattr(st, "rerun"): st.rerun()
    else: st.experimental_rerun()

def read_csv_robust(path: Path) -> pd.DataFrame:
    # 自动探测分隔符
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        for sep in [",", "\t", ";", "|"]:
            try:
                return pd.read_csv(path, sep=sep, engine="python", encoding="utf-8-sig")
            except Exception:
                continue
    raise RuntimeError(f"无法解析 CSV: {path}")

def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        pd.Index(df.columns).astype(str)
          .str.replace("\ufeff","", regex=False)
          .str.replace("\u200b","", regex=False)
          .str.replace("\xa0"," ", regex=False)
          .str.strip().str.lower()
    )
    return df

def find_col_optional(df: pd.DataFrame, candidates, required=True):
    s = set(df.columns)
    for c in candidates:
        if c in s: return c
    for c in candidates:
        for col in s:
            if c in col: return col
    if required:
        raise KeyError(f"列缺失: {candidates} in {list(df.columns)}")
    return None

# GIF 路径解析
MEDIA_ROOT_PATH = Path(MEDIA_ROOT)
def try_paths(name: str) -> str:
    c = Path(name)
    if c.is_file(): return str(c)
    if c.suffix.lower() != ".gif":
        cg = c.with_suffix(".gif")
        if cg.is_file(): return str(cg)
    q = MEDIA_ROOT_PATH / name
    if q.is_file(): return str(q)
    if q.suffix.lower() != ".gif":
        qg = q.with_suffix(".gif")
        if qg.is_file(): return str(qg)
    qq = MEDIA_ROOT_PATH / Path(name).name
    if qq.is_file(): return str(qq)
    if qq.suffix.lower() != ".gif":
        qqg = qq.with_suffix(".gif")
        if qqg.is_file(): return str(qqg)
    return str(c)

def show_gif(path: str, caption=None):
    try:
        st.image(path, caption=caption, use_container_width=True)
    except Exception:
        st.info(f"无法显示：{path}")

def atomic_save(df_new: pd.DataFrame, path: Path):
    if BACKUP:
        ts = time.strftime("%Y%m%d-%H%M%S")
        bak = path.with_suffix(path.suffix + f".{ts}.bak")
        try: shutil.copy2(path, bak)
        except Exception as e: st.warning(f"备份失败：{e}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    df_new.to_csv(tmp, index=False, encoding="utf-8-sig", index_label=False)
    os.replace(tmp, path)

# ---------- 读取数据 ----------
p = Path(NEG_CSV)
if not p.exists():
    st.error(f"找不到负样本候选表：{p}")
    st.stop()

df_all = norm_cols(read_csv_robust(p))

A = find_col_optional(df_all, ["anchor"], required=True)
C = find_col_optional(df_all, ["candidate","cand","sample"], required=True)
S = find_col_optional(df_all, ["score","sim","similarity","machine_score"], required=False)  # score 可选

# 三态标注列（不能把空值当成0）
NEG_FLAG = None
for cand in ["neg_selected(0/1)", "neg_selected", "is_neg", "negative", "neg", "label_neg", "is_negative"]:
    if cand in df_all.columns:
        NEG_FLAG = cand; break
if NEG_FLAG is None:
    NEG_FLAG = "neg_selected(0/1)"
    df_all[NEG_FLAG] = np.nan  # 初始为空（未标）

# 完成标记列（独立）
DONE_FLAG = None
for cand in ["anchor_done(0/1)", "done", "reviewed", "is_done"]:
    if cand in df_all.columns:
        DONE_FLAG = cand; break
if DONE_FLAG is None:
    DONE_FLAG = "anchor_done(0/1)"
    df_all[DONE_FLAG] = 0

# 规整类型
df_all[A] = df_all[A].astype(str)
df_all[C] = df_all[C].astype(str)
if S is not None:
    df_all[S] = pd.to_numeric(df_all[S], errors="coerce").fillna(0.0).clip(0,1)
# 负样本三态列：保留 NaN；若原表有字符串 '','nan' 等，转成 NaN 或 0/1
def _to_tri(x):
    if pd.isna(x): return np.nan
    try:
        v = float(x)
        if math.isfinite(v):
            if v >= 0.5: return 1.0
            if v <= 0.0: return 0.0
            # 其它值按 0/1 以外也可以保留为 NaN，这里做保守映射：
            return 1.0 if v > 0.5 else 0.0
    except Exception:
        pass
    xs = str(x).strip().lower()
    if xs in {"1","yes","y","true"}: return 1.0
    if xs in {"0","no","n","false"}: return 0.0
    return np.nan
df_all[NEG_FLAG] = df_all[NEG_FLAG].map(_to_tri)
df_all[DONE_FLAG] = pd.to_numeric(df_all[DONE_FLAG], errors="coerce").fillna(0).astype(int)

# ---------- 进度（按完成标记） ----------
anchors_all = sorted(df_all[A].unique().tolist())
done_map = df_all.groupby(A)[DONE_FLAG].max().to_dict()
anchors_done = [a for a in anchors_all if done_map.get(a,0) == 1]
anchors_todo = [a for a in anchors_all if done_map.get(a,0) == 0]

total = len(anchors_all)
done  = len(anchors_done)

st.title("负样本甄别（三态：未标/否/是 + 完成标记）")
st.write(f"全局进度（按完成标记）：**{done}/{total}** 个 Anchor 已完成")
st.progress(0 if total==0 else done/total)

# 当前视图
anchors_view = anchors_todo if ONLY_TODO else anchors_all
if not anchors_view:
    st.success("✅ 全部 Anchor 已完成（基于完成标记）！")
    st.stop()

if "anchor_idx" not in st.session_state: st.session_state.anchor_idx = 0
st.session_state.anchor_idx = min(st.session_state.anchor_idx, len(anchors_view)-1)

def _fmt_anchor(x: str) -> str:
    flag = "✔已完成" if done_map.get(x,0)==1 else "未完成"
    return f"{x}  {flag}"

a = st.selectbox(
    "选择 Anchor",
    anchors_view,
    index=st.session_state.anchor_idx,
    format_func=_fmt_anchor
)
st.session_state.anchor_idx = anchors_view.index(a)

# 当前 Anchor 的候选（展示 topN）
sub_all = df_all[df_all[A]==a].copy()
sub = sub_all.copy()
if S is not None: sub = sub.sort_values(S, ascending=True)
else:             sub = sub.sort_values(C, ascending=True)
sub = sub.head(int(TOPN_EACH)).reset_index(drop=True)

st.subheader(f"Anchor：{a}")
show_gif(try_paths(a))

# 标注进度（当前展示范围）
labeled_mask = ~sub[NEG_FLAG].isna()
labeled_cnt  = int(labeled_mask.sum())
st.write(f"本页（TopN={len(sub)}）标注进度：**{labeled_cnt}/{len(sub)}** 已标（未标=空）")

# 三态单选：-1=未标, 0=否, 1=是
def _val_to_idx(v):
    if pd.isna(v): return 0   # 未标
    return 2 if float(v) == 1.0 else 1  # 1->是，0->否
choices = {}
rows = math.ceil(max(len(sub),1) / int(N_COLS))
for r in range(rows):
    cols = st.columns(int(N_COLS))
    for c_i in range(int(N_COLS)):
        i = r*int(N_COLS) + c_i
        if i >= len(sub): break
        with cols[c_i]:
            cand = str(sub.loc[i, C])
            meta = f" · 机器分：{sub.loc[i, S]:.3f}" if S is not None else ""
            st.caption(f"`{cand}`{meta}")
            show_gif(try_paths(cand))
            cur = sub.loc[i, NEG_FLAG]
            idx = _val_to_idx(cur)
            choice = st.radio(
                f"是否为负样本 #{i+1}",
                options=[-1, 0, 1],
                format_func=lambda x: "未标" if x==-1 else ("否 (0)" if x==0 else "是 (1)"),
                index=[-1,0,1].index([-1,0,1][idx]),  # idx: 0->-1, 1->0, 2->1
                key=f"neg::{a}::{cand}"
            )
            choices[i] = choice  # -1/0/1

# 完成标记状态与按钮
is_done = bool(done_map.get(a,0) == 1)
st.write(f"当前 Anchor 完成标记：**{'已完成' if is_done else '未完成'}**")
col_done1, col_done2, col_done3 = st.columns(3)

def save_rows_and_stay(next_anchor: str|None = None):
    df_new = df_all.copy()
    # 写回本页三态
    for i, choice in choices.items():
        val = np.nan if choice == -1 else float(choice)
        df_new.loc[(df_new[A]==a) & (df_new[C]==sub.loc[i, C]), NEG_FLAG] = val
    # 保存
    try:
        atomic_save(df_new, p)
        st.success(f"已保存到：{p}")
        if next_anchor is None:
            _rerun(); return
        # 跳转
        view_after = ([x for x in anchors_all if df_new.groupby(A)[DONE_FLAG].max().get(x,0)==0]
                      if ONLY_TODO else anchors_all)
        if next_anchor in view_after:
            st.session_state.anchor_idx = view_after.index(next_anchor)
        else:
            st.session_state.anchor_idx = 0
        _rerun()
    except Exception as e:
        st.error(f"保存失败：{e}")

# 标记完成/取消完成（不依赖是否有 1）
def mark_done(done_val: int):
    df_new = df_all.copy()
    df_new.loc[df_new[A]==a, DONE_FLAG] = int(done_val)
    try:
        atomic_save(df_new, p)
        st.success("已更新完成标记。")
        _rerun()
    except Exception as e:
        st.error(f"更新完成标记失败：{e}")

with col_done1:
    if st.button("标记此 Anchor 已完成 ✅", type="primary"):
        mark_done(1)
with col_done2:
    if st.button("取消完成标记 ↩"):
        mark_done(0)

# 保存 + 跳转逻辑
def find_next_anchor_after_save(current_a: str) -> str | None:
    # 仅用于“保存并跳转”按钮
    # 若 ONLY_TODO：跳到下一个未完成；否则按 anchors_all 顺序。
    df_state = df_all.copy()
    done_map2 = df_state.groupby(A)[DONE_FLAG].max().to_dict()
    view = [x for x in anchors_all if done_map2.get(x,0)==0] if ONLY_TODO else anchors_all
    if not view: return None
    if current_a not in view: return view[0]
    i = view.index(current_a)
    if len(view) == 1:
        return None if ONLY_TODO and done_map2.get(current_a,0)==1 else current_a
    return view[(i+1) % len(view)]

b1, b2, b3 = st.columns(3)
with b1:
    if st.button("保存当前页标注"):
        save_rows_and_stay(None)
with b2:
    if st.button("保存并跳到下一个 Anchor"):
        nxt = find_next_anchor_after_save(a) if AUTO_NEXT else None
        save_rows_and_stay(nxt)
with b3:
    if st.button("跳到下一个未完成 Anchor"):
        todo_now = [x for x in anchors_all if done_map.get(x,0)==0]
        if not todo_now:
            st.success("✅ 全部 Anchor 已完成（基于完成标记）！")
        else:
            cur = a if a in todo_now else todo_now[0]
            i = todo_now.index(cur)
            nxt = todo_now[(i+1) % len(todo_now)]
            view_now = todo_now if ONLY_TODO else anchors_all
            st.session_state.anchor_idx = view_now.index(nxt) if nxt in view_now else 0
            _rerun()
