# -*- coding: utf-8 -*-
"""
单表相似度打分（按 Anchor 聚合 + 层次分组 + GIF 预览，写回同一张 CSV）
运行：streamlit run survey_app.py
"""
import os, time, math, shutil, re
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="单表相似度打分（Anchor分组+层次对比）", layout="wide")

# ---------- rerun 兼容封装 ----------
def _rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# ---------- 默认路径（侧边栏可改） ----------
DEFAULT_BASE_CSV = r"pair_scores_min.csv"
DEFAULT_MEDIA_ROOT = r"scenes"

# ---------- 侧边栏 ----------
BASE_CSV   = st.sidebar.text_input("总表路径", DEFAULT_BASE_CSV)
MEDIA_ROOT = st.sidebar.text_input("GIF 根目录", DEFAULT_MEDIA_ROOT)

MODE = st.sidebar.selectbox("层次分组模式", ["同分=同层", "分位数分层", "自定义阈值分层"])
if MODE == "分位数分层":
    N_QUANT = st.sidebar.number_input("分位数层数（等频）", min_value=2, max_value=10, value=4, step=1)
elif MODE == "自定义阈值分层":
    THRESH_TXT = st.sidebar.text_input("阈值（高→低，逗号分隔）", "0.85,0.70,0.50")

COLS        = st.sidebar.number_input("每行列数", min_value=2, max_value=6, value=3, step=1)
TOPN_EACH   = st.sidebar.number_input("每层最多展示候选数", min_value=1, value=30, step=1)
AUTO_SAVE_N = st.sidebar.number_input("自动保存间隔（操作数）", min_value=1, value=12, step=1)
SHUFFLE     = st.sidebar.checkbox("随机 Anchor 顺序", value=False)
BACKUP      = st.sidebar.checkbox("保存前备份(.bak)", value=True)
AUTO_NEXT_ANCHOR = st.sidebar.checkbox("标完整个 Anchor 后自动跳到下一个", value=True)

# ---------- 读取总表 ----------
def read_csv_robust(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")

p = Path(BASE_CSV)
if not p.exists():
    st.error(f"找不到总表：{p}")
    st.stop()

df_all = read_csv_robust(p)
need = {"anchor","candidate","score"}
df_all.columns = (
    pd.Index(df_all.columns)
      .astype(str).str.replace("\ufeff","", regex=False)
      .str.strip().str.lower()
)
if not need.issubset(df_all.columns):
    st.error(f"总表缺少列：{need - set(df_all.columns)}"); st.stop()

if "final_score(0-1)_to_fill" not in df_all.columns:
    df_all["final_score(0-1)_to_fill"] = ""

# 统一为数值列（非法即 NaN）
df_all["final_score(0-1)_to_fill"] = pd.to_numeric(df_all["final_score(0-1)_to_fill"], errors="coerce")

df_all["score"] = pd.to_numeric(df_all["score"], errors="coerce").fillna(0.0).clip(0,1)
df_all["anchor"] = df_all["anchor"].astype(str)
df_all["candidate"] = df_all["candidate"].astype(str)

def is_finite(x) -> bool:
    try:
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False

# ---------- 全局完成提示（可选增强） ----------
all_done = df_all["final_score(0-1)_to_fill"].apply(is_finite).all()
if all_done:
    st.success("🎉 全部 Anchor 已完成标注！可以关闭页面或导出结果。")

# ---------- Anchor 列表 ----------
anchors = sorted(df_all["anchor"].unique().tolist())
if SHUFFLE:
    import random; random.Random(42).shuffle(anchors)

# ---------- 会话状态 ----------
if "anchor_idx" not in st.session_state: st.session_state.anchor_idx = 0
if "level_idx"  not in st.session_state: st.session_state.level_idx  = 0
if "ops"        not in st.session_state: st.session_state.ops        = 0

st.title("单表相似度打分（Anchor分组 + 层次对比 + GIF）")

# ---------- Anchor 选择 ----------
def on_select_anchor():
    st.session_state.level_idx = 0

a = st.selectbox(
    "选择 Anchor",
    anchors,
    index=min(max(st.session_state.anchor_idx, 0), len(anchors)-1),
    on_change=on_select_anchor
)
st.session_state.anchor_idx = anchors.index(a)
df = df_all[df_all["anchor"] == a].copy().reset_index(drop=True)

# ---------- 工具函数 ----------
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
    try: st.image(path, caption=caption, use_container_width=True)
    except Exception: st.info(f"无法显示：{path}")


def parse_init_or_fallback(txt, machine_score: float) -> float:
    """人工分可用→用人工分；否则回退机器分；最后裁剪到[0,1]"""
    v = None
    try:
        v = float(txt)
    except Exception:
        v = None
    if v is None or not math.isfinite(v):
        v = float(machine_score)
    return max(0.0, min(1.0, float(v)))

def safe_key(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", s)[:120]

# ---------- Anchor 进度 ----------
st.subheader(f"Anchor：{a}")
show_gif(try_paths(a))
filled_mask_all = df["final_score(0-1)_to_fill"].apply(is_finite)
st.write(f"当前 Anchor 进度：**{int(filled_mask_all.sum())}/{len(df)}** 已填写")
st.progress(0 if len(df)==0 else int(filled_mask_all.sum())/len(df))

# ---------- 层次划分 ----------
def levels_by_exact_score(x): return f"score={x:.2f}"
def levels_by_quantile(series, n_bins=4):
    q = pd.qcut(series.rank(method="first"), q=n_bins, labels=False, duplicates="drop")
    return (n_bins - 1 - q).astype(int)
def levels_by_thresholds(x, thr_list):
    for t in thr_list:
        if x >= t: return f"≥{t:.2f}"
    return f"<{thr_list[-1]:.2f}"

if MODE == "同分=同层":
    df["level"] = df["score"].apply(levels_by_exact_score)
    level_order = [f"score={v:.2f}" for v in sorted(df["score"].unique(), reverse=True)]
elif MODE == "分位数分层":
    n_bins = int(N_QUANT)
    df["_bin"] = levels_by_quantile(df["score"], n_bins)
    df["level"] = df["_bin"].apply(lambda b: f"Q{int(b)+1}/{n_bins}")
    level_order = [f"Q{i}/{n_bins}" for i in range(1, n_bins+1)]
    df = df.drop(columns=["_bin"])
else:
    raw = [x.strip() for x in THRESH_TXT.split(",") if x.strip()]
    thr_list = []
    for t in raw:
        try: thr_list.append(float(t))
        except: pass
    if not thr_list: thr_list = [0.85,0.70,0.50]
    thr_list = sorted(thr_list, reverse=True)
    df["level"] = df["score"].apply(lambda x: levels_by_thresholds(x, thr_list))
    level_order = [f"≥{t:.2f}" for t in thr_list] + [f"<{thr_list[-1]:.2f}"]

levels_avail = [lv for lv in level_order if lv in set(df["level"])]
if not levels_avail:
    st.warning("该 Anchor 无候选。"); st.stop()

st.session_state.level_idx = min(st.session_state.level_idx, len(levels_avail)-1)

# ---------- 层次选择 ----------
level_sel = st.selectbox("选择要打分的层次", levels_avail, index=st.session_state.level_idx)
st.session_state.level_idx = levels_avail.index(level_sel)

# ---------- 当前层展示 ----------
sub = df[df["level"] == level_sel].sort_values("score", ascending=False).head(int(TOPN_EACH)).copy()
st.markdown(f"### 层次：**{level_sel}** · 候选数：{len(sub)}")
st.caption("建议：1.00 极高相似；0.80–0.90 高度相似；0.50–0.70 部分相似；0.20–0.40 弱相似；0.00 不相似。")

filled_mask_lvl = sub["final_score(0-1)_to_fill"].apply(is_finite)
st.write(f"当前层进度：**{int(filled_mask_lvl.sum())}/{len(sub)}** 已填写")
st.progress(0 if len(sub)==0 else int(filled_mask_lvl.sum())/len(sub))

sliders = {}
rows = math.ceil(max(len(sub),1) / int(COLS))
idxs = list(sub.index)

for r in range(rows):
    cols = st.columns(int(COLS))
    for c in range(int(COLS)):
        i = r*int(COLS) + c
        if i >= len(idxs): break
        ridx = idxs[i]
        row  = sub.loc[ridx]
        with cols[c]:
            cand = str(row["candidate"])
            st.caption(f"`{cand}` · 机器分：**{row['score']:.3f}**")
            show_gif(try_paths(cand))

            base_val = parse_init_or_fallback(row["final_score(0-1)_to_fill"], row["score"])
            k = safe_key(f"sld__{a}__{level_sel}__{cand}")
            sliders[ridx] = st.slider("最终分 (0~1)", 0.0, 1.0, value=base_val, step=0.01, key=k)

# ---------- 保存 ----------
def atomic_save(df_all_new: pd.DataFrame, path: Path):
    if BACKUP:
        ts = time.strftime("%Y%m%d-%H%M%S")
        bak = path.with_suffix(path.suffix + f".{ts}.bak")
        try: shutil.copy2(path, bak)
        except Exception as e: st.warning(f"备份失败：{e}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    df_all_new.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)

def write_back(only_fill_empty=False):
    out_anchor = df.copy()
    for ridx, val in sliders.items():
        if only_fill_empty:
            if not is_finite(out_anchor.loc[ridx, "final_score(0-1)_to_fill"]):
                out_anchor.loc[ridx, "final_score(0-1)_to_fill"] = round(float(val), 3)
        else:
            out_anchor.loc[ridx, "final_score(0-1)_to_fill"] = round(float(val), 3)

    df_all_new = df_all.copy()
    mask_anchor = (df_all_new["anchor"] == a)
    df_all_new = pd.concat([df_all_new[~mask_anchor], out_anchor], ignore_index=True)
    df_all_new = df_all_new.sort_values(["anchor","score"], ascending=[True, False]).reset_index(drop=True)
    atomic_save(df_all_new, p)

def goto_next_level_or_anchor():
    """层→下一层；若已无下一层，跳到下一个 Anchor 的第一层。"""
    if st.session_state.level_idx + 1 < len(levels_avail):
        st.session_state.level_idx += 1
    else:
        if st.session_state.anchor_idx + 1 < len(anchors):
            st.session_state.anchor_idx += 1
        else:
            st.session_state.anchor_idx = 0
        st.session_state.level_idx = 0

def save_changes(only_fill_empty=False, switch_next=False):
    try:
        write_back(only_fill_empty=only_fill_empty)
        st.success(f"已保存到：{p}")
        st.session_state.ops = 0
        if switch_next:
            goto_next_level_or_anchor()
        _rerun()
    except Exception as e:
        st.error(f"保存失败：{e}")

# ---------- 操作区 ----------
b1, b2, b3, b4 = st.columns(4)
if b1.button("保存当前层次（写回总表）"):
    save_changes(only_fill_empty=False, switch_next=False)

if b2.button("仅保存空白项（已填不覆盖）"):
    save_changes(only_fill_empty=True, switch_next=False)

if b3.button("保存并切换下一层次 / 下一 Anchor"):
    save_changes(only_fill_empty=False, switch_next=True)

if b4.button("标完整个 Anchor 后一键跳下一个", disabled=not AUTO_NEXT_ANCHOR):
    if filled_mask_all.all():
        goto_next_level_or_anchor()
        _rerun()
    else:
        st.info("该 Anchor 还未全部打分，先完成再跳。")

# ---------- 自动保存 ----------
st.session_state.ops += 1
if st.session_state.ops >= int(AUTO_SAVE_N):
    try:
        write_back(only_fill_empty=False)
        st.toast("已自动保存")
        st.session_state.ops = 0
    except Exception as e:
        st.warning(f"自动保存失败：{e}")
