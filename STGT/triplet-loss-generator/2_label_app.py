# label_app.py
# -*- coding: utf-8 -*-
import os
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Triplet/Pairs/Positive 标注器", layout="wide")

# ====== 全局路径设置 ======
PAIR_TASKS_CSV = 'pair_tasks.csv'
TRIPLET_TASKS_CSV = 'triplet_tasks.csv'
POS_TASKS_CSV = 'pos_candidate_tasks.csv'

PAIR_OUT = 'pair_answers.csv'
TRIPLET_OUT = 'triplet_answers.csv'
POS_OUT = 'pos_candidate_answers.csv'

SHUFFLE_SEED = 42
# ===========================

st.sidebar.title("设置")
MODE = st.sidebar.selectbox('模式', ['Pair', 'Triplet', 'Positive', 'PosCandidate'])
shuffle = st.sidebar.checkbox('随机题序（固定种子）', value=True)

# 工具函数
def read_lines(path):
    if not os.path.exists(path): return []
    with open(path, 'r', encoding='utf-8') as f:
        return [ln.strip() for ln in f if ln.strip()]

def ensure_selected(choice, msg="请先选择一个选项再提交"):
    if choice is None:
        st.warning(msg)
        st.stop()

# ========== Pair 模式 ==========
if MODE == 'Pair':
    st.title("Pair 相似 / 不相似 标注")
    df_all = pd.read_csv(PAIR_TASKS_CSV)
    if shuffle:
        df_all = df_all.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)

    done_ids = set()
    for ln in read_lines(PAIR_OUT):
        parts = ln.split('|')
        if parts: done_ids.add(parts[0])
    df = df_all[~df_all['pair_id'].isin(done_ids)].reset_index(drop=True)
    total, finished = len(df_all), len(done_ids)

    st.caption(f"进度：{finished}/{total}")
    st.progress(finished / total if total > 0 else 0)

    if len(df) == 0:
        st.success(f"🎉 已全部标完！结果保存在 {PAIR_OUT}")
        st.stop()

    if 'idx' not in st.session_state: st.session_state.idx = 0
    idx = st.session_state.idx
    if idx >= len(df): idx = 0

    row = df.iloc[idx]
    col1, col2 = st.columns(2)
    col1.image(row.gif_left, use_column_width=True)
    col2.image(row.gif_right, use_column_width=True)

    choice = st.radio("是否相似？", ['1-相似', '0-不相似'], index=None, horizontal=True)
    if st.button("提交 / 下一题", key='pair_submit'):
        ensure_selected(choice)
        label = choice.split('-')[0]
        with open(PAIR_OUT, 'a', encoding='utf-8') as f:
            f.write(f"{row.pair_id}|{label}\n")
        st.session_state.idx += 1
        st.rerun()

# ========== Triplet 模式 ==========
elif MODE == 'Triplet':
    st.title("Triplet 核验（选出最相似两张）")
    if not os.path.exists(TRIPLET_TASKS_CSV):
        st.warning("未找到 triplet_tasks.csv")
        st.stop()

    df_all = pd.read_csv(TRIPLET_TASKS_CSV)
    if shuffle:
        df_all = df_all.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)

    done_keys = set()
    for ln in read_lines(TRIPLET_OUT):
        parts = ln.split('|')
        if len(parts) >= 3:
            done_keys.add('|'.join(parts[:3]))

    keys_all = df_all['anchor'].astype(str) + '|' + df_all['positive'] + '|' + df_all['negative']
    df = df_all[~keys_all.isin(done_keys)].reset_index(drop=True)
    total, finished = len(df_all), len(done_keys)

    st.caption(f"进度：{finished}/{total}")
    st.progress(finished / total if total > 0 else 0)

    if len(df) == 0:
        st.success(f"🎉 已全部标完！结果保存在 {TRIPLET_OUT}")
        st.stop()

    if 'idx_t' not in st.session_state: st.session_state.idx_t = 0
    idx = st.session_state.idx_t
    if idx >= len(df): idx = 0

    row = df.iloc[idx]
    cols = st.columns(3)
    cols[0].image(row.anchor,   caption="Anchor",   use_column_width=True)
    cols[1].image(row.positive, caption="Positive", use_column_width=True)
    cols[2].image(row.negative, caption="Negative", use_column_width=True)

    choice = st.radio("哪两张最相似？", ['A-P', 'A-N'], index=None, horizontal=True)
    if st.button("提交 / 下一题", key='triplet_submit'):
        ensure_selected(choice)
        with open(TRIPLET_OUT, 'a', encoding='utf-8') as f:
            f.write(f"{row.anchor}|{row.positive}|{row.negative}|{choice}\n")
        st.session_state.idx_t += 1
        st.rerun()

# ========== Positive 模式 ==========
elif MODE == 'Positive':
    st.title("Positive 候选中选出最相似（Anchor - Candidate）")
    if not os.path.exists(POS_TASKS_CSV):
        st.warning("未找到 pos_candidate_tasks.csv")
        st.stop()

    df_all = pd.read_csv(POS_TASKS_CSV)
    if shuffle:
        df_all = df_all.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)

    done_keys = set()
    for ln in read_lines(POS_OUT):
        parts = ln.split('|')
        if len(parts) >= 2:
            done_keys.add(parts[0])

    df = df_all[~df_all['task_id'].isin(done_keys)].reset_index(drop=True)
    total, finished = len(df_all), len(done_keys)

    st.caption(f"进度：{finished}/{total}")
    st.progress(finished / total if total > 0 else 0)

    if len(df) == 0:
        st.success(f"🎉 已全部标完！结果保存在 {POS_OUT}")
        st.stop()

    if 'idx_p' not in st.session_state: st.session_state.idx_p = 0
    idx = st.session_state.idx_p
    if idx >= len(df): idx = 0

    row = df.iloc[idx]
    st.image(row.gif_anchor, caption="Anchor", use_column_width=True)
    st.image(row.gif_candidate, caption="Candidate", use_column_width=True)

    choice = st.radio("是否最相似的候选？", ['是', '否'], index=None, horizontal=True)
    if st.button("提交 / 下一题", key='pos_submit'):
        ensure_selected(choice)
        label = '1' if choice == '是' else '0'
        with open(POS_OUT, 'a', encoding='utf-8') as f:
            f.write(f"{row.task_id}|{label}\n")
        st.session_state.idx_p += 1
        st.rerun()

# ============= PosCandidate 模式 =============
elif MODE == 'PosCandidate':
    st.title("PosCandidate 三选一标注（允许都不相似）")

    POS_TASKS_CSV = 'pos_candidate_tasks.csv'
    POS_OUT = 'pos_candidate_answers.csv'

    if not os.path.exists(POS_TASKS_CSV):
        st.info(f"未找到 {POS_TASKS_CSV}，请先生成任务 CSV")
        st.stop()

    df_all = pd.read_csv(POS_TASKS_CSV)
    if shuffle:
        df_all = df_all.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)

    # 读取已完成记录
    done_lines = read_lines(POS_OUT)
    done_ids = {ln.split('|')[0] for ln in done_lines if ln.strip()}

    df = df_all[~df_all['task_id'].isin(done_ids)].reset_index(drop=True)
    total = len(df_all)
    finished = total - len(df)

    st.caption(f"进度：{finished}/{total}")
    st.progress(finished / total if total > 0 else 0)

    if len(df) == 0:
        st.success(f"🎉 PosCandidate 问卷全部标注完成，结果保存在 {POS_OUT}")
        st.stop()

    if 'idx_pos' not in st.session_state:
        st.session_state.idx_pos = 0
    idx = st.session_state.idx_pos
    if idx >= len(df):
        st.session_state.idx_pos = 0
        idx = 0

    row = df.iloc[idx]

    # ===== 展示 Anchor 动图 =====
    st.markdown("### Anchor 场景")
    st.image(row.gif_anchor, caption="Anchor", use_container_width=True)

    # ===== 展示三个候选 =====
    st.markdown("### 选择与 Anchor 最相似的候选项")
    cand_paths = [row.gif_cand1, row.gif_cand2, row.gif_cand3]
    cols = st.columns([1, 1, 1])

    for i in range(3):
        with cols[i]:
            st.image(cand_paths[i], caption=f"候选{i+1}", use_container_width=True)

    # ===== 选择框（加入都不相似）=====
    choice = st.radio(
        "最相似的是哪一个候选？（若都不像请选 4）",
        options=['1', '2', '3', '4-都不相似'],
        index=None,
        horizontal=True
    )

    # ======= 提交按钮 =======
    if st.button("提交 / 下一题", key='pos_submit'):
        if st.session_state.get('lock_pos', False):
            st.stop()
        st.session_state.lock_pos = True

        ensure_selected(choice)
        label = choice.split('-')[0]  # 提取纯数字部分（如 '1'、'4'）
        with open(POS_OUT, 'a', encoding='utf-8') as f:
            f.write(f"{row.task_id}|{label}\n")

        st.session_state.idx_pos += 1
        st.session_state.lock_pos = False
        st.rerun()

