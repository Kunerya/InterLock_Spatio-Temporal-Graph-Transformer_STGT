# make_pos_from_pair.py
import pandas as pd
import random
from pathlib import Path
from collections import defaultdict

random.seed(11)

# 文件路径
PAIR_TASKS_CSV = 'pair_tasks.csv'
PAIR_ANSWERS_CSV = 'pair_answers.csv'
OUTPUT_CSV = 'pos_candidate_tasks.csv'

# 读取任务和答案
df_task = pd.read_csv(PAIR_TASKS_CSV)
answer_map = {}
with open(PAIR_ANSWERS_CSV, 'r', encoding='utf-8') as f:
    for ln in f:
        if '|' in ln:
            pair_id, label = ln.strip().split('|')
            answer_map[pair_id] = label

# 构建 anchor → set(相似样本) 的反向索引
anchor2positives = defaultdict(set)

for _, row in df_task.iterrows():
    pid, left, right = row['pair_id'], row['gif_left'], row['gif_right']
    label = answer_map.get(pid)
    if label != '1':  # 只保留相似项
        continue

    # 将路径标准化为 scenes/xxx.gif 相对路径
    def to_rel(p): return f"scenes/{Path(p).stem}.gif"

    anchor2positives[to_rel(left)].add(to_rel(right))
    anchor2positives[to_rel(right)].add(to_rel(left))

# 构建三选一问卷
records = []
for anchor, cands in anchor2positives.items():
    if len(cands) < 3:
        continue
    cand_list = list(cands)
    random.shuffle(cand_list)
    for i in range(0, len(cand_list) - 2, 3):
        triplet = cand_list[i:i+3]
        if len(triplet) < 3:
            continue
        records.append([
            f'pos3_{Path(anchor).stem}_{i//3}',
            anchor,
            triplet[0],
            triplet[1],
            triplet[2],
        ])

# 保存
df_out = pd.DataFrame(records, columns=[
    'task_id', 'gif_anchor', 'gif_cand1', 'gif_cand2', 'gif_cand3'
])
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"✔ 已生成基于 Pair 相似标注的三选一问卷，共 {len(df_out)} 条")
