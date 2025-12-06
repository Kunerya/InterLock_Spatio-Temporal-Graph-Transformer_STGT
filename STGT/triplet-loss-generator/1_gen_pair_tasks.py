#!/usr/bin/env python
# gen_pair_tasks.py
"""
从 scene_index_mapping.csv 读取场景列表
→ 直接生成 pair_tasks.csv（Pair 标注列表）

用法示例：
    python gen_pair_tasks.py \
        --csv "" \
        --gif-dir "" \
        --n-each 10 \
        --seed 42 \
        --out pair_tasks.csv
"""

import argparse, random, csv, pathlib
import pandas as pd

def load_scene_ids(csv_path: str) -> list[str]:
    """自动识别列名，返回场景 ID 列表"""
    df = pd.read_csv(csv_path)
    for col in ('scene_id', 'scene_name', 'scene'):
        if col in df.columns:
            return df[col].astype(str).tolist()
    raise ValueError("CSV 内必须包含 scene_id 或 scene_name 列！")

def build_pairs(scene_ids: list[str], n_each: int, seed: int) -> list[tuple[str,str]]:
    """为每个场景随机挑 n_each 个不同场景组成无序对，返回去重后的配对列表"""
    random.seed(seed)
    pair_set = set()
    for s in scene_ids:
        candidates = random.sample([x for x in scene_ids if x != s],
                                   min(n_each, len(scene_ids) - 1))
        for c in candidates:
            pair_set.add(tuple(sorted((s, c))))
    return sorted(pair_set)

def write_pair_csv(pairs: list[tuple[str,str]],
                   gif_dir: pathlib.Path,
                   out_csv: str):
    """输出 pair_tasks.csv：pair_id, gif_left, gif_right"""
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['pair_id', 'gif_left', 'gif_right'])
        for i, (a, b) in enumerate(pairs):
            w.writerow([
                f'p{i:05d}',
                str(gif_dir / f'{a}.gif'),
                str(gif_dir / f'{b}.gif')
            ])
    print(f'✅ 已生成 {len(pairs)} 对 → {out_csv}')

def main():
    parser = argparse.ArgumentParser(description='生成 Pair 标注任务 CSV')
    parser.add_argument('--csv',     default='scene_index_mapping.csv',
                        help='场景映射表 CSV 路径')
    parser.add_argument('--gif-dir', required=True,
                        help='GIF 文件所在目录')
    parser.add_argument('--n-each',  type=int, default=10,
                        help='每个场景随机配对数量')
    parser.add_argument('--seed',    type=int, default=42,
                        help='随机种子（可换以追加配对）')
    parser.add_argument('--out',     default='pair_tasks.csv',
                        help='输出 CSV 文件名')
    args = parser.parse_args()

    scene_ids = load_scene_ids(args.csv)
    pairs     = build_pairs(scene_ids, args.n_each, args.seed)
    write_pair_csv(pairs, pathlib.Path(args.gif_dir), args.out)

if __name__ == '__main__':
    main()
