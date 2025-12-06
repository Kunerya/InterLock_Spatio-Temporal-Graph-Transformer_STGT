import os
import numpy as np
import tenseal as ts
import pandas as pd
import re
import json, os, argparse
import glob, unicodedata
from pathlib import Path
from typing import Optional
from datetime import datetime

# ---- manifest utils ----
def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")

def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_manifest(path: str, data: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def arg_manifest(default="job_manifest.json"):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=os.environ.get("MANIFEST_PATH", default))
    return p.parse_args().manifest

# ---------- 模块 2：平台端 ----------
# Step 2: 读取加密查询样本 + 与明文候选集计算距离 + 保存密文结果

# 提取场景名称（与明文模块一致）
def extract_scene_name(file_name):
    if "REPLAY_" in file_name:
        match = re.search(r"REPLAY_\d+_(.*?)_result", file_name)
        return match.group(1) if match else file_name
    else:
        parts = file_name.split("_")
        return "_".join(parts[:3]) if len(parts) >= 3 else file_name

def load_encrypted_query(path, context):
    data = np.load(path)
    return [ts.ckks_vector_from(context, data[key].tobytes()) for key in data.files]


def compute_encrypted_distance(enc_query_vecs, plain_sample):
    # plain_sample shape: [T, D]，转置后与 enc_query_vecs 对应
    flat_plain = plain_sample.T  # shape: [D, T]
    
    # 对每个维度执行差值平方（结果是密文向量）
    diffs = [(v_enc - v_plain.tolist()) ** 2 for v_enc, v_plain in zip(enc_query_vecs, flat_plain)]
    
    # 先对所有维度求和（结果仍为密文向量），再聚合为单一密文标量
    return sum(diffs).sum()


# 加载标签文件（与明文模块一致）
def get_scene_label_map(label_excel_paths):
    if isinstance(label_excel_paths, str):
        label_excel_paths = [label_excel_paths]
    combined_df = pd.concat([pd.read_excel(p) for p in label_excel_paths], ignore_index=True)
    return dict(zip(combined_df["Scene Name"].astype(str), combined_df["Labels"].astype(str)))

def platform_compute_and_save(query_enc_path, candidate_dir, output_path, pub_ctx_path,
                               query_label="交叉口通行", label_excel_paths=None, scene_name = None):
    context = ts.context_from(open(pub_ctx_path, "rb").read())
    enc_query = load_encrypted_query(query_enc_path, context)

    if query_label is None:
        raise ValueError(f"请提供场景 {scene_name} 的标签")

    scene_label_map = get_scene_label_map(label_excel_paths)
    valid_scene_names = {k for k, v in scene_label_map.items() if v == query_label}

    encrypted_scores = []
    candidate_names = []
    if isinstance(candidate_dir, str):
        candidate_paths = [os.path.join(candidate_dir, fname) for fname in os.listdir(candidate_dir) if fname.endswith(".npy")]
    else:
        candidate_paths = []
        for dir_path in candidate_dir:
            candidate_paths.extend([
                os.path.join(dir_path, fname)
                for fname in os.listdir(dir_path)
                if fname.endswith(".npy")
            ])

    for fpath in candidate_paths:
        fname = os.path.basename(fpath)
        cand_scene = extract_scene_name(fname)
        if cand_scene not in valid_scene_names:
            continue
        tensor = np.load(fpath)  # [T, N, F]
        flat = tensor.reshape(tensor.shape[0], -1)  # [T, D]
        dist = compute_encrypted_distance(enc_query, flat)
        encrypted_scores.append(dist)  # dist 是 CKKSVector
        candidate_names.append(cand_scene)

    os.makedirs(output_path, exist_ok=True)

    # ✅ 保存为 .npz 文件，每个距离是一个密文
    with open(os.path.join(output_path, f"{scene_name}.result.npz"), "wb") as f:
        np.savez(f, *[v.serialize() for v in encrypted_scores])

    # 保存候选文件名列表
    with open(os.path.join(output_path, f"{scene_name}_candidates.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(candidate_names))

    print("平台端密文相似度向量已保存")

HIDDEN_SPACES = {"\u00A0": " ", "\u200B": "", "\u3000": " "}

def clean_path(p: str) -> str:
    if not isinstance(p, str): p = str(p)
    p = unicodedata.normalize("NFKC", p)
    for bad, rep in HIDDEN_SPACES.items(): p = p.replace(bad, rep)
    p = p.strip()
    # 若出现 "dir file.xlsx" 且左侧确为目录，则将第一个空格还原为分隔符
    if " " in p:
        left, right = p.split(" ", 1)
        if os.path.isdir(left) and os.path.sep not in left and not right.startswith(os.path.sep):
            p = left + os.path.sep + right
    return os.path.normpath(p)

def resolve_path(p: str, must_exist: bool=False) -> str:
    p_clean = clean_path(p)
    here = Path(__file__).resolve().parent
    candidates = [Path(p_clean), here / p_clean, here / Path(p_clean).name]
    for c in candidates:
        if not must_exist or c.exists():
            return str(c.resolve())
    return str((here / p_clean).resolve())

def load_manifest_if_any() -> Optional[dict]:
    mpath = os.environ.get("MANIFEST_PATH", "job_manifest.json")
    if os.path.exists(mpath):
        with open(mpath, "r", encoding="utf-8") as f:
            m = json.load(f)
        m["_manifest_path"] = mpath
        return m
    return None

def save_manifest_if_loaded(m: dict) -> None:
    mpath = m.get("_manifest_path")
    if not mpath:
        return
    m.setdefault("meta", {})["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
# =========================  替换 __main__ 从这里开始  =========================
if __name__ == "__main__":
    # 1) 优先读取 manifest；没有就自动发现最近的加密查询样本
    manifest = load_manifest_if_any()

    if manifest:
        scene_name = manifest.get("scene_name")
        paths = manifest.get("paths", {})
        pin   = manifest.get("platform_inputs", {})

        query_enc_path    = paths.get("query_encrypted")
        pub_ctx_path      = paths.get("key_public")
        output_path       = paths.get("results_dir", "results/")
        candidate_dir     = pin.get("candidate_dirs", [])
        label_excel_paths = pin.get("label_excels", [])
        query_label       = pin.get("query_label", "交叉口通行")

        if not scene_name:
            # 兜底：从 query_encrypted 文件名中提取
            if query_enc_path:
                m = re.search(r"query_sample_(.+)_encrypted\.npz$", os.path.basename(query_enc_path))
                if m: scene_name = m.group(1)

    else:
        # 无 manifest：自动发现 query/ 下最新的加密样本，并据此推断 scene_name
        hits = sorted(
            glob.glob("query/query_sample_*_encrypted.npz"),
            key=lambda p: os.path.getmtime(p),
            reverse=True
        )
        if not hits:
            raise FileNotFoundError("未找到加密查询样本：query/query_sample_*_encrypted.npz")
        query_enc_path = hits[0]
        m = re.search(r"query_sample_(.+)_encrypted\.npz$", os.path.basename(query_enc_path))
        if not m:
            raise RuntimeError(f"无法从文件名中解析 scene_name：{query_enc_path}")
        scene_name = m.group(1)

    # 2) 路径清洗 + 存在性检查（避免 “No objects to concatenate” 这类隐性错误）
    query_enc_path = resolve_path(query_enc_path, must_exist=True)
    pub_ctx_path   = resolve_path(pub_ctx_path,   must_exist=True)
    output_path    = resolve_path(output_path,    must_exist=False)

    if isinstance(candidate_dir, str):
        candidate_dir = [candidate_dir]
    candidate_dir = [resolve_path(d, must_exist=True) for d in candidate_dir]

    if not label_excel_paths:
        raise ValueError("label_excel_paths 为空，平台端无法按标签筛选，请检查传参或 manifest。")
    label_excel_paths = [resolve_path(p, must_exist=True) for p in label_excel_paths]

    # 3) 运行主流程
    print("[INFO] scene_name        =", scene_name)
    print("[INFO] query_enc_path    =", query_enc_path)
    print("[INFO] pub_ctx_path      =", pub_ctx_path)
    print("[INFO] output_path       =", output_path)
    print("[INFO] candidate_dir     =", candidate_dir)
    print("[INFO] label_excel_paths =", label_excel_paths)
    print("[INFO] query_label       =", query_label)

    platform_compute_and_save(
        query_enc_path=query_enc_path,
        candidate_dir=candidate_dir,
        output_path=output_path,
        pub_ctx_path=pub_ctx_path,
        query_label=query_label,
        label_excel_paths=label_excel_paths,
        scene_name=scene_name
    )

    # 4) 若来自 manifest，则把平台输出路径写回，便于 module3 使用
    if manifest is not None:
        manifest.setdefault("platform_outputs", {}).update({
            "result_npz":     str(Path(output_path) / f"{scene_name}.result.npz"),
            "candidates_txt": str(Path(output_path) / f"{scene_name}_candidates.txt"),
        })
        save_manifest_if_loaded(manifest)
        print("[INFO] manifest 已更新平台输出路径")

