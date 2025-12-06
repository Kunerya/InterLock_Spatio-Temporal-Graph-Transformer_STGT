import os
import numpy as np
import tenseal as ts
import pandas as pd
import re
# ---- manifest utils ----
import json, os, argparse, datetime

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

# ---------- 模块 3：数据拥有方 ----------
# Step 3: 解密结果 + 开方 + 指数映射 + 排序保存

def decrypt_and_postprocess_results(result_dir, full_ctx_path, scene_name, alpha=0.01):
    context = ts.context_from(open(full_ctx_path, "rb").read())

    result_path = os.path.join(result_dir, f"{scene_name}.result.npz")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"未找到密文结果文件：{result_path}")

    data = np.load(result_path)
    encrypted_dists = [ts.ckks_vector_from(context, data[key].tobytes()) for key in data.files]
    dist_sq_list = [v.decrypt()[0] for v in encrypted_dists]  # 每个密文仅包含一个 float

    candidates_path = os.path.join(result_dir, f"{scene_name}_candidates.txt")
    if not os.path.exists(candidates_path):
        raise FileNotFoundError(f"未找到候选名文件：{candidates_path}")

    with open(candidates_path, "r", encoding="utf-8") as f:
        cand_names = [line.strip() for line in f.readlines()]

    result_list = []
    for name, dist_sq in zip(cand_names, dist_sq_list):
        dist = np.sqrt(dist_sq)
        sim = np.exp(-alpha * dist)
        result_list.append((name, sim))

    result_list.sort(key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(result_list, columns=["Matched_File", "Similarity"])
    csv_path = f"{scene_name}_final_similarity_result.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"最终查重结果已保存为 CSV：{csv_path}")

if __name__ == "__main__":
    manifest_path = arg_manifest()
    manifest = load_manifest(manifest_path)

    scene_name = manifest["scene_name"]
    paths  = manifest["paths"]
    pouts  = manifest["platform_outputs"]
    ppost  = manifest.get("postprocess", {})
    alpha  = float(ppost.get("alpha", 0.01))

    decrypt_and_postprocess_results(
        result_dir=os.path.dirname(pouts["result_npz"]),
        full_ctx_path=paths["key_full"],
        scene_name=scene_name,
        alpha=alpha
    )

    final_csv = f"{scene_name}_final_similarity_result.csv"
    manifest.setdefault("postprocess", {})["final_csv"] = final_csv
    manifest["meta"]["updated_at"] = now_iso()
    save_manifest(manifest_path, manifest)
    print(f"[module3] final csv saved; manifest updated: {manifest_path}")
