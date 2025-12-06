# ---------- 模块 1：数据拥有方 ----------
# Step 1: 生成密钥 + 加密查询样本
import os
import numpy as np
import tenseal as ts
import pandas as pd
import re
import unicodedata
from pathlib import Path

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

HIDDEN_SPACES = {
    "\u00A0": " ",  # 不换行空格
    "\u200B": "",   # 零宽空格
    "\u3000": " ",  # 全角空格
}

def clean_path(p: str) -> str:
    if not isinstance(p, str):
        p = str(p)
    # 1) 规范 Unicode（把全角等归一化）
    p = unicodedata.normalize("NFKC", p)
    # 2) 替换隐藏空白
    for bad, repl in HIDDEN_SPACES.items():
        p = p.replace(bad, repl)
    # 3) 去除两端空白，再规范分隔符
    p = p.strip()
    p = os.path.normpath(p)
    return p

def safe_load_npy(p: str):
    from numpy import load
    cp = clean_path(p)
    if not os.path.exists(cp):
        # 再尝试相对/绝对路径的变体
        alt = str(Path(cp).resolve())
        if os.path.exists(alt):
            cp = alt
        else:
            # 给出诊断信息，方便你定位
            raise FileNotFoundError(
                f"文件不存在：\n  原始: {repr(p)}\n  清洗: {repr(cp)}\n  绝对: {repr(alt)}\n  CWD : {os.getcwd()}"
            )
    return load(cp)

# 提取场景名称（与明文模块一致）
def extract_scene_name(file_name):
    if "REPLAY_" in file_name:
        match = re.search(r"REPLAY_\d+_(.*?)_result", file_name)
        return match.group(1) if match else file_name
    else:
        parts = file_name.split("_")
        return "_".join(parts[:3]) if len(parts) >= 3 else file_name

# 生成密钥对
def generate_and_save_keys(path_prefix="keys/context", scene_name=None):
    if scene_name is not None:
        path_prefix = f"{path_prefix}_{scene_name}"
    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60]
    )
    context.global_scale = 2**40
    context.generate_galois_keys()
    context.generate_relin_keys()

    with open(f"{path_prefix}_full.ctx", "wb") as f:
        f.write(context.serialize(
            save_public_key=True,
            save_secret_key=True,
            save_galois_keys=True,
            save_relin_keys=True
        ))

    with open(f"{path_prefix}_public.ctx", "wb") as f:
        f.write(context.serialize(
            save_public_key=True,
            save_secret_key=False,
            save_galois_keys=True,
            save_relin_keys=True
        ))

    return context

# 加密查询样本
def encrypt_query_sample(context, sample_tensor, scene_name):
    T, N, F = sample_tensor.shape
    flattened = sample_tensor.reshape(T, N * F)
    encrypted_vectors = [ts.ckks_vector(context, flattened[:, i].tolist()) for i in range(flattened.shape[1])]
    out_path = f"query/query_sample_{scene_name}_encrypted.npz"
    np.savez(out_path, *[v.serialize() for v in encrypted_vectors])

if __name__ == "__main__":
    manifest_path = arg_manifest()

    # 1) 读取或初始化 manifest
    manifest = load_manifest(manifest_path) if os.path.exists(manifest_path) else {
        "paths": {}, "platform_inputs": {}, "platform_outputs": {}, "postprocess": {}, "meta": {"version": "v1"}
    }

    # 2) 你自己的输入（可改为 CLI 参数）
    query_feature_path = "data/24052866952_outputscenario_V2_filtered_converted_encoded.npy"

    # 3) 计算 scene_name 与加密
    query_tensor = safe_load_npy(query_feature_path)  # [T, N, F]
    scene_name = extract_scene_name(os.path.basename(query_feature_path))
    ctx = generate_and_save_keys("keys/context", scene_name=scene_name)
    encrypt_query_sample(ctx, query_tensor, scene_name=scene_name)

    # 4) 写入 manifest
    manifest["scene_name"] = scene_name
    manifest["paths"].update({
        "query_feature": query_feature_path,
        "query_encrypted": f"query/query_sample_{scene_name}_encrypted.npz",
        "key_public":     f"keys/context_{scene_name}_public.ctx",
        "key_full":       f"keys/context_{scene_name}_full.ctx",
        "results_dir":    manifest["paths"].get("results_dir", "results/")
    })
    # 如已知平台侧输入，可一并写入，便于模块2直接使用
    manifest.setdefault("platform_inputs", {}).update({
        "candidate_dirs": manifest["platform_inputs"].get("candidate_dirs", []),
        "label_excels":   manifest["platform_inputs"].get("label_excels", []),
        "query_label":    manifest["platform_inputs"].get("query_label", "交叉口通行")
    })
    manifest["meta"]["created_at"] = manifest["meta"].get("created_at", now_iso())
    manifest["meta"]["updated_at"] = now_iso()

    save_manifest(manifest_path, manifest)
    print(f"[module1] manifest saved to: {manifest_path}")
