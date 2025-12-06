# InterLock Spatio-Temporal Graph Transformer (STGT)

**InterLock** is an integrated framework for privacy-preserving scenario similarity computation and deduplication in shared autonomous-driving scenario libraries. Within InterLock, we provide a **PyTorch implementation of the Spatio-Temporal Graph Transformer (STGT)** that encodes ego-centric interaction graphs into fixed-length, homomorphic-encryption-friendly scenario embeddings for metric similarity evaluation. Building on these embeddings, the **ciphertext-domain-computation** module leverages the **CKKS** scheme in **TenSEAL** and consists of three Python scripts that implement an end-to-end privacy-preserving similarity-computation workflow between data owners (users) and the platform.

This repository provides the reference implementation of the Spatio-Temporal Graph Transformer (STGT) used in InterLock, a privacy-preserving framework for ciphertext-domain similarity computation of autonomous driving scenarios, as illustrated below.

<p align="center">
  <img src="STGT\figures\Framework of InterLock.jpg"
       alt="InterLock framework for ciphertext-domain similarity computation of autonomous driving scenarios"
       width="70%">
</p>

STGT takes tensorized ego-centric spatiotemporal interaction graphs as input and produces **compact, fixed-dimensional, metrically comparable embeddings** that are compatible with homomorphic-encryption–based (HE) distance computation in the ciphertext domain.

---

## Highlights

- **HE-friendly scenario embeddings**: fixed-length embeddings with bounded numeric ranges, suitable for CKKS-based encrypted distance evaluation.
- **Spatiotemporal interaction modeling**: spatial graph encoder over multi-agent interaction graphs + temporal Transformer over frame-wise features.
- **Hybrid training objective**: combines triplet loss and regression on human-annotated similarity scores to improve both ranking quality and score calibration.
- **Integration with InterLock**: designed as the embedding backbone for privacy-preserving scenario similarity computation.

---

## Repository Structure

```text
InterLock/
README.md               # This file
  STGT/
    configs/                # Training and evaluation configs (YAML/JSON)
    models/                 # Model components (STGT, encoders, heads, etc.)
    triplet-loss-generator/ # Scripts for building triplet data from annotations
    data/                   # Empty in this repo; see data/README.txt
    results/                # (optional) Plots / metrics

    dataset.py              # Scenario-graph dataset and dataloaders
    model.py                # STGT model definition
    loss.py                 # Loss functions (triplet + regression)
    calculator.py           # Embedding extraction and similarity computation
    train_triplet.py        # Main training script
    plot_metrics.py         # Utilities for plotting training curves
    spatial_ego_rgat.py     # Ego-centric spatial graph encoder
    env.yml                 # Conda environment (recommended)
    run.sh                  # Example training script (optional)

  ciphertext-domain-computation/
    data/                   # Plaintext embeddings used in the demo
    keys/                   # CKKS context, public key, and secret key
    query/                  # Encrypted query embeddings
    results/                # Encrypted distances and decrypted rankings
    module1_user.py         # User-side key generation and query encryption
    module2_platform.py     # Platform-side ciphertext-domain distance computation
    module3_user_*.py       # User-side decryption and ranking script
    README.md               # Usage of the ciphertext-domain computation module
```

The directory `STGT/triplet-loss-generator/` contains a more detailed README.txt describing the multi-stage pipeline used to construct human-annotated triplet data for training.

---

## Installation

We recommend using Conda:
```bash
git clone https://github.com/Kunerya/InterLock_Spatio-Temporal-Graph-Transformer_STGT.git
cd InterLock_STGT/

conda env create -f env.yml
conda activate stgt
```
If you prefer `pip`, please install the dependencies listed in `env.yml` (or in your own `requirements.txt`), including PyTorch, NumPy, Pandas, PyYAML, etc.

---

## Data Preparation

**Real autonomous driving scenario data are not included in this public repository due to privacy constraints.**

In our internal experiments, the `data/` directory contains:

- interaction-extracted graph-tensor files for each scenario (e.g., `*.pkl`);

- a `scene_index_mapping.csv` file that maps raw scenario identifiers and timestamps to tensor indices.

The expected formats of these files are defined by `dataset.py` and the configuration files under `configs/`. To run the code on your own data, you need to:

1. Generate ego-centric spatiotemporal interaction graphs for each scenario (e.g., based on trajectories and interaction labels).

2. Serialize them into tensor files (`.pkl`) following the same structure consumed by `dataset.py`.

3. Build a `scene_index_mapping.csv` that records, for each scenario, the mapping between original IDs / timestamps and the tensor indices.

For quick testing, you may create a small toy dataset with a few synthetic scenarios and adjust the config files accordingly.

Please see `data/README.txt` for a concise description of the expected contents of this directory.

---

## Training STGT

A typical training run with triplet + regression objectives can be launched as:

```bash
python train_triplet.py \
  --config configs/stgt_triplet_example.yaml
```

Key configuration items (see the YAML file for details) include:

- dataset paths and file names under `data/`;

- model hyperparameters (embedding dimension, number of GNN/Transformer layers, attention heads, etc.);

- loss weights for triplet loss and regression loss;

- optimizer, batch size, and learning-rate schedule;

- evaluation settings (validation split, retrieval metrics, logging).

During training, the script periodically evaluates retrieval metrics (e.g., nDCG, rank correlation) on a validation set and saves checkpoints under `checkpoints_v2/` (or another directory specified in the config).

---

## Scenario Embedding and Similarity Evaluation

Once a model is trained (or a pretrained checkpoint is provided), you can extract embeddings and compute similarity scores using:

```bash
python calculator.py \
  --config configs/stgt_eval_example.yaml \
  --checkpoint checkpoints_v2/stgt_best.pth
```

This script will:

1. Load the trained STGT model.

2. Encode each scenario into a fixed-dimensional embedding.

3. Compute distance-based similarity scores (Euclidean).

4. Optionally export embeddings for downstream tasks such as scenario retrieval and ranking, or for encrypted distance evaluation using the ciphertext-domain computation module under ```ciphertext-domain-computation/.```

---

## Ciphertext-Domain Similarity Computation

The actual ciphertext-domain distance computation is implemented in the `ciphertext-domain-computation/` module at the repository root. This module builds on the CKKS scheme provided by TenSEAL and contains three Python scripts that realize a privacy-preserving workflow between the data owner (user) and the platform.

The subdirectories used by this module are:

- `data/` – plaintext scenario embeddings for the library and a sample plaintext query embedding used for verification.
- `keys/` – CKKS context and key files (public key and secret key).
- `query/` – encrypted query embeddings generated by the data owner.
- `results/` – encrypted distance outputs produced by the platform and the final decrypted similarity rankings.

The three steps of ciphertext-domain similarity computation are:

1. **Step 1 – Data owner (user side): `module1_user.py`**  
   The user generates a CKKS context and key pair, encrypts the query scenario embedding with the public key, and writes:
   - the encryption context and public key (e.g., `context_*_full.ctx`)
   - the encrypted query embedding (e.g., `query_*_encrypted.npz`)

   The public key and encrypted query are then sent to the platform, while the secret key is kept locally by the user.

2. **Step 2 – Platform side: `module2_platform.py`**  
   The platform loads the encrypted query and context, computes distances between the encrypted query and the stored scenario-library embeddings **directly in the ciphertext domain**, and saves the results in encrypted form (e.g., `*_candidates.txt`, `*.result.npz`) under `results/`. The platform never accesses plaintext embeddings or similarity scores.

3. **Step 3 – Data owner (user side): decryption and ranking**  
   On the user side, a third script is used to load the encrypted distance results from `results/`, decrypt them with the locally stored secret key, sort the candidates, and export the final ranking as `*_final_similarity_result.csv`.

This separation of roles ensures that the platform can perform scenario similarity search over encrypted embeddings without learning either the raw scenario data or the actual similarity scores, enabling privacy-preserving ciphertext-domain similarity computation within InterLock.

---

## Triplet-Loss Generator

The folder `triplet-loss-generator/` contains a series of scripts used to construct human-annotated triplet data (anchor–positive–negative) from multiple rounds of labeling and post-processing, including:

- generation of pairwise similarity tasks,

- Streamlit-based annotation apps,

- parsing and scoring of answers,

- construction and cleaning of positive / negative candidate sets,

- final fine-grained similarity scoring.

All CSV files derived from real driving scenarios are **not included** in this repository, but the scripts and triplet-loss-generator/README.txt document the expected formats so that you can adapt the pipeline to your own data.

---
## Citation

This code accompanies the following manuscript:

> L. Xiao et al., “InterLock: Interaction Learning under Cipher-Lock for Privacy-Preserving Similarity of Autonomous Driving Scenarios,” manuscript submitted to *IEEE Transactions on Intelligent Transportation Systems (T-ITS)*, 2025.

This work is currently under review.

---

## License

This code is released for non-commercial research purposes only. All rights reserved. Please contact the authors if you are interested in commercial use.