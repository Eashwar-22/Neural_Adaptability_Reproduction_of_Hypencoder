# Scripts Directory

This directory contains all scripts used in the thesis, organized by **Research Question (RQ)** for easy navigation.

## Directory Structure (Thesis-Aligned)

| Directory | Thesis Section | Description |
|---|---|---|
| `rq1_reproduction/` | **RQ1** — Baseline Verification | Training and evaluation scripts for reproducing the original Hypencoder results. |
| `rq2_interpretability/` | **RQ2** — Q-Net Mechanistic Analysis | Q-Net fingerprinting, depth analysis, saliency maps, and UMAP visualizations. |
| `rq3_scaling/` | **RQ3** — SOTA Teacher Distillation | Training with MXBAI teacher and statistical significance testing. |
| `data_prep/` | Chapter 3 — Methodology | Hard negative mining and distillation data preparation. |
| `verification/` | Chapter 4 — Verification | Scripts to verify training improvements (dropout, regularization). |

### Other Directories

*   **`analysis/`**: Additional analysis scripts (cluster properties, dataset stats, thesis plots, etc.).
*   **`benchmarking/`**: Benchmarking and profiling scripts.
*   **`inference/`**: Inference and retrieval scripts.
*   **`train/`**: Additional training scripts (control bi-encoders, distillation variants).
*   **`training/`**: Legacy SLURM training scripts (ColBERT variants).
*   **`setup/`**: Environment setup and data conversion scripts.
*   **`utils/`**: Helper utilities (evaluation, conversion, debugging).
*   **`data/`**: Additional data preparation scripts.
*   **`legacy/`**: Deprecated scripts.

## Quick Start

**RQ1 — Reproduce Hypencoder Training:**
```bash
sbatch scripts/rq1_reproduction/train_hypencoder_retrained.sh
```

**RQ2 — Run Q-Net Fingerprint Analysis:**
```bash
python scripts/rq2_interpretability/QNet_fingerprint.py
```

**RQ3 — Train with MXBAI Teacher:**
```bash
sbatch scripts/rq3_scaling/train_hypencoder_mxbai.sh
```

**Verify Training Improvements:**
```bash
python scripts/verification/verify_training_improvements.py
```
