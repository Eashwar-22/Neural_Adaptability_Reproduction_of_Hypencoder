# Scripts Directory

This directory contains all the shell and python scripts used for training, inference, and verification of the Hypencoder and Hypen-ColBERT models.

## Directory Structure

*   **`training/`**: SLURM and bash scripts to launch training jobs (e.g., `train_8gpu_hypen_colbert_multihead.sh`).
*   **`inference/`**: Scripts to run inference and benchmarks (e.g., `run_msmarco_...`, `run_trec_...`).
*   **`verification/`**: Python scripts to verify claims and architectures (e.g., `verify_multi_head_logic.py`, `verify_claim_3_7.py`).
*   **`setup/`**: Environment setup and data conversion scripts.
*   **`utils/`**: Helper utilities and library code specific to scripts.
*   **`legacy/`**: Deprecated or old scripts.

## Common Tasks

**Train Multi-Head HypenColBERT:**
```bash
sbatch scripts/training/hypencolbert_multihead.sh
```

**Verify Logic:**
```bash
python scripts/verification/verify_multi_head_logic.py
```

## Directory Structure
*   **`training/`**: `hypencolbert_multihead.sh`, `hypencoder_retrained.sh`, etc.
*   **`inference/`**: `inference_hypencoder_retrained.sh`, `inference_hypencoder_pretrained.sh`, `inference_hypencolbert.sh`, etc.
*   **`verification/`**: `verify_claim_3_7.py`, `verify_multi_head_logic.py`, etc.
