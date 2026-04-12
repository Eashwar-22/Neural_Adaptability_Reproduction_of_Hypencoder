# Neural Adaptability: Hypencoder Reproduction & Verification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

This repository contains a rigorous reproduction and scientific verification study of **Hypencoder**, a state-of-the-art neural retrieval model that implements **Neural Adaptability**. 

The core innovation of Hypencoder is the **Q-Net**—a hyper-network that dynamically alters the document encoder's weights based on the query. This allows the model to move beyond static dot-product Bi-Encoders and learn complex, non-linear relevance functions typically reserved for expensive Cross-Encoders.

![Hypencoder Architecture](./imgs/main_figure.jpg)

---

## 🚀 Key Features
- **Neural Adaptability**: Query-dependent document re-encoding via hyper-networks.
- **Scientific Verification**: Targeted experiments validating the core claims of the Hypencoder paper (Section 3.7).
- **H100 Benchmarks**: Real-world efficiency metrics (QPS/Latency) on high-performance hardware.
- **Zero-Shot Generalization**: Robust evaluation across diverse BEIR datasets (TREC-COVID, NFCorpus, FiQA, etc.).

---

## 📊 Benchmark Results

Our retrained model (**Hypencoder (Mine)**) achieves results statistically comparable to the original paper, even exceeding it in top-rank precision (MRR).

| Dataset | Metric | Hypencoder (Paper) | **Hypencoder (Mine)** |
| :--- | :--- | :--- | :--- |
| **TREC DL '19 & '20** | nDCG@10 | 0.736 | **0.727** |
| | MRR | 0.885 | **0.964** |
| **MS MARCO Dev** | MRR@10 | 0.386 | **0.384** |
| **TREC-COVID** | nDCG@10 | 0.651 | **0.661** |
| **NFCorpus** | nDCG@10 | 0.324 | **0.329** |

---

## 🔬 Scientific Verification (Section 3.7)

We conducted three controlled experiments to verify the theoretical foundations of the Hypencoder architecture:

### 1. The Replication Claim
> *"Hypencoder can exactly replicate any standard Bi-Encoder."*
- **Status**: **Partially Verified**. 
- **Finding**: While performance matches Bi-Encoders on in-domain data, there is a slight generalization gap in zero-shot settings, suggesting that while the *architecture* can mimic a Bi-Encoder, the *distillation* adds complexity that impacts transferability.

### 2. The Superiority Claim
> *"Hypencoder can learn non-linear functions that Bi-Encoders cannot."*
- **Status**: ✅ **Verified**.
- **Finding**: When trained on the exact same data, Hypencoder significantly outperforms a standard BERT-base Bi-Encoder (up to **+18.3%** on TREC-COVID), proving the architectural advantage.

### 3. The Expressiveness Claim
> *"Hypencoder is a more efficient knowledge container than a Bi-Encoder."*
- **Status**: ✅ **Verified**.
- **Finding**: Hypencoder closes the performance gap between Bi-Encoders and Cross-Encoder Teachers by over **50%** across multiple datasets.

---

## ⚡ Efficiency (H100 Benchmarks)

Retrieved items are scored efficiently using the **Approximate Graph Search** index, balancing the non-linear power of Hypencoder with the speed required for large-scale retrieval.

| Dataset | Method | Corpus Size | Avg Latency (ms) | QPS |
| :--- | :--- | :--- | :--- | :--- |
| **NFCorpus** | Exact | 3.6k | 30.0 | 32.6 |
| **TREC DL '20** | Approx (Graph) | 8.8M | **368.0** | 2.7 |

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/Eashwar-22/Neural_Adaptability_Reproduction_of_Hypencoder.git
cd Neural_Adaptability_Reproduction_of_Hypencoder

# Install dependencies
pip install -e .
```

---

## 📖 Usage

### Inference (HuggingFace)
You can load the official or retrained checkpoints directly using the model class:
```python
from hypencoder_cb.modeling.hypencoder import HypencoderDualEncoder

model = HypencoderDualEncoder.from_pretrained("Eashwar-22/thesis_hypencoder")
```

### Evaluation
To run the full evaluation suite on BEIR datasets:
```bash
python scripts/run_eval_only.py --config configs/eval_retrained.yaml
```

---

## 📂 Project Structure
- `hypencoder_cb/`: Core modeling logic and Q-Net implementation.
- `scripts/`: Training, evaluation, and verification scripts.
- `configs/`: YAML configurations for experimental runs.
- `imgs/`: Architecture diagrams and training visualizations.

---
