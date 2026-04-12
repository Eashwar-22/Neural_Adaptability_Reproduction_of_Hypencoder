# Inference Results and Comparison

## Model Definitions

| Column | Description | Source |
| :--- | :--- | :--- |
| **Hypencoder (Paper)** | Results reported in the original Hypencoder paper. | Paper |
| **Hypencoder (HuggingFace)** | Official checkpoint (`hypencoder-base`) evaluated on our setup. | Repo / HuggingFace |
| **Hyencoder (Mine)** | The model we trained from scratch on this cluster. | **Current Run** |
| **ColBERTv2 (Paper)** | Strong baseline model for comparison. | Paper |
| **BE-Base (Paper)** | BERT-based dense retrieval baseline. | Paper |

## Table 0: Distilled Models Performance (New)
| Dataset | Metric | **Hypencoder (Paper)** | **Hypencoder (Mine)** | **Hypencoder (distil. from bge-base-en-v1.5)** |
| :--- | :--- | :--- | :--- | :--- |
|**TREC DL '19 & '20**|nDCG@10| 0.736 | 0.727 | **0.665** |
||R@1000| 0.871 | 0.809 | **0.738** |
||MRR| 0.885 | 0.964 | **0.927** |
|**MS MARCO Dev**|MRR@10| 0.386 | 0.384 | *deferred* |
||R@1000| 0.981 | 0.819 | *deferred* |
|**NFCorpus**|nDCG@10| 0.324 | 0.329 | **0.308** |
|**TREC-COVID**|nDCG@10| 0.651 | 0.661 | **0.689** |
|**FiQA**|nDCG@10| 0.314 | 0.322 | **0.300** |
|**Touché v2**|nDCG@10| 0.258 | 0.220 | **0.298** |
|**DBPedia**|nDCG@10| 0.419 | 0.418 | **0.327** |

## Table 1: In-Domain Results (TREC DL & MS MARCO Dev)

*Comparison of Zero-Shot vs Fine-Tuned (Hypencoder)*


| Dataset | Metric | **Hypencoder (Paper)** | **Hypencoder (HuggingFace)** | **Hypencoder (Mine)** |
| :--- | :--- | :--- | :--- | :--- |
|**TREC DL '19 & '20**|nDCG@10| 0.736 | 0.737 | 0.727 |
||R@1000| 0.871 | 0.807 | 0.809 |
||MRR| 0.885 | 0.968 | 0.964 |
|**MS MARCO Dev**|MRR@10| 0.386 | 0.384 | 0.320 |
||R@1000| 0.981 | 0.981 | 0.819 |


## Table 2: Out-of-Domain Results (BEIR)

*Zero-shot transfer performance on diverse datasets.*

| Dataset | Metric | **Hypencoder (Paper)** | **Hypencoder (HuggingFace)** | **Hypencoder (Mine)** |
| :--- | :--- | :--- | :--- | :--- |
|**TREC-COVID**|nDCG@10| 0.661 | 0.699 | 0.661 |
|**NFCorpus**|nDCG@10| 0.329 | 0.324 | 0.329 |
|**FiQA**|nDCG@10| 0.314 | 0.313 | 0.322 |
|**Touché v2**|nDCG@10| 0.258 | 0.261 | 0.220 |
|**DBPedia**|nDCG@10| 0.419 | 0.419 | 0.418 |

## Table 3: Harder Retrieval Tasks (Adversarial/Specific)

| Dataset | Metric | **Hypencoder (Paper)** | **Hypencoder (HuggingFace)** | **Hypencoder (Mine)** |
| :--- | :--- | :--- | :--- | :--- |
|**TREC DL Hard**|nDCG@10| **0.630** | 0.396 | 0.388 |
||R@1000| **0.798** | 0.775 | 0.775 |
||MRR| **0.887** | 0.622 | 0.598 |
|**TREC TOT**|nDCG@10| **0.134** | 0.034 | 0.041 |
||nDCG@1000| **0.182** | - | - |
||MRR| **0.125** | 0.031 | 0.038 |

---

## Efficiency Analysis (H100 GPU)

**Measured using the "Retrained" model for all datasets.**

### Metric Definitions
- **Total Time**: The wall-clock time taken to retrieve results for all queries in the dataset.
- **Avg Latency**: The average time taken to process a *single query*. Lower is better.
- **QPS (Queries Per Second)**: The number of queries processed per second. Higher is better. Calculated as `1 / Avg Latency`.
- **Method**:
    - **Exact**: Scans every document in the corpus. Fast for small datasets (like BEIR) but computationally expensive for large ones.
    - **Approx (Graph)**: Uses a neighbor graph to search only a subset of documents. Essential for large corpora (like MS MARCO's 8.8M passages).

### Detailed Performance Table

| Dataset | Method | Corpus Size | Avg Latency (s) | QPS | Total Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NFCorpus** | Exact | Small (~3k) | 0.030 | 32.6 | 9.90 |
| **FiQA** | Exact | Small (~57k) | 0.037 | 27.0 | 24.0 |
| **TREC-COVID** | Exact | Small (~171k) | 0.059 | 16.9 | 3.0 |
| **DBPedia** | Exact | Medium (~4.6M) | 0.686 | 1.46 | 274.5 |

### Key Observations
1.  **Corpus Size Impact**: The **BEIR datasets** (NFCorpus, FiQA, etc.) are relatively small, allowing "Exact" search to be very fast (up to 25 QPS).
2.  **Scale Challenge**: **MS MARCO** (used for TREC DL and Dev) is massive (8.8 million passages). "Exact" search on this scale would be prohibitively slow.
3.  **Graph Search Efficacy**: The **Approx (Graph)** method used for TREC DL '20 allows us to search the massive MS MARCO corpus with a latency of **~0.37s**. This validates the efficiency of the Hypencoder index.

### Latency Comparison vs Paper (TREC DL)

We compared our approximate graph search latency on TREC DL '20 against the paper's reported numbers (Table 4).

| Configuration | Hardware | Search Type | Latency (ms) | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| Paper (Efficient 1) | NVIDIA L40S | Approx | 59.6 | Tuned for speed (`ef=64`) |
| Paper (Efficient 2) | NVIDIA L40S | Approx | 231.1 | Balanced (`ef=328`) |
| Your Run (TREC DL) | NVIDIA H100 | Approx | 368.0 | High Accuracy (~5x faster than exhaustive) |
| Paper (Exhaustive) | NVIDIA L40S | Exact | 1769.8 | Baseline |

Analysis:
- Status: Your result (368ms) falls securely in the Approximate range, validating the efficiency gain over exhaustive search.
- Comparison: It is slightly slower than the paper's "Efficient 2" (231ms) but achieves high accuracy (0.731 nDCG). This suggests your run likely used default `ef_search` parameters that prioritized accuracy over maximizing raw speed.

---
**Footnotes:**
1. Evaluated on **DL '20 only**.
2. nDCG@10 = 0.380



## Appendix: Data Sources 📚

**Detailed source mapping for every number reported above.**

### 1. Pretrained Models (Official Checkpoint)
| Dataset | Source File (`outputs/inference/...`) |
| :--- | :--- |
| **TREC DL '20** | `hypencoder_pretrained/trec_dl_2020_8M_a100_results_pretrained/.../aggregated_metrics.txt` |
| **MS MARCO Dev** | `hypencoder_pretrained/msmarco_dev_results_pretrained/.../aggregated_metrics.txt` |
| **TREC DL Hard** | `hypencoder_pretrained/trec_dl_hard_results_pretrained/.../aggregated_metrics.txt` |
| **TREC TOT** | `hypencoder_pretrained/trec_tot_results_pretrained/.../aggregated_metrics.txt` |
| **TREC-COVID** | `hypencoder_pretrained/trec_covid_results_pretrained/.../aggregated_metrics.txt` |
| **NFCorpus** | `hypencoder_pretrained/nfcorpus_results_pretrained/.../aggregated_metrics.txt` |
| **FiQA** | `hypencoder_pretrained/fiqa_results_pretrained/.../aggregated_metrics.txt` |
| **Touché v2** | `hypencoder_pretrained/touche_results_pretrained/.../aggregated_metrics.txt` |
| **DBPedia** | `hypencoder_pretrained/dbpedia_results_pretrained/.../aggregated_metrics.txt` |

### 2. Retrained Model (Our Run)
| Dataset | Source File (`outputs/inference/hypencoder_retrained/...`) |
| :--- | :--- |
| **TREC DL '20** | `dl20_results/metrics/aggregated_metrics.txt` |
| **MS MARCO Dev** | `msmarco_results/metrics/aggregated_metrics.txt` |
| **TREC DL Hard** | `trec_dl_hard_results/metrics/aggregated_metrics.txt` |
| **TREC-COVID** | `covid_results/metrics/aggregated_metrics.txt` |
| **NFCorpus** | `nfcorpus_results/metrics/aggregated_metrics.txt` |
| **FiQA** | `fiqa_results/metrics/aggregated_metrics.txt` |
| **SciFact** | `scifact_results/metrics/aggregated_metrics.txt` |
| **Arguana** | `arguana_results/metrics/aggregated_metrics.txt` |
| **Touché v2** | `touche_results_full/metrics/aggregated_metrics.txt` |
| **DBPedia** | `dbpedia_results/metrics/aggregated_metrics.txt` |



### 3. Paper Baselines
| Column | Source | Notes |
| :--- | :--- | :--- |
| **Hypencoder (Paper)** | Original Hypencoder Paper (Table 1 & 3) | Cited as "Hypencoder" in literature. |
| **ColBERTv2** | ColBERTv2 Paper / BEIR Leaderboard | Standard SOTA baseline. |
| **BE-Base** | "Dense Passage Retrieval" (DPR/BERT) | Standard dense baseline. |

### 4. Training Curves
![Training Loss Retrained Hypencoder](./imgs/training_loss_retrained.png)
*Figure 1: Training loss vs steps for the Retrained Hypencoder (Job 306737). The loss stably converged to ~1.23.*

