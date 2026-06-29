# Full-context BERTopic Benchmark — `bertopic_full_context_v1`

## Purpose

One methodologically corrected BERTopic benchmark on the exact 75,121 NMF-matched modelling documents, using full-document multilingual embeddings from `BAAI/bge-m3`. This is a supporting comparison, not a replacement for the primary NMF K=24 model.

## Corrections from previous exploratory benchmark

| Issue | Previous (defective) | This benchmark |
|---|---|---|
| Embedding model | `paraphrase-multilingual-MiniLM-L12-v2` | `BAAI/bge-m3` |
| Token truncation | 128 tokens | None (full context) |
| Embedding input | `cleaned_text` | `modeling_text` (natural text) |
| UMAP computation | Once per config | Once shared across both configs |
| Config selected | Fallback (61.3% outliers > primary 59.4%) | Predeclared rule applied before examining results |
| Stopwords reported | 318 (bug in final run) | 315 (asserted P0=310, P1=315) |
| RAM reported | 0.0 GB (psutil bug) | Correctly reported |

## Predeclared selection rule

The primary configuration is the default.

Fallback may be selected **only** when **all** of the following hold:
1. Primary outlier fraction > 45%
2. Fallback reduces outlier fraction by ≥ 5 absolute percentage points
3. Fallback does not produce a degenerate topic-count distribution
4. Fallback does not materially worsen NPMI, diversity, or redundancy

## Configurations

**Primary HDBSCAN:** `min_cluster_size=100, min_samples=10`  
**Fallback HDBSCAN:** `min_cluster_size=100, min_samples=5`

Both use the same UMAP representation: `n_neighbors=15, n_components=5, metric=cosine, random_state=42`.

## Environment setup

```powershell
# Create isolated venv (Python 3.12, do NOT alter main project venv)
uv python install 3.12
uv venv .venv-bertopic-full --python 3.12
.\.venv-bertopic-full\Scripts\Activate.ps1

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install FlagEmbedding transformers huggingface-hub
pip install bertopic umap-learn hdbscan scikit-learn
pip install numpy scipy pandas psutil

# If HuggingFace downloads fail with SSL errors (Windows corporate cert issue):
# Set the environment variable BEFORE running (do not add to the script):
# $env:HF_HUB_DISABLE_SSL_VERIFICATION = "1"
```

## Running the benchmark

```powershell
.\.venv-bertopic-full\Scripts\Activate.ps1
python experiments/bertopic_full_context/run_benchmark.py
```

Optional flags:
- `--smoke-only`: Run smoke test only and stop
- `--skip-smoke`: Skip smoke test and go directly to full run

## Outputs

All generated outputs go to `outputs/bertopic_full_context_v1/` (gitignored).  
Large files (embeddings, UMAP arrays) are covered by `*.npy` gitignore rule.  
Review bundle: `outputs/bertopic_full_context_v1/bertopic_full_context_review.zip`

## What is NOT in scope

- Temporal BERTopic analysis
- Changing or refitting NMF
- LDA or any other model
- Modifying the modelling corpus
- Changing K=24 for NMF
- Hyperparameter search beyond primary/fallback configs

## Reproducibility

The corpus manifest (`corpus_manifest.json`) contains SHA-256 hashes of the ordered document-ID list and the ordered `modeling_text` content, along with the base Git commit, enabling exact corpus identity verification.
