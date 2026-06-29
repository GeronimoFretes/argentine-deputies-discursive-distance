"""
Full-context BERTopic benchmark — Argentine Chamber of Deputies 2008–2025
Experiment: bertopic_full_context_v1
Model: BAAI/bge-m3 (dense only, no truncation)
Corpus: 75,121 NMF-matched modelling documents (P1 stopword policy)

SSL note: If HuggingFace downloads fail with SSL errors on Windows, set the
environment variable HF_HUB_DISABLE_SSL_VERIFICATION=1 before running this
script. That workaround must NOT be enabled in code committed to the repo.

Usage:
    python experiments/bertopic_full_context/run_benchmark.py [--smoke-only] [--skip-smoke]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = Path(__file__).resolve().parent / "config_v1.json"
STOPWORDS_PATH = REPO_ROOT / "config" / "topic_modeling" / "stopwords_es_p0_v1.txt"
SRC_PATH = REPO_ROOT / "src"

# Add src to path so we can import project modules
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

with CONFIG_PATH.open(encoding="utf-8") as _fh:
    CFG = json.load(_fh)

HANDOFF_DIR = REPO_ROOT / CFG["corpus"]["handoff_dir"]
OUT_DIR = REPO_ROOT / CFG["output_dir"]
REVIEW_DIR = OUT_DIR / "review"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "run_log.txt"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, obj: Any) -> None:
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    part.replace(path)


def write_npy_atomic(path: Path, arr: np.ndarray) -> None:
    # np.save() adds .npy extension if missing, so we name the part without .npy
    part = path.parent / (path.stem + ".part.npy")
    np.save(part, arr)
    part.replace(path)


# ---------------------------------------------------------------------------
# 1. Import project preprocessing
# ---------------------------------------------------------------------------
def load_project_stopwords():
    from argentine_deputies_discursive_distance.topic_preprocessing import (
        load_stopwords,
        clean_natural_text,
        lexical_tokens,
        P1_ADDITIONS,
    )
    sw = load_stopwords(STOPWORDS_PATH, variant="P1")
    assert sw.p0_count == CFG["stopwords"]["expected_p0_count"], (
        f"P0 count mismatch: {sw.p0_count} != {CFG['stopwords']['expected_p0_count']}"
    )
    assert sw.p1_count == CFG["stopwords"]["expected_p1_count"], (
        f"P1 count mismatch: {sw.p1_count} != {CFG['stopwords']['expected_p1_count']}"
    )
    expected_additions = set(CFG["stopwords"]["expected_p1_additions"])
    assert set(P1_ADDITIONS) == expected_additions, (
        f"P1 additions mismatch: {set(P1_ADDITIONS)} != {expected_additions}"
    )
    log.info(
        f"Stopwords: P0={sw.p0_count} P1={sw.p1_count} "
        f"additions={sorted(P1_ADDITIONS)}"
    )
    return sw, clean_natural_text, lexical_tokens


# ---------------------------------------------------------------------------
# 2. Corpus loading
# ---------------------------------------------------------------------------
def load_corpus() -> tuple[list[dict], set[str]]:
    log.info("Loading corpus...")
    zero_path = HANDOFF_DIR / CFG["corpus"]["zero_tfidf_file"]
    if not zero_path.is_file():
        raise FileNotFoundError(f"Zero-TF-IDF ledger missing: {zero_path}")

    zero_ids: set[str] = set()
    for line in zero_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            zero_ids.add(json.loads(line)["document_id"])

    expected_excluded = set(CFG["corpus"]["expected_excluded_ids"])
    assert zero_ids == expected_excluded, (
        f"Excluded IDs mismatch.\n  Got: {sorted(zero_ids)}\n  Expected: {sorted(expected_excluded)}"
    )
    log.info(f"Zero-TF-IDF exclusions confirmed: {sorted(zero_ids)}")

    docs_path = HANDOFF_DIR / CFG["corpus"]["cleaned_documents_file"]
    if not docs_path.is_file():
        raise FileNotFoundError(f"Cleaned documents missing: {docs_path}")

    docs_raw: list[dict] = []
    for line in docs_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if not r.get("modeling_text"):
            raise ValueError(f"Empty modeling_text for doc: {r.get('document_id')}")
        docs_raw.append(r)

    n_input = len(docs_raw)
    assert n_input == CFG["corpus"]["expected_input_count"], (
        f"Input count mismatch: {n_input} != {CFG['corpus']['expected_input_count']}"
    )
    log.info(f"Input records: {n_input}")

    docs = [r for r in docs_raw if r["document_id"] not in zero_ids]
    n_bench = len(docs)
    assert n_bench == CFG["corpus"]["expected_benchmark_count"], (
        f"Benchmark count mismatch: {n_bench} != {CFG['corpus']['expected_benchmark_count']}"
    )

    doc_ids = [r["document_id"] for r in docs]
    assert len(doc_ids) == len(set(doc_ids)), "Duplicate document IDs detected"
    assert all(mt for mt in [r["modeling_text"] for r in docs]), "Empty modeling_text records"

    log.info(f"Benchmark records: {n_bench} | duplicates=0 | empty_texts=0")
    return docs, zero_ids


# ---------------------------------------------------------------------------
# 3. Corpus manifest
# ---------------------------------------------------------------------------
def build_corpus_manifest(docs: list[dict], zero_ids: set[str], base_commit: str) -> dict:
    doc_ids = [r["document_id"] for r in docs]
    id_bytes = "\n".join(doc_ids).encode("utf-8")
    id_hash = sha256_bytes(id_bytes)

    texts = [r["modeling_text"] for r in docs]
    content_bytes = "\n".join(texts).encode("utf-8")
    content_hash = sha256_bytes(content_bytes)

    years = sorted({r["year"] for r in docs})
    session_cats = sorted({r.get("session_category", "") for r in docs} - {""})

    return {
        "base_git_commit": base_commit,
        "handoff_dir": str(HANDOFF_DIR),
        "document_count": len(docs),
        "excluded_ids": sorted(zero_ids),
        "year_range": [min(years), max(years)],
        "session_categories": session_cats,
        "ordered_document_id_sha256": id_hash,
        "ordered_content_sha256": content_hash,
        "created_at": now_iso(),
    }


# ---------------------------------------------------------------------------
# 4. Build P1 vectorizer
# ---------------------------------------------------------------------------
def build_p1_vectorizer(sw, clean_natural_text, lexical_tokens):
    from sklearn.feature_extraction.text import CountVectorizer

    stopword_list = sorted(sw.words)

    def p1_tokenizer(text: str) -> list[str]:
        cleaned = clean_natural_text(text).cleaned_text
        return lexical_tokens(cleaned)

    min_df = CFG["vectorizer"]["min_df"]
    vectorizer = CountVectorizer(
        tokenizer=p1_tokenizer,
        stop_words=stopword_list,
        ngram_range=tuple(CFG["vectorizer"]["ngram_range"]),
        min_df=min_df,
        lowercase=False,
    )
    log.info(f"P1 vectorizer built: {len(stopword_list)} stopwords, min_df={min_df}")
    return vectorizer


# ---------------------------------------------------------------------------
# 5. Environment inspection
# ---------------------------------------------------------------------------
def inspect_environment() -> dict:
    import platform
    import psutil
    import torch

    ram_gb = psutil.virtual_memory().total / (1024**3)
    gpu_name = "N/A"
    gpu_vram_mb = 0
    cuda_version = "N/A"

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        cuda_version = torch.version.cuda or "N/A"

    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(ram_gb, 2),
        "gpu_name": gpu_name,
        "gpu_vram_mb": gpu_vram_mb,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": cuda_version,
        "torch_version": torch.__version__,
    }

    try:
        import bertopic
        env["bertopic_version"] = bertopic.__version__
    except Exception:
        env["bertopic_version"] = "N/A"

    try:
        import umap
        env["umap_version"] = umap.__version__
    except Exception:
        env["umap_version"] = "N/A"

    try:
        import hdbscan
        env["hdbscan_version"] = getattr(hdbscan, "__version__", "N/A")
    except Exception:
        env["hdbscan_version"] = "N/A"

    try:
        import sklearn
        env["sklearn_version"] = sklearn.__version__
    except Exception:
        env["sklearn_version"] = "N/A"

    env["embedding_backend"] = "transformers_automodel_cls_pooling"
    env["flagembedding_version"] = "not_used_openssl_conflict"

    try:
        import transformers
        env["transformers_version"] = transformers.__version__
    except Exception:
        env["transformers_version"] = "N/A"

    try:
        import numpy as np_
        env["numpy_version"] = np_.__version__
    except Exception:
        env["numpy_version"] = "N/A"

    try:
        import scipy
        env["scipy_version"] = scipy.__version__
    except Exception:
        env["scipy_version"] = "N/A"

    try:
        import pandas as pd_
        env["pandas_version"] = pd_.__version__
    except Exception:
        env["pandas_version"] = "N/A"

    log.info(
        f"Environment: {gpu_name} | CUDA {cuda_version} | "
        f"torch {torch.__version__} | RAM {ram_gb:.1f} GB"
    )
    return env


# ---------------------------------------------------------------------------
# 6. Token length audit
# ---------------------------------------------------------------------------
BGE_M3_CACHE = None  # resolved at runtime


def resolve_model_path() -> str:
    """Return local model path from HF cache."""
    import glob
    cache_base = Path.home() / ".cache" / "huggingface" / "hub"
    patterns = [
        str(cache_base / "models--BAAI--bge-m3" / "snapshots" / "*"),
    ]
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[-1]  # most recent snapshot
    raise FileNotFoundError(
        "BAAI/bge-m3 not found in HF cache. "
        "Run the model download step first:\n"
        "  python experiments/bertopic_full_context/download_model.py"
    )


def run_token_audit(docs: list[dict]) -> dict:
    log.info("=== TOKEN LENGTH AUDIT ===")
    from transformers import AutoTokenizer

    model_path = resolve_model_path()
    log.info(f"Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tok_name = tokenizer.name_or_path

    model_max = 8192  # BGE-M3 documented max context
    log.info(f"BGE-M3 supported context: {model_max} tokens")

    log.info(f"Tokenizer: {tok_name} | model_max_length: {model_max}")

    texts = [r["modeling_text"] for r in docs]
    lengths: list[int] = []

    batch_size = 512
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, add_special_tokens=True, truncation=False)
        for ids in enc["input_ids"]:
            lengths.append(len(ids))

    lengths_arr = np.array(lengths, dtype=np.int64)

    above_128 = int((lengths_arr > 128).sum())
    above_256 = int((lengths_arr > 256).sum())
    above_512 = int((lengths_arr > 512).sum())
    above_1024 = int((lengths_arr > 1024).sum())
    above_2048 = int((lengths_arr > 2048).sum())
    above_4096 = int((lengths_arr > 4096).sum())
    above_8192 = int((lengths_arr > 8192).sum())
    obs_max = int(lengths_arr.max())

    assert obs_max <= model_max, (
        f"ANOMALY: max token count {obs_max} exceeds model context {model_max}. "
        f"STOP — do not truncate."
    )

    safe_max_length = min(max(int(np.percentile(lengths_arr, 100)), 64), model_max)

    audit = {
        "tokenizer_model": tok_name,
        "tokenizer_revision": str(model_path),
        "model_max_length": model_max,
        "configured_max_length": safe_max_length,
        "special_tokens_included": True,
        "document_count": len(lengths),
        "min": int(lengths_arr.min()),
        "p25": int(np.percentile(lengths_arr, 25)),
        "p50": int(np.percentile(lengths_arr, 50)),
        "p75": int(np.percentile(lengths_arr, 75)),
        "p90": int(np.percentile(lengths_arr, 90)),
        "p95": int(np.percentile(lengths_arr, 95)),
        "p99": int(np.percentile(lengths_arr, 99)),
        "max": obs_max,
        "above_128": above_128,
        "above_256": above_256,
        "above_512": above_512,
        "above_1024": above_1024,
        "above_2048": above_2048,
        "above_4096": above_4096,
        "above_8192": above_8192,
        "truncated_documents": 0,
        "created_at": now_iso(),
    }

    log.info(
        f"Token audit: min={audit['min']} p50={audit['p50']} "
        f"p95={audit['p95']} p99={audit['p99']} max={obs_max} "
        f"above_128={above_128} above_512={above_512}"
    )

    write_json_atomic(OUT_DIR / "token_length_audit.json", audit)
    dist_path = OUT_DIR / "token_length_distribution.csv"
    with dist_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["document_id", "token_count"])
        for r, tl in zip(docs, lengths):
            writer.writerow([r["document_id"], tl])

    log.info(f"Token audit saved. safe_max_length={safe_max_length}")
    return audit, safe_max_length


# ---------------------------------------------------------------------------
# 7. Embedding with BGE-M3 (via transformers AutoModel)
# Note: FlagEmbedding is not used because its aiohttp dependency causes a
# fatal OPENSSL_Applink DLL conflict on Windows with Python 3.12. We use
# transformers AutoModel directly, which is functionally equivalent for
# dense CLS-token embeddings. BGE-M3 uses CLS pooling for dense retrieval.
# ---------------------------------------------------------------------------
def load_bge_m3(max_length: int):
    import torch
    from transformers import AutoTokenizer, AutoModel

    model_path = resolve_model_path()
    log.info(f"Loading BGE-M3 from: {model_path} | max_length={max_length}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(
        model_path,
        dtype=torch.float16,
        local_files_only=True,
    ).cuda()
    model.eval()

    # Record model info
    model_revision = Path(model_path).name
    log.info(
        f"BGE-M3 loaded | device=cuda | revision={model_revision} | "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )
    return model, tokenizer, model_path


def embed_texts(
    model_tuple,
    texts: list[str],
    max_length: int,
    label: str = "embed",
):
    import torch
    import gc

    model, tokenizer, model_path = model_tuple
    batch_size = CFG["embedding"]["initial_batch_size"]
    min_batch = CFG["embedding"]["min_batch_size"]
    all_embeddings: list[np.ndarray] = []
    i = 0
    t0 = time.time()
    n = len(texts)
    final_batch_size = batch_size

    while i < n:
        batch = texts[i : i + batch_size]
        try:
            # Encode without truncation to check lengths, then encode again with padding for GPU
            no_trunc_enc = tokenizer(batch, add_special_tokens=True, truncation=False)
            max_in_batch = max(len(ids) for ids in no_trunc_enc["input_ids"])
            if max_in_batch > max_length:
                raise AssertionError(
                    f"TRUNCATION WOULD OCCUR: batch max tokens {max_in_batch} > max_length {max_length}"
                )
            enc = tokenizer(
                batch,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )

            enc = {k: v.cuda() for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc)
                # CLS token (BGE-M3 uses CLS for dense embedding)
                embs = out.last_hidden_state[:, 0, :].float().cpu().numpy()

            all_embeddings.append(embs.astype(np.float32))
            final_batch_size = batch_size
            if (i // batch_size) % 50 == 0:
                elapsed = time.time() - t0
                pct = 100 * (i + len(batch)) / n
                log.info(f"[{label}] {i+len(batch)}/{n} ({pct:.1f}%) in {elapsed:.0f}s bs={batch_size}")
            i += batch_size
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and batch_size > min_batch:
                torch.cuda.empty_cache()
                gc.collect()
                batch_size = max(min_batch, batch_size // 2)
                log.warning(f"[{label}] OOM — reducing batch_size to {batch_size}")
            else:
                raise

    result = np.vstack(all_embeddings)
    # L2 normalize
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    result = result / norms

    elapsed = time.time() - t0
    log.info(f"[{label}] done: shape={result.shape} time={elapsed:.1f}s batch_size={final_batch_size}")
    return result, elapsed, final_batch_size


# ---------------------------------------------------------------------------
# 8. Smoke test
# ---------------------------------------------------------------------------
def run_smoke_test(docs: list[dict], model, max_length: int, sw, vectorizer) -> dict:
    log.info("=== SMOKE TEST ===")
    import random
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN

    rng = random.Random(CFG["smoke_test"]["random_state"])
    years_needed = CFG["smoke_test"]["years"]
    target_size = CFG["smoke_test"]["sample_size_approx"]

    by_year: dict[int, list[int]] = {}
    for idx, r in enumerate(docs):
        by_year.setdefault(r["year"], []).append(idx)

    sample_indices: list[int] = []
    for yr in years_needed:
        if yr in by_year:
            pool = by_year[yr]
            k = max(1, target_size // len(years_needed))
            chosen = rng.sample(pool, min(k, len(pool)))
            sample_indices.extend(chosen)

    sample_indices = sorted(set(sample_indices))
    sample_docs = [docs[i] for i in sample_indices]
    sample_texts = [r["modeling_text"] for r in sample_docs]
    sample_lexical = [r["cleaned_text"] for r in sample_docs]
    years_covered = sorted({r["year"] for r in sample_docs})

    log.info(f"Smoke sample: {len(sample_docs)} docs across {len(years_covered)} years")

    embs, smoke_embed_s, batch_used = embed_texts(
        model, sample_texts, max_length, label="smoke"
    )

    assert embs.shape == (len(sample_docs), embs.shape[1]), "Embedding shape mismatch"
    assert np.isfinite(embs).all(), "NaN/Inf in smoke embeddings"
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4), "Embeddings not normalized"
    log.info(f"Smoke embeddings OK: {embs.shape} in {smoke_embed_s:.1f}s")

    umap_cfg = CFG["umap"]
    umap_model = UMAP(
        n_neighbors=umap_cfg["n_neighbors"],
        n_components=umap_cfg["n_components"],
        min_dist=umap_cfg["min_dist"],
        metric=umap_cfg["metric"],
        random_state=umap_cfg["random_state"],
    )
    hdbscan_cfg = CFG["hdbscan_primary"]
    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_cfg["min_cluster_size"],
        min_samples=hdbscan_cfg["min_samples"],
        metric=hdbscan_cfg["metric"],
        cluster_selection_method=hdbscan_cfg["cluster_selection_method"],
    )

    t0 = time.time()
    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        language="multilingual",
        calculate_probabilities=False,
    )
    topics, _ = topic_model.fit_transform(sample_lexical, embeddings=embs)
    smoke_fit_s = time.time() - t0

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(1 for t in topics if t == -1)
    outlier_frac = n_outliers / len(topics)
    log.info(
        f"Smoke: {n_topics} topics | {outlier_frac:.1%} outliers | fit={smoke_fit_s:.1f}s"
    )

    smoke_metrics = {
        "sample_size": len(sample_docs),
        "years_covered": len(years_covered),
        "embedding_shape": list(embs.shape),
        "embedding_dim": embs.shape[1],
        "batch_size_used": batch_used,
        "embed_time_s": round(smoke_embed_s, 1),
        "fit_time_s": round(smoke_fit_s, 1),
        "topic_count": n_topics,
        "outlier_count": n_outliers,
        "outlier_fraction": round(outlier_frac, 4),
        "cuda_used": str(next(iter(["cuda" if "cuda" in str(model) else "cpu"]), "unknown")),
        "nan_count": int(~np.isfinite(embs).all()),
        "normalized": True,
        "created_at": now_iso(),
    }

    smoke_manifest = {
        "sample_indices_count": len(sample_indices),
        "years_covered": years_covered,
        "embedding_dim": embs.shape[1],
        "max_length_used": max_length,
        "truncated": 0,
        "created_at": now_iso(),
    }

    write_json_atomic(OUT_DIR / "smoke_metrics.json", smoke_metrics)
    write_json_atomic(OUT_DIR / "smoke_manifest.json", smoke_manifest)

    log.info("Smoke test PASSED. Proceeding to full corpus.")
    return smoke_metrics


# ---------------------------------------------------------------------------
# 9. Full embeddings
# ---------------------------------------------------------------------------
def generate_full_embeddings(docs: list[dict], model_tuple, max_length: int, env: dict) -> tuple:
    log.info("=== FULL EMBEDDINGS ===")
    emb_path = OUT_DIR / "embeddings.npy"
    id_path = OUT_DIR / "document_ids.txt"

    texts = [r["modeling_text"] for r in docs]
    doc_ids = [r["document_id"] for r in docs]
    _, _, model_path = model_tuple

    embs, elapsed, batch_used = embed_texts(model_tuple, texts, max_length, label="full")

    assert embs.shape[0] == len(docs), f"Row count mismatch: {embs.shape[0]} != {len(docs)}"
    assert np.isfinite(embs).all(), "NaN/Inf in full embeddings"
    norms = np.linalg.norm(embs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), "Full embeddings not normalized"

    write_npy_atomic(emb_path, embs)
    id_path.write_text("\n".join(doc_ids), encoding="utf-8")

    emb_sha = sha256_file(emb_path)
    emb_size = emb_path.stat().st_size

    manifest = {
        "model_id": CFG["embedding"]["model_id"],
        "model_snapshot": str(model_path),
        "model_revision": Path(model_path).name,
        "pooling_strategy": "CLS_token",
        "embedding_backend": "transformers_AutoModel",
        "embedding_shape": list(embs.shape),
        "dtype": str(embs.dtype),
        "byte_size": emb_size,
        "sha256": emb_sha,
        "max_length": max_length,
        "batch_size_final": batch_used,
        "device": CFG["embedding"]["device"],
        "normalize": True,
        "dense_only": True,
        "truncated_documents": 0,
        "runtime_s": round(elapsed, 1),
        "nan_count": 0,
        "infinite_count": 0,
        "mean_norm": float(norms.mean()),
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "document_count": len(docs),
        "created_at": now_iso(),
    }

    write_json_atomic(OUT_DIR / "embedding_manifest.json", manifest)
    log.info(
        f"Embeddings saved: {embs.shape} dtype={embs.dtype} "
        f"time={elapsed:.1f}s sha256={emb_sha[:16]}..."
    )
    return embs, doc_ids, manifest


# ---------------------------------------------------------------------------
# 10. Compute UMAP once
# ---------------------------------------------------------------------------
def compute_umap(embs: np.ndarray) -> np.ndarray:
    log.info("=== UMAP REDUCTION ===")
    from umap import UMAP

    umap_cfg = CFG["umap"]
    umap_model = UMAP(
        n_neighbors=umap_cfg["n_neighbors"],
        n_components=umap_cfg["n_components"],
        min_dist=umap_cfg["min_dist"],
        metric=umap_cfg["metric"],
        random_state=umap_cfg["random_state"],
    )
    t0 = time.time()
    umap_embs = umap_model.fit_transform(embs)
    elapsed = time.time() - t0

    assert umap_embs.shape == (embs.shape[0], umap_cfg["n_components"]), (
        f"UMAP shape mismatch: {umap_embs.shape}"
    )
    assert np.isfinite(umap_embs).all(), "NaN/Inf in UMAP embeddings"

    umap_path = OUT_DIR / "umap_embeddings.npy"
    write_npy_atomic(umap_path, umap_embs.astype(np.float32))
    umap_sha = sha256_file(umap_path)

    manifest = {
        "shape": list(umap_embs.shape),
        "dtype": "float32",
        "byte_size": umap_path.stat().st_size,
        "sha256": umap_sha,
        "random_state": umap_cfg["random_state"],
        "n_neighbors": umap_cfg["n_neighbors"],
        "n_components": umap_cfg["n_components"],
        "min_dist": umap_cfg["min_dist"],
        "metric": umap_cfg["metric"],
        "runtime_s": round(elapsed, 1),
        "nan_count": 0,
        "infinite_count": 0,
        "created_at": now_iso(),
    }
    write_json_atomic(OUT_DIR / "umap_manifest.json", manifest)
    log.info(f"UMAP done: {umap_embs.shape} in {elapsed:.1f}s")
    return umap_embs


# ---------------------------------------------------------------------------
# 11. Coherence metrics helpers
# ---------------------------------------------------------------------------
def compute_npmi(topic_words_list: list[list[str]], corpus_texts: list[str]) -> dict:
    """Compute per-topic NPMI for top-10 words using co-occurrence counts."""
    from sklearn.feature_extraction.text import CountVectorizer
    import scipy.sparse as sp

    all_words = sorted({w for tw in topic_words_list for w in tw[:10]})
    if not all_words:
        return {"mean": 0.0, "median": 0.0, "min": 0.0}

    cv = CountVectorizer(vocabulary=all_words, binary=True, analyzer="word",
                         token_pattern=r"(?u)\b\w+\b")
    try:
        X = cv.fit_transform(corpus_texts)
    except Exception:
        return {"mean": 0.0, "median": 0.0, "min": 0.0}

    n_docs = X.shape[0]
    vocab = cv.vocabulary_
    npmis: list[float] = []

    for tw in topic_words_list:
        words = [w for w in tw[:10] if w in vocab]
        topic_npmis: list[float] = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                wi, wj = vocab[words[i]], vocab[words[j]]
                pi = X[:, wi].nnz / n_docs
                pj = X[:, wj].nnz / n_docs
                pij_col = X[:, wi].multiply(X[:, wj])
                pij = pij_col.nnz / n_docs
                if pij < 1e-9 or pi < 1e-9 or pj < 1e-9:
                    continue
                import math
                npmi = math.log(pij / (pi * pj)) / (-math.log(pij))
                topic_npmis.append(npmi)
        if topic_npmis:
            npmis.append(float(np.mean(topic_npmis)))

    if not npmis:
        return {"mean": 0.0, "median": 0.0, "min": 0.0}

    return {
        "mean": round(float(np.mean(npmis)), 4),
        "median": round(float(np.median(npmis)), 4),
        "min": round(float(np.min(npmis)), 4),
    }


def compute_diversity(topic_words_list: list[list[str]], top_n: int = 10) -> float:
    all_words = [w for tw in topic_words_list for w in tw[:top_n]]
    if not all_words:
        return 0.0
    return round(len(set(all_words)) / len(all_words), 4)


def compute_exclusivity_redundancy(topic_words_list: list[list[str]], top_n: int = 10):
    from collections import Counter
    if not topic_words_list:
        return 0.0, 0.0, 0.0

    word_freq: Counter = Counter()
    for tw in topic_words_list:
        for w in tw[:top_n]:
            word_freq[w] += 1

    n_topics = len(topic_words_list)
    exclusivities: list[float] = []
    for tw in topic_words_list:
        words = tw[:top_n]
        if not words:
            continue
        excl = sum(1.0 / word_freq[w] for w in words) / len(words)
        exclusivities.append(excl)

    mean_excl = round(float(np.mean(exclusivities)) if exclusivities else 0.0, 4)

    # pairwise cosine redundancy via word overlap
    def bow(words):
        v = {}
        for w in words:
            v[w] = v.get(w, 0) + 1
        return v

    redundancies: list[float] = []
    for i in range(n_topics):
        for j in range(i + 1, n_topics):
            wi = set(topic_words_list[i][:top_n])
            wj = set(topic_words_list[j][:top_n])
            overlap = len(wi & wj)
            denom = min(len(wi), len(wj))
            red = overlap / denom if denom > 0 else 0.0
            redundancies.append(red)

    mean_red = round(float(np.mean(redundancies)) if redundancies else 0.0, 4)
    max_red = round(float(np.max(redundancies)) if redundancies else 0.0, 4)
    return mean_excl, mean_red, max_red


def compute_entropy(sizes: list[int]) -> float:
    import math
    total = sum(sizes)
    if total == 0:
        return 0.0
    probs = [s / total for s in sizes]
    max_ent = math.log(len(sizes)) if len(sizes) > 1 else 1.0
    ent = -sum(p * math.log(p) for p in probs if p > 0)
    return round(ent / max_ent if max_ent > 0 else 0.0, 4)


def compute_top_share(sizes: list[int], top_n: int) -> float:
    total = sum(sizes)
    if total == 0:
        return 0.0
    return round(sum(sorted(sizes, reverse=True)[:top_n]) / total, 4)


# ---------------------------------------------------------------------------
# 12. Run HDBSCAN config
# ---------------------------------------------------------------------------
def run_hdbscan_config(
    config_name: str,
    hdbscan_params: dict,
    umap_embs: np.ndarray,
    embs: np.ndarray,
    docs: list[dict],
    vectorizer,
) -> tuple[dict, object, list[int]]:
    log.info(f"=== HDBSCAN: {config_name} ===")
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    umap_cfg = CFG["umap"]

    class PrecomputedUMAP:
        """Wrapper that returns pre-computed UMAP embeddings."""
        def __init__(self, umap_data):
            self.umap_data = umap_data
            self.embedding_ = umap_data

        def fit(self, X, **kwargs):
            return self

        def transform(self, X, **kwargs):
            return self.umap_data

        def fit_transform(self, X, **kwargs):
            return self.umap_data

    hdbscan_model = HDBSCAN(
        min_cluster_size=hdbscan_params["min_cluster_size"],
        min_samples=hdbscan_params["min_samples"],
        metric=hdbscan_params["metric"],
        cluster_selection_method=hdbscan_params["cluster_selection_method"],
        gen_min_span_tree=False,
    )

    umap_wrapper = PrecomputedUMAP(umap_embs)

    lexical_texts = [r["cleaned_text"] for r in docs]

    t0 = time.time()
    topic_model = BERTopic(
        umap_model=umap_wrapper,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        language="multilingual",
        calculate_probabilities=False,
        nr_topics=None,
    )
    topics, _ = topic_model.fit_transform(lexical_texts, embeddings=embs)
    elapsed = time.time() - t0

    topic_set = set(topics)
    n_topics = len(topic_set) - (1 if -1 in topic_set else 0)
    n_outliers = sum(1 for t in topics if t == -1)
    n_assigned = len(topics) - n_outliers
    outlier_frac = n_outliers / len(topics)

    topic_info = topic_model.get_topic_info()
    non_outlier = topic_info[topic_info["Topic"] != -1]
    sizes = list(non_outlier["Count"])

    topic_words_list = []
    for tid in non_outlier["Topic"]:
        tw = topic_model.get_topic(tid)
        if tw:
            topic_words_list.append([w for w, _ in tw[:10]])
        else:
            topic_words_list.append([])

    npmi = compute_npmi(topic_words_list, lexical_texts)
    diversity = compute_diversity(topic_words_list)
    mean_excl, mean_red, max_red = compute_exclusivity_redundancy(topic_words_list)
    entropy = compute_entropy(sizes)
    top1_share = compute_top_share(sizes, 1)
    top5_share = compute_top_share(sizes, 5)

    metrics = {
        "config_name": config_name,
        "hdbscan_params": hdbscan_params,
        "topic_count_excl_outlier": n_topics,
        "total_documents": len(topics),
        "assigned_documents": n_assigned,
        "outlier_count": n_outliers,
        "outlier_fraction": round(outlier_frac, 4),
        "topic_size_min": min(sizes) if sizes else 0,
        "topic_size_median": int(np.median(sizes)) if sizes else 0,
        "topic_size_max": max(sizes) if sizes else 0,
        "normalized_entropy": entropy,
        "top1_share": top1_share,
        "top5_share": top5_share,
        "npmi_mean": npmi["mean"],
        "npmi_median": npmi["median"],
        "npmi_min": npmi["min"],
        "diversity": diversity,
        "mean_exclusivity": mean_excl,
        "mean_redundancy": mean_red,
        "max_redundancy": max_red,
        "runtime_s": round(elapsed, 1),
        "umap_random_state": umap_cfg["random_state"],
        "warnings": [],
        "created_at": now_iso(),
    }

    log.info(
        f"[{config_name}] topics={n_topics} outliers={n_outliers} "
        f"({outlier_frac:.1%}) NPMI={npmi['mean']:.3f} div={diversity:.3f} "
        f"excl={mean_excl:.3f} red_mean={mean_red:.3f} time={elapsed:.1f}s"
    )

    # Save outputs
    prefix = config_name.lower().replace(" ", "_")
    write_json_atomic(OUT_DIR / f"{prefix}_metrics.json", metrics)

    topic_info.to_csv(OUT_DIR / f"{prefix}_topic_info.csv", index=False)

    # Topic terms
    terms_rows = []
    for tid in non_outlier["Topic"]:
        tw = topic_model.get_topic(tid)
        if tw:
            for rank, (word, score) in enumerate(tw[:20]):
                terms_rows.append({
                    "topic_id": tid,
                    "rank": rank,
                    "word": word,
                    "score": round(score, 6),
                })
    pd.DataFrame(terms_rows).to_csv(OUT_DIR / f"{prefix}_topic_terms.csv", index=False)

    # Document assignments
    assign_rows = []
    for r, t in zip(docs, topics):
        assign_rows.append({
            "document_id": r["document_id"],
            "topic_id": t,
            "year": r["year"],
            "session_category": r.get("session_category", ""),
            "speaker_family": r.get("speaker_family", ""),
            "source_record_id": r.get("source_record_id", ""),
            "turn_index": r.get("turn_index", ""),
            "chunk_index": r.get("chunk_index", 0),
            "word_count": r.get("word_count", 0),
        })
    pd.DataFrame(assign_rows).to_csv(OUT_DIR / f"{prefix}_document_assignments.csv", index=False)

    return metrics, topic_model, topics


# ---------------------------------------------------------------------------
# 13. Selection
# ---------------------------------------------------------------------------
def select_configuration(primary_metrics: dict, fallback_metrics: dict) -> dict:
    rule = CFG["selection_rule"]
    pof = primary_metrics["outlier_fraction"]
    fof = fallback_metrics["outlier_fraction"]
    improvement_pp = (pof - fof) * 100

    conditions_met = [
        pof > rule["fallback_conditions"]["primary_outlier_fraction_above"],
        improvement_pp >= rule["fallback_conditions"]["fallback_outlier_improvement_min_pp"],
    ]
    all_conditions = all(conditions_met)

    if all_conditions:
        selected = "fallback"
        reason = (
            f"Primary outlier fraction {pof:.1%} > 45% threshold AND "
            f"fallback improves by {improvement_pp:.1f}pp (>= 5pp minimum)"
        )
    else:
        selected = "primary"
        if pof <= rule["fallback_conditions"]["primary_outlier_fraction_above"]:
            reason = f"Primary outlier fraction {pof:.1%} <= 45% threshold; primary retained"
        else:
            reason = (
                f"Primary outlier fraction {pof:.1%} > 45% but fallback only improves "
                f"by {improvement_pp:.1f}pp (< 5pp minimum); primary retained"
            )

    decision = {
        "selected": selected,
        "rule_applied": "predeclared_v1",
        "primary_outlier_fraction": pof,
        "fallback_outlier_fraction": fof,
        "improvement_pp": round(improvement_pp, 2),
        "threshold_45pct": pof > 0.45,
        "threshold_5pp_improvement": improvement_pp >= 5.0,
        "reason": reason,
        "created_at": now_iso(),
    }

    log.info(f"Selection: {selected.upper()} | {reason}")
    write_json_atomic(OUT_DIR / "selection_decision.json", decision)
    return decision


# ---------------------------------------------------------------------------
# 14. Copy selected native outputs
# ---------------------------------------------------------------------------
def copy_selected_native(selected_name: str) -> None:
    prefix = selected_name
    for suffix in ["_metrics.json", "_topic_info.csv", "_topic_terms.csv",
                   "_document_assignments.csv"]:
        src = OUT_DIR / f"{prefix}{suffix}"
        dst = OUT_DIR / f"selected_native{suffix}"
        if src.is_file():
            import shutil
            shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# 15. Reduce to ~24 topics
# ---------------------------------------------------------------------------
def reduce_to_24(
    selected_model,
    docs: list[dict],
    selected_topics: list[int],
    embs: np.ndarray,
    vectorizer,
) -> dict:
    log.info("=== REDUCTION TO 24 TOPICS ===")
    nr_topics = CFG["reduction"]["nr_topics"]

    lexical_texts = [r["cleaned_text"] for r in docs]

    t0 = time.time()
    selected_model.reduce_topics(lexical_texts, nr_topics=nr_topics)
    reduced_topics = selected_model.topics_
    elapsed = time.time() - t0

    topic_set = set(reduced_topics)
    n_topics = len(topic_set) - (1 if -1 in topic_set else 0)
    n_outliers = sum(1 for t in reduced_topics if t == -1)
    n_assigned = len(reduced_topics) - n_outliers
    outlier_frac = n_outliers / len(reduced_topics)

    log.info(
        f"Reduced: requested={nr_topics} actual={n_topics} "
        f"outliers={n_outliers} ({outlier_frac:.1%}) time={elapsed:.1f}s"
    )

    topic_info = selected_model.get_topic_info()
    non_outlier = topic_info[topic_info["Topic"] != -1]
    sizes = list(non_outlier["Count"])

    topic_words_list = []
    for tid in non_outlier["Topic"]:
        tw = selected_model.get_topic(tid)
        if tw:
            topic_words_list.append([w for w, _ in tw[:10]])
        else:
            topic_words_list.append([])

    npmi = compute_npmi(topic_words_list, lexical_texts)
    diversity = compute_diversity(topic_words_list)
    mean_excl, mean_red, max_red = compute_exclusivity_redundancy(topic_words_list)
    entropy = compute_entropy(sizes)
    top5_share = compute_top_share(sizes, 5)

    metrics = {
        "requested_topics": nr_topics,
        "actual_topic_count": n_topics,
        "total_documents": len(reduced_topics),
        "assigned_documents": n_assigned,
        "outlier_count": n_outliers,
        "outlier_fraction": round(outlier_frac, 4),
        "topic_size_min": min(sizes) if sizes else 0,
        "topic_size_median": int(np.median(sizes)) if sizes else 0,
        "topic_size_max": max(sizes) if sizes else 0,
        "normalized_entropy": entropy,
        "top5_share": top5_share,
        "npmi_mean": npmi["mean"],
        "npmi_median": npmi["median"],
        "npmi_min": npmi["min"],
        "diversity": diversity,
        "mean_exclusivity": mean_excl,
        "mean_redundancy": mean_red,
        "max_redundancy": max_red,
        "runtime_s": round(elapsed, 1),
        "created_at": now_iso(),
    }

    write_json_atomic(OUT_DIR / "reduced_metrics.json", metrics)
    topic_info.to_csv(OUT_DIR / "reduced_topic_info.csv", index=False)

    terms_rows = []
    for tid in non_outlier["Topic"]:
        tw = selected_model.get_topic(tid)
        if tw:
            for rank, (word, score) in enumerate(tw[:20]):
                terms_rows.append({"topic_id": tid, "rank": rank, "word": word, "score": round(score, 6)})
    pd.DataFrame(terms_rows).to_csv(OUT_DIR / "reduced_topic_terms.csv", index=False)

    assign_rows = []
    for r, t in zip(docs, reduced_topics):
        assign_rows.append({
            "document_id": r["document_id"],
            "topic_id": t,
            "year": r["year"],
            "session_category": r.get("session_category", ""),
            "speaker_family": r.get("speaker_family", ""),
            "source_record_id": r.get("source_record_id", ""),
            "turn_index": r.get("turn_index", ""),
            "chunk_index": r.get("chunk_index", 0),
            "word_count": r.get("word_count", 0),
        })
    pd.DataFrame(assign_rows).to_csv(OUT_DIR / "reduced_document_assignments.csv", index=False)

    # Reduction mapping
    mapping_rows = []
    try:
        # Compare original selected_topics vs reduced_topics
        orig_to_new: dict[int, set] = {}
        for orig, new in zip(selected_topics, reduced_topics):
            orig_to_new.setdefault(orig, set()).add(new)
        for orig, news in sorted(orig_to_new.items()):
            mapping_rows.append({"original_topic_id": orig, "reduced_topic_id": sorted(news)[0]})
        pd.DataFrame(mapping_rows).to_csv(OUT_DIR / "reduction_mapping.csv", index=False)
    except Exception as e:
        log.warning(f"Could not save reduction mapping: {e}")

    return metrics, reduced_topics


# ---------------------------------------------------------------------------
# 16. Representative documents
# ---------------------------------------------------------------------------
def save_representative_docs(
    topic_model,
    docs: list[dict],
    topics: list[int],
    prefix: str,
    top_n: int = 5,
) -> None:
    log.info(f"Saving representative docs: {prefix}")
    topic_info = topic_model.get_topic_info()
    non_outlier_topics = sorted(topic_info[topic_info["Topic"] != -1]["Topic"].tolist())

    by_topic: dict[int, list[tuple[int, float]]] = {}
    for idx, (r, t) in enumerate(zip(docs, topics)):
        if t == -1:
            continue
        score = 1.0
        try:
            probs = topic_model.probabilities_
            if probs is not None and idx < len(probs):
                score = float(probs[idx][t]) if hasattr(probs[idx], '__len__') else float(probs[idx])
        except Exception:
            pass
        by_topic.setdefault(t, []).append((idx, score))

    rows = []
    for tid in non_outlier_topics:
        entries = sorted(by_topic.get(tid, []), key=lambda x: -x[1])[:top_n]
        for rank, (idx, score) in enumerate(entries):
            r = docs[idx]
            doc_hash = sha256_bytes(r["modeling_text"].encode("utf-8"))
            rows.append({
                "topic_id": tid,
                "rank": rank,
                "document_id": r["document_id"],
                "source_record_id": r.get("source_record_id", ""),
                "year": r["year"],
                "session_category": r.get("session_category", ""),
                "turn_index": r.get("turn_index", ""),
                "chunk_index": r.get("chunk_index", 0),
                "speaker_family": r.get("speaker_family", ""),
                "word_count": r.get("word_count", 0),
                "score": round(score, 6),
                "text_excerpt": r["modeling_text"][:500].replace("\n", " "),
                "full_document_sha256": doc_hash,
            })

    out_path = OUT_DIR / f"{prefix}_representative_documents.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info(f"Saved {len(rows)} representative documents to {out_path.name}")


# ---------------------------------------------------------------------------
# 17. NMF comparison
# ---------------------------------------------------------------------------
def build_model_comparison(primary_m: dict, fallback_m: dict, reduced_m: dict) -> None:
    log.info("Building model comparison...")
    grid_path = HANDOFF_DIR / "grid_metrics.csv"

    nmf_row: dict = {"model": "NMF P1 K=24", "note": "primary_model"}
    if grid_path.is_file():
        try:
            df_grid = pd.read_csv(grid_path)
            log.info(f"Grid metrics columns: {list(df_grid.columns)}")
            # Try to find the K=24 row
            k_col = "k" if "k" in df_grid.columns else ("n_topics" if "n_topics" in df_grid.columns else None)
            if k_col:
                k24_rows = df_grid[df_grid[k_col] == 24]
            else:
                k24_rows = df_grid.iloc[[0]]
            if len(k24_rows) > 0:
                k24_row = k24_rows.iloc[0]
                nmf_row.update({
                    "documents": 75121,
                    "coverage_fraction": 1.0,
                    "outlier_fraction": 0.0,
                    "topic_count": 24,
                    "npmi_mean": round(float(k24_row.get("mean_npmi_coherence_top10", 0)), 4),
                    "diversity": round(float(k24_row.get("topic_diversity_top10", 0)), 4),
                    "mean_exclusivity": round(float(k24_row.get("mean_topic_exclusivity_top10", 0)), 4),
                    "mean_redundancy": round(float(k24_row.get("redundancy_mean_off_diagonal_cosine", 0)), 4),
                    "max_redundancy": round(float(k24_row.get("redundancy_max_cosine", 0)), 4),
                    "entropy": "N/A",
                    "top5_share": "N/A",
                })
        except Exception as e:
            log.warning(f"Could not read NMF grid metrics: {e}")
    else:
        log.warning("Grid metrics CSV not found, NMF row will be partial")

    prev_row = {
        "model": "BERTopic MiniLM-L12-v2 (exploratory, truncated 128tok)",
        "note": "previous_benchmark_defective",
        "documents": 75121,
        "coverage_fraction": round(1 - 0.6131, 4),
        "outlier_fraction": 0.6131,
        "topic_count": "84 (native) / 23 (reduced)",
        "npmi_mean": "N/A",
        "diversity": "N/A",
        "mean_exclusivity": "N/A",
        "mean_redundancy": "N/A",
        "max_redundancy": "N/A",
        "entropy": "N/A",
        "top5_share": "N/A",
    }

    def m_row(m: dict, label: str, note: str) -> dict:
        return {
            "model": label,
            "note": note,
            "documents": m.get("total_documents", 75121),
            "coverage_fraction": round(m.get("assigned_documents", 0) / m.get("total_documents", 1), 4),
            "outlier_fraction": m.get("outlier_fraction", "N/A"),
            "topic_count": m.get("topic_count_excl_outlier", m.get("actual_topic_count", "N/A")),
            "npmi_mean": m.get("npmi_mean", "N/A"),
            "diversity": m.get("diversity", "N/A"),
            "mean_exclusivity": m.get("mean_exclusivity", "N/A"),
            "mean_redundancy": m.get("mean_redundancy", "N/A"),
            "max_redundancy": m.get("max_redundancy", "N/A"),
            "entropy": m.get("normalized_entropy", "N/A"),
            "top5_share": m.get("top5_share", "N/A"),
        }

    rows = [
        nmf_row,
        prev_row,
        m_row(primary_m, "BERTopic BGE-M3 primary (min_samples=10)", "new_full_context_primary"),
        m_row(fallback_m, "BERTopic BGE-M3 fallback (min_samples=5)", "new_full_context_fallback"),
        m_row(primary_m if primary_m.get("_selected") else fallback_m,
              "BERTopic BGE-M3 selected native", "new_full_context_selected_native"),
        m_row(reduced_m, f"BERTopic BGE-M3 reduced (~24 topics)", "new_full_context_reduced"),
    ]

    pd.DataFrame(rows).to_csv(OUT_DIR / "model_comparison.csv", index=False)

    notes = (
        "# Model Comparison Notes\n\n"
        "## Structural caveats\n\n"
        "- NMF produces soft topic mixtures (documents have fractional membership across all topics).\n"
        "  BERTopic/HDBSCAN produces hard cluster assignments with explicit outlier class (-1).\n"
        "- c-TF-IDF weights (BERTopic representation) and NMF component loadings are not mathematically equivalent.\n"
        "- NPMI, diversity, exclusivity, and redundancy metrics are computed analogously but not identically across models.\n"
        "- Coverage fraction is essential for longitudinal prevalence estimation.\n"
        "  NMF covers 100% of documents; HDBSCAN outliers are excluded from prevalence.\n\n"
        "## Previous benchmark defects\n\n"
        "- paraphrase-multilingual-MiniLM-L12-v2 truncated documents at 128 tokens.\n"
        "- Fallback configuration was mistakenly selected despite HIGHER outlier fraction than primary.\n"
        "- Final run reported 318 stopwords instead of correct 315.\n"
        "- RAM reported as 0.0 GB due to psutil bug.\n\n"
        "## This benchmark corrections\n\n"
        "- BAAI/bge-m3 dense embeddings with full-context (no truncation).\n"
        "- modeling_text (natural text) used for embeddings; cleaned_text for c-TF-IDF.\n"
        "- Primary and fallback UMAP computed once and shared.\n"
        "- Selection rule declared before examining results.\n"
        "- Only selected model reduced to ~24 topics.\n"
        "- P1 stopwords asserted = 315.\n"
        "- RAM correctly reported.\n"
    )
    (OUT_DIR / "model_comparison_notes.md").write_text(notes, encoding="utf-8")
    log.info("Model comparison saved.")


# ---------------------------------------------------------------------------
# 18. Review bundle
# ---------------------------------------------------------------------------
def build_review_bundle(env: dict, pip_freeze: str) -> str:
    log.info("Building review bundle...")
    write_json_atomic(REVIEW_DIR / "environment.json", env)
    (REVIEW_DIR / "requirements_freeze.txt").write_text(pip_freeze, encoding="utf-8")

    review_files = [
        "token_length_audit.json",
        "token_length_distribution.csv",
        "embedding_manifest.json",
        "umap_manifest.json",
        "smoke_metrics.json",
        "smoke_manifest.json",
        "primary_metrics.json",
        "fallback_metrics.json",
        "selection_decision.json",
        "reduced_metrics.json",
        "primary_topic_info.csv",
        "fallback_topic_info.csv",
        "selected_native_topic_info.csv",
        "reduced_topic_info.csv",
        "selected_native_topic_terms.csv",
        "reduced_topic_terms.csv",
        "reduction_mapping.csv",
        "selected_native_representative_documents.jsonl",
        "reduced_representative_documents.jsonl",
        "model_comparison.csv",
        "model_comparison_notes.md",
        "run_log.txt",
    ]

    sums: dict[str, str] = {}
    for fname in review_files:
        src = OUT_DIR / fname
        if src.is_file():
            import shutil
            shutil.copy2(src, REVIEW_DIR / fname)
            sums[fname] = sha256_file(REVIEW_DIR / fname)

    sum_lines = "\n".join(f"{h}  {f}" for f, h in sorted(sums.items()))
    (REVIEW_DIR / "SHA256SUMS.txt").write_text(sum_lines + "\n", encoding="utf-8")

    zip_path = OUT_DIR / "bertopic_full_context_review.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src_path in sorted(REVIEW_DIR.rglob("*")):
            if src_path.is_file():
                zf.write(src_path, src_path.relative_to(REVIEW_DIR))

    zip_sha = sha256_file(zip_path)
    log.info(f"Review ZIP: {zip_path} | SHA256: {zip_sha}")
    write_json_atomic(OUT_DIR / "review_zip_sha256.json", {
        "path": str(zip_path),
        "sha256": zip_sha,
        "size_bytes": zip_path.stat().st_size,
        "created_at": now_iso(),
    })
    return zip_sha


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Load saved embeddings from disk and resume from UMAP")
    args = parser.parse_args()

    log.info("=== BERTopic full-context benchmark v1 started ===")
    t0_global = time.time()

    import subprocess
    base_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT
    ).strip()
    log.info(f"Base commit: {base_commit}")

    # 1. Stopwords
    sw, clean_natural_text, lexical_tokens = load_project_stopwords()

    # 2. Corpus
    docs, zero_ids = load_corpus()

    # 3. Corpus manifest
    corpus_manifest = build_corpus_manifest(docs, zero_ids, base_commit)
    write_json_atomic(OUT_DIR / "corpus_manifest.json", corpus_manifest)
    log.info(f"Corpus manifest: doc_id_hash={corpus_manifest['ordered_document_id_sha256'][:16]}...")

    # 4. Vectorizer
    vectorizer = build_p1_vectorizer(sw, clean_natural_text, lexical_tokens)

    # 5. Environment
    env = inspect_environment()

    # 6. Token audit
    token_audit, safe_max_length = run_token_audit(docs)

    if args.resume:
        # Load previously saved embeddings and skip model loading
        emb_path = OUT_DIR / "embeddings.npy"
        id_path = OUT_DIR / "document_ids.txt"
        if not emb_path.is_file():
            raise FileNotFoundError(f"--resume: embeddings.npy not found at {emb_path}")
        log.info(f"--resume: loading embeddings from {emb_path}")
        embs = np.load(emb_path)
        doc_ids = id_path.read_text(encoding="utf-8").splitlines()
        assert embs.shape[0] == len(docs), f"Embedding rows {embs.shape[0]} != docs {len(docs)}"
        assert len(doc_ids) == len(docs), f"doc_ids count {len(doc_ids)} != docs {len(docs)}"
        assert np.isfinite(embs).all(), "NaN/Inf in loaded embeddings"
        norms = np.linalg.norm(embs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), "Loaded embeddings not normalized"

        # Save embedding manifest
        emb_sha = sha256_file(emb_path)
        model_path_str = resolve_model_path()
        emb_manifest = {
            "model_id": CFG["embedding"]["model_id"],
            "model_snapshot": str(model_path_str),
            "model_revision": Path(model_path_str).name,
            "pooling_strategy": "CLS_token",
            "embedding_backend": "transformers_AutoModel",
            "embedding_shape": list(embs.shape),
            "dtype": str(embs.dtype),
            "byte_size": emb_path.stat().st_size,
            "sha256": emb_sha,
            "max_length": safe_max_length,
            "device": CFG["embedding"]["device"],
            "normalize": True,
            "dense_only": True,
            "truncated_documents": 0,
            "nan_count": 0,
            "infinite_count": 0,
            "mean_norm": float(norms.mean()),
            "min_norm": float(norms.min()),
            "max_norm": float(norms.max()),
            "document_count": len(docs),
            "loaded_from_cache": True,
            "created_at": now_iso(),
        }
        id_path.write_text("\n".join(doc_ids), encoding="utf-8")
        write_json_atomic(OUT_DIR / "embedding_manifest.json", emb_manifest)
        log.info(f"Embeddings loaded: {embs.shape} sha256={emb_sha[:16]}...")
    else:
        # 7. Load model (returns tuple: model, tokenizer, model_path)
        model_tuple = load_bge_m3(safe_max_length)

        # 8. Smoke test
        if not args.skip_smoke:
            run_smoke_test(docs, model_tuple, safe_max_length, sw, vectorizer)

        if args.smoke_only:
            log.info("--smoke-only: stopping after smoke test.")
            return

        # 9. Full embeddings
        embs, doc_ids, emb_manifest = generate_full_embeddings(docs, model_tuple, safe_max_length, env)

        # Free model from GPU after embedding
        model_nn, model_tok, model_path = model_tuple
        del model_nn, model_tok, model_tuple
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    # 10. UMAP (once — or load from cache if resuming)
    umap_path = OUT_DIR / "umap_embeddings.npy"
    if args.resume and umap_path.is_file():
        log.info(f"--resume: loading UMAP embeddings from {umap_path}")
        umap_embs = np.load(umap_path)
        assert umap_embs.shape == (len(docs), CFG["umap"]["n_components"]), (
            f"UMAP shape mismatch: {umap_embs.shape}"
        )
        assert np.isfinite(umap_embs).all(), "NaN/Inf in loaded UMAP embeddings"
        log.info(f"UMAP loaded: {umap_embs.shape}")
    else:
        umap_embs = compute_umap(embs)

    # 11. Primary HDBSCAN
    primary_metrics, primary_model, primary_topics = run_hdbscan_config(
        "primary",
        CFG["hdbscan_primary"],
        umap_embs,
        embs,
        docs,
        vectorizer,
    )

    # 12. Fallback HDBSCAN
    fallback_metrics, fallback_model, fallback_topics = run_hdbscan_config(
        "fallback",
        CFG["hdbscan_fallback"],
        umap_embs,
        embs,
        docs,
        vectorizer,
    )

    # 13. Selection
    decision = select_configuration(primary_metrics, fallback_metrics)
    selected_name = decision["selected"]
    selected_model = primary_model if selected_name == "primary" else fallback_model
    selected_topics = primary_topics if selected_name == "primary" else fallback_topics
    selected_metrics = primary_metrics if selected_name == "primary" else fallback_metrics
    selected_metrics["_selected"] = True

    copy_selected_native(selected_name)

    # Save selected native representative docs
    save_representative_docs(selected_model, docs, selected_topics, "selected_native")

    # 14. Reduction
    reduced_metrics, reduced_topics = reduce_to_24(
        selected_model, docs, selected_topics, embs, vectorizer
    )

    # Save reduced representative docs
    save_representative_docs(selected_model, docs, reduced_topics, "reduced")

    # 15. Model comparison
    build_model_comparison(primary_metrics, fallback_metrics, reduced_metrics)

    # 16. Pip freeze
    try:
        pip_out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except subprocess.CalledProcessError:
        # uv venvs don't include pip; use uv pip list instead
        try:
            pip_out = subprocess.check_output(
                ["uv", "pip", "freeze", "--python", sys.executable], text=True
            )
        except Exception as e2:
            pip_out = f"# Could not capture pip freeze: {e2}\n"

    # 17. Review bundle
    zip_sha = build_review_bundle(env, pip_out)

    total_s = time.time() - t0_global
    log.info(f"=== COMPLETE in {total_s:.0f}s | review ZIP SHA256: {zip_sha} ===")

    # Final assertions
    assert len(docs) == 75121
    assert all(d.get("modeling_text") for d in docs)
    assert len(set(d["document_id"] for d in docs)) == len(docs)

    log.info("All final assertions PASSED.")


if __name__ == "__main__":
    main()
