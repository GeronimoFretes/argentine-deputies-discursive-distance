# NMF Grid

`fit-nmf-grid` preprocesses the locked modelling corpus primary universe and fits a
configured sparse TF-IDF plus NMF grid. It does not select a winning K.

Default command:

```powershell
uv run deputies-distance fit-nmf-grid
```

Production defaults read:

- `data/processed/modeling_corpus/documents.jsonl`
- `data/processed/modeling_corpus/export_manifest.json`
- `data/qa/modeling_corpus_lock_v1.json`
- `data/qa/topic_modeling/corpus_profile_v1/profile_manifest.json`
- `config/topic_modeling/nmf_grid_v1.json`
- `config/topic_modeling/stopwords_es_p0_v1.txt`

Outputs are written transactionally to `data/qa/topic_modeling/nmf_grid_v1/`.
The command refuses a nonempty output directory unless `--force` is supplied.

## Lineage Checks

Before preprocessing, the stage reuses the corpus-profile locked-input validator for
the export manifest and corpus lock. It then verifies that the corpus-profile manifest
hashes match the current documents, export manifest, corpus lock, and
`config/topic_modeling/corpus_profile_v1.json`; every corpus-profile reconciliation
check is `true`; all-session totals are 86,270 documents and 18,315,187 words; and
the primary universe is 75,123 documents, 34,060 source turns, 243 sessions, and
15,901,236 words.

## Cleaning

The original `modeling_text` is preserved in the cleaned JSONL output. A separate
`cleaned_text` is produced by applying, in order:

1. Join alphabetic text separated by U+00AD soft hyphen and optional whitespace,
   then remove any remaining U+00AD. Soft hyphens are never replaced with spaces.
2. Repeatedly join alphabetic line-break hyphenation matching an alphabetic
   character, ASCII `-`, whitespace, and another alphabetic character until stable.
3. Normalize with Unicode NFKC and `casefold()`.

No stemming, lemmatization, accent stripping, named-entity removal, or province-name
removal is performed.

## Stopwords

P0 is a local conservative Spanish function-word list with 310 normalized entries.
It is stored as UTF-8 without BOM at
`config/topic_modeling/stopwords_es_p0_v1.txt`.

P0 SHA-256:
`b4d338c3aed3e225105bc0b9cdaf3ae775f131e68cfbc6d9a0e61a8152179c3a`

Normalization policy: each nonblank, non-comment line is NFKC-normalized and
casefolded at runtime. P0 excludes protected substantive terms such as `estado`,
`ley`, `derecho`, `trabajo`, `nación`, `comisión`, `presupuesto`, `salud`, and the
other protected terms listed in tests. P1 equals P0 plus exactly `señor`, `señora`,
`señores`, `presidente`, and `presidenta`, for 315 entries. The production grid uses
P1; P0 remains available for later sensitivity runs.

## TF-IDF

The vectorizer uses sparse scikit-learn TF-IDF with the project lexical tokenizer:
Unicode alphanumeric clusters are considered, but only purely alphabetic tokens of
length at least 3 are retained. Accents and `ñ` are preserved, numeric and
mixed-alphanumeric tokens are excluded, and text is already NFKC/casefolded before
tokenization.

Production settings are `ngram_range=(1, 2)`, `min_df=20`, `max_df=0.95`,
`max_features=40000`, `sublinear_tf=True`, `smooth_idf=True`, `norm="l2"`,
`dtype=float32`, `lowercase=False`, and no accent stripping. The fitted vectorizer is
saved with joblib and reused for every K. Serialized model bytes may vary across
library versions and platforms.

## NMF And Metrics

Production K values are 12, 16, 20, 24, and 28. Each model uses coordinate descent,
Frobenius loss, NNDSVDa initialization, random state 42, maximum 400 iterations,
tolerance 0.0001, and no sparsity regularization.

Metrics are reported per K:

- Topic diversity@10: unique terms across all topic top-10 lists divided by `K * 10`.
- Topic exclusivity@10: for each selected topic-term pair, the topic weight divided
  by that term's summed weight across all topics; reported as per-topic means and a
  global mean.
- Topic redundancy: pairwise cosine similarity between dense topic-term vectors,
  with K less than 2 handled explicitly.
- NPMI coherence@10: binary document co-occurrence from the vectorized primary
  corpus, computed only for the union of top terms. Zero co-occurrence pairs receive
  NPMI `-1`.
- Document-topic concentration: nonzero document-topic rows are normalized to sum to
  one; dominant-topic weight, normalized entropy, and zero-weight rows are reported.
