# Corpus Profile

## Purpose

`profile-modeling-corpus` is the first topic-modelling analysis stage. It profiles the locked modelling corpus and emits evidence for later preprocessing choices. It does not fit NMF, LDA, BERTopic, embeddings or any other topic model.

The stage is diagnostic. It preserves the modelling corpus exactly as exported and does not mutate `modeling_text`.

## Input contract

Canonical production inputs are:

```text
data/processed/modeling_corpus/documents.jsonl
data/processed/modeling_corpus/export_manifest.json
data/qa/modeling_corpus_lock_v1.json
config/topic_modeling/corpus_profile_v1.json
```

The profiler validates that all files exist, that the corpus lock matches the export manifest hash, that `documents.jsonl` matches the export manifest hash, that every required locked total and version field exists in both the export manifest and corpus lock, that locked totals match manifest totals, and that the exporter, speaker-turn pipeline and content-classifier versions are the expected checked-in versions.

Every JSONL document is validated strictly for required fields, duplicate `document_id`, `word_count` in `1..300`, `len(modeling_text.split()) == word_count`, optional `modeling_word_count == word_count`, valid session date/year/period, and an eligible speaker family:

```text
named_or_role_unspecified
executive_official
```

Malformed records fail the run. No records are silently skipped.

## Analytical universes

Every relevant row is labelled with a universe:

```text
all_sessions
primary
```

`all_sessions` includes every modelling document. `primary` includes documents whose `session_category` is configured as `legislative_debate` in `corpus_profile_v1.json`.

## Tokenizer policy

The profiling tokenizer is `diagnostic_lexical_nfkc_casefold_alnum_v1`.

Policy:

```text
input field: modeling_text
Unicode normalization: NFKC
case normalization: casefold
split rule: deterministic split on non-alphanumeric Unicode boundaries
accented letters: preserved
lemmatization: none
stopword removal: none
name/province removal: none
numeric and mixed-alphanumeric tokens: retained
```

Token classes are reported separately for alphabetic, numeric, mixed alphanumeric, one-character, two-character, and three-or-more-character tokens.

## Exact versus sampled statistics

Full-corpus exact statistics include corpus counts, grouped counts, document-length statistics and histograms, lexical token totals and classes, vocabulary growth by year, unigram total frequency, unigram document frequency, candidate-stopword counts and reasons, and suspicious-token counts.

Sampled outputs are limited to deterministic sample membership, sampled bigram frequency, candidate-token context examples, and preprocessing examples. Exact full-corpus bigram counting is not attempted. The profiler uses a deterministic stratified sample and counts bigrams only over that sample. `corpus_profile.json` records structured scope metadata under `statistics_scope`.

## Candidate stopwords

`candidate_stopwords.csv` contains diagnostic candidates only. Tokens are never removed in this stage, and `selected_for_removal` is always `false`.

Candidate reasons include:

```text
high_document_fraction
high_total_frequency
very_short_alpha_token
procedural_seed_match
```

The procedural seed list is small and conservative: `señor`, `señora`, `presidente`, `presidenta`, `diputado`, `diputada`, `honorable`, `cámara`, `sesión`, `palabra`, and `gracias`.

## Suspicious tokens

`suspicious_tokens.csv` records evidence for possible text-quality or tokenization issues, including replacement characters, mojibake-like sequences, soft hyphens, mixed letter/digit tokens, repeated characters, numeric-only tokens, unusually long tokens, raw-text control characters, one-character alphabetic tokens, and unexpected raw Unicode categories such as private-use, surrogate, unassigned or format characters outside the separate soft-hyphen diagnostic.

Suspicious-token snippets include `snippet_text_kind`. Token-derived diagnostics use snippets from NFKC-casefolded modelling text so normalized tokens can be located consistently. Raw anomalies such as replacement characters, soft hyphens, mojibake-like sequences, control characters and unexpected raw Unicode categories use raw flattened `modeling_text` snippets.

These rows are diagnostics only. They are not automatic exclusions.

## Output schemas

Default output directory:

```text
data/qa/topic_modeling/corpus_profile_v1/
```

Outputs:

```text
profile_manifest.json
corpus_profile.json
corpus_profile.md
counts_by_year.csv
counts_by_temporal_period.csv
counts_by_session_category.csv
counts_by_speaker_family.csv
counts_by_year_and_category.csv
document_length_histogram.csv
token_frequency.csv
candidate_stopwords.csv
suspicious_tokens.csv
sampled_documents.jsonl
sampled_bigram_frequency.csv
preprocessing_examples.jsonl
```

CSV count files include `universe`, document count, source-turn count, session count, word total, and mean document length. `token_frequency.csv` includes total frequency, document frequency, document fraction, and ranks by both total frequency and document frequency. `hapax_count` in `corpus_profile.json` means total corpus frequency exactly one; document-frequency singleton counts are reported separately as `tokens_appearing_in_exactly_1_document`. JSONL sample files include deterministic sample membership and bounded preprocessing examples.

`profile_manifest.json` records input paths, input SHA-256 hashes, locked counts, processed counts, universe counts, sample counts, configuration hash, output hashes and reconciliation checks.

## Reproducibility

Configuration is versioned in `config/topic_modeling/corpus_profile_v1.json`.

Sampling is deterministic and order-independent. For each document, the sampler ranks:

```text
SHA256(random_seed + ":" + document_id)
```

It stratifies by `year` and `session_category`, keeping the smallest stable hashes up to `sample_documents_per_stratum`.

Outputs are written as UTF-8 without BOM through `.part` files. The profiler refuses a nonempty output directory unless `--force` is supplied. Outputs are promoted only after reconciliation checks pass.

Repeated runs with identical inputs are byte-identical except for `generated_at_utc` in `profile_manifest.json`.

## CLI usage

Default production command:

```text
uv run python -m argentine_deputies_discursive_distance.cli profile-modeling-corpus
```

Explicit paths:

```text
uv run python -m argentine_deputies_discursive_distance.cli profile-modeling-corpus \
  --documents data/processed/modeling_corpus/documents.jsonl \
  --export-manifest data/processed/modeling_corpus/export_manifest.json \
  --corpus-lock data/qa/modeling_corpus_lock_v1.json \
  --config config/topic_modeling/corpus_profile_v1.json \
  --output-dir data/qa/topic_modeling/corpus_profile_v1
```

Use `--force` only to replace an existing nonempty profile output directory.

## Why no topic models

This stage exists to make preprocessing decisions auditable before modelling. Fitting a topic model here would mix corpus diagnostics with downstream modelling assumptions. Topic-model fitting belongs in a later, separately versioned stage after stopword, token, vocabulary and sampling decisions are reviewed and frozen.
