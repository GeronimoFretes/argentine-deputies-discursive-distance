# Modeling Corpus Export

## Purpose

The modeling corpus is the primary text input for downstream topic modeling. Its estimand is spoken parliamentary discourse reconstructed from speaker-turn outputs. Formal documentary agenda text, bills, reports, annexes, and other inserted records are not part of the primary spoken-discourse corpus unless they remain in a turn as `spoken_text` after the versioned speaker-turn parser and content classifier.

The exporter is deterministic and provenance-preserving. It does not modify parser outputs, content spans, speaker turns, structural segmentation, PDF extraction, or full-corpus QA artifacts.

## Inputs

The exporter reads successful speaker-turn document directories from:

```text
data/interim/speaker_turns/<source_record_id>/
```

Every input `speaker_turns.json` manifest must declare:

```text
pipeline_version = "2"
content_classifier_version = "2"
```

Session metadata is read from the full-corpus run summary, by default:

```text
data/qa/full_corpus_run_summary.json
```

Exceptional modeling decisions are read from:

```text
config/modeling_turn_overrides.json
```

The override manifest is versioned and must require the same parser and classifier versions as the exporter inputs.

## Override Layer

Supported override actions are:

* `exclude_turn`
* `retain_from_anchor_and_relabel`

Override validation is strict. The exporter fails if the source record is missing, the turn index is missing or duplicated, the upstream parser speech word count differs from `expected_speech_word_count`, the required input versions do not match, the configured anchor occurrence count does not match, an unsupported action or field is present, or any declared override is not applied exactly once.

For `retain_from_anchor_and_relabel`, the exporter discards spoken text before `start_anchor`, retains the anchor and all following spoken text, updates the effective `speaker_family` and `normalized_label`, and records the discarded prefix in `exclusion_ledger.jsonl`. Persisted parser files are not edited.

## Inclusion Policy

Only source-turn text reconstructed from `content_spans.jsonl` records with:

```text
content_kind == "spoken_text"
include_in_speech == true
```

is eligible for export. Span order and source fragments are preserved exactly.

The upstream parser computes `speech_word_count` as the sum of `word_count` over selected spoken content spans. It does not count words over one concatenated turn string. The exporter therefore keeps two explicit count concepts:

* `original_upstream_speech_word_count`: parser-compatible span-sum count used for override validation.
* `post_override_modeling_word_count`: whitespace-token count over the deterministic reconstructed spoken text after overrides, used for thresholding, chunk limits, retained totals, and manifest word totals.

When a turn has multiple selected spoken spans, the exporter reconstructs spoken text by inserting this separator between distinct selected fragments:

```text
\n\n
```

These separator characters are synthetic. They are not attributed to PDF/source offsets.

After overrides, the retained speaker families are:

* `named_or_role_unspecified`
* `executive_official`

Chair, secretary, collective, anonymous, unattributed, and other out-of-scope speaker families are excluded from the modeling corpus, while their decisions are still recorded.

The minimum length threshold is exactly 25 whitespace-delimited words after overrides:

* 24 words: excluded
* 25 words: retained

Zero-speech turns are recorded as `excluded_zero_speech`.

## Chunking

Each retained source turn is split into ordered modeling documents.

Chunking rules:

* separate source turns are never combined;
* separate speakers are never combined;
* chunks are non-overlapping and ordered;
* every retained source word belongs to exactly one chunk;
* no chunk exceeds 300 whitespace-delimited words by default;
* inside the final quarter of the chunk window, paragraph boundaries are preferred over sentence boundaries, which are preferred over weaker clause boundaries;
* when the final quarter contains no usable boundary, the latest boundary in the latter half of the window is selected, regardless of type, with boundary strength used only as a tie-breaker;
* when every available boundary occurs before the latter half of the window, the exporter uses the hard word cap rather than creating a pathologically short chunk;
* an indivisible unit longer than the maximum is split deterministically by whitespace;
* empty chunks are not emitted.

`exact_text` is deterministic reconstructed spoken text. It contains exact source fragments plus any documented synthetic `\n\n` separators required between independently persisted spoken spans. It is not necessarily one contiguous source substring. Concatenating `exact_text` for a turn's chunks in `chunk_index` order reconstructs the retained post-override source-turn text exactly, with no separator insertion. `modeling_text` is conservatively whitespace-normalized for downstream modeling convenience.

Provenance entries use `fragment_kind`:

* `source_fragment`: exact content-span source identifiers, source offsets, reconstructed-text offsets, character count, and fragment hash.
* `synthetic_separator`: reconstructed-text offsets, separator policy, character count, and fragment hash, with no source offsets.

## Temporal Periods

Each record is assigned exactly one period from its session year:

```text
2008-2011
2012-2015
2016-2019
2020-2023
2024-2025
```

Dates outside 2008 through 2025 are rejected.

## CLI

Default export command:

```text
uv run python -m argentine_deputies_discursive_distance.cli export-modeling-corpus
```

The command supports explicit paths for the speaker-turn root, override file, metadata summary, and output directory, plus `--minimum-words`, `--maximum-chunk-words`, and `--force`.

The canonical output directory is:

```text
data/processed/modeling_corpus/
```

The exporter refuses to overwrite a nonempty output directory unless `--force` is supplied.

## Outputs

### `documents.jsonl`

One record per modeling chunk. Key fields:

```text
document_id
source_record_id
session_date
year
temporal_period
session_category
turn_index
chunk_index
chunk_count_for_turn
speaker_label
normalized_label
speaker_family
exact_text
modeling_text
word_count
modeling_word_count
source_turn_modeling_word_count_after_override
override_applied
override_action
source_pages_covered
provenance
```

`document_id` is stable and source/turn/chunk based.

### `source_turns.jsonl`

One record per retained post-override source turn before chunking. It includes source and session metadata, original and effective speaker metadata, upstream and modeling word counts, override information, chunk count, deterministic reconstructed retained text, provenance, source pages covered, and a stable content hash.

### `turn_decisions.jsonl`

One record for every parsed source turn, including zero-speech turns. Decisions include:

```text
retained
excluded_zero_speech
excluded_by_override
excluded_speaker_family
excluded_below_minimum_words
```

Each source turn appears exactly once.

### `exclusion_ledger.jsonl`

One record per excluded turn or trimmed fragment. Records include source and turn identity, reason, modeling word count, speaker metadata, override metadata, exact excluded reconstructed text where available, content hash, and provenance.

### `export_manifest.json`

The manifest records exporter and input versions, override manifest version, the SHA-256 of `config/modeling_turn_overrides.json`, chunking configuration, inclusion policy, corpus counts, grouped counts and words, output file sizes, output file SHA-256 hashes, and an override application ledger.

It also records reconciliation checks for parsed-turn accounting, retained modeling-word accounting, chunk word reconstruction, duplicate IDs and decisions, maximum chunk length, minimum retained-turn length, allowed speaker families, date bounds, source-turn/document mapping, separator policy, and one-time application of every override.

## Reproducibility

Records are emitted with deterministic ordering and JSON serialization. JSONL outputs are streamed to temporary part files while running counts, hashes, grouped statistics, override application counts, and reconciliation state are maintained. The exporter does not serialize complete corpus JSONL files in memory. Outputs are validated before final replacement. A failed run must not leave a partially updated final corpus.

Repeated exports with the same inputs produce byte-identical JSONL files. The manifest is deterministic except for `generated_at_utc`.

## Known Limitation

The primary corpus deliberately excludes documentary agenda material that is not classified as spoken discourse. This improves alignment with the spoken-discourse estimand but means the modeling corpus is not a complete record of all text printed in proceedings documents.
