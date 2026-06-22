# Speaker-Turn Batch Output Contract

## Status

Proposed contract for the `feat/speaker-turn-batch` workstream.

The contract must be reviewed before the persistence implementation is finalized.

## Purpose

This pipeline integrates the validated speaker-marker, speaker-turn, and turn-content parsers into a deterministic, resumable corpus-processing stage.

The stage has four responsibilities:

1. read validated structural-segmentation outputs;
2. construct lossless speaker turns;
3. classify exact turn-content spans;
4. persist deterministic outputs and a downstream speaker inventory.

This stage does not resolve legislators, assign parliamentary blocs, or assign political alignment.

## Inputs

The batch-level input is the structural-segmentation batch summary:

```text
data/qa/structural_segmentation_summary.json
```

Each selected record must provide:

```text
label
source_record_id
session_date
period
meeting_number
session_category
structural_blocks_path
structure_path
```

For every document, the pipeline must read and validate:

```text
structure.json
structural_blocks.jsonl
```

The pipeline must verify that:

* the source record identifier is present and safe;
* the structural summary belongs to the expected record;
* the structural-block file exists;
* the structural-block file size matches its recorded size;
* the structural-block SHA-256 matches its recorded hash;
* all structural-block records belong to the same source record;
* page and reading-order references are unique.

Only blocks whose structural zone is `proceedings` are passed to speaker-turn parsing.

Excluded proceedings blocks must still be supplied because they can contain barriers or explicit markers that affect attribution.

## Per-document directory

Each document is written under:

```text
data/interim/speaker_turns/<source_record_id>/
```

The directory contains:

```text
turns.jsonl
turn_segments.jsonl
content_spans.jsonl
speaker_turns.json
```

## `turns.jsonl`

This file contains one record per parsed speaker turn.

Records are ordered by ascending `turn_index`.

Required fields:

```text
source_record_id
turn_index
marker
marker_page_number
marker_reading_order
marker_block_reference
marker_block_included
normalized_label
speaker_family
is_unattributed
segment_count
character_count
word_count
content_span_count
speech_span_count
speech_word_count
documentary_word_count
stage_direction_word_count
editorial_note_word_count
first_reference
last_reference
documentary_boundary
```

### `marker`

`marker` is `null` for unattributed turns.

For explicit turns, it is an object containing:

```text
start
end
raw_marker
raw_title
normalized_title
raw_label
normalized_label
family
separator
separator_kind
position
is_multiline
detection_method
detection_confidence
```

Marker offsets remain relative to the source block containing the marker.

### `documentary_boundary`

`documentary_boundary` is `null` when no documentary transition was detected.

When present, it contains:

```text
turn_offset
page_number
reading_order
source_reference
source_offset
cue
matched_text
classification_method
classification_confidence
documentary_word_count
```

### Text policy

The full synthetic joined-turn text is not persisted in `turns.jsonl`.

Exact text remains available through `turn_segments.jsonl`.

This avoids treating synthetic newlines inserted between source segments as source characters.

## `turn_segments.jsonl`

This file contains one record per exact source segment assigned to a turn.

Records are ordered by:

```text
turn_index
segment_index
```

Required fields:

```text
source_record_id
turn_index
segment_index
page_number
reading_order
block_reference
start
end
text
attribution_method
character_count
word_count
```

`segment_index` is one-based within each turn.

Offsets are relative to the source structural block.

The serialized text must exactly equal:

```python
source_block_text[start:end]
```

## `content_spans.jsonl`

This file contains the exact functional classification of every source span assigned to a turn.

Records are ordered by:

```text
turn_index
content_span_index
```

Required fields:

```text
source_record_id
turn_index
content_span_index
source_segment_index
page_number
reading_order
block_reference
start
end
text
content_kind
include_in_speech
classification_method
classification_confidence
attribution_method
character_count
word_count
```

`content_span_index` is one-based within each turn.

`source_segment_index` links the span to the exact row in `turn_segments.jsonl`.

Allowed `content_kind` values are:

```text
spoken_text
documentary_insert
stage_direction
editorial_note
unattributed_text
```

No content span may cross a source-block boundary.

A logical event may occupy multiple content spans when it crosses source blocks.

## `speaker_turns.json`

This is the per-document summary and cache manifest.

Required top-level fields:

```text
pipeline_version
processed_at_utc
source_record_id
source
outputs
statistics
```

### `source`

Required fields:

```text
structure_path
structure_sha256
segmenter_version
structural_blocks_path
structural_blocks_sha256
structural_blocks_size_bytes
```

### `outputs`

For each output file, record its path, SHA-256, and size:

```text
turns_path
turns_sha256
turns_size_bytes
turn_segments_path
turn_segments_sha256
turn_segments_size_bytes
content_spans_path
content_spans_sha256
content_spans_size_bytes
```

### `statistics`

Required fields:

```text
turn_count
explicit_marker_count
unattributed_turn_count
unattributed_segment_count
barrier_reset_count
assigned_segment_count
assigned_character_count
content_span_count
speech_span_count
speech_word_count
documentary_span_count
documentary_word_count
stage_direction_span_count
stage_direction_word_count
editorial_note_span_count
editorial_note_word_count
unattributed_content_span_count
unattributed_content_word_count
zero_speech_turn_count
maximum_speech_word_count
speaker_family_counts
content_kind_counts
attribution_method_counts
```

All count mappings must use deterministically sorted keys.

## Batch outputs

The batch command produces:

```text
data/qa/speaker_turn_batch_summary.json
data/interim/speaker_turns/speaker_inventory.csv
```

The speaker inventory may later be moved to another deterministic output path if the downstream interface requires it, but its schema must remain versioned and documented.

## `speaker_turn_batch_summary.json`

Required top-level fields:

```text
batch_version
started_at_utc
finished_at_utc
source_structure_summary
output_root
speaker_inventory
record_count
parsing_reused_count
records
aggregate_statistics
```

Each document record must preserve:

```text
label
source_record_id
session_date
period
meeting_number
session_category
parsing_reused
turn_count
explicit_marker_count
unattributed_turn_count
barrier_reset_count
speech_word_count
documentary_word_count
stage_direction_word_count
editorial_note_word_count
speaker_label_count
turns_path
turn_segments_path
content_spans_path
speaker_turns_path
```

## Speaker inventory

The speaker inventory contains explicit marker-derived speaker labels only.

Unattributed material is summarized in the document and batch QA statistics rather than represented as a fictional speaker.

Non-legislative families remain in the inventory for auditing. They are not silently discarded.

The grouping key is:

```text
source_record_id
speaker_raw
speaker_normalized
speaker_family
```

Required columns:

```text
session_id
session_date
period
meeting_number
session_category
source_record_id
speaker_raw
speaker_normalized
speaker_family
turn_count
speech_span_count
speech_word_count
first_reference
sample_text
```

At this stage:

```text
session_id = source_record_id
speaker_raw = marker.raw_label
speaker_normalized = marker.normalized_label
speaker_family = marker.family
```

This preserves distinct printed variants even when they normalize to the same label.

### Speaker-inventory text sample

`sample_text` is derived only from spans where:

```text
include_in_speech = true
```

The sample is:

* taken from the earliest eligible speech for that inventory row;
* normalized only by collapsing whitespace;
* not spelling-corrected;
* limited to 300 Unicode characters;
* stored without synthetic metadata or ellipsis when the text is shorter than the limit.

### Speaker-inventory ordering

Rows are sorted by:

```text
session_date
source_record_id
speaker_normalized
speaker_raw
speaker_family
```

CSV output must use UTF-8 encoding and deterministic column order.

## Reuse policy

The per-document output may be reused only when all of the following match:

* pipeline version;
* source record identifier;
* structural segmenter version;
* structural summary SHA-256;
* structural-block SHA-256;
* all expected output paths;
* all recorded output sizes;
* all recorded output SHA-256 values.

A reused run must not rewrite any per-document output.

The returned in-memory result may add:

```text
reused = true
```

without modifying the persisted summary.

Changing parsing behavior or a persisted schema requires incrementing the pipeline version.

## Atomic-write policy

All per-document output files must first be written to sibling `.part` files.

The final paths are replaced only after every temporary output has been written successfully.

On failure:

* temporary files are removed;
* previously valid final outputs remain untouched;
* no partial cache manifest is written.

The per-document summary is written last.

The batch summary and speaker inventory are also written atomically.

## Determinism

The following must be deterministic:

* document processing order;
* turn order;
* segment order;
* content-span order;
* mapping-key order in JSON;
* JSON indentation and encoding;
* JSONL field order through sorted JSON keys;
* speaker-inventory row order;
* speaker-inventory column order;
* newline convention.

Timestamps may differ only when outputs are genuinely regenerated.

A cache reuse run must leave output bytes unchanged.

## Required invariants

The implementation must validate at least the following:

1. Turn indices are consecutive and begin at one.
2. Segment indices are consecutive within each turn.
3. Content-span indices are consecutive within each turn.
4. Each serialized segment exactly matches its source-block substring.
5. Each content span exactly matches its source-block substring.
6. Every content span lies within exactly one serialized turn segment.
7. Content spans reconstruct every turn segment exactly.
8. No content spans overlap within the same source segment.
9. Marker ranges match the marker source text.
10. Turn-level statistics reconcile with serialized segments and spans.
11. Document-level statistics reconcile with all turn records.
12. Batch-level statistics reconcile with all document summaries.
13. Speaker-inventory totals reconcile with explicit speaker turns and eligible speech spans.
14. Output file hashes and sizes match the cache manifest.
15. Reprocessing a valid document reuses its outputs.
16. Forced reprocessing produces deterministic non-timestamp outputs.

## Proposed CLI command

The command name is:

```text
parse-speaker-turns
```

Proposed usage:

```powershell
uv run deputies-distance parse-speaker-turns `
  --structure-summary data/qa/structural_segmentation_summary.json `
  --output-dir data/interim/speaker_turns `
  --summary data/qa/speaker_turn_batch_summary.json `
  --speaker-inventory data/interim/speaker_turns/speaker_inventory.csv
```

Optional flag:

```text
--force
```

`--force` rebuilds per-document outputs even when the cache is valid.

## Non-goals

This stage does not:

* resolve a speaker label to a legislator;
* assign a stable `legislator_id`;
* infer political party or parliamentary bloc;
* assign government or opposition alignment;
* remove unresolved speakers;
* aggregate legislator-session documents;
* generate embeddings;
* calculate semantic distance;
* alter validated parser behavior to improve metadata coverage.
