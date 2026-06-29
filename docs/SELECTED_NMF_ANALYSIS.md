# Selected NMF Analysis

This stage assigns topic weights with the selected fitted NMF model. It loads the
production P1 grid artifacts, validates their manifest-recorded hashes and byte
sizes, applies the saved vectorizer to `cleaned_primary_documents.jsonl`, excludes
only the exact zero TF-IDF ledger rows, and calls the saved NMF model's
`transform()` method. It does not refit the vectorizer or NMF model.

Default command:

```powershell
uv run deputies-distance analyze-selected-nmf
```

Equivalent explicit production command:

```powershell
uv run deputies-distance analyze-selected-nmf --grid-input-dir data/qa/topic_modeling/nmf_grid_v1 --config config/topic_modeling/selected_nmf_k024_v1.json --output-dir data/qa/topic_modeling/selected_nmf_k024_v1
```

Use `--force` only to overwrite an existing nonempty output directory.

## Weight Semantics

Document-topic rows are inferred selected-model weights using the fixed fitted
topic-term matrix. Each nonzero document row is normalized to sum to one and stored
as float32 in `document_topic_weights.npz`.

Aggregation definitions:

- Document weighting: average normalized document-topic vectors directly.
- Source-turn weighting: average chunks within each `(source_record_id, turn_index)`,
  then average source turns equally within each year or period.
- Session weighting: average source-turn vectors within each `source_record_id`,
  then average sessions equally within each year or period.

The source-turn specification is the main analysis. Session weighting starts from
source-turn vectors and does not revert to direct document weighting inside
sessions.

## Outputs

The output directory contains:

- `selected_model_manifest.json`
- `selected_model_report.md`
- `topic_metadata.csv`
- `document_topic_weights.npz`
- `document_topic_metadata.csv`
- `document_topic_assignments.csv`
- `source_turn_topic_weights.npz`
- `source_turn_metadata.csv`
- `session_topic_weights.npz`
- `session_metadata.csv`
- `annual_topic_prevalence.csv`
- `period_topic_prevalence.csv`
- `temporal_denominators.csv`
- `topic_change_summary.csv`
- `grid_prevalence_comparison.csv`

The two zero TF-IDF documents remain primary corpus documents in denominator
outputs, but they do not receive modelled topic vectors.
