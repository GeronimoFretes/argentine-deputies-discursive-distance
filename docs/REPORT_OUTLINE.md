# Report Outline

This outline reserves the report's section structure. Stable methodological sections
are drafted now. Sections marked **PLACEHOLDER** must not be filled in until the
analysis that produces them has actually run; this document does not anticipate or
estimate any result.

## 1. Research question and motivation

Drafted from [`README.md`](../README.md) and
[`docs/research_protocol.md`](research_protocol.md): how discursive distance between
government and opposition blocs evolved during the period with reliable speaker and
bloc identification, and why this redesign was necessary (see
[`docs/legacy_provenance.md`](legacy_provenance.md)).

## 2. Official source corpus

Describes the official Diario de Sesiones source, the candidate window (2008-01-01
onward), and why an arbitrary starting year is not used (see
[`docs/research_protocol.md`](research_protocol.md)).

## 3. Session discovery

Drafted from the validated discovery results recorded in
[`docs/TEAMMATE_HANDOFF.md`](TEAMMATE_HANDOFF.md) (period count, record counts, PDF
availability, candidate date range as of the cited validation run). Any updated counts
from a later discovery run must cite the regenerated summary file, not this document.

## 4. PDF extraction

Drafted from the validated, resumable, column-aware extraction pipeline and its six
audited pilot documents.

## 5. Structural segmentation

Drafted from the validated structural-segmentation boundary audits across the six
pilots.

## 6. Speaker-turn construction

Drafted from the validated speaker-marker detection and speaker-turn parsing behavior
in [`speaker.py`](../src/argentine_deputies_discursive_distance/speaker.py) and
[`speaker_turns.py`](../src/argentine_deputies_discursive_distance/speaker_turns.py).

## 7. Speech and documentary-content classification

Drafted from the validated lossless classification behavior in
[`turn_content.py`](../src/argentine_deputies_discursive_distance/turn_content.py) and
its exact non-speech validation results across the six pilots.

## 8. Speaker identity resolution

Methodology drafted from
[`docs/MANUAL_IDENTITY_QA_PROTOCOL.md`](MANUAL_IDENTITY_QA_PROTOCOL.md). **PLACEHOLDER**
for: unresolved-speaker rate, resolution-status distribution, and any manual-override
summary, once the speaker inventory and QA pass exist.

## 9. Political bloc and alignment metadata

Methodology drafted from
[`docs/POLITICAL_METADATA_METHODOLOGY.md`](POLITICAL_METADATA_METHODOLOGY.md) and the
schemas in [`data/reference/README.md`](../data/reference/README.md). **PLACEHOLDER**
for: political alignment coverage (share of speech mapped to a core side), and counts
of `ambiguous_independent`/`excluded` material excluded from the primary comparison.

## 10. Legislator-session document construction

Drafted from [`docs/TEAMMATE_HANDOFF.md`](TEAMMATE_HANDOFF.md) §2 ("Text units"): one
text representation per legislator per session, aggregating all eligible turns. Final
aggregation rules (e.g. whether and how `documentary_insert` content is fully excluded)
are owned by the pipeline workstream and cited here once implemented.

## 11. Embedding model

**PLACEHOLDER.** Reserved for the embedding model selected by the pipeline workstream,
its provenance, and any preprocessing applied specifically for embedding input (per
[`docs/research_protocol.md`](research_protocol.md), preprocessing preserves grammar,
stopwords, and names; only structural/documentary artifacts are removed).

## 12. Distance metric

Drafted from [`docs/TEAMMATE_HANDOFF.md`](TEAMMATE_HANDOFF.md) §2 ("Distance
construction"): equal-weighted per-legislator embeddings, averaged and normalized per
side, cosine distance between side centroids, reported at the session level. This
report must describe the result as semantic/discursive distance, never as a direct
measurement of ideological polarization.

## 13. Data-quality threshold and selected start date

Methodology drafted from [`docs/research_protocol.md`](research_protocol.md) (session
eligibility rules, year-level quality rule). **PLACEHOLDER** for the selected start
year itself and the year-by-year quality metrics that produced it.

## 14. Main results — PLACEHOLDER

No result, trend, or directional claim may be written here before the corresponding
analysis has run and its evidence is recorded in
[`docs/EVIDENCE_LEDGER.md`](EVIDENCE_LEDGER.md).

## 15. Robustness checks — PLACEHOLDER

Reserved for the planned robustness analyses listed in
[`docs/research_protocol.md`](research_protocol.md) (lexical divergence, strict/broad
alignment definitions, alternative participation thresholds, informative-session
sensitivity, leave-one-legislator-out, start-year sensitivity).

## 16. Limitations

Drafted limitations from this workstream are listed in
[`docs/POLITICAL_METADATA_METHODOLOGY.md`](POLITICAL_METADATA_METHODOLOGY.md) §10.
**PLACEHOLDER** for pipeline-side limitations (extraction failure modes, embedding
truncation, etc.) once that workstream documents them.

## 17. Reproducibility

Drafted from the project's deterministic, resumable design: discovery, extraction, and
segmentation are designed to be byte-identical on repeated runs; reference-data
validation is deterministic
([`scripts/validate_reference_data.py`](../scripts/validate_reference_data.py)).
**PLACEHOLDER** for the final reproducibility statement covering the embedding and
distance-calculation stages.

## 18. Conclusion — PLACEHOLDER

Cannot be drafted before Sections 14–15 exist.
