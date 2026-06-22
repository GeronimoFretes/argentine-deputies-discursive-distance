# Teammate Handoff and Parallel Work Plan

## 1. Project overview

This repository studies how the discursive distance between governing and opposition blocs evolved in Argentina’s Chamber of Deputies during the recent period for which speaker identity and political alignment can be identified reliably.

The main research question is:

> How did the discursive distance between governing and opposition blocs evolve in Argentina’s Chamber of Deputies during the recent period with reliable speaker and bloc identification?

The repository and all permanent documentation, code, variable names, comments, commit messages, report text, and presentation text must be written in English.

The analysis measures semantic or discursive distance. It must not be described as a direct measurement of ideological polarization.

## 2. Fixed methodological decisions

The following decisions are already established and should not be changed without an explicit discussion and a separate methodological decision record.

### Corpus

* Candidate sessions begin on 2008-01-01.
* The candidate corpus ends at the latest available official session.
* The final starting year will be selected through data-quality analysis rather than hardcoded in advance.
* The likely reliable starting period is approximately 2010–2012, but this remains subject to speaker and metadata coverage validation.
* The source documents are official Chamber of Deputies session transcripts.

### Text units

* The source extraction unit is the speaker turn.
* The analytical text unit is the legislator-session document.
* All eligible turns by the same legislator in the same session will be aggregated.
* The final outcome is a session-level distance between governing and opposition discourse.

### Political alignment

The permitted alignment labels are:

* `government_core`
* `opposition_core`
* `ambiguous_independent`
* `excluded`

Only `government_core` and `opposition_core` are included in the primary distance estimate.

Ambiguous, independent, non-legislative, administrative, and unresolved speakers must not be forced into either analytical side.

### Text preservation

The corpus should preserve:

* grammar
* stopwords
* names
* institutions
* normal legislative language

Only structural and documentary artifacts should be removed from the speech representation.

### Distance construction

The intended primary calculation is:

1. Construct one text representation per legislator-session.
2. Compute one embedding per legislator-session.
3. Give equal weight to eligible legislators within each side.
4. Average the legislator embeddings within each side.
5. Normalize each side centroid.
6. Calculate cosine distance between the government and opposition centroids.
7. Report the result at the session level.

## 3. Current repository state

### Session discovery

The repository contains deterministic discovery of official session records.

Validated discovery results:

* 33 parliamentary periods
* 1,128 discovered records
* 1,092 records with PDFs
* 36 records without PDFs
* 412 candidate records from 2008 onward
* 398 candidate PDFs
* 14 candidate records without PDFs
* Full discovered date range from 1983-11-29 through 2026-05-20
* Candidate date range from 2008-02-22 through 2026-05-20

### PDF extraction

A resumable, column-aware PDF extraction pipeline has been implemented and validated.

Six pilot documents cover different document structures:

* `older_ordinary`
* `continuation`
* `remote_session`
* `long_recent_session`
* `informative_session`
* `latest_ordinary_special`

Repeated extraction runs reuse existing downloads and produce byte-identical outputs.

### Structural segmentation

The structural layer separates proceedings from front matter, appendices, and excluded document regions.

All six pilot documents passed exact start and end boundary audits.

### Speaker marker recognition

The parser detects explicit speaker markers, including:

* standard `Sr.` and `Sra.` forms
* multiline markers
* audited punctuation variants
* one audited missing-period variant
* dash-only separators

Validated explicit marker counts across the six pilots total 2,924.

### Speaker turns

The speaker-turn parser:

* starts a new turn at each explicit speaker marker
* carries prior attribution across markerless continuation blocks
* preserves unresolved material as unattributed
* handles procedural barriers
* preserves source references and exact offsets
* supports marker seeding from excluded procedural blocks when appropriate

### Turn content classification

Each turn can be divided losslessly into:

* `spoken_text`
* `documentary_insert`
* `stage_direction`
* `editorial_note`
* `unattributed_text`

The implementation preserves exact source coverage and provenance.

The permanent test suite currently contains 82 passing tests. Ruff and mypy also pass.

The six real pilot documents pass exact non-speech validation:

| Pilot                   | Stage spans | Editorial spans | Speech words | Coverage |
| ----------------------- | ----------: | --------------: | -----------: | -------- |
| older_ordinary          |          56 |               2 |       73,377 | PASS     |
| continuation            |          94 |              18 |       41,371 | PASS     |
| remote_session          |         107 |               5 |       90,481 | PASS     |
| long_recent_session     |         221 |               8 |      114,763 | PASS     |
| informative_session     |          46 |               0 |       38,152 | PASS     |
| latest_ordinary_special |         173 |              33 |       93,379 | PASS     |

These values count exact source spans. A single logical stage direction can occupy multiple spans when it crosses source-block boundaries.

## 4. Parallel work structure

### Main developer workstream

The main developer owns the technical critical path:

1. Integrate speaker-turn parsing into a resumable batch pipeline.
2. Persist turn and content-span outputs.
3. Add CLI commands and batch summaries.
4. Produce full-corpus speaker inventories.
5. Implement deterministic speaker normalization.
6. Connect parsed speakers to stable legislator identifiers.
7. Aggregate eligible speech into legislator-session documents.
8. Generate embeddings.
9. Calculate government-opposition semantic distance.
10. Produce robustness checks and final result tables.

### Teammate workstream

The teammate owns the political metadata, provenance, manual QA, and documentation workstream.

The teammate must not independently redesign the parser or modify the currently validated parsing modules unless a separate issue explicitly requests it.

## 5. Primary teammate assignment

The primary assignment is to create a provenance-first political metadata layer that can later connect resolved speakers to legislators, blocs, and analytical alignments.

### Required deliverables

Create the following files:

```text
docs/POLITICAL_METADATA_METHODOLOGY.md
docs/MANUAL_IDENTITY_QA_PROTOCOL.md
data/reference/README.md
data/reference/legislators.csv
data/reference/legislator_aliases.csv
data/reference/bloc_membership.csv
data/reference/bloc_alignment.csv
data/reference/sources.csv
```

A lightweight validation script may also be added:

```text
scripts/validate_reference_data.py
tests/test_reference_data.py
```

The teammate should avoid editing existing parsing files.

## 6. Required schemas

### `data/reference/legislators.csv`

Required columns:

```text
legislator_id
canonical_name
given_names
surname
province
chamber
valid_from
valid_to
source_id
review_status
notes
```

Rules:

* `legislator_id` must be stable and repository-defined.
* Dates must use ISO format `YYYY-MM-DD`.
* Names must preserve accents.
* A person serving in multiple periods retains one stable identifier.
* Every row must have a source.
* Uncertain records must use an explicit review status rather than a guessed value.

### `data/reference/legislator_aliases.csv`

Required columns:

```text
alias_raw
alias_normalized
legislator_id
valid_from
valid_to
alias_type
confidence
review_status
source_id
notes
```

Suggested `alias_type` values:

```text
official_name
transcript_surname
transcript_full_name
initials_variant
accent_variant
compound_surname_variant
manual_exception
```

Rules:

* Do not map an isolated surname when more than one eligible legislator could match it for the same date.
* Date validity must be considered.
* Ambiguous aliases must remain unresolved.
* Automated normalization and manual exceptions must be distinguishable.

### `data/reference/bloc_membership.csv`

Required columns:

```text
legislator_id
bloc_name_raw
bloc_name_normalized
valid_from
valid_to
source_id
confidence
review_status
notes
```

Rules:

* Bloc membership must be time-bounded.
* Mid-term bloc changes require separate rows.
* Overlapping membership intervals for the same legislator require explicit review.
* Raw official names must be preserved separately from normalized names.

### `data/reference/bloc_alignment.csv`

Required columns:

```text
bloc_name_normalized
valid_from
valid_to
alignment
reason
source_id
review_status
notes
```

Allowed values for `alignment`:

```text
government_core
opposition_core
ambiguous_independent
excluded
```

Rules:

* Alignment is time-dependent.
* A bloc must not be assigned based only on intuition or contemporary knowledge.
* The `reason` field must explain why the bloc is classified as core government, core opposition, ambiguous, independent, or excluded.
* Coalitional support that varies by bill should normally remain outside the two core groups unless a stable period-level classification is defensible.

### `data/reference/sources.csv`

Required columns:

```text
source_id
source_type
title
publisher
url
retrieved_at
coverage_start
coverage_end
local_snapshot
notes
```

Rules:

* Prefer official Chamber sources.
* Record retrieval dates.
* Preserve enough information to reproduce every metadata decision.
* Archive or snapshot source material when legally and technically appropriate.
* Do not use an LLM response as a factual source.

## 7. Political metadata methodology document

`docs/POLITICAL_METADATA_METHODOLOGY.md` must explain:

1. Why political alignment is time-dependent.
2. Why party, electoral alliance, parliamentary bloc, and government alignment are different concepts.
3. Which unit is used for the primary analysis.
4. How bloc changes are represented.
5. How government transitions are handled.
6. How ambiguous or independent blocs are treated.
7. How source conflicts are resolved.
8. Which decisions require manual review.
9. How the final metadata layer connects to session dates.
10. Which limitations remain.

The document should be concise, technical, and reproducible. It should not contain unsupported political interpretations.

## 8. Manual speaker identity QA protocol

`docs/MANUAL_IDENTITY_QA_PROTOCOL.md` must define how parsed speaker labels will be checked once the main developer provides a speaker inventory.

The protocol must include:

* exact matching
* normalized matching
* date-constrained surname matching
* role and office-holder exclusions
* ambiguous surname handling
* compound surname handling
* initials and accent variants
* transcript errors
* unresolved speaker preservation
* confidence levels
* required evidence for manual overrides
* reviewer and review-date fields

Suggested resolution statuses:

```text
resolved_exact
resolved_normalized
resolved_date_constrained
resolved_manual
ambiguous
non_legislator
unresolved
```

No speaker should be assigned to a legislator merely because one candidate appears plausible.

## 9. Validation requirements

The reference-data validator should check at least:

* required columns are present
* identifiers are non-empty
* ISO dates parse correctly
* `valid_from <= valid_to`
* allowed enum values are respected
* all `source_id` values exist
* all referenced `legislator_id` values exist
* exact duplicate rows are absent
* contradictory overlapping intervals are reported
* ambiguous rows are not marked as fully reviewed
* confidence values are valid and documented

The validator should fail loudly on structural errors and produce review warnings for political or historical ambiguity.

## 10. Report preparation assignment

The teammate may prepare the report structure, but must not invent or estimate final results.

Create:

```text
docs/REPORT_OUTLINE.md
docs/PRESENTATION_OUTLINE.md
docs/EVIDENCE_LEDGER.md
```

### Report outline

The report should reserve sections for:

1. Research question and motivation
2. Official source corpus
3. Session discovery
4. PDF extraction
5. Structural segmentation
6. Speaker-turn construction
7. Speech and documentary-content classification
8. Speaker identity resolution
9. Political bloc and alignment metadata
10. Legislator-session document construction
11. Embedding model
12. Distance metric
13. Data-quality threshold and selected start date
14. Main results
15. Robustness checks
16. Limitations
17. Reproducibility
18. Conclusion

Stable methodological sections may be drafted now.

The following sections must remain placeholders until the analysis is complete:

* final corpus size
* final start year
* unresolved-speaker rate
* political alignment coverage
* model-specific results
* time-series interpretation
* robustness outcomes
* substantive conclusions

### Presentation outline

Prepare an approximately 8–10 slide structure:

1. Research question
2. Why legislative discourse is difficult to measure
3. Official corpus and processing pipeline
4. Speaker and content reconstruction
5. Political alignment methodology
6. Analytical representation and distance metric
7. Data-quality validation
8. Main result
9. Robustness and limitations
10. Conclusion

Slides 8–10 must initially contain explicit result placeholders.

### Evidence ledger

`docs/EVIDENCE_LEDGER.md` should map every important report claim to:

* source file
* generated table
* validation command
* figure
* responsible script
* current status

Suggested columns:

```text
claim_id
report_section
claim
evidence_type
source_path
generation_command
status
notes
```

Allowed status values:

```text
available
pending_pipeline
pending_validation
pending_analysis
approved
```

## 11. Git workflow

The teammate should work from a current `main` branch:

```powershell
git switch main
git pull
git switch -c feat/political-metadata
```

Before making changes:

```powershell
uv sync
uv run pytest
uv run ruff check .
uv run mypy
```

Commit work in small logical checkpoints.

Suggested commits:

```text
Document political metadata methodology
Add provenance-first reference data schemas
Add reference data validation
Define manual speaker identity QA protocol
Add report and presentation scaffolds
```

Do not commit temporary downloads, exploratory files, large source snapshots, or generated outputs unless the repository conventions explicitly require them.

Before opening a pull request:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run mypy
git diff --check
git status
```

## 12. Pull request requirements

The pull request must explain:

* the scope of the metadata layer
* all files added
* all sources used
* unresolved ambiguities
* schema decisions
* validation performed
* known limitations
* what remains dependent on the speaker inventory

The pull request should not combine unrelated parsing changes.

## 13. Coordination points

The two workstreams must coordinate on three interfaces.

### Interface 1: stable legislator identifier

The teammate proposes the identifier convention. The main developer reviews and approves it before speaker-resolution code depends on it.

### Interface 2: speaker inventory

The main developer will later provide a machine-generated inventory containing at least:

```text
session_id
session_date
speaker_raw
speaker_normalized
turn_count
speech_word_count
sample_text
```

The teammate will use that inventory to expand aliases and conduct manual QA.

### Interface 3: alignment join

The main developer will join session dates and resolved legislators to:

* time-bounded bloc membership
* time-bounded bloc alignment

The teammate must provide non-overlapping, documented intervals that make this join deterministic.

## 14. Prohibited shortcuts

Do not:

* assign political alignment from general memory
* use an LLM answer as a source
* infer identity from surname alone when multiple candidates exist
* assign ambiguous blocs to government or opposition for higher coverage
* silently repair source conflicts
* remove unresolved records
* edit parser behavior merely to improve metadata coverage
* write final results before they are generated
* describe semantic distance as direct ideological polarization

## 15. Completion criteria for the teammate workstream

The first teammate pull request is complete when:

* methodology documents exist
* all reference schemas exist
* every populated metadata row has provenance
* the validator passes
* unresolved cases remain explicit
* report and presentation structures exist
* result-dependent statements remain placeholders
* no validated parser files were modified
* the branch is rebased or merged cleanly against current `main`
