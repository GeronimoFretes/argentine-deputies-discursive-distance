# Project Decision Log

This file records approved decisions that affect implementation, methodology, or collaboration.

Do not silently change an accepted decision. Proposed changes should be discussed in a GitHub issue or pull request and then recorded here.

## D001 — Repository language

Status: accepted

All permanent code, comments, documentation, schemas, commit messages, report text, and presentation text must be written in English.

## D002 — Candidate corpus period

Status: accepted

The candidate corpus begins on 2008-01-01 and ends at the latest available official session.

The final analytical starting year will be selected through data-quality validation. It must not be hardcoded before speaker and political metadata coverage are evaluated.

## D003 — Analytical units

Status: accepted

The source parsing unit is the speaker turn.

The analytical text unit is the legislator-session document.

The final outcome is session-level government-opposition semantic distance.

## D004 — Political alignment classes

Status: accepted

Allowed analytical alignment values are:

* `government_core`
* `opposition_core`
* `ambiguous_independent`
* `excluded`

Only `government_core` and `opposition_core` enter the primary estimate.

Ambiguous cases must not be forced into a core class.

## D005 — Distance construction

Status: accepted

The intended primary calculation is:

1. aggregate eligible speech by legislator and session;
2. embed each legislator-session document;
3. give equal weight to eligible legislators within each side;
4. average embeddings within each side;
5. normalize both side centroids;
6. calculate cosine distance between the centroids.

The metric is semantic or discursive distance, not direct ideological polarization.

## D006 — Text preservation

Status: accepted

Preserve grammar, stopwords, personal names, institutions, and normal legislative language.

Remove only structural, documentary, stage-direction, and editorial artifacts supported by the validated classification layer.

## D007 — Exact provenance

Status: accepted

Parsed records must preserve source page, block, offsets, attribution method, and classification information.

A logical event may be represented by multiple spans when it crosses source blocks.

## D008 — Parallel ownership

Status: accepted

The main developer owns the batch parsing and analytical pipeline.

The teammate owns the reference-data framework, provenance methodology, manual QA protocol, and report/presentation scaffolds.

Validated parser files are protected from teammate changes unless a separate issue explicitly authorizes them.

## D009 — Initial reference CSV population

Status: accepted

The first reference-data framework pull request must keep all production reference CSVs header-only.

No seed, example, synthetic, current-office-holder, or historical factual rows should be committed before the observed speaker inventory is available.

Population will be driven by speakers actually present in the processed corpus.

Temporary fixtures inside tests are allowed.

## D010 — Factual provenance

Status: accepted

Every populated factual metadata row must reference a documented source.

An LLM response is not an acceptable factual source.

Uncertainty must be represented explicitly rather than silently resolved.

## Pending decisions

The following remain pending human review:

* stable `legislator_id` format
* interval endpoint convention
* representation of unknown or open-ended dates
* overlap policy for membership intervals
* source hierarchy for conflicting records
* minimum evidence required for core political alignment
* final embedding model
* final reliable starting year
