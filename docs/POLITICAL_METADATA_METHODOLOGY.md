# Political Metadata Methodology

## Purpose

This document explains how the project connects a parsed parliamentary speaker to a
stable legislator identity, a time-bounded parliamentary bloc, and a time-bounded
analytical political alignment (`government_core`, `opposition_core`,
`ambiguous_independent`, `excluded`). It governs the schemas in
[`data/reference/`](../data/reference/README.md) and the validator in
[`scripts/validate_reference_data.py`](../scripts/validate_reference_data.py).

This document does not contain final corpus statistics, a selected analysis start
year, or any political alignment decision made from general background knowledge. Every
factual claim here about a specific legislator or bloc must trace to a `source_id` in
`data/reference/sources.csv`.

## 1. Why political alignment is time-dependent

A legislator's bloc, a bloc's relationship to the sitting government, and even a
bloc's name can change within a single parliamentary period. Coalitions split, blocs
rename themselves, and a bloc that supports the government in one period may oppose it
in the next. Treating alignment as a static attribute of a party label would conflate
distinct political facts and silently misclassify speech from one period using
information that only became true later (or had already stopped being true). Every
alignment fact in this project is therefore stored as a time interval
(`valid_from`/`valid_to`), and the same bloc name may have multiple
`bloc_alignment.csv` rows for non-overlapping periods.

## 2. Party, electoral alliance, parliamentary bloc, and government alignment are different concepts

* **Party**: a formally registered political organization. Not directly modeled in
  this layer; it is upstream of bloc formation.
* **Electoral alliance**: a coalition formed to contest an election. Multiple parties
  may run under one alliance and then split into separate parliamentary blocs, or one
  alliance may form one bloc.
* **Parliamentary bloc** (`bloque`): the actual organizational unit inside the Chamber
  that a legislator sits in. This is the unit recorded in `bloc_membership.csv`. A
  bloc's raw official name (`bloc_name_raw`) is preserved exactly as filed; a
  normalized name (`bloc_name_normalized`) is used as the join key against
  `bloc_alignment.csv` because the same bloc is sometimes filed under minor textual
  variants across periods.
* **Government alignment**: a classification of a bloc's relationship to the sitting
  national government during a specific interval, independent of party or alliance
  labels. This is the only concept directly used by the primary distance analysis.

Conflating these would let an electoral-alliance label or a party brand silently stand
in for a bloc's actual, period-specific governing relationship. This project keeps
them as separate columns and separate tables so that each can be sourced and revised
independently.

## 3. Which unit is used for the primary analysis

The primary analysis uses **parliamentary bloc, time-bounded** as the unit that is
classified into `government_core` / `opposition_core` / `ambiguous_independent` /
`excluded` (`bloc_alignment.csv`). A legislator inherits an analytical side only
indirectly: through their bloc membership interval (`bloc_membership.csv`) joined, by
session date, against the bloc's alignment interval. No legislator is assigned an
alignment value directly; alignment is always reached through the legislator's
documented bloc membership at the time of the session.

## 4. How bloc changes are represented

A legislator who changes blocs mid-term produces two (or more) rows in
`bloc_membership.csv`: one ending on the last day in the old bloc, one beginning on the
first day in the new bloc. Rows for the same `legislator_id` must not have overlapping
`[valid_from, valid_to]` intervals; the validator reports overlaps for manual review
rather than silently picking one.

A bloc that is renamed but is the same organizational continuity (for example, after a
internal restructuring with continuous membership) is represented as a
`bloc_name_normalized` change with adjacent, non-overlapping `bloc_alignment.csv` rows,
with the relationship documented in `notes`. The validator does not attempt to infer
bloc continuity automatically; that judgment is recorded by a human reviewer.

## 5. How government transitions are handled

A change of national government (a new president taking office, or a cabinet
realignment that changes which blocs support the executive) is represented purely
through `bloc_alignment.csv` interval boundaries. The `valid_from` of a new alignment
row is set to the date the changed relationship took effect, sourced to an official
record (inauguration date, formal bloc declaration of support or opposition, or
equivalent). The bloc's `bloc_membership.csv` rows are unaffected by a government
transition; only the bloc's `alignment` interval changes.

## 6. How ambiguous or independent blocs are treated

Blocs that:

* are genuinely independent of the government/opposition axis (e.g. blocs whose
  declared position is non-aligned for an entire interval), or
* provide coalitional support that varies bill-by-bill rather than holding a stable,
  period-level position, or
* cannot be classified with available primary evidence,

are recorded as `ambiguous_independent`. Administrative, non-partisan, or
non-legislative entries (e.g. a presiding officer role rather than a bloc) are recorded
as `excluded`. Both `ambiguous_independent` and `excluded` blocs are deliberately kept
out of the primary `government_core`/`opposition_core` comparison; they may appear in
sensitivity analyses described in [`docs/research_protocol.md`](research_protocol.md).
A bloc must never be pushed into a core category to increase analytical coverage.

## 7. How source conflicts are resolved

See the source hierarchy in
[`data/reference/README.md`](../data/reference/README.md#source-hierarchy). Official
Chamber and government sources strictly outrank secondary sources. When sources
disagree:

1. The row's facts follow the official source.
2. Both `source_id` values are still recorded (the official source as the primary
   `source_id`; the conflicting secondary source referenced in `notes`).
3. `review_status` is set to `conflicting_sources`.
4. The conflict is never resolved silently; an LLM is never used to adjudicate it.

If two official sources disagree with each other, the row is set to
`needs_manual_decision` and the conflict is described in `notes` rather than guessed.

## 8. Which decisions require manual review

* Any bloc classification into `government_core` or `opposition_core` where coalition
  support is not stable across the full interval.
* Any `legislator_id` collision candidate (two distinct people who fold to the same
  canonical identifier components).
* Any alias that could plausibly match more than one eligible legislator for the same
  date (see [`docs/MANUAL_IDENTITY_QA_PROTOCOL.md`](MANUAL_IDENTITY_QA_PROTOCOL.md)).
* Any overlapping `bloc_membership.csv` interval for the same legislator.
* Any row flagged `conflicting_sources` or `needs_manual_decision`.

## 9. How the final metadata layer connects to session dates

The pipeline workstream joins a resolved legislator's speech, for a given session, to
this layer using the session date as the join key:

1. Resolve `speaker_raw`/`speaker_normalized` (from the speaker inventory) to a
   `legislator_id` using `legislator_aliases.csv`, constrained by the session date
   falling inside the alias's `[valid_from, valid_to]` interval.
2. Find the `bloc_membership.csv` row for that `legislator_id` whose interval contains
   the session date.
3. Find the `bloc_alignment.csv` row for that bloc's `bloc_name_normalized` whose
   interval contains the session date.
4. The resulting `alignment` value (or the absence of a matching row) determines
   whether the legislator's speech for that session enters the primary
   `government_core`/`opposition_core` comparison.

This project does not provide that join code; it provides the tables and the
guarantee, enforced by the validator, that the intervals are non-overlapping and
internally consistent enough for the join to be deterministic.

## 10. Limitations

* This layer can only be as complete as the official record. Some intervals,
  especially short-lived bloc splits, may be under-documented by official sources and
  will remain `pending_research` or `reviewed_uncertain`.
* `bloc_name_normalized` continuity judgments (rename vs. genuinely new bloc) are
  manual and may be revised as better primary sources are found.
* The deterministic `legislator_id` convention depends on `surname`, `given_names`,
  and `first_known_period` remaining stable inputs; a later-discovered earlier period
  for an already-recorded legislator would require recomputing that legislator's
  identifier, which is a breaking change requiring coordinated review with the
  pipeline workstream.
* This metadata layer does not itself decide the analysis start year, and no
  alignment, membership, or source row in this layer should be read as implying one.
