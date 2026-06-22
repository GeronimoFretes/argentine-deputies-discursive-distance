# Manual Speaker Identity QA Protocol

## Purpose

This protocol defines how parsed speaker labels are checked against
`data/reference/legislators.csv` and `data/reference/legislator_aliases.csv` once the
pipeline workstream provides a speaker inventory (Interface 2 in
[`docs/TEAMMATE_HANDOFF.md`](TEAMMATE_HANDOFF.md)):

```text
session_id
session_date
speaker_raw
speaker_normalized
turn_count
speech_word_count
sample_text
```

No speaker is assigned to a legislator merely because one candidate appears plausible.
This protocol exists to make every resolution decision explicit, sourced, and
reviewable.

## Resolution statuses

Each `(speaker_raw, session_id)` pair (or, where the inventory aggregates across
sessions, each distinct `speaker_normalized` form) is assigned exactly one resolution
status:

```text
resolved_exact              - exact match to a recorded alias, unambiguous for the date
resolved_normalized         - match after normalization (case/accent/whitespace) only
resolved_date_constrained   - surname/initials match disambiguated by session date and
                               documented bloc/period membership
resolved_manual             - resolved by a human reviewer using external evidence not
                               captured by the automated steps above
ambiguous                   - multiple legislators remain plausible for this date
non_legislator               - the label belongs to a chair, secretary, executive
                               official, or other non-legislator role
unresolved                  - no candidate found; preserved as-is, not guessed
```

`ambiguous` and `unresolved` speakers are preserved in the inventory and excluded from
legislator identity, not dropped and not forced into a best guess.

## Resolution steps, in order

### 1. Exact matching

Compare `speaker_raw` against `alias_raw` in `legislator_aliases.csv` for an exact
string match. If exactly one alias matches and the session date falls inside that
alias's `[valid_from, valid_to]` interval, status is `resolved_exact`.

### 2. Normalized matching

If no exact match, compare the accent-insensitive, uppercase, punctuation-stripped
form (the same normalization already used by
[`speaker.py`](../src/argentine_deputies_discursive_distance/speaker.py)'s
`fold_text`) of `speaker_raw` against `alias_normalized`. If exactly one alias matches
and the date is within range, status is `resolved_normalized`.

### 3. Date-constrained surname matching

If the normalized label is a bare surname (or surname plus initials) and multiple
aliases share that surname, narrow candidates to those whose
`legislator_aliases.csv` interval contains the session date **and** whose
`bloc_membership.csv` shows them seated in that period. If exactly one candidate
survives, status is `resolved_date_constrained`. Per
[`docs/TEAMMATE_HANDOFF.md`](TEAMMATE_HANDOFF.md), an isolated surname is never mapped
to an alias when more than one eligible legislator could match it for the same date —
if narrowing by date and membership still leaves more than one candidate, the status is
`ambiguous`, not a guess.

### 4. Role and office-holder exclusion

Before attempting legislator resolution, check the marker's
`SpeakerLabelFamily` from `speaker.py` (already computed by the parser, not modified by
this protocol). Labels classified as `CHAIR`, `CHAMBER_SECRETARY`, or
`EXECUTIVE_OFFICIAL` are checked against whether the role is being exercised by a
sitting legislator (a deputy who is also presiding) or by a non-legislator
office-holder. If the office-holder is not a deputy for that date, status is
`non_legislator`. If the office-holder is a deputy exercising a chamber role, normal
resolution proceeds for their legislator identity, and the role is recorded in `notes`,
not modeled as a separate alignment category.

### 5. Ambiguous surname handling

A surname shared by multiple sitting legislators in the same date range, with no
disambiguating initial, compound form, or sample-text evidence, is `ambiguous`. Reviewers
must record every candidate considered in the review note, not just the rejected ones.

### 6. Compound surname handling

Spanish compound surnames (e.g. maternal/paternal combinations, hyphenated or
multi-word surnames) are recorded with `alias_type = compound_surname_variant`. A
compound surname must not be partially matched against a single-component alias for a
different legislator unless an explicit `manual_exception` alias documents that
specific transcript spelling.

### 7. Initials and accent variants

Variants that only change initials (`alias_type = initials_variant`) or accent marks
(`alias_type = accent_variant`) are linked to the same `legislator_id` as the official
name once a reviewer confirms the variant is unambiguous for its date range.

### 8. Transcript errors

Misspellings or OCR/extraction artifacts specific to one document are recorded as
`alias_type = manual_exception` with `confidence = low` or `medium` depending on how
certain the correction is, and `notes` describing the specific transcript and why the
correction is believed correct (e.g. cross-referenced against `sample_text` and a
roster). A transcript-error alias is never inferred from general knowledge of who
"should" have spoken; it requires textual or source evidence.

### 9. Unresolved speaker preservation

A speaker with no plausible candidate at all is `unresolved`. Unresolved speakers
remain in the inventory with their raw and normalized forms untouched; they are not
removed, merged into a similar-sounding legislator, or silently excluded from
inventory statistics.

## Confidence levels

`legislator_aliases.csv` confidence values (`high`/`medium`/`low`) reflect the
strength of the *alias mapping* evidence, independent of the resolution status above:

* `high`: official roster spelling, or an exact match with no ambiguity.
* `medium`: a documented variant (compound surname, initials, accent) confirmed by a
  reviewer but not itself an official primary-source spelling.
* `low`: a plausible correction of a transcript error, or a date-constrained match
  resolved from limited evidence; prioritized for follow-up QA.

## Required evidence for manual overrides

A `resolved_manual` status requires, in the review note:

1. The specific evidence used (roster excerpt, contemporaneous news report only as a
   last resort and never as the sole source for contested facts, official biography,
   etc.) with a `source_id`.
2. Why automated steps 1–3 above did not resolve it.
3. Which other candidates were considered and ruled out, and why.

An LLM response is never acceptable evidence for a manual override.

## Reviewer and review-date fields

Every manually reviewed alias or membership row records, in `notes`, the reviewer's
name (or initials) and the ISO review date, in the form:

```text
reviewed by <name> on <YYYY-MM-DD>: <summary of evidence>
```

This is a notes-field convention rather than a separate schema column, to avoid
widening the schema before it is reviewed by the pipeline workstream (Interface 1).
If a dedicated `reviewer`/`review_date` column proves necessary once real QA volume
begins, that is a schema change requiring the same review process as any other
schema change.
