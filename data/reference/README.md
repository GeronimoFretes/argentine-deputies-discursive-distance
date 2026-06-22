# Reference Data: Political Metadata Layer

This directory holds the provenance-first reference tables that will later connect
parsed parliamentary speakers to legislators, time-bounded parliamentary blocs, and
time-bounded analytical political alignment.

These tables are populated manually and incrementally, driven by the speaker
inventory the pipeline workstream will provide (see
[`docs/TEAMMATE_HANDOFF.md`](../../docs/TEAMMATE_HANDOFF.md), Interface 2). Until that
inventory exists, the files below contain headers only. No row may be added without a
documented source.

See [`docs/POLITICAL_METADATA_METHODOLOGY.md`](../../docs/POLITICAL_METADATA_METHODOLOGY.md)
for the reasoning behind these conventions, and
[`docs/MANUAL_IDENTITY_QA_PROTOCOL.md`](../../docs/MANUAL_IDENTITY_QA_PROTOCOL.md) for how
parsed speaker labels will be checked against `legislators.csv` and
`legislator_aliases.csv`.

## Files

| File | Purpose |
| --- | --- |
| `legislators.csv` | One stable identity per person who has served as a deputy. |
| `legislator_aliases.csv` | Transcript spellings and name variants mapped to a `legislator_id`, time-bounded. |
| `bloc_membership.csv` | Time-bounded parliamentary bloc membership per legislator. |
| `bloc_alignment.csv` | Time-bounded classification of each bloc into `government_core`, `opposition_core`, `ambiguous_independent`, or `excluded`. |
| `sources.csv` | Provenance registry. Every `source_id` referenced elsewhere must appear here exactly once. |

## Identifier convention

### `legislator_id`

`legislator_id` is a deterministic, repository-computed identifier, not an externally
assigned number. It mirrors the convention already used for
`source_record_id` in
[`identifiers.py`](../../src/argentine_deputies_discursive_distance/identifiers.py):

```text
canonical = (
    "legislator|"
    f"surname={fold_for_matching(surname)}|"
    f"given_names={fold_for_matching(given_names)}|"
    f"first_known_period={first_known_period}"
)
legislator_id = sha256(canonical.encode("utf-8")).hexdigest()[:20]
```

* `surname` and `given_names` are folded with the same accent-insensitive,
  punctuation-stripped uppercase normalization used elsewhere in the pipeline
  (`fold_for_matching` / `fold_text`).
* `first_known_period` is the chamber period number (an integer, matching
  `SessionManifestRecord.period`) in which the legislator is first documented as
  seated. It is a stabilizing fact, not a guess: it must be recorded with its own
  `source_id`.
* A person who serves across multiple non-contiguous periods keeps exactly one
  `legislator_id`, computed once from their first documented period. Later periods are
  represented as additional `bloc_membership.csv` rows, not new identities.
* This convention is a **proposal** pending review by the pipeline workstream owner
  before identity-resolution code depends on it (Interface 1 in
  `docs/TEAMMATE_HANDOFF.md`). It must not be changed unilaterally once approved.

### `source_id`

`source_id` values are short, stable, human-assigned slugs (for example
`hcdn-diputados-actuales-2024`), not hashes. They are assigned once in `sources.csv`
and never reused for a different source.

## Date-interval convention

* All dates use ISO 8601 (`YYYY-MM-DD`).
* `valid_from` is always required and represents the first day the fact is known to
  hold.
* `valid_to` represents the last day the fact is known to hold. **An empty string in
  `valid_to` means the interval is open-ended** (still in effect as of the last
  review), not unknown. Do not use a sentinel date such as `9999-12-31`.
* When `valid_to` is present, `valid_from <= valid_to` must hold.
* Intervals are treated as closed (`[valid_from, valid_to]` inclusive) for overlap
  checks.
* A change mid-period (for example, a legislator leaving one bloc and joining another)
  is represented as two adjacent, non-overlapping rows, not as one row with a gap.

## Source hierarchy

When sources conflict, official Chamber of Deputies and other official government
sources (Boletín Oficial, official bloc communications filed with the Chamber)
**strictly outrank** secondary sources (academic literature, journalism, archival
aggregators).

* A secondary source may be used to fill a gap only when no official source covers the
  fact.
* A secondary source must never silently override an official source. If a secondary
  source contradicts an official source, both are recorded with their respective
  `source_id` values, the row uses the official source's facts, and `review_status` is
  set to a value indicating the conflict (see
  `docs/POLITICAL_METADATA_METHODOLOGY.md`), with the contradiction described in
  `notes`.
* An LLM response is never a source. `source_type` has no value representing an LLM or
  other generative output.

## Controlled vocabularies

These values are enforced by `scripts/validate_reference_data.py` and must stay in
sync with that script.

### `review_status` (legislators, aliases, bloc_membership, bloc_alignment)

```text
pending_research        - no review has been performed yet
reviewed_confident       - reviewed against an official source with no contradiction
reviewed_uncertain       - reviewed, but the available evidence is incomplete or weak
conflicting_sources      - official and secondary sources disagree; unresolved
needs_manual_decision    - requires a human judgment call beyond source lookup
```

A row with `alignment = ambiguous_independent` (in `bloc_alignment.csv`) must not use
`reviewed_confident`: by construction, an ambiguous classification reflects unresolved
or contested evidence, even when the supporting sources themselves are solid.

### `confidence` (aliases, bloc_membership)

```text
high     - exact, unambiguous evidence (e.g. an official roster entry)
medium   - reasonable inference with minor uncertainty (e.g. a documented compound surname)
low      - plausible but weakly supported; should be prioritized for manual QA
```

### `alias_type` (aliases)

```text
official_name
transcript_surname
transcript_full_name
initials_variant
accent_variant
compound_surname_variant
manual_exception
```

### `alignment` (bloc_alignment)

```text
government_core
opposition_core
ambiguous_independent
excluded
```

### `source_type` (sources)

```text
official_chamber_record       - HCDN roster, session records, bloc filings
official_government_record    - Boletín Oficial, executive-branch official records
secondary_academic            - peer-reviewed or academic secondary literature
secondary_journalistic        - news reporting used only to fill documented gaps
archival_snapshot             - a preserved copy of an official or secondary source
other                          - any source type not covered above; must be justified in notes
```

## Provenance requirement

Every populated row in every reference table must carry a `source_id` that resolves to
a row in `sources.csv`. Rows describing a stabilizing fact used inside an identifier
(such as `first_known_period`) must also be traceable to a `source_id`, recorded in
`legislators.csv` or `notes`.

## What this directory does not contain

* No final list of which sessions or years enter the primary analysis.
* No assignment of contested coalition blocs to `government_core` or
  `opposition_core` without a prior, explicit review (see
  `docs/POLITICAL_METADATA_METHODOLOGY.md`, "Decisions requiring manual review").
* No row sourced from an LLM response.
