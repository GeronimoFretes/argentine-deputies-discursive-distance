# Parallel Work Protocol

## Purpose

This protocol defines how two developers work concurrently on the repository without duplicating work, creating avoidable merge conflicts, or weakening methodological traceability.

## Branch ownership

Each active feature branch has exactly one owner.

A developer must not push commits to another developer’s feature branch unless both developers explicitly agree beforehand.

All permanent changes enter `main` through a pull request.

Direct commits to `main` are prohibited.

## Current workstreams

### Pipeline workstream

Owner: main developer

Primary scope:

* batch speaker-turn processing
* content-span persistence
* corpus-level QA summaries
* speaker inventory generation
* speaker normalization and identity-resolution code
* legislator-session document construction
* embeddings
* session-level discursive-distance calculation
* robustness analysis

Primary owned paths:

```text
src/argentine_deputies_discursive_distance/
tests/
config/
data/qa/
```

The pipeline owner should avoid editing teammate-owned political metadata files except when reviewing or integrating an approved schema.

### Political metadata and documentation workstream

Owner: teammate

Primary scope:

* political metadata methodology
* provenance system
* legislator reference schema
* speaker alias schema
* time-bounded bloc membership
* time-bounded analytical alignment
* metadata validation
* manual identity QA protocol
* report outline
* presentation outline
* evidence ledger

Primary owned paths:

```text
data/reference/
docs/POLITICAL_METADATA_METHODOLOGY.md
docs/MANUAL_IDENTITY_QA_PROTOCOL.md
docs/REPORT_OUTLINE.md
docs/PRESENTATION_OUTLINE.md
docs/EVIDENCE_LEDGER.md
scripts/validate_reference_data.py
tests/test_reference_data.py
```

The teammate must not modify the validated parser modules or their tests without a separate issue and explicit approval.

## Protected parser files

The following files must not be changed by the political metadata workstream:

```text
src/argentine_deputies_discursive_distance/speaker.py
src/argentine_deputies_discursive_distance/speaker_turns.py
src/argentine_deputies_discursive_distance/turn_content.py
tests/test_speaker.py
tests/test_speaker_turns.py
tests/test_turn_content.py
```

## Shared files

The following files may affect both workstreams and require coordination before editing:

```text
README.md
pyproject.toml
uv.lock
src/argentine_deputies_discursive_distance/cli.py
.gitignore
```

Before changing a shared file, the developer must leave a comment on the relevant GitHub issue stating:

* which file will change
* why the change is necessary
* which branch will own the change
* whether the other workstream depends on it

Only one branch should modify a shared file at a time.

## Interface contracts

### Legislator identifier

The political metadata workstream proposes the stable `legislator_id` convention.

The pipeline workstream must review and approve it before identity-resolution code depends on it.

The convention must remain stable across parliamentary periods.

### Speaker inventory

The pipeline workstream will generate an inventory containing at least:

```text
session_id
session_date
speaker_raw
speaker_normalized
turn_count
speech_word_count
sample_text
```

The political metadata workstream will use this inventory to:

* identify observed legislators
* create aliases
* prioritize manual QA
* avoid researching unused historical records

### Political alignment join

The political metadata workstream must provide deterministic, time-bounded tables for:

* legislator-to-bloc membership
* bloc-to-analytical-alignment classification

The pipeline must join these tables using the session date.

Overlapping or contradictory intervals must fail validation or be explicitly marked for review.

## Synchronization procedure

At the start of every work session:

```powershell
git fetch origin
git switch <owned-branch>
git status
git pull --ff-only origin <owned-branch>
```

Before incorporating changes from `main`:

```powershell
git fetch origin
git switch <owned-branch>
git merge origin/main
```

Feature branches should use normal merge synchronization rather than force-pushing rewritten history.

When a conflict occurs, the developer must inspect both versions. An LLM must not resolve a semantic conflict without human review.

## Commit requirements

Commits should be small and logically scoped.

Each commit should represent one coherent change, such as:

* add a schema
* add validation
* document a methodology
* add one pipeline stage
* add tests for one behavior

Generated data, temporary audits, downloaded source files, and exploratory notebooks must not be committed unless repository policy explicitly requires them.

## Pre-push checks

Before every push:

```powershell
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run mypy
git diff --check
git status
```

## Pull request requirements

Every pull request must describe:

* purpose
* files changed
* methodological decisions
* tests and validation performed
* unresolved questions
* downstream dependencies
* known limitations

A pull request must not combine unrelated pipeline, metadata, and presentation changes.

## LLM usage rules

An LLM may assist with:

* code generation
* schema design
* test drafting
* documentation drafting
* code review
* source summarization

An LLM response is not a factual source.

Political affiliation, bloc membership, office-holder status, dates, and historical facts require external provenance.

The LLM must not:

* invent sources
* infer contested political alignment without evidence
* silently resolve ambiguous identities
* rewrite validated parser behavior to increase metadata coverage
* invent final analytical results

## Merge order

The expected merge sequence is:

1. coordination documentation
2. political metadata framework
3. batch speaker-turn pipeline
4. speaker inventory
5. populated observed-speaker metadata
6. identity-resolution pipeline
7. legislator-session documents
8. embeddings and distance calculation
9. report and presentation results

This order may change only when the interface dependencies remain explicit and validated.
