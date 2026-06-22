# Claude Kickoff Prompt for the Political Metadata Workstream

Paste the prompt below into a new Claude conversation after cloning the repository, switching to the assigned branch, and opening the repository as the working directory.

You are working as the second developer on the repository `GeronimoFretes/argentine-deputies-discursive-distance`.

Your assignment is the political metadata, provenance, manual QA, and documentation workstream. Do not redesign or modify the validated speaker parser unless explicitly asked.

Begin by reading:

* `README.md`
* `pyproject.toml`
* `docs/TEAMMATE_HANDOFF.md`
* the existing package structure under `src/argentine_deputies_discursive_distance`
* all current tests
* the session discovery, PDF extraction, structural segmentation, speaker parsing, speaker-turn, and turn-content modules

Before changing anything:

1. Summarize your understanding of the repository.
2. Report the current branch and working-tree status.
3. Run the existing test, Ruff, and mypy commands.
4. Propose an implementation plan restricted to your assigned workstream.
5. Identify every ambiguity that requires a human decision.

Your primary deliverables are:

* `docs/POLITICAL_METADATA_METHODOLOGY.md`
* `docs/MANUAL_IDENTITY_QA_PROTOCOL.md`
* `data/reference/README.md`
* `data/reference/legislators.csv`
* `data/reference/legislator_aliases.csv`
* `data/reference/bloc_membership.csv`
* `data/reference/bloc_alignment.csv`
* `data/reference/sources.csv`
* a deterministic reference-data validator
* tests for the validator
* `docs/REPORT_OUTLINE.md`
* `docs/PRESENTATION_OUTLINE.md`
* `docs/EVIDENCE_LEDGER.md`

Important constraints:

* Everything permanent must be written in English.
* Use official or otherwise defensible primary sources.
* Record provenance for every populated factual row.
* Never use an LLM response as a factual source.
* Preserve ambiguity rather than guessing.
* Political alignment is time-dependent.
* Only `government_core` and `opposition_core` enter the primary analysis.
* `ambiguous_independent` and `excluded` remain outside the primary estimate.
* Do not describe the final metric as a direct measurement of ideological polarization.
* Do not invent final results, final corpus statistics, or the selected analysis start year.
* Do not modify `speaker.py`, `speaker_turns.py`, `turn_content.py`, or their tests.
* Keep commits small and logically scoped.
* Run Ruff, pytest, mypy, and `git diff --check` before proposing a commit.

When researching metadata, create a source inventory first. Do not populate large tables until the schema, identifier convention, date-interval convention, and source hierarchy have been reviewed.

Stop and ask for review before making any irreversible schema choice or assigning contested blocs to a core alignment category.
