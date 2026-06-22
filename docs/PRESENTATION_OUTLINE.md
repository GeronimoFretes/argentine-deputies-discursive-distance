# Presentation Outline

Approximately 8–10 slides. Slides 8–10 must keep explicit result placeholders until
the analysis exists; no figure or number may be drafted for them in advance.

## 1. Research question

How did the discursive distance between governing and opposition blocs evolve in
Argentina's Chamber of Deputies during the recent period with reliable speaker and
bloc identification?

## 2. Why legislative discourse is difficult to measure

Speaker attribution ambiguity, mixed procedural/documentary/spoken content within a
single transcript block, time-varying bloc membership and alignment, and the
difference between a semantic-distance metric and a direct ideological measurement.

## 3. Official corpus and processing pipeline

Diario de Sesiones source, discovery → PDF extraction → structural segmentation
pipeline stages, each validated against audited pilot documents before scaling.

## 4. Speaker and content reconstruction

Explicit speaker-marker detection, speaker-turn construction (including carried-forward
continuation and procedural barriers), and lossless content classification into
spoken/documentary/stage/editorial/unattributed spans.

## 5. Political alignment methodology

Party vs. electoral alliance vs. parliamentary bloc vs. government alignment as
distinct, time-bounded concepts; provenance-first reference tables; the
`government_core`/`opposition_core`/`ambiguous_independent`/`excluded` taxonomy and why
the latter two stay outside the primary comparison.

## 6. Analytical representation and distance metric

Legislator-session documents, equal-weighted per-legislator embeddings, normalized side
centroids, cosine distance, reported at the session level. Framed explicitly as
discursive/semantic distance, not ideological polarization.

## 7. Data-quality validation

Session eligibility criteria and the year-level quality rule used to select the
analysis start year from data rather than convention.

## 8. Main result — PLACEHOLDER

Reserved for the session-level distance series once computed. No figure exists yet.

## 9. Robustness and limitations — PLACEHOLDER

Reserved for the planned robustness analyses and their outcomes, plus the limitations
recorded in [`docs/POLITICAL_METADATA_METHODOLOGY.md`](POLITICAL_METADATA_METHODOLOGY.md)
and the pipeline workstream's own limitations.

## 10. Conclusion — PLACEHOLDER

Cannot be drafted before Slides 8–9 exist.
