# Final Project Scope Freeze

**Project:** Topic modelling of Argentine parliamentary discourse
**Corpus:** Cámara de Diputados de la Nación Argentina
**Period:** 2008–2025
**Freeze status:** Final analytical scope locked
**Deadline:** June 29, 2026, 07:00 — recuperatorio instance

## 1. Final research question

What thematic structures emerge from substantive spoken discourse in Argentina’s Chamber of Deputies, and how did their prevalence evolve between 2008 and 2025?

The analysis aims to produce:

1. a defensible and interpretable map of parliamentary topics;
2. a justified topic-model specification;
3. a temporal analysis of topic prevalence;
4. representative textual evidence;
5. robustness and sensitivity checks.

The project describes thematic prevalence in parliamentary discourse. It does not directly measure ideology, polarization, political distance or causal effects.

## 2. Relationship to the original proposal

The original proposal included:

* topic discovery;
* temporal analysis;
* comparisons by session type;
* comparisons by political family;
* LDA;
* NMF;
* BERTopic.

The final project retains the central approved objective of topic discovery and temporal evolution.

The following extensions were excluded after a feasibility and methodological audit:

* political-family comparisons;
* government–opposition distance;
* session-type comparison;
* LDA.

This is a deliberate scope refinement. The final project prioritizes depth, evaluation and interpretability over executing every exploratory idea from the preliminary proposal.

## 3. Final corpus

The corpus is constructed from official Chamber of Deputies proceedings.

Locked corpus facts:

* 392 eligible official PDFs were identified.
* 391 PDFs contain analysable proceedings.
* One document corresponds to a session that did not proceed because of the absence of deputies.
* The final temporal window is 2008–2025.
* The primary temporal universe contains 243 legislative-debate sessions.
* The modelling corpus contains 75,123 primary documents.
* These documents correspond to 34,060 retained source turns.
* The primary corpus contains approximately 15.9 million words.
* Two documents produce zero TF-IDF vectors under the final P1 representation.
* The final NMF model therefore contains 75,121 modelled documents.

The unit hierarchy is:

`session → source turn → non-overlapping modelling document`

Modelling documents are bounded chunks of no more than 300 whitespace-delimited words. Separate source turns and speakers are never combined.

## 4. Final textual estimand

The primary estimand is substantive spoken parliamentary discourse reconstructed from speaker turns.

The primary corpus includes spoken text associated with:

* named or role-unspecified speakers;
* executive officials.

The primary corpus excludes, where identified:

* documentary agenda material;
* complete inserted bills;
* reports and annexes;
* stage directions;
* editorial notes;
* chair and secretarial interventions;
* anonymous or unresolved speaker families;
* source turns below the minimum word threshold.

A small amount of residual procedural or stage-direction language remains visible in some latent components and is reported as a limitation.

## 5. Final preprocessing specification

The selected preprocessing policy is P1.

P1 consists of:

* the frozen Spanish P0 stopword list;
* plus `señor`;
* `señora`;
* `señores`;
* `presidente`;
* `presidenta`.

Text preparation includes:

* soft-hyphen repair;
* explicit line-break hyphen repair;
* Unicode NFKC normalization;
* case folding;
* alphabetic lexical tokens of at least three characters;
* accents and `ñ` preserved;
* unigrams and bigrams;
* no stemming;
* no lemmatization.

The contextual embeddings used in the separate BERTopic benchmark receive natural `modeling_text`; P1 applies to BERTopic’s lexical topic representation and common lexical metrics.

## 6. Final primary model

The primary model is:

* Non-negative Matrix Factorization;
* TF-IDF representation;
* P1 preprocessing;
* K = 24;
* coordinate-descent solver;
* Frobenius loss;
* NNDSVDa initialization;
* random seed 42.

The bounded model grid evaluated:

* K = 12;
* K = 16;
* K = 20;
* K = 24;
* K = 28.

K = 24 is selected because it provides the strongest overall quality–granularity tradeoff:

* highest mean NPMI coherence in the grid;
* highest top-term diversity;
* highest mean exclusivity;
* acceptable redundancy;
* interpretable substantive and institutional structure.

K = 28 slightly reduces redundancy but loses coherence. K = 12 did not converge.

## 7. P0–P1 sensitivity decision

The final representation remains P1.

At K = 24:

| Metric               |     P0 |     P1 |
| -------------------- | -----: | -----: |
| Mean NPMI            | 0.2887 | 0.3226 |
| Median NPMI          | 0.2723 | 0.3220 |
| Topic diversity      | 0.9417 | 0.9583 |
| Mean exclusivity     | 0.7632 | 0.7888 |
| Mean redundancy      | 0.0562 | 0.0638 |
| Maximum redundancy   | 0.2993 | 0.3046 |
| Mean dominant weight | 0.3439 | 0.3650 |
| Normalized entropy   | 0.5789 | 0.5690 |

P1 improves coherence, diversity, exclusivity and assignment sharpness. The small redundancy increase does not outweigh those gains.

The second zero-vector document under P1 contains only a non-substantive parliamentary formula and its exclusion is expected.

## 8. Final temporal estimator

Each modelled document receives a normalized mixture of 24 topic weights.

The primary temporal estimator is source-turn weighting:

1. average the document chunks belonging to each source turn;
2. give each retained source turn equal weight within the relevant year or period;
3. average its normalized topic vector.

The primary reported values are therefore estimated average topic weights among source turns.

They are not:

* percentages of speeches exclusively assigned to a topic;
* percentages of words;
* percentages of legislators;
* binary topic classifications.

Document weighting and session weighting are retained as robustness specifications.

The main conclusions must remain substantively consistent across all three aggregation schemes.

## 9. Final working topic taxonomy

### Substantive policy and institutional topics

* Topic 1 — Criminal law and procedural codes
* Topic 4 — Justice and the judiciary
* Topic 5 — Provinces, Buenos Aires and federal territory
* Topic 8 — Productive development and the economy
* Topic 9 — Rights, gender and reproductive health
* Topic 10 — Labour, employment and working conditions
* Topic 16 — Executive powers, emergency decrees and constitutional authority
* Topic 17 — Budget, public expenditure and inflation
* Topic 18 — Pensions and the social-security system
* Topic 19 — Debt, the IMF and financial policy
* Topic 20 — Taxes and fiscal policy

### Parliamentary-procedure topics

* Topic 2 — Article-by-article drafting
* Topic 3 — Committees, reports and parliamentary business
* Topic 7 — Bills and legislative initiatives
* Topic 13 — Bloc positions and explanations of votes
* Topic 14 — Motions, standing orders and agenda management
* Topic 15 — Amendments to bill text
* Topic 21 — Chamber and session dynamics
* Topic 22 — Insertion of speeches into the official record
* Topic 23 — Questions of privilege

### Broad political or rhetorical dimensions

* Topic 0 — General argumentation and confrontation
* Topic 11 — Political identity and partisan historical memory
* Topic 12 — Government–opposition positioning

### Residual component

* Topic 6 — Applause and reactions in the chamber

Topics 12 and 15 contain coherent content but are partly contaminated by recurrent broken-word extraction patterns.

## 10. Locked primary findings

### Finding 1 — Justice and the judiciary

Topic 4 shows an exceptional concentration in 2013.

Source-turn-weighted prevalence:

* 2013: 6.66%.

The maximum occurs in 2013 under document, source-turn and session weighting.

Approved wording:

> Justice and judicial discourse shows an exceptional concentration in 2013, robust to the choice of temporal aggregation.

### Finding 2 — Rights, gender and reproductive health

Topic 9 reaches its maximum in 2018 and rises again in 2020.

Source-turn-weighted prevalence:

* 2018: 9.53%;
* 2020: 8.06%.

The 2018 maximum occurs under all three aggregation methods.

Approved wording:

> Rights, gender and reproductive-health discourse shows its largest concentration in 2018 and another pronounced rise in 2020.

The topic must not be labelled only as abortion because it also includes women’s rights, human rights, health, gender and violence.

### Finding 3 — Fiscal prominence

Topic 17, budget, public expenditure and inflation:

* 2021 source-turn prevalence: 6.56%.

Topic 20, taxes and fiscal policy:

* 2021 source-turn prevalence: 5.17%;
* remains elevated in 2022 and 2023.

Both topics peak in 2021 under all three aggregation schemes.

Approved wording:

> Fiscal discourse becomes particularly prominent between 2021 and 2023, with both budget and tax components reaching their maximum in 2021.

### Finding 4 — Pensions and social security

Topic 18 shows the strongest sustained recent increase.

Source-turn-weighted period prevalence:

* 2008–2011: 3.20%;
* 2012–2015: 2.50%;
* 2016–2019: 3.77%;
* 2020–2023: 4.41%;
* 2024–2025: 5.51%.

Change between the first and final periods:

* +2.30 percentage points;
* approximately +72% relative to the initial period.

Approved wording:

> Pension and social-security discourse becomes persistently more prominent in the later years.

The long-period increase is more robust than selecting one exact annual maximum.

### Finding 5 — Questions of privilege

Topic 23 is a procedural finding.

Source-turn-weighted period prevalence:

* 2008–2011: 2.90%;
* 2024–2025: 5.81%.

Approved wording:

> The model also identifies a strong recent increase in procedural discourse, particularly questions of privilege.

This finding must not be presented as a public-policy agenda.

## 11. BERTopic’s bounded role

BERTopic is a supporting methodological benchmark.

The corrected benchmark must:

* use the exact same 75,121 documents;
* use full-context BGE-M3 embeddings;
* produce zero truncated documents;
* keep primary and fallback HDBSCAN results separate;
* select the fallback only under the predeclared rule;
* report native outliers;
* reduce only the selected native result toward 24 topics;
* use the exact P1 lexical representation for topic terms and common metrics.

BERTopic may affect only:

* the model-comparison section;
* one presentation slide;
* one conclusion sentence;
* the benchmark section of the README;
* technical appendix material.

BERTopic may not change:

* the research question;
* the selected NMF model;
* the temporal estimator;
* the four primary substantive findings;
* the project’s central narrative.

Interpretation gates:

* outliers above 45% — unsuitable for prevalence analysis;
* outliers between 30% and 45% — exploratory only;
* outliers below 30% — viable supporting comparison;
* high redundancy or incoherent topics — reject regardless of outlier rate.

## 12. Final limitations

The final project acknowledges:

1. Topic names are human interpretations based on top terms and representative documents.
2. Topic weights are soft mixtures rather than binary classifications.
3. A small residual stage-direction component remains.
4. Some components contain broken-word extraction patterns.
5. The number of sessions and interventions varies by year.
6. Historical events help interpret topic peaks but the model does not establish causality.
7. The project does not estimate ideological position or polarization.
8. Political-family and session-type comparisons were excluded from the final scope.

## 13. No-go work after the freeze

The following work is prohibited unless a critical error is found:

* LDA;
* additional K values;
* additional stopword variants;
* additional embedding models;
* UMAP or HDBSCAN grids;
* political-family analysis;
* government–opposition distance;
* session-type analysis;
* causal analysis;
* new parser features;
* changes to the modelling corpus;
* refitting NMF;
* relabelling headline topics without representative-document evidence;
* adding new headline findings.

## 14. Change-control rule

A frozen decision may be reopened only if one of the following occurs:

1. a source artifact is shown to be corrupted;
2. a reported number fails deterministic reproduction;
3. the BERTopic corpus does not match the locked NMF document IDs;
4. a topic label is contradicted by its representative documents;
5. a major claim changes direction under source-turn and session weighting;
6. an instructor requirement has been omitted.

A visual preference, a new speculative interpretation or an alternative model result is not sufficient reason to reopen the scope.

## 15. Final project thesis

The final presentation is built around this thesis:

> A validated NLP pipeline can recover an interpretable map of substantive and institutional themes from heterogeneous Argentine parliamentary proceedings. NMF identifies both policy agendas and the procedural language of lawmaking, while temporal prevalence reveals event-shaped concentrations and sustained changes between 2008 and 2025.
