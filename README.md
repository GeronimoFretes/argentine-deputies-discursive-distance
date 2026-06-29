# Topic Modeling in Argentina's Chamber of Deputies

This project builds an interpretable map of recurring topics in Argentina's Chamber of Deputies between 2008 and 2025 and examines how their estimated prevalence changed over time.

## Research question

What substantive, procedural, and rhetorical topics appear in parliamentary debate, and how did their estimated prevalence evolve between 2008 and 2025?

The analysis is descriptive. Temporal peaks and trends are not interpreted as causal effects.

## Corpus

The source collection contains 392 eligible official PDF records. Proceedings were available and analyzable for 391 of them.

The final analytical corpus contains:

| Unit | Count |
| --- | ---: |
| Official PDF records | 392 |
| Analyzable proceedings | 391 |
| Legislative sessions | 243 |
| Source speaker turns | 34,060 |
| Primary modeling documents | 75,123 |
| NMF-modeled documents | 75,121 |
| Words | 15,901,236 |

Documents contain spoken parliamentary discourse only. Eligible turns were divided into non-overlapping chunks of at most 300 words, with a minimum retained length of 25 words. Two documents were excluded from NMF because their TF-IDF vectors were zero after vocabulary filtering.

## Method

The final pipeline is:

1. discover and download official session PDFs;
2. extract layout-aware page text;
3. segment structural content and speaker turns;
4. retain spoken discourse from eligible speakers;
5. split long turns into non-overlapping modeling documents;
6. apply Spanish lexical preprocessing and the P1 stopword specification;
7. fit TF-IDF and an NMF grid with K in {12, 16, 20, 24, 28};
8. select and interpret the final topic solution;
9. aggregate normalized topic weights by source turn for annual and period prevalence;
10. evaluate document- and session-weighted sensitivity;
11. compare the selected model with a full-context BERTopic benchmark.

## Selected model

The primary model is NMF with preprocessing specification P1 and K=24.

P1 extends the base Spanish stopword list with:

```text
presidenta
presidente
señor
señora
señores
```

K=24 produced the strongest overall balance among the evaluated solutions:

| Metric | K=24 |
| --- | ---: |
| Mean NPMI coherence | 0.323 |
| Topic diversity | 0.958 |
| Mean exclusivity | 0.789 |
| Mean redundancy | 0.064 |

K=28 slightly reduced mean redundancy, but coherence, diversity, and exclusivity were lower. K=12 did not converge.

## Topic structure

The interpreted 24-topic solution contains:

- 11 substantive policy or institutional topics;
- 9 parliamentary-procedure topics;
- 3 rhetorical or political-positioning topics;
- 1 residual chamber-reaction topic.

Topic labels are human interpretations based on top terms and representative documents. They are not supervised ground-truth classes.

## Main findings

The main temporal estimator gives equal weight to each source speaker turn after averaging its chunks.

- Justice and judicial discourse reaches an exceptional concentration in 2013.
- Rights, gender, and reproductive-health discourse reaches its maximum in 2018 and rises again in 2020.
- Budget and tax discourse is especially prominent from 2021 to 2023.
- Pension and social-security discourse rises from 3.20% in 2008–2011 to 5.51% in 2024–2025.

The principal peak interpretations remain visible under document-, source-turn-, and session-weighted aggregation.

Prevalence is the mean normalized NMF topic weight under the selected aggregation. It is not the percentage of speeches wholly devoted to one topic.

## BERTopic benchmark

A secondary benchmark used full-context BAAI/bge-m3 embeddings and HDBSCAN.

| Solution | Topics | Coverage | Outliers |
| --- | ---: | ---: | ---: |
| NMF P1 K=24 | 24 | 100.0% | 0.0% |
| BERTopic native | 124 | 61.2% | 38.8% |
| BERTopic reduced | 23 | 61.2% | 38.8% |

Full-context BGE-M3 substantially improved BERTopic coverage relative to the earlier truncated MiniLM benchmark. Since both the embedding model and available context changed, the improvement cannot be attributed exclusively to truncation.

BERTopic remains an exploratory semantic benchmark. NMF remains the primary model for longitudinal prevalence because it provides complete corpus coverage and soft topic weights for every modeled document.

The current BERTopic lexical coherence, exclusivity, and redundancy implementations are not compared directly with NMF because their definitions differ.

## Repository guide

Important final artifacts:

```text
docs/FINAL_PROJECT_STATE.md
docs/FINAL_SCOPE_FREEZE.md
docs/final_evidence_ledger.csv
docs/figures/final_presentation/
docs/bertopic_review/
experiments/bertopic_full_context/
scripts/build_final_evidence_ledger.py
scripts/build_final_figures.py
scripts/build_bertopic_representatives.py
scripts/finalize_bertopic_integration.py
scripts/refresh_final_figure_layouts.py
```

Large generated corpora, embeddings, matrices, and model outputs are intentionally excluded from Git.

## Validation and limitations

The main limitations are:

- topic labels require human judgment;
- NMF produces soft topic mixtures rather than binary classifications;
- small amounts of formatting and stage-direction residue remain;
- the number of sessions and interventions varies across years;
- unsupervised topic prevalence supports descriptive comparison, not causal inference;
- the BERTopic benchmark leaves 38.8% of documents unassigned.

## Legacy provenance

This repository originated as a redesign of an earlier exploratory project:

`https://github.com/fedesaroka/nlp-sesiones-diputados`

The original work was developed by Federico Saroka, Ramón Eppens, Geronimo Fretes, and Felipe Merlo. See [`docs/legacy_provenance.md`](docs/legacy_provenance.md).
