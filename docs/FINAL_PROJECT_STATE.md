# Final Project State

Last updated: 2026-06-29

## Status

The final analytical scope is frozen. The primary topic model, temporal findings, robustness checks, BERTopic benchmark, evidence ledger, and presentation figures are complete.

The previous government–opposition discursive-distance objective was abandoned. This file supersedes `docs/CURRENT_STATE.md` as the canonical project summary.

## Final objective

Construct a defensible and interpretable map of substantive topics in Argentina's Chamber of Deputies between 2008 and 2025, select an appropriate topic solution, and analyze how topic prevalence evolved over time.

## Scope

Included:

- official parliamentary proceedings from 2008–2025;
- spoken parliamentary discourse;
- named legislators, role-unspecified legislators, and eligible executive officials;
- NMF model selection and interpretation;
- annual and multi-year temporal prevalence;
- document-, source-turn-, and session-weighting sensitivity;
- full-context BERTopic as a bounded benchmark.

Excluded from the final scope:

- government–opposition discursive distance;
- legislator or bloc ideology;
- political-family comparisons;
- session-type comparisons;
- causal claims;
- LDA;
- supervised topic classification.

## Final corpus

| Measure | Value |
| --- | ---: |
| Eligible official PDF records | 392 |
| Analyzable proceedings | 391 |
| Sessions | 243 |
| Source turns | 34,060 |
| Primary documents | 75,123 |
| NMF-modeled documents | 75,121 |
| Words | 15,901,236 |

One record dated 2010-03-10 contained no proceedings. Two documents were excluded from NMF because they became zero TF-IDF rows after vocabulary filtering.

## Primary model

Specification:

```text
Model: NMF
Preprocessing: P1
K: 24
Temporal estimator: equal source-turn weighting
```

P1 adds `presidenta`, `presidente`, `señor`, `señora`, and `señores` to the P0 stopword list.

K=24 was selected from K in {12, 16, 20, 24, 28}.

| Metric | K=24 |
| --- | ---: |
| Mean NPMI coherence | 0.322587 |
| Median NPMI coherence | 0.321973 |
| Topic diversity | 0.958333 |
| Mean exclusivity | 0.788844 |
| Mean redundancy | 0.063797 |
| Maximum redundancy | 0.304619 |
| Mean dominant document weight | 0.365019 |
| Mean normalized entropy | 0.569029 |

K=12 did not converge. K=28 reduced mean redundancy but performed worse on coherence, diversity, and exclusivity.

## Interpreted topic structure

Substantive or institutional topics:

```text
1  Criminal law and procedural codes
4  Justice and the judiciary
5  Provinces, Buenos Aires, and federal territory
8  Productive development and the economy
9  Rights, gender, and reproductive health
10 Labour and employment
16 Executive powers, DNU, and constitutional authority
17 Budget, expenditure, and inflation
18 Pensions and social security
19 Debt, IMF, and financial policy
20 Taxes and fiscal policy
```

Parliamentary-procedure topics:

```text
2  Article drafting
3  Committees, reports, and parliamentary business
7  Bills and initiatives
13 Bloc positions and vote explanations
14 Motions, standing orders, and agenda
15 Amendments
21 Chamber and session dynamics
22 Insertion in the official record
23 Questions of privilege
```

Rhetorical or political-positioning topics:

```text
0  General argument and confrontation
11 Identity and history
12 Government and opposition
```

Residual topic:

```text
6  Applause, reactions, and stage-direction leakage
```

The broad secondary grouping accounts for approximately 46.5% substantive, 33.5% procedure, 17.7% rhetorical, and 2.2% residual average topic weight. This grouping is a human analytical layer, not a model output.

## Locked findings

### Justice

Topic 4 reaches its strongest annual concentration in 2013.

Source-turn prevalence:

```text
2013: 6.66%
```

The same peak year appears under document and session weighting.

### Rights, gender, and reproductive health

Topic 9 reaches its highest annual source-turn prevalence in 2018 and another large concentration in 2020.

```text
2018: 9.53%
2020: 8.06%
```

The topic is broad and should not be reduced to one legislative event.

### Fiscal discourse

Both fiscal topics peak under the principal estimator in 2021.

```text
Topic 17, budget/expenditure/inflation: 6.56%
Topic 20, taxes/fiscal policy: 5.17%
```

Tax discourse remains elevated in 2022 and 2023. The period 2021–2023 should be described as a concentration, without a causal attribution.

### Pensions and social security

Topic 18 shows the clearest sustained recent increase across the predefined periods.

```text
2008–2011: 3.20%
2012–2015: 2.50%
2016–2019: 3.77%
2020–2023: 4.41%
2024–2025: 5.51%
```

Initial-to-final change:

```text
+2.30 percentage points
approximately +72% relative
```

### Questions of privilege

Topic 23 rises from 2.90% in 2008–2011 to 5.81% in 2024–2025. This is an optional procedural result and is not required in the main presentation.

## Meaning of prevalence

For every NMF-modeled document, the 24 topic weights are normalized to sum to one.

For the primary estimator:

1. chunks belonging to the same source turn are averaged;
2. every source turn receives equal weight;
3. topic weights are averaged within each year or period.

A reported value such as 6.66% therefore means that the topic's mean normalized weight among interventions was 6.66%. It does not mean that 6.66% of speeches were exclusively about that topic.

## Robustness

The main findings were checked under:

- document weighting;
- source-turn weighting;
- session weighting.

The justice 2013, rights/gender 2018, budget 2021, and tax 2021 peak years remain under all three approaches.

## BERTopic benchmark

Full-context BERTopic configuration:

```text
Embedding model: BAAI/bge-m3
Embedding backend: transformers AutoModel
Pooling: CLS token
Embedding dimension: 1024
Documents: 75,121
Truncated documents: 0
UMAP components: 5
Primary HDBSCAN: min_cluster_size=100, min_samples=10
```

Results:

| Solution | Topics | Assigned documents | Coverage | Outliers |
| --- | ---: | ---: | ---: | ---: |
| Native | 124 | 45,952 | 61.2% | 38.8% |
| Reduced | 23 | 45,952 | 61.2% | 38.8% |

The fallback HDBSCAN configuration did not satisfy the predeclared selection rule and was not selected.

Centroid-based representative documents were regenerated from the saved normalized BGE-M3 embeddings.

Final interpretation:

> Full-context BGE-M3 substantially improved native BERTopic coverage relative to the earlier truncated MiniLM benchmark. Since both the embedding model and available context changed, the improvement cannot be attributed exclusively to truncation. With 38.8% outliers, BERTopic remains an exploratory comparison and NMF remains the primary longitudinal model.

## Evidence and figures

The final evidence ledger is:

```text
docs/final_evidence_ledger.csv
```

Status:

```text
28 LOCKED
0 pending
```

The final presentation figures are stored in:

```text
docs/figures/final_presentation/
```

The figure package contains nine PNG files and a final hash inventory.

## Interpretation boundaries

Allowed:

- estimated prevalence;
- concentration;
- peak;
- increase or decrease;
- descriptive temporal association;
- robustness across weighting schemes.

Not allowed:

- causal explanation without external research;
- ideological polarization claims;
- treating soft weights as binary speech labels;
- asserting that every topic is a public-policy agenda;
- directly comparing the existing BERTopic and NMF lexical metrics;
- attributing BERTopic improvement exclusively to removal of truncation.

## Final deliverables

The final submission consists of:

- slide presentation;
- repository link;
- final figures;
- evidence ledger;
- documented NMF selection;
- BERTopic benchmark;
- optional appendix materials.

Large data and model artifacts remain outside version control.
