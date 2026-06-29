
from __future__ import annotations

from pathlib import Path


README = """# Topic Modeling in Argentina's Chamber of Deputies

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
"""


FINAL_STATE = """# Final Project State

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
"""


SCRIPT_DESCRIPTION_OLD = (
    'description = "NLP analysis of government–opposition discursive distance '
    'in Argentina\'s Chamber of Deputies."'
)
SCRIPT_DESCRIPTION_NEW = (
    'description = "Topic modeling and temporal analysis of parliamentary '
    'discourse in Argentina\'s Chamber of Deputies."'
)

SUPERSEDED_NOTICE = (
    "> **Superseded:** This operational summary describes an abandoned "
    "government–opposition discursive-distance scope. The canonical final "
    "summary is [`FINAL_PROJECT_STATE.md`](FINAL_PROJECT_STATE.md).\n\n"
)


def write_atomic(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]

    readme_path = root / "README.md"
    final_state_path = root / "docs" / "FINAL_PROJECT_STATE.md"
    current_state_path = root / "docs" / "CURRENT_STATE.md"
    pyproject_path = root / "pyproject.toml"

    required = [current_state_path, pyproject_path]
    missing = [path for path in required if not path.is_file()]

    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing repository documentation inputs:\n" + formatted
        )

    write_atomic(readme_path, README)
    final_state_path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(final_state_path, FINAL_STATE)

    current_state = current_state_path.read_text(encoding="utf-8-sig")
    if "Superseded:" not in current_state[:500]:
        current_state = SUPERSEDED_NOTICE + current_state
        write_atomic(current_state_path, current_state)

    pyproject = pyproject_path.read_text(encoding="utf-8-sig")

    if SCRIPT_DESCRIPTION_OLD in pyproject:
        pyproject = pyproject.replace(
            SCRIPT_DESCRIPTION_OLD,
            SCRIPT_DESCRIPTION_NEW,
            1,
        )
        write_atomic(pyproject_path, pyproject)
    elif SCRIPT_DESCRIPTION_NEW not in pyproject:
        raise AssertionError(
            "Could not find either the old or new project description "
            "in pyproject.toml."
        )

    print("Final repository documentation updated successfully.")
    print(f"README: {readme_path}")
    print(f"Final state: {final_state_path}")
    print(f"Superseded notice checked: {current_state_path}")
    print(f"Project description checked: {pyproject_path}")


if __name__ == "__main__":
    main()
