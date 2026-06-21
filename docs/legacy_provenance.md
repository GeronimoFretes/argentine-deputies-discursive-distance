# Legacy Project Provenance

This project is a methodological redesign of an earlier exploratory project on topic modelling in Argentina's Chamber of Deputies.

## Legacy repository

Repository
`https://github.com/fedesaroka/nlp-sesiones-diputados`

Frozen reference commit
`<LEGACY_COMMIT_SHA>`

## Original team

The original data-collection and exploratory topic-modelling work was developed by:

* Federico Saroka
* Ramón Eppens
* Geronimo Fretes
* Felipe Merlo

## Relationship with this repository

The legacy project constructed the initial parliamentary-session corpus and explored LDA, NMF, and BERTopic models. Its results exposed important methodological limitations involving document granularity, speaker segmentation, destructive preprocessing, embedding truncation, and model evaluation.

Those findings motivated the present redesign.

This repository does not treat the legacy BERTopic model or its processed chunks as valid inputs for the final analysis. Instead, it rebuilds the analytical corpus using page-aware extraction, intervention-level parsing, conservative speaker resolution, model-specific preprocessing, and explicit quality criteria.

The legacy repository remains available as project provenance and as evidence of the exploratory process that informed the final methodology.
