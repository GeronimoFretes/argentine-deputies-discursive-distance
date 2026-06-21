# Discursive Distance in Argentina's Chamber of Deputies

This project studies how the discursive distance between governing and opposition blocs evolved in Argentina's Chamber of Deputies during the recent period for which parliamentary speakers and their active blocs can be identified reliably.

## Research question

How did the discursive distance between governing and opposition blocs evolve during the recent period with reliable speaker and block identification?

## Planned methodology

The pipeline will:

1. discover official parliamentary-session transcripts;
2. extract page-aware text from the source PDFs;
3. identify speaker turns and procedural sections;
4. resolve speakers to legislators and contemporaneous parliamentary blocs;
5. aggregate interventions into legislator-session documents;
6. represent those documents using semantic embeddings;
7. calculate session-level government–opposition distance; and
8. validate the results through lexical and sensitivity analyses.

The final historical window will be selected using predefined data-quality and speaker-resolution criteria rather than an arbitrary starting year.

## Repository status

The project is currently under active development. The acquisition and validation pipeline is being rebuilt before any final model results are produced.

Generated corpora, PDFs, embeddings, and model artifacts are not committed to the repository.

## Legacy project

This repository is a redesign of an earlier exploratory topic-modelling project:

`https://github.com/fedesaroka/nlp-sesiones-diputados`

The original work was developed by Federico Saroka, Ramón Eppens, Geronimo Fretes, and Felipe Merlo. See [`docs/legacy_provenance.md`](docs/legacy_provenance.md) for details.
