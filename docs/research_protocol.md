# Research Protocol

## Research question

How did the discursive distance between governing and opposition blocs evolve in
Argentina's Chamber of Deputies during the recent period with reliable speaker and
block identification?

## Candidate corpus

The initial candidate corpus begins on 1 January 2008 and ends on the most recent
session discovered in the official source snapshot.

The final starting year will not be chosen in advance. It will be selected through
predefined extraction, speaker-resolution, and political-metadata quality criteria.

## Units of analysis

- Source unit: one parliamentary speaker turn.
- Text representation unit: one legislator within one session.
- Measurement unit: one parliamentary session.

Multiple interventions from the same legislator in the same session will be aggregated
before political-side centroids are calculated.

## Primary political comparison

The primary analysis compares:

- `government_core`
- `opposition_core`

Blocks classified as `ambiguous_or_independent` are excluded from the primary result
and may be included in sensitivity analyses.

## Primary session eligibility

A session must:

1. have taken place;
2. contain a valid and readable transcript;
3. contain substantive parliamentary debate;
4. allow debate text to be separated from appendices and voting tables;
5. have at least three matched legislators on each political side;
6. have at least 1,000 substantive words on each side;
7. achieve at least 85% word-weighted speaker-match coverage; and
8. belong to an included session type.

## Year-level quality rule

A year is considered high quality when:

1. at least 70% of candidate substantive sessions pass structural QA;
2. median word-weighted speaker-match coverage is at least 90%;
3. a sufficient number of sessions contain representation from both sides; and
4. no systematic year-specific extraction failure is detected.

The primary analysis starts with the first of at least two consecutive high-quality years.

An incomplete final calendar year may be included in session-level analyses but will not
be treated as directly comparable to complete annual periods without qualification.

## Text preprocessing

Embedding inputs will preserve natural grammatical text, including names, syntax, and
stopwords. Only extraction artifacts, repeated page furniture, and structurally identified
procedural material will be removed.

Lexical analyses will use a separate preprocessing pipeline. Model-specific preprocessing
will be documented independently.

## Main metric

Each eligible legislator-session document will be represented as a semantic vector.
Legislators will receive equal weight within each side. Session-level discursive distance
will be measured as cosine distance between government and opposition centroids.

The metric will be described as semantic or discursive distance, not as a direct measure
of ideological polarization.

## Robustness analyses

Planned robustness analyses include:

- lexical divergence;
- strict and broad political-alignment definitions;
- alternative minimum-participation thresholds;
- inclusion and exclusion of informative sessions;
- leave-one-legislator-out influence analysis; and
- sensitivity to the selected starting year.

## Legacy model

The BERTopic model from the previous repository is retained only as an exploratory
prototype and methodological motivation. It will not be used to support final temporal
or political conclusions.