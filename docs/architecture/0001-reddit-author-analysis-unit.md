# ADR 0001: Use one curated Reddit report per known author

## Status

Accepted

## Context

The Reddit curation pipeline can select several distinct records written by the same
pseudonymous author. Treating those records as independent patients gives prolific
authors more influence during extraction, estimation, and row-level bootstrap.

Reddit comments also include the initial submission for context. That text may have
been written by someone other than the author whose comment is being analyzed.

## Decision

For each experiment, concatenate all curated candidate records with the same non-null
`author_key` before LLM extraction. Keep records without an author key as separate
analysis units. Remove exact permalink duplicates and order records deterministically
by creation date and permalink.

Each record must distinguish `Author's own text` from
`Context written by another Reddit user`. The combined artifact remains a single
string-valued `report`, so downstream extraction and estimation contracts do not
change.

Aggregation applies only to records already selected by subreddit, date, and treatment
matching. It does not retrieve an author's unrelated Reddit history.

## Consequences

- A known Reddit author contributes at most one row to an experiment.
- Existing row-level bootstrap treats the combined row as one analysis unit.
- Conflicting treatments or outcomes remain visible to the extraction model rather
  than being resolved by a new structured-data reducer.
- Reports from prolific authors are longer and may use more model input tokens.
- Missing, deleted, and removed authors are not incorrectly clustered together.
