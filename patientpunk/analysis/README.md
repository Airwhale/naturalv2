# analysis — how the study got built

These scripts **produced** the inputs the pipeline consumes. You do not need to run any of them to
run a trial (see [../README.md](../README.md)); you need them to change *which* trials are in the
study, or to re-derive the coverage numbers quoted in the docs.

**No data lives here.** Everything reads from and writes to S3-backed paths under
`s3://patientpunk/trial_superset/` (see [Data](#data) below).

## The pipeline that builds a study

Roughly in dependency order.

| script | what it does |
|---|---|
| `convert_corpus.py` | Reddit JSONL dumps → the partitioned parquet corpus the pipeline reads |
| `seed_terms.py` | per-condition keyword classifier that replaces upstream's substring matcher (findings A1), plus the broadened CT.gov search scope (B1) |
| `run_study.py` | thin driver around naturalv2's `create_study` with our config |
| `m3_pool.py`, `broaden.py` | pull the candidate trial pool from CT.gov at the broadened scope |
| `adapt_registries.py` | adapt non-CT.gov trials (ISRCTN) into the same schema, flagged `registry_adapted` |
| `litlabels/` | **papers-as-labels**: for a completed trial with no posted results, find the paper via Europe PMC, extract the primary endpoint, and shape it like an `Experiment`. `europe_pmc.py` and `cache.py` are vendored HTTP/caching helpers |
| `build_augmented.py`, `build_improved.py`, `build_master_csv.py` | merge structured, registry-adapted and paper-extracted trials into one manifest |
| `relaxed_test_universe.py` | recruiting-inclusive test universe, since `status:act` excludes recruiting trials (findings A4) |
| `build_core5_study.py` | emit the core-5 study in naturalv2's own study-YAML format |

## Measuring corpus coverage — the trial-selection criterion

This is the part that is genuinely ours: deciding whether there is enough patient discussion of a
treatment to estimate anything from it.

| script | what it does |
|---|---|
| `measure_coverage.py` | distinct authors mentioning each drug across the corpus, by word-boundary alias regex |
| `validate_coverage.py` | samples ~20 mentions per drug and has an LLM judge whether each is a genuine on-target usage report. Clean-alias controls (fluvoxamine, oxaloacetate, LDN) sanity-check the method |
| `studies_list.py` | joins trials + coverage + validation into the ranked list of runnable trials |
| `long_covid_eval.py` | the arm-level prediction-target set, including the factorial relabel |
| `benchmark_treatment_aliases.py`, `count_covidlonghaulers_*.py` | corpus signal counts per candidate treatment |

`effective_authors = distinct authors × LLM-validated on-target fraction`, and the inclusion gate is
≥ 50. Replacing an earlier self-obtainability gate with this took the set from 5 to 19 trials.

## Audits and label QA

| script | what it does |
|---|---|
| `build_labels_sidecar.py` | per-arm label sidecar with `endpoint_type`, `clean_outcome` and change-from-baseline detection. Largely superseded upstream (findings A2), kept for the residual `NUMBER`/`COUNT_OF_UNITS` case and as the change-detection reference |
| `audit_conditions.py` | evidence for the condition-matcher audit (A1) |
| `binary_compare.py` | what the `binary` preset costs versus `notbinary` (−92% of the set) |
| `endpoint_classify.py`, `drug_classify.py` | endpoint and drug-class labelling |
| `sanity_check.py`, `verify_m1.py`, `extract_validate.py` | reproduction checks against upstream's own study |
| `review_*.py`, `mine_*.py`, `pull_*.py` | one-off pulls and manual-review helpers |

## Data

Nothing in this repository. Everything is under **`s3://patientpunk/trial_superset/`**:

| prefix | contents |
|---|---|
| `natural_corpus_parquet/` | the corpus (~458 MB): 147,333 submissions + 2,304,485 comments, three Long-COVID subreddits, 2020-07 → 2026-06 |
| `m3_labeled/long_covid/nct_reports/` | trial records with posted results |
| `relaxed_test/nct_reports_test/` | prediction-target trial records |
| `core5/` | the core-5 study YAML |
| `*.csv` | the analysis outputs: `drug_coverage`, `coverage_validation`, `studies_list`, `benchmark_19`, `long_covid_eval_set`, `labels_sidecar`, `master_pulled_data` |

`../scripts/00_fetch_inputs.sh` pulls what a run needs. These scripts expect a local mirror; set
`SAVE_PATH` and sync the prefixes you need.

## Running them

They were written against `trial_superset/` as the working directory and assume a local `data/`
mirror, so paths in them are relative to that layout. They are included for reference and reuse
rather than as a turnkey second pipeline — if you re-run one, check its module docstring first, and
expect to point it at your own local mirror of the S3 prefixes above.
