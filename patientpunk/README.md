# patientpunk — running NATURAL on a pre-built patient corpus

All the code needed to reproduce our runs lives in this directory — nothing outside this repository
is required. **No data lives here**: the corpus, trial records and study definition are on S3, and
`scripts/00_fetch_inputs.sh` pulls them. You supply API keys.

For *what we found* rather than *how to run it*, see
[docs/patientpunk/findings.md](../docs/patientpunk/findings.md); for the full write-up of the method
and its open problems, [docs/patientpunk/pipeline_overview.md](../docs/patientpunk/pipeline_overview.md).

```
patientpunk/
  serving/            get a vLLM server up on rented GPU, and tear it down again
    launch_gpu.py             create/cancel a dispersed GPU job
    probe_long_logprobs.py    fail-fast check that a server can do prompt_logprobs
    run_estimate_chain.py     launch -> wait -> probe -> estimate -> ALWAYS stop
    vllm-server/              the container image (env-driven; one image serves any model)
  scripts/            the pipeline, stage by stage (00_fetch_inputs first)
  analysis/           how the study itself was built: trial selection, corpus coverage,
                      papers-as-labels, audits. Not needed to run a trial
```

**Data lives on S3**, under `s3://patientpunk/trial_superset/`:

| prefix | what |
|---|---|
| `natural_corpus_parquet/` | the corpus, ~458 MB |
| `core5/` | the study definition |
| `m3_labeled/long_covid/nct_reports/` | trial records with posted results |
| `relaxed_test/nct_reports_test/` | prediction-target trial records |
| `*.csv` | analysis outputs — coverage, validation, the trial list, label sidecar |

## What you supply

| | |
|---|---|
| **corpus** | partitioned parquet, `content_type=<submissions\|comments>/bucket=<subreddit>/*.parquet`. Ours is 147,333 submissions + 2,304,485 comments from three Long-COVID subreddits, 2020-07 → 2026-06. Point `PP_CORPUS_DIR` at it |
| **API key** | an OpenRouter key for the generative stages (`OPENROUTER_API_KEY`) |
| **GPU** | anything serving vLLM with `prompt_logprobs`. `serving/` automates one on dispersed; a local card or any rented box works too |

Put keys in a `.env` at the repo root (or point `PP_ENV_FILE` at one). It is read by the scripts and
never committed.

## Environment

| variable | default | meaning |
|---|---|---|
| `PP_CORPUS_DIR` | — | **required.** The pre-built parquet corpus |
| `SAVE_PATH` | `<repo>/outputs` | where naturalv2 writes studies, experiments, curated data, results |
| `PP_PYTHON` | `python` | interpreter (activate your venv, or set this) |
| `PP_ENV_FILE` | `<repo>/.env` | file of API keys to source |
| `EXPERIMENT` | `noparallel_notbinary` | preset name; appears in every output path |
| `CONDITION` | `Long Covid` | condition string used to name the study |
| `GEN` / `CHEAP` | gemini-2.5-flash / -flash-lite | generative model, and the cheap one for the two filter stages |
| `PROBS_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | the model served for `prompt_logprobs` |

## Run it

**0. Fetch inputs and build experiments.** No data lives in this repository — it is all on S3.

```bash
export SAVE_PATH=$PWD/outputs
bash patientpunk/scripts/00_fetch_inputs.sh            # study + trial records (~1 MB)
# add --corpus to also sync the ~458 MB corpus into $PP_CORPUS_DIR

# NOTE: no --require_binary_endpoint. This is a notbinary study; the flag defaults ON and would
# drop every one of these continuous-endpoint trials.
python -m scripts.build_experiments_from_study \
  --study_yaml "$SAVE_PATH/studies/long_covid_noparallel_notbinary_apo_study.yaml" \
  --save_path "$SAVE_PATH"
```

> **The study filename is load-bearing.** Every CLI locates the study via `get_study_filepaths`,
> which *derives* the name from the condition and experiment: `Long Covid` + `noparallel_notbinary`
> + `ate: False` → `long_covid_noparallel_notbinary_apo_study.yaml`. Rename it and the pipeline
> reports a missing study rather than a naming problem.

**1. Seed what the skipped stages would have written.** We skip `condition_filter` (needs Reddit
OAuth) and `treatment_synonyms` (needs OpenAI web-search), so two small things must be supplied by
hand. Skipping this step is the single easiest way to get **zero curated records and no error**.

```bash
python patientpunk/scripts/01_seed_study_dataset.py \
  --subreddits covidlonghaulers LongCovid LongHaulersRecovery

# only needed where arm labels are not searchable text, e.g. a factorial "Placebo/LDN"
python patientpunk/scripts/02_seed_treatment_aliases.py --nct NCT06366724 \
  --alias "Placebo/LDN=naltrexone,low dose naltrexone,ldn" \
  --alias "Pyridostigmine/Placebo=pyridostigmine,mestinon" \
  --alias "Pyridostigmine/LDN=naltrexone,low dose naltrexone,ldn,pyridostigmine,mestinon"
```

**2. Fix change-from-baseline endpoints** (see findings A6). Re-run this any time you rebuild
experiments — `build_experiments_from_study` regenerates the YAML and discards the rewrite.

```bash
python patientpunk/scripts/03_fix_change_outcomes.py --nct NCT05618587
```

**3. Curate.** Free: no LLM, CPU only. The first run contextualises the whole corpus (minutes); later
trials reuse it.

```bash
export PP_CORPUS_DIR=/path/to/natural_corpus_parquet
NCT=NCT05618587 SPLIT=train bash patientpunk/scripts/run_filter_curate.sh
```

**4. Estimate.** Needs the vLLM endpoint. Either bring your own:

```bash
VLLM_BASE=http://your-host:8000/v1 NCT=NCT05618587 SPLIT=train \
  bash patientpunk/scripts/run_estimate.sh
```

…or let `serving/` rent one and tear it down for you:

```bash
python patientpunk/serving/launch_gpu.py            # dry-run: prints the job spec, spends nothing
python patientpunk/serving/launch_gpu.py --go       # creates the job; BILLING STARTS
NCT=NCT05618587 SPLIT=train python patientpunk/serving/run_estimate_chain.py <job_uuid>
python patientpunk/serving/launch_gpu.py --stop <job_uuid>   # if the chain could not
```

## Things that will bite you

- **`--require_binary_endpoint` defaults ON** in `build_experiments_from_study`, and silently drops
  every continuous-endpoint trial. Omit it for a `notbinary` study.
- **Skipping step 1 yields zero records and no error.** `_build_registry` skips any experiment whose
  condition maps to no subreddits.
- **`prompt_logprobs` needs LOW `gpu_memory_utilization`** (~0.55, not the usual 0.90). It allocates
  a transient `[prompt_tokens × vocab]` logits tensor *outside* vLLM's preallocated pool — about
  3.5 GB for a 5.7k-token prompt at a 152k vocab. Because it is a fraction, **a bigger GPU does not
  help**: more VRAM just becomes more KV cache. The image defaults are set correctly.
- **The default estimator cannot run continuous outcomes.** `conf/estimate_ate.yaml` uses
  `natural_ipw`, whose `conditional_extraction` enumerates options over the outcome; continuous
  endpoints have none and it dies with a bare `ZeroDivisionError`. Use `estimate_mc.yaml`
  (`run_estimate.sh` already does).
- **A crashed stage leaves a 0-byte CSV** in `results/`, which the resume logic then reads →
  `EmptyDataError`. `wait_and_estimate.sh` deletes them before each run.
- **Changing which stages run invalidates caches.** `SampleTYStage` writes its CSV *before*
  discretising, so a stale downstream cache gives `KeyError: treatment_taken_discretized`. Delete
  caches downstream of any composition change.
- **Cost is dominated by `relevance_filter`**, which scores every curated report once *per outcome*.
  A four-outcome trial over 42k reports is hours and tens of dollars; that is what `CHEAP` is for.
