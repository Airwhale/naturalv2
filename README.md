# NATURAL-v2

This repository extends [NATURAL](https://arxiv.org/abs/2407.07018) ([code](https://github.com/nikitadhawan/natural)) to larger data and evaluation scales. Given a medical condition, it uses LLMs to extract treatment effects from real-world text (Reddit posts, PubMed articles) and benchmarks them against ground-truth outcomes from completed clinical trials on [ClinicalTrials.gov](https://clinicaltrials.gov). It can also be applied to active trials with complete recruitment to predict and pre-register results before they are published.

The pipeline supports two evaluation modes, controlled by the `ate` flag throughout:
- **APO mode** (`ate=False`, default) — estimates per-arm average potential outcomes for each treatment
- **ATE mode** (`ate=True`) — estimates head-to-head treatment comparisons, i.e. average treatment effects

---

## Contents

- [Setup](#setup)
- [Step 1 — Create a Study](#step-1--create-a-study)
- [Step 2 — Filter and Curate Data](#step-2--filter-and-curate-data)
- [Step 3 — Estimate Potential Outcomes](#step-3--estimate-potential-outcomes)
- [Extending the Pipeline](#extending-the-pipeline)
  - [Adding a New Source](#adding-a-new-source)
  - [Adding a New Estimator](#adding-a-new-estimator)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Setup

It is recommended to use [uv](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) for dependency management and virtual environments.

```bash
git clone https://github.com/nikitadhawan/naturalv2.git
cd naturalv2
uv sync --no-cache --dev
```

> [!NOTE]
> Add `--active` to `uv sync` to use your current virtual environment. Otherwise, `.venv` will be created in the project root.

Copy the example environment file and edit as needed, e.g. to add API keys:

```bash
cp .env.example .env
```

---

## Step 1 — Create a Study

Creates a study for a given condition using clinical trials from [ClinicalTrials.gov](https://clinicaltrials.gov), matched against MeSH terms. Training and validation sets are constructed with a temporal split of completed trials, while the test set contains active trials for which recruitment is complete, enabling retrospective and prospective evaluation respectively.

**Prerequisites:** ~4 GB disk space. The first run downloads and caches all trials; subsequent runs use the local cache.

```bash
uv run --active --env-file=.env create_study conditions=["Migraine","Migraine Disorders"]
```

**Output:** `{save_path}/studies/{condition}_study.yaml` — trial metadata and train/val/test splits.

> [!NOTE]
> Set `save_path` to control the output directory.

> [!NOTE]
> Use the same `conditions` and `save_path` in all three steps so each step can find the outputs of the previous one.

---

## Step 2 — Filter and Curate Data

Filters and curates per-trial datasets from one or more sources, for trials in the `train`, `val`, or `test` set, according to the `split` parameter.

**Available sources:**

| Source | Config | Description |
|---|---|---|
| Reddit (top 20k) | `reddit_20k` | Filters the top 20k subreddits via the Reddit API, then downloads and curates relevant posts |
| Reddit (full archive) | `reddit_archive` | Downloads and processes the full Pushshift Reddit archive |
| PubMed | `pubmed` | Fetches and curates PubMed abstracts and PMC full texts |

Sources are enabled in `conf/common.yaml` defaults. Use `~source@sources.<name>=<config>` to disable a source at runtime (e.g. `~source@sources.pubmed=pubmed`).

**Prerequisites:** Reddit API credentials (`PRAW_CLIENT_ID`, `PRAW_CLIENT_SECRET`, `PRAW_USERNAME`, `PRAW_PWD`, `PRAW_AGENT`) set in `.env`. Obtain them by registering an app at [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps). LLM API key(s) for the model(s) specified below.

```bash
uv run --active --env-file=.env filter_curate \
    '~source@sources.pubmed=pubmed' \
    sample_model.model_id="gemini/gemini-2.5-pro" \
    sample_model.rpm=145 \
    sample_model.tpm=1950000 \
    sample_model.rpd=10000 \
    +sample_model.thinking.type=enabled \
    +sample_model.thinking.budget_tokens=2048 \
    sample_model.max_tokens=4096 \
    conditions=["Migraine","Migraine Disorders"] \
    filter_by_date=True \
    split=val \
    experiment_name=test
```

**Output:**
- `{save_path}/experiments/{nct_id}.yaml` — per-trial metadata (treatments, outcomes, covariates)
- `{save_path}/{source}_data/` — downloaded source data
- `{save_path}/curation_results/` — curated per-trial datasets
- `{save_path}/studies/{condition}_study_dataset.yaml` — bookkeeping for curated data paths and sizes

> [!NOTE]
> `filter_by_date=True` restricts curated data to posts and articles published before the trial results were made public, preventing data leakage. Set to `False` to include all available data.

> [!NOTE]
> Model choices and rate limits can be adjusted based on your API tier.

---

## Step 3 — Estimate Potential Outcomes

Runs the extraction pipeline on curated data and estimates average potential outcomes or average treatment effects.

**Available estimators:**

| Estimator | Config | Description |
|---|---|---|
| NATURAL-IPW | `natural_ipw` | Inverse Propensity score Weighting — uses P(T, Y \| X) from vLLM logits to reweight observations |
| NATURAL-OI | `natural_oi` | Outcome Imputation — uses P(Y \| T, X) from vLLM logits to for outcomes per treatment arm |
| NATURAL-MC | `natural_mc` | Monte Carlo variant using off-the-shelf IPW and OI estimators |

**Prerequisites:** CUDA GPU. Install vLLM with:

```bash
uv sync --no-cache --dev --extra vllm
```

```bash
uv run --active --env-file=.env estimate_ate \
    '~source@sources.pubmed=pubmed' \
    '~pipeline.stages.relevance_filter' \
    model@cheap_model=openai \
    cheap_model.model_id=gpt-4.1-mini \
    sample_model.model_id=gemini/gemini-2.5-pro \
    +sample_model.thinking.type=enabled \
    +sample_model.thinking.budget_tokens=512 \
    sample_model.max_tokens=1024 \
    model@imputations_model=openai \
    imputations_model.model_id=gpt-4.1 \
    probs_model.model_id="google/gemma-3-4b-it" \
    probs_model.model_kwargs.gpu_memory_utilization=0.4 \
    probs_model.model_kwargs.tensor_parallel_size=2 \
    estimator=natural_ipw \
    conditions=["Migraine","Migraine Disorders"] \
    experiment_name=test \
    split=val
```

**Output:** `{save_path}/results/{nct_id}_{experiment_name}/` — CSV files with predicted and ground-truth treatment effects per trial.

> [!NOTE]
> Model choices and parameters can be adjusted based on your budget and hardware.

---

## Extending the Pipeline

### Adding a New Source

Implement your stages in `naturalv2/sources/`, add a config YAML to `conf/source/`, and register it in `conf/common.yaml`.

Each stage subclasses `CurationStage` (or `SourceStage` for filesystem helpers) and implements `async run`:

```python
from naturalv2.sources.core import CurationContext, SourceStage, StageState
from naturalv2.utils import get_experiment_filepath

class MyFetchStage(SourceStage):
    async def run(self, context: CurationContext, state: StageState) -> StageState:
        # fetch data, build a DataFrame with a "report" text column
        df = ...
        state.payload = df
        return state

class MyCurateStage(SourceStage):
    async def run(self, context: CurationContext, state: StageState) -> StageState:
        df = state.payload
        curated_paths = {}
        for experiment in context.experiments:
            # filter df to rows mentioning experiment treatments, save to disk
            path = ...
            curated_paths[experiment.nct_id] = path

            # Register the curated path on the experiment so estimate_ate can find it.
            # Use the _no_date_filter suffix when filter_by_date=False.
            source_key = context.source_name if context.filter_by_date else f"{context.source_name}_no_date_filter"
            experiment.source_paths[source_key] = path

            # Persist the updated experiment YAML (treatments, outcomes, source paths).
            experiment.to_yaml(get_experiment_filepath(context.save_dir, experiment.nct_id))

        # Record curated paths and sizes in the study dataset YAML.
        self.persist_dataset(context, per_experiment_paths=curated_paths)
        return state
```

Then add `conf/source/my_source.yaml`:

```yaml
# @package _global_
stages:
  fetch:
    _target_: naturalv2.sources.my_source.MyFetchStage
  curate:
    _target_: naturalv2.sources.my_source.MyCurateStage
```

To enable it, either register it in `conf/common.yaml`:

```yaml
defaults:
  - source@sources.my_source: my_source
```

Or add it on the command line:

```bash
+source@sources.my_source=my_source
```

### Adding a New Estimator

Implement your estimator class in `naturalv2/estimators/`. The only required method is `get_individual_treatment_effects`, which takes the extraction DataFrame produced by the pipeline and returns a `(num_treatments, num_samples)` array of individual treatment responses:

```python
import numpy as np
import pandas as pd
from naturalv2.experiment import Experiment

class MyEstimator:
    def __init__(self, experiment: Experiment) -> None:
        self.experiment = experiment

    def get_individual_treatment_effects(self, conditionals: pd.DataFrame) -> np.ndarray:
        # conditionals contains P(T, Y | X) or P(Y | X, T) columns per treatment arm,
        # plus discretized covariate columns ({covariate}_discretized).
        # Return shape: (num_treatments, num_samples)
        ...
```

Then add `conf/estimator/my_estimator.yaml`:

```yaml
_target_: naturalv2.estimators.my_estimator.MyEstimator
experiment:
```

Select it at runtime with:

```bash
estimator=my_estimator
```

---

## Troubleshooting

> [!WARNING]
> If a run fails partway through, it may leave behind an empty CSV. Subsequent runs will then fail with `No columns to parse from file`. Delete the empty CSV and re-run.

---

## Acknowledgements

Franklin Ogidi scaled up the data collection and curation, including improvements to overall pipeline efficiency and LLM orchestration.
We would like to thank Kieran Quinn for providing medical expertise and advice on clinical trial emulation, Elizabeth Uleryk for pointers on PubMed curation, as well as Amrit Krishnan, Rahul Krishnan, Chris Maddison, and the Vector Institute for resources and support throughout.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
