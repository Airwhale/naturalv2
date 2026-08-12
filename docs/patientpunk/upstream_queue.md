# Upstream queue — what to file, in what order

Everything here is a defect in `naturalv2` itself, found by running the pipeline end to end on a
Long-COVID Reddit corpus (147k submissions, 2.3M comments, three subreddits, 2020-07 → 2026-06).
Pinned at `7a2e006`.

**Ordered by how much it damages a result, not by how easy it is to fix.** Where we have a working
fix, a PR sketch sits directly under the issue. Where we do not, the issue says what we would need
from Nikita to write one.

Evidence for every claim is in [findings.md](findings.md); the method it came from is in
[pipeline_overview.md](pipeline_overview.md). Bug IDs match those documents.

| | count |
|---|---|
| issues to file | 12 |
| with a PR ready to open | 3 |
| additive contributions offered | 2 |
| already fixed by Nikita in `7a2e006` | 2 |

---

## Group 1 — Results are silently wrong

The pipeline completes, returns a well-formed number, and the number is meaningless. No error, no
warning.

### 1. A11 — a comment inherits the outcome of the thread it replies to

**Impact: every estimate drawn from comment evidence. No fix proposed — see below.**

`curate.py` (`fmt_comment`, ~line 834) embeds the full text of the initial post in every comment's
report. `sample_ty` then asks for the value *"the individual described in the report"* would give. A
comment's report describes two people — the original poster and the commenter — and describes the
poster far more fully. The model answers about the wrong one.

Measured on our two runs:

| | Brain Fog (lithium) | Fatigue (lithium) | Functional Capacity (LIFT) |
|---|---|---|---|
| evidence rows | 32 | 9 | 3,127 |
| distinct threads behind them | 7 | 4 | 494 |
| rows whose own comment names the treatment | 4 | 1 | 735 |
| rows where only the thread names it | **28 (88%)** | **8 (89%)** | **2,375 (76%)** |
| median length of the comment itself | 99 chars | 54 chars | 142 chars |

For lithium's Brain Fog, **21 of 32 rows carry the identical value −35.0**. The largest source thread
is titled *"Long Covid Recovery to 90% — Antihistamine Treatment"* and supplies 11 rows on its own.
The comments scored −35 read, in full: *"Yes! Those are horrible when they happen.. Thanks for
sharing."*, *"Hey! We're you able to eventually ween off the antihistamines?"*, and *"It's cheap.
Will give it a try. I take generic zyrtec and Benadryl at night"*. None reports an outcome; none is
about lithium.

The separation is clean: the 4 rows whose own text discusses lithium average **−2.5**; the 28 that do
not average **−33.2**. The trial's answer is **−9.0**.

**Why no PR.** Four fixes are possible and choosing between them is a research decision, not a
mechanical one:

1. require the report's own `report_text` to mention the matched treatment (would have dropped 88% of
   lithium's evidence — which is the point);
2. name the subject in the prompt, and mark the initial post as background about a different person;
3. give the outcome question a different context from the relevance and treatment questions;
4. aggregate to the thread, or cluster-bootstrap by it.

We have a validated precedent for option 2 in our own extraction stack, and can port it if useful.

**Also fixes:** A8, largely — see issue 9.

### 2. A12 — extracted outcome values are never range-checked

**Impact: any trial where one parse goes wrong. Mechanical fix; we can write it on request.**

`sample_ty` asks for *"a single number … on the same scale as the outcome description"*, and whatever
comes back is used. Nothing compares it against the range the description states.

LIFT, FUNCAP55, a 0–55 scale, 3,127 rows:

| | |
|---|---|
| values outside 0–55 | **462 (14.8%)** |
| literal `inf` | 2 |
| maximum | **4,444,000** |
| mean of finite values | **+1,479.6** |

One value of 4.4 million on a 55-point scale moves the mean by more than a thousand.

**Proposed fix.** Reject, do not clamp — a clamp turns a parse failure into a confident extreme.
`Experiment` already knows the endpoint range wherever the description states it. Log the rejection
rate; it is a useful extraction-quality signal in its own right.

### 3. A9 — treatment matching drops 3-character drug abbreviations

**Impact: 79% of LDN evidence. Fix written and verified.**

Two independent filters use the same threshold. `build_treatment_automaton` skips any canonical form
of `<= 3` chars, and `RedditCurateStage._prepare_registry_config` separately skips aliases with
`len(alias) <= 3`. Both drop `ldn` — the near-universal patient term for low-dose naltrexone, and one
of the most-discussed Long-COVID treatments there is. `vns`, `hbot` and similar go the same way.

The threshold exists to stop `mg` and `ml` matching, which is a real problem — but the actual cause of
those false positives is that Aho-Corasick matches substrings, so `ldn` also fires inside `couldn't`.
Fixing the matching lets the threshold drop safely.

> #### PR — boundary-aware treatment matching
>
> **Branch:** `fix/boundary-aware-alias-matching` · **~10 lines across 2 files**
>
> `sources/components/helpers.py` — `extract_mentions` already receives the match end position from
> Aho-Corasick and discards it. Use it to require a non-alphanumeric character (or the string edge)
> on both sides:
>
> ```python
> for end_idx, canonical_alias in automaton.iter(canonical_text):
>     start_idx = end_idx - len(canonical_alias) + 1
>     before = canonical_text[start_idx - 1] if start_idx > 0 else " "
>     after = canonical_text[end_idx + 1] if end_idx + 1 < len(canonical_text) else " "
>     if before.isalnum() or after.isalnum():
>         continue
>     found_mentions.add(canonical_alias)
> ```
>
> With word boundaries enforced, both length floors move from `<= 3` to `< 3` — still excluding `mg`
> and `ml`, now admitting `ldn`.
>
> **Verified:** 31,567 → 138,618 matching rows on our corpus, a **4.4×** increase, with no new false
> positives found in a manual sample. Costs no LLM calls — the matcher is free and deterministic.
>
> **Risk to her runs:** changes which rows are curated, so any cached curation must be rebuilt. Every
> alias that matched before still matches.

---

## Group 2 — Wrong trials go into the benchmark

These do not corrupt an individual estimate. They decide which trials are eligible at all, so they
set what the benchmark is measuring.

### 4. A1 — the condition matcher over- and under-matches

**Impact: the composition of her Long-COVID set. Needs her design decision.**

`find_condition_ncts` uses a plain substring test in both directions (`cond in trial_cond` or
`trial_cond in cond`). On her own shared Long-COVID study that admits **12 acute-COVID trials** and
drops **7 genuine post-COVID ones**. "COVID-19" is a substring of "Post COVID-19 Condition", so acute
trials match; a trial tagged only `PASC` does not.

We replaced it with a per-condition keyword classifier. Whether that shape belongs in her repo is her
call — hence an issue rather than a PR.

### 5. A6 — change-from-baseline labels compared against absolute predictions

**Impact: guaranteed error on every change-scored trial, regardless of model quality.**

Lithium's Fatigue Severity Scale is scored **1–49**, and the trial reports **−11.3** — an improvement
*from baseline*, not a level. `Experiment` passes that through as the label while the pipeline is
asked to predict a level. Comparing them is comparing *"the patient weighs 80 kg"* with *"the patient
lost 5 kg"*: both in kilograms, and the comparison means nothing.

The error is guaranteed and says nothing about how good the estimate was.

**Three defensible fixes, which is why this is an issue and not a PR:** handle change endpoints in
`Experiment`; rewrite the outcome description so the sampled quantity matches the label (what we do —
`fix_change_outcomes.py`, ours, happy to share); or exclude change-scored trials from the benchmark.

### 6. A4 — the test universe excludes recruiting trials

**Impact: which trials can be prediction targets. Arguably a tradeoff, not a defect.**

The test aggFilters use `status:act`, which in CT.gov means *active, not recruiting*. Every currently
*recruiting* trial is therefore excluded — including LIFT, which is the most useful prospective
target in Long COVID right now. The pinned code also disagrees with the study she shared.

**Question rather than a fix:** which status selection is canonical for the test universe? We run a
recruiting-inclusive one and can share it.

---

## Group 3 — Statistical validity

### 7. A13 — `NaturalMC`'s docstring says not to use it for APOs, which is what we do

**Impact: upstream of every number we have produced. We need her answer before trusting any of it.**

```python
class NaturalMC:
    """NATURAL Monte Carlo Estimator for individual treatment responses.
    TODO: Do not use for APOs; off-the-shelf estimators do not trivially extend to APOs.
```

We run `ate: False` — an APO per arm — with `estimator: natural_mc_oi`.

Reading the `oi` branch, it imputes the outcome under each treatment value for every unit and the
caller averages, which is the standardisation estimator for `E[Y(t)]` and looks correct for an APO.
So either the caution is about the `ipw` branch, or about something we have not spotted.

The `ipw` branch does look wrong for APOs, and may be what the note means: it computes
`all_ites[t, :] = individual_outcomes * t_mask`, zeroing every unit that did not take arm `t`, and
`estimate_ate.py` then averages over *all* units — returning `E[Y · 1{T=t}]`, shrunk by arm `t`'s
share of the sample. A single-arm experiment hides it, since the mask is all ones.

**Questions:** Is `natural_mc_oi` sound for `ate: False`? Is the TODO about `ipw`, both, or something
else? If MC is unsuitable for APOs generally, what should a continuous-endpoint APO study use, given
`natural_ipw` cannot run one at all (issue 10)?

### 8. A10 — `author` is discarded, so prolific posters count as many patients

**Impact: independence, which both the estimator and the bootstrap assume.**

The raw corpus carries `author`; neither the contextualised dataset nor the curated output keeps it.
Nothing after stage 2 can deduplicate, cluster or weight by patient. Lithium's evidence pool,
pre-filter:

| | |
|---|---|
| documents mentioning lithium | 1,012 |
| distinct authors | 365 |
| median documents per author | **1** |
| **maximum** | **99** |
| top 10 authors' share | **30.3%** |

One person supplies a tenth of the evidence; ten supply nearly a third. We cannot measure the
post-filter concentration *because the field is gone*, which is itself part of the argument.

**Mechanical half is small:** retain a salted hash of the author through contextualisation and
curation — clustering without storing handles. **Statistical half is a real choice:** cluster-bootstrap
by author, weight by 1/nᵢ, or deduplicate. With a median of one document per author, the tail is doing
the damage, so a cap may beat a global reweighting.

### 9. A8 — bootstrap intervals are implausibly tight

**Impact: any coverage or calibration metric computed from these intervals.**

Brain Fog: **−33.89, 95% CI (−34.87, −32.92)** — ±1 point from 32 noisy Reddit reports, missing the
truth (−9.0) by 24 points. Fatigue, from n=9, gives the *wider* interval, so tightness scales the
wrong way with n.

**Root cause is mostly A11, not `bootstrap_size`.** 21 of the 32 Brain Fog values are identical,
because they are comments on the same recovery threads. Resampling a column that is nearly constant
returns the same number whatever `B` is. `bootstrap_size: 10` is genuinely too small — with B = 10
the 2.5th and 97.5th percentiles are just the min and max of ten numbers — but raising it alone would
not have widened these intervals much.

Worth knowing whether she treats these as descriptive rather than calibrated. If they are meant to be
calibrated, coverage across the benchmark is measurable today and nobody has measured it.

---

## Group 4 — Crashes and robustness

### 10. A5 — the default estimator cannot run a `notbinary` study

**Impact: a hard crash, so at least it is not silent. Config workaround exists.**

`conditional_extraction` enumerates a grid over the outcome's option list. Continuous endpoints have
no options, so the grid is empty and it dies at `conditional_extraction.py:624` with a bare
`ZeroDivisionError` from `num_workers // len(interleaved_options)` — in the worker-pool setup,
hundreds of lines from the empty option list that caused it, so it reads as an infrastructure error.

Our `conf/estimate_mc.yaml` routes around it (issue 14). Changing the *shipped default* is a design
call. At minimum, a clear error at the point the option set comes back empty would save the next
person the trace.

### 11. A14 — a bare `except:` turns any fit failure into silent `NaN`

**Impact: a failed fit is indistinguishable from a legitimate empty result.**

`natural_mc.py`, `get_individual_treatment_effects`:

```python
try:
    model.fit(data)
    ...
except:
    all_ites = np.full((self._num_treat, len(observational_data)), np.nan)
return all_ites
```

A bare `except` catches everything — singular design matrix, dtype problem, `KeyboardInterrupt` — and
returns `NaN` with no log line. The failure surfaces later as a missing result that looks like a data
problem.

Not hypothetical for studies like ours: the outcome model is an unregularised `LinearRegression` over
8 covariates plus treatment, so with an intercept it fits 10 parameters, and our Fatigue outcome had
**9 reports**. Fewer rows than parameters is exactly where a fit degenerates. Our runs did not hit it,
which is luck rather than evidence it is safe.

**Proposed fix:** catch the specific exceptions, log which unit and outcome failed and why, let
anything unexpected propagate.

### 12. A7 — `detokenize=False` kills a hosted vLLM server

**Impact: none on her in-process path. Fatal for anyone serving vLLM over HTTP. Fix written.**

`conditional_extraction.py` hardcodes `detokenize=False` on every `prompt_logprobs` call, at two call
sites, with the note *"avoid vLLM detokenizing logprob token ids (OverflowError on some models/TP
setups)"*. Correct for the in-process `VLLMModel`. Against a hosted vLLM OpenAI server the serving
layer still builds `decoded_token` for every prompt logprob and has nothing to detokenize with, so
the flag aborts EngineCore and takes the server down. Because it is hardcoded, no config override can
reach it.

> #### PR — send `detokenize` only for the in-process model
>
> **Branch:** `fix/hosted-vllm-detokenize` · **+12 −2, one file**
>
> ```python
> def _detokenize_kwargs(llm: object) -> dict:
>     """Sampling kwargs that are only valid when vLLM runs in-process. ..."""
>     return {"detokenize": False} if isinstance(llm, VLLMModel) else {}
> ```
>
> Used as `**_detokenize_kwargs(llm),` at both call sites (~line 516 and ~line 816).
>
> **Risk to her runs: none.** The in-process path receives exactly the flag it does today; only the
> hosted path changes. Our first attempt simply flipped the hardcode to `True`, which would have
> broken her in-process runs — that is deliberately *not* what we propose.

---

## Group 5 — Additive contributions

Neither fixes a bug. Both are offered because they were needed to run at all.

### 13. Prebuilt-corpus source stage

> #### PR — `RedditPrebuiltParquet`
>
> **Branch:** `feat/prebuilt-parquet-source` · **+109 lines, one new file plus one config**
>
> `sources/reddit/stages/prebuilt_parquet.py` + `conf/source/reddit_prebuilt.yaml`. A `SourceStage`
> that points the existing contextualiser at an already-built partitioned parquet corpus instead of
> downloading `.zst` archives.
>
> **Why it may be worth having:** `RedditDownloadAndClean` fetches from the-eye's Pushshift mirror,
> which has been down for over a month. Anyone without a cached corpus currently cannot run the
> Reddit path at all. This also lets a team bring a corpus they already trust.
>
> **Risk: none.** Purely additive — a new stage and a new config, selected only when asked for.
> Nothing existing changes.

### 14. NATURAL-MC pipeline wiring and an OpenRouter provider

`conf/estimate_mc.yaml` composes the stages a continuous-endpoint study needs — `sample_ty` in,
`conditional_extraction` out, `estimator: natural_mc_oi`. Every piece is hers; only the wiring is
ours, and her own `estimate_ate.yaml` already carries `SampleTYStage` commented out in the same slot.
Offered as a config others can copy rather than as a change to the shipped default (issue 10).

`conf/model/openrouter.yaml` sits beside the existing gemini/anthropic/openai provider configs.
Convenience only; take it or leave it.

---

## Already fixed by Nikita in `7a2e006` — verified

Both were flagged by us before the 2026-08-09 push and are confirmed fixed. Listed so nothing gets
re-reported.

| id | what it was | verification |
|---|---|---|
| **A2** | `notbinary` labels computed as `value / N` — a response rate for binary endpoints, meaningless for a continuous mean, and ~84% of our rows are continuous | `_normalize_outcome_value` now returns the raw value for MEAN/MEDIAN/LSM. Our core-5 study builds through her `Experiment` with correct labels (lithium `[−11.3, −9.0]`, fluvoxamine `−47.3`). One residual: `NUMBER`/`COUNT_OF_UNITS` are still divided by N |
| **A3** | factorial arms named `"X/Placebo"` classified as placebo *by title* and silently dropped, including LIFT's LDN main effect | arms now typed by CT.gov `ArmGroupType` via `check_arm`. Verified on LIFT: all three treatment arms survive |

---

## Filing order

1. Open the three PRs first — issues 3, 12, 13. All are self-contained, none changes her default
   path, and they establish the working relationship before anything contentious.
2. File A11 (issue 1) as an issue with the measurements, asking which of the four directions she
   prefers. It is the largest finding and the one where her judgment decides the fix.
3. File A13 (issue 7) as a question, not a bug. Her answer gates whether our results mean anything.
4. Everything else as issues, in the order above.
