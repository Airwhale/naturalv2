# Findings — bugs and gotchas from running NATURAL end to end

<!-- Category A is what matters for review: real issues in naturalv2 itself. B is a
ClinicalTrials.gov search gotcha, C is bugs in our own code (C3 is kept because it is the mirror
image of A6), D is serving/ops. References to `trial_superset/` are historical: that code now lives
in `patientpunk/analysis/`, and the serving kit in `patientpunk/serving/`. -->

A single registry of every bug we hit running NATURAL end-to-end on a Long-COVID Reddit corpus.
**Start with the fix-by-fix summary below** — it lists each change we made and what it is for. The
numbered sections after it carry the evidence.

> **Filing to upstream?** Use [upstream_queue.md](upstream_queue.md) instead. It carries the same
> issues ordered by how much each damages a result, with a PR sketch under each one we can already
> fix. This document stays the evidence registry behind it.

Category A is real issues in `naturalv2` (they affect Nikita's *own* results, not just ours);
B is a CT.gov search gotcha; C is bugs in our own code (kept because C3 is the mirror image of A6);
D is serving/ops. Each entry: what · where · evidence · impact · fix/status.

Full detail for A1, A2 and A4 also lives in the per-topic docs alongside this one: [condition_filter_audit.md](condition_filter_audit.md), [label_normalization.md](label_normalization.md), [test_universe_status.md](test_universe_status.md).

> **Status update — we re-pinned `naturalv2` to main `7a2e006` (Nikita's 2026-08-09 push).**
> That push **fixes A3** (arms now classified by structured `ArmGroupType`, not the title string) and
> **mostly fixes A2** (continuous means are no longer divided by N). Verified end-to-end: our core-5
> study builds through her native `Experiment` with correct labels (e.g. lithium `[-11.3, -9.0]`,
> fluvoxamine `-47.3`). **A1 and A4 are untouched by that push.** The index table reflects post-re-pin
> status; each item's original evidence is kept below as the record of what was wrong.

## Summary

### Closed — what is now fixed

Twenty-two defects found; **ten are closed**. "Upstream" means Nikita fixed it in her 2026-08-09 push
after we flagged it; "ours" means we wrote the fix.

| # | what was wrong | fixed by | how |
|---|---|---|---|
| **A2** | `notbinary` labels computed as `value / N` — meaningless for a continuous mean, and ~84% of our rows are continuous | upstream | `_normalize_outcome_value` now returns the raw value for MEAN/MEDIAN/LSM. One residual: `NUMBER`/`COUNT_OF_UNITS` are still divided by N |
| **A3** | factorial arms named `"X/Placebo"` classified as placebo *by title* and silently dropped — including LIFT's LDN main effect | upstream | arms now typed by CT.gov `ArmGroupType` via `check_arm`; verified on LIFT, all three treatment arms survive |
| **A7** | `detokenize=False` hardcoded on every `prompt_logprobs` call; harmless in-process, fatal against a hosted vLLM server | **ours** | `_detokenize_kwargs(llm)` sends it only for in-process `VLLMModel`, so her path is untouched |
| **A9** | treatment matching dropped ≤3-char aliases, costing **79% of LDN evidence** | **ours** | boundary-aware matching using the position Aho-Corasick already returned, then floor lowered to 3. **4.4× more matching rows** |
| **B1** | `query.cond="COVID"` silently missed trials tagged `SARS-CoV-2` / `PASC` | **ours** | broadened the search scope; recovered +4 with-results and +33 without |
| **C1** | `inject_one` returned `None` for two different outcomes, so the caller re-added a never-written trial | **ours** | returns `False` when nothing is written |
| **C2** | misc slips: EPMC double-`/PMC/` URL, missing `enrollmentInfo.type`, synthetic-results ValidationError, Windows `cp1252` crashes | **ours** | all fixed during development |
| **C3** | our own change-from-baseline detector read only the outcome *title*, so it missed the very trial that motivated it | **ours** | now also reads `timeFrame` and the description, and flags any value outside the range the description states |
| **D1** | `prompt_logprobs` OOM-killed vLLM; a bigger GPU did not help | **ours** | `gpu_memory_utilization` **lowered** to 0.55 — it is a fraction, so more VRAM just becomes more KV cache |
| **D2** | crashed stages left 0-byte CSVs that the resume logic then read | **ours** | deleted before every run |

### Open — still outstanding

| # | issue | owner | why it is still open |
|---|---|---|---|
| **A1** | condition matcher over/under-matches (~half her Long-COVID set is acute-COVID) | Nikita | ours is a keyword classifier; the right shape for her repo is her call |
| **A4** | `status:act` excludes recruiting trials; pinned code ≠ her shared study | Nikita | a tradeoff, not a defect — which status selection is canonical? |
| **A5** | default estimator cannot run a `notbinary` study (`ZeroDivisionError`) | Nikita | we supply `estimate_mc.yaml`, but changing the *default* is a design call |
| **A6** | change-from-baseline labels compared against absolute predictions | Nikita | fixing it in `Experiment`, rewriting descriptions, or excluding such trials are all defensible |
| **A8** | bootstrap intervals implausibly tight (±1 on n=32) | Nikita | mostly downstream of A11 — 21 of 32 values are identical, so any resample returns the same number |
| **A10** | `author` is dropped, so one prolific poster counts as many patients | Nikita | schema change is small; the statistics of what to do about it are open |
| **A11** | a comment inherits the **outcome of the thread it replies to** — 76–88% of our evidence rows never mention the treatment in their own text | Nikita | measurement error, and the likeliest cause of both our overshoot and A8 |
| **A12** | extracted outcome values are never range-checked — LIFT produced 4,444,000 on a 0–55 scale | Nikita | mechanical; the endpoint range is already known where the description states it |
| **A13** | `NaturalMC`'s docstring says not to use it for APOs — which is our configuration | Nikita | a question for her, and upstream of every number we have produced |
| **A14** | a bare `except:` turns any fit failure into silent `NaN` | Nikita | small and independent; matters most for the small-n outcomes |
| **C4** | our analysis scripts assume a working directory and a local data mirror | ours | works, but not portable — see `patientpunk/analysis/README.md` |
| **D3** | the GPU is held for a whole run but used only by `inclusion_prob` | ours | ~85% of GPU spend is idle time; needs the runner to acquire it later |

### The changes we wrote (for review)

| fix | what changed | file | addresses | status |
|---|---|---|---|---|
| **1. Prebuilt-corpus source** | new `SourceStage` that contextualises an already-built parquet corpus instead of downloading `.zst` archives | `naturalv2/sources/reddit/stages/prebuilt_parquet.py` + `conf/source/reddit_prebuilt.yaml` | no bug — additive; also a way around the-eye outage | **PR-ready**, purely additive |
| **2. Hosted-vLLM guard** | send `detokenize=False` only when the model is an in-process `VLLMModel` | `naturalv2/pipeline/conditional_extraction.py` (+12 −2) | **A7** | **PR-ready**, her path untouched |
| **3. Pipeline wiring for NATURAL-MC** (config only — the estimator is yours) | `sample_ty` in, `conditional_extraction` out, estimator `natural_mc_oi` | `conf/estimate_mc.yaml` | **A5** | **PR-ready**, but the default is a design call |
| **4. OpenRouter provider** | provider config beside the existing gemini/anthropic/openai | `conf/model/openrouter.yaml` | none — convenience | optional |
| **5. Serving config** | `gpu_memory_utilization 0.55`, no chunked prefill, 1 seq | *ours, not her code* | **D1** | docs note only |
| **6. Change-scale rewrite** | restate a change-from-baseline outcome's description so the sampled quantity matches the label | *ours* (`fix_change_outcomes.py`) | **A6** | **issue, not a PR** — one option among several |
| **7. Boundary-aware treatment matching** | use the match position Aho-Corasick already returns to reject matches inside words, then lower the alias floor from `>3` to `>=3` chars | `naturalv2/sources/components/helpers.py` + `sources/reddit/stages/curate.py` (~10 lines) | **A9** | **PR-ready** — recovers 4.4× the LDN evidence |
| **8. Alias seeding** | hand-seed `treatment_common_names` when `treatment_synonyms` can't run | *ours* | works around missing web-search | stopgap only |



---

## Index

| # | bug | location | affects HER results? | severity | status |
|---|---|---|---|---|---|
| **A1** | condition matcher over/under-matches | `naturalv2` `find_condition_ncts` | **yes** | high | flagged for Nikita; we use a clean classifier |
| **A2** | `notbinary` label = `value/N` for continuous | `naturalv2` `Experiment` | **yes** | **high** | **mostly FIXED upstream** (`7a2e006`); sidecar redundant except `NUMBER`/`COUNT_OF_UNITS` residual |
| **A3** | factorial arms `"X/Placebo"` dropped | ~~`check_nonplacebo`~~ → `check_arm`/`ArmGroupType` | **yes** | medium | **FIXED upstream** (`7a2e006`); our relabel now redundant |
| **A4** | `status:act` excludes recruiting; pinned ≠ shared | `naturalv2` test aggFilters | **yes** | medium | flagged; relaxed universe shipped |
| **A5** | default estimator can't run a `notbinary` study | `naturalv2` `conditional_extraction` | **yes** | **high** | flagged; we run NATURAL-MC instead |
| **A6** | change-from-baseline label vs absolute prediction | `naturalv2` `Experiment` + our sidecar | **yes** | **high** | **open** — root cause of our first run's error |
| **A7** | `detokenize=False` kills a hosted vLLM server | `naturalv2` `conditional_extraction` | only off-repo runs | medium | **FIXED** — in-process guard, PR-ready |
| **A8** | bootstrap CIs implausibly tight (±1 on n=32) | `naturalv2` `bootstrap_size` | **yes** | medium | **open** — largely a symptom of A11 |
| **A9** | ≤3-char drug aliases dropped (79% of LDN evidence) | `build_treatment_automaton` + curate | **yes** | **high** | **FIXED** — boundary-aware matching, PR-ready |
| **B1** | `query.cond="COVID"` misses SARS-CoV-2/PASC tags | CT.gov search (our scope layer) | indirect | medium | fixed in `seed_terms` scope |
| **C1** | `inject_one` ambiguous `None` return | our `build_augmented.py` | no | medium | **fixed** |
| **C2** | misc implementation slips | our code | no | low | **fixed** |
| **A10** | `author` dropped, so prolific posters count many times | `naturalv2` contextualise + curate | **yes** | **high** | **open** — schema change small, statistics open |
| **A11** | a comment inherits the thread's outcome | `curate.py` `fmt_comment` + `sample_ty` prompt | **yes** | **highest** | **open** — measurement error; drives the overshoot and A8 |
| **A12** | extracted outcomes never range-checked | `sample_ty` parsing | **yes** | **high** | **open** — 14.8% of LIFT rows out of range, max 4.4M on a 0–55 scale |
| **A13** | `NaturalMC` docstring warns against APO use | `natural_mc.py` | **yes** | **high** | **open** — needs her answer; `ipw` branch looks wrong for APOs |
| **A14** | bare `except:` → silent `NaN` | `natural_mc.py` | **yes** | medium | **open** — our runs did not hit it; small-n fits are the risk |
| **C3** | `is_change` title regex missed untitled change endpoints | our `build_labels_sidecar.py` | no | **high** | **FIXED** — now reads timeFrame + range-checks the value |
| **C4** | analysis scripts assume a cwd and a local data mirror | our `patientpunk/analysis/` | no | low | **open** — not portable |
| **D1** | `gpu_memory_utilization` starves `prompt_logprobs` | serving config | no | high | fixed (0.55) |
| **D2** | stage CSV caches survive a pipeline-shape change | `naturalv2` stage resume | no | low | **FIXED** — 0-byte caches deleted before each run |
| **D3** | GPU held for a whole run, used only by `inclusion_prob` | our runner | no | medium | **open** — ~85% of GPU spend is idle |

---

## A. Bugs in `naturalv2` (affect Nikita's own results)

### A1 — Condition matcher over- and under-matches
**Where:** `find_condition_ncts` — keeps a trial if any mesh/conditions term `tc` and the condition
string `c` satisfy `c in tc` **or** `tc in c` (plain substring, both directions, lowercased).
**Evidence (Long COVID, the worst case):** 22 trials matched, only **10 genuine**.
- **Over-match (12 acute-COVID admitted):** `NCT04359901`, `NCT04382924`, `NCT04385199`, … — 2020
  hospitalization/ARDS trials, matched only because `"covid"` ⊂ `"long covid"`.
- **Under-match (7 genuine post-COVID dropped):** `NCT05104749`, `NCT05633407`, `NCT05445427`, … —
  tagged `"post-acute covid-19 syndrome"` / `"post covid syndrome"`, which don't substring-match
  `"long covid"`.

Same mechanism elsewhere: dysautonomia matched **7 of 66** valid trials (30 orthostatic/autonomic
dropped); ME/CFS admits generic cancer/renal "fatigue" via `"fatigue"` ⊂ `"chronic fatigue syndrome"`.
**Impact on her:** her shared Long-COVID study is **~half acute-COVID contamination while missing
real post-COVID trials** — those acute trials are *why* M1 reproduced her retro 21/21. **Root cause
is shared across all conditions** (the substring-both-directions match).
**Fix/status:** we replaced it with a per-condition keyword classifier (`seed_terms.CLASSIFY`);
recommend she do the same. Full detail: condition_filter_audit.md.

### A2 — `notbinary` label is `value / N` for continuous endpoints
**Where:** `Experiment` (notbinary preset) sets `avg_potential_outcome = value / N` for **every**
endpoint (`value/100` if the unit says percent). Correct as a response *rate* for binary/count;
**meaningless for a continuous mean** — it mixes the effect size with the arm's sample size.
**Evidence:**

| trial | endpoint | correct value | her label = value/N |
|---|---|---|---|
| NCT02499302 | steps/day | 7217 (n=21) | **343.7** ⚠️ |
| NCT04158427 | VAS fatigue 0–100 | 72.8 (n=5) | **14.6** ⚠️ |
| NCT05559021 | FIQ score | 44.07 (n=8) | **5.5** ⚠️ |

**Impact on her:** **~84% of current sidecar rows are continuous**, so most evaluation labels in the
notbinary path are affected (only ~52% land in [0,1], extremes to ~387). This is the prediction *target* - arguably
the highest-severity item. Present in her own notbinary study. The `binary` preset is not an escape:
it collapses the set 255 -> 21 train+val (-92%) because these symptom conditions use continuous primaries.
**Fix/status:** we keep every trial and add a model-ready **label sidecar** (`endpoint_type` +
`clean_outcome` [raw mean for continuous] + `scale_proportion`). Pinned `naturalv2` does not read
that sidecar by itself, so she would need to consume it explicitly or fix the normalization
(use the mean / standardize). Full detail: label_normalization.md.
**UPDATE (main `7a2e006`):** fixed upstream in `6390055`. Her new `_normalize_outcome_value()` returns
the **raw value** for `MEAN`/`MEDIAN`/`LEAST_SQUARES_MEAN` (divides by N only for count/binary) — exactly
our `clean_outcome`. Verified end-to-end: all 5 core-5 trials build with correct means (lithium
`[-11.3, -9.0]`, fluvoxamine `-47.3`, cyclobenzaprine `-2.2`). **Sidecar now redundant for the core-5.**
One residual: her `_COUNT_PARAM_TYPES` still includes `NUMBER` and `COUNT_OF_UNITS`, so those are still
÷N (our narrower `{COUNT_OF_PARTICIPANTS}` disagrees — the "impossible rate" edge). No core-5 endpoint
hits it; **unverified against the broader 19** — the sidecar is retained as the tool to check that.

### A3 — Factorial arms named `"X/Placebo"` are silently dropped
**Where:** `check_nonplacebo` decides "is this a real treatment arm?" by the arm **title**. Factorial
arms named `"X/Placebo"` contain the word "Placebo", so they're classified as placebo and dropped.
**Evidence (LIFT, NCT06366724, a 2×2 factorial of LDN × pyridostigmine):**

| LIFT arm | what it is | her pipeline keeps it? |
|---|---|---|
| Pyridostigmine/LDN | the stack (both) | ✅ |
| Pyridostigmine/Placebo | **pyridostigmine main effect** | ❌ dropped |
| Placebo/LDN | **LDN main effect** | ❌ dropped |
| Placebo/Placebo | control | ✅ dropped (correct) |

**Impact on her:** run naively, LIFT keeps **only the stack** — the **LDN-alone arm (the highest
corpus-signal target, 5183 distinct authors) is lost.** Affects **any** 2×2 factorial, silently.
**Fix/status:** relabel factorial arms to their non-placebo component before her filter
(`"Placebo/LDN" → "Low-Dose Naltrexone"`); implemented in `long_covid_eval.py::relabel`. Full detail:
long_covid_focus.md.
**UPDATE (main `7a2e006`):** fixed upstream in `6390055`. Title-string `check_nonplacebo` is retired;
arms are now classified by CT.gov `ArmGroupType` via `check_arm`, which keeps `EXPERIMENTAL` +
`ACTIVE_COMPARATOR` and drops `PLACEBO_COMPARATOR`/`NO_INTERVENTION`/`SHAM_COMPARATOR`. A factorial
`Placebo/LDN` arm is typed `EXPERIMENTAL`, so it survives natively — **our relabel is redundant** (given
CT.gov populated `ArmGroupType`, which well-formed records do). Caveat: LIFT still depends on A4 (open)
to be pullable into the test universe at all.

### A4 — `status:act` excludes recruiting trials, and the pinned code ≠ her shared study
**Where:** test universe `aggFilters=studyType:int,results:without,status:act`, where `status:act`
= "Active, not recruiting" only** — it drops every still-**recruiting** trial.
**Evidence:** LIFT (`overallStatus = RECRUITING`) is dropped solely by `status:act`. For Long COVID,
strict `status:act` = 13 test trials vs relaxed = 50 (+37). Critically, her *shared* study's 51-trial
test set matches the **relaxed** universe **48/51** but strict only **13/51** — so **the pinned repo's
`status:act` does not reproduce her own shared test set** (older/edited config or manual curation).
**Impact on her:** in-flight trials (incl. LIFT) can't be prediction targets under the pinned code,
and the repo disagrees with her published test set. (Training is unaffected — recruiting trials have
no labels.) **It's a tradeoff, not purely a defect** (recruitment-complete = more stable target), but
the pinned-vs-shared mismatch needs resolving. **Fix/status:** we provide a recruiting-inclusive
relaxed universe; open question for Nikita: which status selection is canonical. Full detail:
test_universe_status.md.

### A5 — The default estimator cannot run a `notbinary` study
**Where:** `conf/estimate_ate.yaml` defaults to `estimator: natural_ipw`, whose `conditional_extraction`
stage enumerates multiple-choice options over the outcome (`experiment.options[<outcome>]`).
**Evidence:** for lithium (`NCT05618587`) every covariate has options (`Age: [Adult, Other]`,
`Sex: [Male, Female]`, `treatment_taken: [Lithium]`) but both continuous outcomes are **empty lists** —
there is nothing to enumerate for a 1–49 severity score. `_prepare_for_conditional_extraction` returns
zero options and the stage dies on `num_workers // len(interleaved_options)`:
```
ZeroDivisionError: integer division or modulo by zero
```
**Impact on her:** the `noparallel_notbinary` preset is the one that keeps continuous-endpoint trials
(the `binary` preset collapses the set −92%, see A2) — so **her default config cannot estimate her own
notbinary studies**, and the failure is a bare `ZeroDivisionError` with no explanation of the cause.
**Fix/status:** NATURAL-**MC** samples `(T, Y)` instead of enumerating, which is what a continuous
endpoint needs. Her config knows this — `# sample_ty: Use this for NATURAL-MC in place of
conditional_extraction` — but it is commented out and the default is IPW. We added
[`estimate_mc.yaml`](../../conf/estimate_mc.yaml) (`sample_ty` in,
`conditional_extraction` out, `estimator: natural_mc_oi`); it runs to completion. Recommend she either
flip the default for notbinary presets or raise a readable error.

### A6 — Change-from-baseline labels compared against absolute predictions
**Where:** the trial label is a *change*, but the outcome description (which is what the LLM is shown)
describes an *absolute* scale — nothing marks the difference.
**Evidence (lithium, `NCT05618587`):**

| field | value |
|---|---|
| title | `Fatigue Severity Scale` |
| description | `Score range 1-49 with higher values signifying worse outcome` |
| paramType | `MEAN` |
| reported value | **−11.3** |

−11.3 is impossible on a 1–49 scale, so the trial reported an 11.3-point *improvement from baseline*
under an absolute-scale description. NATURAL reads the description, samples an **absolute** severity
from patient reports, and compares it to a **change**:

| outcome | predicted (95% CI) | trial label | abs error |
|---|---|---|---|
| Fatigue Severity Scale | 12.86 (7.31–18.40) | −11.3 | 24.16 |
| Brain Fog Severity Scale | 10.66 (9.31–12.01) | −9.0 | 19.66 |

The predictions are *plausible absolute severities*; the sign flip is the units mismatch, not model
error. **Impact on her:** any trial reporting change under an absolute-scale description scores as a
large error regardless of how good the estimate is — it silently penalises the model on the benchmark.

**Confirmed by re-run.** We rewrote the outcome description to state the change quantity and its sign
convention (`fix_change_outcomes.py`) and re-ran:

| outcome | before (absolute) | after (change scale) | truth | error |
|---|---|---|---|---|
| Fatigue | +12.86 | **−28.47** (−38.6, −18.4) | −11.3 | 24.16 → **17.17** |
| Brain Fog | +10.66 | **−33.89** (−34.9, −32.9) | −9.0 | 19.66 → **24.89** |

**The units bug is fixed** — both predictions are now negative, on the change scale, in the right
direction. **Accuracy is not** — mean absolute error moved 21.9 → 21.0 (noise), and the model now
*overshoots improvement* (−33.9 vs −9.0, on a scale whose maximum possible improvement is −48). Those
are two separate problems and only the first was addressed here; the overshoot is consistent with
Reddit selection bias (people post dramatic outcomes) and with a 7B model on n=9/32.
**Fix/status:** detection fixed our side (C3). Upstream this is a **design question, not a patch** —
rewriting the description is one option; handling it in `Experiment`, or excluding such trials, are
others. Raise as an issue rather than a PR.

### A8 — Bootstrap intervals are implausibly tight
**Where:** `bootstrap_size: 10` in `conf/estimate_ate.yaml`, resampling per-report values that are
already computed.
**Evidence:** Brain Fog predicted **−33.89 with a 95% CI of (−34.87, −32.92)** — **±1 point from 32
noisy Reddit reports**, and the interval misses the truth (−9.0) by 24 points. The Fatigue interval
(−38.6, −18.4) from n=9 is wider and more believable, so the tightness scales the wrong way with n.
**Why it matters:** the bootstrap captures sampling variation *among extracted values*, not uncertainty
in whether the extractions are right — which is the dominant error source when an LLM infers an
outcome from a Reddit post. **Overconfident intervals are worse than wrong point estimates on a
benchmark**: they make a model look reliably wrong rather than uncertain, and any downstream
calibration or coverage metric inherits the error.
**Root cause — revised.** We first read this as `bootstrap_size: 10` being too small. Going back to
the extracted rows shows something more direct: **21 of the 32 Brain Fog values are identical
(−35.0)**, because they are comments on the same recovery threads and inherited that thread's
outcome ([A11](#a11--a-comment-inherits-the-outcome-of-the-thread-it-replies-to)). Resampling rows
that are nearly all the same number returns nearly the same number, whatever B is. That also explains
why n = 9 gave the *wider* interval — Fatigue's values are more spread, not less clustered by luck.
`bootstrap_size: 10` is still too small, and [A10](#a10--author-is-discarded-so-prolific-posters-count-as-many-patients)
still breaks independence, but neither is the leading term.
**Fix/status:** open, needs Nikita's view — she may already treat these as descriptive rather than
calibrated. Raising B is worth doing but will not fix this on its own: an interval computed over
degenerate values is precise about the wrong thing. Worth checking coverage across the full benchmark
before trusting any interval.

### A7 — `detokenize=False` kills a hosted vLLM server
**Where:** `conditional_extraction.py` hardcodes `detokenize=False` on every `prompt_logprobs` call
(lines ~509 and ~809), with the note "avoid vLLM detokenizing logprob token ids (OverflowError on some
models/TP setups)".
**Applicability:** fine for her in-process `VLLMModel`. Against a **hosted** vLLM OpenAI server
(`model@probs_model=hosted_vllm`) the serving layer still builds `decoded_token` for each prompt
logprob and has nothing to detokenize with. Because it is hardcoded, a config override cannot reach it.
**Fix/status:** fixed with a guard — `_detokenize_kwargs(llm)` sends the flag only when `llm` is an
in-process `VLLMModel`, so her path is unchanged and the hosted path stops crashing. (Our first patch
simply flipped the hardcode to `True`; that would have broken her in-process runs, so it is *not* what
we propose.) PR-able as-is.

### A9 — Treatment matching drops 3-character drug abbreviations
**Where:** two filters, same threshold. `build_treatment_automaton` skips any canonical form of
`<= 3` chars ("Very short canonical forms … generate too many false positives (e.g. 'mg', 'ml')"),
and `RedditCurateStage._prepare_registry_config` independently skips aliases with `len(alias) <= 3`.
**Evidence (LDN, the most-discussed Long-COVID treatment, in our 2.45M-document corpus):**

| | documents |
|---|---|
| spell out "naltrexone" | 6,368 |
| use the bare abbreviation "LDN" | 25,619 |
| **never spell it out — reachable only via "LDN"** | **23,950** |
| **share of LDN evidence lost** | **79%** |

**Root cause is not the threshold.** `extract_mentions` runs `automaton.iter()` over canonicalised
text with **no word-boundary check**, so a 3-character pattern would match inside ordinary words —
"ldn" hits "cou**ldn**'t", "wou**ldn**'t". (We hit exactly this in our own coverage tooling and fixed
it with `\b` anchors.) The length filter is a *workaround for substring matching*, so **raising the
threshold alone would trade silent under-matching for silent over-matching** — we deliberately did
not do that.
**Impact on her:** any treatment whose common name is a ≤3-char abbreviation is largely invisible to
curation — LDN, VNS, and similar. The trial still runs and reports a number, so the loss is silent.

**Fix/status: FIXED, PR-ready.** Aho-Corasick already reports the *end position* of every match and
`extract_mentions` was discarding it (`for _, canonical_alias in ...`). We use it to check the
characters either side and reject a match unless both are non-alphanumeric or the string edge, then
lower the floor from `> 3` to `>= 3` in both places. 1–2 character fragments ("mg", "ml") stay
excluded, so the original rationale is preserved.

```python
for end_idx, canonical_alias in automaton.iter(canonical_text):
    start_idx = end_idx - len(canonical_alias) + 1
    before = canonical_text[start_idx - 1] if start_idx > 0 else " "
    after = canonical_text[end_idx + 1] if end_idx + 1 < len(canonical_text) else " "
    if before.isalnum() or after.isalnum():
        continue
    found_mentions.add(canonical_alias)
```

About ten lines across `helpers.py` and `curate.py`. Verified on seven cases: "LDN" matches at
sentence start, before punctuation and inside parentheses; "couldnt", "wouldnt" and "aldnamide" do
not; multi-word aliases are unaffected.

**Measured effect:** rows matching the LDN alias set on our contextualised corpus go from **31,567 to
138,618 — 4.4×**. Note this changes matching for every trial, so it wants landing before any frozen-
configuration evaluation run.

**Still open, separately:** this fixes *matching* an alias once supplied. **Discovering** which
aliases to supply is unsolved — upstream expands synonyms by web search, we hand-seed, and both fail
silently on misspellings and unlisted brand names. Worth noting `naturalv2` already imports
`get_drugbank_aliases` and exposes `Experiment.drugbank_names`, but expects a DrugBank
`full_database.xml.gz` we have never supplied — the "DrugBank data file not found" warning in every
run. That is an ontology-backed synonym source already wired in and unused.

### A10 — `author` is discarded, so prolific posters count as many patients
**Where:** the raw corpus carries `author`, but neither the contextualised dataset nor the curated
output retains it. Nothing after stage 2 can deduplicate, cluster or weight by patient.
**Evidence (lithium's evidence pool, pre-filter):**

| | |
|---|---|
| documents mentioning lithium | 1,012 |
| distinct authors | 365 |
| median documents per author | **1** |
| **maximum** | **99** |
| top 10 authors' share | **30.3%** |
| single most prolific author | **9.8% of all the evidence** |

One person supplies a tenth of it; ten people supply nearly a third. We cannot measure the
post-filter concentration *because* the field is gone — which is itself part of the argument.
**Impact on her:** every surviving report becomes one synthetic patient, so 99 posts from one person
become 99 "patients". That breaks the independence assumption behind both the estimator and the
bootstrap, and is a plausible **second mechanism behind A8** — under clustering this severe the
effective sample size is far below the nominal count, so intervals are narrower than the data
warrants. It also amplifies selection bias, since prolific posters are more engaged and plausibly
evangelising something that worked.
**Fix/status:** open. The mechanical half is small — retain a **salted hash** of the author through
contextualisation and curation, which gives clustering without storing handles. The statistical half
is a real choice: cluster-bootstrap by author (most defensible first move), weight by 1/nᵢ, or
deduplicate. With a median of one document per author, the tail is doing the damage, so a cap may
beat a global reweighting.

### A11 — A comment inherits the outcome of the thread it replies to
**Where:** `sources/reddit/stages/curate.py` (`fmt_comment`, ~line 834) embeds the **full text of the
initial post** in every comment's report. The extraction prompt
(`prompts/templates/sample_ty.yaml`) then asks for the value *"the individual described in the
report"* would give. A comment's report describes **two** individuals — the original poster and the
commenter — and the original poster is described far more fully. The model answers about the wrong
one.

**Evidence.** Reading back the extracted rows from both of our runs:

| | Brain Fog (lithium) | Fatigue (lithium) | Functional Capacity (LIFT) |
|---|---|---|---|
| evidence rows | 32 | 9 | 3,127 |
| distinct **threads** behind them | 7 | 4 | 494 |
| rows whose **own comment** names the treatment | 4 | 1 | 735 |
| rows where **only the thread** names it | **28 (88%)** | **8 (89%)** | **2,375 (76%)** |
| median length of the comment itself | 99 chars | 54 chars | 142 chars |

For lithium's Brain Fog, **21 of 32 rows carry the identical extracted value, −35.0**. The three
largest source threads are recovery announcements; the biggest is titled *"Long Covid Recovery to
90% — Antihistamine Treatment"*, and it alone supplies 11 rows. The comments scored −35 read, in
full, *"Yes! Those are horrible when they happen.. Thanks for sharing."*, *"Hey! We're you able to
eventually ween off the antihistamines?"* and *"It's cheap. Will give it a try. I take generic
zyrtec and Benadryl at night"*. None reports an outcome; none is about lithium. Each is a comment on
a thread whose **title** announces 90% recovery, and each is scored as near-total recovery.

The separation is clean: the 4 rows whose own text discusses lithium average **−2.5**, while the 28
that do not average **−33.2**. The trial's answer is **−9.0**. The rows that are actually about the
drug are the closest to correct; the estimate is driven by the ones that are not.

**Impact on her:** this is a measurement error, not a modelling one — `Y_i` is not the outcome of the
individual the row claims to represent. It propagates three ways:

1. **Point estimates inflate toward whatever the thread announced.** Long, popular threads are
   disproportionately recovery stories, and their comment counts are exactly what makes them supply
   many rows.
2. **Intervals collapse.** With 21 of 32 values identical, every bootstrap resample returns the same
   number. This is a **more likely cause of A8 than `bootstrap_size` alone** — the values have almost
   no spread to resample.
3. **The effective sample is the thread, not the comment.** 32 rows from 7 threads is not n = 32 for
   any purpose. Compounds with [A10](#a10--author-is-discarded-so-prolific-posters-count-as-many-patients).

It also means we have been attributing the wrong *treatment*: `treatment_taken` reads `Lithium` on
comments discussing zyrtec and Benadryl.

**Fix/status:** open, and we think this is the highest-value thing in this document. Directions, in
rough order of cost:

- **Require the report to discuss the treatment.** The cheapest filter — drop rows whose own
  `report_text` never mentions the matched treatment. Would have removed 88% of lithium's evidence,
  which is the point: that evidence was not evidence.
- **Name the subject in the prompt.** Ask for the outcome of *the comment's author*, and mark the
  initial post explicitly as background about a different person.
- **Reconsider whether thread context belongs at all** for the outcome question. It helps the
  relevance and treatment questions and actively harms this one; it need not be the same context for
  every question.
- **Aggregate to the thread, or cluster-bootstrap by it**, so 11 comments on one recovery post cannot
  count as 11 patients.

We have not implemented a fix — the choice among these is a research decision, not a mechanical one,
and it is Nikita's call which belongs upstream.

### A12 — Extracted outcome values are never range-checked
**Where:** `sample_ty` asks for *"a single number … on the same scale as the outcome description"*,
and whatever comes back is parsed and used. Nothing compares it against the range the description
states.

**Evidence (LIFT, FUNCAP55, a 0–55 scale, 3,127 rows):**

| | |
|---|---|
| values outside 0–55 | **462 (14.8%)** |
| literal `inf` | 2 |
| maximum | **4,444,000** |
| minimum | −55 |
| mean of finite values | **+1,479.6** |

A single value of 4.4 million on a 55-point scale moves the mean by more than a thousand points. The
mean is not a summary of anything.

**Impact on her:** silent and total. There is no error, no warning, and the resulting estimate is a
well-formed number. Any trial whose extraction produces one bad parse gets a meaningless answer that
looks exactly like a good one. Trials with wide or unusual scales are the most exposed.

**Fix/status:** open, and unlike A11 this one is mechanical. The `Experiment` already knows the
endpoint's range wherever the description states it — the same source our own `C3` range-check uses.
Rejecting (not clamping — a clamp turns a parse failure into a confident extreme) out-of-range
values, and logging the rate, would be a small patch. The rate itself is a useful extraction-quality
signal.

### A13 — `NaturalMC`'s docstring says not to use it for APOs, which is what we do
**Where:** `naturalv2/estimators/natural_mc.py`, the class docstring:

```
class NaturalMC:
    """NATURAL Monte Carlo Estimator for individual treatment responses.
    TODO: Do not use for APOs; off-the-shelf estimators do not trivially extend to APOs.
```

We run `ate: False`, which is an APO per arm, with `estimator: natural_mc_oi`.

**This is a question, not an accusation.** Reading the `oi` branch, it imputes the outcome under each
treatment value for every unit and the caller averages those — which is the standardisation estimator
(g-formula) for `E[Y(t)]`, and correct for an APO as far as we can tell. So either the caution is
about the `ipw` branch specifically, or it is about a subtlety we have not spotted. **We would rather
ask than guess**, because every number in this document depends on the answer.

The `ipw` branch does look wrong for APOs, which may be what the note means. It computes
`all_ites[t, :] = individual_outcomes * t_mask`, zeroing every unit that did not take arm `t`, and
`estimate_ate.py` then averages over *all* units. That returns `E[Y · 1{T=t}]`, not `E[Y(t)]` —
shrunk toward zero by arm `t`'s share of the sample. With a single-arm experiment the mask is all
ones and the bug is invisible, which is why our lithium run is unaffected.

**Questions for Nikita:** is `natural_mc_oi` sound for `ate: False`? Is the TODO about the `ipw`
branch, about both, or about something else? If MC is unsuitable for APOs generally, what should a
continuous-endpoint APO study use, given `natural_ipw` cannot run one at all (A5)?

**Fix/status:** open — blocked on her answer, and the highest-leverage question in this document,
since it is upstream of every estimate we have produced.

### A14 — A bare `except:` turns any fit failure into silent `NaN`
**Where:** `naturalv2/estimators/natural_mc.py`, `get_individual_treatment_effects`:

```python
try:
    model.fit(data)
    ...
except:
    all_ites = np.full((self._num_treat, len(observational_data)), np.nan)
return all_ites
```

**Why it matters:** a bare `except` catches everything — a singular design matrix, a dtype problem, a
`KeyboardInterrupt` — and returns `NaN` with no log line and no exception. The caller cannot tell a
failed fit from a fit that legitimately produced nothing, and the failure surfaces much later as a
missing result that looks like a data problem rather than a modelling one.

This is not hypothetical for studies like ours. The outcome model is an unregularised
`LinearRegression` over 8 discretised covariates plus treatment, and our Fatigue outcome had **n = 9
reports** — as many parameters as rows. That is exactly the regime where a fit degenerates, and the
place where a silent `NaN` is most likely and least noticeable.

**Our runs did not hit it** — both lithium estimates came back as numbers, so the fits succeeded.
That is luck rather than evidence it is safe.

**Fix/status:** open, small, and independent of everything else — catch the specific exceptions,
log which unit and outcome failed and why, and let anything unexpected propagate. Worth doing before
any run large enough that nobody reads the per-trial output.

---

## B. Data-source gotcha (CT.gov search)

### B1 — `query.cond="COVID"` misses `SARS-CoV-2` / `PASC` / `Post-COVID-19 Condition` tags
**Where:** our condition-scoped download layer (`run_study.py`, `m3_pool.py`, `relaxed_test_universe.py`)
passes `query.cond=<scope>`. A trial tagged `"Post-Acute Sequelae of SARS-CoV-2"` has **no "COVID"
substring**, and CT.gov does **not** auto-expand `COVID` to it — so the scope silently misses it.
**Evidence:** broadening the Long-COVID scope to
`COVID OR SARS-CoV-2 OR PASC OR Post-Acute Sequelae of SARS-CoV-2 OR Post-COVID-19 Condition OR
Chronic COVID OR Long-haul COVID` recovered **+4** `results:with` and **+33** `results:without`
genuine Long-COVID trials.
**Applicability to her:** her own `download_clinical_trials` pulls the *whole* corpus with **no**
condition filter, so she doesn't hit B1 in the download — **but** her matcher (A1) drops those same
SARS-CoV-2/PASC-tagged trials anyway, and **anyone scoping a pull by condition string hits B1
directly.** Worth a one-line caution in her docs.
**Fix/status:** broadened scope wired into `seed_terms.py`. Full detail:
long_covid_focus.md.

---

## C. Bugs in our own code (fixed)

### C1 — `inject_one` ambiguous `None` return → `Study` reads a missing file
**Where:** `build_augmented.py::inject_one`. It returned `None` for **two** different outcomes:
"no usable numeric arm → nothing written" **and** "written but no result date". The caller then
re-added a never-written trial to the `Study` (because the schema still had arms), and her
`build_exp` crashed trying to read the absent JSON.
**Trigger:** `NCT04574050` — the first paper extraction with arms but no numeric values, surfaced by
the broadened-scope re-run.
**Fix/status:** `inject_one` now returns `False` when nothing is written; the caller appends only
when a file was actually written. **Fixed** (commit `d74d713`).

### C3 — `is_change` misses change endpoints whose title says nothing about change
**Where:** `build_labels_sidecar.py::is_change` regexes the outcome **title** for `change`,
`from baseline`, `reduction in`, etc.
**Evidence:** lithium's endpoints are titled exactly `Fatigue Severity Scale` and `Brain Fog Severity
Scale` — no change wording anywhere in title *or* description — yet both report change values
(−11.3, −9.0). Our heuristic passes them through as absolute scores, so the sidecar's
`is_change_from_baseline` flag is **false-negative on the very trial that motivated it** (A6).
**Better detector:** the description states the scale (`Score range 1-49`); a reported value outside
that range cannot be absolute. Range violation is checkable from data we already parse, and unlike
title wording it cannot be worded around. Titles remain useful as a secondary signal.
**Status:** open — implement the range check, then re-run the core-5 and see how many labels move.

### C2 — Misc implementation slips (all fixed during development)
- **EPMC full-text 404** — URL had a double `/PMC/` segment; corrected to the bare PMCID path
  (`{base}/{pmcid}/fullTextXML`).
- `enrollmentInfo` missing `type` — some CT.gov records fail her pydantic model; wrapped
  `ClinicalTrial.from_json_file` in try/except and skip.
- **Synthetic `resultsSection` ValidationError** — her model requires `participantFlowModule` +
  `baselineCharacteristicsModule`; we stub them empty and carry only `outcomeMeasuresModule`.
- **Windows `cp1252` UnicodeEncodeError** — em-dash / non-ASCII in trial titles crash console prints;
  use ASCII-safe output.

### C4 — The analysis scripts are not portable
**Where:** `patientpunk/analysis/`. They were written with `trial_superset/` as the working
directory and a local `data/` mirror of the S3 prefixes, so paths inside them are relative to that
layout.
**Impact:** none on results — they produced the study we are running, and are correct. But someone
picking them up cannot run them from a fresh checkout without repointing paths, which makes
re-deriving the coverage numbers harder than it should be.
**Fix/status:** open, low priority. The run-time pipeline (`patientpunk/scripts/`, `serving/`) has
already been made path-independent; the analysis tooling has not. Documented in
`patientpunk/analysis/README.md` so nobody loses an afternoon to it.

---

## D. Serving / ops gotchas (running the pipeline, not bugs in anyone's code)

Both cost real time on the first end-to-end run; neither is obvious from the failure message.

### D1 — `gpu_memory_utilization` starves `prompt_logprobs`
**Why this stage needs a GPU at all:** `inclusion_prob` scores a prompt *we* supply — teacher-forced
token probabilities — and no hosted chat API returns those; they give probabilities only for tokens
the model generates. That one stage is what forces a self-hosted vLLM server. Everything else in the
pipeline runs against an ordinary API, which is why the GPU is worth acquiring late (see D3).
**Symptom:** vLLM returns `EngineCore encountered an issue` and the container dies; every later request
then reports `Cannot connect to host`, which reads like the node was preempted. It is not.
**Cause:** `prompt_logprobs` needs a **transient** `[prompt_tokens × vocab]` logits tensor (~3.5 GB for
a 5.7k-token prompt at Qwen's ~152k vocab — `5,700 × 152,064 × 4 bytes`) allocated **outside** the
pool vLLM preallocates. The usual `gpu_memory_utilization: 0.90` leaves ~10% free and the allocation
fails.
**The trap:** it is a *fraction*, not an amount, so **moving to a bigger card does not help**. `0.90`
reserves 21.6 GB of a 24 GB card and 72 GB of an 80 GB one: the pool scales with the card, the
leftover stays about a tenth, and the 3.5 GB tensor is a fixed cost that has to fit in that leftover.
Going 24 GB → 32 GB simply bought a bigger KV cache and the same ~10% headroom. We burned five GPU
cycles on hardware before lowering the fraction — the fix is to make the number *smaller*, which is
the opposite of what an out-of-memory error usually suggests. Serving config that works:
`GPU_MEM_UTIL=0.55`, `--max-num-seqs 1`, `--max-model-len 8192`.
**Debugging lesson:** a single-request smoke test passed throughout, because a **7-token** prompt needs
~4 MB of logits and never exercises the failure. `patientpunk/serving/probe_long_logprobs.py`
now sends one **realistic-length** report before the pipeline runs, so a bad serving config costs
seconds instead of a full pass. Size the probe like production, not like a hello-world.

### D2 — Stage caches survive a pipeline-shape change
**Where:** stages resume by reading `outputs/results/<NCT>_<exp>/*.csv` if present (a genuine feature —
it saved the whole OpenRouter spend across several retries).
**Two ways it bites:** (1) a crashed stage leaves a **0-byte CSV** that the resume logic then reads →
`pandas.errors.EmptyDataError: No columns to parse from file`; (2) after changing which stages run,
a cache written by the *old* shape is stale — `SampleTYStage` writes its CSV **before** calling
`discretize_ty()`, so the `_discretized` columns live only in the frame it passes downstream, and a
stale last-stage cache yields `KeyError: treatment_taken_discretized`.
**Fix:** delete 0-byte CSVs before every run (the runner now does), and drop caches for any stage
downstream of a composition change.

### D3 — The GPU is held for the whole run but used by one stage
**Where:** our runner (`patientpunk/serving/run_estimate_chain.py`) acquires a GPU job, runs the
entire pipeline against it, then cancels.
**Evidence (the LIFT attempt):** the first outcome spent **two hours** in the OpenRouter stages —
`relevance_filter` alone scored 42,456 reports — before touching the GPU at `inclusion_prob`. The
card was idle for roughly 90% of that, billing the whole time. Compounded by the scheduler placing a
single-GPU request on a six-GPU node at $4.14/hr, we paid something like 50× what the GPU work
needed.
**Impact:** cost only, but it dominates the bill on any large trial. A four-outcome LIFT run would
have been ~$33 of GPU against roughly $5 of actual use.
**Fix/status:** open. The runner should acquire the GPU immediately before the logprob stages and
release it after, rather than wrapping the whole pipeline. Worth doing before the LIFT re-run.

---

## Cross-cutting note for the Nikita hand-off
A1 and B1 share a root cause: **substring/keyword matching on free-text condition fields is brittle.**
The durable fix is a small controlled vocabulary / classifier for condition assignment instead of
substring tests. A3 was the arm-role instance of this same brittleness and is **now fixed** — she moved
arm classification off the title string onto structured `ArmGroupType`; A1's condition matcher is the
remaining instance. A2 (the highest-severity label issue) is now **mostly fixed upstream**.

**A6/C3 are the same lesson applied to labels:** the trustworthy signal is the *structured* field
(paramType, the scale range in the description), never free-text titles. Our own `is_change` regex made
exactly the mistake we flagged in hers — it reads the title, and lithium's title says nothing about
change. Prefer checks that data cannot word around.

**Open after the re-pin and the first end-to-end run:** **A1**, **A4**, **A5**, **A6** (plus **C3** on
our side). A6 is the one that matters most right now — until change-from-baseline labels are detected,
every trial scores a large error for reasons unrelated to how well NATURAL actually estimates.
