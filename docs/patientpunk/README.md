# PatientPunk × NATURAL-v2 — integration branch

`main` on this fork is **identical to `nikitadhawan/naturalv2`**. Everything below lives only on this
branch, so the diff *is* the review.

We ran NATURAL end-to-end on a **Long-COVID Reddit corpus** (147k submissions + 2.3M comments,
r/covidlonghaulers + r/LongCovid + r/LongHaulersRecovery, 2020-07 → 2026-06) against the
low-dose-lithium trial `NCT05618587`, serving `probs_model` from a hosted vLLM endpoint rather than
in-process. This branch is what that took.

## What's here

| file | why |
|---|---|
| `naturalv2/sources/reddit/stages/prebuilt_parquet.py` | new `SourceStage`: use an already-built parquet corpus instead of downloading `.zst` archives |
| `conf/source/reddit_prebuilt.yaml` | source config wiring that stage in place of `download_and_clean` |
| `conf/estimate_mc.yaml` | NATURAL-**MC** variant, required for continuous outcomes |
| `conf/model/openrouter.yaml` | OpenRouter provider config, alongside the existing gemini/anthropic/openai ones |
| `naturalv2/pipeline/conditional_extraction.py` | 12-line fix: send `detokenize=False` only when vLLM is in-process |
| `docs/patientpunk/findings.md` | every bug and gotcha we hit, with evidence |

## The three findings worth your time

1. **The default estimator cannot run a `notbinary` study.** `conf/estimate_ate.yaml` defaults to
   `natural_ipw`, whose `conditional_extraction` enumerates options over the outcome. Continuous
   endpoints have none, so it dies on `num_workers // len(interleaved_options)` with a bare
   `ZeroDivisionError`. NATURAL-MC (`sample_ty`) is the path that works — your config says so in a
   comment, but the default is IPW. `conf/estimate_mc.yaml` is that config, made runnable.

2. **`detokenize=False` kills a hosted vLLM server.** Fine in-process; over HTTP the serving layer
   still builds a decoded token per prompt logprob and aborts EngineCore, taking the server down.
   Guarded rather than removed, so the in-process path is untouched.

3. **Change-from-baseline labels are compared against absolute predictions.** `NCT05618587` is titled
   `Fatigue Severity Scale`, described as `Score range 1-49`, and reports **−11.3** — impossible as an
   absolute score. Only `timeFrame` ("Change from baseline to day 21") says it's a change. The model
   is asked for a value "on the same scale as the outcome description", correctly returns an absolute
   severity, and is scored against a change. **This is a design question, not a patch** — see
   `docs/patientpunk/findings.md` A6 for the evidence and the options. Nothing here changes it.

## Serving note (not a code change)

`prompt_logprobs` needs a transient `[prompt_tokens × vocab]` logits tensor (~3.5 GB for a 5.7k-token
prompt at a 152k vocab) allocated **outside** the pool vLLM preallocates. The usual
`gpu_memory_utilization: 0.90` leaves no room and the engine OOMs — and moving to a bigger card does
**not** help, because the fraction just buys more KV cache. We serve at `0.55`. Worth a line in the
docs for anyone pointing `probs_model` at a server.

## Status

Run end-to-end on lithium: corpus → curate → `sample_ty` → `inclusion_prob` → estimate. The estimates
themselves are not meaningful yet (Qwen2.5-**7B** smoke model, n=9 and n=32 evidence units) — the point
of this branch is that the path runs and what it took to get there.
