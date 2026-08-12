"""Fail-fast probe: can this server do prompt_logprobs on a REALISTIC prompt?

Every serving failure we hit aborted vLLM's engine on prompt_logprobs, and the only requests that
ever succeeded were tiny -- a 7-token prompt needs ~4 MB of logits and never exercises the failure,
so a hello-world smoke test passes while the pipeline dies. This sends a prompt the size of a real
curated report, so a bad serving config costs seconds instead of a full pipeline pass.

    python probe_long_logprobs.py http://host:8000/v1 [--nct NCT05618587] [--tokens 5000]

With --nct it uses the longest real curated report for that trial; otherwise it synthesises a prompt
of roughly --tokens length, so the probe works before anything has been curated.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import litellm

litellm.suppress_debug_info = True


def _prompt(nct: str | None, save_path: str, approx_tokens: int) -> str:
    if nct:
        pattern = os.path.join(save_path, "reddit_data", "*", f"reddit_{nct}", "*.parquet")
        files = sorted(glob.glob(pattern))
        if files:
            import polars as pl
            df = pl.read_parquet(files)
            col = "report" if "report" in df.columns else df.columns[-1]
            row = df.sort(pl.col(col).str.len_chars(), descending=True).row(0, named=True)
            return row[col]
        print(f"no curated parquet for {nct} under {save_path}; using a synthetic prompt")
    # ~3.5 chars/token, varied text so it does not compress to a trivial token sequence.
    unit = ("I started this treatment about three weeks ago and my fatigue has changed noticeably. "
            "Some days are better than others, and the brain fog is the part I notice most. ")
    return (unit * (int(approx_tokens * 3.5 / len(unit)) + 1))[: int(approx_tokens * 3.5)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", help="vLLM OpenAI base url, e.g. http://host:8000/v1")
    ap.add_argument("--model", default=os.environ.get("PP_MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--nct", default=os.environ.get("NCT"))
    ap.add_argument("--save-path", default=os.environ.get("SAVE_PATH", "outputs"))
    ap.add_argument("--tokens", type=int, default=5000)
    args = ap.parse_args()

    prompt = _prompt(args.nct, args.save_path, args.tokens)
    print(f"probe prompt: {len(prompt)} chars (~{len(prompt) / 3.5:.0f} tokens)")

    try:
        resp = litellm.text_completion(
            model=f"hosted_vllm/{args.model}", prompt=prompt, api_base=args.base,
            api_key="EMPTY", max_tokens=1, temperature=1.0, prompt_logprobs=0,
        )
    except Exception as e:  # noqa: BLE001 - any failure here means "do not start the pipeline"
        print(f"PROBE FAIL: {type(e).__name__}: {str(e)[:300]}")
        return 1

    d = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    slots = (d.get("choices") or [{}])[0].get("prompt_logprobs")
    if not slots:
        print("PROBE FAIL: response carried no prompt_logprobs")
        return 1
    print(f"PROBE PASS: {len(slots)} prompt-token logprob slots returned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
