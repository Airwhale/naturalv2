#!/usr/bin/env python3
"""Launch the vLLM-on-Dispersed smoke server (Qwen2.5-7B-Instruct) for NATURAL's probs_model.

SAFE BY DEFAULT: with no flags it prints the exact job spec and exits (--dry-run) — no network
write, no spend. Pass --go to actually create the job, which STARTS BILLING at the GPU's hourly
rate (~$0.35/hr on a 4090). Needs DISPERSED_PUBLIC_KEY / DISPERSED_SECRET_KEY in the env for --go
(dry-run needs neither).

    python dispersed/launch_smoke.py            # dry-run: show the spec, no spend
    python dispersed/launch_smoke.py --go       # create the job (billing starts)
    python dispersed/launch_smoke.py --stop UUID # cancel a job (stops billing)

The image is generic (model chosen via the MODEL env), so the same server image serves the 72B
later — just change DEFAULT_ENV["MODEL"].
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API = os.environ.get("DISPERSED_API", "https://api.dispersed.com")
# Public image; rebuild from ./vllm-server and point PP_VLLM_IMAGE elsewhere to use your own.
IMAGE = os.environ.get("PP_VLLM_IMAGE", "ghcr.io/airwhale/pp-vllm")
TAG = os.environ.get("PP_VLLM_TAG", "latest")
PORT = int(os.environ.get("PP_VLLM_PORT", "8000"))
# 32GB, not the 24GB 4090: prompt_logprobs makes vLLM materialise a [prompt_tokens x vocab] logits
# tensor (Qwen vocab ~152k), which OOM-crashed the engine on a 4090 once ~15GB went to weights.
GPU_NAME = os.environ.get("PP_GPU_NAME", "NVIDIA RTX 5090")
MIN_VRAM_GB = int(os.environ.get("PP_MIN_VRAM_GB", "32"))
DEFAULT_ENV = {
    "MODEL": os.environ.get("PP_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
    # prompt_logprobs aborts EngineCore when a prompt gets CHUNKED, so the whole config exists to
    # guarantee one prompt = one chunk: inputs are capped at 20k chars (~5.7k tokens) upstream,
    # max_num_batched_tokens >= max_model_len, and chunked prefill is off. 8192 also keeps the
    # [tokens x 152k vocab] logits tensor near 5GB, which fits beside ~15GB of weights.
    "MAX_MODEL_LEN": os.environ.get("PP_MAX_MODEL_LEN", "8192"),
    # 0.55, NOT the usual 0.90: prompt_logprobs needs a TRANSIENT [prompt_tokens x 152k vocab]
    # logits tensor (~3.5GB at 5.7k tokens) that lives OUTSIDE the pool vLLM preallocates. At 0.90
    # the KV cache expands to fill the card and that allocation OOMs -- which is why moving 24GB ->
    # 32GB changed nothing: the fraction just bought more KV cache, never headroom.
    "GPU_MEM_UTIL": os.environ.get("PP_GPU_MEM_UTIL", "0.55"),
    "EXTRA_ARGS": "--max-num-seqs 1 --max-num-batched-tokens 8192 --no-enable-chunked-prefill",
}


def _signed_request(method, path, body=None, *, pk, sk, query=""):
    body_bytes = (json.dumps(body, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode("utf-8")
                  if body is not None else b"")
    body_sha = hashlib.sha256(body_bytes).hexdigest()
    ts = str(int(time.time() * 1000))
    nonce = os.urandom(16).hex()
    canonical = f"{pk}|{ts}|{nonce}|{method}|{path}|{query}|{body_sha}"
    sig = hmac.new(sk.encode(), canonical.encode(), hashlib.sha256).hexdigest()
    headers = {"X-API-Key": pk, "X-Time": ts, "X-Nonce": nonce, "X-Signature": sig,
               "Content-Type": "application/json"}
    req = Request(API + path + (("?" + query) if query else ""),
                  data=body_bytes if body is not None else None, headers=headers, method=method)
    try:
        with urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode() or "{}")
    except HTTPError as e:
        sys.exit(f"HTTP {e.code} on {method} {path}: {e.read().decode(errors='ignore')[:300]}")
    except (URLError, TimeoutError, OSError) as e:
        sys.exit(f"Could not reach Dispersed API: {getattr(e, 'reason', e)}")


def _public_ip():
    try:
        with urlopen("https://api.ipify.org", timeout=10) as r:
            return r.read().decode().strip() + "/32"
    except Exception:
        return None


def _load_keys():
    pk = os.environ.get("DISPERSED_PUBLIC_KEY", "")
    sk = os.environ.get("DISPERSED_SECRET_KEY", "")
    if pk and sk:
        return pk, sk
    env = os.environ.get("PP_ENV_FILE") or os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".env"))  # repo-root .env
    if os.path.exists(env):
        with open(env, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    if k.strip() in ("DISPERSED_PUBLIC_KEY", "DISPERSED_SECRET_KEY"):
                        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    return os.environ.get("DISPERSED_PUBLIC_KEY", ""), os.environ.get("DISPERSED_SECRET_KEY", "")


def build_body(allowed_ip):
    return {
        "task": "PERSISTENT",
        "title": "pp-natural-vllm-qwen7b",
        "gpu_count": 1,
        "min_vram_gb": MIN_VRAM_GB,
        "gpu_name": GPU_NAME,
        "parameters": {"type": "docker", "parameters": {
            "image": IMAGE,
            "tag": TAG,
            "ports": [PORT],
            "allowed_ips": [allowed_ip] if allowed_ip else [],
            "env": dict(DEFAULT_ENV),
        }},
    }


def poll_for_endpoint(uuid, *, pk, sk, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(10)
        runs = _signed_request("GET", "/v1/job-runs", pk=pk, sk=sk)
        runs = runs.get("data", runs) if isinstance(runs, dict) else runs
        run = next((r for r in (runs or []) if r.get("job_uuid") == uuid), None)
        urls = (run or {}).get("node_urls") or []
        if urls:
            u = next((x for x in urls if str(x.get("description")) == str(PORT)), urls[0])
            scheme = "https" if u.get("tls") else "http"
            return f"{scheme}://{u.get('hostname')}:{u.get('port')}"
        print(f"    run status={run.get('status') if run else '?'} ...")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="Actually create the job (STARTS BILLING).")
    ap.add_argument("--stop", metavar="UUID", default=None, help="Cancel a job by uuid (stops billing).")
    args = ap.parse_args()

    if args.stop:
        pk, sk = _load_keys()
        r = _signed_request("PUT", f"/v1/jobs/{args.stop}/cancel", {"reason": "smoke stop"}, pk=pk, sk=sk)
        print(f"cancel {args.stop} -> {r.get('status') if isinstance(r, dict) else r}")
        return

    ip = _public_ip()
    body = build_body(ip)

    if not args.go:
        print("DRY-RUN - this is the exact job spec that --go would POST to /v1/jobs (no spend):\n")
        print(json.dumps(body, indent=2))
        print(f"\n  egress IP for allowed_ips: {ip}")
        print(f"  image: {IMAGE}:{TAG}   model: {DEFAULT_ENV['MODEL']}   gpu: {GPU_NAME}")
        print("\n  Re-run with --go to create it (billing starts). Requires the GHCR package to be PUBLIC.")
        return

    pk, sk = _load_keys()
    if not (pk and sk):
        sys.exit("DISPERSED_PUBLIC_KEY / DISPERSED_SECRET_KEY not found (env or repo-root .env).")
    if not ip:
        sys.exit("Could not detect public IP for allowed_ips; set it manually before --go.")

    print(f"Creating PERSISTENT vLLM job: {IMAGE}:{TAG}  model={DEFAULT_ENV['MODEL']}  gpu={GPU_NAME}")
    created = _signed_request("POST", "/v1/jobs", body, pk=pk, sk=sk)
    uuid = created.get("uuid") if isinstance(created, dict) else None
    if not uuid:
        sys.exit(f"No job uuid in response: {created}")
    print(f"  job uuid: {uuid}  status: {created.get('status')}  (BILLING STARTED)")
    print("  waiting for node_urls (reachable host:port)...")
    endpoint = poll_for_endpoint(uuid, pk=pk, sk=sk)
    if not endpoint:
        sys.exit(f"Timed out waiting for node_urls. Check job {uuid}; stop it with --stop {uuid} if needed.")
    print(f"\n  vLLM reachable at: {endpoint}")
    print(f"  OpenAI base_url for NATURAL: {endpoint}/v1")
    print(f"  point probs_model at: hosted_vllm/{DEFAULT_ENV['MODEL']}")
    print(f"\n  stop it when done: python dispersed/launch_smoke.py --stop {uuid}")


if __name__ == "__main__":
    main()
