#!/usr/bin/env python3
"""Drive one estimate against a dispersed GPU job, and ALWAYS stop the job afterwards.

Waits for the job's endpoint -> waits for vLLM to serve -> probes it -> runs the estimate ->
cancels the job in a `finally`. The auto-stop is the point: billing is hourly and starts at job
creation, so a crash without it leaves a GPU running.

    NCT=NCT05618587 SPLIT=train python run_estimate_chain.py <job_uuid>

Note the GPU is held for the whole pipeline, but only `inclusion_prob` uses it -- most of the
wall-clock is API stages with the card idle. For a large trial that is most of the GPU bill, and
acquiring the card only around the logprob stages is the obvious improvement.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch_gpu as gpu  # noqa: E402

SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts")


def _lst(x):
    return x.get("data", x) if isinstance(x, dict) else x


def endpoint_for(uuid: str, pk: str, sk: str, timeout: int = 900) -> str | None:
    """The reachable host:port lives on the job-RUN, not the job object."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        runs = _lst(gpu._signed_request("GET", "/v1/job-runs", pk=pk, sk=sk)) or []
        run = next((r for r in runs if r.get("job_uuid") == uuid), None)
        for u in (run or {}).get("node_urls") or []:
            if str(u.get("description")) == str(gpu.PORT):
                scheme = "https" if u.get("tls") else "http"
                return f"{scheme}://{u.get('hostname')}:{u.get('port')}/v1"
        print(f"  waiting for node_urls (run status={run.get('status') if run else '?'}) ...")
        time.sleep(15)
    return None


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    uuid = sys.argv[1]
    pk, sk = gpu._load_keys()
    try:
        base = endpoint_for(uuid, pk, sk)
        if not base:
            print("no endpoint appeared; stopping job")
            return 1
        nct = os.environ.get("NCT", "")
        split = os.environ.get("SPLIT", "train")
        print(f"endpoint: {base}\nestimating {nct or '(NCT unset)'} (split={split})\n")
        cmd = ["bash", os.path.join(SCRIPTS, "wait_and_estimate.sh"), base]
        return subprocess.run(cmd, text=True, env={**os.environ}).returncode
    finally:
        try:
            r = gpu._signed_request("PUT", f"/v1/jobs/{uuid}/cancel",
                                    {"reason": "estimate chain finished"}, pk=pk, sk=sk)
            print(f"\n[auto-stop] job {uuid} -> {r.get('status') if isinstance(r, dict) else r}")
        except SystemExit as e:  # _signed_request exits on HTTP error; do not mask the run result
            print(f"\n[auto-stop] FAILED to cancel {uuid}: {e} -- stop it manually!")


if __name__ == "__main__":
    raise SystemExit(main())
