"""Seed `treatment_common_names` so curate can find a trial's drugs in patient text.

The `treatment_synonyms` stage normally fills this by web search (OpenAI-only). Without it, curate
matches the raw arm labels -- and a factorial label like "Placebo/LDN" appears nowhere in patient
text, so curation silently returns zero records.

Aliases are supplied as `ARM=alias,alias,...`, repeatable:

    python 02_seed_treatment_aliases.py --nct NCT06366724 \
        --alias "Placebo/LDN=naltrexone,low dose naltrexone,ldn" \
        --alias "Pyridostigmine/Placebo=pyridostigmine,mestinon"

Aliases of 1-2 characters are dropped by the matcher; 3+ are fine since matching became
word-boundary aware (see docs/patientpunk/findings.md, A9).
"""

from __future__ import annotations

import argparse
import os

import yaml

from naturalv2.utils import get_experiment_filepath


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-path", default=os.environ.get("SAVE_PATH", "outputs"))
    ap.add_argument("--nct", required=True)
    ap.add_argument("--experiment", default="noparallel_notbinary")
    ap.add_argument("--source", default="reddit")
    ap.add_argument("--alias", action="append", required=True, metavar="ARM=a,b,c")
    args = ap.parse_args()

    mapping: dict[str, list[str]] = {}
    for spec in args.alias:
        arm, _, csv = spec.partition("=")
        if not csv:
            raise SystemExit(f"--alias needs ARM=alias,alias : got {spec!r}")
        mapping[arm.strip()] = [a.strip() for a in csv.split(",") if a.strip()]

    fp = get_experiment_filepath(args.save_path, args.nct, args.experiment)
    doc = yaml.safe_load(open(fp, encoding="utf-8"))

    known = set(doc.get("_treatment_names") or doc.get("treatment_names") or [])
    unknown = [a for a in mapping if known and a not in known]
    if unknown:
        print(f"WARNING: these arms are not in the experiment's treatments {sorted(known)}: {unknown}")

    doc["treatment_common_names"] = {args.source: mapping}
    yaml.safe_dump(doc, open(fp, "w", encoding="utf-8"), sort_keys=False, width=120)

    short = sorted({a for v in mapping.values() for a in v if len(a) < 3})
    print(f"seeded treatment_common_names[{args.source}] in {fp}")
    for arm, al in mapping.items():
        print(f"   {arm}: {al}")
    if short:
        print(f"NOTE: dropped by the matcher (under 3 chars): {short}")


if __name__ == "__main__":
    main()
