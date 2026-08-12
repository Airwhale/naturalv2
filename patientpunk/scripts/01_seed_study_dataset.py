"""Seed the condition -> subreddit map that `condition_filter` would normally write.

We skip `condition_filter` (it discovers subreddits through the Reddit API, which needs OAuth and is
unnecessary for a corpus already scoped to known subreddits). But `curate` still needs the map:
`_build_registry` does `condition_map.get(condition, [])` and **silently skips the experiment** when
that is empty, producing zero curated records with no error.

The key must match `experiment.conditions` EXACTLY, and **the study and its experiments can disagree
on capitalisation** -- our study says "Long Covid" while its experiments say "Long COVID", because
the experiment takes the condition from the trial record rather than from the study config. Keying
off the study alone therefore reproduces the zero-records bug. We seed every condition string found
in the built experiments, plus the study's own, and let the extra keys be harmless.

    python 01_seed_study_dataset.py --subreddits covidlonghaulers LongCovid LongHaulersRecovery
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import yaml

from naturalv2.study import Study, StudyDataset, get_study_filepaths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-path", default=os.environ.get("SAVE_PATH", "outputs"))
    ap.add_argument("--condition", default="Long Covid", help="condition as used in the STUDY")
    ap.add_argument("--experiment", default="noparallel_notbinary")
    ap.add_argument("--ate", action="store_true", help="study was built with ate=True")
    ap.add_argument("--source", default="reddit")
    ap.add_argument("--subreddits", nargs="+", required=True)
    args = ap.parse_args()

    paths = get_study_filepaths(
        base_dir=args.save_path, condition=args.condition,
        experiment_name=args.experiment, ate=args.ate,
    )
    study = Study.from_yaml(paths["study"])

    # curate matches on the EXPERIMENT's condition strings, which may differ in case from the
    # study's. Collect both.
    conditions: list[str] = list(study.conditions)
    exp_dir = os.path.join(args.save_path, "experiments", args.experiment)
    for fp in sorted(glob.glob(os.path.join(exp_dir, "*.yaml"))):
        doc = yaml.safe_load(open(fp, encoding="utf-8")) or {}
        for c in (doc.get("_conditions") or doc.get("conditions") or []):
            if c not in conditions:
                conditions.append(c)

    if not conditions:
        print("ERROR: no condition strings found in the study or its experiments; "
              "curate would match nothing", file=sys.stderr)
        raise SystemExit(1)

    sd = StudyDataset(study.conditions, [args.source])
    sd.sources[args.source] = {c: list(args.subreddits) for c in conditions}
    sd.to_yaml(paths["study_dataset"])

    print(f"seeded {args.source} map for {len(conditions)} condition string(s): {conditions}")
    print(f"   -> {args.subreddits}")
    print(f"-> {paths['study_dataset']}")
    if not glob.glob(os.path.join(exp_dir, "*.yaml")):
        print(f"NOTE: no experiments under {exp_dir} yet — only the study's conditions were seeded. "
              "Re-run after building experiments.", file=sys.stderr)


if __name__ == "__main__":
    main()
