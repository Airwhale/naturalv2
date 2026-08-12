"""Align the sampled quantity with the label for change-from-baseline endpoints.

`sample_ty` asks the model for a value "on the same scale as the outcome description above". When a
trial reports a CHANGE but describes an ABSOLUTE scale, the model correctly returns an absolute
score and is then graded against a change -- a guaranteed large error that says nothing about
estimate quality. See docs/patientpunk/findings.md, A6.

NCT05618587 is the worked example: titled "Fatigue Severity Scale", described as "Score range 1-49",
reported value -11.3. A value of -11.3 is impossible on that scale, and only `timeFrame` ("Change
from baseline to day 21") says so.

Detection therefore uses three signals, strongest last:
  * change wording in the title,
  * change wording in `timeFrame` or the description,
  * a reported value outside the range the description itself states.
The range check is the reliable one: wording can be omitted, an out-of-range value cannot.

    python 03_fix_change_outcomes.py --nct NCT05618587

Rewrites `_outcome_desc` in place. Re-running after `build_experiments_from_study` is required,
since that regenerates the experiment YAML and discards the rewrite.
"""

from __future__ import annotations

import argparse
import json
import os
import re

import yaml

from naturalv2.utils import get_experiment_filepath

# \bchanges?\b, not \bchange\b: NCT06366724 states "Changes in % of predicted ..." and the singular
# form silently missed three of its four primary outcomes.
_CHANGE_RE = re.compile(
    r"\bchanges?\b|from baseline|between baseline|\bΔ\b|reduction in|improvement (in|from)|"
    r"\b(decrease|increase) (in|from)\b|difference from baseline",
    re.I,
)
# "Score range 1-49", "range: 0 to 100" -- the scale the description claims the value lives on.
_RANGE_RE = re.compile(
    r"rang\w*\s*:?\s*(-?\d+(?:\.\d+)?)\s*(?:to|through|–|—|-)\s*(-?\d+(?:\.\d+)?)", re.I
)

# Deliberately direction-NEUTRAL. Scales disagree on which way is better -- a severity scale improves
# downward, a capacity scale (LIFT's FUNCAP55) improves upward -- so asserting "negative means
# improvement" is right for one and backwards for the other. Define the arithmetic only.
TEMPLATE = (
    "CHANGE in the {name} ({timeframe}), i.e. follow-up value minus baseline value. {orig} "
    "Report that CHANGE, not the absolute score: a POSITIVE number means the score went up and a "
    "NEGATIVE number means it went down, per the scale described above; 0 means no change{hint}."
)


def is_change(title: str, timeframe: str = "", desc: str = "", value: float | None = None) -> bool:
    if _CHANGE_RE.search(" ".join(filter(None, (title, timeframe, desc)))):
        return True
    if value is None:
        return False
    m = _RANGE_RE.search(desc or "")
    if not m:
        return False
    lo, hi = sorted((float(m.group(1)), float(m.group(2))))
    return not (lo <= value <= hi)


def _trial_path(save_path: str, nct: str) -> str:
    """Completed trials live in nct_reports/, prospective targets in nct_reports_test/."""
    for sub in ("nct_reports", "nct_reports_test"):
        fp = os.path.join(save_path, sub, f"{nct}.json")
        if os.path.exists(fp):
            return fp
    raise FileNotFoundError(f"no trial JSON for {nct} under {save_path}")


def _primary_outcomes(trial_path: str) -> dict[str, dict]:
    """Primaries from posted results if present, else from the protocol.

    An active trial has no resultsSection, and the protocol names the fields differently
    (`measure`/`description`/`timeFrame`). With no reported value there, only wording detection
    applies -- there is nothing to range-check.
    """
    trial = json.load(open(trial_path, encoding="utf-8"))
    oms = (trial.get("resultsSection", {}).get("outcomeMeasuresModule", {})
           .get("outcomeMeasures", []) or [])
    if oms:
        return {om.get("title", ""): om for om in oms if om.get("type") == "PRIMARY"}
    protocol = (trial.get("protocolSection", {}).get("outcomesModule", {})
                .get("primaryOutcomes", []) or [])
    return {o.get("measure", ""): {"title": o.get("measure", ""),
                                   "timeFrame": o.get("timeFrame", ""),
                                   "description": o.get("description", "")} for o in protocol}


def _first_value(om: dict) -> float | None:
    for cls in om.get("classes", []) or []:
        cats = cls.get("categories") or []
        for m in ((cats[0].get("measurements") or []) if cats else []):
            try:
                return float(str(m.get("value")).replace(",", ""))
            except (TypeError, ValueError):
                return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-path", default=os.environ.get("SAVE_PATH", "outputs"))
    ap.add_argument("--nct", required=True)
    ap.add_argument("--experiment", default="noparallel_notbinary")
    args = ap.parse_args()

    fp = get_experiment_filepath(args.save_path, args.nct, args.experiment)
    doc = yaml.safe_load(open(fp, encoding="utf-8"))
    oms = _primary_outcomes(_trial_path(args.save_path, args.nct))

    changed = []
    for name, orig in list((doc.get("_outcome_desc") or {}).items()):
        om = oms.get(name)
        if not om:
            continue
        tf = om.get("timeFrame", "") or ""
        if not is_change(name, tf, orig or "", _first_value(om)):
            continue
        m = _RANGE_RE.search(orig or "")
        hint = ""
        if m:
            lo, hi = sorted((float(m.group(1)), float(m.group(2))))
            span = hi - lo
            hint = f" (plausible range about {-span:g} to {span:g})"
        doc["_outcome_desc"][name] = TEMPLATE.format(
            name=name, timeframe=tf.strip().rstrip(".") or "change from baseline",
            orig=(orig or "").strip().rstrip(".") + ".", hint=hint,
        )
        changed.append(name)

    if not changed:
        print("no change-from-baseline outcomes detected; nothing rewritten")
        return

    yaml.safe_dump(doc, open(fp, "w", encoding="utf-8"), sort_keys=False, width=100)
    print(f"rewrote {len(changed)} outcome description(s) in {fp}:")
    for name in changed:
        print(f"\n  [{name}]\n    {doc['_outcome_desc'][name]}")


if __name__ == "__main__":
    main()
