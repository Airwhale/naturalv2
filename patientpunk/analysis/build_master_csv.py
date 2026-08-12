"""Consolidate everything pulled so far into one master CSV, fully flagged.

One row per (trial, outcome, arm) label for train/val, plus one row per (test trial, registered
primary). Every row marked data_source = 'trial_listing' (CT.gov structured results) vs 'paper'
(LLM-extracted from the results publication) vs 'registry_adapted' (non-CT.gov ISRCTN trial: design
from the registry + per-arm outcome from its paper, adapted to CT.gov shape). Carries: trial
metadata, per-arm labels, endpoint classification + match-to-test,
and the biological/methodological flags (masking, comparator, combination, intervention type,
drug class, drug accessibility, candidate-drug, underpowered, leakage date, representation,
cross-condition duplicate).

Joins: augmented manifest, labels_sidecar, m3_extractions.jsonl, endpoint_classification.csv,
drug_classification.csv, trial JSONs.
Run: trial_superset/.venv/Scripts/python.exe trial_superset/build_master_csv.py
Output: data/master_pulled_data.csv  (gitignored)
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict

from seed_terms import CANDIDATE_DRUGS

DATA = "trial_superset/data"
MANIFEST = f"{DATA}/training_set_manifest_augmented.csv"
SIDECAR = f"{DATA}/labels_sidecar.csv"
JSONL = f"{DATA}/m3_extractions.jsonl"
CLASSIFY = f"{DATA}/endpoint_classification.csv"
DRUGCLASS = f"{DATA}/drug_classification.csv"
LABELED = f"{DATA}/m3_labeled"
OUT = f"{DATA}/master_pulled_data.csv"            # PRIMARY: Long COVID benchmark
CLUSTER_OUT = f"{DATA}/cluster_benchmark.csv"     # SEPARATE: adjacent conditions (ME/CFS, fibro, dysautonomia, lyme)

DRUG_TYPES = {"DRUG", "BIOLOGICAL", "DIETARY_SUPPLEMENT"}
DRUG_ALIASES = {a.lower(): d for d, al in CANDIDATE_DRUGS.items() for a in al}

COLS = ["nct", "condition", "split", "data_source", "is_prediction_target", "in_nikita_seed", "has_label",
        "title", "phase", "overall_status", "enrollment", "primary_completion_date", "results_public_date",
        "interventions", "condition_tags", "primary_intervention", "intervention_types",
        "drug_class", "drug_accessibility", "is_candidate_drug", "candidate_drug",
        "masking", "is_open_label", "comparator_type", "is_combination_arm", "underpowered",
        "cross_condition_duplicate",
        "paper_pmcid", "paper_link_via", "llm_confidence",
        "outcome", "arm", "endpoint_type", "is_change_from_baseline", "representation",
        "raw_value", "n", "clean_outcome", "scale_proportion",
        "endpoint_domain", "endpoint_modality", "self_reportable", "instrument", "endpoint_match_to_test",
        "is_corpus_learnable", "corpus_learnable_tier"]


PREDICTION_TARGETS = {"NCT06366724": "LIFT", "NCT07128082": "Tirzepatide", "NCT06305793": "IVIG"}


def _nikita_seed():
    """NCTs in Nikita's original shared study (faithfully reproduced at M1)."""
    import yaml
    f = f"{DATA}/m1_outputs/studies/long_covid_noparallel_notbinary_apo_study.yaml"
    if not os.path.exists(f):
        return set()
    d = yaml.safe_load(open(f, encoding="utf-8"))
    return {list(x.keys())[0] for s in ("train_trials", "val_trials", "test_trials")
            for x in (d.get(s) or [])}


NIKITA_SEED = _nikita_seed()


def _trial_json(slug, nct, split):
    sub = "nct_reports_test" if split == "test" else "nct_reports"
    p = os.path.join(LABELED, slug, sub, f"{nct}.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    if split == "test":  # fallback: recruiting prediction targets (e.g. LIFT) aren't in the strict test pool
        rp = os.path.join(DATA, "relaxed_test", "nct_reports_test", f"{nct}.json")
        if os.path.exists(rp):
            return json.load(open(rp, encoding="utf-8"))
    return None


def trial_fields(slug, nct, split, drugcls):
    j = _trial_json(slug, nct, split)
    if not j:
        return {}
    ps = j.get("protocolSection", {})
    dm = ps.get("designModule", {}) or {}
    aim = ps.get("armsInterventionsModule", {}) or {}
    arms = aim.get("armGroups", []) or []
    ivs = aim.get("interventions", []) or []

    masking = ((dm.get("designInfo", {}) or {}).get("maskingInfo", {}) or {}).get("masking", "") or ""
    arm_types = [a.get("type", "") for a in arms]
    arm_text = " ".join(((a.get("label", "") or "") + " " + " ".join(a.get("interventionNames", []) or [])).lower()
                        for a in arms)
    if "placebo" in arm_text or "sham" in arm_text or "PLACEBO_COMPARATOR" in arm_types or "SHAM_COMPARATOR" in arm_types:
        comparator = "placebo"
    elif "ACTIVE_COMPARATOR" in arm_types:
        comparator = "active_comparator"
    elif "NO_INTERVENTION" in arm_types:
        comparator = "no_treatment"
    else:
        comparator = "single_arm_or_other"
    combo = any(len(a.get("interventionNames", []) or []) > 1
                for a in arms if a.get("type") in ("EXPERIMENTAL", "ACTIVE_COMPARATOR"))
    itypes = sorted({i.get("type", "") for i in ivs})
    drug_names = [i.get("name", "") for i in ivs if i.get("type") in DRUG_TYPES
                  and "placebo" not in (i.get("name", "") or "").lower()
                  and "sham" not in (i.get("name", "") or "").lower()]
    primary = (drug_names[0] if drug_names else (ivs[0].get("name", "") if ivs else ""))[:120]
    enr = (dm.get("enrollmentInfo") or {}).get("count")
    cand = ""
    blob = " ".join(i.get("name", "") for i in ivs).lower()
    for alias, drug in DRUG_ALIASES.items():
        if alias in blob:
            cand = drug
            break
    if drug_names:  # a real drug -> use the LLM classification
        dc = drugcls.get(primary, {})
        drug_class, accessibility = dc.get("drug_class", ""), dc.get("drug_accessibility", "")
    else:  # behavioral / device / procedure / diagnostic intervention -> non-pharmacologic
        drug_class, accessibility = "non_pharmacologic", "behavioral_or_device"
    return {
        "results_first_post_date": (ps.get("statusModule", {}).get("resultsFirstPostDateStruct") or {}).get("date", ""),
        "primary_intervention": primary,
        "intervention_types": "|".join(t for t in itypes if t),
        "drug_class": drug_class,
        "drug_accessibility": accessibility,
        "is_candidate_drug": bool(cand),
        "candidate_drug": cand,
        "masking": masking,
        "is_open_label": masking.upper() in ("", "NONE"),
        "comparator_type": comparator,
        "is_combination_arm": combo,
        "underpowered": (isinstance(enr, int) and enr < 50),
    }


def main() -> None:
    cls = {r["endpoint_text"]: r for r in csv.DictReader(open(CLASSIFY, encoding="utf-8-sig"))}
    drugcls = {r["intervention"]: r for r in csv.DictReader(open(DRUGCLASS, encoding="utf-8-sig"))} if os.path.exists(DRUGCLASS) else {}

    def endpoint_fields(text):
        c = cls.get((text or "").strip(), {})
        return {k: c.get(k, "") for k in ("endpoint_domain", "endpoint_modality", "self_reportable", "instrument")}

    prov = {}
    if os.path.exists(JSONL):
        for line in open(JSONL, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("extractable") and r.get("schema"):
                s = r["schema"]
                prov[r["nct"]] = {"paper_pmcid": r.get("pmcid", ""), "paper_link_via": r.get("via", ""),
                                  "llm_confidence": s.get("confidence", ""),
                                  "result_public_date": s.get("result_public_date", "")}

    labels = defaultdict(list)
    for r in csv.DictReader(open(SIDECAR, encoding="utf-8-sig")):
        labels[(r["nct"], r["condition"], r["split"])].append(r)

    manifest = list(csv.DictReader(open(MANIFEST, encoding="utf-8-sig")))
    # ensure all 3 prediction targets appear (LIFT is recruiting -> not in the strict test manifest)
    have = {m["nct"] for m in manifest}
    for nct in PREDICTION_TARGETS:
        if nct not in have:
            manifest.append({"nct": nct, "condition": "long_covid", "split": "test",
                             "label_source": "ctgov_structured"})
    dup = {n for n, c in Counter(m["nct"] for m in manifest).items() if c > 1}

    rows = []
    for m in manifest:
        nct, slug, split, lsrc = m["nct"], m["condition"], m["split"], m["label_source"]
        data_source = ("paper" if lsrc == "paper_extracted"
                       else "registry_adapted" if lsrc == "registry_adapted"
                       else "trial_listing")  # registry_adapted = non-CT.gov (ISRCTN) design + paper outcome
        tf = trial_fields(slug, nct, split, drugcls)
        base = {"nct": nct, "condition": slug, "split": split, "data_source": data_source,
                "is_prediction_target": nct in PREDICTION_TARGETS,
                "in_nikita_seed": nct in NIKITA_SEED,
                "cross_condition_duplicate": nct in dup}
        # metadata
        j = _trial_json(slug, nct, split)
        if j:
            ps = j["protocolSection"]
            base.update(title=ps.get("identificationModule", {}).get("briefTitle", "")[:160],
                        phase="/".join((ps.get("designModule", {}) or {}).get("phases", []) or []),
                        overall_status=ps.get("statusModule", {}).get("overallStatus", ""),
                        enrollment=(ps.get("designModule", {}).get("enrollmentInfo") or {}).get("count", ""),
                        primary_completion_date=(ps.get("statusModule", {}).get("primaryCompletionDateStruct") or {}).get("date", ""),
                        interventions="; ".join(i.get("name", "") for i in
                                                (ps.get("armsInterventionsModule", {}).get("interventions", []) or []))[:200],
                        condition_tags="; ".join(ps.get("conditionsModule", {}).get("conditions", []) or [])[:160])
        base.update({k: v for k, v in tf.items() if k != "results_first_post_date"})
        if data_source == "paper":
            p = prov.get(nct, {})
            base.update({k: p.get(k, "") for k in ("paper_pmcid", "paper_link_via", "llm_confidence")})
            base["results_public_date"] = p.get("result_public_date", "")
        else:
            base["results_public_date"] = tf.get("results_first_post_date", "")

        if split == "test":
            for measure in (registered_primaries(slug, nct, split) or [""]):
                row = dict(base, has_label=False, outcome=measure, arm="", representation="")
                row.update(endpoint_fields(measure))
                rows.append(row)
        else:
            for lr in labels.get((nct, slug, split), []):
                et = lr["endpoint_type"]
                chg = lr["is_change_from_baseline"]
                rep = ("change_from_baseline" if chg == "True" else "absolute") if et == "continuous" else \
                      ("percentage" if et == "percentage" else "rate" if et == "binary" else "")
                row = dict(base, has_label=True, outcome=lr["outcome"], arm=lr["arm"],
                           endpoint_type=et, is_change_from_baseline=chg, representation=rep,
                           raw_value=lr["raw_value"], n=lr["n"], clean_outcome=lr["clean_outcome"],
                           scale_proportion=lr["scale_proportion"])
                row.update(endpoint_fields(lr["outcome"]))
                rows.append(row)

    # per-condition TEST endpoint profiles -> endpoint_match_to_test
    tdom, tmod, tinstr = defaultdict(set), defaultdict(set), defaultdict(set)
    for r in rows:
        if r["split"] == "test":
            if r.get("endpoint_domain"):
                tdom[r["condition"]].add(r["endpoint_domain"])
            if r.get("endpoint_modality"):
                tmod[r["condition"]].add(r["endpoint_modality"])
            if r.get("instrument"):
                tinstr[r["condition"]].add(r["instrument"].lower())
    for r in rows:
        cond = r["condition"]
        if r["split"] == "test":
            r["endpoint_match_to_test"] = "target"
        elif cond not in tdom and cond not in tmod:
            r["endpoint_match_to_test"] = "no_test_targets"
        elif r.get("instrument") and r["instrument"].lower() in tinstr.get(cond, set()):
            r["endpoint_match_to_test"] = "same_instrument"
        elif r.get("endpoint_domain") and r["endpoint_domain"] in tdom.get(cond, set()):
            r["endpoint_match_to_test"] = "same_domain"
        elif r.get("endpoint_modality") and r["endpoint_modality"] in tmod.get(cond, set()):
            r["endpoint_match_to_test"] = "same_modality"
        else:
            r["endpoint_match_to_test"] = "none"

    # corpus-learnable verdict: does NATURAL's premise (patient self-report -> outcome) apply?
    # single-agent + blinded, an accessible drug, and a self-reportable endpoint.
    for r in rows:
        single = not r.get("is_combination_arm")
        blinded = not r.get("is_open_label")
        acc, sr = r.get("drug_accessibility", ""), r.get("self_reportable", "")
        if single and blinded and acc == "self_obtainable" and sr == "yes":
            tier = "strict"
        elif single and blinded and acc in ("self_obtainable", "prescription_oral") and sr in ("yes", "partial"):
            tier = "relaxed"
        else:
            tier = "off_premise"
        r["corpus_learnable_tier"] = tier
        r["is_corpus_learnable"] = tier in ("strict", "relaxed")

    rows.sort(key=lambda r: (r["condition"], r["split"], r["data_source"], r["nct"]))
    # split into two datasets: Long COVID (primary) vs the adjacent-conditions cluster (separate)
    lc_rows = [r for r in rows if r["condition"] == "long_covid"]
    cluster_rows = [r for r in rows if r["condition"] != "long_covid"]
    for path, rs in ((OUT, lc_rows), (CLUSTER_OUT, cluster_rows)):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rs)

    tr = [r for r in lc_rows if r["split"] in ("train", "val")]
    print(f"master_pulled_data.csv (Long COVID): {len(lc_rows)} rows  ({len(COLS)} columns)")
    print(f"cluster_benchmark.csv (adjacent conditions): {len(cluster_rows)} rows")
    print("LC by data_source:", dict(Counter(r["data_source"] for r in lc_rows)))
    print("training drug_accessibility:", dict(Counter(r.get("drug_accessibility", "") for r in tr)))
    print("training comparator_type:", dict(Counter(r.get("comparator_type", "") for r in tr)))
    print("training is_open_label:", dict(Counter(r.get("is_open_label", "") for r in tr)))
    print("training is_combination_arm:", dict(Counter(r.get("is_combination_arm", "") for r in tr)))
    test = [r for r in lc_rows if r["split"] == "test"]
    dtrials = lambda rs: len({(r["nct"], r["condition"]) for r in rs})
    print("LC training corpus_learnable_tier:", dict(Counter(r["corpus_learnable_tier"] for r in tr)))
    print(f"LC is_corpus_learnable: training {sum(r['is_corpus_learnable'] for r in tr)} labels / "
          f"{dtrials([r for r in tr if r['is_corpus_learnable']])} trials; "
          f"test {dtrials([r for r in test if r['is_corpus_learnable']])} target trials")
    print(f"-> {OUT}  +  {CLUSTER_OUT}")


def registered_primaries(slug, nct, split):
    j = _trial_json(slug, nct, split)
    if not j:
        return []
    return [(o.get("measure") or "").strip()[:120]
            for o in j["protocolSection"].get("outcomesModule", {}).get("primaryOutcomes", []) or [] if o.get("measure")]


if __name__ == "__main__":
    main()
