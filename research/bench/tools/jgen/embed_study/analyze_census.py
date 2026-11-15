"""Summarize F census measurements and compare reference-free detectors with reference labels."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def qsummary(values):
    x = np.asarray(list(values), float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "median": float(np.median(x)), "p05": float(np.percentile(x, 5)),
            "p95": float(np.percentile(x, 95)), "max": float(np.max(x))}


def bond_key(row):
    return tuple(sorted((str(row["type"][0]), str(row["type"][1]))) + [float(row["type"][2])])


def angle_key(row):
    return (tuple(sorted((str(row["type"][0]), str(row["type"][2])))), str(row["type"][1]), bool(row["contains_H"]))


def truth_flags(row):
    bad = {"bond": False, "angle": False, "planarity": False, "stereo": False}
    for b in row.get("bonds", []):
        if abs(b["delta_A"]) > .15:
            bad["bond"] = True
    for a in row.get("angles", []):
        if not a.get("solver_dof") and abs(a["embed_deg"] - a["reference_deg"]) > 15.0:
            bad["angle"] = True
    if any(abs(i.get("abs_delta_deg", 0.0)) > 20.0 for i in row.get("planarity", [])):
        bad["planarity"] = True
    st = row.get("stereo", {})
    if st.get("tetrahedral_mismatches") or st.get("ez_mismatches"):
        bad["stereo"] = True
    return bad


def detectors(row, bond_ranges, angle_ranges):
    b = False
    for x in row.get("bonds", []):
        rr = bond_ranges.get(bond_key(x))
        if rr and not (rr[0] <= x["embed_A"] <= rr[1]):
            b = True
            break
    if not b:
        for x in row.get("angles", []):
            if x.get("solver_dof"):
                continue
            rr = angle_ranges.get(angle_key(x))
            if rr and not (rr[0] <= x["embed_deg"] <= rr[1]):
                b = True
                break
    s = row.get("stereo", {})
    c_bonded = bool(row.get("bonded_graph_anomalies"))
    c_nonbonded = bool(row.get("nonbonded_clashes"))
    c_stereo = bool(s.get("input_tetrahedral_lost") or s.get("input_ez_lost"))
    c = c_bonded or c_nonbonded or c_stereo
    return {"b": b, "c_bonded": c_bonded, "c_nonbonded": c_nonbonded,
            "c_stereo": c_stereo, "c": c, "union": b or c}


def confusion(labels, preds):
    tp = sum(a and b for a, b in zip(labels, preds))
    fp = sum((not a) and b for a, b in zip(labels, preds))
    fn = sum(a and (not b) for a, b in zip(labels, preds))
    tn = sum((not a) and (not b) for a, b in zip(labels, preds))
    def div(a, b): return float(a / b) if b else None
    sensitivity, specificity = div(tp, tp + fn), div(tn, tn + fp)
    # The detectors are binary classifiers here, so their AUROC is the balanced accuracy.
    auc = (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "sensitivity": sensitivity, "specificity": specificity,
            "precision": div(tp, tp + fp), "recall": sensitivity, "auroc_binary": auc}


def cluster_bootstrap(records, detector, n=1000, seed=20261028):
    by_parent = defaultdict(list)
    for r in records:
        by_parent[r["code"]].append(r)
    parents = sorted(by_parent)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [p for p in (rng.choice(parents) for _ in parents)]
        rr = [row for p in sample for row in by_parent[p]]
        cm = confusion([r["bad"] for r in rr], [r[detector] for r in rr])
        vals.append((cm["sensitivity"], cm["specificity"]))
    ans = {}
    for i, name in enumerate(("sensitivity", "specificity")):
        x = np.asarray([v[i] for v in vals if v[i] is not None], float)
        ans[name + "_ci95"] = [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else None
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--bootstraps", type=int, default=1000)
    args = ap.parse_args()
    parents = list(map(json.loads, open(args.input)))
    ligands = [dict(code=p["code"], cohort=p.get("cohort"), split=p.get("split"), **l)
               for p in parents for l in p.get("ligands", []) if l.get("status") == "ok"]
    surf = [r for r in ligands if r["cohort"] == "surface_fit_sample"]
    bond_values, angle_values = defaultdict(list), defaultdict(list)
    for r in surf:
        for b in r.get("bonds", []):
            bond_values[bond_key(b)].append(b["reference_A"])
        for a in r.get("angles", []):
            if not a.get("solver_dof"):
                angle_values[angle_key(a)].append(a["reference_deg"])
    bond_ranges = {}
    angle_ranges = {}
    for k, v in bond_values.items():
        bond_ranges[k] = [float(np.percentile(v, .5)), float(np.percentile(v, 99.5))]
    for k, v in angle_values.items():
        angle_ranges[k] = [float(np.percentile(v, .5)), float(np.percentile(v, 99.5))]

    variants = []
    distributions = defaultdict(lambda: defaultdict(list))
    detector_counts = Counter()
    for r in ligands:
        bad = truth_flags(r)
        pred = detectors(r, bond_ranges, angle_ranges)
        variants.append({"code": r["code"], "ligand_idx": r["ligand_idx"], "variant_idx": r.get("variant_idx", 0),
                         "cohort": r["cohort"], "bad": any(bad.values()), "bad_components": bad, **pred})
        for k, v in bad.items():
            if v: detector_counts["reference_bad_" + k] += 1
        for k, v in pred.items():
            if v: detector_counts["detector_" + k] += 1
        c = r.get("ligand_class", "unknown")
        for b in r.get("bonds", []):
            distributions[c]["bond_abs_delta_A"].append(abs(b["delta_A"]))
        for a in r.get("angles", []):
            if not a.get("solver_dof"):
                name = "H_angle_abs_delta_deg" if a.get("contains_H") else "heavy_angle_abs_delta_deg"
                distributions[c][name].append(abs(a["embed_deg"] - a["reference_deg"]))
        for ring in r.get("rings", []):
            distributions[c]["ring_plane_abs_delta_A"].append(abs(ring["plane_rms_delta_A"]))
        for imp in r.get("impropers", []):
            distributions[c]["improper_abs_delta_deg"].append(abs(imp["improper_delta_deg"]))
        for pl in r.get("planarity", []):
            distributions[c]["sp2_planarity_abs_delta_deg"].append(abs(pl["abs_delta_deg"]))

    by_ligand = defaultdict(list)
    for v in variants:
        by_ligand[(v["code"], v["ligand_idx"])].append(v)
    records = []
    for (code, ligand_idx), vs in by_ligand.items():
        # The production engine enumerates every permitted E/Z variant. A ligand is reference-bad only when
        # every production variant is bad; a reference-compatible variant makes the graph representable.
        records.append({"code": code, "ligand_idx": ligand_idx, "cohort": vs[0]["cohort"],
                        "n_ez_variants": len(vs), "bad": all(v["bad"] for v in vs),
                        "bad_components": {k: all(v["bad_components"][k] for v in vs)
                                           for k in vs[0]["bad_components"]},
                        **{d: all(v[d] for v in vs) for d in ("b", "c_bonded", "c_nonbonded", "c_stereo", "c", "union")},
                        "variant_truth": [{"variant_idx": v["variant_idx"], "bad": v["bad"],
                                           "bad_components": v["bad_components"], "b": v["b"], "c": v["c"]} for v in vs]})

    methods = {}
    for d in ("b", "c_bonded", "c_nonbonded", "c_stereo", "c", "union"):
        cm = confusion([r["bad"] for r in records], [r[d] for r in records])
        cm.update(cluster_bootstrap(records, d, n=args.bootstraps))
        methods[d] = cm
    by_class = {}
    for cl, data in distributions.items():
        by_class[cl] = {k: qsummary(v) for k, v in data.items()}
    cal_ligands = [r for r in records if r["cohort"] == "cal_tail"]
    surf_ligands = [r for r in records if r["cohort"] == "surface_fit_sample"]
    bad_component_ligand_counts = {k: sum(r["bad_components"][k] for r in records)
                                   for k in records[0]["bad_components"]}
    out = {"n_parent_records": len(parents), "parent_status_counts": dict(Counter(p["status"] for p in parents)),
           "n_ligand_variant_records": len([l for p in parents for l in p.get("ligands", [])]),
           "n_valid_ligand_variant_records": len(ligands), "n_source_ligand_instances": len(records),
           "n_reference_bad_ligands": sum(r["bad"] for r in records),
           "reference_bad_component_ligand_counts": bad_component_ligand_counts,
           "variant_reference_bad_count": sum(r["bad"] for r in variants),
           "cal_tail_valid_ligands": len(cal_ligands), "surface_fit_sample_valid_ligands": len(surf_ligands),
           "surface_fit_range_type_counts": {"bond": len(bond_ranges), "angle": len(angle_ranges)},
           "surface_fit_range_support": {"bond_min": min(map(len, bond_values.values()), default=0),
                                          "bond_median": float(np.median([len(x) for x in bond_values.values()])) if bond_values else 0,
                                          "angle_min": min(map(len, angle_values.values()), default=0),
                                          "angle_median": float(np.median([len(x) for x in angle_values.values()])) if angle_values else 0},
           "distributions_by_ligand_class": by_class, "reference_bad_and_detector_counts": dict(detector_counts),
           "detector_recovery_against_reference_bad": methods,
           "bad_ligand_records": [r for r in records if r["bad"]],
           "variant_records": variants,
           "range_tables": {"bond": [[list(k[:2]), k[2], *v, len(bond_values[k])] for k, v in sorted(bond_ranges.items())],
                            "angle": [[list(k[0]), k[1], k[2], *v, len(angle_values[k])] for k, v in sorted(angle_ranges.items())]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"valid_ligands": len(ligands), "reference_bad": out["n_reference_bad_ligands"],
                      "detectors": methods, "output": str(args.output)}))


if __name__ == "__main__":
    main()
