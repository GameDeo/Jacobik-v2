"""Paired CPU evaluation of the F-2 research-side embed arms."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "research" / "bench")]

from tools.jgen import inputs  # noqa: E402
from tools.jgen.embed_study.alternatives import ARMS, embed_arm  # noqa: E402
from tools.jgen.embed_study.analyze_census import truth_flags  # noqa: E402
from tools.jgen.embed_study.measure_embeds import _ligand_metrics, _reference_by_rdkit  # noqa: E402
from tools.jgen.bench.panels import TAIL  # noqa: E402
from jacobik.builder.ligands import _build_ligand_groups, ligand_inputs_from_mol  # noqa: E402
from jacobik.builder.embed import ez_variants  # noqa: E402


def normalized_error(metrics):
    values = [abs(x["delta_A"]) / .15 for x in metrics["bonds"]]
    values += [abs(x["embed_deg"] - x["reference_deg"]) / 15.0
               for x in metrics["angles"] if not x.get("solver_dof")]
    values += [abs(x["abs_delta_deg"]) / 20.0 for x in metrics.get("planarity", [])]
    st = metrics.get("stereo", {})
    for i in set(st.get("reference_tetrahedral", {})):
        values.append(float(st.get("embed_tetrahedral", {}).get(i) != st["reference_tetrahedral"][i]))
    for i in set(st.get("reference_ez", {})):
        values.append(float(st.get("embed_ez", {}).get(i) != st["reference_ez"][i]))
    return float(np.median(values)) if values else None


def _work(arg):
    code, rows = arg
    shard, dataset = inputs.frame_shards()[code]
    mol = inputs.load_mol(code, shard, dataset)
    if mol is None:
        return [{"code": code, "ligand_idx": ix, "label": label, "arm": arm, "status": "failed",
                 "error": "parent geometry unavailable", "normalized_error": None, "seconds": 0.0}
                for ix, label in rows for arm in ARMS]
    groups = _build_ligand_groups(mol, mol.smiles)
    ligands = ligand_inputs_from_mol(mol, groups)
    out = []
    for ix, label in rows:
        if ix >= len(groups) or ix >= len(ligands):
            for arm in ARMS:
                out.append({"code": code, "ligand_idx": ix, "label": label, "arm": arm, "status": "failed",
                            "error": "ligand index unavailable", "normalized_error": None, "seconds": 0.0})
            continue
        lig, group = ligands[ix], groups[ix]
        variants = ez_variants(lig.smiles, lig.donors) or [[]]
        bend_pivots = set()
        try:
            from jacobik.ik.angle_bends import bend_specs
            xyz_to_rdkit = {int(x): int(r) for r, x in group["rdkit_to_xyz"].items()}
            bend_pivots = {xyz_to_rdkit[int(sp[0])] for sp in bend_specs(mol, mol.ligands[ix])
                           if int(sp[0]) in xyz_to_rdkit}
        except Exception:
            pass
        from rdkit import Chem
        ref_base = _reference_by_rdkit(mol, group, Chem.AddHs(Chem.MolFromSmiles(lig.smiles)))
        for arm in ARMS:
            from jacobik.builder.engine import _GFN2_CACHE
            _GFN2_CACHE.clear()
            t0 = time.perf_counter()
            variant_scores, variant_bad, errors = [], [], []
            err_msg = None
            for ez in variants:
                try:
                    rd, xyz, status, _ = embed_arm(arm, lig.smiles, lig.donors, seed=42, ez=ez,
                                                   geometry=mol.coord_geom, metal=mol.metal_symbol, charge=mol.charge)
                    ref = ref_base
                    metrics = _ligand_metrics(lig.smiles, lig.donors, xyz, ref, bend_pivots)
                    value = normalized_error(metrics)
                    if value is not None:
                        errors.append(value)
                    variant_bad.append(any(truth_flags(metrics).values()))
                    variant_scores.append(status)
                except Exception as exc:
                    err_msg = f"{type(exc).__name__}: {exc}"
            elapsed = time.perf_counter() - t0
            status = "ok" if errors else "failed"
            if errors and all(s == "not_applicable" for s in variant_scores): status = "not_applicable"
            out.append({"code": code, "ligand_idx": ix, "label": label, "arm": arm, "status": status,
                        "error": err_msg, "normalized_error": min(errors) if errors else None,
                        "bad": bool(all(variant_bad)) if variant_bad else None,
                        "variant_errors": errors, "n_variants_ok": len(errors), "seconds": elapsed})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--resonance-scan", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    if not 1 <= args.workers <= 3:
        ap.error("F-2 caps CPU workers at 3")
    census = [json.loads(x) for x in open(args.census)]
    analysis = json.load(open(args.analysis))
    scan = json.load(open(args.resonance_scan))
    lig_by_key = {(p["code"], l["ligand_idx"]): dict(l, cohort=p.get("cohort"))
                  for p in census for l in p.get("ligands", [])}
    all_keys = set(lig_by_key)
    bad_rows = analysis["bad_ligand_records"]
    bad_keys = {(r["code"], r["ligand_idx"]) for r in bad_rows}
    valid_by_code = Counter(code for code, ix in all_keys if lig_by_key[(code, ix)].get("status") == "ok")
    bad_count_by_code = Counter(code for code, _ in bad_keys)
    worst_codes = sorted(bad_count_by_code, key=lambda c: (-bad_count_by_code[c],
                          -bad_count_by_code[c]/max(valid_by_code[c],1), c))[:20]
    failure = {k: "corrected_label_worst20" for k in bad_keys if k[0] in worst_codes}
    for key in (("QEZVEY", 2), ("HEDJOP", 0), ("ZOLQIC", 1)):
        failure[key] = "named_failure"
    nitro_keys = set()
    for cohort in ("cal_tail", "surface_fit_sample"):
        for m in scan[cohort]["details"]:
            if m["kind"] == "nitro_malformed_N0_Ominus2_single":
                nitro_keys.add((m["code"], m["ligand_idx"]))
    for key in nitro_keys:
        failure[key] = "uncorrected_nitro"
    failure = {k:v for k,v in failure.items() if k in all_keys}
    bad_keys = set(failure)
    def scoreable(row):
        return bool(row.get("bonds") or any(not a.get("solver_dof") for a in row.get("angles", []))
                    or row.get("planarity") or row.get("stereo", {}).get("reference_tetrahedral")
                    or row.get("stereo", {}).get("reference_ez"))
    good_pool = sorted(k for k in all_keys if k not in {(r["code"],r["ligand_idx"]) for r in bad_rows}
                       and lig_by_key[k].get("status") == "ok" and lig_by_key[k].get("cohort") == "cal_tail"
                       and scoreable(lig_by_key[k]))
    good = random.Random(20261029).sample(good_pool, 40)
    jobs_by_code = defaultdict(list)
    for key,label in failure.items(): jobs_by_code[key[0]].append((key[1],label))
    for code,ix in good: jobs_by_code[code].append((ix,"good_control"))
    jobs = sorted((c, sorted(set(rows))) for c, rows in jobs_by_code.items())
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(_work,j) for j in jobs]
        for f in as_completed(fs): records.extend(f.result())
    records.sort(key=lambda r:(r["code"],r["ligand_idx"],r["arm"]))
    summary={}
    for arm in ARMS:
        summary[arm]={}
        for label in ("failure_set","named_failure","uncorrected_nitro","corrected_label_worst20","good_control"):
            rr=[r for r in records if r["arm"]==arm and (r["label"]==label or
                 (label=="failure_set" and r["label"]!="good_control"))]
            vals=[r["normalized_error"] for r in rr if r["normalized_error"] is not None]
            summary[arm][label]={"n":len(rr),"n_failed":sum(r["status"]=="failed" for r in rr),
                "failure_rate":sum(r["status"]=="failed" for r in rr)/len(rr) if rr else None,
                "median_error":float(np.median(vals)) if vals else None,
                "median_seconds":float(np.median([r["seconds"] for r in rr])) if rr else None}
        rr=[r for r in records if r["arm"]==arm]
        summary[arm]["all"]={"n":len(rr),"n_failed":sum(r["status"]=="failed" for r in rr),
            "failure_rate":sum(r["status"]=="failed" for r in rr)/len(rr) if rr else None,
            "median_seconds":float(np.median([r["seconds"] for r in rr])) if rr else None}
    result={"failure_keys":sorted([list(k) for k in failure]),"failure_labels":{f"{k[0]}:{k[1]}":v for k,v in failure.items()},
            "worst20_parents":worst_codes,"good_control_keys":[list(k) for k in good],"nitro_keys":sorted([list(k) for k in nitro_keys]),
            "summary":summary,"records":records,"workers":args.workers,"seed":20261029}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,default=str)+"\n")
    print(json.dumps({"failure_ligands":len(failure),"good_control":len(good),"summary":summary,"output":str(args.output)}))


if __name__ == "__main__": main()
