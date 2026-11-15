"""Validate the source-only audit review and materialize versioned target overlays."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from .gate_audit import _digest, _read, _source, _typing, _write

CLEAN = "no_input_discrepancy_detected"
REPAIR = "confirmed_error_repaired_input"
ERROR = "confirmed_error_unrepaired"
AMBIGUOUS = "ambiguous_target"
MISSING = "missing_or_unmappable"
DISPOSITIONS = {CLEAN, REPAIR, ERROR, AMBIGUOUS, MISSING}


def _record(root, code):
    path = root / "run_v1" / "records" / f"{code}.json"
    return path, _read(path)


def collect(root: Path, decisions: dict):
    """Fail closed on a missing parent, absent review or stale evidence hash."""
    manifest = _read(root / "manifest.json")
    rows = []
    expected_decisions = set()
    for entry in manifest["parents"]:
        code = entry["code"]
        path, evidence = _record(root, code)
        if evidence.get("id") != code or evidence.get("status") == "ok" and (
                evidence.get("code") != code or evidence.get("source_sha256") != entry["source_sha256"]):
            raise ValueError(f"{code}: inconsistent evidence identity")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        flags = evidence.get("review_flags", {})
        if evidence.get("status") != "ok" or flags:
            expected_decisions.add(code)
            decision = decisions[code]
            if decision.get("evidence_sha256") != digest or sorted(decision.get("reviewed_flags", [])) != sorted(flags):
                raise ValueError(f"{code}: stale evidence digest or review flags")
            if decision["disposition"] not in DISPOSITIONS or not decision.get("reason"):
                raise ValueError(f"{code}: missing disposition or explanation")
            if decision["disposition"] == REPAIR and not decision.get("repair"):
                raise ValueError(f"{code}: missing repair specification")
            if decision["disposition"] != REPAIR and decision.get("repair"):
                raise ValueError(f"{code}: a repair requires a confirmed error")
            if evidence.get("status") != "ok" and decision["disposition"] != MISSING:
                raise ValueError(f"{code}: failed source audit cannot verify target")
        else:
            decision = {"disposition": CLEAN, "reason": "No discrepancy in the recorded source-only checks.",
                        "reference_compatible": True, "reviewed_flags": []}
        rows.append({**entry, **decision, "evidence_file": str(path.relative_to(root)), "evidence_sha256": digest,
                     "review_flags": flags})
    if set(decisions) != expected_decisions:
        raise ValueError(f"unexpected decisions: {sorted(set(decisions) - expected_decisions)}")
    if len(rows) != 230 or Counter(row["stage"] for row in rows) != {1: 120, 2: 110}:
        raise ValueError("wrong audit coverage")
    return rows


def validate_overlay(entry):
    """Runner task: materialize a repaired target in the existing builder input format.

    This is deliberately only a static target. It does not call the builder, embed,
    or solve, and does not opt either gate arm into a new runtime typing profile.
    """
    import numpy as np
    from jacobik.core.types import CoordinationSite
    from jacobik.topology.coord_geom import classify_with_margin
    from jacobik.builder.ligands import _build_ligand_groups

    source = _source(entry)
    repair = entry["repair"]
    mode = repair["typing_basis"]
    if mode not in {"t1d", "explicit_reference_sites"}:
        raise ValueError("unsupported repair typing basis")
    original, raw_typed = _typing(source, False)
    mol, typed = _typing(source, mode == "t1d")
    if typed["mapping"]["mapping_path"] != "smiles" or not typed["sites"]:
        raise ValueError("repair needs an aligned source graph and classified sites")
    expected = sorted(repair["donor_sites_xyz"])
    if sorted(s["atoms"] for s in raw_typed["sites"]) == expected:
        raise ValueError("repair does not change original target topology")
    if mode == "explicit_reference_sites":
        old_sites = {tuple(sorted(s.donor_indices)): s for s in mol.coord_sites}
        sites = []
        for atom_ids in expected:
            key = tuple(atom_ids)
            if key in old_sites:
                sites.append(old_sites[key])
            elif len(atom_ids) == 1:
                atom = mol.get_atom(atom_ids[0])
                sites.append(CoordinationSite("kappa", atom_ids, np.asarray(atom.position), atom.symbol))
            else:
                raise ValueError("new haptic target needs separate chemical validation")
        if len(set(i for s in sites for i in s.donor_indices)) != sum(len(s.donor_indices) for s in sites):
            raise ValueError("overlapping sites in proposed target")
        mol.coord_sites = sites
        mol.coord_geom, mol.coord_geom_alt, mol.coord_geom_gap = classify_with_margin(mol)
        typed["sites"] = [{"atoms": s.donor_indices, "eta": s.haptic_order, "label": s.label} for s in sites]
    if sorted(s["atoms"] for s in typed["sites"]) != expected:
        raise ValueError("repair does not match source-supported donor groups")
    if not mol.coord_geom or mol.coord_geom == "unknown":
        raise ValueError("unclassified corrected geometry")
    groups = _build_ligand_groups(mol, source["smiles"])
    old_groups = _build_ligand_groups(original, source["smiles"])
    if [g["raw_smiles"] for g in groups] != [g["raw_smiles"] for g in old_groups]:
        raise ValueError("donor-only repair unexpectedly changed the ligand SMILES")
    mapped_sites = sorted(sorted(g["rdkit_to_xyz"][i] for i in donor)
                         for g in groups for donor in g["donor_groups_rdkit"])
    if mapped_sites != expected or any(not g["raw_smiles"] or not g["donor_groups_rdkit"] for g in groups):
        raise ValueError("repaired builder target did not preserve every donor group")
    target = {"metal": mol.metal_symbol, "geometry": mol.coord_geom, "charge": mol.charge,
              "spin": mol.spin_state,
              "ligands": [{"smiles": g["raw_smiles"], "donors": g["donor_groups_rdkit"]} for g in groups]}
    return {"code": entry["code"], "source_sha256": entry["source_sha256"],
            "evidence_sha256": entry["evidence_sha256"], "reason": entry["reason"],
            "original_sites_xyz": raw_typed["sites"], "corrected_sites_xyz": typed["sites"],
            "original_geometry": original.coord_geom, "corrected_geometry": mol.coord_geom,
            "corrected_geometry_alt": mol.coord_geom_alt, "corrected_geometry_gap": mol.coord_geom_gap,
            "reference_compatible": entry["reference_compatible"],
            "reference_reuse_requires_original_identity": True,
            "target": target, "target_sha256": _digest(target), "static_validation": "exact donor mapping"}


def compile_decisions(root, notes):
    """Bind manually reviewed findings to the immutable records they explain."""
    with Path(notes).open(newline="") as f:
        findings = list(csv.DictReader(f, delimiter="\t"))
    manual = {row["code"]: row for row in findings}
    if len(manual) != len(findings):
        raise ValueError("duplicate review notes")
    decisions = {}
    labels = {"clean": CLEAN, "repair": REPAIR, "error": ERROR, "ambiguous": AMBIGUOUS, "missing": MISSING}
    for entry in _read(root / "manifest.json")["parents"]:
        code = entry["code"]
        path, rec = _record(root, code)
        flags = rec.get("review_flags", {})
        row = manual.pop(code, None)
        if rec.get("status") == "ok" and not flags:
            if row:
                raise ValueError(f"{code}: manual note without a review trigger")
            continue
        if row is None:
            if set(flags) != {"typing_difference"} or rec.get("status") != "ok":
                raise ValueError(f"{code}: missing manual review")
            p, d = (rec["typing"][mode] for mode in ("production", "t1d"))
            groups = lambda m: [(s["atoms"], s["eta"]) for s in m["sites"]]
            if groups(p) != groups(d) or p["geometry"] != d["geometry"]:
                raise ValueError(f"{code}: non-label change requires manual review")
            changes = [f'{a["atoms"]}: {a["subclass"]} -> {b["subclass"]}'
                       for a, b in zip(p["sites"], d["sites"]) if a["subclass"] != b["subclass"]]
            row = {"decision": "clean", "reason": "Typing changed only labels; donor atom groups, hapticity, "
                   "and geometry are identical." + (" Subclasses: " + "; ".join(changes) if changes else "")}
        decision = {"disposition": labels[row["decision"]], "reason": row["reason"],
                    "reference_compatible": row["decision"] in {"clean", "repair"},
                    "reviewed_flags": sorted(flags), "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        if row["decision"] == "repair":
            if row["sites"] == "t1d":
                sites = [s["atoms"] for s in rec["typing"]["t1d"]["sites"]]
                basis = "t1d"
            else:
                sites = json.loads(row["sites"])
                basis = "explicit_reference_sites"
            decision["repair"] = {"donor_sites_xyz": sorted(sites), "typing_basis": basis}
        decisions[code] = decision
    if manual:
        raise ValueError(f"manual notes for out-of-scope parents: {sorted(manual)}")
    _write(root / "review_decisions.json", decisions)


def freeze_review(root):
    rows = collect(root, _read(root / "review_decisions.json"))
    _write(root / "audit_dispositions.json", {"schema": 1, "parents": rows})
    with (root / "audit_dispositions.csv").open("w", newline="") as f:
        columns = ("code", "stage", "stage1_primary", "disposition", "reference_compatible", "reason", "evidence_file")
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    repairs = [row for row in rows if row["disposition"] == REPAIR]
    _write(root / "repair_validation_manifest.json", {"schema": 1, "parents": repairs})
    print(json.dumps({"audit_count": len(rows), "counts": dict(Counter(row["disposition"] for row in rows)),
                      "repairs": [row["code"] for row in repairs]}, sort_keys=True))


def publish_overlays(root):
    rows = _read(root / "repair_validation_manifest.json")["parents"]
    results = []
    for entry in rows:
        code = entry["code"]
        path = root / "repair_validation" / "records" / f"{code}.json"
        got = _read(path)
        if (got.get("status") != "ok" or got["code"] != code or got["source_sha256"] != entry["source_sha256"]
                or got["evidence_sha256"] != entry["evidence_sha256"]):
            raise ValueError(f"{code}: missing/invalid static repair validation")
        results.append({k: v for k, v in got.items() if k not in {"status", "wall_s", "id"}})
    out = {"schema": 1, "scope": "corrected benchmark targets; no gate results changed", "overlays": results}
    _write(root / "target_overlays_v1.json", out)
    print(json.dumps({"validated_overlays": len(results), "overlay_sha256": _digest(out)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("freeze-review", "publish-overlays"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--notes", type=Path, help="TSV manual findings to bind to audit evidence")
    args = ap.parse_args()
    if args.action == "freeze-review":
        if args.notes:
            compile_decisions(args.out, args.notes)
        freeze_review(args.out)
    else:
        publish_overlays(args.out)


if __name__ == "__main__":
    main()
