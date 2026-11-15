"""Run one F-4 arm on footprint parents with runkit's unrelaxed metrics only."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "research" / "bench")]


def run_parent(code, seeds, evaluator, stats):
    from tools.runkit import policy, xtb, geom
    from tools.jgen import inputs
    from jacobik.core.env import build_builder_and_inputs
    from tools.jdev.budget.common import stats_from_db
    from jacobik.builder.atom_map import build_atom_mapping
    from jacobik.builder import engine
    from rdkit import Chem

    policy.assert_allowed([code])
    xtb.require()
    shard, dataset = inputs.frame_shards()[code]
    mol = inputs.load_mol(code, shard, dataset)
    if mol is None:
        raise RuntimeError("no parent geometry")
    builder, _, err = build_builder_and_inputs(mol)
    if builder is None:
        raise RuntimeError(f"builder failed: {err}")
    # Preserve the exact candidate donor-to-site assignments so every top-1
    # regression can be attributed to the assignment that produced that row.
    original_solve_for_assignments = builder._solve_for_assignments
    assignment_capture = {}
    def capture_assignments(ctx, assignments, beam_width, n_ik_steps):
        assignment_capture["donors"] = [
            {"donor_order": di, "ligand_idx": li, "group_idx": gi, "atom_indices": list(grp)}
            for di, (li, gi, grp) in enumerate(ctx.all_donor_groups)
        ]
        assignment_capture["candidates"] = [np.asarray(a, dtype=int).tolist() for a in assignments]
        return original_solve_for_assignments(ctx, assignments, beam_width, n_ik_steps)
    builder._solve_for_assignments = capture_assignments
    ligand_keys = {}
    anionic_donors = []
    for idx, lig in enumerate(builder.ligands):
        base = Chem.MolFromSmiles(lig.smiles)
        if base is None:
            continue
        ligand_keys.setdefault(Chem.MolToSmiles(base), []).append(idx)
        for group in lig.donors or []:
            if len(group) == 1:
                atom_idx = int(group[0])
                charge = int(base.GetAtomWithIdx(atom_idx).GetFormalCharge())
                if charge < 0:
                    anionic_donors.append({"ligand_idx": idx, "atom_idx": atom_idx, "formal_charge": charge,
                                           "smiles": lig.smiles})
    ref_sym = [a.symbol for a in mol.atoms]
    ref_X = np.asarray([a.position for a in mol.atoms], dtype=float)
    ref_structure = (ref_sym, ref_X, mol.charge, mol.spin_state)
    records = {"code": code, "seeds": {}, "embed_events": [], "anionic_kappa_donors": anionic_donors,
               "assignment_donors": assignment_capture.get("donors", []),
               "assignment_candidates": assignment_capture.get("candidates", [])}
    for seed in seeds:
        t0 = time.perf_counter()
        res = builder.solve(seed=int(seed), stats=stats, angle_bends=True, device="cuda")
        elapsed_solve = time.perf_counter() - t0
        for event in engine.pop_embed_bond_check_events():
            emol = Chem.MolFromSmiles(event.get("smiles", ""))
            if emol is not None:
                canonical = Chem.MolToSmiles(Chem.RemoveHs(emol))
                event["ligand_indices"] = ligand_keys.get(canonical, [])
            records["embed_events"].append(event)
        if not res or not res.conformers:
            records["seeds"][str(seed)] = {"status": "failed", "reason": "empty_conformers",
                                            "n_rows": 0, "solve_wall_s": elapsed_solve}
            continue
        scores = np.asarray([c.score for c in res.conformers], dtype=float)
        finite = np.isfinite(scores)
        if not finite.any():
            records["seeds"][str(seed)] = {"status": "failed", "reason": "no_finite_scores",
                                            "n_rows": len(scores), "solve_wall_s": elapsed_solve}
            continue
        top = int(np.argmax(np.where(finite, scores, -np.inf)))
        top_conf = res.conformers[top]
        aidx = int(top_conf.assignment_idx)
        candidate_table = assignment_capture.get("candidates", [])
        donor_table = assignment_capture.get("donors", [])
        top_assignment = []
        if 0 <= aidx < len(candidate_table):
            top_assignment = [dict(d, site_idx=int(candidate_table[aidx][d["donor_order"]]))
                              for d in donor_table]
        structures = [ref_structure]
        for c in res.conformers:
            X = np.asarray([a.position for a in c.complex.atoms], dtype=float)
            symbols = [atom.symbol for atom in c.complex.atoms]
            structures.append((symbols, X, mol.charge, mol.spin_state))
        t1 = time.perf_counter()
        energies = np.asarray(evaluator.energies_kj(structures), dtype=float)
        mace_wall_s = time.perf_counter() - t1
        dE = energies[1:] - energies[0]
        dE[~np.isfinite(scores)] = np.nan
        good = np.isfinite(dE)
        if not good.any() or not np.isfinite(dE[top]):
            records["seeds"][str(seed)] = {"status": "failed", "reason": "no_finite_energy",
                                            "n_rows": len(scores), "solve_wall_s": elapsed_solve,
                                            "mace_wall_s": mace_wall_s}
            continue
        metal = geom.find_metal(ref_sym)
        P_aut, _ = geom.automorphisms(ref_X, ref_sym, metal)
        heavy = [i for i, s in enumerate(ref_sym) if s != "H"]
        X_top = np.asarray([a.position for a in res.conformers[top].complex.atoms], dtype=float)
        am = build_atom_mapping(mol, gen_mol=res.conformers[top].complex, X_ref=ref_X, X_gen=X_top)
        if am is None:
            raise RuntimeError("top-1 atom mapping failed")
        X_top_ref_order = X_top[np.asarray(am.gen_to_ref_idx, dtype=int)]
        top_rmsd = float(geom.sym_rmsd_to_ref(X_top_ref_order[None, heavy], ref_X[heavy], P_aut)[0])
        worst = int(np.nanargmax(dE))
        records["seeds"][str(seed)] = {
            "status": "ok", "n_rows": len(scores), "top1_dE_unrel": float(dE[top]),
            "top1_assignment_idx": aidx, "top1_assignment": top_assignment,
            "floor_dE_unrel": float(np.nanmin(dE)), "worst_slot_dE_unrel": float(dE[worst]),
            "worst_slot": worst, "obscene_rate": float(np.mean(dE[good] > 200.0)),
            "top1_rmsd_ref_A": top_rmsd, "solve_wall_s": elapsed_solve, "mace_wall_s": mace_wall_s,
            "n_finite_energies": int(good.sum())}
    records["status"] = "ok" if any(r.get("status") == "ok" for r in records["seeds"].values()) else "failed"
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", type=Path, required=True, help="JSON array of permitted parent codes")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", default="11")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from tools.runkit.mace import Evaluator
    from tools.jdev.budget.common import stats_from_db
    from tools.jgen import inputs
    from jacobik.core import profile

    evaluator = Evaluator(device="cuda", threads=a.threads)
    stats = stats_from_db(inputs.DB_PATH)
    seeds = [int(x) for x in a.seeds.split(",") if x]
    codes = json.loads(a.codes.read_text())
    out = {"arm_flags": json.loads(os.environ.get("JACOBIK_PROFILE_OVERRIDES", "{}")),
           "fingerprint": profile.snapshot()["fingerprint"], "seeds": seeds, "codes": []}
    for code in codes:
        try:
            rec = run_parent(code, seeds, evaluator, stats)
        except Exception as exc:
            rec = {"code": code, "status": "failed", "reason": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-1500:]}
        out["codes"].append(rec)
        print(json.dumps({"code": code, "status": rec.get("status"),
                          "seeds": rec.get("seeds", {})}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"arm_fingerprint": out["fingerprint"], "codes": len(codes),
                      "failed_parents": [r["code"] for r in out["codes"] if r["status"] != "ok"],
                      "output": str(a.out)}))


if __name__ == "__main__":
    main()
