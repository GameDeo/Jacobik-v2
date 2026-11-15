"""Effective pool loader for the Step 2 control pools (shared by the diag scripts; CPU only).

The effective pool is the raw pool with the S2.1 repair overlay applied, where an overlay exists
(repair_s2_1/repair_manifest.json -> pools[key].overlay_path, relative to frame_v1/). Overlay content by repair:
  R1_mace    energies_kj, ref_energy_kj, top1_idx, floor_idx, top1_dE_kj, floor_dE_kj (recomputed MACE)
  R2_mapping gen_to_ref_idx, ref_to_gen_idx, mapping_ambiguous, num_automorphisms,
             coord_intact, coord_sigma_shift_A, coord_haptic_slip_A, coord_haptic_shift_A
  R3_health  health fields only (no energies, no mapping, no coordination)
Keys absent from an overlay fall back to the raw pool.

Returned dict (all per row unless noted):
  score, rank, asg, X (generated order), Z (float), terms (n x 6, Q1..Q6, already times Z),
  gen_to_ref (int array, as used by the orchestrator: X[:, gen_to_ref] is reference order),
  ref_X, ref_symbols, coord_intact (bool or None), dE (kJ/mol, energies - ref_energy),
  repairs (list), source ("raw" or "overlay:<repairs>").
"""
import json
import os

import numpy as np

TERMS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]


class Repairs:
    def __init__(self, scaled_root):
        self.root = scaled_root
        self.pools = json.load(open(os.path.join(scaled_root, "repair_s2_1", "repair_manifest.json")))["pools"]

    def repairs_for(self, code, seed, control="control"):
        ov = self.pools.get("%s:%d:%s" % (code, seed, control))
        return [] if ov is None else list(ov.get("repairs", []))

    def overlay(self, code, seed, control="control"):
        ov = self.pools.get("%s:%d:%s" % (code, seed, control))
        if ov is None:
            return None
        return np.load(os.path.join(self.root, ov["overlay_path"]))


def load_effective(pools_dir, repairs, code, seed, control="control"):
    base = os.path.join(pools_dir, "%s_%d_%s" % (code, seed, control))
    z = np.load(base + ".npz", allow_pickle=True)
    if "error" in z.files:
        return None
    mj = json.load(open(base + ".mace64.json"))
    ov = repairs.overlay(code, seed, control)
    src = "raw"
    energies = np.asarray(mj["energies_kj"], float)
    ref_e = float(mj["ref_energy_kj"])
    gen_to_ref = np.asarray(z["gen_to_ref_idx"], int) if "gen_to_ref_idx" in z.files else None
    coord = np.asarray(z["coord_intact"], bool) if "coord_intact" in z.files else None
    if ov is not None:
        src = "overlay:" + ",".join(repairs.repairs_for(code, seed, control))
        if "energies_kj" in ov.files:
            energies = np.asarray(ov["energies_kj"], float)
            ref_e = float(ov["ref_energy_kj"])
        if "gen_to_ref_idx" in ov.files:
            gen_to_ref = np.asarray(ov["gen_to_ref_idx"], int)
        if "coord_intact" in ov.files:
            coord = np.asarray(ov["coord_intact"], bool)
    dE = energies - ref_e
    return dict(score=np.asarray(z["score"], float), rank=np.asarray(z["rank"], float),
                asg=np.asarray(z["asg"], int), X=np.asarray(z["X"], float), Z=float(z["Z"]),
                terms=np.stack([np.asarray(z["terms_" + q], float) for q in TERMS], 1),
                gen_to_ref=gen_to_ref, ref_X=np.asarray(z["ref_X"], float),
                ref_symbols=[str(s) for s in z["ref_symbols"]], coord_intact=coord, dE=dE,
                repairs=repairs.repairs_for(code, seed, control), source=src)


def control_tasks(manifest_path, splits=("calibration", "motion_validation")):
    m = json.load(open(manifest_path))
    for key, rec in m["tasks"].items():
        if rec.get("status") == "success" and rec.get("split") in splits:
            yield rec["code"], int(rec["seed"]), rec
