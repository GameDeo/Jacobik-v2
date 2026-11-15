"""Paired CPU light gate for H-2 embed flags on selected calibration parents."""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
from rdkit import Chem
sys.path[:0] = ["src", ".", "research/bench"]

from tools.jgen import inputs
from jacobik.core.env import build_builder_and_inputs
from jacobik.core.env import MACE_MODEL_PATH
from tools.jdev.budget.common import stats_from_db
from jacobik.builder.atom_map import build_atom_mapping
from jacobik.core import profile
from jacobik.builder import engine
from jacobik.core.energy import MACEPolarEvaluator

OUT = Path("research/data/jgen/campaign/H/h2_census/light_gate.json")

class FallbackWatch(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        msg = record.getMessage()
        if "GFN2 embed refinement failed" in msg:
            self.messages.append(msg)

def xyz(m):
    return np.asarray([a.position for a in m.atoms], dtype=float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True, help="comma-separated cal parent codes")
    ap.add_argument("--output", default=str(OUT))
    a = ap.parse_args()
    if not __import__("shutil").which("xtb"):
        raise SystemExit("xtb is required for this gate")
    calc = MACEPolarEvaluator(model="polar-1-s", device="cpu", dtype="float64",
                              model_path=str(MACE_MODEL_PATH), threads=1)
    index = inputs.frame_shards()
    watch = FallbackWatch()
    logging.getLogger("jacobik.builder.engine").addHandler(watch)
    out = {"profile": profile.snapshot(), "results": []}
    for code in [c.strip().upper() for c in a.codes.split(",") if c.strip()]:
        engine._GFN2_CACHE.clear()
        engine._GFN2_BOND_CHECK_CACHE.clear()
        watch.messages.clear()
        sh, ds = index[code]
        ref = inputs.load_mol(code, sh, ds)
        builder, _, err = build_builder_and_inputs(ref)
        if builder is None:
            out["results"].append({"code": code, "error": str(err)})
            continue
        # Score a returned, solver-placed complex. A raw _build_shell frame has
        # no reliable inter-ligand placement and is not a valid full-complex MACE input.
        result = builder.solve(seed=11, stats=stats_from_db(inputs.DB_PATH), angle_bends=True,
                               device="cpu", n_ik_steps=250, n_conformers=16)
        if not result.conformers:
            out["results"].append({"code":code,"error":"no conformers returned"})
            continue
        conf = result.conformers[0]
        shell = conf.complex
        Xr, Xg = xyz(ref), xyz(shell)
        amap = build_atom_mapping(ref, gen_mol=shell, X_ref=Xr, X_gen=Xg)
        ref_symbols = [atom.symbol for atom in ref.atoms]
        shell_symbols = [atom.symbol for atom in shell.atoms]
        symbols_match = bool(amap.verify_symbols(ref_symbols, shell_symbols))
        if not symbols_match or len(Xr) != len(Xg) or len(Xg) != len(shell_symbols):
            raise RuntimeError(f"{code}: atom mapping/count mismatch: ref={len(Xr)}, shell={len(Xg)}, symbols={symbols_match}")
        if "Li" in shell_symbols:
            raise RuntimeError(f"{code}: temporary Li cap leaked into returned shell")
        if int(ref.charge) != int(shell.charge) or int(ref.spin_state) != int(shell.spin_state):
            raise RuntimeError(f"{code}: charge/spin mismatch: ref={ref.charge}/{ref.spin_state}, shell={shell.charge}/{shell.spin_state}")
        if watch.messages:
            raise RuntimeError(f"{code}: GFN2 fell back: {watch.messages}")
        source_to_shell = np.asarray(amap.gen_to_ref_idx, dtype=int)
        shell_to_source = np.asarray(amap.ref_to_gen_idx, dtype=int)
        Xref_shell = Xr[shell_to_source]
        metal_ref = int(shell_to_source[int(shell.metal_idx)-1])
        haptic_anions = []
        for li, lig in enumerate(builder.ligands):
            rdmol = Chem.MolFromSmiles(lig.smiles)
            if rdmol is None:
                continue
            for group in lig.donors:
                if len(group) > 1:
                    charge = sum(int(rdmol.GetAtomWithIdx(int(i)).GetFormalCharge()) for i in group)
                    if charge < 0:
                        haptic_anions.append({"ligand":li,"atoms":list(group),"net_formal_charge":charge})
        rows = []
        for b in ref.bonds:
            i, j = int(b.i)-1, int(b.j)-1
            if min(i,j) < 0 or max(i,j) >= len(ref.atoms):
                continue
            # Internal ligand bonds only; the metal shell is included for MACE below.
            if i == metal_ref or j == metal_ref:
                continue
            ei = float(np.linalg.norm(Xg[source_to_shell[i]]-Xg[source_to_shell[j]]))
            er = float(np.linalg.norm(Xr[i]-Xr[j]))
            rows.append({"atoms_ref":[i,j], "elements":"-".join(sorted((ref.atoms[i].symbol,ref.atoms[j].symbol))),
                         "embed_A":ei, "crystal_A":er, "delta_A":ei-er})
        energies = calc.energies([(shell_symbols, Xg, int(shell.charge), int(shell.spin_state)),
                                  (shell_symbols, Xref_shell, int(shell.charge), int(shell.spin_state))], batch_size=1)
        events = engine.pop_embed_bond_check_events() if hasattr(engine, "pop_embed_bond_check_events") else []
        rows_over = [r for r in rows if abs(r["delta_A"]) > .1]
        out["results"].append({"code":code,"top1_score":float(conf.score),
            "haptic_anionic_groups":haptic_anions, "internal_bonds":rows,
            "mapping_symbols_match":symbols_match,"atom_count":len(shell_symbols),"hydrogen_count":shell_symbols.count("H"),
            "li_count_in_shell":shell_symbols.count("Li"),"charge":int(shell.charge),"spin_state":int(shell.spin_state),
            "internal_bonds_gt_0p1A":len(rows_over), "max_abs_internal_delta_A":max([abs(r["delta_A"]) for r in rows] or [0]),
            "mace_embed_minus_crystal_kj_mol":float((energies[0]-energies[1])*96.4853321233),
            "mace_embed_eV":float(energies[0]), "mace_crystal_eV":float(energies[1]),
            "bond_check_events":events})
        print(code, "bonds_gt_0.1", len(rows_over), "MACE(embed-crystal)", out["results"][-1]["mace_embed_minus_crystal_kj_mol"], flush=True)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2))

if __name__ == "__main__": main()
