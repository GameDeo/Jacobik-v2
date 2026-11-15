"""Energy-scale study: symmetry-aware heavy-atom RMSD of every pooled conformer to the reference (exact min over graph
automorphisms, StructureDeduper._full). Writes <pool>.rmsd.json {"rmsd": [...], "n_mapped": k, "n_heavy": n}."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from pools import OUT, set_cases, setup  # noqa: E402


def main():
    setup()
    import torch
    from jacobik.ik.dedup import StructureDeduper
    from tools.jdev.cases import get_case

    names = {}
    for s in ("n48", "heldout28", "ring10", "dev32"):
        names.update({c: n for c, n in set_cases(s)})
    shells = {}
    for f in sorted((OUT / "pools").rglob("*.npz")):
        jf = f.with_suffix(".rmsd.json")
        if jf.exists():
            continue
        z = np.load(f, allow_pickle=False)
        if "error" in z.files:
            continue
        code = f.stem.rsplit("_s", 1)[0]
        try:
            if code not in shells:
                case = get_case(names[code])
                b = case.make_builder()
                mol, _, _ = b._build_shell(False, False)
                m = case.tmqm_mol
                rb = {a.idx: i for i, a in enumerate(m.atoms)}
                amap = {0: rb[m.metal_idx]}  # the solved complex has the metal in row 0 (as run_panel's map)
                ys = [a.symbol for a in mol.atoms]
                for lig, lg in zip(sorted(mol.ligands, key=lambda l: l.ligand_id), case.ligand_groups):
                    for rd, x1 in lg["rdkit_to_xyz"].items():
                        if rd < len(lig.atom_indices) and ys[lig.atom_indices[rd] - 1] != "H":
                            amap[lig.atom_indices[rd] - 1] = rb[x1]
                shells[code] = (mol, amap, StructureDeduper(mol))
            mol, amap, dd = shells[code]
            sy = [str(s) for s in z["symbols"]]
            heavy = dd.heavy.numpy()
            if [mol.atoms[k].symbol for k in range(len(mol.atoms))] != sy:
                raise RuntimeError("atom order of the shell differs from the solved complex")
            ok = [k for k in heavy if k in amap]
            if len(ok) != len(heavy):
                raise RuntimeError(f"only {len(ok)} of {len(heavy)} heavy atoms mapped")
            ref = np.asarray(z["ref_X"])[[amap[k] for k in heavy]]
            if [str(z["ref_symbols"][amap[k]]) for k in heavy] != [sy[k] for k in heavy]:
                raise RuntimeError("mapped reference elements differ")
            A = torch.tensor(ref - ref.mean(0), dtype=torch.float64)
            B = torch.tensor(z["X"][:, heavy], dtype=torch.float64)
            B = B - B.mean(1, keepdim=True)
            r = dd._full(A, B).numpy().tolist()
            Bm = B.clone()
            Bm[..., 0] = -Bm[..., 0]  # mirror image (the deduper never merges mirrors; energy cannot tell them apart)
            rm = dd._full(A, Bm).numpy().tolist()
            jf.write_text(json.dumps({"rmsd": r, "rmsd_mirror": rm, "n_heavy": int(len(heavy))}))
            print(f"rmsd {f.parent.name}/{f.stem}: min {min(r):.2f} #1 {r[int(np.argmax(z['score']))]:.2f}", flush=True)
        except Exception as exc:
            jf.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"[:300]}))
            print(f"rmsd {f.parent.name}/{f.stem}: ERROR {exc}", flush=True)


if __name__ == "__main__":
    main()
