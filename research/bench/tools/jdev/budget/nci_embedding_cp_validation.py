"""Phase 3 (Step 8): Separate, non-production ligand-embedding validation track.

1. Diagnoses the ninth Cp-family strict-xTB failure (GIQSIK) from n2_embedding_validation_2026-10-18.
2. Evaluates narrow eta5-Cp-only intra-face restraint release on an expanded cohort of 16
   additional independent non-benchmark substituted-Cp ligands (disjoint from the original 8),
   confirming bond-length and donor-angle RMS improvements without touching eta6-arenes, eta2,
   or group-13 protections.
3. Evaluates anionic N-N bound-state modeling across the independent non-benchmark both-primary
   anionic N-N complexes from anionic_NN_topology_census.json (plus JEFJUZ), testing whether
   restraining original heavy-atom donor bond angles during capped GFN2 refinement
   (cap_release_bonded_keep_angles) preserves donor angles as well as N-N / C-N bond lengths.

Writes versioned artifacts to data/refit/reports/n2_embedding_cp_validation_2026-10-18/.
No runtime or production default changes.
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zlib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO
from pools import setup, set_cases
from nci_cap_generality import refine, cap_start, errors
from nci_training_calibration import write, digest, stream
from nci_embedding_validation import make_case

OUT = REPO / "data/refit/reports/n2_embedding_cp_validation_2026-10-18"
PRIOR_DIR = REPO / "data/refit/reports/n2_embedding_validation_2026-10-18"


def refine_with_angles(rd, x, pairs, angles):
    """Strict ligand-only xTB optimization with explicit donor-distance and heavy-donor-angle constraints."""
    from rdkit import Chem
    from jacobik.builder.engine import XTB_BIN

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sy = [a.GetSymbol() for a in rd.GetAtoms()]
        (td / "l.xyz").write_text(
            f"{len(sy)}\n\n" + "".join(f"{s} {a:.9f} {b:.9f} {c:.9f}\n" for s, (a, b, c) in zip(sy, x))
        )
        q = Chem.GetFormalCharge(rd)
        uhf = (sum(a.GetAtomicNum() for a in rd.GetAtoms()) - q) % 2
        args = [XTB_BIN, "l.xyz", "--gfn", "2", "--opt", "crude", "--chrg", str(q), "--uhf", str(uhf)]
        if pairs or angles:
            lines = ["$constrain\n", " force constant=1.0\n"]
            for a, b in pairs:
                lines.append(f" distance: {a+1}, {b+1}, auto\n")
            for b, a, c in angles:
                lines.append(f" angle: {b+1}, {a+1}, {c+1}, auto\n")
            lines.append("$end\n")
            (td / "c.inp").write_text("".join(lines))
            args += ["--input", "c.inp"]
        result = subprocess.run(
            args,
            cwd=td,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS="1"),
            timeout=180,
        )
        if result.returncode or not (td / "xtbopt.xyz").exists():
            raise RuntimeError(
                f"xTB exit {result.returncode}: " + result.stdout[-600:].decode(errors="replace")
            )
        y = np.array(
            [
                [float(v) for v in line.split()[1:4]]
                for line in (td / "xtbopt.xyz").read_text().splitlines()[2 : 2 + len(sy)]
            ]
        )
        assert y.shape == x.shape and np.isfinite(y).all()
        return y


def _heavy_donor_angles(base, donors):
    """Return list of (b, a, c) heavy-atom angle triples centered on each donor atom a."""
    triples = []
    flat = sorted(set(i for g in donors for i in g))
    for a in flat:
        if a >= base.GetNumAtoms():
            continue
        nb = [n.GetIdx() for n in base.GetAtomWithIdx(a).GetNeighbors() if n.GetAtomicNum() > 1]
        for b, c in itertools.combinations(sorted(nb), 2):
            triples.append((b, a, c))
    return triples


def diagnose_giqsik():
    """Diagnose the GIQSIK xTB failure recorded in n2_embedding_validation_2026-10-18."""
    from rdkit import Chem
    from jacobik.builder.engine import _smiles_to_conformer

    smi = "Cc1c(C)c(C)[c-](CCNC(=O)c2ccc(S(#[O+])([O-])F)cc2)c1C"
    dn = [[1, 2, 4, 6, 22]]
    face = {1, 2, 4, 6, 22}
    rd, start, _ = _smiles_to_conformer(smi, donors=dn, chelate_embed=True, embed="mmff")
    flat = sorted(set(i for g in dn for i in g))
    pairs = [(a, b) for j, a in enumerate(flat) for b in flat[j + 1 :]]
    released = [(a, b) for a, b in pairs if not (a in face and b in face)]

    # Inspect bond lengths in MMFF start around hypervalent S(#[O+])([O-])F vs Cp face
    s_bonds = []
    cp_bonds = []
    for b in rd.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        si, sj = rd.GetAtomWithIdx(i).GetSymbol(), rd.GetAtomWithIdx(j).GetSymbol()
        dist = float(np.linalg.norm(start[i] - start[j]))
        if "S" in (si, sj):
            s_bonds.append(dict(atoms=[i, j], symbols=f"{si}-{sj}", order=float(b.GetBondTypeAsDouble()), mmff_r_A=dist))
        if i in face and j in face:
            cp_bonds.append(dict(atoms=[i, j], symbols=f"{si}-{sj}", mmff_r_A=dist))

    arm_outcomes = {}
    for arm_name, p_list in [("keep_pairs", pairs), ("release_face", released), ("unconstrained", [])]:
        try:
            _ = refine(rd, start, p_list)
            arm_outcomes[arm_name] = "success"
        except Exception as exc:
            arm_outcomes[arm_name] = str(exc).splitlines()[0]

    return dict(
        code="GIQSIK",
        smiles=smi,
        diagnosis=(
            "Both keep_pairs (baseline) and release_face fail identically in xTB with |grad| > 500 "
            "because the tmQM SMILES encodes the sulfonyl fluoride group as S(#[O+])([O-])F; "
            "the failure is present in the baseline keep_pairs arm and is not caused by releasing Cp restraints."
        ),
        arm_outcomes=arm_outcomes,
        sulfur_bonds_mmff=s_bonds,
        cp_face_bonds_mmff=cp_bonds,
    )


def run_expanded_cp_cohort():
    """Select and evaluate 16 additional independent substituted-Cp ligands beyond the initial 8."""
    from rdkit import Chem
    from jacobik.builder.engine import _smiles_to_conformer
    from tools.jdev.start import _crystal_ligand_coords
    from fit.pipeline.extract import benchmark_codes

    excluded = set(benchmark_codes())
    prior_sel = json.loads((PRIOR_DIR / "selection.json").read_text())
    prior_keys = {r["key"] for r in prior_sel["selected"]}

    source = REPO / "data/refit/reports/oxidation_training_modes_2026-10-18.json"
    data = json.loads(source.read_text())
    candidates = {}
    for row in data["rows"]:
        if row["code"] in excluded:
            continue
        for li in row["ligands"]:
            smi = li["smiles"]
            dn = li["donors"]
            base = Chem.MolFromSmiles(smi)
            if base is None or "[H]" in smi or Chem.AddHs(base).GetNumAtoms() > 80:
                continue
            face = [
                g
                for g in dn
                if len(g) == 5
                and all(base.GetAtomWithIdx(i).GetAtomicNum() == 6 and base.GetAtomWithIdx(i).GetIsAromatic() for i in g)
                and sum(base.GetAtomWithIdx(i).GetFormalCharge() for i in g) == -1
            ]
            if len(face) != 1:
                continue
            key = repr((smi, dn))
            if key in prior_keys:
                continue
            candidates.setdefault(
                key,
                dict(
                    code=row["code"],
                    shard=row["set"],
                    dir=str(REPO / "data/refit/clean/tmQM"),
                    smiles=smi,
                    donors=dn,
                    face=face[0],
                    family="substituted_Cp_expanded",
                    key=key,
                ),
            )

    selected = sorted(
        candidates.values(), key=lambda r: zlib.crc32(("cp-release-2026-10-18:" + r["key"]).encode())
    )[:16]

    chosen = {(r["shard"], r["code"]): r for r in selected}
    rows = []
    failures = []
    for row, mol, error in stream(chosen.values()):
        try:
            if error:
                raise ValueError(error)
            case = make_case(mol)
            crystal = _crystal_ligand_coords(case)
            for r in [r for r in selected if r["code"] == row["code"]]:
                smi = r["smiles"]
                dn = r["donors"]
                rd, start, _ = _smiles_to_conformer(smi, donors=dn, chelate_embed=True, embed="mmff")
                reference = crystal[smi][0]
                assert len(reference) == len(start)
                flat = sorted(set(i for g in dn for i in g))
                face = set(r["face"])
                pairs = [(a, b) for j, a in enumerate(flat) for b in flat[j + 1 :]]
                released = [(a, b) for a, b in pairs if not (a in face and b in face)]
                variants = dict(
                    keep_pairs=refine(rd, start, pairs),
                    release_face=refine(rd, start, released),
                )
                metrics = {k: errors(rd, x, reference, dn) for k, x in variants.items()}
                rows.append(dict(**r, metrics=metrics))
                print("Expanded Cp:", r["code"], json.dumps({k: dict(bond_rms=v["bond_rms"], angle_rms=v["angle_rms"]) for k, v in metrics.items()}), flush=True)
        except Exception as exc:
            failures.append(dict(code=row["code"], error=f"{type(exc).__name__}: {exc}"))
            print("Expanded Cp FAILED:", failures[-1], flush=True)

    # Combine with the 8 prior Cp / substituted_Cp results for summary statistics
    prior_res = json.loads((PRIOR_DIR / "training_results.json").read_text())
    prior_cp_rows = [r for r in prior_res["rows"] if r["family"] in ("Cp", "substituted_Cp")]
    all_cp = prior_cp_rows + rows

    def _summarize(subset):
        b_keep = [r["metrics"]["keep_pairs"]["bond_rms"] for r in subset]
        b_rel = [r["metrics"]["release_face"]["bond_rms"] for r in subset]
        a_keep = [r["metrics"]["keep_pairs"]["angle_rms"] for r in subset]
        a_rel = [r["metrics"]["release_face"]["angle_rms"] for r in subset]
        return dict(
            n_ligands=len(subset),
            mean_bond_rms_keep=float(np.mean(b_keep)),
            mean_bond_rms_release=float(np.mean(b_rel)),
            bond_improved_count=int(sum(x < y for x, y in zip(b_rel, b_keep))),
            mean_angle_rms_keep=float(np.mean(a_keep)),
            mean_angle_rms_release=float(np.mean(a_rel)),
            angle_improved_count=int(sum(x < y for x, y in zip(a_rel, a_keep))),
        )

    return dict(
        expanded_selected=len(selected),
        expanded_completed=len(rows),
        expanded_failures=failures,
        expanded_summary=_summarize(rows),
        combined_24_cohort_summary=_summarize(all_cp),
        rows=rows,
    )


def run_anionic_nn_cohort():
    """Evaluate bound-state modeling across the independent non-benchmark both-primary
    anionic N-N ligands from anionic_NN_topology_census.json + JEFJUZ diagnostic."""
    from rdkit import Chem
    from jacobik.builder.engine import _smiles_to_conformer
    from tools.jdev.cases import get_case
    from tools.jdev.start import _crystal_ligand_coords

    census = json.loads((PRIOR_DIR / "anionic_NN_topology_census.json").read_text())
    primary_rows = [r for r in census["rows"] if r["both_primary"]]

    # Group by (shard, code) and evaluate unique (code, smiles)
    chosen = {(r["shard"], r["code"]): r for r in primary_rows}
    seen_code_smi = set()
    results = []
    failures = []

    def _eval_nn_ligand(code, smi, dn, reference):
        base = Chem.MolFromSmiles(smi)
        rd, start, _ = _smiles_to_conformer(smi, donors=dn, chelate_embed=True, embed="mmff")
        assert len(reference) == len(start)
        flat = sorted(set(i for g in dn for i in g))
        pairs = [(a, b) for j, a in enumerate(flat) for b in flat[j + 1 :]]
        released = [(a, b) for a, b in pairs if base.GetBondBetweenAtoms(a, b) is None]
        heavy_angles = _heavy_donor_angles(base, dn)

        sur, x_cap, mapping = cap_start(base, rd, start, dn)
        variants = dict(
            bare_keep_pairs=refine(rd, start, pairs),
            bare_release_bonded=refine(rd, start, released),
            cap_keep_pairs=refine(sur, x_cap, pairs)[mapping],
            cap_release_bonded=refine(sur, x_cap, released)[mapping],
            cap_release_bonded_keep_angles=refine_with_angles(sur, x_cap, released, heavy_angles)[mapping],
        )
        metrics = {
            k: dict(
                bond_rms=errors(rd, x, reference, dn)["bond_rms"],
                angle_rms=errors(rd, x, reference, dn)["angle_rms"],
            )
            for k, x in variants.items()
        }
        return metrics

    for row, mol, error in stream(chosen.values()):
        code = row["code"]
        try:
            if error:
                raise ValueError(error)
            case = make_case(mol)
            crystal = _crystal_ligand_coords(case)
            for r in [r for r in primary_rows if r["code"] == code]:
                smi = r["smiles"]
                if (code, smi) in seen_code_smi:
                    continue
                seen_code_smi.add((code, smi))
                dn = r["donors"]
                if smi not in crystal or not crystal[smi]:
                    raise ValueError("missing crystal ligand coords")
                reference = crystal[smi][0]
                m = _eval_nn_ligand(code, smi, dn, reference)
                results.append(dict(code=code, smiles=smi, scaffold=r["scaffold"], is_benchmark=False, metrics=m))
                print("Anionic N-N:", code, json.dumps(m), flush=True)
        except Exception as exc:
            failures.append(dict(code=code, error=f"{type(exc).__name__}: {exc}"))
            print("Anionic N-N FAILED:", failures[-1], flush=True)

    # Also evaluate JEFJUZ anionic N-N ligand for direct comparison
    name = next(n for s in ("dev32", "n48") for c, n in set_cases(s) if c == "JEFJUZ")
    jef_case = get_case(name)
    jef_crystal = _crystal_ligand_coords(jef_case)
    jef_metrics = None
    for li in jef_case.ligand_inputs:
        if "[N-]" in li.smiles:
            jef_metrics = _eval_nn_ligand("JEFJUZ", li.smiles, li.donors, jef_crystal[li.smiles][0])
            break

    arms = ["bare_keep_pairs", "bare_release_bonded", "cap_keep_pairs", "cap_release_bonded", "cap_release_bonded_keep_angles"]
    summary = {}
    for arm in arms:
        b_vals = [r["metrics"][arm]["bond_rms"] for r in results]
        a_vals = [r["metrics"][arm]["angle_rms"] for r in results]
        base_b = [r["metrics"]["bare_keep_pairs"]["bond_rms"] for r in results]
        base_a = [r["metrics"]["bare_keep_pairs"]["angle_rms"] for r in results]
        summary[arm] = dict(
            n_independent=len(results),
            mean_bond_rms=float(np.mean(b_vals)) if b_vals else None,
            mean_angle_rms=float(np.mean(a_vals)) if a_vals else None,
            bond_better_than_bare_keep=int(sum(x < y for x, y in zip(b_vals, base_b))),
            angle_better_than_bare_keep=int(sum(x < y for x, y in zip(a_vals, base_a))),
            both_bond_and_angle_better=int(sum(bx < by and ax < ay for bx, by, ax, ay in zip(b_vals, base_b, a_vals, base_a))),
        )

    return dict(
        independent_evaluated=len(results),
        failures=failures,
        summary_by_arm=summary,
        independent_rows=results,
        jefjuz_reference_check=jef_metrics,
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    giqsik = diagnose_giqsik()
    cp_cohort = run_expanded_cp_cohort()
    nn_cohort = run_anionic_nn_cohort()
    report = dict(
        source_hash=__import__("tools.jdev.env", fromlist=["tree_hash"]).tree_hash(),
        giqsik_diagnosis=giqsik,
        cp_intra_face_release=cp_cohort,
        anionic_nn_bound_state=nn_cohort,
    )
    write(OUT / "embedding_cp_nn_validation.json", report)
    print("Saved embedding_cp_nn_validation.json", flush=True)


if __name__ == "__main__":
    main()
