"""test_embed: haptic Li cap scope/placement, GFN2 start-geometry cache, output window, and retired profile env switches."""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import pytest


def test_haptic_li_cap_charge_and_sigma_placement(monkeypatch):
    from rdkit import Chem
    from jacobik.builder import engine as E

    monkeypatch.setattr(E, "EMBED_HAPTIC_LI_CAP", True)
    monkeypatch.setattr(E, "EMBED_LI_CAP", False)
    monkeypatch.setattr(E, "EMBED_BOND_CHECK", False)

    for smiles, expected_caps in (
        ("c1cc[cH-]c1", 1),
        ("Cc1c(C)c(C)[c-](C)c1C", 1),
        ("Cc1c(C)[c-](C)c2c(C)[c-](C)c(C)c12", 1),
        ("[CH2-]C=C", 1),
    ):
        for kekule in (False, True):
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            aromatic = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
            face = aromatic or [0, 1]
            if kekule:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            positions = np.arange(mol.GetNumAtoms() * 3, dtype=float).reshape(-1, 3) * 0.1
            monkeypatch.setattr(E, "_GFN2_CACHE", {})
            observed = []

            def run_charge(args, *, cwd, **kwargs):
                p = Path(cwd)
                lines = (p / "l.xyz").read_text().splitlines()
                assert int(lines[0]) == len(positions) + expected_caps
                assert sum(line.startswith("Li ") for line in lines[2:]) == expected_caps
                observed.append(True)
                (p / "xtbopt.xyz").write_text("\n".join(lines) + "\n")

            monkeypatch.setattr("subprocess.run", run_charge)
            actual = E._gfn2_refine(mol, positions, [face])
            assert observed and np.allclose(actual, positions, atol=1e-6), (smiles, kekule)

    for smiles, face, expected_caps, sigma_atom in (
        ("[P+](C)(C)(C)[CH-]C=O", [4, 5], 0, None),
        ("CC(=S)[S-]", [1, 3], 1, 3),
        ("[C-]#[O+]", [0, 1], 0, None),
    ):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        positions = np.arange(mol.GetNumAtoms() * 3, dtype=float).reshape(-1, 3) * 0.1
        monkeypatch.setattr(E, "_GFN2_CACHE", {})
        observed = []

        def run_sigma(args, *, cwd, **kwargs):
            p = Path(cwd)
            lines = (p / "l.xyz").read_text().splitlines()
            X = np.asarray([[float(v) for v in line.split()[1:4]] for line in lines[2:]])
            assert len(X) == len(positions) + expected_caps
            if sigma_atom is not None:
                assert np.linalg.norm(X[-1] - positions[sigma_atom]) == pytest.approx(1.9, abs=1e-6)
            observed.append(True)
            (p / "xtbopt.xyz").write_text("\n".join(lines) + "\n")

        monkeypatch.setattr("subprocess.run", run_sigma)
        actual = E._gfn2_refine(mol, positions, [face])
        assert observed and np.allclose(actual, positions, atol=1e-6), smiles


def test_gfn2_cache_keyed_on_start_geometry(monkeypatch):
    """Two different embeds of one ligand must not share a GFN2 cache entry (solve_multi_embed diversity)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import jacobik.builder.engine as E

    monkeypatch.setattr(E, "XTB_BIN", "definitely-not-xtb-2026")
    monkeypatch.setattr(E, "_GFN2_CACHE", {})
    m = Chem.AddHs(Chem.MolFromSmiles("OCCO"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    p1 = m.GetConformer().GetPositions()
    p2 = p1 + np.array([0.3, 0.0, 0.0]) * (np.arange(len(p1))[:, None] % 2)
    out1 = E._gfn2_refine(m, p1, donors=[[0], [3]])
    out2 = E._gfn2_refine(m, p2, donors=[[0], [3]])
    assert np.allclose(out1, p1)
    assert np.allclose(out2, p2), "second embed got the first embed's cached geometry"
    assert np.allclose(E._gfn2_refine(m, p1, donors=[[0], [3]]), p1)
    assert len(E._GFN2_CACHE) == 2


def test_output_window_drops_only_rows_far_below_the_best(stats, monkeypatch):
    """JACOBIK_OUT_WINDOW: every returned conformer lies within OUT_WINDOW summed nats of the best; the best is kept."""
    import jacobik.builder.engine as E
    from jacobik import ComplexBuilder, LigandInput

    def build():
        return ComplexBuilder("Co", "octahedral", charge=3, spin=1,
                              ligands=[LigandInput("NCCN", donors=[[0], [3]]) for _ in range(3)])

    monkeypatch.setattr(E, "OUT_WINDOW", 0.0)
    r0 = build().solve(seed=3, stats=stats, n_ik_steps=20, beam_width=4, n_coord_samples=2, n_conformers=50)
    s0 = np.array([c.score for c in r0.conformers])
    Z = float(r0.scorer._agg_norm())
    win = (s0.max() - s0) * Z
    w = float(np.quantile(win, 0.5)) if win.max() > 0 else 1.0
    monkeypatch.setattr(E, "OUT_WINDOW", w)
    r1 = build().solve(seed=3, stats=stats, n_ik_steps=20, beam_width=4, n_coord_samples=2, n_conformers=50)
    s1 = np.array([c.score for c in r1.conformers])
    assert np.isclose(s1.max(), s0.max())
    assert ((s1.max() - s1) * Z <= w + 1e-6).all()
    assert len(s1) <= len(s0)


def test_profile_fields_overrides_and_retired_switches():
    """Retired env switches raise loudly and JACOBIK_PROFILE_OVERRIDES validation runs at import."""
    import jacobik.core.profile as P

    src = str(Path(P.__file__).resolve().parents[1])
    code = (
        "import json, jacobik.core.profile as P, jacobik.ik.scorer as S, jacobik.builder.engine as E; "
        "print(json.dumps([S.Q6_MODEL, S.SCORE_AGG, E.OUT_WINDOW, P.snapshot()['differs_from_profile']]))"
    )
    env = {k: v for k, v in os.environ.items() if not (k.startswith("JACOBIK_") and k in P.RETIRED_ENV)}
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    env["JACOBIK_PROFILE"] = "production"
    env.pop("JACOBIK_PROFILE_OVERRIDES", None)

    def run(extra):
        return subprocess.run([sys.executable, "-c", code], env=dict(env, **extra), capture_output=True, text=True)

    r = run({})
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout.strip().splitlines()[-1]) == ["repulsive", "sum", 200.0, {}]
    r = run({"JACOBIK_Q6_AGG": "sum"})
    assert r.returncode != 0 and "retired environment switches" in r.stderr
    r = run({"JACOBIK_PROFILE_OVERRIDES": json.dumps({"no_such_field": 1})})
    assert r.returncode != 0 and "unknown fields" in r.stderr
