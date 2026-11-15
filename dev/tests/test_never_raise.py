"""test_never_raise: merged from test_guards."""
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIX = HERE / "regression" / "fixtures"
EDTA = "[O-]C(=O)CN(CC(=O)[O-])CCN(CC(=O)[O-])CC(=O)[O-]"


def _isolated(code: str, timeout: float, db_path) -> str:
    """Run `code` in a fresh interpreter (src on the path, DB path in env). Fails on timeout."""
    prelude = f"import sys; sys.path.insert(0, {str(REPO / 'src')!r})\n"
    env = dict(os.environ, JACOBIK_DB=str(db_path))
    try:
        p = subprocess.run([sys.executable, "-c", prelude + textwrap.dedent(code)], capture_output=True,
                           text=True, timeout=timeout, env=env, cwd=str(REPO))
    except subprocess.TimeoutExpired:
        pytest.fail(f"did not finish within {timeout:.0f} s (hang / combinatorial blow-up?)")
    assert p.returncode == 0, p.stderr[-3000:]
    return p.stdout


def test_dedup_homoleptic_and_cn9_aqua_finish(db_path):
    """Symmetry-enumeration dedup stays fast for CN8/CN9 monodentates, and end-to-end CN9 aqua solve finishes."""
    out = _isolated("""
        import os, time, numpy as np
        from jacobik import ComplexBuilder, LigandInput
        from jacobik.ik.site_enumeration import enumerate_unique_assignments
        from jacobik.stats.stats import StatsLookup, load_db
        for n, geom in ((8, "square_anti"), (9, "tricap_pris")):
            hop = np.full((n, n), -1, dtype=int)
            t0 = time.perf_counter()
            r = enumerate_unique_assignments(n, geom, hop, donor_equivalence_groups=[list(range(n))])
            dt = time.perf_counter() - t0
            assert len(r.assignments) == 1, (n, geom, len(r.assignments))
            assert dt < 10.0, f"{n} {geom} dedup took {dt:.1f} s"
        st = StatsLookup(load_db(os.environ["JACOBIK_DB"]))
        res = ComplexBuilder("Ce", "tricap_pris", charge=3, spin=2,
                             ligands=[LigandInput("O", donors=[[0]]) for _ in range(9)]).solve(
                                 stats=st, seed=11, n_ik_steps=4, beam_width=2, n_coord_samples=2)
        print(len(res.conformers))
    """, timeout=180, db_path=db_path)
    assert int(out.strip().splitlines()[-1]) > 0


def test_co_edta_hexadentate_fully_chelates(stats):
    """Hexadentate Co-EDTA forms all 6 Co-D bonds under both donor orderings, GFN2 default embed, and MMFF score band."""
    from jacobik import ComplexBuilder, LigandInput

    def _co_d_dists(best):
        X = np.array([a.position for a in best.complex.atoms])
        return [float(np.linalg.norm(X[i - 1] - X[0])) for i in best.complex.ligands[0].donor_atoms]

    # Default embed (gfn2) forms all 6 bonds
    r_gfn2 = ComplexBuilder("Co", "octahedral", charge=-1, spin=1,
                            ligands=[LigandInput(EDTA, donors=[[i] for i in [0, 4, 7, 11, 15, 19]])]).solve(
                                stats=stats, seed=42)
    d_gfn2 = _co_d_dists(r_gfn2.conformers[0])
    assert len(d_gfn2) == 6 and max(d_gfn2) < 2.45, f"Co-EDTA (gfn2) unbound arm: {np.round(d_gfn2, 2)}"

    # Reference embed (mmff) across both donor indexings and seeds satisfies binding + historical score band
    for donors, seed in (([0, 4, 7, 11, 15, 19], 42), ([0, 4, 8, 11, 15, 19], 11)):
        r_ref = ComplexBuilder("Co", "octahedral", charge=-1, spin=1,
                               ligands=[LigandInput(EDTA, donors=[[i] for i in donors])]).solve(
                                   stats=stats, seed=seed, embed="mmff")
        d_ref = _co_d_dists(r_ref.conformers[0])
        assert len(d_ref) == 6 and max(d_ref) < 2.45, f"Co-EDTA (mmff) unbound arm: {np.round(d_ref, 2)}"
        assert r_ref.conformers[0].score > -3.5, f"Co-EDTA (mmff) best score {r_ref.conformers[0].score:.2f}"


def test_benchmark_harness_donors_and_co_edta_parser():
    """Stream tmQM_P.xyz once to verify benchmark harness donor counts and Co-EDTA coord site parsing."""
    from fit.parser import TMQMParser
    from jacobik.builder.ligands import _build_ligand_groups, ligand_inputs_from_mol

    harness_codes = ("VUCVUN", "ALADIC", "ZENDII", "HAKBUU", "LIXMIR")
    co_edta_codes = ("IDACCO", "TELBAN", "CEDGEX")
    wanted = set(harness_codes) | set(co_edta_codes)
    mols = {m.csd_code: m for m in TMQMParser(FIX / "tmqm").stream_split("P") if m.csd_code in wanted}
    assert set(mols) == wanted

    txt = (FIX / "tmqm" / "xyz" / "tmQM_P.xyz").read_text()
    mnd = {
        code: int(re.search(rf"CSD_code = {code} .*?MND = (\d+)", txt).group(1))
        for code in wanted
    }

    for code in harness_codes:
        m = mols[code]
        lig = ligand_inputs_from_mol(m, _build_ligand_groups(m, m.smiles))
        assert sum(len(l.donors) for l in lig) == mnd[code], code

    for code in co_edta_codes:
        m = mols[code]
        assert len(m.coord_sites) == mnd[code] == 6, code
        assert m.coord_geom == "octahedral", code
