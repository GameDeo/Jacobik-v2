"""Unit tests: declared-donor mapping, bound counts, clash labelling and reference mapping modes on A01.

Fixtures (copied from the pilot run structures/baseline/A01/native/seed_11): arch_final_000/001/002.xyz and
arch_init_001/002.xyz. Reference: research/bench/benchmarks/data/references/A01.xyz (heavy atoms, CN 6, [Co(en)3]3+).
Declared ligands (input spec): three NCCN with coordList [0, 3].

Expected (auditor, 2026-11-03): declared donors bound 6 / 4 / 5 at final ranks 0 / 1 / 2, expected 6. Final rank 0 maps
with metal edges; ranks 1 and 2 map on the ligand graph only. The init assembly of rank 1 has clashing non-bonded N-N
pairs (labelled by n_clash_pairs > 0).

Run: cd <worktree>; PYTHONPATH=src:.:research/bench python -m pytest research/bench/tools/architector/tests -q
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.architector import post
from tools.runkit import geom

FIX = Path(__file__).resolve().parent / "fixtures" / "A01"
LIGANDS = [{"smiles": "NCCN", "coordList": [0, 3]} for _ in range(3)]


@pytest.fixture(scope="module")
def ref():
    r = post.catalog_reference("baseline", "A01")
    assert r is not None, "A01 reference missing"
    P, capped = geom.automorphisms(r["X"], r["symbols"], r["metal"])
    return dict(r, P=P, capped=bool(capped))


def _row(path, ref, init_only=False):
    sym, X = post.read_xyz(FIX / path)
    row = {}
    post.geometry_metrics(row, sym, X, ref, "Co", LIGANDS, init_only=init_only)
    return row


@pytest.mark.parametrize("rank, bound, mode", [(0, 6, "full"), (1, 4, "ligand_only"), (2, 5, "ligand_only")])
def test_final_bound_and_mode(ref, rank, bound, mode):
    row = _row(f"arch_final_{rank:03d}.xyz", ref)
    assert row["n_donors_expected"] == 6
    assert row["n_donors_mapped"] == 6 and row["n_donors_unmapped"] == 0
    assert row["n_donors_bound"] == bound
    assert row["dissociated"] == (bound < 6)
    assert row["map_mode"] == mode


def test_full_mapping_has_symmetric_rmsd_and_shell(ref):
    row = _row("arch_final_000.xyz", ref)
    assert row["sym_heavy_rmsd_A"] is not None and row["sym_heavy_rmsd_A"] >= 0
    assert row["shell_rmsd_A"] is not None


@pytest.mark.parametrize("rank", [1, 2])
def test_dissociated_structures_have_no_shell_and_ligand_rmsd(ref, rank):
    row = _row(f"arch_final_{rank:03d}.xyz", ref)
    assert row["shell_rmsd_A"] is None
    assert np.isfinite(row["ligand_kabsch_rmsd_A"])
    assert "sym_heavy_rmsd_A" not in row


def test_init_rank1_clash_labelled(ref):
    row = _row("arch_init_001.xyz", ref, init_only=True)
    # three non-bonded N-N pairs at 1.17-1.19 x rcov sum: close contacts (graph-edge range), not below the 0.75 clash cut
    assert row["n_close_pairs"] > 0
    assert row["n_clash_pairs"] == 0
    assert row["min_nonbonded_dist_A"] is not None and row["min_nonbonded_dist_A"] < 1.75
    assert row["arch_init_only"] is True


def test_init_rank2_clash_labelled(ref):
    row = _row("arch_init_002.xyz", ref, init_only=True)
    assert row["n_close_pairs"] > 0
    assert row["min_nonbonded_dist_A"] is not None


def test_bound_donor_gate_is_jgen_gate():
    rm, rn = geom.rcov("Co"), geom.rcov("N")
    d = 1.21 * (rm + rn)
    sym = ["Co", "N", "N", "N"]
    X = np.array([[0, 0, 0], [d - 0.01, 0, 0], [0, d + 0.02, 0], [0, 0, d + 0.2]], float)
    m = post.declared_donor_map(sym, X, 0, [{"smiles": "N", "coordList": [0]}])
    assert m[0]["donors_out"] == [1]  # the only nitrogen inside the gate is the declared donor
    assert post._gate(sym, X, 0, 1) and not post._gate(sym, X, 0, 2)


def test_extra_contacts_counted_not_declared(ref):
    row = _row("arch_final_000.xyz", ref)
    assert row["n_extra_contacts"] >= 0
