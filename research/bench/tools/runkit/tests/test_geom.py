"""A4's RMSD tests re-run through the runkit.geom facade, plus an agreement check with an explicit slow oracle."""
import itertools
from pathlib import Path

import numpy as np
import pytest

from tools.runkit import geom

HERE = Path(__file__).resolve()
STORED = HERE.parents[5].parent / "research/data/jgen/campaign/A/a4/pilot/solve/pool/QEZVEY_s11.npz"
if not STORED.exists():
    STORED = Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/jgen/campaign/A/a4/pilot/solve/pool/QEZVEY_s11.npz")


def _fe():
    sym = ["Fe", "N", "N", "N", "Cl", "Cl", "Cl"]
    X = np.array([[0, 0, 0], [2.1, 0, 0], [0, 2.1, 0], [0, 0, 2.1], [-2.3, 0, 0], [0, -2.3, 0], [0, 0, -2.3]], float)
    return sym, X


def kabsch_rmsd(A, B):
    A, B = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(A.T @ B)
    S[-1] *= np.sign(np.linalg.det(U @ Vt))
    return float(np.sqrt(max((A ** 2).sum() + (B ** 2).sum() - 2 * S.sum(), 0) / len(A)))


def test_automorphisms_identity_and_count():
    sym, X = _fe()
    P, capped = geom.automorphisms(X, sym, 0)
    assert not capped and (P[0] == np.arange(7)).all() and len(P) == 36


def test_relabelled_copy_zero_and_oracle_agreement():
    sym, X = _fe()
    P, _ = geom.automorphisms(X, sym, 0)
    rng = np.random.default_rng(3)
    Y = X + rng.normal(scale=0.15, size=X.shape)
    Y[[1, 2]] = Y[[2, 1]]
    r = geom.sym_rmsd_to_ref(Y[None], X, P)[0]
    # slow oracle: explicit minimum over the same 36 permutations of heavy atoms
    oracle = min(kabsch_rmsd(Y, X[p]) for p in P)
    assert abs(r - oracle) < 1e-9
    assert geom.sym_rmsd_to_ref(X[[0, 2, 1, 3, 4, 5, 6]][None], X, P)[0] < 1e-6


@pytest.mark.skipif(not STORED.exists(), reason="pilot pool not present")
def test_stored_pool_through_facade():
    d = np.load(STORED)
    sym = [str(s) for s in d["symbols"]]
    ref = d["ref"].astype(float)
    metal = geom.find_metal(sym)
    P, _ = geom.automorphisms(ref, sym, metal)
    heavy = [i for i, s in enumerate(sym) if s != "H"]
    r = geom.sym_rmsd_to_ref(d["rows_X"][:5].astype(float)[:, heavy], ref[heavy], P)
    assert (r > 0).all() and np.isfinite(r).all()
    adj = geom.ligand_adjacency(ref, sym, metal)
    classes = np.array([hash(s) % 997 for s in sym])
    Xh = d["rows_X"][:8].astype(float)[:, heavy]
    D = geom.sym_rmsd_matrix(Xh, P, classes[heavy], 0.5)
    lab = geom.cluster_single_linkage(D, 0.5)
    assert lab.max() + 1 <= 8 and adj.shape[0] == len(sym)
