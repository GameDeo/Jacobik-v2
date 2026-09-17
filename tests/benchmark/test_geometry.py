import numpy as np
import pytest

from jacobik.benchmark.geometry import evaluate_geometry, kabsch, symmetry_corrected_kabsch


def test_kabsch_removes_rigid_transform_without_reflection():
    q = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    r = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    p = q @ r.T + np.array([4., -2., 7.])
    rmsd, aligned = kabsch(p, q)
    assert rmsd == pytest.approx(0.0, abs=1e-12)
    assert aligned == pytest.approx(q, abs=1e-12)


def test_symmetry_corrects_half_turn_ring_assignment():
    angles = np.arange(6) * np.pi / 3
    q = np.c_[np.cos(angles), np.sin(angles), np.zeros(6)]
    # Deliberately distort one vertex so Kabsch cannot absorb the relabelling.
    q[0, 0] += 0.2
    perm = (3, 4, 5, 0, 1, 2)
    p = q[list(perm)]
    plain = kabsch(p, q)[0]
    corrected, _, chosen = symmetry_corrected_kabsch(p, q, [range(6), perm])
    assert plain > 0.05
    assert corrected == pytest.approx(0.0, abs=1e-12)
    assert chosen == perm


def test_coordination_and_clash_metrics_are_reported():
    ref = np.array([[0., 0., 0.], [2., 0., 0.], [-2., 0., 0.], [0., 3., 0.]])
    pred = ref.copy()
    pred[1] = [2.2, 0., 0.]
    pred[3] = [2.21, 0., 0.]  # nonbonded severe clash with atom 1
    metrics = evaluate_geometry(
        pred, ref, donor_indices=[1, 2], metal_index=0,
        bonds=[(0, 1), (0, 2)], atomic_numbers=[26, 7, 7, 6],
        sphere_indices=[0, 1, 2], ligand_groups=[[1, 3], [2]],
    )
    assert metrics.metal_donor_mae_angstrom == pytest.approx(0.1)
    assert metrics.severe_clash_count >= 1
    assert metrics.coordination_retained_fraction == 1.0


def test_bad_permutation_rejected():
    coords = np.zeros((3, 3))
    with pytest.raises(ValueError):
        symmetry_corrected_kabsch(coords, coords, [(0, 0, 2)])
