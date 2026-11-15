"""tests/test_chem_elements.py: shared element tables, geometry helpers, and device utilities."""
import numpy as np
import torch

from jacobik.core import chemistry as E
from jacobik.core.env import resolve_device, torch_device
from jacobik.core.geometry import dihedral_deg, kabsch_fit_batch, kabsch_fit_np
from jacobik.topology import mapping


def test_element_tables_stay_separate_and_unified():
    # the bond gate (typing) and the Q6 fallback (scoring) use different tables; merging them would move keys or scores
    assert E.covalent_radius("B") == 0.85 and E.Q6_FALLBACK_RADII["B"] == 0.84
    assert E.covalent_radius("Xx") == 0.77 and E.covalent_radius("Fe") == E.BOND_GATE_RADII["Fe"]
    assert set(E.D_GROUP) <= E.METALS and E.TRANSITION_METALS <= E.METALS | {"La", "Ce"}
    assert mapping.METALS is E.METALS


def test_kabsch_dihedral_and_device_helpers():
    rng = np.random.default_rng(3)
    P = rng.normal(size=(8, 3))
    R0, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    R0 *= np.sign(np.linalg.det(R0))
    Q = P @ R0.T + 0.5
    rmsd, max_dev, R = kabsch_fit_np(P, Q)
    assert rmsd < 1e-12 and max_dev < 1e-12
    r2, _m2, R2 = kabsch_fit_batch(np.stack([P, P]), np.stack([Q, Q]))
    assert np.allclose(R2[1], R) and np.allclose(r2, rmsd)

    p = [np.array(v, float) for v in ([1, 0, 0], [0, 0, 0], [0, 0, 1], [0, 1, 1])]
    assert abs(dihedral_deg(*p) - 90.0) < 1e-9
    assert abs(dihedral_deg(p[0], p[1], p[2], np.array([1.0, 0, 1]))) < 1e-9

    assert torch_device(None) == torch.device("cpu") and torch_device("cpu").type == "cpu"
    assert resolve_device("cpu") == "cpu"
