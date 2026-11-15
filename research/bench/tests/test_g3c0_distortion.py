"""Scientific diagnostic checks on ideal and deliberately distorted octahedra."""
from types import SimpleNamespace as NS
import numpy as np
import torch
from tools.jgen.g3c0_distortion import measure


def test_shape_rotation_q2_padding_and_recovered_assignment():
    from jacobik.ik.targets import IDEAL_GEOMETRIES
    ideal = np.array(IDEAL_GEOMETRIES['octahedral'])
    pairs = torch.triu_indices(6, 6, 1)
    class Geometry:
        metal_pos_idx = 0
        def donor_positions(self, X):
            return X[:, 1:]
        def q2_dmd_angles(self, X):
            u = torch.nn.functional.normalize(X[:, 1:], dim=-1)
            return torch.acos((u @ u.transpose(-1,-2))[:, pairs[0], pairs[1]].clamp(-1,1))
    def conformer(d):
        atoms = [NS(symbol='Fe', position=np.zeros(3))] + [NS(symbol='N', position=x) for x in d]
        mol = NS(atoms=atoms, coord_sites=[NS(position=x) for x in d], metal_idx=0)
        mol.get_atom = lambda idx: atoms[idx]
        return NS(complex=mol, assignment_idx=0)
    # The unused padded mean at cos=1 must never become the nearest mean.
    mu = torch.tensor([[0., -1., 1.]]).expand(15, -1)
    weights = torch.tensor([[.5, .5, 0.]]).expand(15, -1)
    scorer = NS(_q2_cand=None, _q2_means=mu, _q2_weights=weights)
    ctx = NS(device='cpu', ideal=ideal, symmetric_donor_groups=[])
    angle = .731
    rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0,0,1]])
    baseline = conformer(ideal @ rot.T)
    damaged = ideal.copy(); damaged[0] = damaged[1]
    rows = measure([baseline, conformer(damaged)], ctx, [np.arange(6)], Geometry(), scorer, 'octahedral')
    assert abs(rows[0]['shape_deviation']) < 2e-6
    assert rows[0]['max_q2_deviation_deg'] < .05
    assert rows[0]['class_changed'] is False
    assert rows[1]['shape_deviation'] > .01
    assert rows[1]['max_q2_deviation_deg'] > 80
