"""Read-only returned-row diagnostics; no solver or scorer state is changed."""
import hashlib
import math
import numpy as np
import torch
from torch.nn import functional as F


def start_digests(rows):
    return [hashlib.sha256(b''.join(t.detach().cpu().numpy().tobytes() for t in row[0])).hexdigest()
            for row in rows]


def measure(confs, ctx, assignments, ge, scorer, geometry):
    from jacobik.ik.kabsch import _optimal_rotation_nocentroid
    from jacobik.ik.site_enumeration import topological_class_keys
    from jacobik.topology.coord_geom import classify, best_fit_assignment
    from tools.jgen.atom_export import symbols_for_conformer
    X = torch.tensor(np.array([[a.position for a in c.complex.atoms] for c in confs]),
                     dtype=torch.float32, device=ctx.device)
    candidate = [int(c.assignment_idx) for c in confs]
    ideal = torch.tensor(np.array([ctx.ideal[np.asarray(assignments[c])] for c in candidate]),
                         dtype=X.dtype, device=X.device)
    dirs = ge.donor_positions(X) - X[:, ge.metal_pos_idx, None, :]
    u = F.normalize(dirs, dim=-1)
    with torch.no_grad():
        R = _optimal_rotation_nocentroid(ideal, u)
        sq = (torch.einsum('bij,bnj->bni', R, ideal) - u).square().sum(-1)
        shape = (torch.logsumexp(2 * sq, -1) - math.log(sq.shape[-1])) / 2
        angles = ge.q2_dmd_angles(X)
        if scorer._q2_cand is not None:
            mu, _, lw, _ = scorer._q2_cand
            mu = mu[candidate]; valid = lw[candidate] > -1000
        else:
            mu = scorer._q2_means[None].expand(len(confs), -1, -1)
            valid = (scorer._q2_weights > 0)[None].expand_as(mu)
        means = torch.acos(mu.clamp(-1, 1))
        nearest = torch.where(valid, (angles[..., None] - means).abs(), torch.inf).amin(-1)
        max_angle = nearest.amax(-1) * (180 / math.pi) if angles.shape[-1] else X.new_zeros(len(confs))
    keys = topological_class_keys(assignments, ctx.symmetric_donor_groups, geometry)
    values = []
    for i, c in enumerate(confs):
        symbols = symbols_for_conformer(c)
        assert symbols == [a.symbol for a in c.complex.atoms]
        label = classify(c.complex)
        a = best_fit_assignment(dirs[i].detach().cpu().numpy(), label)
        fitted = None if a is None else topological_class_keys([a], ctx.symmetric_donor_groups, label)[0]
        changed = None if fitted is None else bool(label != geometry or fitted != keys[candidate[i]])
        values.append(dict(shape_deviation=float(shape[i]), max_q2_deviation_deg=float(max_angle[i]),
                           starting_geometry=geometry, returned_geometry=label,
                           starting_assignment=list(map(int, assignments[candidate[i]])),
                           returned_assignment=a, starting_class=keys[candidate[i]], returned_class=fitted,
                           class_changed=changed, symbols=symbols,
                           X_A=[a.position.tolist() for a in c.complex.atoms]))
    return values
