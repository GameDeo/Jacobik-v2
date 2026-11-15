"""Step 4 offline response calibration (roadmapv2 §§3.7, 3.9, 9).

No production inputs are opened here. Callers supply frozen paths, baseline raw
nats (including bend), split IDs and surfaces. Reference labels never depend on
surfaces. CPU MACE is opt-in and bounded by atoms and single-point counts; it is
not a replacement for the certified production solve/validation path.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable, Mapping

for _key, _value in {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                     "OPENBLAS_NUM_THREADS": "1"}.items():
    os.environ[_key] = _value

import numpy as np
import torch
from scipy.optimize import lsq_linear
from research.archive.code.interactions.nci_v2.nci_v2 import (CalibratedArtifact, SurfaceArtifact,
                                 RT_KJ_MOL, features_torch)

torch.set_num_threads(1)
T_GRID = (-1., -.5, 0., .5, 1.)
AMPLITUDES = {"torsion": math.pi / 18, "rotation": math.pi / 18,
              "bend": math.pi / 90, "translation": .1}


def jsonable(value):
    """Strict JSON conversion, including an explicit null for infinite condition."""
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, torch.Tensor):
        return jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    """CLI-stage helper; serialization performs no labeling or fitting."""
    from research.archive.code.interactions.nci_v2.nci_v2_devio import write_json
    write_json(path,jsonable(value))


def _cpu(value):
    if isinstance(value, torch.Tensor) and value.device.type != 'cpu':
        raise ValueError('response calibration requires CPU tensors')
    x = torch.as_tensor(value, dtype=torch.float64, device='cpu')
    if not torch.isfinite(x).all():
        raise ValueError('nonfinite coordinates/design')
    return x


def solver_dofs(theta, forward: Callable, kinds: Mapping[int, str], *, prefix='ik'):
    """Explicit solver adapter: forward(theta[B,Q]) -> full X[B,N,3].

    Kinds must come from the actual solver layout (6 base columns, absolute
    torsions, optional branch/macro joints, then angle bends). No inference of
    covalent bends or ring-closure freedom is made. Selection checks validity.
    """
    theta = _cpu(theta).reshape(-1)
    out = []
    for column, kind in sorted(kinds.items()):
        if kind not in AMPLITUDES or not 0 <= column < len(theta):
            raise ValueError('invalid explicit solver DOF')
        direction = torch.zeros_like(theta)
        direction[column] = AMPLITUDES[kind]

        def path(t, direction=direction):
            return forward((theta + t * direction)[None])[0]

        out.append(dict(id=f'{prefix}:{column}', kind=kind, path=path,
                        definition=dict(type='solver', theta0=theta.tolist(),
                                        column=column, amplitude=AMPLITUDES[kind]),
                        development_only=False))
    return out


def graph_dofs(X, graph=None, mol=None):
    """Conservative development fallback: rigid metal-free ligand components.

    Graph is {atom-position: neighbor-positions}, or mol has atoms/bonds/metal_idx.
    No internal torsions/bends are guessed. Full solver mapping remains required
    for production attribution. Rigid translations/rotations are solver base DOFs.
    """
    X = _cpu(X)
    metal = None
    if mol is not None:
        pos = {a.idx: i for i, a in enumerate(mol.atoms)}
        metal = pos.get(getattr(mol, 'metal_idx', -1))
        from jacobik.topology.graph import get_ligand_neighbors
        neighbors = get_ligand_neighbors(mol.atoms, mol.bonds, getattr(mol, 'metal_idx', -1))
        graph = {pos[a]: [pos[b] for b in adjacent] for a, adjacent in neighbors.items()
                 if pos[a] != metal}
    if graph is None:
        raise ValueError('explicit DOFs, graph or mol required')
    seen, out = set(), []
    for root in sorted(graph):
        if root in seen or root == metal:
            continue
        component = [root]
        seen.add(root)
        for i in component:
            for j in sorted(graph.get(i, ())):
                if j != metal and j not in seen:
                    seen.add(j)
                    component.append(j)
        idx = sorted(component)
        pivot = X[idx].mean(0)
        for kind in ('translation', 'rotation'):
            for axis_index in range(3):
                axis = torch.eye(3, dtype=X.dtype)[axis_index]

                def path(t, idx=idx, axis=axis, kind=kind, pivot=pivot):
                    Y = X.clone()
                    a = t * AMPLITUDES[kind]
                    if kind == 'translation':
                        Y[idx] = X[idx] + a * axis
                    else:
                        q = X[idx] - pivot
                        Y[idx] = pivot + q * torch.cos(a) + torch.linalg.cross(
                            axis.expand_as(q), q) * torch.sin(a) + (
                            q * axis).sum(-1, keepdim=True) * axis * (1 - torch.cos(a))
                    return Y

                out.append(dict(id=f'graph:{root}:{kind}:{axis_index}', kind=kind,
                                path=path, development_only=True,
                                definition=dict(type='rigid', atoms=idx, pivot=pivot.tolist(),
                                                axis=axis.tolist(), kind=kind,
                                                amplitude=AMPLITUDES[kind])))
    return out


def freeze_motion(path: Callable, *, motion_id, group_id, complex_id=None,
                  center='reference', definition=None, validity=None,
                  fd_step=1e-3, multiplier=1., development_only=False, include_half_step=False):
    """Freeze geometry BEFORE energy calls; save exact small FD points and tangent.

    validity(X[B,N,3]) must apply the independent frozen bond/clash/state and
    coordination rules, using the original reference shell even at other centers.
    Without this adapter, paths are rejected and retained as records.
    """
    if center not in ('reference', 'baseline', 'early_solve', 'mid_solve'):
        raise ValueError('unknown center label')
    if not 0 < fd_step < .5 or not multiplier > 0:
        raise ValueError('invalid path amplitude/finite difference step')
    def scaled(t):
        return _cpu(path(t * multiplier))
    t0 = torch.tensor(0., dtype=torch.float64)
    X0, tangent = torch.autograd.functional.jvp(scaled, t0, torch.ones_like(t0))
    positions = torch.stack([scaled(torch.tensor(t, dtype=torch.float64)) for t in T_GRID])
    fd = torch.stack([scaled(torch.tensor(t, dtype=torch.float64)) for t in (-fd_step, fd_step)])
    half_fd = torch.stack([scaled(torch.tensor(t, dtype=torch.float64))
                           for t in (-fd_step / 2, fd_step / 2)]) if include_half_step else None
    reasons = []
    if positions.ndim != 3 or positions.shape[-1] != 3 or X0.shape != tangent.shape:
        raise ValueError('path must return full X[N,3]')
    if not torch.allclose(positions[2], X0):
        raise ValueError('path is not deterministic')
    if not torch.allclose((fd[1] - fd[0]) / (2 * fd_step), tangent, atol=1e-6, rtol=1e-4):
        reasons.append('tangent_fd_mismatch')
    if float(tangent.norm()) <= 1e-10:
        reasons.append('zero_motion')
    check = None
    if validity is None:
        reasons.append('independent_validity_not_supplied')
    else:
        check = validity(torch.cat([positions, fd] + ([half_fd] if half_fd is not None else [])).detach())
        intact = check.get('valid', check.get('intact')) if isinstance(check, dict) else check
        if intact is None or not bool(np.asarray(jsonable(intact), dtype=bool).all()):
            reasons.append('independent_geometry_invalid')
    return jsonable(dict(schema='nci_v2_motion_v1', motion_id=motion_id,
                         group_id=str(group_id), complex_id=str(complex_id or group_id),
                         center=center, t=T_GRID, positions=positions, dXdt0=tangent,
                         fd_step=fd_step, fd_positions=fd, half_fd_positions=half_fd, multiplier=multiplier,
                         definition=definition, valid=not reasons, rejection_reasons=reasons,
                         validity=check, development_only=development_only,
                         units=dict(positions='Angstrom', t='dimensionless', dXdt0='Angstrom/t')))


def generate_motions(X, opportunities, *, group_id, complex_id=None, dofs=None,
                     mol=None, graph=None, center='reference', validity=None,
                     schema=None, wider_multiplier=4., max_local=3, include_half_step=False):
    """Geometry-only ranking; up to three valid paths plus one wider scan/family.

    Ranking is squared regularized feature displacement, never reward or energy.
    Wider scans use the best family-informative DOF and a predeclared multiplier.
    All attempted paths (including rejected ones) are returned. A family without
    an informative valid DOF gets an explicit unsupported record.
    """
    X = _cpu(X)
    candidates = dofs if dofs is not None else graph_dofs(X, graph, mol)
    ranked, records = [], []
    for dof in sorted(candidates, key=lambda d: d['id']):
        record = freeze_motion(dof['path'], motion_id=dof['id'], group_id=group_id,
                               complex_id=complex_id, center=center, validity=validity,
                               definition=dof.get('definition'),
                               development_only=dof.get('development_only', False),include_half_step=include_half_step)
        if not torch.allclose(_cpu(record['positions'])[2], X, atol=1e-6, rtol=1e-6):
            record['valid'] = False
            record['rejection_reasons'].append('solver_center_mismatch')
        feat = features_torch(_cpu(record['positions']), opportunities, schema).features
        scores = {f: float(((v - v[2:3]) ** 2).sum()) for f, v in feat.items()}
        record['radial_scores'] = {f: float(((v[..., 0] - v[2:3, :, 0]) ** 2).sum())
                                   for f, v in feat.items()}
        record['geometry_scores'] = scores
        record['selected'] = False
        record['scan'] = 'local'
        records.append(record)
        if record['valid'] and sum(scores.values()) > 1e-16:
            ranked.append((sum(scores.values()), dof, record))
    ranked.sort(key=lambda r: (-r[0], r[1]['id']))
    for _, _, record in ranked[:min(3, max(0, max_local))]:
        record['selected'] = True
    for family, occs in sorted(opportunities.opportunities.items()):
        if not occs:
            continue
        choices = sorted(ranked, key=lambda r: (-r[2]['radial_scores'].get(family, 0), r[1]['id']))
        if not choices or choices[0][2]['radial_scores'].get(family, 0) <= 1e-16:
            records.append(dict(group_id=str(group_id), family=family, scan='wider',
                                valid=False, selected=False,
                                rejection_reasons=['no_approach_retreat_solver_dof']))
            continue
        _, dof, _ = choices[0]
        record = freeze_motion(dof['path'], motion_id=f"{dof['id']}:wider:{family}",
                               group_id=group_id, complex_id=complex_id, center=center,
                               definition=dof.get('definition'), validity=validity,
                               multiplier=wider_multiplier,
                               development_only=dof.get('development_only', False),include_half_step=include_half_step)
        record.update(scan='wider', family=family, selected=record['valid'])
        records.append(record)
    return records


class MockBackend:
    """Independent analytic potential X[B,N,3] -> E[B] in kJ/mol.

    The backend has no surface/opportunity argument. Forces are minus its exact
    coordinate gradient in kJ/mol/Angstrom. Default is an origin harmonic well.
    """
    def __init__(self, potential=None):
        self.potential = potential or (lambda X: .5 * X.square().sum((1, 2)))
        self.evaluations = 0

    def evaluate(self, symbols, positions, charge, multiplicity, *, forces=False):
        X = _cpu(positions).detach().requires_grad_(forces)
        E = self.potential(X)
        if E.shape != (len(X),) or not torch.isfinite(E).all():
            raise ValueError('backend energy must be finite B-vector')
        F = None
        if forces:
            F = -_gradient(E.sum(), X)
        self.evaluations += len(X)
        return E.detach().numpy(), None if F is None else F.detach().numpy()


class MACEBackend:
    """Lazy opt-in CPU float64 full-complex single points, one at a time.

    No child processes. max_evaluations includes center and FD calls. Bounds
    limit work count/atoms, not hard wall time or RSS. No model download allowed.
    """
    def __init__(self, model_path, *, max_atoms=256, max_evaluations=100):
        if max_atoms < 1 or max_evaluations < 1:
            raise ValueError('positive MACE budgets required')
        self.model_path = Path(model_path)
        self.max_atoms, self.max_evaluations = max_atoms, max_evaluations
        self.evaluations = 0
        self._evaluator = None

    def evaluate(self, symbols, positions, charge, multiplicity, *, forces=False):
        from jacobik.benchmark.energy import MACEPolarEvaluator, EV_TO_KJ_MOL
        X = _cpu(positions).numpy()
        if X.ndim != 3 or X.shape[1:] != (len(symbols), 3):
            raise ValueError('full-complex symbols/positions mismatch')
        if len(symbols) > self.max_atoms or self.evaluations + len(X) > self.max_evaluations:
            raise RuntimeError('MACE CPU evaluation budget exceeded')
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        if self._evaluator is None:
            self._evaluator = MACEPolarEvaluator(model='polar-1-s', device='cpu', dtype='float64',
                                                model_path=self.model_path, threads=1)
        energies, force_values = [], []
        for x in X:
            self.evaluations += 1
            if forces:
                props = self._evaluator.properties(symbols, x, charge, multiplicity)
                energies.append(float(props['energy']) * EV_TO_KJ_MOL)
                force_values.append(np.asarray(props['forces']) * EV_TO_KJ_MOL)
            else:
                energies.append(self._evaluator.energy(symbols, x, charge, multiplicity) * EV_TO_KJ_MOL)
        return np.asarray(energies), np.asarray(force_values) if forces else None


def _gradient(value, X):
    if not value.requires_grad:
        return torch.zeros_like(X)
    grad = torch.autograd.grad(value, X, retain_graph=True, allow_unused=True)[0]
    return torch.zeros_like(X) if grad is None else grad


def _cells(surfaces):
    for family, art in sorted(surfaces.items()):
        for key, surface in sorted(art.surfaces.items()):
            mask = np.asarray(jsonable(surface.get('ref_only_mask', [])), dtype=bool)
            support = np.asarray(jsonable(surface['support_a']))
            if mask.size and mask.shape != support.shape:
                raise ValueError('ref_only mask/support shape mismatch')
            for index in np.argwhere(mask & (support > 0)) if mask.size else []:
                idx = tuple(int(i) for i in index)
                yield json.dumps([family, key, list(idx)], separators=(',', ':')), family, key, idx


def measure_motion(motion, symbols, charge, multiplicity, baseline, opportunities,
                   surfaces, backend, *, schema=None, occurrence_fn=None,
                   allow_development=False, include_cell_columns=True, profile=None):
    """Return JSONable energy rows plus one force row; all derivatives per t.

    ref_only cell contributions are computed from actual masks by isolating each
    basis cell and evaluating occurrences, never from a caller traversal boolean.
    These additive columns allow exact removal before calibration if unsupported.
    """
    if not motion.get('valid') or not motion.get('selected', True):
        raise ValueError('only selected geometrically valid paths may be labeled')
    if motion.get('development_only') and not allow_development:
        raise ValueError('graph-only DOFs require explicit development labeling')
    if occurrence_fn is None:
        from research.archive.code.interactions.nci_v2.nci_v2_surface import occurrence_rewards
        occurrence_fn = occurrence_rewards
    X = _cpu(motion['positions'])
    fd = _cpu(motion['fd_positions'])
    tangent = _cpu(motion['dXdt0'])
    if X.shape != (5, len(symbols), 3) or fd.shape != (2, len(symbols), 3):
        raise ValueError('invalid saved full-complex path')
    half_fd = motion.get('half_fd_positions')
    extra = [_cpu(half_fd)] if half_fd is not None else []
    allX = torch.cat([X, fd] + extra).requires_grad_(True)
    E, _ = backend.evaluate(symbols, allX.detach(), charge, multiplicity)
    _, F = backend.evaluate(symbols, X[2:3], charge, multiplicity, forces=True)
    E, F = np.asarray(E), np.asarray(F)
    if E.shape != (len(allX),) or F.shape != (1, len(symbols), 3) or not np.isfinite(E).all() or not np.isfinite(F).all():
        raise ValueError('invalid backend energies/forces')
    stage_t0=time.perf_counter(); B = baseline(allX)
    if profile is not None:profile['baseline_seconds']=profile.get('baseline_seconds',0.)+time.perf_counter()-stage_t0
    if B.shape != (len(allX),) or not torch.isfinite(B).all():
        raise ValueError('baseline adapter must return finite raw nats B-vector including bend')
    stage_t0=time.perf_counter()
    b_slope = (_gradient(B[2], allX)[2] * tangent).sum()
    if profile is not None:profile['baseline_gradient_seconds']=profile.get('baseline_gradient_seconds',0.)+time.perf_counter()-stage_t0
    stage_t0=time.perf_counter()
    u = occurrence_fn(allX, opportunities, surfaces, schema=schema)
    if profile is not None:profile['occurrence_seconds']=profile.get('occurrence_seconds',0.)+time.perf_counter()-stage_t0

    def columns(rewards):
        out = {}
        for fam, value in rewards.items():
            if value.ndim != 2 or len(value) != len(allX) or not torch.isfinite(value).all():
                raise ValueError('occurrence_rewards must return finite (B,O) tensors')
            total = value.sum(-1)
            grad_t0=time.perf_counter()
            slope = (_gradient(total[2], allX)[2] * tangent).sum()
            if profile is not None:profile['nci_gradient_seconds']=profile.get('nci_gradient_seconds',0.)+time.perf_counter()-grad_t0
            out[fam] = (-RT_KJ_MOL * (total - total[2]), -RT_KJ_MOL * slope)
        return out

    design = columns(u)
    cell_columns, cell_traversal = {}, {}
    for cell_id, fam, key, idx in (_cells(surfaces) if include_cell_columns else []):
        cell_t0=time.perf_counter()
        isolated = {fam: copy.deepcopy(surfaces[fam])}
        for art in isolated.values():
            for surface in art.surfaces.values():
                surface['support_a'] = np.zeros_like(np.asarray(jsonable(surface['support_a'])))
        isolated[fam].surfaces[key]['support_a'][idx] = float(
            np.asarray(jsonable(surfaces[fam].surfaces[key]['support_a']))[idx])
        masked_occurrences = occurrence_fn(allX, opportunities, isolated, schema=schema, ref_only=True)
        cell_columns[cell_id] = columns(masked_occurrences).get(fam, (torch.zeros(len(allX)), torch.tensor(0.)))
        occ = masked_occurrences.get(fam, allX[:, :0, 0])
        delta = -RT_KJ_MOL * (occ - occ[2:3])
        cell_traversal[cell_id] = delta.abs().amax(-1) if occ.shape[1] else torch.zeros(len(allX))
        if profile is not None:profile['cell_seconds']=profile.get('cell_seconds',0.)+time.perf_counter()-cell_t0
    reference_slope = float(-(F[0] * tangent.numpy()).sum())
    fd_slope = float((E[6] - E[5]) / (2 * motion['fd_step']))
    common = dict(group_id=motion['group_id'], complex_id=motion.get('complex_id', motion['group_id']),
                  motion_id=motion['motion_id'], center=motion['center'],
                  scan=motion.get('scan', 'local'), development_only=motion.get('development_only', False),
                  units='kJ/mol; derivatives per dimensionless t')
    rows = []
    for k, t in enumerate(motion['t']):
        if t == 0:
            continue
        rows.append(dict(common, kind='energy', t=t,
                         baseline_energy_change=float((-RT_KJ_MOL * (B[k] - B[2])).detach()),
                         reference_energy_change=float(E[k] - E[2]),
                         y=float(E[k] - E[2] + RT_KJ_MOL * (B[k] - B[2]).detach()),
                         D_f={f: float(v[0][k].detach()) for f, v in design.items()},
                         D_ref_only={c: float(v[0][k].detach()) for c, v in cell_columns.items()},
                         ref_only_occurrence_max_abs_D={c: float(v[k].detach()) for c, v in cell_traversal.items()}))
    rows.append(dict(common, kind='force', t=0., g=reference_slope + RT_KJ_MOL * float(b_slope.detach()),
                     reference_energy_slope=reference_slope,
                     baseline_energy_slope=-RT_KJ_MOL * float(b_slope.detach()),
                     G_f={f: float(v[1].detach()) for f, v in design.items()},
                     G_ref_only={c: float(v[1].detach()) for c, v in cell_columns.items()},
                     force_fd=dict(reference_slope=reference_slope, finite_difference=fd_slope,
                                   error=reference_slope - fd_slope, h=motion['fd_step'],
                                   baseline_slope=float(b_slope.detach()),
                                   baseline_fd=float(((B[6] - B[5]) / (2 * motion['fd_step'])).detach())),
                     evaluation_count=len(allX) + 1))
    if half_fd is not None:
        curvature = {}
        for name, minus, plus, h in (('full', 5, 6, motion['fd_step']),
                                     ('half', 7, 8, motion['fd_step'] / 2)):
            curvature[name] = dict(reference=float((E[minus] - 2 * E[2] + E[plus]) / h**2),
                baseline=float((-RT_KJ_MOL * (B[minus] - 2 * B[2] + B[plus]) / h**2).detach()),
                D_f={f: float(((v[0][minus] + v[0][plus]) / h**2).detach()) for f, v in design.items()})
        rows[-1]['curvature'] = curvature
    return jsonable(rows)


def _weights(rows, group_weights=None):
    """Equal group, then complex, then row weights (separately E and F)."""
    groups = sorted({r['group_id'] for r in rows})
    if not groups:
        return np.empty(0)
    gw = {g: float((group_weights or {}).get(g, 1.)) for g in groups}
    if any(not math.isfinite(w) or w <= 0 for w in gw.values()):
        raise ValueError('group weights must be positive finite')
    norm = sum(gw.values())
    complexes = defaultdict(set)
    counts = defaultdict(int)
    for r in rows:
        group = r['group_id']; complex_id = r.get('complex_id', group)
        complexes[group].add(complex_id); counts[(group, complex_id)] += 1
    return np.fromiter((gw[r['group_id']] / norm / len(complexes[r['group_id']]) /
                        counts[(r['group_id'], r.get('complex_id', r['group_id']))]
                        for r in rows),dtype=np.float64,count=len(rows))


def _robust_scale(values, weights):
    if not len(values):
        return 1.
    def median(x):
        order = np.argsort(x, kind='stable')
        cumulative = np.cumsum(weights[order])
        halfway = weights.sum() / 2
        index = min(len(order) - 1, int(np.searchsorted(cumulative, halfway - 1e-12)))
        if index + 1 < len(order) and np.isclose(cumulative[index], halfway, atol=1e-12, rtol=0):
            return (x[order[index]] + x[order[index + 1]]) / 2
        return x[order[index]]
    return max(1., float(1.4826 * median(np.abs(values - median(values)))))


def _assemble(rows, families, lambda_F=1., scales=None, group_weights=None):
    matrices, targets, scales_out = [], [], []
    for target, column, coefficient in (('y', 'D_f', 1.), ('g', 'G_f', lambda_F)):
        subset = [r for r in rows if target in r]
        y = np.array([r[target] for r in subset], dtype=float)
        A = np.array([[r.get(column, {}).get(f, 0.) for f in families] for r in subset], dtype=float).reshape(len(subset), len(families))
        w = _weights(subset, group_weights)
        scale = scales[len(scales_out)] if scales is not None else _robust_scale(y, w)
        scales_out.append(scale)
        matrices.append(A * np.sqrt(coefficient * w[:, None]) / scale)
        targets.append(y * np.sqrt(coefficient * w) / scale)
    return np.vstack(matrices), np.concatenate(targets), scales_out


def _eligible(rows, families):
    return np.array([len({r['group_id'] for r in rows if
                         abs(r.get('D_f', {}).get(f, 0.)) > 1e-10 or
                         abs(r.get('G_f', {}).get(f, 0.)) > 1e-10}) >= 10 for f in families])


def _fit(A, y, eligible, penalty=None):
    shared_column = A.sum(1)
    denom = float(shared_column @ shared_column)
    identified = denom > 1e-12
    shared = max(0., float(shared_column @ y) / denom) if identified else 0.
    gamma = np.full(A.shape[1], shared)
    if penalty is not None and identified and eligible.any():
        fixed = ~eligible
        design = A[:, eligible]
        target = y - A[:, fixed] @ gamma[fixed]
        n = int(eligible.sum())
        result = lsq_linear(np.vstack([design, math.sqrt(penalty) * np.eye(n)]),
                            np.concatenate([target, math.sqrt(penalty) * np.full(n, shared)]),
                            bounds=(0., np.inf), tol=1e-12)
        if not result.success:
            raise RuntimeError(f'family NNLS failed: {result.message}')
        gamma[eligible] = result.x
    return shared, gamma, identified


def _disable_uncovered(surfaces, labels):
    """Disable individual uncovered cells and remove their exact design columns.

    Cell contributions are mandatory whenever masks are active. Accepting an
    external traversal flag would allow unrelated occurrences to certify a well.
    """
    surfaces, labels = copy.deepcopy(surfaces), copy.deepcopy(labels)
    counts, passed, disabled = {}, {f: True for f in surfaces}, []
    for cell, fam, key, idx in list(_cells(surfaces)):
        if any(('y' in r and cell not in r.get('D_ref_only', {})) or
               ('g' in r and cell not in r.get('G_ref_only', {})) for r in labels):
            raise ValueError('active ref_only cells require measured per-cell response columns')
        groups = {r['group_id'] for r in labels if r.get('ref_only_occurrence_max_abs_D', {}).get(
            cell, abs(r.get('D_ref_only', {}).get(cell, 0.))) > .1}
        counts[cell] = len(groups)
        if len(groups) >= 5:
            continue
        disabled.append(cell)
        passed[fam] = False
        s = surfaces[fam].surfaces[key]
        support = np.asarray(jsonable(s['support_a']), dtype=float).copy()
        support[idx] = 0.
        s['support_a'] = support
        for r in labels:
            for col, refcol in (('D_f', 'D_ref_only'), ('G_f', 'G_ref_only')):
                if refcol in r:
                    r.setdefault(col, {})[fam] = r.get(col, {}).get(fam, 0.) - r[refcol].get(cell, 0.)
    return surfaces, labels, passed, counts, disabled


def fit_response_scales(surfaces, motion_labels, calibration_split, spec=None, *, progress_callback=None):
    """Grouped five-fold shared-first NNLS, one-SE selection and bootstrap.

    spec: lambda_F (default 1), bootstrap_samples (200), seed (0), group_weights,
    allow_development (False). Lambdas and minimum evidence rules are fixed by
    §3.7. No validation/production energies are read. Noncalibration labels are
    excluded before coverage, scales, CV and bootstrap. Call separately for
    development diagnostics; trajectory centers are excluded from fitting.
    """
    sp = dict(spec or {})
    if isinstance(surfaces, SurfaceArtifact):
        surfaces = {surfaces.family: surfaces}
    families = sorted(surfaces)
    if not families:
        raise ValueError('at least one surface family required')
    allowed = {str(g) for g in calibration_split}
    labels = [copy.deepcopy(r) for r in motion_labels if str(r['group_id']) in allowed
              and r.get('center', 'reference') in ('reference', 'baseline')]
    for r in labels:
        r['group_id'] = str(r['group_id'])
        if r.get('development_only') and not sp.get('allow_development', False):
            raise ValueError('development-only labels cannot calibrate a production artifact')
        for target, col in (('y', 'D_f'), ('g', 'G_f')):
            if target in r and not np.isfinite([r[target], *r.get(col, {}).values()]).all():
                raise ValueError('nonfinite response label')
        if 'y' not in r and 'g' not in r:
            raise ValueError('label must contain y or g')
    if sp.get('coverage_frozen', False):
        # Step 4 A3 already traversed every connected ref-only peak and froze
        # unsupported cells out of support_a. In this mode labels need not carry
        # redundant per-cell response columns; family responses use that frozen
        # surface directly.
        surfaces, labels = copy.deepcopy(surfaces), copy.deepcopy(labels)
        passed = {f: True for f in surfaces}
        counts, disabled = {}, []
    else:
        surfaces, labels, passed, counts, disabled = _disable_uncovered(surfaces, labels)
    lambda_F = float(sp.get('lambda_F', 1.))
    if not math.isfinite(lambda_F) or lambda_F < 0:
        raise ValueError('lambda_F must be nonnegative finite')
    gw = sp.get('group_weights')
    groups = sorted({r['group_id'] for r in labels}, key=lambda g: hashlib.sha256(
        (str(sp.get('seed', 0)) + ':' + g).encode()).hexdigest())
    fold_records = []
    eligible = _eligible(labels, families)
    candidates = (.1, 1., 10.)
    if len(groups) >= 5:
        folds = [groups[i::5] for i in range(5)]
        for k, heldout in enumerate(folds):
            train = [r for r in labels if r['group_id'] not in heldout]
            test = [r for r in labels if r['group_id'] in heldout]
            A, y, scales = _assemble(train, families, lambda_F, group_weights=gw)
            shared, _, _ = _fit(A, y, _eligible(train, families))
            losses = {str(p): [] for p in candidates}
            shared_losses = []
            fits = {p: _fit(A, y, _eligible(train, families), p)[1] for p in candidates}
            for group in heldout:
                Ag, yg, _ = _assemble([r for r in test if r['group_id'] == group], families,
                                      lambda_F, scales=scales, group_weights=gw)
                shared_losses.append(float(np.square(Ag @ np.full(len(families), shared) - yg).sum()))
                for p in candidates:
                    losses[str(p)].append(float(np.square(Ag @ fits[p] - yg).sum()))
            fold_records.append(dict(fold=k, train_groups=[g for g in groups if g not in heldout],
                                     heldout_groups=heldout, s_E=scales[0], s_F=scales[1],
                                     gamma_shared=shared, shared_losses=shared_losses,
                                     family_losses=losses))
    selected, best_penalty, comparison = 'shared', None, {}
    if fold_records and eligible.any():
        # Paired fold-level differences: the five folds are the CV replicates.
        best = min(candidates, key=lambda p: (np.mean([np.mean(f['family_losses'][str(p)])
                                                     for f in fold_records]), -p))
        differences = np.array([np.mean(f['shared_losses']) - np.mean(f['family_losses'][str(best)])
                                for f in fold_records])
        improvement = float(differences.mean())
        se = float(differences.std(ddof=1) / math.sqrt(5))
        comparison = dict(lambda_gamma=best, improvement=improvement, standard_error=se)
        if improvement > se and improvement > 1e-12:
            selected, best_penalty = 'family', best
    A, y, scales = _assemble(labels, families, lambda_F, group_weights=gw)
    shared, gamma, identified = _fit(A, y, eligible, best_penalty)
    singular = np.linalg.svd(A, compute_uv=False) if A.size else np.zeros(len(families))
    rank = int(np.count_nonzero(singular > max(1e-12, singular[0] * 1e-10))) if len(singular) else 0
    condition = float(singular[0] / singular[-1]) if rank == len(families) and len(singular) else math.inf
    # Compare energy-only shared fit; do not use it for candidate selection.
    AE, yE, _ = _assemble(labels, families, 0., group_weights=gw)
    shared_energy = _fit(AE, yE, eligible)[0]
    rng = np.random.default_rng(int(sp.get('seed', 0)))
    draws = []
    n_boot = int(sp.get('bootstrap_samples', 200))
    if n_boot < 0:
        raise ValueError('bootstrap_samples must be nonnegative')
    labels_by_group = defaultdict(list)
    for row in labels:
        labels_by_group[row['group_id']].append(row)
    for boot_index in range(n_boot if groups else 0):
        resampled, boot_weights = [], {}
        for k, original in enumerate(rng.choice(groups, size=len(groups), replace=True)):
            name = f'resample:{k}'
            boot_weights[name] = (gw or {}).get(original, 1.)
            resampled.extend(dict(r, group_id=name) for r in labels_by_group[original])
        Ab, yb, _ = _assemble(resampled, families, lambda_F, group_weights=boot_weights)
        gb, fb, _ = _fit(Ab, yb, eligible, best_penalty)
        draws.append([gb, *fb])
        if progress_callback is not None and (boot_index+1)%250==0:
            progress_callback(boot_index+1,n_boot)
    uncertainty = {}
    if draws:
        values = np.asarray(draws)
        for i, name in enumerate(['shared', *families]):
            parameter_identified = identified if i == 0 or selected == 'shared' else rank == len(families)
            uncertainty[name] = dict(identified=parameter_identified,
                std=float(values[:, i].std()) if parameter_identified else None,
                ci95=np.quantile(values[:, i], [.025, .975]).tolist() if parameter_identified else None)
    return CalibratedArtifact(
        surface_artifacts=surfaces, gamma_shared=shared,
        family_scales={f: float(gamma[i]) for i, f in enumerate(families)},
        selected_mode=selected, singular_values=singular.tolist(), condition_number=condition,
        ref_only_traversal_passed=passed, spec=sp,
        metadata=jsonable(dict(schema='nci_v2_response_v1', identified=identified,
                               family_design_rank=rank, n_groups=len(groups),
                               s_E=scales[0], s_F=scales[1], lambda_F=lambda_F,
                               family_eligible=dict(zip(families, eligible.tolist())),
                               cv_folds=fold_records, cv_comparison=comparison,
                               selected_lambda_gamma=best_penalty,
                               gamma_shared_energy_only=shared_energy,
                               bootstrap_samples=len(draws), uncertainty=uncertainty,
                               coverage_frozen=bool(sp.get('coverage_frozen', False)),
                               coverage_provenance=sp.get('coverage_provenance'),
                               a3_disabled_ref_only_components=sp.get('a3_disabled_ref_only_components'),
                               a3_disabled_ref_only_cells=sp.get('a3_disabled_ref_only_cells'),
                               ref_only_cell_traversing_groups=counts, disabled_ref_only_cells=disabled,
                               limitations=['Graph-only motions are development-only.',
                                            'CPU MACE budget bounds work count/atoms, not wall time or RSS.',
                                            'Trajectory centers are diagnostics only.',
                                            'Requires the new context-aware surface evaluator; certified runtime drops corrections.'])))


def artifact_from_json(data):
    """Stage-runner helper to load the plain JSON deployment artifact."""
    data = copy.deepcopy(data)
    data['surface_artifacts'] = {f: SurfaceArtifact(**a) for f, a in data['surface_artifacts'].items()}
    if data.get('condition_number') is None:
        data['condition_number'] = math.inf
    return CalibratedArtifact(**data)


class RawScorerBaseline:
    """Recover frozen scorer raw nats without changing its quantity weights.

    Scorer.score returns (weighted normalized total, breakdown). BVS and NCI
    already present in that frozen scorer remain included. No term is toggled.
    The separate solver bend prior must be supplied in raw nats when active;
    caller declares bends_active because the scorer cannot reveal FK bend state.
    Production scorer/solver reconstruction is deliberately not wired here.
    """
    def __init__(self, scorer, row_group=0, bend_reward=None, *, bends_active=False):
        if bends_active and bend_reward is None:
            raise ValueError('active solver bends require raw bend_reward callback')
        self.scorer, self.row_group, self.bend_reward = scorer, int(row_group), bend_reward

    def __call__(self, X):
        X = _cpu(X)
        row_groups = torch.full((len(X),), self.row_group, dtype=torch.long, device=X.device)
        # S2b scorer tensors are reconstructed in the solver's float32 dtype.
        # Keep the cast differentiable so response slopes still flow to X.
        total, _ = self.scorer.score(X.to(dtype=torch.float32), row_groups=row_groups)
        norm = float(self.scorer._agg_norm())
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError('invalid scorer normalization')
        raw = total * norm
        if self.bend_reward is not None:
            bend = self.bend_reward(X)
            if bend.shape != raw.shape:
                raise ValueError('raw bend reward must be B-vector')
            raw = raw + bend
        return raw


def harmonic_mock_baseline(X):
    """Explicit development-only harmonic baseline, raw nats, not a fitted scorer."""
    return -.5 * X.square().sum((1, 2)) / RT_KJ_MOL


def response_diagnostics(motion_labels, artifact, *, groups=None,
                         curvature_rtol=.05, curvature_atol=1e-3, sign_tolerance=1e-8):
    """Pure Gate 3 diagnostics on explicitly supplied labels; never opens data.

    Group -> parent/complex -> row balance for E/F and group -> parent -> path
    balance for regret. Total-energy sampled minima include the saved zero
    center. Curvature sign attribution requires both h and h/2 and numerical
    convergence of reference AND predicted total energy curvature. Missing
    half-step data are explicitly unresolved. No untouched validation is labeled.
    """
    allowed = None if groups is None else {str(g) for g in groups}
    labels = [copy.deepcopy(r) for r in motion_labels if allowed is None or str(r['group_id']) in allowed]
    # Diagnostic designs must match the disabled support in the fitted artifact.
    for cell in artifact.metadata.get('disabled_ref_only_cells', []):
        fam = json.loads(cell)[0]
        for r in labels:
            for col, refcol in (('D_f', 'D_ref_only'), ('G_f', 'G_ref_only')):
                if col in r and cell not in r.get(refcol, {}):
                    raise ValueError('diagnostics require per-cell responses for disabled cells')
                if refcol in r:
                    r.setdefault(col, {})[fam] = r.get(col, {}).get(fam, 0.) - r[refcol].get(cell, 0.)
    scales = artifact.family_scales
    def prediction(r, key):
        return sum(v * scales.get(f, artifact.gamma_shared) for f, v in r.get(key, {}).items())
    energy = [r for r in labels if 'y' in r]
    slope = [r for r in labels if 'g' in r]
    def rmse(rows, target, column):
        if not rows:
            return None
        errors = np.array([prediction(r, column) - r[target] for r in rows])
        return float(np.sqrt(_weights(rows) @ errors**2))
    def wrong(rows, target, column):
        flags = [r for r in rows if r[target] * prediction(r, column) < -sign_tolerance**2]
        return dict(count=len(flags), n=len(rows), balanced_fraction=float(
            _weights(rows) @ np.array([r[target] * prediction(r, column) < -sign_tolerance**2
                                       for r in rows])) if rows else None)
    total_energy = [dict(r, actual=r.get('reference_energy_change', r['y'] + r['baseline_energy_change']),
                         total_prediction={'total': r['baseline_energy_change'] + prediction(r, 'D_f')})
                    for r in energy if 'baseline_energy_change' in r]
    total_slope = [dict(r, actual=r['reference_energy_slope'],
                        total_prediction={'total': r['baseline_energy_slope'] + prediction(r, 'G_f')})
                   for r in slope if 'reference_energy_slope' in r and 'baseline_energy_slope' in r]
    def total_wrong(rows):
        w = _weights(rows)
        flags = [r['actual'] * r['total_prediction']['total'] < -sign_tolerance**2 for r in rows]
        return dict(count=sum(flags), n=len(rows), balanced_fraction=float(w @ flags) if rows else None)
    paths = {}
    for r in energy:
        key = (r['group_id'], r.get('complex_id', r['group_id']), r.get('motion_id', 'unspecified'))
        paths.setdefault(key, []).append(r)
    regrets, unavailable = [], 0
    for (group, parent, mid), rows in paths.items():
        if any('baseline_energy_change' not in r for r in rows):
            unavailable += 1
            continue
        actual = [0.] + [r.get('reference_energy_change', r['y'] + r['baseline_energy_change']) for r in rows]
        predicted = [0.] + [r['baseline_energy_change'] + prediction(r, 'D_f') for r in rows]
        # Ties use the center first, then saved scan order; regret is sampled only.
        picked = int(np.argmin(predicted))
        regrets.append(dict(group_id=group, complex_id=parent, motion_id=mid,
                            regret=float(actual[picked] - min(actual))))
    curvature = dict(n=len(slope), resolved=0, unresolved=0, unavailable=0, wrong_sign=0, paths=[])
    for r in slope:
        c = r.get('curvature')
        item = dict(group_id=r['group_id'], motion_id=r.get('motion_id'))
        if not c or 'full' not in c or 'half' not in c or artifact.metadata.get('disabled_ref_only_cells'):
            # Per-cell curvature removal is not stored: conservatively unavailable
            # after any support changes rather than reporting a stale sign.
            curvature['unavailable'] += 1
            curvature['unresolved'] += 1
            item.update(status='unavailable', numerical_uncertainty=None)
        else:
            ref = [c[k]['reference'] for k in ('full', 'half')]
            pred = [c[k]['baseline'] + prediction(c[k], 'D_f') for k in ('full', 'half')]
            errors = [abs(v[1] - v[0]) / 3 for v in (ref, pred)]
            resolved = all(e <= curvature_atol + curvature_rtol * abs(v[1])
                           for e, v in zip(errors, (ref, pred)))
            resolved = resolved and abs(ref[1]) > 5 * errors[0]
            curvature['resolved' if resolved else 'unresolved'] += 1
            wrong_sign = resolved and ref[1] * pred[1] < -sign_tolerance**2 and all(
                abs(v[1]) > e + curvature_atol for e, v in zip(errors, (ref, pred)))
            curvature['wrong_sign'] += int(wrong_sign)
            item.update(status='resolved' if resolved else 'unresolved',
                        reference=ref[1], predicted=pred[1], numerical_uncertainty=errors,
                        wrong_sign=wrong_sign)
        curvature['paths'].append(item)
    resolved_paths=[p for p in curvature['paths'] if p['status']=='resolved']
    curvature['median_absolute_error']=float(np.median([abs(p['reference']-p['predicted']) for p in resolved_paths])) if resolved_paths else None
    positive=[p['predicted']/p['reference'] for p in resolved_paths if p['reference']>0 and p['predicted']>0]
    curvature['median_positive_ratio']=float(np.median(positive)) if positive else None
    by_group={}
    for group in sorted({r['group_id'] for r in energy}):
        rr=[r for r in energy if r['group_id']==group];w=_weights(rr)
        by_group[group]=[float(w@np.array([r['y']**2 for r in rr])),float(w@np.array([(prediction(r,'D_f')-r['y'])**2 for r in rr]))]
    control_comparison={'status':'unavailable','incumbent':'not_supplied'}
    if by_group:
        values=np.array(list(by_group.values()));rng=np.random.default_rng(0)
        draws=rng.integers(0,len(values),size=(4000,len(values)));means=values[draws].mean(1)
        improvements=np.sqrt(means[:,0])-np.sqrt(means[:,1])
        control_comparison=dict(status='development_diagnostic',seed=0,bootstrap_groups=4000,
            energy_rmse_improvement=float(np.sqrt(values[:,0].mean())-np.sqrt(values[:,1].mean())),
            ci95=np.quantile(improvements,[.025,.975]).tolist(),incumbent='not_supplied')
    return jsonable(dict(schema='nci_v2_response_diagnostics_v1',
                         control_comparison=control_comparison,
                         n_groups=len({r['group_id'] for r in labels}),
                         energy_rmse=rmse(energy, 'y', 'D_f'),
                         slope_rmse=rmse(slope, 'g', 'G_f'),
                         wrong_sign_energy=wrong(energy, 'y', 'D_f'),
                         wrong_sign_slope=wrong(slope, 'g', 'G_f'),
                         wrong_sign_total_energy=total_wrong(total_energy),
                         wrong_sign_total_slope=total_wrong(total_slope),
                         sampled_minimum_regret=dict(mean=float(_weights(regrets) @ np.array(
                             [r['regret'] for r in regrets])) if regrets else None,
                             unavailable_paths=unavailable, paths=regrets),
                         curvature=curvature,
                         units='energy/regret kJ/mol; slopes kJ/mol/t; curvature kJ/mol/t^2',
                         limitations=['Sampled-minimum regret is limited to measured paths.',
                                      'Curvature uncertainty unavailable without step-halving points.',
                                      'Production baseline reconstruction is not wired.']))
