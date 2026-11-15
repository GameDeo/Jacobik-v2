"""Reconstruct H-3x NUWFAM starts; no IK optimization or source mutation."""
import argparse
import json
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--arm', choices=['cap', 'control'], required=True)
a = ap.parse_args()
os.environ['JACOBIK_PROFILE'] = 'u1'
os.environ['JACOBIK_PROFILE_OVERRIDES'] = json.dumps({
    'nci': False, 'deterministic_embed': True,
    'embed_haptic_li_cap': a.arm == 'cap'})
os.environ['OMP_NUM_THREADS'] = '3'

import numpy as np
import torch
from tools.jgen.diag import ctxio
from tools.jdev import multiseed
from tools.jgen.embedding_validity import EmbeddingValidity
from jacobik.core.env import build_builder_and_inputs
from jacobik.builder import engine as E
from jacobik.ik import scorer as S, angle_bends as AB
from jacobik.ik.fk_compose import ComplexFK
from jacobik.ik.kabsch import _ref_rows
from jacobik.stats.orientation import haptic_features
from jacobik.stats.stats import StatsLookup, load_db

# Restore the old per-H descriptors in this diagnostic process only.
S.CAP_GEMINAL_FACE = False
AB.GEOMETRIC_LEAN_WALL = False


def leans(x):
    return [float(np.degrees(np.arcsin(np.clip(haptic_features(
        x[0], x[[1, 2]], False, [(x[1], x[h])])[1], -1, 1))))
        for h in (40, 41)]


def angle(x):
    u, v = x[[40, 41]] - x[1]
    return float(np.degrees(np.arccos(np.clip(
        u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))


def main():
    stats = StatsLookup(load_db(str(ctxio.CONSTRAINTS)))
    guard = EmbeddingValidity(E).install()
    guard.parent = 'NUWFAM'
    b, _, _ = build_builder_and_inputs(ctxio.load_parent('NUWFAM'))
    variants = b._ez_combinations()
    b._ez_inner = True
    root = Path('research/data/jgen/campaign/H/h3x/nuwfam_2060') / a.arm / 'xyz/NUWFAM'
    out = []
    with torch.no_grad(), multiseed.safe_cuda_eigh_chunking():
        for vi, fixes in enumerate(variants):
            for ligand, fix in zip(b.ligands, fixes):
                ligand.ez_fix = fix
            ctx, assign, width, _ = multiseed.prepare(
                b, 11, device='cuda', angle_bends=True, stats=stats)
            (fks, _, ge, sc), rows = multiseed.seed_rows(
                b, ctx, assign, width, (11, 12, 13))
            guard.check('NUWFAM')
            assert ge.metal_pos_idx == 0
            descriptors = []
            for h in sc._so_hapt:
                for j in range(len(h['sa'])):
                    if int(h['sa'][j, 0]) == 1:
                        descriptors.append((h['face'][j].tolist(), h['sx'][j].tolist()))
            assert ([1, 2], [40]) in descriptors and ([1, 2], [41]) in descriptors
            keep = b._class_budget_rows(ctx, assign, width) if E.CLASS_BUDGET == 'split' else None
            cfk = ComplexFK(fks)
            for seed, (theta, xref, _, cand) in zip((11, 12, 13), rows):
                if keep is not None:
                    kt = torch.as_tensor(keep, device='cuda')
                    theta = [t[kt] for t in theta]
                    xref = xref[kt] if xref.dim() == 3 else xref
                    cand = cand[keep]
                initial = cfk.apply(theta, xref) if cfk.supported else _ref_rows(xref, theta[0].shape[0])
                if not cfk.supported:
                    for fk, t in zip(fks, theta):
                        initial = fk.apply(t, initial)
                initial = initial.cpu().numpy()
                for p in sorted(root.glob(f'*s{seed}_row*.xyz')):
                    lines = p.read_text().splitlines()
                    header = json.loads(lines[1])
                    if header['row_key'][0] != vi:
                        continue
                    trial = int(header['row_key'][2])
                    assert int(cand[trial]) == int(header['row_key'][1])
                    final = np.asarray([[float(v) for v in line.split()[1:4]] for line in lines[2:]])
                    start_lean = leans(initial[trial])
                    final_angle = angle(final)
                    out.append(dict(arm=a.arm, path=str(p), seed=seed, variant=vi,
                                    assignment=int(cand[trial]), trial=trial,
                                    row=header['row_index'], start_face_leans_deg=start_lean,
                                    start_direction='both_away' if max(start_lean) < 0 else
                                    'both_toward' if min(start_lean) > 0 else 'mixed',
                                    mean_start_bend=float(np.mean(np.sin(np.radians(start_lean)))),
                                    final_angle_deg=final_angle, collapsed_lt15=final_angle < 15,
                                    is_top1=header['is_top1'], is_floor=header['is_floor'],
                                    mace_energy_kj=header['mace_energy_kj']))
    guard.close()
    assert len(out) == 96
    bad = [r for r in out if r['collapsed_lt15']]
    result = dict(arm=a.arm, rows=out, summary=dict(
        pool_rows=len(out), collapsed=len(bad),
        collapsed_both_away=sum(r['start_direction'] == 'both_away' for r in bad),
        collapsed_both_toward=sum(r['start_direction'] == 'both_toward' for r in bad),
        collapsed_mixed=sum(r['start_direction'] == 'mixed' for r in bad),
        collapsed_positive_mean_bend=sum(r['mean_start_bend'] > 0 for r in bad),
        start_both_abs_lean_lt10=sum(max(map(abs, r['start_face_leans_deg'])) < 10 for r in out)),
        caveat='Starts reconstructed by variant/assignment/trial; original exports have no initial-state hashes. No Li positions retained, so substituent signs are the directive proxy, not a direct Li/metal face measurement.')
    target = Path('research/data/jgen/campaign/H/h3d')
    target.mkdir(parents=True, exist_ok=True)
    (target / f'step0_{a.arm}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
