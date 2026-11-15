"""Coordination-sphere audit of each seed-11 #1 pick: per donor (Q1) and donor pair (Q2), the solved value, the crystal
value, and the table's log-likelihood at each (the pick's own candidate table, fit-time floors, as the ranking uses).

Question: when the coordination sphere is off, does the statistic prefer the crystal value (other terms / frozen
internals pull the solver away) or the solved value (the statistic itself points to the wrong place)?
Writes <OUT>/coord_audit.csv."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from pools import OUT, load_stats, set_cases, setup  # noqa: E402


START = sys.argv[1] if len(sys.argv) > 1 else "etkdg"


def symmetric_remap(mol, Xp, Xc):
    """The crystal's heavy atoms relabelled by the automorphism of the bonded heavy-atom graph (coordination bonds
    included) that best superimposes them on the pick, mirror images allowed. Without it, swapped identical donors
    or ligands (the same structure) were compared with the wrong partner: 42 of 116 picks showed D-M-D 'errors'
    > 30 deg, of which only 2 survive a permutation-invariant comparison (2026-10-15)."""
    import torch
    from jacobik.ik.dedup import StructureDeduper

    dd = StructureDeduper(mol)
    h = dd.heavy.numpy()
    A = torch.tensor(Xp[h] - Xp[h].mean(0), dtype=torch.float64)
    best = (-np.inf, None, None)
    for mirror in (False, True):
        B = Xc[h] - Xc[h].mean(0)
        if mirror:
            B = B * np.array([-1.0, 1.0, 1.0])
        B = torch.tensor(B, dtype=torch.float64)
        for s in range(0, dd.maps.shape[0], 2048):
            mp = dd.maps[s:s + 2048]
            H = torch.einsum("ni,mnj->mij", A, B[mp])  # sum_n A[n] B[map[n]]^T
            bound = 0.5 * ((A ** 2).sum() + (B ** 2).sum())
            ov = dd._max_overlap(H, bound.expand(H.shape[0]))
            j = int(torch.argmax(ov))
            if float(ov[j]) > best[0]:
                best = (float(ov[j]), mp[j].numpy(), mirror)
    out = Xc.copy()
    out[h] = Xc[h][best[1]]  # solver heavy atom n <- crystal heavy atom map[n] (distances/angles are mirror-blind)
    return out


def main():
    setup()
    import torch
    import jacobik.ik.scorer as SC
    from tools.jdev.cases import get_case

    names = {}
    for s in ("n48", "heldout28", "ring10", "dev32"):
        names.update({c: (s, n) for c, n in set_cases(s)})
    st = load_stats()
    rows = []
    for f in sorted((OUT / "pools" / START).rglob("*_s11.npz")):
        z = np.load(f, allow_pickle=False)
        if "error" in z.files:
            continue
        code = f.stem[:-4]
        try:
            case = get_case(names[code][1])
            r = case.make_builder().solve(seed=11, stats=st, angle_bends=True, n_conformers=1, n_ik_steps=2,
                                          device="cpu")
            sc, mol = r.scorer, r.conformers[0].complex
            sc.set_gmm_floor("fit")
            # crystal heavy atoms in solver order (H copied from the pick: irrelevant for Q1/Q2)
            m = case.tmqm_mol
            rb = {a.idx: i for i, a in enumerate(m.atoms)}
            amap = {0: rb[m.metal_idx]}
            ys = [a.symbol for a in mol.atoms]
            for lig, lg in zip(sorted(mol.ligands, key=lambda l: l.ligand_id), case.ligand_groups):
                for rd, x1 in lg["rdkit_to_xyz"].items():
                    if rd < len(lig.atom_indices) and ys[lig.atom_indices[rd] - 1] != "H":
                        amap[lig.atom_indices[rd] - 1] = rb[x1]
            i1 = int(np.argmax(z["score"]))
            Xp = z["X"][i1].copy()
            Xc = Xp.copy()
            for k, v in amap.items():
                Xc[k] = case.ref_X[v]
            Xc = symmetric_remap(mol, Xp, Xc)
            X = torch.tensor(np.stack([Xp, Xc]), dtype=torch.float32)
            asg = int(z["asg"][i1])
            with torch.no_grad():
                d = sc._ge.q1_distances(X)  # (2, n_sites)
                ang = sc._ge.q2_dmd_angles(X)  # (2, n_pairs)
            q1c = getattr(sc, "_q1_cand", None)
            if q1c is not None and asg < q1c[0].shape[0]:
                mu, ev, lw = (t[asg] for t in q1c)
            else:
                mu, ev, lw = sc._q1_means, sc._q1_eff_vars, sc._q1_log_w
            lp1 = torch.logsumexp(-0.5 * np.log(2 * np.pi) - 0.5 * torch.log(ev)[None] - 0.5 * (d[..., None] - mu[None]) ** 2
                                  / ev[None] + lw[None], -1)  # (2, n_sites)
            q2c = getattr(sc, "_q2_cand", None)
            if q2c is not None and asg < q2c[0].shape[0]:
                lp2 = SC._cos_mix_logp(torch.cos(ang), q2c[0][asg][None].expand(2, -1, -1), q2c[1][asg][None].expand(2, -1, -1),
                                       q2c[2][asg][None].expand(2, -1, -1), q2c[3][asg][None].expand(2, -1),
                                       has_t=sc._q2_cand_has_t)
            else:
                lp2 = SC._cos_mix_logp(torch.cos(ang), sc._q2_means, sc._q2_eff_vars, sc._q2_log_w, sc._q2_nu, has_t=sc._q2_has_t)
            for s in range(d.shape[1]):
                k = int(torch.argmax(lw[s] - 0.5 * (d[1, s] - mu[s]) ** 2 / ev[s]))  # component nearest the crystal
                rows.append(dict(set=names[code][0], case=code, q="Q1", i=s, solved=float(d[0, s]), crystal=float(d[1, s]),
                                 lp_solved=float(lp1[0, s]), lp_crystal=float(lp1[1, s]),
                                 mode_near_crystal=float(mu[s, k]), sd_mode=float(ev[s, k].sqrt()),
                                 n_modes=int((lw[s] > -1e3).sum())))
            for p in range(ang.shape[1]):
                rows.append(dict(set=names[code][0], case=code, q="Q2", i=p, solved=float(np.degrees(ang[0, p])),
                                 crystal=float(np.degrees(ang[1, p])), lp_solved=float(lp2[0, p]),
                                 lp_crystal=float(lp2[1, p])))
            print(f"{code}: ok", flush=True)
        except Exception as exc:
            print(f"{code}: ERROR {type(exc).__name__}: {str(exc)[:150]}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / f"coord_audit_{START}.csv", index=False)


if __name__ == "__main__":
    main()
