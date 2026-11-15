"""Why does the Q6 PMF lose to the walls? (2026-10-16 f) Component decomposition and component-wise re-ranking.

Stage `terms` (one scorer build per pool, CPU): for the conformers of BOTH arms' pools of a complex / seed (the wall
arm u1+lean and the PMF arm u1+lean+pmf), every Q6 pair's wall term and PMF term, summed per conformer by pair
category and region:
    categories  hp (polar H ... acceptor, the H-bond pairs), m (metal ... X), c4 (exactly 4 ligand bonds), c5 (rest),
                nt (pairs without a PMF table: they keep the wall in both models)
    regions     core (d < the table's onset r0, GFN2 core) / data (d >= r0, -ln g)
plus the core excess w(d) - w(r0), the polar-H pairs without the angle classes and their attractive part, and the
other ranking terms. Written to <OUT>/q6diag/<set>/<code>_s<seed>.npz.

Stage `report`: re-ranks the union of the two pools under Q6 variants (the rest of the u1 ranking score fixed) and
compares each variant's #1 (MACE float64 dE) with the walls' #1 on the same conformers; then accounts for the PMF
arm's picks term by term.

    python tools/jdev/budget/q6_diag.py terms   (JACOBIK_PROFILE=u1, JACOBIK_PROFILE_OVERRIDES q6_model=pmf)
    python tools/jdev/budget/q6_diag.py report
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import pool_energy, pool_files, pool_path, read_pool  # noqa: E402
from pools import OUT, load_stats, rank_terms, set_cases, setup  # noqa: E402

ARM_W, ARM_P = "etkdg_u1_lean", "etkdg_u1_lean_pmf"
CATS = ("hp", "m", "c4", "c5", "nt")
KCAL = 4.184


def pair_arrays(sc, mol, X):
    """Per pair (B, M): distance d, wall term, PMF term as run (eff), raw PMF w (angle-mixed), w without the angle
    classes, core mask, core excess; per pair (M,): category masks, onset r0, class label."""
    import torch
    from research.archive.code.interactions import q6pmf as Q

    rt, ge = sc._q6_pmf, sc._ge
    a, b = ge._q6_a, ge._q6_b

    atoms = list(mol.atoms)
    sep = Q.ligand_separation(mol)
    cls = [Q.pair_class(atoms[i].idx, atoms[j].idx, mol.metal_idx, sep) for i, j in zip(a.tolist(), b.tolist())]
    has, hp = rt["has"], rt["hp"]
    cat = {"nt": ~has, "hp": has & hp}
    for c, lab in (("m", "m"), ("c4", "4"), ("c5", "5")):
        cat[c] = has & ~hp & torch.tensor([x == lab for x in cls])
    d = (X[:, a] - X[:, b]).norm(dim=-1)
    I = lambda rows, dd: Q._interp(rt["T"], rows, dd, rt["r_lo"], rt["dr"], rt["G"])
    w_tot = I(rt["tot"], d)
    r0 = rt["r0"][rt["tot"]]
    w = w_tot
    w0 = I(rt["tot"], r0[None].expand_as(d))
    if rt["hb_any"]:
        H, Dn, A = X[:, rt["h"]], X[:, rt["d"]], X[:, rt["acc"]]
        u, v = Dn - H, A - H
        c = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1)).clamp(min=1e-8)
        s = torch.sigmoid((Q.ANGLE_SPLIT_COS - c) / 0.08)
        r0e = r0[None].expand_as(d)
        wl, wb = I(rt["lin"], d), I(rt["bent"], d)
        w = torch.where(rt["hb"][None], s * wl + (1 - s) * wb, w)
        w0 = torch.where(rt["hb"][None], s * I(rt["lin"], r0e) + (1 - s) * I(rt["bent"], r0e), w0)
    t = (rt["mu"] - d).clamp(min=0.0)
    wall = t ** 2 / (2.0 * rt["sigma"].clamp(min=1e-7) ** 2)
    eff = torch.where(hp[None], w, w.clamp(min=0.0))     # as PMF_WELLS = "hbond"
    eff = torch.where(has[None], eff, wall)
    core = d < r0[None]
    xc = torch.where(core, (w - w0).clamp(min=0.0), torch.zeros_like(w))   # core excess above w(r0)
    return dict(d=d, wall=wall, eff=eff, w=w, w_tot=w_tot, core=core, xc=xc, cat=cat, r0=r0, cls=cls)


def pair_terms(sc, mol, X):
    """{name: (B,) kT} sums of the Q6 pair terms (see module doc) for conformers X (B, N, 3)."""
    pa = pair_arrays(sc, mol, X)
    d, wall, eff, w, w_tot, core, xc, cat = (pa[k] for k in ("d", "wall", "eff", "w", "w_tot", "core", "xc", "cat"))
    rt = sc._q6_pmf
    import torch
    out = {}
    for c, m in cat.items():
        m = m[None]
        out[f"W_{c}"] = (wall * m).sum(-1)
        out[f"Pc_{c}"] = (eff * (m & core)).sum(-1)
        out[f"Pd_{c}"] = (eff * (m & ~core)).sum(-1)
        out[f"Xc_{c}"] = (torch.where(m, xc, torch.zeros_like(xc))).sum(-1)
        out[f"max_{c}"] = (torch.maximum(eff, wall) * m).sum(-1)
    m = cat["hp"][None]
    out["hp_pos"] = (w.clamp(min=0.0) * m).sum(-1)
    out["hp_negdata"] = (w.clamp(max=0.0) * (m & ~core)).sum(-1)
    out["hp_noang"] = (w_tot * m).sum(-1)
    out["hp_noang_negdata"] = (w_tot.clamp(max=0.0) * (m & ~core)).sum(-1)
    # where the polar-H attraction sits: data-region well by distance band
    for lo, hi in ((0.0, 2.2), (2.2, 2.8), (2.8, 6.0)):
        band = m & ~core & (d >= lo) & (d < hi)
        out[f"hp_neg_{lo:.1f}_{hi:.1f}"] = (w.clamp(max=0.0) * band).sum(-1)
    out["n_hp_lt22"] = ((d < 2.2) & m).sum(-1).float()
    if rt.get("n_m13", 0):
        raise SystemExit("metal 1-3 rows in the runtime: run with q6_m13 off")
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


def cmd_terms(a):
    setup()
    import torch
    import jacobik.ik.scorer as SC
    from tools.jdev.cases import get_case

    assert SC.Q6_MODEL == "pmf", "run with JACOBIK_PROFILE_OVERRIDES q6_model=pmf"
    st = load_stats()
    for s in a.sets:
        names = dict(set_cases(s))
        od = OUT / "q6diag" / s
        od.mkdir(parents=True, exist_ok=True)
        for _, code, seed, fw in pool_files(ARM_W, [s]):
            fp = pool_path(ARM_P, s, code, seed)
            fo = od / fw.name
            if fo.exists() or not fp.exists():
                continue
            zs = [read_pool(f) for f in (fw, fp)]
            es = [pool_energy(f, "mace64") for f in (fw, fp)]
            if any(z is None for z in zs) or any(e is None for e in es):
                continue
            try:
                r = get_case(names[code]).make_builder().solve(seed=int(seed), stats=st, angle_bends=True,
                                                                n_conformers=1, n_ik_steps=2, device="cpu")
                sc = r.scorer
                if sc._q6_pmf is None:
                    print(f"{code} s{seed}: no PMF runtime", flush=True)
                    continue
                X = torch.tensor(np.concatenate([z["X"] for z in zs]), dtype=torch.float32)
                rg = torch.as_tensor(np.concatenate([z["asg"] for z in zs]), dtype=torch.long)
                with torch.no_grad():
                    tot, bd = rank_terms(sc, X, rg)
                    pt = pair_terms(sc, r.conformers[0].complex, X)
                Z = float(sc._agg_norm())
                qw = float(sc._q6_qweight)
                save = dict(arm=np.concatenate([np.full(len(z["X"]), i) for i, z in enumerate(zs)]),
                            mace=np.concatenate(es),
                            extra=np.concatenate([z["score"] - z["rank"] for z in zs]),
                            stored_rank=np.concatenate([z["rank"] for z in zs]),
                            rank_pmf=tot.numpy(), Z=Z, qw=qw, n_pairs=int(sc._ge.n_q6))
                save.update({f"t_{q}": v.numpy() for q, v in bd.items()})
                save.update({f"q6_{k}": v for k, v in pt.items()})
                np.savez_compressed(fo, **save)
                # consistency: the walls reproduce the wall arm's stored ranking, the PMF the PMF arm's
                base = save["rank_pmf"] - save["t_Q6"]
                Wt = sum(pt[f"W_{c}"] for c in CATS)
                Pt = sum(pt[f"Pc_{c}"] + pt[f"Pd_{c}"] for c in CATS)
                err_w = np.abs((base - qw * Wt / Z) - save["stored_rank"])[save["arm"] == 0].max()
                err_p = np.abs((base - qw * Pt / Z) - save["stored_rank"])[save["arm"] == 1].max()
                print(f"{s}/{code} s{seed}: {len(X)} conformers, {sc._ge.n_q6} pairs; |rank - stored| wall "
                      f"{err_w:.2e} pmf {err_p:.2e}", flush=True)
            except Exception as exc:
                print(f"{code} s{seed}: skipped ({type(exc).__name__}: {str(exc)[:200]})", flush=True)


def variants(q):
    """{name: (B,) Q6 penalty in kT} from the stored pair sums."""
    S = lambda p, cats=CATS: sum(q[f"{p}_{c}"] for c in cats)
    P = S("Pc") + S("Pd")
    hpP = q["Pc_hp"] + q["Pd_hp"]
    nonhp = ("m", "c4", "c5")
    return {
        "W  walls (adopted)": S("W"),
        "P  PMF as run (hbond wells)": P,
        "P - polar-H wells (hp clamped >= 0)": P - hpP + q["hp_pos"],
        "P - angle classes (hp total table)": P - hpP + q["hp_noang"],
        "P core only (data region 0, all pairs)": S("Xc", ("hp",) + nonhp) + q["W_nt"],
        "P core, + polar-H full": S("Xc", nonhp) + hpP + q["W_nt"],
        "P - barriers (non-hp data region 0)": P - S("Pd", nonhp),
        "P, metal pairs by walls": P - q["Pc_m"] - q["Pd_m"] + q["W_m"],
        "P, c4 pairs by walls": P - q["Pc_c4"] - q["Pd_c4"] + q["W_c4"],
        "P, c5 pairs by walls": P - q["Pc_c5"] - q["Pd_c5"] + q["W_c5"],
        "P, polar-H pairs by walls": P - hpP + q["W_hp"],
        "max(W, P) per pair": S("max"),
        "W + polar-H wells (data region < 0)": S("W") + q["hp_negdata"],
        "W + polar-H wells, no angle": S("W") + q["hp_noang_negdata"],
        "W + polar-H wells < 2.2 A only": S("W") + q["hp_neg_0.0_2.2"],
        "W, polar-H pairs by PMF": S("W") - q["W_hp"] + hpP,
    }


def cmd_report(a):
    rng = np.random.default_rng(0)
    files = sorted(f for s in a.sets for f in (OUT / "q6diag" / s).glob("*.npz"))
    pools = []
    for f in files:
        z = dict(np.load(f))
        z["pool"] = f"{f.parent.name}|{f.stem}"
        pools.append(z)
    print(f"{len(pools)} pools (wall arm + PMF arm conformers each)")
    base_fn = lambda z: z["rank_pmf"] - z["t_Q6"] + z["extra"]

    def pick(z, pen, mask=None):
        sc = base_fn(z) - z["qw"] * pen / z["Z"]
        if mask is not None:
            sc = np.where(mask, sc, -np.inf)
        return int(np.argmax(sc))

    def summarize(name, d_by_case):
        pc = np.array(list(d_by_case.values()))
        bs = np.array([rng.choice(pc, len(pc)).mean() for _ in range(4000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return f"{name:42s} {pc.mean():+7.1f} [{lo:+6.1f}, {hi:+6.1f}]  better/worse {(pc < -KCAL).sum():2d}/{(pc > KCAL).sum():2d}"

    print("\n== search vs ranking (kJ/mol, MACE float64; per complex mean over seeds) ==")
    rows = {}
    for z in pools:
        w, p = z["arm"] == 0, z["arm"] == 1
        V = variants({k[3:]: v for k, v in z.items() if k.startswith("q6_")})
        e = z["mace"]
        r = dict(floorW=e[w].min(), floorP=e[p].min(), floorU=e.min(),
                 W_on_W=e[pick(z, V["W  walls (adopted)"], w)], P_on_P=e[pick(z, V["P  PMF as run (hbond wells)"], p)])
        for k, v in V.items():
            r[k] = e[pick(z, v)]
        rows[z["pool"]] = r
    case = lambda pool: pool.split("|")[0] + "|" + pool.split("|")[1].rsplit("_s", 1)[0]
    import collections

    def per_case(fn):
        acc = collections.defaultdict(list)
        for pool, r in rows.items():
            acc[case(pool)].append(fn(r))
        return {k: float(np.mean(v)) for k, v in acc.items()}
    print(summarize("arm result: P#1 (own pool) - W#1 (own pool)", per_case(lambda r: r["P_on_P"] - r["W_on_W"])))
    print(summarize("search: floor of P pool - floor of W pool", per_case(lambda r: r["floorP"] - r["floorW"])))
    print(summarize("ranking: P vs W on the union pool", per_case(lambda r: r["P  PMF as run (hbond wells)"] - r["W  walls (adopted)"])))
    print("\n== Q6 variants re-ranking the union pool, #1 vs the walls' #1 (negative = better) ==")
    for k in variants({k[3:]: v for k, v in pools[0].items() if k.startswith("q6_")}):
        print(summarize(k, per_case(lambda r, k=k: r[k] - r["W  walls (adopted)"])))

    print("\n== accounting: PMF arm #1 vs wall arm #1, pools where the PMF pick is > 20 kJ/mol worse ==")
    print("   (differences b1 - a1 of log p in raw nats; positive = the term prefers the PMF pick)")
    acc = collections.defaultdict(list)
    n = 0
    for z in pools:
        w, p = z["arm"] == 0, z["arm"] == 1
        Z = z["Z"]
        sw = z["stored_rank"] + z["extra"]
        a1 = np.flatnonzero(w)[np.argmax(sw[w])]
        b1 = np.flatnonzero(p)[np.argmax(sw[p])]
        if z["mace"][b1] - z["mace"][a1] <= 20:
            continue
        n += 1
        acc["MACE dE (kJ/mol)"].append(z["mace"][b1] - z["mace"][a1])
        for k in z:
            if k.startswith("t_") and k != "t_Q6":
                acc[f"term {k[2:]}"].append((z[k][b1] - z[k][a1]) * Z)
        acc["bend prior / ring (extra)"].append((z["extra"][b1] - z["extra"][a1]) * Z)
        for k in z:
            if k.startswith("q6_") and (k[3:5] in ("Pc", "Pd", "W_") or k.startswith("q6_hp_")):
                acc[f"-{k[3:]}"].append(-(z[k][b1] - z[k][a1]))
        acc["polar-H pairs < 2.2 A (count)"].append(z["q6_n_hp_lt22"][b1] - z["q6_n_hp_lt22"][a1])
    print(f"   {n} pools")
    for k, v in acc.items():
        v = np.array(v)
        print(f"   {k:34s} mean {v.mean():+8.2f}  median {np.median(v):+8.2f}  share > 0 {np.mean(v > 0):.2f}")


def cmd_pairs(a):
    """The pairs behind a PMF loss: wall arm #1 (a1) vs PMF arm #1 (b1) of each named complex / seed, the pairs whose
    wall or PMF term differs most between the two; xyz of both to data/inspect/<code>/q6pmf_*."""
    setup()
    import torch
    from tools.jdev.cases import get_case

    st = load_stats()
    insp = OUT.parents[0] / "inspect"
    for s in a.sets:
        names = dict(set_cases(s))
        for code in a.cases:
            if code not in names:
                continue
            for seed in a.seeds:
                fw, fp = (pool_path(arm, s, code, seed) for arm in (ARM_W, ARM_P))
                zw, zp = np.load(fw), np.load(fp)
                ew, ep = (pool_energy(f, "mace64") for f in (fw, fp))
                a1, b1 = int(np.argmax(zw["score"])), int(np.argmax(zp["score"]))
                r = get_case(names[code]).make_builder().solve(seed=seed, stats=st, angle_bends=True, n_conformers=1,
                                                               n_ik_steps=2, device="cpu")
                sc, mol = r.scorer, r.conformers[0].complex
                X = torch.tensor(np.stack([zw["X"][a1], zp["X"][b1]]), dtype=torch.float32)
                with torch.no_grad():
                    pa = pair_arrays(sc, mol, X)
                atoms = list(mol.atoms)
                lab = [f"{at.symbol}{i}" for i, at in enumerate(atoms)]
                ia, ib = sc._ge._q6_a.tolist(), sc._ge._q6_b.tolist()
                keys = sc._ge._q6_elem_keys
                catn = np.empty(len(ia), dtype=object)
                for c, m in pa["cat"].items():
                    catn[m.numpy()] = c
                d, wl, ef = (pa[k].numpy() for k in ("d", "wall", "eff"))
                Z = float(sc._agg_norm())
                print(f"\n#### {s}/{code} s{seed}: wall #1 dE {ew[a1]:.0f}, PMF #1 dE {ep[b1]:.0f} kJ/mol; Z {Z:.1f}; "
                      f"sum wall {wl[0].sum():.1f} -> {wl[1].sum():.1f} kT, sum PMF {ef[0].sum():.1f} -> {ef[1].sum():.1f} kT")
                dw, dp = wl[1] - wl[0], ef[1] - ef[0]
                order = np.argsort(-(np.abs(dw) + np.abs(dp)))[: a.top]
                print(f"  {'pair':14s} {'key':12s} cat  r0    d(a1)  d(b1)  wall a1 -> b1     PMF a1 -> b1")
                for k in order:
                    print(f"  {lab[ia[k]] + '-' + lab[ib[k]]:14s} {'_'.join(keys[k]):12s} {catn[k]:3s} "
                          f"{float(pa['r0'][k]):.2f}  {d[0, k]:.2f}   {d[1, k]:.2f}   {wl[0, k]:6.2f} -> {wl[1, k]:6.2f}   "
                          f"{ef[0, k]:6.2f} -> {ef[1, k]:6.2f}")
                for c in CATS:
                    m = pa["cat"][c].numpy()
                    print(f"  cat {c:3s}: n {m.sum():4d}  wall {wl[0, m].sum():7.1f} -> {wl[1, m].sum():7.1f}   "
                          f"PMF {ef[0, m].sum():7.1f} -> {ef[1, m].sum():7.1f}")
                od = insp / code
                od.mkdir(parents=True, exist_ok=True)
                sy = [str(x) for x in zw["symbols"]]
                for tag, Xi, e in (("wall1", zw["X"][a1], ew[a1]), ("pmf1", zp["X"][b1], ep[b1])):
                    with open(od / f"q6pmf_s{seed}_{tag}.xyz", "w") as fh:
                        fh.write(f"{len(sy)}\n{code} s{seed} {tag} MACE dE {e:.1f} kJ/mol\n")
                        fh.writelines(f"{x} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n" for x, p in zip(sy, Xi))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["terms", "report", "pairs"])
    p.add_argument("--sets", nargs="+", default=["dev32", "n48"])
    p.add_argument("--cases", nargs="*", default=[])
    p.add_argument("--seeds", nargs="*", type=int, default=[11, 12, 13])
    p.add_argument("--top", type=int, default=15)
    a = p.parse_args()
    {"terms": cmd_terms, "report": cmd_report, "pairs": cmd_pairs}[a.stage](a)


if __name__ == "__main__":
    main()
