"""Energy-scale study, stage 1: solve pools and record the production ranking score per term.

Run from the repo root with the pixi env and MKL_THREADING_LAYER=SEQUENTIAL:
    python <this> solve --sets n48 heldout28 ring10 dev32 --seeds 11 12 13 --start etkdg
    python <this> solve --sets n48 heldout28 --seeds 11 --start full_oracle
    python <this> solve --sets dev32 n48 --seeds 11 12 13 --isomer ref     (reference isomer only -> <start>_isoref)
    python <this> mace [--dtype float64] [--dirs etkdg_u1_lean ...]   (float64 -> <code>_s<seed>.mace64.json)
Output: <OUT>/pools/<start>/<set>/<code>_s<seed>.npz with
    X (n,N,3), symbols, score (final, as ranked), asg (assignment idx), terms_<Q> (summed nats, ranking settings),
    rank (ranking_score), Z (per-complex constant), n_<Q> (feature counts), q7_raw
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent))
from common import DB, OUT, REPO, stats_from_db  # noqa: E402,F401  (re-exported: golden.py, gate.py, ... import them from here)

# DB (common.py, JACOBIK_BUDGET_DB): every stage reading these pools (gate, analyze, coord_audit) must see the same DB,
# so the solve stage records it in <OUT>/pools_db.txt and refuses another.
DEV = REPO / "research/bench/tools/q6_ablation/dev_tmqm" if (REPO / "research/bench/tools/q6_ablation/dev_tmqm").exists() else REPO / "tools/q6_ablation/dev_tmqm"


def setup():
    this_repo = Path(__file__).resolve().parents[4]
    for p in (this_repo / "src", this_repo):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import logging, warnings
    logging.basicConfig(level=logging.ERROR)
    warnings.filterwarnings("ignore")
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")


def set_cases(name):
    import re
    from tools.jdev import bench
    if name == "dev32":
        codes = re.findall(r"CSD_code = (\w+)", (DEV / "xyz" / "tmQM_D.xyz").read_text())
        return [(c, f"tmqm:{DEV}:D:{c}") for c in codes]
    d, split, _ = bench.TMQM_SETS[name]
    return [(c, f"tmqm:{d}:{split}:{c}") for c in bench.case_list(name)]


def db_compatible(recorded: Path) -> bool:
    """Only the recorded DB is accepted; verified migrations retain a separate provenance report."""
    rec_str = str(recorded).replace("\\", "/")
    return Path(rec_str).name == DB.name or Path(rec_str).resolve() == DB


def check_db():
    """Refuse pools from another DB (see pools_db.txt and any explicit migration report beside it)."""
    rec = OUT / "pools_db.txt"
    if rec.exists() and not db_compatible(Path(rec.read_text().strip())):
        raise SystemExit(f"{OUT} holds pools of {rec.read_text().strip()}; set JACOBIK_BUDGET_DB to it (now {DB})")


def load_stats():
    check_db()
    return stats_from_db(DB)


def rank_terms(sc, X, rg):
    """Per-term values of the production ranking score (scorer.ranking_score), as summed nats (x Z)."""
    import torch
    import jacobik.ik.scorer as SC
    old_agg, old_q1c = SC.SCORE_AGG, getattr(sc, "_q1_cand", None)
    try:
        if SC.RANK_AGG == "sum":
            SC.SCORE_AGG = "sum"
        import jacobik.ik.ik_solver as IK
        sc.set_gmm_floor("fit" if IK._RANK_FLOOR else SC.GMM_FLOOR_MODE)  # as scorer.ranking_score
        q1c = getattr(sc, "_q1_cand", None)
        if SC.Q1_TAIL_SIGMA > 0 and q1c is not None and getattr(sc, "_q1_cand_var", None) is not None:
            mu, ev, lw = q1c
            real = lw > -1e3
            keep = real & ~((sc._q1_cand_var.sqrt() > SC.Q1_TAIL_SIGMA) | (mu < 1.2))
            keep = torch.where(keep.any(-1, keepdim=True), keep, real)
            lw2 = torch.where(keep, lw, torch.full_like(lw, -1e30))
            sc._q1_cand = (mu, ev, lw2 - torch.logsumexp(lw2, -1, keepdim=True))
        with torch.no_grad():
            tot, bd = sc.score(X, row_groups=rg)
    finally:
        SC.SCORE_AGG = old_agg
        sc.set_gmm_floor(SC.GMM_FLOOR_MODE)
        if old_q1c is not None:
            sc._q1_cand = (old_q1c[0], sc._q1_cand[1], old_q1c[2])
    return tot, bd


def cmd_solve(a):
    setup()
    import numpy as np
    import torch
    from tools.jdev.cases import get_case
    from tools.jdev.start import starting_geometry
    OUT.mkdir(parents=True, exist_ok=True)
    rec = OUT / "pools_db.txt"
    if rec.exists() and not db_compatible(Path(rec.read_text().strip())):
        raise SystemExit(f"{OUT} holds pools of {rec.read_text().strip()}, not {DB}: set JACOBIK_BUDGET_OUT")
    if not rec.exists():
        rec.write_text(str(DB))
    st = load_stats()
    for set_name in a.sets:
        cases = set_cases(set_name)
        if a.cases:
            cases = [(c, n) for c, n in cases if c in set(a.cases)]
        if a.limit:
            cases = cases[: a.limit]
        tag = a.start + ("_isoref" if a.isomer == "ref" else "") + {"db": "", "crystal": "_crystgt", "reach": "_reach"}[a.targets]
        from jacobik.core import profile as _P
        if _P.ACTIVE_NAME != _P.DEFAULT_PROFILE or _P.OVERRIDES:  # another configuration: its own pools
            tag += f"_{_P.ACTIVE_NAME}" + (f"_{a.tag}" if a.tag else "")
        od = OUT / "pools" / tag / set_name
        od.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        todo = [(c, n, s) for c, n in cases for s in a.seeds if not (od / f"{c}_s{s}.npz").exists()]
        for k, (code, name, seed) in enumerate(todo, 1):
            f = od / f"{code}_s{seed}.npz"
            try:
                case = get_case(name)
                make, iso_dev = case.make_builder, None
                if a.isomer == "ref":  # the reference's isomer only (ARCHITECTURE.md, isomer-matched track)
                    from tools.jdev.budget.isomer import isomer_builder
                    make, iso_dev = isomer_builder(case)
                    if make is None:
                        raise LookupError(f"no reference isomer: {iso_dev}")
                ctx = contextlib.nullcontext()
                if a.targets in ("crystal", "reach"):  # DIAGNOSTICS (oracle_targets.py)
                    from oracle_targets import crystal_in_engine_order, crystal_targets, reach_target
                    with starting_geometry(case, a.start):
                        pre = make().solve(seed=seed, stats=st, angle_bends=True, n_conformers=1, n_ik_steps=2,
                                           device="cpu")
                    mol = pre.conformers[0].complex
                    pf = OUT / "pools" / "etkdg" / set_name / f"{code}_s{seed}.npz"
                    zp = np.load(pf, allow_pickle=False) if pf.exists() else None
                    if zp is not None and "error" not in zp.files:
                        Xp = zp["X"][int(np.argmax(zp["score"]))]
                    else:
                        Xp = np.array([at.position for at in mol.atoms])
                    Xc, mapped = crystal_in_engine_order(case, mol, Xp)
                    ctx = crystal_targets(Xc, mapped) if a.targets == "crystal" else reach_target(Xc, mapped)
                t = time.time()
                with starting_geometry(case, a.start), ctx:
                    r = make().solve(seed=seed, stats=st, angle_bends=True, device=a.device)
                ts = time.time() - t
                confs = r.conformers
                X = np.array([[at.position for at in c.complex.atoms] for c in confs], dtype=np.float64)
                sy = np.array([at.symbol for at in confs[0].complex.atoms])
                asg = np.array([c.assignment_idx or 0 for c in confs])
                sc = r.scorer
                Xt = torch.tensor(X, dtype=torch.float32, device=next(iter([sc._q6_mu])).device if hasattr(sc, "_q6_mu") else "cpu")
                rg = torch.as_tensor(asg, dtype=torch.long, device=Xt.device)
                tot, bd = rank_terms(sc, Xt, rg)
                Z = float(sc._agg_norm())
                save = dict(X=X, symbols=sy, score=np.array([c.score for c in confs]), asg=asg,
                            rank=tot.cpu().numpy(), Z=Z, t_solve=ts, ref_X=case.ref_X,
                            ref_symbols=np.array(case.ref_symbols), charge=case.charge, mult=case.mult)
                if iso_dev is not None:
                    save["iso_site_dev"] = iso_dev
                from jacobik.core.profile import snapshot
                save["profile"] = json.dumps(snapshot(), default=str)  # every result-changing setting
                for q, v in bd.items():
                    v = v.detach().cpu().numpy()
                    save[f"terms_{q}"] = v * Z if q != "Q7" else v
                ge = sc._ge
                save.update(n_Q1=sc._n_q1, n_Q2=sc._n_q2, n_SO=sc._so_n, n_Q5=ge.n_q5, n_Q6=ge.n_q6)
                np.savez_compressed(f, **save)
                msg = f"{len(X)} conf, {ts:.1f}s"
            except Exception as exc:
                np.savez_compressed(f, error=f"{type(exc).__name__}: {str(exc)[:400]}")
                msg = f"ERROR {type(exc).__name__}: {str(exc)[:150]}"
                if a.verbose:
                    traceback.print_exc()
            if a.device.startswith("cuda"):
                torch.cuda.empty_cache()
            el = time.time() - t0
            print(f"[{a.start}/{set_name}] {k}/{len(todo)} {code} s{seed}: {msg} (eta {el / k * (len(todo) - k) / 60:.0f} min)",
                  flush=True)


def cmd_mace(a):
    setup()
    import numpy as np
    from jacobik.core.energy import MACEPolarEvaluator
    ev = MACEPolarEvaluator(model="polar-1-s", device=a.device, dtype=a.dtype,
                            model_path=str(REPO / "research/bench/tools/jdev/MACE-POLAR-1-S.model"))
    # float32 energies jitter by up to 5-10 kJ/mol (CHANGELOG 2026-10-16 e); float64 goes to its own file
    suffix = ".mace.json" if a.dtype == "float32" else ".mace64.json"
    roots = [OUT / "pools" / d for d in a.dirs] if a.dirs else [OUT / "pools"]
    for f in sorted(f for r in roots for f in r.rglob("*.npz")):
        ef = f.with_suffix(suffix)
        if ef.exists():
            continue
        z = np.load(f, allow_pickle=False)
        if "error" in z.files:
            continue
        sy = [str(s) for s in z["symbols"]]
        q, m = int(z["charge"]), int(z["mult"])
        jobs = [(sy, X, q, m) for X in z["X"]] + [([str(s) for s in z["ref_symbols"]], z["ref_X"], q, m)]
        E = [float(e) for e in ev.energies(jobs)]
        ref = E[-1]
        ef.write_text(json.dumps({"ref_eV": ref, "dE": [(e - ref) * 96.485 for e in E[:-1]]}))
        print(f"mace {f.parent.parent.name}/{f.parent.name}/{f.stem}: {len(E) - 1}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["solve", "mace"])
    p.add_argument("--sets", nargs="+", default=["n48"])
    p.add_argument("--seeds", nargs="+", type=int, default=[11])
    p.add_argument("--start", default="etkdg")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32", help="mace: model precision")
    p.add_argument("--dirs", nargs="*", default=None, help="mace: only these pool directories (default: all)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--cases", nargs="*", default=None, help="only these CSD codes")
    p.add_argument("--targets", choices=["db", "crystal", "reach"], default="db",
                   help="DIAGNOSTICS. crystal: every fitted curve translated onto the crystal value (-> <tag>_crystgt); "
                        "reach: the solver minimises its distance to the crystal (closest reachable pose, -> <tag>_reach)")
    p.add_argument("--isomer", choices=["free", "ref"], default="free",
                   help="ref: solve only the reference's donor->site assignment (pools/<start>_isoref/)")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--tag", default="", help="suffix of a non-default profile's pool directory (<start>_<profile>_<tag>)")
    a = p.parse_args()
    {"solve": cmd_solve, "mace": cmd_mace}[a.stage](a)


if __name__ == "__main__":
    main()
