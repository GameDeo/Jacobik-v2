"""Trope suite runner: one parent per subprocess (CPU solver, fixed seeds).

    python -m tools.runkit.trope.suite parent --code C --arm A --flags JSON --seeds 11,12,13 --out DIR      (one parent, one arm)
    python -m tools.runkit.trope.suite run --arm A --flags JSON --footprint FOOT.json --out DIR [--only C,D]  (whole suite, 2 processes)
    python -m tools.runkit.trope.suite compare --control DIR --arm DIR --footprint FOOT.json --out report.json

Per (parent, seed) the record has the unrelaxed results (score, dE_unrel = MACE(row) - MACE(reference), kJ/mol; RMSD to the reference) and
SHA-256 digests of the whole solver output (all scores, all geometries) used for the bit-identity check outside the footprint. The solver runs
on CPU through builder.solve (the golden check's call; no CUDA graph, no multiseed).

Real strain, per isomer bucket (measurement only; the result rows stay unrelaxed): the buckets are the solver's own donor-to-site assignments
(ConformerResult.assignment_idx, with embed_idx for E/Z variants); nothing is captured or changed in the solver. For the top-1 row and
the best-scoring row of each bucket (best 8 buckets, the number that existed is recorded) the row is relaxed with GFN2 `--opt crude`
(cycle cap 200, runkit.xtb). strain = E_MACE(row) - E_MACE(relaxed); offset = E_MACE(relaxed) - E_MACE(reference). Relaxation outputs are
derived data and are not in the digests. `md_bond_dev` is the old bond-length deviation proxy, kept only under this name: bends and torsions are
rigid motions for it, so it must not be read as strain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

MAX_BUCKETS = 8
RELAX_CYCLES = 200
CONTROL_OVERRIDES = {"nci": False, "deterministic_embed": True}
K_MD, K_XH, K_HV = 1000.0, 3000.0, 2500.0


def md_bond_dev(sym, ref, X, metal, adj, donors) -> float:
    iu = np.argwhere(np.triu(adj))
    e = 0.0
    for i, j in iu:
        k = K_XH if "H" in (sym[i], sym[j]) else K_HV
        e += 0.5 * k * (np.linalg.norm(X[i] - X[j]) - np.linalg.norm(ref[i] - ref[j])) ** 2
    for d in donors:
        e += 0.5 * K_MD * (np.linalg.norm(X[d] - X[metal]) - np.linalg.norm(ref[d] - ref[metal])) ** 2
    return float(e)


def _digest(arrs) -> str:
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"{type(obj).__name__} is not JSON serializable")


def run_parent(code: str, arm: str, flags: dict, seeds, out: Path, *, unrelaxed_only=False, mace_device="cpu", solve_steps=200) -> dict:
    """In-process (call from a fresh interpreter): env first, then jacobik."""
    from .. import resources
    os.environ.update(resources.env(threads=1, cuda=(mace_device != "cpu")))
    os.environ["JACOBIK_NUM_WORKERS"] = "1"
    os.environ["JACOBIK_PROFILE"] = "u1"
    os.environ["JACOBIK_PROFILE_OVERRIDES"] = json.dumps({**CONTROL_OVERRIDES, **flags})
    if mace_device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    t0 = time.perf_counter()
    rec = dict(code=code, arm=arm, flags=flags, seeds=list(seeds), host=platform.node(), status="failed", reason="", per_seed={},
               unrelaxed_only=bool(unrelaxed_only), mace_device=mace_device, solve_steps=solve_steps)
    from .. import policy, xtb, geom, mace
    try:
        policy.assert_allowed([code])
        if not unrelaxed_only:
            xtb.require()
        import torch
        torch.set_num_threads(1)
        from jacobik.core import profile as P
        from tools.jgen import inputs
        from jacobik.core.env import build_builder_and_inputs
        from jacobik.builder.atom_map import build_atom_mapping
        from tools.jdev.budget.common import stats_from_db
        rec["fingerprint"] = P.snapshot()["fingerprint"]
        shard, ds = inputs.frame_shards()[code]
        mol = inputs.load_mol(code, shard, ds)
        if mol is None:
            raise RuntimeError("no geometry")
        builder, _, err = build_builder_and_inputs(mol)
        if builder is None:
            raise RuntimeError(f"unmappable: {err}")
        ref_sym = [a.symbol for a in mol.atoms]
        ref_X = np.array([a.position for a in mol.atoms], float)
        with xtb_watch() as fw:
            t1 = time.perf_counter()
            st = stats_from_db(inputs.DB_PATH)
            res = {}
            for s in seeds:       # production CPU path, one seed per call (the golden check's call)
                try:
                    res[s] = builder.solve(seed=s, stats=st, angle_bends=True, device="cpu", n_ik_steps=solve_steps)
                except Exception as exc:
                    rec["per_seed"][str(s)] = dict(status="solve_failed", reason=f"{type(exc).__name__}: {str(exc)[:150]}")
            rec["solve_s"] = time.perf_counter() - t1
        rec["embed_fallback"] = fw.hit
        am = None
        for s in seeds:
            if res.get(s) and res[s].conformers:
                c0 = res[s].conformers[0]
                am = build_atom_mapping(mol, gen_mol=c0.complex, X_ref=ref_X,
                                        X_gen=np.array([a.position for a in c0.complex.atoms], float))
                break
        if am is None:
            raise RuntimeError("atom mapping failed")
        g2r = np.asarray(am.gen_to_ref_idx, int)
        metal = geom.find_metal(ref_sym)
        adj = geom.ligand_adjacency(ref_X, ref_sym, metal)
        donors = geom.donors_of(ref_X, ref_sym, metal)
        heavy = [i for i, s in enumerate(ref_sym) if s != "H"]
        P_aut, _ = geom.automorphisms(ref_X, ref_sym, metal)
        ev = mace.Evaluator(mace_device)
        wr = resources.workroot(out)
        sel = {}                                               # seed -> list of selected buckets
        for s in seeds:
            r = res.get(s)
            if str(s) in rec["per_seed"]:
                continue                                       # solve failed, already recorded
            if not r or not r.conformers:
                rec["per_seed"][str(s)] = dict(status="empty")
                continue
            sc = np.array([c.score for c in r.conformers], float)
            top = int(np.argmax(sc))
            Xs = [np.array([a.position for a in c.complex.atoms], float)[g2r] for c in r.conformers]
            best = {}                                          # bucket key -> index of its best-scoring row
            for k, c in enumerate(r.conformers):
                key = (int(c.embed_idx or 0), int(c.assignment_idx if c.assignment_idx is not None else -1))
                if key not in best or sc[k] > sc[best[key]]:
                    best[key] = k
            order = sorted(best.items(), key=lambda kv: -sc[kv[1]])
            sel[s] = [(key, k) for key, k in order[:MAX_BUCKETS]]
            assert sel[s][0][1] == top                         # the top-1 row is the best row of its bucket
            rec["per_seed"][str(s)] = dict(status="ok", n_conf=len(sc), top1_score=float(sc[top]), scores_digest=_digest([sc]),
                                           geom_digest=_digest(Xs), top1_digest=_digest([Xs[top]]), n_buckets=len(best),
                                           n_buckets_kept=len(sel[s]), buckets=[])
            for key, k in sel[s]:
                d = dict(embed_idx=key[0], assignment_idx=key[1], row=int(k), score=float(sc[k]), is_top1=(k == top))
                if unrelaxed_only:
                    d.update(relax_status="deferred", relax_cycles=0, relax_wall_s=0.0, relax_stderr="")
                    d["_X"], d["_Xr"] = Xs[k], None
                else:
                    relax = xtb.opt(ref_sym, Xs[k], mol.charge, mol.spin_state, wr, cycles=RELAX_CYCLES)
                    d.update(relax_status=relax["status"], relax_cycles=relax["cycles"], relax_wall_s=relax["wall_s"],
                             relax_stderr=("" if relax["status"] in ("ok", "not_converged") else relax["stderr_tail"][-200:]))
                    d["_X"], d["_Xr"] = Xs[k], relax["X"]
                d["md_bond_dev"] = md_bond_dev(ref_sym, ref_X, Xs[k], metal, adj, donors)
                d["rmsd_ref"] = float(geom.sym_rmsd_to_ref(Xs[k][None, heavy], ref_X[heavy], P_aut)[0])
                rec["per_seed"][str(s)]["buckets"].append(d)
        import warnings
        structs = [(ref_sym, ref_X, mol.charge, mol.spin_state)]
        offsets = {}
        if unrelaxed_only:
            for s in sel:
                ps = rec["per_seed"][str(s)]
                # Save only top-1 coordinates for the final, explicitly limited relaxation pass.
                Xs = [np.array([a.position for a in c.complex.atoms], float)[g2r] for c in res[s].conformers]
                top = int(np.argmax([c.score for c in res[s].conformers]))
                ps.update(top1_X=Xs[top].tolist(), atom_symbols=ref_sym, charge=int(mol.charge), spin_state=int(mol.spin_state))
                offsets[s] = len(structs)
                structs.extend((ref_sym, X, mol.charge, mol.spin_state) for X in Xs)
        else:
            for s, bl in sel.items():
                for d in rec["per_seed"][str(s)]["buckets"]:
                    structs.append((ref_sym, d["_X"], mol.charge, mol.spin_state))
                    if d["_Xr"] is not None:
                        structs.append((ref_sym, d["_Xr"], mol.charge, mol.spin_state))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if mace_device == "cuda":
                with resources.gpu_lock("CodexB", timeout=900, check_smi=False):
                    E = ev.energies_kj(structs)
                    del ev
                    import torch
                    torch.cuda.empty_cache()
            else:
                E = ev.energies_kj(structs)
        if unrelaxed_only:
            for s, off in offsets.items():
                ps = rec["per_seed"][str(s)]
                all_de = np.asarray(E[off:off + ps["n_conf"]]) - E[0]
                ps.update(all_dE_unrel=all_de.tolist(), worst_dE_unrel=float(np.max(all_de)),
                          obscene_count=int(np.sum(all_de > 200.0)), obscene_rate=float(np.mean(all_de > 200.0)),
                          dE_unrel=float(all_de[int(np.argmax([c.score for c in res[s].conformers]))]))
                for d in ps["buckets"]:
                    d["dE_unrel"] = float(all_de[d["row"]])
                    d["strain"] = d["offset"] = None
                    del d["_X"], d["_Xr"]
                t = ps["buckets"][0]
                ps.update({k: t[k] for k in ("strain", "offset", "md_bond_dev", "rmsd_ref", "relax_status")})
            rec["ref_energy_kj"] = float(E[0])
            rec["n_atoms"] = len(ref_sym)
            rec["status"] = "ok" if any(v.get("status") == "ok" for v in rec["per_seed"].values()) else "failed"
            if rec["embed_fallback"]:
                rec["status"], rec["reason"] = "invalid", f"{rec['embed_fallback']} embeds fell back to MMFF"
            rec["wall_s"] = time.perf_counter() - t0
            rec["manifest"] = resources.manifest(torch_info=False)
            out.mkdir(parents=True, exist_ok=True)
            (out / f"{code}.json").write_text(json.dumps(rec, indent=1, default=_json_default))
            return rec
        i = 1
        for s in sel:
            ps = rec["per_seed"][str(s)]
            for d in ps["buckets"]:
                d["dE_unrel"] = float(E[i] - E[0])
                i += 1
                if d["_Xr"] is not None:
                    d["strain"] = float(E[i - 1] - E[i])
                    d["offset"] = float(E[i] - E[0])
                    i += 1
                else:
                    d["strain"] = d["offset"] = None
                del d["_X"], d["_Xr"]
            t = ps["buckets"][0]
            ps.update({k: t[k] for k in ("dE_unrel", "strain", "offset", "md_bond_dev", "rmsd_ref", "relax_status")})
        rec["ref_energy_kj"] = float(E[0])
        rec["n_atoms"] = len(ref_sym)
        rec["status"] = "ok" if any(v.get("status") == "ok" for v in rec["per_seed"].values()) else "failed"
        if rec["embed_fallback"]:
            rec["status"], rec["reason"] = "invalid", f"{rec['embed_fallback']} embeds fell back to MMFF"
    except BaseException as exc:
        rec["reason"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        rec["tb"] = traceback.format_exc()[-1500:]
    rec["wall_s"] = time.perf_counter() - t0
    rec["manifest"] = resources.manifest(torch_info=False)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{code}.json").write_text(json.dumps(rec, indent=1, default=_json_default))
    return rec


def xtb_watch():
    from .. import xtb
    return xtb.FallbackWatch()


# ---------------------------------------------------------------- orchestration and comparison

def seeds_for(code: str, tags, foot: dict, full=(11, 12, 13), single=(11,)):
    """Seed ensembles only inside the footprint (declared parents or parents carrying a declared tag)."""
    inside = code in foot.get("parents", []) or bool(set(tags) & set(foot.get("tags", [])))
    return (full if inside else single), inside


def load_suite(path=None):
    p = Path(path) if path else Path(__file__).resolve().parents[4].parent / "research/archive/code/trope_suite/suite.json"
    return json.loads(p.read_text())


def run_suite(arm, flags, foot, out: Path, only=None, procs: int = 2, suite=None, agent="runkit",
              unrelaxed_only=False, mace_device="cpu", seeds_override=None, solve_steps=200):
    from .. import resources
    suite = suite or load_suite()
    budget = resources.Budget(agent, cpus=procs)
    todo = [p for p in suite["parents"] if not only or p["code"] in only]
    running = []
    for p in todo:
        seeds, inside = seeds_for(p["code"], p["all_tags"], foot)
        if seeds_override:
            seeds = tuple(seeds_override)
        if (out / f"{p['code']}.json").exists():
            continue
        while len(budget.running()) >= procs:
            time.sleep(2)
        cmd = [sys.executable, "-m", "tools.runkit.trope.suite", "parent", "--code", p["code"], "--arm", arm, "--flags",
               json.dumps(flags), "--seeds", ",".join(map(str, seeds)), "--out", str(out)]
        if unrelaxed_only:
            cmd.append("--unrelaxed-only")
        cmd += ["--mace-device", mace_device, "--solve-steps", str(solve_steps)]
        child_env = resources.env(threads=1, cuda=(mace_device != "cpu"))
        tag = f"trope-{arm}-{p['code']}"
        if mace_device == "cuda":
            # The solver remains on CPU, but the child must see the GPU for its
            # later, separately locked MACE energy batch.
            with resources.gpu_lock(agent, timeout=900, check_smi=False):
                pr = resources.spawn(cmd, budget, tag, log=out / "logs" / f"{p['code']}.log",
                                     cpus=1, env=child_env, gpu=True)
        else:
            pr = resources.spawn(cmd, budget, tag, log=out / "logs" / f"{p['code']}.log",
                                 cpus=1, env=child_env, gpu=False)
        running.append(pr)
    for pr in running:
        pr.wait()


def _mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None


def parent_summary(rec):
    ok = [d for d in rec["per_seed"].values() if d.get("status") == "ok"]
    return dict(score=_mean([d["top1_score"] for d in ok]), dE_unrel=_mean([d["dE_unrel"] for d in ok]),
                strain=_mean([d.get("strain") for d in ok]), offset=_mean([d.get("offset") for d in ok]),
                md_bond_dev=_mean([d.get("md_bond_dev") for d in ok]), rmsd=_mean([d["rmsd_ref"] for d in ok]), n_seeds=len(ok),
                n_buckets=_mean([d["n_buckets"] for d in ok]))


def bucket_deltas(rc, ra):
    """Per-bucket arm - control for buckets present in both (matched by seed and (embed_idx, assignment_idx))."""
    out, unmatched = [], 0
    for s, dc in rc["per_seed"].items():
        da = ra["per_seed"].get(s)
        if dc.get("status") != "ok" or not da or da.get("status") != "ok":
            continue
        ac = {(b["embed_idx"], b["assignment_idx"]): b for b in da["buckets"]}
        for b in dc["buckets"]:
            o = ac.get((b["embed_idx"], b["assignment_idx"]))
            if o is None:
                unmatched += 1
                continue
            out.append(dict(seed=int(s), bucket=[b["embed_idx"], b["assignment_idx"]], is_top1=b["is_top1"],
                            d_score=o["score"] - b["score"], d_dE_unrel=o["dE_unrel"] - b["dE_unrel"],
                            d_strain=None if None in (o["strain"], b["strain"]) else o["strain"] - b["strain"],
                            d_offset=None if None in (o["offset"], b["offset"]) else o["offset"] - b["offset"]))
    return out, unmatched


def compare(ctrl_dir: Path, arm_dir: Path, foot: dict, suite=None) -> dict:
    """Outside the footprint the digests must be equal (bit-identical); inside, per-parent and per-trope deltas arm - control."""
    suite = suite or load_suite()
    rows, violations = [], []
    for p in suite["parents"]:
        c = p["code"]
        fc, fa = ctrl_dir / f"{c}.json", arm_dir / f"{c}.json"
        if not fc.exists() or not fa.exists():
            rows.append(dict(code=c, status="missing")); continue
        rc, ra = json.loads(fc.read_text()), json.loads(fa.read_text())
        if rc["host"] != ra["host"] or (rc.get("fingerprint") != ra.get("fingerprint") and not foot.get("fingerprint_change_ok")):
            rows.append(dict(code=c, status="unpaired", note="different host or fingerprint")); continue
        if rc["status"] != "ok" or ra["status"] != "ok":
            rows.append(dict(code=c, status="failed", control=rc["status"] + rc["reason"], arm=ra["status"] + ra["reason"])); continue
        seeds, inside = seeds_for(c, p["all_tags"], foot)
        ident = all(rc["per_seed"][s].get("geom_digest") == ra["per_seed"].get(s, {}).get("geom_digest") and
                    rc["per_seed"][s].get("scores_digest") == ra["per_seed"].get(s, {}).get("scores_digest") for s in rc["per_seed"])
        sc, sa = parent_summary(rc), parent_summary(ra)
        bd, unmatched = bucket_deltas(rc, ra)
        dd = lambda k: None if sa[k] is None or sc[k] is None else sa[k] - sc[k]
        row = dict(code=c, inside=inside, bit_identical=ident, n_seeds=sa["n_seeds"], tags=p["all_tags"],
                   d_score=dd("score"), d_dE_unrel=dd("dE_unrel"), d_strain=dd("strain"), d_offset=dd("offset"),
                   d_md_bond_dev=dd("md_bond_dev"), n_buckets_control=sc["n_buckets"], n_buckets_arm=sa["n_buckets"],
                   buckets=bd, buckets_unmatched=unmatched,
                   d_bucket_strain=_mean([b["d_strain"] for b in bd]), d_bucket_offset=_mean([b["d_offset"] for b in bd]),
                   d_bucket_dE_unrel=_mean([b["d_dE_unrel"] for b in bd]), control=sc, arm=sa)
        if not inside and not ident:
            violations.append(c)
            row["status"] = "VIOLATION (outside footprint, not bit-identical)"
        else:
            row["status"] = "ok"
        rows.append(row)
    per_tag = {}
    for r in rows:
        if r.get("inside") and r["status"] == "ok":
            for t in r["tags"]:
                if t in foot.get("tags", []) or r["code"] in foot.get("parents", []):
                    per_tag.setdefault(t, []).append(r)
    keys = ("d_score", "d_dE_unrel", "d_strain", "d_offset", "d_bucket_dE_unrel", "d_bucket_strain", "d_bucket_offset")
    tag_tab = {t: dict(parents=[r["code"] for r in rs], **{k: _mean([r[k] for r in rs]) for k in keys}) for t, rs in sorted(per_tag.items())}
    return dict(footprint=foot, rows=rows, per_trope=tag_tab, outside_violations=violations,
                outside_ok=not violations, note="a filter before panels; the panels decide. <=12 parents: no CI claimed.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("parent")
    a.add_argument("--code", required=True); a.add_argument("--arm", required=True); a.add_argument("--flags", default="{}")
    a.add_argument("--seeds", default="11"); a.add_argument("--out", required=True)
    a.add_argument("--unrelaxed-only", "--no-strain", action="store_true"); a.add_argument("--mace-device", choices=("cpu", "cuda"), default="cpu")
    r = sp.add_parser("run")
    r.add_argument("--arm", required=True); r.add_argument("--flags", default="{}"); r.add_argument("--footprint", required=True)
    r.add_argument("--out", required=True); r.add_argument("--only", default=""); r.add_argument("--procs", type=int, default=2)
    r.add_argument("--unrelaxed-only", "--no-strain", action="store_true"); r.add_argument("--mace-device", choices=("cpu", "cuda"), default="cpu")
    r.add_argument("--seeds", default="")
    r.add_argument("--agent", default="CodexB")
    c = sp.add_parser("compare")
    c.add_argument("--control", required=True); c.add_argument("--arm", required=True); c.add_argument("--footprint", required=True)
    c.add_argument("--out", required=True)
    a.add_argument("--solve-steps", type=int, default=200)
    r.add_argument("--solve-steps", type=int, default=200)
    n = ap.parse_args(argv)
    if n.cmd == "parent":
        rec = run_parent(n.code, n.arm, json.loads(n.flags), [int(s) for s in n.seeds.split(",")], Path(n.out),
                         unrelaxed_only=n.unrelaxed_only, mace_device=n.mace_device, solve_steps=n.solve_steps)
        print(n.code, rec["status"], rec["reason"], round(rec["wall_s"], 1), "s", flush=True)
        return 0 if rec["status"] == "ok" else 1
    if n.cmd == "run":
        run_suite(n.arm, json.loads(n.flags), json.loads(Path(n.footprint).read_text()), Path(n.out),
                  {x for x in n.only.split(",") if x} or None, n.procs,
                  agent=n.agent,
                  unrelaxed_only=n.unrelaxed_only, mace_device=n.mace_device,
                  seeds_override=[int(s) for s in n.seeds.split(",") if s] or None, solve_steps=n.solve_steps)
        return 0
    rep = compare(Path(n.control), Path(n.arm), json.loads(Path(n.footprint).read_text()))
    Path(n.out).write_text(json.dumps(rep, indent=1))
    print("outside footprint bit-identical:", rep["outside_ok"], "| violations:", rep["outside_violations"])
    for t, v in rep["per_trope"].items():
        f = lambda x: "   n/a" if x is None else f"{x:+.2f}"
        print(f"{t:20s} n={len(v['parents'])} top1: d_score={v['d_score']:+.4f} d_dE_unrel={f(v['d_dE_unrel'])} d_strain={f(v['d_strain'])} "
              f"d_offset={f(v['d_offset'])} | buckets: d_dE={f(v['d_bucket_dE_unrel'])} d_strain={f(v['d_bucket_strain'])} d_offset={f(v['d_bucket_offset'])}")
    return 0 if rep["outside_ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
