"""
Regression panel for src: accuracy + speed on a fixed set of complexes, compared to a stored
baseline. Run at the end of every major change (about 10 min on a laptop; MACE optional):

    python tests/regression/run_panel.py                  # run + compare to baseline.json
    python tests/regression/run_panel.py --cases VUCVUN,aqua_Ce3_9
    python tests/regression/run_panel.py --update-baseline   # accept the current results

Devices and process layout (``--device`` sets both; ``--solver-device`` / ``--mace-device`` override one):

  * ``--device cpu`` (default; MACE still goes to the GPU when ``--mace-device auto`` finds one): cases are solved
    in a bounded pool of persistent loky processes (``--jobs``), each solve using ``--solver-workers`` threads.
  * MACE on CUDA: ONE model, loaded once in this (driver) process and kept on the GPU for the whole invocation.
    Solves return their structures and the driver evaluates them, so pool workers never touch the GPU and the
    model is never reloaded or moved.
  * ``--device cuda``: the solver runs on the GPU too. Cases then run one after another inside the driver process
    (``--case-executor inline``, no worker processes or solver threads): one CUDA context, shared with MACE.

Each case gets fresh builder / solver / result state and deterministic seeds; a process retains only expensive,
read-mostly infrastructure (imports, parsed constraint DBs, the MACE calculator, parsed fixture splits).
``--plan plan.json`` runs several result files (sets x DBs x configurations x seeds) in this one process, so the
pool, the DBs and the model stay warm across all of them (tools/jdev/bench.py builds the plan).
With a CUDA solver, result files of a plan that differ only in the seed are solved together: per case ONE solver
pass for all seeds (``--seeds-in-one-pass``, tools/jdev/multiseed.py), still one result file per seed; a seed's
``solve_s`` is then its share of the pass.
A hard per-case timeout kills and rebuilds the loky pool, so a hang is reported as TIMEOUT instead of stalling the
run; inline cases cannot be killed and are only flagged when they exceed it. ``--case-executor subprocess`` keeps the
old fresh-process-per-case path as a strict-isolation diagnostic (CPU only).
Results go to tests/regression/results/<timestamp>.json.

Per case it records: solve time, best score, converged count, donor count vs crystal, max |M-D error|
vs the crystal (tmQM/aqua cases), water tilt (aqua) and the MACE-POLAR-1 matched-geometry energy
difference E(best conformer) - E(reference) in kJ/mol (only if a MACE model is available: env
JACOBIK_MACE_MODEL_PATH, else script_notebooks/MACE-POLAR-1-M.model). MACE only MEASURES here.

Regression rules vs baseline (per case):
  FAIL  : error / timeout; donor count changed; max|M-D err| grew by > 0.15 A and is now > 0.3 A;
          |dE| worse by more than max(25 kJ/mol, 25 %); score worse by more than 1.0.
  WARN  : solve time > 2x baseline (only compared when run on the same machine as the baseline).
  BETTER: |dE| better by more than max(25 kJ/mol, 25 %).

Panel (fixtures/ is self-contained; nothing is read from data/ except the constraints DB):
  aqua_*  : 8 DOE Ln(H2O)8/9 3+ and Fe(H2O)6 2+/3+ systems (DFT references, never fitted on).
  tmQM    : VUCVUN (rigid pentadentate cage) + 12 random tmQM complexes whose benchmark donor count
            matches the crystal (Re/Fe/Ru/Co oct, Pt/Pd sq-planar, Cu/Zn tet, Ni sq-pyr, Co tbp).
  legacy_*: hand-built Co-EDTA (test-suite donors), La-EDTA(H2O)3 CN9 (symmetry enumeration with
            chelates), Co(en)3.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, wait
import json
import os
import platform
import re
import statistics as st
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FIX = HERE / "fixtures"
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
from jacobik.core.env import PRODUCTION_DB_NAME, main_checkout  # noqa: E402

#: the production constraints DB ($JACOBIK_DB overrides; a worktree has no data/ of its own)
DEFAULT_DB = Path(os.environ.get("JACOBIK_DB") or main_checkout(REPO) / "data" / "db" / PRODUCTION_DB_NAME)
DEFAULT_MACE = REPO / "script_notebooks" / "MACE-POLAR-1-M.model"

AQUA = ["Ce3_9H2O", "Dy3_8H2O", "Er3_8H2O", "La3_8H2O", "Eu3_9H2O", "Yb3_9H2O", "Fe2_6H2O", "Fe3_6H2O"]
TMQM = ["VUCVUN", "ALADIC", "DEFCPO", "ZENDII", "JUFNON", "HAKBUU", "TAYXOK", "GEFPIU", "NAHNAN",
        "KAQHOA", "LIXMIR", "YARHIM", "GUJCUJ"]
LEGACY = ["legacy_co_edta", "legacy_la_edta_cn9", "legacy_co_en3"]
ALL_CASES = [f"aqua_{a}" for a in AQUA] + TMQM + LEGACY
TIMEOUT_S = 600


def env_context() -> dict:
    """Per-run settings from the JDEV_* environment (the single-run interface; --plan passes them explicitly)."""
    solve_kw = dict(seed=int(os.environ.get("JDEV_SEED", "11")))
    if os.environ.get("JDEV_SOLVE"):  # extra solve() keyword arguments as JSON, e.g. '{"angle_bends": true}'
        solve_kw.update(json.loads(os.environ["JDEV_SOLVE"]))  # solve() defaults otherwise (beam 16, 200 steps, 8 coord samples, sym-enum on)
    if os.environ.get("JDEV_FAST_EMBED"):
        solve_kw["fast_embed"] = os.environ["JDEV_FAST_EMBED"] == "1"
    return {
        "solve_kw": solve_kw,
        "tmqm_dir": os.environ.get("JDEV_TMQM_DIR") or None,
        "tmqm_split": os.environ.get("JDEV_TMQM_SPLIT") or None,
        "save_xyz": os.environ.get("JDEV_SAVE_XYZ") or None,
        "ensemble": bool(os.environ.get("JDEV_ENSEMBLE_DE")),
        "q1_variant": os.environ.get("JDEV_Q1_VARIANT") or None,
        "orientation_variant": os.environ.get("JDEV_ORIENTATION_VARIANT") or None,
    }


SOLVE_KW = env_context()["solve_kw"]  # kept for callers that import it

EDTA = "[O-]C(=O)CN(CC(=O)[O-])CCN(CC(=O)[O-])CC(=O)[O-]"
SPIN = {"La": 1, "Ce": 2, "Pr": 3, "Nd": 4, "Pm": 5, "Sm": 6, "Eu": 7, "Gd": 8, "Tb": 7, "Dy": 6, "Ho": 5,
        "Er": 4, "Tm": 3, "Yb": 2, "Lu": 1, "Fe2": 5, "Fe3": 6}
GEOM = {8: "square_anti", 9: "tricap_pris", 6: "octahedral"}


# Worker-local, read-mostly resources.  A loky process handles only one case at a time, so the MACE
# ASE calculator is never used concurrently.  Solver/build/result objects are deliberately not
# cached: they are created afresh inside _worker for case isolation.
_DB_CACHE = {}
_MACE_CACHE = {}
_TMQM_SPLIT_CACHE = {}


def _stats_for(db: str, ctx: dict):
    """Fresh StatsLookup over a process-cached parsed DB."""
    sys.path.insert(0, str(REPO / "src"))
    from jacobik.stats.stats import StatsLookup, load_db

    db_key = str(Path(db).resolve())
    if db_key not in _DB_CACHE:
        _DB_CACHE[db_key] = load_db(db_key)
    # StatsLookup carries diagnostic trace fields, so make one per case.  The underlying fitted
    # distributions are read-mostly and are safe to retain; lookup only assigns their deterministic
    # resolution-level annotation.
    return StatsLookup(_DB_CACHE[db_key], q1_variant=ctx.get("q1_variant"),
                       orientation_variant=ctx.get("orientation_variant"))


def _evaluator(mace_cfg: dict | None):
    """The process-wide MACE evaluator for this configuration (built once, kept on its device)."""
    if not mace_cfg:
        return None
    sys.path.insert(0, str(REPO / "src"))
    key = (str(Path(mace_cfg["path"]).resolve()), mace_cfg["mode"], mace_cfg["dtype"], mace_cfg["threads"],
           mace_cfg["device"])
    if key not in _MACE_CACHE:
        from jacobik.core.energy import MACEPolarEvaluator

        _MACE_CACHE[key] = MACEPolarEvaluator(model_path=key[0], mode=key[1], dtype=key[2], threads=key[3],
                                              device=key[4])
    return _MACE_CACHE[key]


def _apply_energies(out: dict, e) -> None:
    """Energies of out's structures (reference first, then conformers in score order) -> dE fields (kJ/mol)."""
    from jacobik.core.energy import EV_TO_KJ_MOL

    e_ref = e[0]
    if "scores_all" in out:
        out["dE_all"] = [float((x - e_ref) * EV_TO_KJ_MOL) for x in e[1:]]
        out["dE"] = out["dE_all"][0]
    else:
        out["dE"] = float((e[1] - e_ref) * EV_TO_KJ_MOL)


def _tmqm_case(data_dir: Path, split: str, case: str):
    """Return one fixture complex from a worker-local, incrementally built split index.

    Keeping the parser iterator means the first case only reads as far as that case.  Every molecule
    passed on the way is retained, so later cases never restart at the beginning of the split.
    """
    from fit.parser import TMQMParser

    key = (str(data_dir.resolve()), split)
    if key not in _TMQM_SPLIT_CACHE:
        xyz = (data_dir / "xyz" / f"tmQM_{split}.xyz").read_text()
        mnd = {
            code: int(n)
            for code, n in re.findall(r"CSD_code\s*=\s*([^\s|]+)[^\n]*?\bMND\s*=\s*(\d+)", xyz)
        }
        _TMQM_SPLIT_CACHE[key] = {
            "mols": {}, "mnd": mnd, "stream": iter(TMQMParser(data_dir).stream_split(split)), "done": False,
        }
    entry = _TMQM_SPLIT_CACHE[key]
    while case not in entry["mols"] and not entry["done"]:
        try:
            mol = next(entry["stream"])
        except StopIteration:
            entry["done"] = True
            entry["stream"] = None
            break
        entry["mols"][mol.csd_code] = mol
    if case not in entry["mols"]:
        raise KeyError(f"{case} not found in tmQM split {split!r} under {data_dir}")
    return entry["mols"][case], entry["mnd"][case]


# ----------------------------------------------------------------------------------------------------
# worker (runs inside the per-case subprocess)
# ----------------------------------------------------------------------------------------------------
def _worker(case: str, db: str, ctx: dict, mace_cfg: dict | None = None, solver_device: str = "cpu",
            return_structures: bool = False) -> dict:
    """Solve one case. With ``mace_cfg`` the energies are evaluated here; with ``return_structures`` the
    structures to evaluate are returned under "_structures" (reference first) for the driver's evaluator."""
    sys.path.insert(0, str(REPO / "src"))
    import logging

    logging.getLogger("jacobik").setLevel(logging.ERROR)
    import numpy as np

    from jacobik import ComplexBuilder, LigandInput

    setup_t0 = time.perf_counter()
    stats = _stats_for(db, ctx)
    ev = None if return_structures else _evaluator(mace_cfg)
    setup_s = time.perf_counter() - setup_t0
    SOLVE_KW = dict(ctx["solve_kw"])
    if solver_device != "cpu":
        SOLVE_KW["device"] = solver_device
    want_energy = bool(ev) or return_structures

    def best_of(res):
        c = max(res.conformers, key=lambda c: c.score)
        X = np.array([a.position for a in c.complex.atoms])
        return c, X, [a.symbol for a in c.complex.atoms]

    def conv(res):
        return int(sum(bool(getattr(c, "converged", False)) for c in res.conformers))

    out: dict = {"case": case, "setup_s": setup_s, "worker_pid": os.getpid(), "solver_device": solver_device}

    def finish(structures):
        """Evaluate here (worker-side MACE) or hand the structures to the driver."""
        if return_structures:
            out["_structures"] = [(list(sy_), np.asarray(X_, dtype=np.float64), int(q_), int(m_))
                                  for sy_, X_, q_, m_ in structures]
        elif ev:
            t0 = time.perf_counter()
            _apply_energies(out, ev.energies(structures))
            out["energy_s"] = time.perf_counter() - t0
        return out

    if case.startswith("aqua_"):
        name = case[5:]
        metal, q, cn = re.match(r"([A-Z][a-z]?)(\d)_(\d)H2O", name).groups()
        q, cn = int(q), int(cn)
        spin = SPIN[metal if metal != "Fe" else f"Fe{q}"]
        L = [x.split() for x in (FIX / "doe_aqua" / f"{name}.xyz").read_text().splitlines()[2:] if x.strip()]
        sy, Xr = [x[0] for x in L], np.array([[float(v) for v in x[1:4]] for x in L])
        mi = sy.index(metal)
        t0 = time.perf_counter()
        res = ComplexBuilder(metal, GEOM[cn], charge=q, spin=spin,
                             ligands=[LigandInput("O", donors=[[0]]) for _ in range(cn)]).solve(stats=stats, **SOLVE_KW)
        out["solve_s"] = time.perf_counter() - t0
        c, X, ys = best_of(res)

        def tilt(sy_, X_, m_):
            v = []
            for i, s in enumerate(sy_):
                if s != "O" or np.linalg.norm(X_[i] - X_[m_]) > 3.2:
                    continue
                hs = [j for j, t in enumerate(sy_) if t == "H" and np.linalg.norm(X_[j] - X_[i]) < 1.25]
                b = (X_[hs[0]] + X_[hs[1]]) / 2 - X_[i]
                m = X_[m_] - X_[i]
                v.append(np.degrees(np.arccos(np.dot(b, m) / np.linalg.norm(b) / np.linalg.norm(m))))
            return float(np.mean(v))

        dref = sorted(np.linalg.norm(Xr[i] - Xr[mi]) for i, s in enumerate(sy) if s == "O" and np.linalg.norm(Xr[i] - Xr[mi]) < 3.2)
        dgot = sorted(np.linalg.norm(X[i] - X[0]) for i, s in enumerate(ys) if s == "O" and np.linalg.norm(X[i] - X[0]) < 3.5)
        out.update(score=float(c.score), converged=conv(res), n_donors=cn, n_donors_crystal=len(dref),
                   max_md_err=float(max(abs(a - b) for a, b in zip(dref, dgot))) if len(dgot) == len(dref) else None,
                   tilt=tilt(ys, X, 0), tilt_ref=tilt(sy, Xr, mi))
        if want_energy:
            return finish([(sy, Xr, q, spin), (ys, X, q, spin)])
        return out

    if case.startswith("legacy_"):
        if case == "legacy_co_edta":
            b = ComplexBuilder("Co", "octahedral", charge=-1, spin=1,
                               ligands=[LigandInput(EDTA, donors=[[0], [4], [7], [11], [15], [19]])])
        elif case == "legacy_la_edta_cn9":
            b = ComplexBuilder("La", "tricap_pris", charge=-1, spin=1,
                               ligands=[LigandInput(EDTA, donors=[[4], [11], [0], [8], [15], [19]])]
                               + [LigandInput("O", donors=[[0]]) for _ in range(3)])
        else:
            b = ComplexBuilder("Co", "octahedral", charge=3, spin=1,
                               ligands=[LigandInput("NCCN", donors=[[0], [3]]) for _ in range(3)])
        t0 = time.perf_counter()
        res = b.solve(stats=stats, **SOLVE_KW)
        out["solve_s"] = time.perf_counter() - t0
        c, X, ys = best_of(res)
        d = [float(np.linalg.norm(X[i - 1] - X[0])) for lig in c.complex.ligands for i in lig.donor_atoms]
        out.update(score=float(c.score), converged=conv(res), n_donors=len(d), max_md=max(d))
        return out

    # tmQM case
    from jacobik.builder.ligands import _build_ligand_groups, ligand_inputs_from_mol

    # JDEV_TMQM_DIR / JDEV_TMQM_SPLIT: take tmQM cases from another split (e.g. a held-out sample); default = fixtures
    tdir = Path(ctx.get("tmqm_dir") or (FIX / "tmqm")); tsplit = ctx.get("tmqm_split") or "P"
    m, mnd = _tmqm_case(tdir, tsplit, case)
    groups = _build_ligand_groups(m, m.smiles)
    lig_in = ligand_inputs_from_mol(m, groups)
    t0 = time.perf_counter()
    res = ComplexBuilder(metal=m.metal_symbol, geometry=m.coord_geom, charge=m.charge, spin=m.spin_state,
                         ligands=lig_in).solve(stats=stats, **SOLVE_KW)
    out["solve_s"] = time.perf_counter() - t0
    c, X, ys = best_of(res)
    rb = {a.idx: i for i, a in enumerate(m.atoms)}
    Xr = np.array([a.position for a in m.atoms])
    mr = rb[m.metal_idx]
    errs = []
    for lig, li, lg in zip(sorted(c.complex.ligands, key=lambda l: l.ligand_id), lig_in, groups):
        for g in li.donors:
            for dd in g:
                if dd in lg["rdkit_to_xyz"]:
                    dc = np.linalg.norm(Xr[rb[lg["rdkit_to_xyz"][dd]]] - Xr[mr])
                    errs.append(abs(np.linalg.norm(X[lig.atom_indices[dd] - 1] - X[0]) - dc))
    # n_donor_atoms (2026-10-16): tmQM's MND counts every metal-bonded atom (Cp 5, arene 6); compare with it, not groups
    out.update(score=float(c.score), converged=conv(res), n_donors=sum(len(li.donors) for li in lig_in),
               n_donor_atoms=sum(len(g) for li in lig_in for g in li.donors),
               n_donors_crystal=mnd, max_md_err=float(max(errs)) if errs else None, n_atoms=len(m.atoms))
    if ctx.get("save_xyz"):
        # best conformer + crystal + heavy-atom map (solved row -> crystal row) for overlays (tools/jdev/viz.py)
        sd = Path(ctx["save_xyz"]); sd.mkdir(parents=True, exist_ok=True)
        amap = {0: mr}
        for lig, lg in zip(sorted(c.complex.ligands, key=lambda l: l.ligand_id), groups):
            for rd, x1 in lg["rdkit_to_xyz"].items():
                if rd < len(lig.atom_indices) and ys[lig.atom_indices[rd] - 1] != "H":
                    amap[lig.atom_indices[rd] - 1] = rb[x1]
        _w = lambda f, s_, P: f.write_text(f"{len(s_)}\n{case}\n" + "".join(f"{a} {x:.6f} {y:.6f} {z:.6f}\n" for a, (x, y, z) in zip(s_, P)))
        _w(sd / f"{case}_solved.xyz", ys, X)
        _w(sd / f"{case}_crystal.xyz", [a.symbol for a in m.atoms], Xr)
        (sd / f"{case}_map.json").write_text(json.dumps({str(k): int(v) for k, v in amap.items()}))
    if want_energy:
        ref = ([a.symbol for a in m.atoms], Xr, m.charge, m.spin_state)
        if ctx.get("ensemble"):
            # every returned conformer (score order): for top-1 / within-ensemble rank correlation (tools/jdev/bench)
            confs = sorted(res.conformers, key=lambda cc: -cc.score)
            out["scores_all"] = [float(cc.score) for cc in confs]
            return finish([ref] + [
                ([a.symbol for a in cc.complex.atoms], np.array([a.position for a in cc.complex.atoms]),
                 m.charge, m.spin_state)
                for cc in confs
            ])
        return finish([ref, (ys, X, m.charge, m.spin_state)])
    return out


# ----------------------------------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------------------------------
#: upper limit of solver rows per multi-seed pass (GPU memory: about 0.35-0.5 MB per row on the fast path)
SEEDS_MAX_ROWS = int(os.environ.get("JDEV_SEEDS_MAX_ROWS", "6000"))


def _worker_seeds(case: str, db: str, ctxs: dict, solver_device: str, return_structures: bool) -> dict:
    """One case at several seeds with ONE solver pass (tools/jdev/multiseed.py): {seed: result as from _worker}.

    `ctxs` maps seed -> that seed's run context (they differ only in the seed and the xyz folder). _worker runs once
    per seed, unchanged; ComplexBuilder.solve is replaced for the duration, so the first call solves every seed
    together and each call gets its own seed's result. Needs the no-culling GPU solver; if the joint pass is not
    possible (other solver, out of memory) every seed is solved by itself, as before. `solve_s` of each seed is its
    share of the pass (pass time / seeds); `solve_pass_s` and `seeds_in_pass` say what was measured."""
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO))
    from jacobik import ComplexBuilder
    from tools.jdev import multiseed

    seeds = list(ctxs)
    orig = ComplexBuilder.solve
    state = {"res": None, "busy": False, "n": len(seeds), "why": None}

    def solve(self, **kw):
        if state["busy"] or state["n"] == 1:
            return orig(self, **kw)
        if state["res"] is None:
            rest = {k: v for k, v in kw.items() if k != "seed"}
            state["busy"] = True
            try:
                state["res"], state["info"] = multiseed.solve_seeds(self, seeds, max_rows=SEEDS_MAX_ROWS, _solve=orig,
                                                                    **rest)
            except RuntimeError as exc:
                state["n"], state["why"] = 1, str(exc).splitlines()[0][:200]      # every seed by itself from here on
                if "out of memory" in str(exc).lower():
                    import torch

                    torch.cuda.empty_cache()
                return orig(self, **kw)
            finally:
                state["busy"] = False
        return state["res"][int(kw["seed"])]

    ComplexBuilder.solve = solve
    outs = {}
    try:
        for s in seeds:
            try:
                r = _worker(case, db, ctxs[s], None, solver_device, return_structures)
                r["status"] = "ok"
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                r = {"case": case, "status": "ERROR", "error": traceback.format_exc()[-4000:]}
            outs[s] = r
    finally:
        ComplexBuilder.solve = orig
    ok = [r for r in outs.values() if r.get("status") == "ok" and r.get("solve_s") is not None]
    if state["res"] is not None and ok:
        total = sum(r["solve_s"] for r in ok)
        for r in ok:
            r.update(solve_s=total / len(ok), solve_pass_s=total, seeds_in_pass=len(ok))
    elif state["why"]:
        for r in outs.values():
            r["seeds_in_pass_skipped"] = state["why"]
    return outs


def _run_case_subprocess(case, db, ctx, mace_cfg, timeout):
    """Strict-isolation fallback: a brand-new interpreter for one case (settings travel in JDEV_* variables)."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", case, "--db", str(db)]
    if mace_cfg:
        cmd += ["--mace-model", str(mace_cfg["path"]), "--mace-mode", mace_cfg["mode"], "--mace-dtype",
                mace_cfg["dtype"], "--mace-threads", str(mace_cfg["threads"]), "--mace-device", mace_cfg["device"]]
    else:
        cmd += ["--no-energy"]
    kw = dict(ctx["solve_kw"])
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"),
               JDEV_SEED=str(kw.pop("seed", 11)), JDEV_SOLVE=json.dumps(kw))
    for var, key in (("JDEV_TMQM_DIR", "tmqm_dir"), ("JDEV_TMQM_SPLIT", "tmqm_split"), ("JDEV_SAVE_XYZ", "save_xyz"),
                     ("JDEV_Q1_VARIANT", "q1_variant"), ("JDEV_ORIENTATION_VARIANT", "orientation_variant")):
        if ctx.get(key):
            env[var] = str(ctx[key])
    if ctx.get("ensemble"):
        env["JDEV_ENSEMBLE_DE"] = "1"
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(REPO), env=env)
    except subprocess.TimeoutExpired:
        return {"case": case, "status": "TIMEOUT", "wall_s": timeout}
    line = [x for x in p.stdout.splitlines() if x.startswith("@@RESULT ")]
    if p.returncode != 0 or not line:
        return {"case": case, "status": "ERROR", "error": p.stderr[-1500:], "wall_s": time.perf_counter() - t0}
    r = json.loads(line[-1][len("@@RESULT "):])
    r["status"] = "ok"
    r["wall_s"] = time.perf_counter() - t0
    return r


def _run_case_pooled(case, db, ctx, mace_cfg, return_structures):
    """loky task. Exceptions become ordinary result records unless the worker itself dies."""
    try:
        r = _worker(case, db, ctx, mace_cfg, "cpu", return_structures)
        r["status"] = "ok"
        return r
    except BaseException:
        return {"case": case, "status": "ERROR", "error": traceback.format_exc()[-4000:]}


def _loky_executor(jobs):
    """A bounded reusable process pool with native libraries kept single-threaded per solver chunk."""
    from joblib.externals.loky import get_reusable_executor

    return get_reusable_executor(
        max_workers=max(1, jobs),
        timeout=3600,          # idle workers stay warm between the runs of a plan
        kill_workers=True,
        reuse=False,
        env={
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "1"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "1"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "1"),
        },
    )


class CaseRunner:
    """Runs cases for one or many result files with everything expensive created once.

    executor "loky": persistent CPU worker processes (kept across runs). executor "inline": cases run in this
    process, one at a time, on ``solver_device``. ``driver_mace`` (always for a CUDA MACE and for inline): this
    process owns the single MACE evaluator and evaluates the structures the solves return, so the model is
    loaded once and never leaves its device.
    """

    def __init__(self, executor: str, jobs: int, solver_device: str, mace_cfg: dict | None, timeout: float,
                 mace_where: str = "auto"):
        self.executor, self.jobs, self.solver_device = executor, max(1, jobs), solver_device
        self.mace_cfg, self.timeout = mace_cfg, timeout
        in_driver = (executor == "inline" or str((mace_cfg or {}).get("device")).startswith("cuda")
                     or mace_where == "driver")
        if mace_where == "workers" and executor == "loky":
            in_driver = False     # one model per worker process (CPU: parallel energies; CUDA: one copy each)
        self.driver_mace = bool(mace_cfg) and in_driver and executor != "subprocess"
        self.ev = None
        self._pool = None
        self._pooled_task = None
        self.energy_s_total = 0.0
        if self.driver_mace:
            self.ev = _evaluator(mace_cfg).load()

    # -- shared: driver-side energies ------------------------------------------------------------
    def _energies(self, r: dict) -> dict:
        structures = r.pop("_structures", None)
        if structures is None or self.ev is None or r.get("status") != "ok":
            return r
        t0 = time.perf_counter()
        try:
            _apply_energies(r, self.ev.energies(structures))
        except BaseException:
            r["status"] = "ERROR"
            r["error"] = "MACE evaluation failed:\n" + traceback.format_exc()[-3000:]
        r["energy_s"] = time.perf_counter() - t0
        self.energy_s_total += r["energy_s"]
        if r.get("wall_s") is not None:
            r["wall_s"] += r["energy_s"]
        return r

    # -- inline -----------------------------------------------------------------------------------
    def _run_inline(self, cases, db, ctx):
        cuda_solver = self.solver_device.startswith("cuda")
        for case in cases:
            t0 = time.perf_counter()
            try:
                if cuda_solver:
                    import torch

                    torch.cuda.reset_peak_memory_stats()
                r = _worker(case, str(db), ctx, None, self.solver_device, return_structures=bool(self.ev))
                r["status"] = "ok"
                if cuda_solver:
                    r["solver_gpu_peak_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
                    # hand the solver's cached blocks back before MACE batches the ensemble on the same GPU
                    torch.cuda.empty_cache()
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                r = {"case": case, "status": "ERROR", "error": traceback.format_exc()[-4000:]}
            r = self._energies(r)
            r["wall_s"] = time.perf_counter() - t0
            if r["status"] == "ok" and r["wall_s"] > self.timeout:
                r["status"] = "TIMEOUT"  # inline cases cannot be killed; flagged after the fact
            yield case, r

    def run_seeds(self, cases, db, ctxs):
        """Inline only: yield (case, {seed: result}); all seeds of a case go through the solver in one pass."""
        cuda_solver = self.solver_device.startswith("cuda")
        for case in cases:
            t0 = time.perf_counter()
            if cuda_solver:
                import torch

                torch.cuda.reset_peak_memory_stats()
            by_seed = _worker_seeds(case, str(db), ctxs, self.solver_device, return_structures=bool(self.ev))
            peak = None
            if cuda_solver:
                peak = round(torch.cuda.max_memory_allocated() / 2**20, 1)
                torch.cuda.empty_cache()     # hand the solver's cached blocks back before MACE batches the ensembles
            share = (time.perf_counter() - t0) / max(len(by_seed), 1)
            for r in by_seed.values():
                if peak is not None and r.get("status") == "ok":
                    r["solver_gpu_peak_mb"] = peak
                r["wall_s"] = share
                self._energies(r)            # adds the seed's own MACE time to its wall_s
                if r["status"] == "ok" and r["wall_s"] > self.timeout:
                    r["status"] = "TIMEOUT"
            yield case, by_seed

    # -- loky -------------------------------------------------------------------------------------
    def _run_loky(self, cases, db, ctx):
        """Yield completed (case, result) pairs; rebuild the whole pool after a hard timeout.

        At most ``jobs`` futures are submitted, so a task's deadline starts when a worker slot is
        reserved rather than while it is sitting behind an unbounded queue.  If one task hangs we must
        kill the process that owns it; loky cannot kill only that worker safely, so other in-flight cases
        are resubmitted to a clean pool.  Completed results are never repeated.
        """
        if self._pooled_task is None:
            # When this file is launched as a script, submitting the __main__ function makes cloudpickle
            # serialize its globals by value for every task.  Resolve the callable through an importable
            # module name instead, so all tasks in a worker share that module's resource caches.
            import importlib

            if str(REPO) not in sys.path:
                sys.path.insert(0, str(REPO))
            self._pooled_task = importlib.import_module("tests.regression.run_panel")._run_case_pooled
        if self._pool is None:
            self._pool = _loky_executor(self.jobs)
        worker_mace = None if self.driver_mace else self.mace_cfg
        remaining = list(cases)
        pending = {}

        def fill():
            while remaining and len(pending) < self.jobs:
                case = remaining.pop(0)
                pending[self._pool.submit(self._pooled_task, case, str(db), ctx, worker_mace,
                                          self.driver_mace)] = (case, time.perf_counter())

        fill()
        while pending:
            done, _ = wait(tuple(pending), timeout=0.2, return_when=FIRST_COMPLETED)
            now = time.perf_counter()
            finished = []
            for future in done:
                case, started = pending.pop(future)
                try:
                    result = future.result()
                except BaseException:
                    result = {"case": case, "status": "ERROR", "error": traceback.format_exc()[-4000:]}
                result["wall_s"] = now - started
                finished.append((case, result))
            fill()   # keep the CPU workers busy while the driver evaluates energies
            for case, result in finished:
                yield case, self._energies(result)

            now = time.perf_counter()
            expired = [(future, case, started) for future, (case, started) in pending.items()
                       if now - started >= self.timeout]
            if not expired:
                continue

            expired_futures = {future for future, _, _ in expired}
            retry = []
            for future, (case, started) in list(pending.items()):
                pending.pop(future)
                if future in expired_futures:
                    yield case, {"case": case, "status": "TIMEOUT", "wall_s": self.timeout}
                else:
                    retry.append(case)
            remaining[:0] = retry
            self._pool.shutdown(wait=True, kill_workers=True)
            self._pool = _loky_executor(self.jobs)
            fill()

    # -- subprocess -------------------------------------------------------------------------------
    def _run_subprocess(self, cases, db, ctx):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            futs = {c: pool.submit(_run_case_subprocess, c, db, ctx, self.mace_cfg, self.timeout) for c in cases}
            for c in cases:
                yield c, futs[c].result()

    def run(self, cases, db, ctx):
        return {"inline": self._run_inline, "loky": self._run_loky, "subprocess": self._run_subprocess}[
            self.executor](cases, db, ctx)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True, kill_workers=True)
            self._pool = None


def _compare(r, b, same_machine):
    """-> (verdict, [reasons])"""
    if r["status"] != "ok":
        return "FAIL", [r["status"]]
    if not b or b.get("status") != "ok":
        return "NEW", []
    why, verdict = [], "ok"
    if r.get("n_donors") != b.get("n_donors"):
        verdict = "FAIL"; why.append(f"donors {b.get('n_donors')}->{r.get('n_donors')}")
    for key in ("max_md_err",):
        if r.get(key) is not None and b.get(key) is not None and r[key] > b[key] + 0.15 and r[key] > 0.3:
            verdict = "FAIL"; why.append(f"{key} {b[key]:.2f}->{r[key]:.2f} A")
    if r.get("max_md") is not None and b.get("max_md") is not None and r["max_md"] > max(b["max_md"] + 0.15, 2.6):
        verdict = "FAIL"; why.append(f"unbound donor: max M-D {b['max_md']:.2f}->{r['max_md']:.2f} A")
    if r.get("dE") is not None and b.get("dE") is not None:
        a, c = abs(b["dE"]), abs(r["dE"])
        tol = max(25.0, 0.25 * a)
        if c > a + tol:
            verdict = "FAIL"; why.append(f"|dE| {a:.0f}->{c:.0f}")
        elif c < a - tol:
            why.append(f"BETTER |dE| {a:.0f}->{c:.0f}")
            verdict = "BETTER" if verdict == "ok" else verdict
    if r["score"] < b["score"] - 1.0:
        verdict = "FAIL"; why.append(f"score {b['score']:.2f}->{r['score']:.2f}")
    if same_machine and r.get("solve_s") and b.get("solve_s") and r["solve_s"] > 2 * b["solve_s"] + 2:
        why.append(f"SLOWER {b['solve_s']:.0f}->{r['solve_s']:.0f}s")
        verdict = "WARN" if verdict in ("ok", "BETTER") else verdict
    return verdict, why


def _resolve_devices(a):
    """(solver_device, mace_device) from --device / --solver-device / --mace-device."""
    sys.path.insert(0, str(REPO / "src"))
    from jacobik.core.energy import resolve_device

    solver = a.solver_device or (a.device if a.device else "cpu")
    mace = a.mace_device or (a.device if a.device else "auto")
    return resolve_device(solver), resolve_device(mace)


def _device_banner(solver_device, mace_device, mace):
    import torch

    lines = [f"torch {torch.__version__} (CUDA build: {torch.version.cuda}); CUDA available: {torch.cuda.is_available()}"]
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(i).total_memory / 2**30
        lines.append(f"GPU {i}: {torch.cuda.get_device_name(i)} ({total:.1f} GB)")
    lines.append(f"solver device: {solver_device}   MACE device: {mace_device if mace else '-'}")
    return "\n".join(lines)


class _RunFile:
    """One result file being filled: per-case lines while it runs, summary and JSON at the end."""

    def __init__(self, run, a, tag=""):
        self.run, self.a, self.tag = run, a, tag
        self.cases = run.get("cases") or ALL_CASES
        self.db, self.ctx = run["db"], run["ctx"]
        self.base = json.loads(Path(a.baseline).read_text()) if Path(a.baseline).exists() else {"cases": {}}
        self.same_machine = self.base.get("machine") == platform.node()
        self.t0 = time.perf_counter()
        self.results = {}

    def record(self, c, r):
        results, cases, base = self.results, self.cases, self.base
        results[c] = r
        v, why = _compare(r, base["cases"].get(c), self.same_machine)
        r["verdict"], r["why"] = v, why
        b = base["cases"].get(c, {})
        fmt = lambda x, f: (f % x) if isinstance(x, (int, float)) else "-"
        print(f"{v:7s} [{len(results):>3d}/{len(cases)}] {self.tag}{c:22s} "
              f"t {fmt(r.get('solve_s'), '%6.1f'):>6s}s  score {fmt(r.get('score'), '%7.2f'):>7s} "
              f"(base {fmt(b.get('score'), '%7.2f'):>7s})  dE {fmt(r.get('dE'), '%8.1f'):>8s} "
              f"(base {fmt(b.get('dE'), '%8.1f'):>8s})  maxMDerr {fmt(r.get('max_md_err'), '%.2f'):>5s}  "
              f"donors {r.get('n_donors', '-')}/{r.get('n_donors_crystal', '-')}  {'; '.join(why)}", flush=True)
        perf = [f"wall {fmt(r.get('wall_s'), '%.1f')}s", f"setup {fmt(r.get('setup_s'), '%.2f')}s"]
        if r.get("energy_s") is not None:
            perf.append(f"MACE {r['energy_s']:.1f}s")
        if r.get("solver_gpu_peak_mb") is not None:
            perf.append(f"solver GPU peak {r['solver_gpu_peak_mb']:.0f} MB")
        if r.get("seeds_in_pass"):
            perf.append(f"{r['seeds_in_pass']} seeds in one pass of {r['solve_pass_s']:.1f}s")
        if r.get("seeds_in_pass_skipped"):
            perf.append(f"seeds solved one by one ({r['seeds_in_pass_skipped']})")
        if r.get("worker_pid") is not None:
            perf.append(f"worker {r['worker_pid']}")
        print("        " + " | ".join(perf), flush=True)
        if r.get("status") == "ERROR" and r.get("error"):
            print("        " + str(r["error"]).strip().splitlines()[-1][:300], flush=True)

    def finish(self, runner, mace, solver_workers, devices):
        """Summary, comparison to the baseline, results JSON. Returns the regression count."""
        a, run, cases, base, db, ctx = self.a, self.run, self.cases, self.base, self.db, self.ctx
        # Completion order is useful for live progress; persisted results retain the requested case order.
        results = {c: self.results[c] for c in cases}

        # group summaries
        def med(keys, field, src):
            v = [abs(src[k][field]) for k in keys if k in src and src[k].get(field) is not None]
            return (st.median(v), len(v)) if v else (None, 0)

        print()
        for label, keys in (("aqua", [c for c in cases if c.startswith("aqua_")]),
                            ("tmQM", [c for c in cases if not c.startswith(("aqua_", "legacy_"))])):
            now, n = med(keys, "dE", results)
            was, _ = med(keys, "dE", base["cases"])
            if now is not None:
                print(f"{label:5s} median |dE| {now:7.1f} kJ/mol over {n}" + (f"   (baseline {was:.1f})" if was is not None else ""))
        n_fail = sum(r["verdict"] == "FAIL" for r in results.values())
        n_warn = sum(r["verdict"] == "WARN" for r in results.values())
        panel_wall_s = time.perf_counter() - self.t0
        worker_pids = {r.get("worker_pid") for r in results.values() if r.get("worker_pid") is not None}
        print(f"\n{self.tag}{'REGRESSIONS: ' + str(n_fail) if n_fail else 'no regressions'}; warnings: {n_warn}")
        print(f"panel wall: {panel_wall_s:.1f}s; worker processes used: {len(worker_pids) or '-'}")

        stamp = time.strftime("%Y%m%d-%H%M%S")
        (HERE / "results").mkdir(exist_ok=True)
        try:
            sys.path.insert(0, str(REPO))
            from tools.jdev.env import tree_hash

            src_hash = tree_hash()
        except Exception:  # pragma: no cover
            src_hash = None
        kw = dict(ctx["solve_kw"])
        payload = {"machine": platform.node(), "time": stamp, "db": db, "mace": mace,
                   "mace_mode": a.mace_mode if mace else None,
                   "mace_dtype": a.mace_dtype if mace else None,
                   "mace_threads": runner.mace_cfg["threads"] if mace else None,
                   "solver_device": devices[0], "mace_device": devices[1] if mace else None,
                   "mace_info": runner.ev.describe() if runner.ev is not None else None,
                   "cases": results,
                   "src_hash": src_hash, "solve_kw": ctx["solve_kw"],
                   "case_executor": runner.executor, "jobs": runner.jobs, "solver_workers": solver_workers,
                   "panel_wall_s": panel_wall_s,
                   # every result-changing setting as the solver saw it (jacobik/profile.py; same env as the workers)
                   "profile": __import__("jacobik.core.profile", fromlist=["snapshot"]).snapshot(),
                   "env": {"JDEV_Q1_VARIANT": ctx.get("q1_variant"), "JDEV_ORIENTATION_VARIANT": ctx.get("orientation_variant"),
                           "JDEV_FAST_EMBED": os.environ.get("JDEV_FAST_EMBED"), "JDEV_SEED": str(kw.pop("seed", None)),
                           "JDEV_TMQM_DIR": ctx.get("tmqm_dir"), "JDEV_TMQM_SPLIT": ctx.get("tmqm_split"),
                           "JDEV_SOLVE": json.dumps(kw), "JACOBIK_NUM_WORKERS": os.environ.get("JACOBIK_NUM_WORKERS"),
                           "JACOBIK_PROFILE": os.environ.get("JACOBIK_PROFILE"),
                           "JACOBIK_PROFILE_OVERRIDES": os.environ.get("JACOBIK_PROFILE_OVERRIDES")}}
        out_path = Path(run["out"]) if run.get("out") else HERE / "results" / f"{stamp}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1))
        os.replace(tmp, out_path)
        print(f"results: {out_path}", flush=True)
        if a.update_baseline:
            merged = base if base.get("machine") == platform.node() else {"cases": {}}
            merged.update(machine=platform.node(), time=stamp, db=db, mace=mace)
            merged.setdefault("cases", {}).update({c: r for c, r in results.items() if r["status"] == "ok"})
            Path(a.baseline).write_text(json.dumps(merged, indent=1))
            print(f"baseline updated: {a.baseline}")
        return n_fail


def _run_one(runner, run, a, mace, solver_workers, devices):
    """One result file: run its cases, compare to the baseline, write the JSON. Returns the regression count."""
    f = _RunFile(run, a)
    for c, r in runner.run(f.cases, f.db, f.ctx):
        f.record(c, r)
    return f.finish(runner, mace, solver_workers, devices)


def _seed_group_key(run):
    """Runs with the same key differ only in the seed (and the per-seed output paths): one solver pass per case."""
    ctx = {k: v for k, v in run["ctx"].items() if k != "save_xyz"}
    ctx["solve_kw"] = {k: v for k, v in ctx["solve_kw"].items() if k != "seed"}
    return json.dumps([run["db"], run.get("cases"), ctx], sort_keys=True)


def _run_group(runner, runs, a, mace, solver_workers, devices):
    """Several result files that differ only in the seed: per case one solver pass for all seeds, one file per seed."""
    files = {int(run["ctx"]["solve_kw"]["seed"]): _RunFile(run, a, tag=f"s{run['ctx']['solve_kw']['seed']} ") for run in runs}
    first = next(iter(files.values()))
    ctxs = {seed: f.ctx for seed, f in files.items()}
    for c, by_seed in runner.run_seeds(first.cases, first.db, ctxs):
        for seed, f in files.items():
            f.record(c, by_seed[seed])
    return sum(f.finish(runner, mace, solver_workers, devices) for f in files.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="all")
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--plan", default=None,
                    help="JSON file {'runs': [{out, db, cases, ctx}, ...]}: several result files in this one process "
                         "(pool, DBs and MACE stay loaded across them); built by tools/jdev/bench.py")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None,
                    help="set BOTH the solver and the MACE device (cuda: solver on the GPU, cases inline)")
    ap.add_argument("--solver-device", default=None, help="cpu (default) | cuda | cuda:N | auto")
    ap.add_argument("--mace-device", default=None, help="auto (default: CUDA when available) | cpu | cuda | cuda:N")
    ap.add_argument("--mace-model", default=os.environ.get("JACOBIK_MACE_MODEL_PATH") or (str(DEFAULT_MACE) if DEFAULT_MACE.exists() else None))
    ap.add_argument("--mace-mode", choices=("energy-only", "full"), default="energy-only",
                    help="energy-only batches fast single points; full retains forces/stress and the historical ASE path")
    ap.add_argument("--mace-dtype", choices=("float32", "float64"), default="float32",
                    help="MACE inference precision (default: float32; use float64 for precision-sensitive force work)")
    ap.add_argument("--mace-threads", type=int, default=None,
                    help="CPU threads per MACE evaluator (default: the per-case solver thread budget; unused on CUDA)")
    ap.add_argument("--mace-where", choices=("auto", "driver", "workers"), default="auto",
                    help="auto: one model in this process when MACE is on CUDA or cases run inline, else one per "
                         "loky worker; driver / workers force it")
    ap.add_argument("--no-energy", action="store_true")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S)
    ap.add_argument("--jobs", type=int, default=1, help="cases solved in parallel (CPU solver, loky executor)")
    ap.add_argument("--solver-workers", type=int, default=None,
                    help="CPU solver threads per case (default: divide available CPUs across --jobs, capped at 8)")
    ap.add_argument("--case-executor", choices=("loky", "inline", "subprocess"), default=None,
                    help="loky: warm CPU worker processes (default for a CPU solver); inline: cases run in this "
                         "process (default and only choice for a CUDA solver); subprocess: fresh process per case")
    ap.add_argument("--seeds-in-one-pass", choices=("auto", "on", "off"), default="auto",
                    help="--plan only: result files that differ only in the seed are solved together, per case one "
                         "solver pass for all seeds (tools/jdev/multiseed.py). auto: when the plan asks for it and the "
                         "solver runs inline on CUDA with the recorded-step solver")
    ap.add_argument("--baseline", default=str(HERE / "baseline.json"))
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--out", default=None, help="write the results JSON here (default results/<timestamp>.json)")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    mace = None if a.no_energy else a.mace_model
    solver_device, mace_device = _resolve_devices(a)

    if a.worker:  # child of --case-executor subprocess
        cfg = dict(path=mace, mode=a.mace_mode, dtype=a.mace_dtype, threads=max(1, a.mace_threads or 1),
                   device=mace_device) if mace else None
        print("@@RESULT " + json.dumps(_worker(a.worker, a.db, env_context(), cfg, solver_device)), flush=True)
        return

    executor = a.case_executor or ("inline" if solver_device.startswith("cuda") else "loky")
    if solver_device.startswith("cuda") and executor != "inline":
        ap.error("a CUDA solver runs its cases inline (one CUDA context, one case at a time); "
                 "use --case-executor inline or --solver-device cpu")
    jobs = 1 if executor == "inline" else max(1, a.jobs)

    # A CPU solve parallelises its batch in up to 8 torch worker threads.  Budget those threads across
    # concurrently running cases instead of letting --jobs multiply them past the machine's CPU count.
    # An explicit CLI or environment setting always wins.  (A CUDA solve is one batch on the GPU.)
    if a.solver_workers is not None:
        os.environ["JACOBIK_NUM_WORKERS"] = str(max(1, a.solver_workers))
    elif not os.environ.get("JACOBIK_NUM_WORKERS"):
        os.environ["JACOBIK_NUM_WORKERS"] = str(max(1, min(8, (os.cpu_count() or 1) // jobs)))
    solver_workers = 1 if solver_device.startswith("cuda") else int(os.environ["JACOBIK_NUM_WORKERS"])
    mace_threads = max(1, a.mace_threads or solver_workers)
    mace_cfg = dict(path=str(mace), mode=a.mace_mode, dtype=a.mace_dtype, threads=mace_threads,
                    device=mace_device) if mace else None

    plan_wants_groups = False
    if a.plan:
        plan_doc = json.loads(Path(a.plan).read_text())
        runs = plan_doc["runs"]
        plan_wants_groups = bool(plan_doc.get("seeds_in_one_pass"))
    else:
        runs = [{"out": a.out, "db": a.db, "cases": ALL_CASES if a.cases == "all" else a.cases.split(","),
                 "ctx": env_context()}]

    print(f"src: {REPO / 'src'}")
    print(_device_banner(solver_device, mace_device, mace))
    mace_desc = f"{mace} ({a.mace_mode}, {a.mace_dtype})" if mace else "off"
    print(f"MACE: {mace_desc}")
    t_load = time.perf_counter()
    runner = CaseRunner(executor, jobs, solver_device, mace_cfg, a.timeout, a.mace_where)
    if runner.ev is not None:
        print(f"MACE loaded once in the driver on {runner.ev.device} in {time.perf_counter() - t_load:.1f}s "
              f"(kept for all {len(runs)} run(s))")
    elif mace:
        print(f"MACE evaluated in the {jobs} worker process(es) on {mace_device}, {mace_threads} threads each")
    layout = (f"inline on {solver_device}" if executor == "inline"
              else f"{executor}: {jobs} processes x {solver_workers} solver threads")
    print(f"cases: {layout}\nbaseline: {a.baseline}\n", flush=True)

    group_seeds = executor == "inline" and bool(a.plan) and (
        a.seeds_in_one_pass == "on"
        or (a.seeds_in_one_pass == "auto" and plan_wants_groups and solver_device.startswith("cuda")))
    n_fail = 0
    t_all = time.perf_counter()
    try:
        done = set()
        for i, run in enumerate(runs, 1):
            if i in done:
                continue
            if a.plan:
                if run.get("skip_existing", True) and run.get("out") and Path(run["out"]).exists():
                    print(f">>> [{i}/{len(runs)}] {run.get('label', run['out'])}: exists, skipped", flush=True)
                    continue
                print(f">>> [{i}/{len(runs)}] {run.get('label', run['out'])}  db: {run['db']}", flush=True)
            group = [run]
            if group_seeds:
                # the later runs of the plan that differ from this one only in the seed (and are still to do)
                key = _seed_group_key(run)
                for k in range(i + 1, len(runs) + 1):
                    other = runs[k - 1]
                    if k in done or _seed_group_key(other) != key:
                        continue
                    if other.get("skip_existing", True) and other.get("out") and Path(other["out"]).exists():
                        continue
                    done.add(k)
                    group.append(other)
                    print(f">>> [{k}/{len(runs)}] {other.get('label', other['out'])}: same pass", flush=True)
            if len(group) > 1:
                n_fail += _run_group(runner, group, a, mace, solver_workers, (solver_device, mace_device))
            else:
                n_fail += _run_one(runner, run, a, mace, solver_workers, (solver_device, mace_device))
    finally:
        runner.close()
    if a.plan:
        info = runner.ev.describe() if runner.ev is not None else {}
        print(f"plan finished: {len(runs)} run(s) in {time.perf_counter() - t_all:.0f}s; MACE time "
              f"{runner.energy_s_total:.0f}s {info}", flush=True)
        sys.exit(0)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
