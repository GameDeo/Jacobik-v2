"""Benchmark runs for the notebooks: named solver configurations x seeds x case sets, each run through the
regression panel runner (one continuous process: warm case workers or an inline CUDA solver, one MACE model),
results as tidy tables.

    from tools.jdev import bench
    cfg = {"v2": {}, "v2+bends": {"angle_bends": True}, "v1-like": {"chelate_embed": False, "orientation_model": "q3q4"}}
    paths = bench.run_matrix(cfg, seeds=[11, 12, 13], case_set="panel", db=DB, out_dir=RUNS)
    df = bench.load(paths)                       # one row per (config, seed, case)
    bench.per_case(df)                           # mean / sd / min / max |dE| per case and config

Every result file carries the src hash, DB path, solve() keyword arguments and seed (DEV_RULES section 2).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .env import CODE_REPO, REPO

PANEL = CODE_REPO / "dev" / "tests" / "regression" / "run_panel.py"
#: 40-complex random tmQM sample of 2026-09-23 (split "S"); 12 of its complexes are in the panel, the other 28 are HELDOUT28.
SAMPLE40 = CODE_REPO / "dev" / "tests" / "regression" / "fixtures" / "tmqm_sample40"

#: Case sets. "panel" = the 24-case regression panel; "heldout28" = the 40-complex random tmQM sample of 2026-09-23
#: minus the 12 panel complexes (never used for diagnosis or tuning).
HELDOUT28 = ("HAMLEM HAFZOG GIZGUR GOZXEY PEVGAB JETXEP VEGMIH KUGWIV AHIGOQ WUMLIZ YEVDUZ VIYKIX KEHCEF IJEVIC QOSGIP "
             "VOSCAK PUSSIF RAMDOA QELROS HATSAZ TINBIE BIDHEB XECSON MELLUN OKADUB PIDXOT AXUVEV SOWKIY").split()
#: N48 (48 random tmQM complexes, 2026-09-29) and RING10 (held-out small-donor-ring set, 2026-09-30): both excluded
#: from the 2026-10 refit by CSD code (src/fit/data/benchmark_holdout_codes.json). Split "N".
FIXTURES = CODE_REPO / "dev" / "tests" / "regression" / "fixtures"
TMQM_SETS = {
    "heldout28": (SAMPLE40, "S", HELDOUT28),
    "n48": (FIXTURES / "n48", "N", None),
    "ring10": (FIXTURES / "ring10", "N", None),
    # DEV32 (2026-10-16): the 32 dev complexes of the Q6 ablation / error budget, held out of the refit since 10-15
    "dev32": (REPO / "research/bench/tools/q6_ablation/dev_tmqm" if (REPO / "research/bench/tools/q6_ablation/dev_tmqm").exists() else REPO / "tools/q6_ablation/dev_tmqm", "D", None),
}


def case_list(case_set: str) -> List[str]:
    d, split, codes = TMQM_SETS[case_set]
    if codes:
        return list(codes)
    if (d / "codes.json").exists():
        return json.loads((d / "codes.json").read_text())
    import re

    return re.findall(r"CSD_code\s*=\s*(\w+)", (d / "xyz" / f"tmQM_{split}.xyz").read_text())


def _solver_on_cuda(device: Optional[str], solver_device: Optional[str]) -> bool:
    """Whether the solver will run on CUDA for these flags ("auto" resolved against this machine's torch)."""
    name = solver_device or device or "cpu"
    if name == "auto":
        import torch

        return bool(torch.cuda.is_available())
    return str(name).startswith("cuda")


def _spec(name: str, solve_kw: dict, seed: int, db: Path, out_dir: Path, cases: Optional[Sequence[str]],
          tmqm_dir: Optional[Path], tmqm_split: Optional[str], save_xyz: bool, tag: str, mace_mode: str,
          mace_dtype: str, cuda_solver: bool, ensemble: Optional[bool], skip_existing: bool):
    """(run entry for the panel's --plan, environment switches of this configuration)."""
    out_dir = Path(out_dir)
    # Mode and precision change measured energies and runtime, so they are part of the cache key; so is a CUDA
    # solver (other random stream and float rounding than the CPU solver: different conformers). Older
    # CPU / full / float64 files are neither overwritten nor silently reused.
    # CUDA solves default to the recorded-step fast path since 2026-10-03 (JACOBIK_GPU_REPLAY=0: eager loop).
    # The two never share a result file, and neither reuses a file written by the eager CUDA solver before that.
    cuda_tag = "__solver-cuda-replay" if os.environ.get("JACOBIK_GPU_REPLAY", "1") != "0" else "__solver-cuda"
    stem = f"{tag}__{name}__s{seed}__mace-{mace_mode}-{mace_dtype}" + (cuda_tag if cuda_solver else "")
    kw = dict(solve_kw)
    # per-configuration environment, e.g. {"JACOBIK_PROFILE_OVERRIDES": '{"donor_bends": true}'} (jacobik/profile.py;
    # the old per-setting switches are retired and raise)
    extra_env = kw.pop("_env", {}) or {}
    ctx = {"solve_kw": dict(kw, seed=int(seed)),
           "tmqm_dir": str(tmqm_dir) if tmqm_dir else None,
           "tmqm_split": str(tmqm_split or "S") if tmqm_dir else None,
           "save_xyz": str(out_dir / "xyz" / f"{tag}__{name}__s{seed}") if save_xyz else None,
           "ensemble": bool(os.environ.get("JDEV_ENSEMBLE_DE")) if ensemble is None else bool(ensemble),
           "q1_variant": os.environ.get("JDEV_Q1_VARIANT") or None,
           "orientation_variant": os.environ.get("JDEV_ORIENTATION_VARIANT") or None}
    run = {"label": f"{tag} | {name} | seed {seed}", "out": str(out_dir / f"{stem}.json"), "db": str(db),
           "cases": list(cases) if cases else None, "ctx": ctx, "skip_existing": bool(skip_existing)}
    return run, tuple(sorted((str(k), str(v)) for k, v in extra_env.items()))


def run_plan(specs: Sequence[tuple], out_dir: Path, jobs: int = 1, timeout: Optional[float] = None,
             verbose: bool = True, case_executor: Optional[str] = None, solver_workers: Optional[int] = None,
             mace_mode: str = "energy-only", mace_dtype: str = "float32", mace_threads: Optional[int] = None,
             device: Optional[str] = None, solver_device: Optional[str] = None,
             mace_device: Optional[str] = None, mace_where: Optional[str] = None,
             seeds_in_one_pass: Optional[bool] = None) -> List[Path]:
    """Run (run entry, env switches) specs. Everything that shares its environment switches goes through ONE panel
    process: its worker pool, parsed DBs and MACE model are created once and stay loaded for every result file
    (on CUDA: one context, the model never leaves the GPU). Returns the result paths in spec order.

    ``seeds_in_one_pass`` (None = on for a CUDA solver): result files that differ only in the seed are solved
    together, per complex one solver pass for all seeds; still one result file per seed, same names as before.
    Needs the recorded-step GPU solver (no row culling); otherwise the runner solves seed by seed."""
    out_dir = Path(out_dir)
    groups: Dict[tuple, List[dict]] = {}
    for run, env_key in specs:
        if not (run["skip_existing"] and Path(run["out"]).exists()):
            groups.setdefault(env_key, []).append(run)
    for gi, (env_key, runs) in enumerate(groups.items()):
        plan_dir = out_dir / "_plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan = plan_dir / f"plan_{time.strftime('%Y%m%d-%H%M%S')}_{gi}.json"
        group = _solver_on_cuda(device, solver_device) if seeds_in_one_pass is None else bool(seeds_in_one_pass)
        plan.write_text(json.dumps({"runs": runs, "seeds_in_one_pass": group}, indent=1))
        cmd = [sys.executable, str(PANEL), "--plan", str(plan), "--jobs", str(jobs),
               "--mace-mode", mace_mode, "--mace-dtype", mace_dtype]
        if seeds_in_one_pass is not None:
            cmd += ["--seeds-in-one-pass", "on" if seeds_in_one_pass else "off"]
        for flag, val in (("--case-executor", case_executor), ("--timeout", timeout),
                          ("--solver-workers", solver_workers), ("--mace-threads", mace_threads),
                          ("--device", device), ("--solver-device", solver_device), ("--mace-device", mace_device),
                          ("--mace-where", mace_where)):
            if val is not None:
                cmd += [flag, str(val)]
        env = dict(os.environ, **dict(env_key))
        if verbose:
            # Stream stdout line-by-line through print() so Jupyter cell output captures it.
            # (capture_output=False would write to raw fd-level stdout, which Jupyter silently drops.)
            stderr_buf: List[str] = []
            proc = subprocess.Popen(cmd, cwd=str(REPO), env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            def _drain_stderr():
                for ln in proc.stderr:
                    stderr_buf.append(ln)
            t = threading.Thread(target=_drain_stderr, daemon=True)
            t.start()
            for line in proc.stdout:
                print(line, end="", flush=True)
            proc.wait()
            t.join()
            stdout_tail, stderr_tail, code = "", "".join(stderr_buf[-50:]), proc.returncode
        else:
            r = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
            stdout_tail, stderr_tail, code = (r.stdout or "")[-2000:], (r.stderr or "")[-2000:], r.returncode
        missing = [run["label"] for run in runs if not Path(run["out"]).exists()]
        if missing:
            raise RuntimeError(f"panel process ended (exit {code}) without results for {missing[:5]}"
                               f"{' ...' if len(missing) > 5 else ''}:\n{stdout_tail}\n{stderr_tail}")
    return [Path(run["out"]) for run, _ in specs]


def run(name: str, solve_kw: dict, seed: int, db: Path, out_dir: Path, cases: Optional[Sequence[str]] = None,
        tmqm_dir: Optional[Path] = None, tmqm_split: Optional[str] = None, jobs: int = 1, save_xyz: bool = False,
        timeout: Optional[float] = None, tag: str = "panel", skip_existing: bool = True, quiet: bool = True,
        verbose: bool = False, mace_mode: str = "energy-only", mace_dtype: str = "float32",
        ensemble: Optional[bool] = None, device: Optional[str] = None, solver_device: Optional[str] = None,
        **launch) -> Path:
    """One panel run of one configuration at one seed. Returns the results JSON path.

    ``verbose=True`` streams the panel's per-case output (verdict, time, score, dE, maxMDerr, donors).
    ``device`` / ``solver_device`` / ``mace_device`` and the other launch options: see run_plan.
    """
    spec = _spec(name, solve_kw, seed, db, out_dir, cases, tmqm_dir, tmqm_split, save_xyz, tag, mace_mode,
                 mace_dtype, _solver_on_cuda(device, solver_device), ensemble, skip_existing)
    return run_plan([spec], out_dir, jobs=jobs, timeout=timeout, verbose=verbose,
                    mace_mode=mace_mode, mace_dtype=mace_dtype, device=device, solver_device=solver_device,
                    **launch)[0]


def matrix_specs(configs: Dict[str, dict], seeds: Iterable[int], db: Path, out_dir: Path, case_set: str = "panel",
                 tmqm_dir: Optional[Path] = None, cases: Optional[Sequence[str]] = None,
                 tmqm_split: Optional[str] = None, save_xyz: bool = False, mace_mode: str = "energy-only",
                 mace_dtype: str = "float32", ensemble: Optional[bool] = None, skip_existing: bool = True,
                 device: Optional[str] = None, solver_device: Optional[str] = None) -> List[tuple]:
    """The specs of every configuration x seed on one case set (see run_matrix)."""
    tag = case_set if not cases else f"{case_set}-sub"
    if case_set in TMQM_SETS:
        d, split, _ = TMQM_SETS[case_set]
        cases = cases or case_list(case_set)
        tmqm_dir = tmqm_dir or d
        tmqm_split = tmqm_split or split
    cuda_solver = _solver_on_cuda(device, solver_device)
    return [_spec(name, skw, seed, db, out_dir, cases, tmqm_dir, tmqm_split, save_xyz, tag, mace_mode, mace_dtype,
                  cuda_solver, ensemble, skip_existing)
            for seed in seeds for name, skw in configs.items()]


_SPEC_KEYS = ("tmqm_dir", "cases", "tmqm_split", "save_xyz", "ensemble", "skip_existing")


def run_matrix(configs: Dict[str, dict], seeds: Iterable[int], db: Path, out_dir: Path, case_set: str = "panel",
               verbose: bool = True, mace_mode: str = "energy-only", mace_dtype: str = "float32",
               device: Optional[str] = None, solver_device: Optional[str] = None, **kw) -> List[Path]:
    """Every configuration at every seed on one case set ("panel", "heldout28", "n48", "ring10"); ``cases``
    restricts it. Existing results are reused (a restricted run gets its own tag so it never masks a full one).
    A configuration may carry "_env": {VAR: value} for environment switches of that configuration.

    All of it runs in one panel process (see run_plan); ``verbose=True`` streams the per-complex output live.
    To keep the process (and the model on the GPU) alive across several case sets and DBs, use run_all.
    """
    spec_kw = {k: kw.pop(k) for k in _SPEC_KEYS if k in kw}
    specs = matrix_specs(configs, seeds, db, out_dir, case_set, mace_mode=mace_mode, mace_dtype=mace_dtype,
                         device=device, solver_device=solver_device, **spec_kw)
    return run_plan(specs, out_dir, verbose=verbose, mace_mode=mace_mode, mace_dtype=mace_dtype, device=device,
                    solver_device=solver_device, **kw)


def run_all(sets: Sequence[str], dbs: Dict[str, Path], configs: Dict[str, dict], seeds: Iterable[int],
            out_dir: Path, verbose: bool = True, mace_mode: str = "energy-only", mace_dtype: str = "float32",
            device: Optional[str] = None, solver_device: Optional[str] = None, **kw) -> List[Path]:
    """The whole benchmark (case sets x DBs x configurations x seeds) in ONE continuous panel process: the worker
    pool, both parsed DBs and the MACE model are created once and reused for every result file. Configuration
    names become "<db label>-<name>". Returns all result paths (existing ones are reused)."""
    seeds = list(seeds)
    spec_kw = {k: kw.pop(k) for k in _SPEC_KEYS if k in kw}
    specs = []
    for s in sets:
        for label, db in dbs.items():
            cfgs = {f"{label}-{name}": c for name, c in configs.items()}
            specs += matrix_specs(cfgs, seeds, db, out_dir, s, mace_mode=mace_mode, mace_dtype=mace_dtype,
                                  device=device, solver_device=solver_device, **spec_kw)
    return run_plan(specs, out_dir, verbose=verbose, mace_mode=mace_mode, mace_dtype=mace_dtype, device=device,
                    solver_device=solver_device, **kw)


def _valid(case: str, r: dict, donor_atoms: Optional[dict]) -> bool:
    mnd = r.get("n_donors_crystal")
    if mnd is None:
        return True
    n = r.get("n_donor_atoms")
    if n is None and donor_atoms and case in donor_atoms:
        n = donor_atoms[case]
    return (n if n is not None else r.get("n_donors")) == mnd


def load(paths: Iterable[Path], donor_atoms: Optional[dict] = None) -> pd.DataFrame:
    """Tidy table: one row per (set, config, seed, case). `donor_atoms` {case: donor atoms given to the solver} for
    result files older than the n_donor_atoms field (validity check, see _valid)."""
    rows = []
    for p in paths:
        p = Path(p)
        d = json.loads(p.read_text())
        tag, name, s, *_identity = p.stem.split("__")
        for case, r in d["cases"].items():
            rows.append(dict(set=tag.replace("-sub", ""), config=name, seed=int(s[1:]), case=case,
                             group="aqua" if case.startswith("aqua_") else "legacy" if case.startswith("legacy_") else "tmQM",
                             dE=r.get("dE"), absdE=abs(r["dE"]) if r.get("dE") is not None else np.nan,
                             score=r.get("score"), solve_s=r.get("solve_s"), max_md_err=r.get("max_md_err"),
                             n_donors=r.get("n_donors"), n_donors_crystal=r.get("n_donors_crystal"),
                             status=r.get("status"), src_hash=d.get("src_hash"), db=d.get("db"),
                             mace_mode=d.get("mace_mode"), mace_dtype=d.get("mace_dtype"),
                             mace_threads=d.get("mace_threads"),
                             solver_device=d.get("solver_device") or "cpu", mace_device=d.get("mace_device"),
                             energy_s=r.get("energy_s"),
                             **_ensemble_metrics(r),
                             # benchmark input sane: the harness gave the solver the crystal's donor count, in
                             # donor ATOMS (tmQM's MND counts each atom of a haptic ligand; 2026-10-16: comparing
                             # donor groups marked 24 correct dev32 / N48 inputs invalid). Files written before
                             # n_donor_atoms existed: `donor_atoms` ({case: atoms}, see donor_atoms.py) or groups.
                             valid=_valid(case, r, donor_atoms)))
    return pd.DataFrame(rows)


#: rho_lo: the within-ensemble rank correlation restricted to conformers within this many kJ/mol of the ensemble's
#: lowest energy (2026-10-15). The whole-ensemble rho is dominated by separating broken conformers from decent ones.
RHO_LO_WINDOW = 50.0


def _ensemble_metrics(r: dict) -> dict:
    """From a run with JDEV_ENSEMBLE_DE=1 (every returned conformer measured): top1 = #1 within 1 kcal/mol of the
    lowest returned energy; top3 = the lowest is among the 3 best-scored; rho = Spearman(score, -dE) in the ensemble.
    Error budget (2026-10-15): best_dE = the lowest returned energy (the pool floor: what generation produced),
    regret = #1 minus that floor (what ranking lost), so dE(#1) = best_dE + regret; rho_lo = rho among the conformers
    within RHO_LO_WINDOW of the floor (>= 5 of them), where the ranking decision is actually made."""
    e, s = r.get("dE_all"), r.get("scores_all")
    if not e or not s or len(e) != len(s):
        return {}
    e, s = np.asarray(e, float), np.asarray(s, float)
    i1 = int(np.argmax(s))
    out = dict(n_conf=len(e), top1=bool(e[i1] <= e.min() + 4.184), top3=bool(int(np.argmin(e)) in np.argsort(-s)[:3]),
               best_absdE=float(np.abs(e).min()), best_dE=float(e.min()), regret=float(e[i1] - e.min()))
    if len(e) >= 3 and np.ptp(s) > 0 and np.ptp(e) > 0:
        from scipy.stats import spearmanr
        out["rho"] = float(spearmanr(s, -e).correlation)
        lo = e <= e.min() + RHO_LO_WINDOW
        if lo.sum() >= 5 and np.ptp(s[lo]) > 0 and np.ptp(e[lo]) > 0:
            out["rho_lo"] = float(spearmanr(s[lo], -e[lo]).correlation)
    return out


def targets(df: pd.DataFrame) -> pd.DataFrame:
    """The project targets per (set, config): #1 |dE| median (< 100) and count above 200 (none), top-1 share, median
    within-ensemble rho (> 0.6), median solve time (< 20 s). Needs JDEV_ENSEMBLE_DE=1 runs for top1 / rho."""
    g = df.groupby(["set", "config"])
    out = pd.DataFrame({"median_absdE": g["absdE"].median(), "mean_absdE": g["absdE"].mean(),
                        "over_200": g["absdE"].apply(lambda v: int((v > 200).sum())), "runs": g.size(),
                        "solve_s_median": g["solve_s"].median()})
    if "top1" in df:
        out["top1"] = g["top1"].mean()
        out["top3"] = g["top3"].mean()
    if "rho" in df:
        out["rho_median"] = g["rho"].median()
    if "rho_lo" in df:
        out["rho_lo_median"] = g["rho_lo"].median()
    if "best_dE" in df:  # error budget: #1 = floor (generation) + regret (ranking)
        out["floor_median"] = g["best_dE"].median()
        out["regret_median"] = g["regret"].median()
        out["regret_mean"] = g["regret"].mean()
    return out.round(3)


def paired_test(df: pd.DataFrame, a: str, b: str, value: str = "absdE", threshold: float = 10.0,
                n_boot: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Set-level paired comparison of configuration b against a (2026-10-15), per set.

    Per case the seed-mean of (b - a); over cases: mean and median difference, a cluster-bootstrap 95 % CI of the mean
    (cases resampled, seeds kept together), counts better / worse by more than `threshold`, and the minimum detectable
    effect (MDE) of this design: 2.8 x sd(per-case diffs) / sqrt(n) (two-sided alpha 0.05, power 0.8). `verdict` is
    "better" / "worse" only when the CI excludes 0; otherwise "no detectable change" -- which, when the MDE is large,
    means the experiment could not have seen an effect of the size it was meant to find.

    Reference point (two runs of identical code + DB on CUDA, 2026-10-03 vs 10-13): per-run |dE| moved by a median of
    9 kJ/mol, about 1 case in 6 crossed +-10 kJ/mol in each direction, set medians moved by up to 29 kJ/mol, and
    whole-ensemble rho medians by up to 0.07. Compare set medians or better/worse counts only against that."""
    rng = np.random.default_rng(seed)
    m = df[df.config.isin([a, b])].pivot_table(index=["set", "case", "seed"], columns="config", values=value)
    m = m.dropna()
    d = (m[b] - m[a]).groupby(level=[0, 1]).mean()  # per (set, case): mean over seeds
    rows = []
    for st, x in d.groupby(level=0):
        x = x.to_numpy()
        n = len(x)
        bs = np.array([rng.choice(x, n).mean() for _ in range(n_boot)]) if n > 1 else np.array([x.mean()])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        mde = 2.8 * x.std(ddof=1) / np.sqrt(n) if n > 1 else np.inf
        verdict = "better" if hi < 0 else "worse" if lo > 0 else "no detectable change"
        rows.append(dict(set=st, cases=n, mean_diff=x.mean(), ci_lo=lo, ci_hi=hi, median_diff=float(np.median(x)),
                         better=int((x < -threshold).sum()), worse=int((x > threshold).sum()), mde=mde,
                         verdict=verdict))
    return pd.DataFrame(rows).set_index("set").round(2)


def per_case(df: pd.DataFrame, value: str = "absdE") -> pd.DataFrame:
    """mean / sd / min / max over seeds, per case (rows) and configuration (column groups)."""
    g = df.groupby(["group", "case", "config"])[value].agg(["mean", "std", "min", "max", "count"])
    return g.unstack("config").round(1)


def paired_diff(df: pd.DataFrame, a: str, b: str, value: str = "absdE") -> pd.DataFrame:
    """Per case: mean over seeds of (b - a), its seed sd, and whether |mean diff| exceeds 2 x the pooled seed sd.
    A per-case screen only (3 seeds give a poor sd); decide set-level questions with paired_test."""
    m = df.pivot_table(index=["group", "case", "seed"], columns="config", values=value)
    d = (m[b] - m[a]).groupby(level=[0, 1]).agg(["mean", "std", "count"])
    var = df[df.config.isin([a, b])].groupby(["group", "case", "config"])[value].var()
    d["pooled_seed_sd"] = np.sqrt(var.groupby(level=[0, 1]).mean())  # within-configuration seed spread only
    d["clear"] = d["mean"].abs() > 2 * d["pooled_seed_sd"].fillna(np.inf)
    return d.round(1)


def summary(df: pd.DataFrame, value: str = "absdE", valid_only: bool = True) -> pd.DataFrame:
    """Per (set, group, config): median, mean, max, count over cases > 300 and > 100 of the seed-averaged value.
    valid_only: drop cases whose benchmark input is wrong (donor count differs from the crystal; see load())."""
    if valid_only and "valid" in df:
        bad = sorted(df.loc[~df["valid"].astype(bool), "case"].unique())
        if bad:
            print(f"excluded (benchmark donor count != crystal): {bad}")
        df = df[df["valid"].astype(bool)]
    cm = df.groupby(["set", "group", "config", "case"])[value].mean().reset_index()
    return cm.groupby(["set", "group", "config"])[value].agg(
        median="median", mean="mean", max="max", n="count",
        over_300=lambda v: int((v > 300).sum()), over_100=lambda v: int((v > 100).sum())).round(1)
