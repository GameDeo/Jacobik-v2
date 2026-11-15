"""run_refit.py -- the full tmQM + LnQM refit against src, stage by stage (the notebook
notebooks/03_full_refit.ipynb calls the same stage functions and shows their tables).

    python refit/run_refit.py all                 # every stage, in order
    python refit/run_refit.py clean extract       # selected stages
    python refit/run_refit.py fit --workers 12 --root "C:/.../Jacobik v3"

Stages (each writes under <root>/data/refit/ and is skipped when its outputs exist, unless --force):
  audit    oxidation-state label audit (fast, no geometry)
  clean    contact cleaning (data-derived thresholds) + sharding of tmQM X1-X3 and LnQM
  extract  sharded extraction (Q1-Q6, QO, QG, QP, QB) + merge per dataset, benchmark complexes removed
           (benchmark_holdout_codes.json: N48 screen + RING10)
  select   held-out selection (complexes split by CSD-code hash): Q1 ladder variant x pooling relevance r,
           Q2 relevance r, S_orient hold-out; the evaluations are cached (reports/holdout/pool_Q*.csv) and the pick
           re-made on every run (Q1 among the state-keeping variants: Config.q1_require_state)
  fit      the full fit, ONE DB with every entry, adopted and experimental (experimental entries are read only when
           their profile field is on): Q1 / Q2 pooled with the selected (variant, r), Q3-Q5, the Q6 walls (5th-percentile
           form, the benchmark's choice 10-15), S_orient (QO), priors (QG / QP), Q6P tables (PMF + metal 1-3 core
           tables and walls), QV / QVR bond valence (stats/bvs.py), NCI tables (stats/nci.py);
           writes <final_dir>/constraints.json + constraints.meta.json and fit_summary.json
  check    audits: far Q1 components, Q6 walls, DB summary
  install  copy the new DB into src/jacobik/ (backup kept); only with --install

The GFN2 thermal widths (refit/thermal_widths.py: Hessians of a sample of training complexes) are a separate,
expensive measurement; its results are the THERMAL_SD constants in ik/scorer.py.

Configuration: tmQM in data/tmQM, LnQM as exported by refit/export_lnqm.py (data/lnQM/lnqm.h5 ->
data/refit/source/lnQM; stage clean runs the export when that folder is missing); override with --root / --tm / --ln /
--workers or a JSON file (--config) with the same keys as Config.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent


@dataclasses.dataclass
class Config:
    root: str = str(HERE.parent)                 # folder containing src/ and refit/
    tm_dir: str = "data/tmQM"                    # tmQM (xyz/ bo/ tmQM_y.csv), relative to root unless absolute
    ln_dir: str = "data/refit/source/lnQM"       # LnQM exporter output (refit/export_lnqm.py; Angstrom)
    ln_xyz_in_bohr: bool = False
    refit_dir: str = "data/refit"
    final_dir: str = "final"                     # fit / check / install output, relative to refit_dir
    workers: int = max(1, (os.cpu_count() or 2) - 1)
    expand_workers: int = 3                      # quantities expanded at once (pure Python, memory-bound; 16 GB PC)
    shard_size: int = 4000
    tm_splits: tuple = ("X1", "X2", "X3")
    ln_split: str = "LNf"
    q1_variants: tuple = ("mo_state", "cn_state", "m_sub_early", "orig", "no_state", "metal_first")
    q1_require_state: bool = True                # Q1 variant chosen among the state-keeping ones only (owner, 10-15)
    extract_only: tuple = ()                     # shard / dataset names: re-extract only these (--shards), keep the rest
    pool_r: tuple = (4.0, 10.0, 25.0)
    holdout_mod: int = 10
    force: bool = False
    install: bool = False

    # ---- derived paths
    def p(self, rel: str) -> Path:
        q = Path(rel)
        return q if q.is_absolute() else Path(self.root) / q

    @property
    def refit(self) -> Path:
        return self.p(self.refit_dir)

    @property
    def obs(self) -> Path:
        return self.refit / "observations"

    @property
    def hold(self) -> Path:
        return self.refit / "reports" / "holdout"

    @property
    def final(self) -> Path:
        return self.refit / self.final_dir

    @property
    def db(self) -> Path:
        return self.final / "constraints.json"

    @property
    def contacts(self) -> Path:
        return self.refit / "reports" / "contacts"

    @property
    def datasets(self) -> List[str]:
        return list(self.tm_splits) + [self.ln_split]

    def setup(self):
        sys.path.insert(0, str(self.p("src")))
        sys.path.insert(0, str(self.p("refit")))
        os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
        for d in (self.obs, self.hold, self.final, self.contacts):
            d.mkdir(parents=True, exist_ok=True)
        try:
            from rdkit import RDLogger

            RDLogger.DisableLog("rdApp.*")
        except Exception:
            pass


QS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]


def _say(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
def stage_audit(cfg: Config):
    import refit_lib as R

    aud = R.os_label_audit(cfg.p(cfg.tm_dir), splits=cfg.tm_splits)
    (cfg.refit / "reports" / "os_audit.json").write_text(json.dumps(aud, indent=1, default=str))
    _say(f"oxidation-state audit: {aud.get('reasons')}")
    return aud


def stage_clean(cfg: Config):
    import refit_lib as R

    if (cfg.contacts / "shards.json").exists() and not cfg.force:
        _say("clean: shards.json exists -- skipped")
        return json.loads((cfg.contacts / "contact_filter_params.json").read_text())
    tm, ln = cfg.p(cfg.tm_dir), cfg.p(cfg.ln_dir)
    h5 = cfg.p("data/lnQM/lnqm.h5")
    if not (ln / "xyz" / "tmQM_LN.xyz").exists() and h5.exists():
        from export_lnqm import export

        _say(f"clean: exporting LnQM {h5} -> {ln}: {export(h5, ln)}")
    specs =[dict(name=sp, group="TM", src=tm, src_split=sp, out=cfg.refit / "clean" / "tmQM", split=sp, scale=1.0)
             for sp in cfg.tm_splits]
    if ln.exists():
        specs.append(dict(name=cfg.ln_split, group="LN", src=ln, src_split="LN", out=cfg.refit / "clean" / "lnQM",
                          split=cfg.ln_split, scale=(R._BOHR if cfg.ln_xyz_in_bohr else 1.0)))
    else:
        _say(f"clean: no LnQM folder at {ln} -- tmQM only")
    t = time.time()
    thr = R.prepare_clean(specs, cfg.contacts, n_workers=cfg.workers, shard_size=cfg.shard_size)
    _say(f"clean: {len(thr['shards'])} shards in {time.time() - t:.0f}s")
    return thr


def stage_extract(cfg: Config):
    import refit_lib as R

    shards = json.loads((cfg.contacts / "shards.json").read_text())
    datasets = sorted({s["dataset"] for s in shards})
    if all((cfg.obs / f"observations_Q1_{d}.parquet").exists() for d in datasets) and not cfg.force \
            and not cfg.extract_only:
        _say("extract: merged observations exist -- skipped")
        return None
    todo = [s for s in shards if not cfg.extract_only or s["shard"] in cfg.extract_only
            or s["dataset"] in cfg.extract_only]
    t = time.time()
    st = R.extract_shards(todo, cfg.obs / "shards", cfg.workers)
    _say(f"extract: {len(todo)} shards in {time.time() - t:.0f}s")
    prev = cfg.refit / "reports" / "extract_shard_stats.json"
    if cfg.extract_only and prev.exists():  # the other shards' files (and stats) are kept from the previous run
        st = dict(json.loads(prev.read_text()), **st)
    prev.write_text(json.dumps(st, indent=1, default=float))
    excl = R.benchmark_codes()
    stats = {}
    for d in datasets:
        names = [s["shard"] for s in shards if s["dataset"] == d]
        rows = R.merge_shards(cfg.obs / "shards", cfg.obs, d, names, exclude=excl)
        have = [st[n] for n in names if n in st]  # shards without stats: kept from a run before extract_shard_stats
        qs = sorted({q for s in have for q in s["quantity_errors"]})
        stats[d] = {"rows": rows, "n_complexes": sum(s["n_complexes"] for s in have),
                    "shards_with_stats": f"{len(have)}/{len(names)}",
                    "n_errors": sum(s["n_errors"] for s in have),
                    "n_parse_failed": sum(s.get("n_parse_failed", 0) for s in have),
                    "quantity_errors": {q: sum(s["quantity_errors"].get(q, 0) for s in have) for q in qs}}
    (cfg.refit / "reports" / "extract_stats.json").write_text(json.dumps(stats, indent=1))
    _say(f"extract: merged; {len(excl)} benchmark codes excluded")
    return stats


def stage_select(cfg: Config):
    """Q1: ladder variant x r; Q2: r; S_orient hold-out. Writes reports/holdout/selection.json."""
    import pandas as pd
    import refit_lib as R

    out = cfg.hold / "selection.json"
    old = json.loads(out.read_text()) if out.exists() and not cfg.force else {}
    ds = [d for d in cfg.datasets if (cfg.obs / f"observations_Q1_{d}.parquet").exists()]
    sel = {}
    for q, variants in (("Q1", list(cfg.q1_variants)), ("Q2", None)):
        t = time.time()
        csv = cfg.hold / f"pool_{q}.csv"
        if csv.exists() and not cfg.force:  # the held-out evaluation is the expensive part; the pick is re-made
            ev = pd.read_csv(csv)
            ev["variant"] = ev["variant"].astype(object).where(ev["variant"].notna(), None)
        else:
            ev = R.pooled_holdout([cfg.obs / f"observations_{q}_{d}.parquet" for d in ds], q, cfg.hold / f"pool_{q}",
                                  rs=cfg.pool_r, variants=variants, n_workers=cfg.workers, mod=cfg.holdout_mod)
            ev.to_csv(csv, index=False)
        sel[q] = R.pick_pooling(ev, require_state=cfg.q1_require_state)
        _say(f"select {q}: {sel[q]}  ({time.time() - t:.0f}s)")
    if "QO" in old:
        sel["QO"] = old["QO"]
        _say("select QO: cached")
    else:
        qo = R.load_qo(cfg.obs, ds)
        t = time.time()
        evq = R.qo_holdout_eval(qo, mod=cfg.holdout_mod, n_jobs=cfg.workers)
        evq["by_class"].to_csv(cfg.hold / "qo_holdout_by_class.csv")
        sel["QO"] = {k: v for k, v in evq.items() if k != "by_class"}
        _say(f"select QO: {sel['QO'].get('loglik')}  ({time.time() - t:.0f}s)")
    out.write_text(json.dumps(sel, indent=1, default=float))
    return sel


def stage_fit(cfg: Config):
    import refit_lib as R
    from jacobik.stats.aggregator import aggregate
    from jacobik.stats.fitter import fit_all
    from jacobik.stats import pooling as PL

    db = cfg.db
    if db.exists() and not cfg.force:
        _say(f"fit: {db.name} exists -- skipped")
        return json.loads((cfg.final / "fit_summary.json").read_text()) if (cfg.final / "fit_summary.json").exists() else None
    selp = cfg.hold / "selection.json"
    sel = json.loads(selp.read_text()) if selp.exists() else {}
    variant = (sel.get("Q1") or {}).get("variant") or "mo_state"
    pooling = {"Q1": (sel.get("Q1") or {}).get("r", PL.R_DEFAULT), "Q2": (sel.get("Q2") or {}).get("r", PL.R_DEFAULT)}
    if not sel:
        _say(f"fit: no selection.json -- defaults: variant {variant}, r {pooling}")
    ds = [d for d in cfg.datasets if (cfg.obs / f"observations_Q1_{d}.parquet").exists()]
    for q in QS:
        for d in ds:
            shutil.copy(cfg.obs / f"observations_{q}_{d}.parquet", cfg.final / f"observations_{q}_{d}.parquet")
    t = time.time()
    _say(f"aggregate: {aggregate(cfg.final, ds, QS)}")
    _say(f"expand: {R.expand_parallel(cfg.final, cfg.final / 'exp', QS, cfg.expand_workers)}  ({time.time() - t:.0f}s)")
    t = time.time()
    summ = fit_all(cfg.final / "exp", db, quantities=QS, n_workers=cfg.workers, pooled=pooling, q1_variant=variant,
                   q6_model="p5")
    _say(f"fit Q1-Q6: {time.time() - t:.0f}s")
    t = time.time()
    qo = R.load_qo(cfg.obs, ds)
    qo_entries = R.qo_fit_full(qo, n_jobs=cfg.workers)
    (cfg.final / "orientation_fits.json").write_text(json.dumps(qo_entries))
    R.add_entries_to_db(db, qo_entries, "QO")
    pri = R.fit_priors(cfg.obs, ds)
    for q in ("QG", "QP"):
        R.add_entries_to_db(db, {k: v for k, v in pri.items() if k.startswith(q + ":")}, q)
    _say(f"fit QO / QG / QP: {len(qo_entries)} / {sum(k.startswith('QG:') for k in pri)} / "
         f"{sum(k.startswith('QP:') for k in pri)} keys  ({time.time() - t:.0f}s)")
    _fit_tables(cfg, db, ds)
    R.write_meta(cfg.final, variant, pooling, extra={"q6_model": "p5_hp"}, name="constraints")
    summ = {"Q1-Q6": summ, "variant": variant, "pooling": pooling, "datasets": ds}
    (cfg.final / "fit_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    _say(f"DB: {R.db_summary(db).to_string()}")
    return summ


def _fit_tables(cfg: Config, db: Path, ds: List[str]) -> None:
    """The table-type entries, each one fit over every dataset: Q6P (PMF tables from q6hist_<ds>.npz; metal 1-3 core
    tables and walls from q6hist_m13_<ds>.npz), QV / QVR (bond valence, from the Q1 observations), NCI (from
    ncihist_<ds>.npz)."""
    import refit_lib as R
    from jacobik.stats import bvs, nci, q6pmf

    def hists(prefix):
        return [cfg.obs / f"{prefix}_{d}.npz" for d in ds if (cfg.obs / f"{prefix}_{d}.npz").exists()]

    q6p = {}  # one Q6P write: add_entries_to_db replaces every entry of the quantity
    hp = hists("q6hist")
    if hp:
        pmf = q6pmf.fit_tables(q6pmf.load_hist(hp))
        q6p.update(pmf)
        (cfg.final / "q6pmf_summary.json").write_text(json.dumps(
            {k: {f: v[f] for f in ("n", "n_key", "r0", "core_b", "core_kappa")} for k, v in pmf.items()}, indent=1))
        _say(f"fit Q6P: {len(pmf)} tables from {[p.name for p in hp]}")
    hm = hists("q6hist_m13")
    if hm:
        h = q6pmf.load_hist(hm)
        tabs = {k: v for k, v in q6pmf.fit_tables(h).items() if any(f"'{c}'" in k for c in q6pmf.M13_CLASSES)}
        tabs.update(q6pmf.fit_m13_walls(h))
        q6p.update(tabs)
        _say(f"fit Q6P metal 1-3: {len(tabs)} tables and walls")
    if q6p:
        R.add_entries_to_db(db, q6p, "Q6P")
    qv = bvs.fit_entries(cfg.obs, ds)
    R.add_entries_to_db(db, {k: v for k, v in qv.items() if k.startswith("QV:")}, "QV")
    R.add_entries_to_db(db, {k: v for k, v in qv.items() if k.startswith("QVR:")}, "QVR")
    _say(f"fit QV / QVR: {sum(k.startswith('QV:') for k in qv)} levels / {sum(k.startswith('QVR:') for k in qv)} R0")
    hn = hists("ncihist")
    if hn:
        tabs = nci.fit_tables(nci.load_hist(hn))
        R.add_entries_to_db(db, tabs, "NCI")
        _say(f"fit NCI: {len(tabs)} tables")


def stage_check(cfg: Config):
    import pandas as pd
    import refit_lib as R

    db = cfg.db
    far = R.far_component_audit(db, far_A=3.0)
    raw = json.loads(db.read_text())
    q6 = pd.DataFrame({k[3:]: {"mu": v["mu"], "sigma": v["sigma"], "n": v["n"]} for k, v in raw.items()
                       if k.startswith("Q6:")}).T.sort_values("n", ascending=False)
    rep = {"far_components": int(len(far)), "db": R.db_summary(db).reset_index().to_dict("records"),
           "q6_top": q6.head(40).round(3).reset_index().to_dict("records")}
    (cfg.refit / "reports" / "check.json").write_text(json.dumps(rep, indent=1, default=float))
    _say(f"check: {len(far)} far Q1 components; Q6 walls (largest keys):\n{q6.head(15).round(3).to_string()}")
    return rep


def stage_install(cfg: Config):
    if not cfg.install:
        _say("install: not requested (--install)")
        return False
    tgt = cfg.p("src") / "jacobik" / "constraints.json"
    bak = tgt.with_name("constraints.pre_refit.json")
    if tgt.exists() and not bak.exists():
        shutil.copy(tgt, bak)
    shutil.copy(cfg.db, tgt)
    shutil.copy(cfg.db.with_name("constraints.meta.json"), tgt.with_name("constraints.meta.json"))
    _say(f"installed {tgt} (backup {bak.name})")
    return True


STAGES = {"audit": stage_audit, "clean": stage_clean, "extract": stage_extract, "select": stage_select,
          "fit": stage_fit, "check": stage_check, "install": stage_install}
ORDER = ["audit", "clean", "extract", "select", "fit", "check", "install"]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=ORDER + ["all"])
    ap.add_argument("--config", help="JSON file with Config fields")
    ap.add_argument("--root"); ap.add_argument("--tm"); ap.add_argument("--ln")
    ap.add_argument("--workers", type=int); ap.add_argument("--force", action="store_true")
    ap.add_argument("--install", action="store_true")
    ap.add_argument("--final", help="output folder of fit / check / install, relative to the refit dir (default final)")
    ap.add_argument("--shards", nargs="*", default=None,
                    help="extract: only these shards or datasets (e.g. LNf X1s000); the other shard files are kept")
    a = ap.parse_args(argv)
    cfg = Config(**(json.loads(Path(a.config).read_text()) if a.config else {}))
    for k, v in (("root", a.root), ("tm_dir", a.tm), ("ln_dir", a.ln), ("workers", a.workers), ("final_dir", a.final)):
        if v is not None:
            setattr(cfg, k, v)
    cfg.force, cfg.install = a.force, a.install
    if a.shards:
        cfg.extract_only = tuple(a.shards)
    cfg.setup()
    todo = ORDER if "all" in a.stages else [s for s in ORDER if s in a.stages]
    for s in todo:
        _say(f"=== {s} ===")
        STAGES[s](cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
