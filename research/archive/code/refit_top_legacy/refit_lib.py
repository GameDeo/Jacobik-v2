"""refit_lib.py -- tooling for the tmQM + LnQM refit against `src` (all pipeline changes are NATIVE in src;
nothing is monkeypatched).  This file only contains: contact cleaning (data-derived thresholds), parallel sharded extraction,
benchmark hold-out exclusion, held-out selection of the Q1 ladder variant and the pooling relevance factor, the S_orient
and hypothesis-prior (QG / QP) fits, and audits (oxidation-state labels, far Q1 components, Shannon fallback hold-out).
The stage-by-stage driver is run_refit.py (CLI) / notebooks/03_full_refit.ipynb (same functions).
"""
from __future__ import annotations

import ast
import json
import re
import logging
import math
import os
import random
import zlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger("refit")

LN = frozenset("La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split())
SEED = 20260920

def holdout_mask(codes: pd.Series, mod: int = 10) -> pd.Series:
    """Held-out complexes by hash of the independent-structure id (LnQM siblings of one ligand set fall on the same
    side; mixture.structure_id), so no ligand set is both trained on and scored."""
    from jacobik.stats.mixture import structure_ids

    sid = pd.Series(structure_ids(codes.to_numpy()), index=codes.index)
    return sid.map(lambda c: zlib.crc32(c.encode()) % mod == 0)


def split_observations(obs_paths: Sequence[Path], q: str, out_dir: Path, mod: int = 10) -> Tuple[int, int]:
    """Write observations_<q>.parquet (train) and heldout_<q>.parquet from per-dataset observation files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_parquet(p) for p in obs_paths], ignore_index=True)
    h = holdout_mask(df["csd_code"], mod)
    df[~h].to_parquet(out_dir / f"observations_{q}.parquet", index=False)
    df[h].to_parquet(out_dir / f"heldout_{q}.parquet", index=False)
    return int((~h).sum()), int(h.sum())



def _mix_logpdf(e: dict, x: float, sigma_floor: float) -> float:
    w, mu, v = np.asarray(e["weights"]), np.asarray(e["means"]), np.maximum(np.asarray(e["vars"]), sigma_floor ** 2)
    return float(np.log(np.sum(w * np.exp(-0.5 * (x - mu) ** 2 / v) / np.sqrt(2 * math.pi * v)) + 1e-300))


def evaluate_q1_ladders(db_json: Path, heldout_parquet: Path, variants: Optional[Iterable[str]] = None,
                        n_min_values=(30, 100), sigma_floor: float = 0.1, max_rows: Optional[int] = 200_000) -> pd.DataFrame:
    """For each held-out Q1 observation, resolve its key under each ladder variant / n_min against a DB fitted on TRAIN
    complexes only (through the SAME ladders module the runtime uses); report mean log-lik (sigma floored), MAE of the
    resolved mixture mean, and resolved-level shares."""
    from jacobik.stats import ladders as L
    db = {k[3:]: v for k, v in json.load(open(db_json)).items() if k.startswith("Q1:")}
    ho = pd.read_parquet(heldout_parquet)
    if max_rows and len(ho) > max_rows:
        ho = ho.sample(max_rows, random_state=0)
    keys = [ast.literal_eval(k) for k in ho["key"]]
    vals = ho["value"].to_numpy()
    grp = np.array(["Ln" if k[0] in LN else "TM" for k in keys])
    rows = []
    for var in (variants or L.Q1_VARIANTS):
        for nm in n_min_values:
            ll = np.full(len(keys), np.nan); ae = np.full(len(keys), np.nan); lvl = [""] * len(keys)
            for i, k in enumerate(keys):
                cands = L.q1_candidates(k, var)
                names = {kk: n for n, kk in L.q1_named_levels(k)[::-1]}
                best = None
                for kk in cands:
                    e = db.get(repr(kk))
                    if e is None:
                        continue
                    best = (kk, e)                       # last found (mirrors StatsLookup._resolve)
                    if e["n"] >= nm:
                        break
                if best is None:
                    continue
                kk, e = best
                lvl[i] = names.get(kk, "?")
                ll[i] = _mix_logpdf(e, vals[i], sigma_floor)
                ae[i] = abs(float(np.dot(e["weights"], e["means"])) - vals[i])
            for g in ("Ln", "TM"):
                s = (grp == g) & ~np.isnan(ll)
                if s.sum() == 0:
                    continue
                share = pd.Series(np.array(lvl)[s]).value_counts(normalize=True).round(3).to_dict()
                rows.append(dict(variant=var, n_min=nm, group=g, n=int(s.sum()), unresolved=int(((grp == g) & np.isnan(ll)).sum()),
                                 mean_loglik=ll[s].mean(), MAE_mean=ae[s].mean(), levels=share))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------
# Data preparation / extra splits
# ----------------------------------------------------------------------------------------------------------------
_BOHR = 0.529177210903
def _blocks_xyz(p, lo: int = 0, hi: Optional[int] = None):
    """Yield (comment, atoms) for structures lo <= i < hi; structures before lo are skipped without parsing."""
    idx = 0
    with open(p) as f:
        while True:
            l = f.readline()
            if not l:
                return
            if not l.strip():
                continue          # tmQM xyz has a blank line between structures
            n = int(l)
            if hi is not None and idx >= hi:
                return
            if idx < lo:
                for _ in range(n + 1):
                    f.readline()
                idx += 1
                continue
            c = f.readline()
            yield c, [f.readline().split() for _ in range(n)]
            idx += 1


def _blocks_bo(p, lo: int = 0, hi: Optional[int] = None):
    """Yield BO blocks (list of lines) for structures lo <= i < hi."""
    cur = None
    idx = -1
    with open(p) as f:
        for l in f:
            l = l.rstrip("\n")
            if l.startswith("CSD_code"):
                if cur:
                    yield cur
                    cur = None
                idx += 1
                if hi is not None and idx >= hi:
                    return
                cur = [l] if idx >= lo else None
            elif not l.strip():
                if cur:
                    yield cur
                    cur = None
            elif cur is not None:
                cur.append(l)
        if cur:
            yield cur


# ----------------------------------------------------------------------------------------------------------------
# Contact cleaning: which metal-X Mayer contacts are real first-shell donors?
#
# Two artifacts of a Mayer-BO >= 0.08 "bond" definition are removed, both with thresholds DERIVED FROM THE DATA
# (no hand-typed numbers; every derived value and its sensitivity is printed and saved):
#   through_bond  M-Y-X with Y a metal-bound atom, X covalently bonded to Y (BO >= 0.5), BO(M,Y) > BO(M,X): the M...X
#                 "bond" is a 1,3 through-bond term (carbonyl O of M-C#O, cyanide N, ...), not a donor.  These form
#                 a population at M-Y-X ~ 170-180 deg, well separated from eta/chelate arrangements (~70-80 deg);
#                 the cut is the density minimum on the low side of the linear population.
#   second_sphere per (dataset group, element of X): a distance beyond the valley between the first-shell peak and a
#                 separate second-sphere hump in the M-X distance histogram.  Applied ONLY where such a valley
#                 exists (valley < 0.5 x the smaller flanking peak); otherwise no cutoff is applied for that element.
# Nothing is deleted from the record: every dropped contact goes to dropped_contacts_<split>.parquet with its reason,
# so a future quantity (outer-sphere H bonds, weak contacts, ...) can be modelled without re-extraction.
# ----------------------------------------------------------------------------------------------------------------
_TM = frozenset("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg La".split())
METALS = _TM | LN
BO_FLOOR = 0.08          # parser.BO_FLOOR
BOND_BO_MIN = 0.5        # X-Y counts as a covalent bond for the through-bond test

# Distance distributions are pooled by periodic row, not by one undifferentiated
# transition-metal class.  This retains enough contacts for robust valley finding
# while avoiding the 3d/4d/5d bond-length smearing that obscures a real second
# population (especially for O donors).  Lanthanides remain one series.
_D3 = frozenset("Sc Ti V Cr Mn Fe Co Ni Cu Zn".split())
_D4 = frozenset("Y Zr Nb Mo Tc Ru Rh Pd Ag Cd".split())
_D5 = frozenset("Hf Ta W Re Os Ir Pt Au Hg".split())


def metal_distance_class(metal: str) -> str:
    """Data-pooling class for first-shell distance distributions."""
    if metal in LN:
        return "Ln"
    if metal in _D3:
        return "3d"
    if metal in _D4:
        return "4d"
    if metal in _D5:
        return "5d"
    return "TM"


def _angle(a, b, c):
    u, v = a - b, c - b
    return float(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12), -1, 1))))


def _parse_bo_block(bl):
    """BO block lines -> (csd_code, {i: {j: bo}}); indices 0-based."""
    m = re.search(r"CSD_code\s*=\s*(\S+)", bl[0])
    nb: Dict[int, Dict[int, float]] = {}
    for line in bl[1:]:
        p = line.split()
        if len(p) < 3:
            continue
        i = int(p[0]) - 1
        d = nb.setdefault(i, {})
        for k in range(3, len(p) - 2, 3):
            d[int(p[k + 1]) - 1] = float(p[k + 2])
    return (m.group(1) if m else "?"), nb


def _metal_contacts(sym, X, nb, m):
    """List of dicts for every metal contact with BO >= BO_FLOOR; through-bond partner y/angle when applicable."""
    cont = {j: b for j, b in nb.get(m, {}).items() if b >= BO_FLOOR and j != m}
    out = []
    for x, b in cont.items():
        best = None
        for y, by in cont.items():
            if y != x and by > b and nb.get(x, {}).get(y, 0.0) >= BOND_BO_MIN and (best is None or by > best[1]):
                best = (y, by)
        rec = dict(x=x, el=sym[x], dist=float(np.linalg.norm(X[m] - X[x])), bo=b, y=-1, y_el="", angle=np.nan)
        if best is not None:
            rec.update(y=best[0], y_el=sym[best[0]], angle=_angle(X[m], X[best[0]], X[x]))
        out.append(rec)
    return out


def _iter_complexes(src_dir: Path, src_split: str, scale: float = 1.0, limit: Optional[int] = None, stride: int = 1,
                    start: int = 0, stop: Optional[int] = None):
    if limit is not None:
        stop = limit if stop is None else min(stop, limit)
    for n, ((cm, at), bl) in enumerate(zip(_blocks_xyz(Path(src_dir) / "xyz" / f"tmQM_{src_split}.xyz", start, stop),
                                            _blocks_bo(Path(src_dir) / "bo" / f"tmQM_{src_split}.BO", start, stop)), start):
        if stride > 1 and n % stride:
            continue
        sym = [a[0] for a in at]
        m = next((i for i, e in enumerate(sym) if e in METALS), None)
        if m is None:
            continue
        X = np.array([[float(v) * scale for v in a[1:4]] for a in at])
        code, nb = _parse_bo_block(bl)
        yield n, cm, at, sym, X, bl, code, nb, m


def _smooth(h, sigma_bins):
    r = int(4 * sigma_bins) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_bins) ** 2)
    return np.convolve(h, k / k.sum(), mode="same")


def derive_angle_theta(angles, fallback: float = 150.0, bin_deg: float = 5.0):
    """Density minimum on the low side of the linear (~180 deg) M-Y-X population.  Returns (theta, info)."""
    a = np.asarray(angles, float)
    a = a[np.isfinite(a)]
    edges = np.arange(0, 180 + bin_deg, bin_deg)
    h, _ = np.histogram(a, edges)
    c = edges[:-1] + bin_deg / 2
    sm = _smooth(h.astype(float), 1.0)
    hi = c >= 150
    info = dict(n=int(len(a)), method="valley")
    if len(a) < 200 or sm[hi].max() <= 0:
        info.update(method="fallback:too_few", theta=fallback)
        return fallback, info
    k_hi = int(np.flatnonzero(hi)[np.argmax(sm[hi])])
    rng = np.flatnonzero((c >= 125) & (c <= c[k_hi] - 10))
    if len(rng) == 0:
        info.update(method="fallback:no_range", theta=fallback)
        return fallback, info
    k_min = int(rng[np.argmin(sm[rng])])
    lo_peak = sm[(c >= 40) & (c < 115)].max()
    ok = sm[k_min] <= 0.5 * min(sm[k_hi], lo_peak)
    theta = float(c[k_min]) if ok else fallback
    info.update(theta=theta, linear_mode=float(c[k_hi]), min_density=float(sm[k_min]), linear_peak=float(sm[k_hi]),
                eta_peak=float(lo_peak), method="valley" if ok else "fallback:weak_valley")
    return theta, info


def derive_gap_cutoff(d, bin_A: float = 0.05, lo: float = 1.4, hi: float = 6.0, min_n: int = 200, min_tail: int = 30,
                      depth: float = 0.5, smooth_A: float = 0.15):
    """Valley between the first-shell peak and a separate second-sphere hump in a metal-X distance histogram.
    The hump is the highest local maximum beyond first-peak + 0.6 A; the valley is the density minimum between the
    two, and is accepted only if it is < `depth` x the smaller of the two peaks and >= `min_tail` contacts lie beyond
    it.  Returns (cutoff or None, info).  None = no separated second population -> keep every contact."""
    d = np.asarray(d, float)
    d = d[(d >= lo) & (d < hi)]
    if len(d) < min_n:
        return None, dict(n=int(len(d)), reason="too_few_contacts")
    edges = np.arange(lo, hi + bin_A, bin_A)
    h, _ = np.histogram(d, edges)
    c = edges[:-1] + bin_A / 2
    sm = _smooth(h.astype(float), smooth_A / bin_A)
    k1 = int(np.argmax(sm[c < 3.5]))
    peak1 = sm[k1]
    start = k1 + int(round(0.6 / bin_A))
    cand = [i for i in range(start, len(sm) - 1) if sm[i] >= sm[i - 1] and sm[i] > sm[i + 1]]
    if not cand:
        return None, dict(n=int(len(d)), first_peak=float(c[k1]), reason="no_second_population")
    j2 = max(cand, key=lambda i: sm[i])
    v = k1 + int(np.argmin(sm[k1:j2 + 1]))
    peak2 = sm[j2]
    if sm[v] <= depth * min(peak1, peak2) and h[v:].sum() >= min_tail:
        return float(c[v]), dict(n=int(len(d)), first_peak=float(c[k1]), valley=float(c[v]), hump=float(c[j2]),
                                 valley_density=float(sm[v]), peak1=float(peak1), peak2=float(peak2), tail=int(h[v:].sum()), reason="valley")
    return None, dict(n=int(len(d)), first_peak=float(c[k1]), hump=float(c[j2]), reason="no_second_population")


def scan_contacts(specs: Sequence[dict], max_complexes: int = 12000, limit: Optional[int] = None) -> pd.DataFrame:
    """Sample contacts for period-row × donor-element cutoff derivation."""
    rows = []
    for sp in specs:
        n_tot = sp.get("n_complexes")
        stride = 1 if not n_tot else max(1, int(n_tot) // max_complexes)
        for n, cm, at, sym, X, bl, code, nb, m in _iter_complexes(sp["src"], sp["src_split"], sp.get("scale", 1.0), limit, stride):
            for r in _metal_contacts(sym, X, nb, m):
                rows.append((metal_distance_class(sym[m]), sp["name"], sym[m], r["el"], r["dist"], r["bo"], r["angle"]))
    return pd.DataFrame(rows, columns=["metal_class", "dataset", "metal", "el", "dist", "bo", "angle"])


def count_complexes(src_dir: Path, src_split: str) -> int:
    n = 0
    with open(Path(src_dir) / "xyz" / f"tmQM_{src_split}.xyz") as f:
        while True:
            l = f.readline()
            if not l:
                break
            if not l.strip():
                continue
            k = int(l)
            for _ in range(k + 1):
                f.readline()
            n += 1
    return n


def derive_thresholds(scan: pd.DataFrame, theta_fallback: float = 150.0) -> dict:
    theta, tinfo = derive_angle_theta(scan["angle"].to_numpy(), fallback=theta_fallback)
    nt = scan[~(scan["angle"] >= theta)]        # contacts that survive the through-bond rule
    cuts, cinfo = {}, {}
    for (metal_class, el), s in nt.groupby(["metal_class", "el"]):
        if el in METALS or el == "H":       # H: the mapping only treats BO >= 0.75 H as donors (hydrides)
            continue
        c, info = derive_gap_cutoff(s["dist"].to_numpy())
        cinfo[f"{metal_class}|{el}"] = info
        if c is not None:
            cuts[f"{metal_class}|{el}"] = c
    # sensitivity (counts of contacts flagged in the sample)
    sens_theta = {t: int((scan["angle"] >= t).sum()) for t in (130, 140, 150, 160, 170)}
    sens_cut = {}
    for k, c in cuts.items():
        metal_class, el = k.split("|")
        s = nt[(nt["metal_class"] == metal_class) & (nt["el"] == el)]["dist"]
        sens_cut[k] = {f"{dd:+.1f}": int((s > c + dd).sum()) for dd in (-0.2, -0.1, 0.0, 0.1, 0.2)}
    return dict(theta=theta, theta_info=tinfo, cutoffs=cuts, cutoff_info=cinfo, sens_theta=sens_theta, sens_cut=sens_cut)


def clean_dataset(spec: dict, thr: dict, out_dir: Path, limit: Optional[int] = None, start: int = 0,
                  stop: Optional[int] = None, out_split: Optional[str] = None) -> pd.DataFrame:
    """Write a cleaned copy of complexes [start, stop) of one dataset (xyz in Angstrom, BO with flagged contacts
    removed) as split `out_split`, and return the dropped contacts.  Only the donor DEFINITION changes; no
    coordinate value is altered (LnQM Bohr -> A is a unit change)."""
    out_dir = Path(out_dir)
    (out_dir / "xyz").mkdir(parents=True, exist_ok=True)
    (out_dir / "bo").mkdir(exist_ok=True)
    scale = spec.get("scale", 1.0)
    theta, cuts = thr["theta"], thr["cutoffs"]
    dropped = []
    n_kept = 0
    out_split = out_split or spec["split"]
    xyz_out = out_dir / "xyz" / f"tmQM_{out_split}.xyz"
    bo_out = out_dir / "bo" / f"tmQM_{out_split}.BO"
    with open(xyz_out, "w") as fx, open(bo_out, "w") as fb:
        for n, cm, at, sym, X, bl, code, nb, m in _iter_complexes(spec["src"], spec["src_split"], scale, limit, 1, start, stop):
            fx.write(f"{len(at)}\n{cm}" + "".join(f"{a} {x[0]:.6f} {x[1]:.6f} {x[2]:.6f}\n" for a, x in zip(sym, X)))
            drop = set()
            for r in _metal_contacts(sym, X, nb, m):
                reason = None
                if np.isfinite(r["angle"]) and r["angle"] >= theta:
                    reason = "through_bond"
                else:
                    c = cuts.get(f"{metal_distance_class(sym[m])}|{r['el']}")
                    if c is not None and r["dist"] > c:
                        reason = "second_sphere"
                if reason:
                    drop.add(r["x"])
                    dropped.append(dict(dataset=spec["name"], csd_code=code, metal=sym[m], metal_idx=m, x_idx=r["x"], x_el=r["el"],
                                        dist=r["dist"], bo=r["bo"], reason=reason, y_idx=r["y"], y_el=r["y_el"], angle=r["angle"]))
                else:
                    n_kept += 1
            fb.write(bl[0] + "\n")
            for line in bl[1:]:
                p = line.split()
                if len(p) < 3:
                    fb.write(line + "\n")
                    continue
                i = int(p[0]) - 1
                out = p[:3]
                for k in range(3, len(p) - 2, 3):
                    j = int(p[k + 1]) - 1
                    if (i == m and j in drop) or (j == m and i in drop):
                        continue
                    out += p[k:k + 3]
                fb.write(" ".join(out) + "\n")
            fb.write("\n")
    df = pd.DataFrame(dropped, columns=["dataset", "csd_code", "metal", "metal_idx", "x_idx", "x_el", "dist", "bo", "reason",
                                         "y_idx", "y_el", "angle"])
    df.attrs["n_kept"] = n_kept
    return df


def _scan_task(args):
    sp, max_complexes, limit = args
    return scan_contacts([sp], max_complexes, limit)


def _clean_task(args):
    spec, thr, shard, lo, hi, report_dir = args
    d = clean_dataset(spec, thr, spec["out"], None, lo, hi, out_split=shard)
    d.to_parquet(Path(report_dir) / f"dropped_contacts_{shard}.parquet", index=False)
    return dict(dataset=spec["name"], shard=shard, kept=d.attrs["n_kept"], **{f"dropped_{k}": int(v) for k, v in d.groupby("reason").size().items()})


def _loky_unordered(func, tasks: Sequence, n_workers: int):
    """Yield task results from joblib's resilient, Windows-safe process pool.

    Cleaning and extraction are file-oriented tasks, so workers receive only
    small task descriptors and independently read/write their shard files.
    ``loky`` is preferable to ``ProcessPoolExecutor`` here: it bounds queued
    work, streams completed results, and avoids the standard Windows executor
    shutdown/IPC hang seen with large refit runs.
    """
    from joblib import Parallel, delayed, parallel_config

    if not tasks:
        return
    worker_count = min(max(1, n_workers), len(tasks))
    with parallel_config(backend="loky", inner_max_num_threads=1):
        results = Parallel(
            n_jobs=worker_count,
            return_as="generator_unordered",
            pre_dispatch="2 * n_jobs",
        )(delayed(func)(task) for task in tasks)
        yield from results


def prepare_clean(specs: Sequence[dict], report_dir: Path, max_scan: int = 12000, limit: Optional[int] = None,
                  n_workers: Optional[int] = None, shard_size: int = 4000) -> dict:
    """specs: dicts with name, group ('TM'|'LN'), src, src_split, out, split, scale (Bohr->A for raw LnQM; else 1).
    Parallel: (1) sampled scan per dataset, (2) thresholds derived once in the parent, (3) cleaned copies written in
    shards of `shard_size` complexes (split names `<split>s000`, ...) so extraction can be parallel too.  Writes
    contact_filter_params.json and shards.json to report_dir."""
    import shutil
    n_workers = n_workers or max(1, (os.cpu_count() or 2))
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    specs = [dict(s) for s in specs]
    for s in specs:
        s["n_complexes"] = count_complexes(s["src"], s["src_split"])
        if limit is not None:
            s["n_complexes"] = min(s["n_complexes"], limit)
        Path(s["out"]).mkdir(parents=True, exist_ok=True)
        y = Path(s["src"]) / "tmQM_y.csv"
        if y.exists() and not (Path(s["out"]) / "tmQM_y.csv").exists():
            shutil.copy(y, Path(s["out"]) / "tmQM_y.csv")
    scans = list(_loky_unordered(_scan_task, [(sp, max_scan, limit) for sp in specs], n_workers))
    scan = pd.concat(scans, ignore_index=True)
    thr = derive_thresholds(scan)
    (report_dir / "contact_filter_params.json").write_text(json.dumps(thr, indent=2, default=float))
    tasks, shards = [], []
    for s in specs:
        for k, lo in enumerate(range(0, s["n_complexes"], shard_size)):
            name = f"{s['split']}s{k:03d}"
            tasks.append((s, thr, name, lo, min(lo + shard_size, s["n_complexes"]), str(report_dir)))
            shards.append(dict(dataset=s["split"], shard=name, dir=str(s["out"])))
    summary = []
    for i, result in enumerate(_loky_unordered(_clean_task, tasks, n_workers), 1):
        summary.append(result)
        if i % 5 == 0 or i == len(tasks):
            print(f"  cleaned {i}/{len(tasks)} shards", flush=True)
    (report_dir / "shards.json").write_text(json.dumps(shards, indent=1))
    thr["summary"] = pd.DataFrame(summary).fillna(0).groupby("dataset").sum(numeric_only=True)
    thr["scan_rows"] = len(scan)
    thr["shards"] = shards
    return thr


# ----------------------------------------------------------------------------------------------------------------
# Parallel extraction over shards (src is imported natively in every worker; nothing to re-apply)
# ----------------------------------------------------------------------------------------------------------------
def _extract_task(args):
    data_dir, shard, obs_dir, hist_exclude, hist_weight = args
    import warnings
    warnings.filterwarnings("ignore")
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    from jacobik.stats.extractor import extract_split
    # the Q6 PMF histograms cannot be filtered after extraction: benchmark codes are left out here (merge_shards drops
    # them from every other quantity)
    return extract_split(data_dir, shard, obs_dir, progress=False, q6_hist_exclude=set(hist_exclude),
                         q6_hist_weight=hist_weight)


def _extract_task_with_shard(item):
    """Return shard metadata with its extraction result for unordered loky use."""
    sh, task = item
    return sh, _extract_task(task)


def sibling_weights(data_dir: Path) -> Dict[str, float]:
    """{code: 1 / siblings} for codes that share an independent structure (LnQM ligand sets; mixture.structure_id),
    from the dataset's tmQM_y.csv; empty for tmQM."""
    from collections import Counter
    from jacobik.stats.mixture import structure_id

    p = Path(data_dir) / "tmQM_y.csv"
    if not p.exists():
        return {}
    codes = pd.read_csv(p, sep=";", usecols=["CSD_code"])["CSD_code"].astype(str).tolist()
    sid = {c: structure_id(c) for c in codes}
    n = Counter(sid.values())
    return {c: 1.0 / n[s] for c, s in sid.items() if n[s] > 1}


def extract_shards(shards: Sequence[dict], obs_dir: Path, n_workers: Optional[int] = None) -> Dict[str, dict]:
    n_workers = n_workers or max(1, (os.cpu_count() or 2))
    obs_dir = Path(obs_dir)
    obs_dir.mkdir(parents=True, exist_ok=True)
    excl = benchmark_codes()
    wts = {d: sibling_weights(d) for d in sorted({sh["dir"] for sh in shards})}
    out = {}
    task_shards = [(sh, (sh["dir"], sh["shard"], str(obs_dir), excl, wts[sh["dir"]])) for sh in shards]
    for i, (sh, st) in enumerate(_loky_unordered(_extract_task_with_shard, task_shards, n_workers), 1):
        out[sh["shard"]] = st
        print(f"  extracted {i}/{len(task_shards)} {sh['shard']}: {st['n_complexes']} complexes, {st['ms_per_mol']:.0f} ms/mol, "
              f"parse_failed={st.get('n_parse_failed', 0)} errors={st['n_errors']} "
              f"quantity_errors={st['quantity_errors']}", flush=True)
    bad = {k: v for k, v in out.items()
           if v["n_errors"] or v.get("n_parse_failed") or any(v["quantity_errors"].values())}
    if bad:
        print("WARNING: extraction reported errors in shards:",
              {k: (v.get("n_parse_failed", 0), v["n_errors"], v["quantity_errors"]) for k, v in bad.items()})
    return out


def _expand_one(args):
    in_dir, out_dir, q = args
    from jacobik.stats.expander import expand_quantity
    return q, expand_quantity(in_dir, out_dir, quantities=[q])


def expand_parallel(in_dir: Path, out_dir: Path, quantities: Sequence[str], n_workers: int = 3) -> Dict[str, dict]:
    """expander.expand_quantity, one process per quantity (independent files; each is a single-threaded Python loop
    over its rows). Same output as the serial call: each quantity's reservoir is seeded on its own."""
    order = sorted(quantities, key=lambda q: -(Path(in_dir) / f"observations_{q}.parquet").stat().st_size
                   if (Path(in_dir) / f"observations_{q}.parquet").exists() else 0)
    out = {}
    for q, st in _loky_unordered(_expand_one, [(str(in_dir), str(out_dir), q) for q in order], n_workers):
        out.update(st)
        print(f"  expanded {q}: {st.get(q)}", flush=True)
    return out


def merge_shards(shard_dir: Path, out_dir: Path, dataset: str, shards: Sequence[str],
                 quantities=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "QO", "QG", "QP", "QB"), seed: int = SEED,
                 exclude: Optional[Iterable[str]] = None) -> Dict[str, int]:
    """Concatenate shard observation files into observations_<Q>_<dataset>.parquet.  Q6 is re-capped per element-pair key
    with the extractor's own caps, so the result matches a single-process extraction of the whole dataset (the cap is per split).
    ``exclude``: CSD codes dropped from every quantity (default: the benchmark hold-out list, benchmark_codes()).
    QO / QG / QP / QB are optional (older shard sets lack them)."""
    excl = set(benchmark_codes() if exclude is None else exclude)
    from jacobik.stats.extractor import Q6_DEFAULT_CAP, Q6_GLOBAL_CAP
    shard_dir, out_dir = Path(shard_dir), Path(out_dir)
    rng = np.random.default_rng(seed + zlib.crc32(dataset.encode()))
    rows = {}
    for q in quantities:
        paths = [shard_dir / f"observations_{q}_{sh}.parquet" for sh in sorted(shards)]
        if q in ("QO", "QG", "QP", "QB") and not all(p.exists() for p in paths):
            print(f"  {dataset}: no {q} shard files (older extraction) -- re-extract to fit them")
            continue
        tabs = [pq.read_table(p) for p in paths]
        t = pa.concat_tables(tabs)
        if excl and "csd_code" in t.column_names and t.num_rows:
            keep = ~pd.Series(t.column("csd_code").to_pylist()).isin(excl).to_numpy()
            if not keep.all():
                t = t.filter(pa.array(keep))
        if q == "Q6" and t.num_rows:
            keys = t.column("key").to_pylist()
            uk = pd.Series(keys)
            cap = {k: (lambda ab: Q6_GLOBAL_CAP.get(ab, Q6_DEFAULT_CAP))(tuple(k[2:-2].split("', '"))) for k in uk.unique()}
            r = pd.DataFrame({"key": uk, "r": rng.random(len(uk))})
            rank = r.groupby("key")["r"].rank(method="first")
            capv = uk.map(lambda k: cap[k] if cap[k] is not None else 10 ** 12)
            t = t.filter(pa.array((rank <= capv).to_numpy()))
        pq.write_table(t, out_dir / f"observations_{q}_{dataset}.parquet", compression="zstd")
        rows[q] = t.num_rows
    # the uncapped histograms (extractor.HIST_PASSES: Q6 PMF, metal 1-3, NCI; benchmark codes already left out at
    # extraction): summed into <prefix>_<dataset>.npz, which stage_fit reads
    from jacobik.stats.extractor import HIST_PASSES

    for prefix in HIST_PASSES:
        hp = [shard_dir / f"{prefix}_{sh}.npz" for sh in sorted(shards)]
        if all(p.exists() for p in hp):
            rows[f"{prefix}_labels"] = len(save_hist_sum(hp, out_dir / f"{prefix}_{dataset}.npz"))
        else:
            print(f"  {dataset}: {prefix} shard files missing ({sum(not p.exists() for p in hp)} of {len(hp)}) -- "
                  f"no tables from them")
    return rows


def save_hist_sum(paths: Sequence[Path], out: Path) -> Dict[str, np.ndarray]:
    """Sum histogram files (arrays "h:<label>") into one file; the Q6 distance grid is stored alongside."""
    from jacobik.stats import q6pmf

    h = q6pmf.load_hist(paths)
    np.savez_compressed(out, edges=q6pmf.EDGES, **{"h:" + k: v for k, v in h.items()})
    return h



STATE_KEEPING = ("orig", "m_sub_early", "mo_state", "cn_state")


def pick_q1_variant(ev: pd.DataFrame, n_min: int = 30, tol: float = 0.01) -> dict:
    """Choose the Q1 ladder variant: rank by n-weighted held-out mean log-likelihood at `n_min`; if a variant that KEEPS
    oxidation-state/spin information in its specific levels is within `tol` (log-lik units) of the overall best, prefer it
    (a state-free ladder that is merely tied on tmQM/LnQM would mis-handle the states these datasets under-sample)."""
    e = ev[ev["n_min"] == n_min].copy()
    g = e.groupby("variant").apply(lambda d: pd.Series({
        "loglik": float(np.average(d["mean_loglik"], weights=d["n"])),
        "MAE": float(np.average(d["MAE_mean"], weights=d["n"])),
        "unresolved": int(d["unresolved"].sum()),
        "loglik_Ln": float(d.loc[d["group"] == "Ln", "mean_loglik"].mean()) if (d["group"] == "Ln").any() else np.nan,
        "loglik_TM": float(d.loc[d["group"] == "TM", "mean_loglik"].mean()) if (d["group"] == "TM").any() else np.nan,
    })).sort_values(["loglik", "MAE"], ascending=[False, True])
    best = str(g.index[0])
    keep = [v for v in g.index if v in STATE_KEEPING and g.loc[v, "loglik"] >= g.loc[best, "loglik"] - tol]
    chosen = keep[0] if keep else best
    return dict(variant=chosen, best_raw=best, table=g, preferred_state_keeping=(chosen != best))


def split_by_metal(obs_paths: Sequence[Path], q: str, out_dir: Path, metals: Sequence[str]) -> Tuple[int, int]:
    """Leave-metal-out split: train has NO rows for `metals`; heldout has only those (tests fallback to pooled levels)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_parquet(p) for p in obs_paths], ignore_index=True)
    h = df["metal"].isin(list(metals))
    df[~h].to_parquet(out_dir / f"observations_{q}.parquet", index=False)
    df[h].to_parquet(out_dir / f"heldout_{q}.parquet", index=False)
    return int((~h).sum()), int(h.sum())


# ----------------------------------------------------------------------------------------------------------------
# Audits
# ----------------------------------------------------------------------------------------------------------------
def os_label_audit(tmqm_dir: Path, splits: Sequence[str] = ("X1",), limit: Optional[int] = None) -> dict:
    """Round-trip the oxidation-state labeller over tmQM headers+SMILES (no geometry needed).  Reports accept/reject reasons
    and the per-metal OS histogram so implausible or suspicious states are visible before they enter any key."""
    import csv
    from collections import Counter, defaultdict
    from jacobik.core import oxidation as O
    csv.field_size_limit(10 ** 8)
    smi = {r["CSD_code"]: r["SMILES"] for r in csv.DictReader(open(Path(tmqm_dir) / "tmQM_y.csv"), delimiter=";")}
    reasons, hist = Counter(), defaultdict(Counter)
    n = 0
    for sp in splits:
        for line in open(Path(tmqm_dir) / "xyz" / f"tmQM_{sp}.xyz"):
            if not line.startswith("CSD_code"):
                continue
            m = re.match(r"CSD_code = (\w+) \| q = (-?\d+)", line)
            code, q = m.group(1), int(m.group(2))
            s = smi.get(code)
            metal = next((a for a in re.findall(r"\[([A-Z][a-z]?)[^\]]*\]", s or "") if a in O._GROUP or a in O._LN_Z), None)
            if not s or not metal:
                reasons["no_smiles_or_metal"] += 1
                continue
            o, why = O.os_from_smiles(s, q, metal)
            reasons[why] += 1
            if o is not None:
                hist[metal][o] += 1
            n += 1
            if limit and n >= limit:
                break
    return dict(reasons=dict(reasons), histogram={m: dict(sorted(c.items())) for m, c in hist.items()})


def far_component_audit(db_json: Path, far_A: float = 3.0, min_n: int = 30) -> pd.DataFrame:
    """List fitted Q1 mixture components whose mean is beyond `far_A` (second-sphere contamination that survived cleaning)."""
    rows = []
    for k, v in json.load(open(db_json)).items():
        if not k.startswith("Q1:") or v.get("n", 0) < min_n:
            continue
        for w, mu in zip(v["weights"], v["means"]):
            if mu > far_A and w > 0.01:
                rows.append(dict(key=k[3:], n=v["n"], weight=round(w, 3), mean=round(mu, 3)))
    return pd.DataFrame(rows).sort_values("n", ascending=False) if rows else pd.DataFrame(columns=["key", "n", "weight", "mean"])


def shannon_holdout_eval(q1_heldout_obs: Path, q1_train_db: Path, n_min: int = 30, min_rows: int = 60) -> pd.DataFrame:
    """Validation of the Shannon fallback WITHOUT touching the DOE benchmark.  For every (metal, spin, OS) state with data
    in the held-out observations, hide that state's own keys from the DB, predict its Q1 mean either (a) by the pooled
    state-free ladder or (b) by borrowing the best-populated other state of the same (metal, donor, geom) and applying the
    Shannon radius shift; report the MAE of each on the real observations."""
    from jacobik.stats import ladders as L, shannon as SH
    db = {k[3:]: v for k, v in json.load(open(q1_train_db)).items() if k.startswith("Q1:")}
    ho = pd.read_parquet(q1_heldout_obs)
    ks = [ast.literal_eval(k) for k in ho["key"]]
    ho = ho.assign(metal_=[k[0] for k in ks], spin=[k[1] for k in ks], os=[k[2] for k in ks], sub=[k[3] for k in ks],
                   geom=[k[4] for k in ks])
    ho = ho[ho["os"] != L.OS_UNKNOWN]
    ref_index: Dict[tuple, list] = {}
    for kstr, e in db.items():
        k = ast.literal_eval(kstr)
        if len(k) == 5 and isinstance(k[1], int) and isinstance(k[2], int) and e["n"] >= n_min:
            ref_index.setdefault((k[0], k[3], k[4]), []).append((k, e))
    out = []
    for (m, sp, o), g in ho.groupby(["metal_", "spin", "os"]):
        if len(g) < min_rows:
            continue
        err_pool, err_shan, n_sh = [], [], 0
        for _, r in g.iterrows():
            pooled = db.get(repr((m, r["sub"])))
            refs = [(k, e) for k, e in ref_index.get((m, r["sub"], r["geom"]), []) if (k[1], k[2]) != (sp, o)]
            if pooled is None or not refs:
                continue
            k, e = max(refs, key=lambda t: t[1]["n"])
            d = SH.radius_shift(m, (k[2], k[1]), (o, sp), L.geom_cn(r["geom"]))
            if d is None:
                continue
            err_pool.append(abs(float(np.dot(pooled["weights"], pooled["means"])) - r["value"]))
            err_shan.append(abs(float(np.dot(e["weights"], e["means"])) + d - r["value"]))
        if err_pool:
            out.append(dict(metal=m, spin=sp, OS=o, n=len(err_pool), MAE_pooled=np.mean(err_pool), MAE_shannon=np.mean(err_shan)))
    return pd.DataFrame(out)



# ---------------------------------------------------------------------------------------------------------------------
# Joint donor orientation (S_orient, quantity "QO"; see src/jacobik/stats/orientation.py)
# ---------------------------------------------------------------------------------------------------------------------
def load_qo(obs_dir: Path, datasets: Sequence[str]) -> pd.DataFrame:
    """All QO rows of the given datasets (observations_QO_<ds>.parquet), with a 'dataset' column."""
    return pd.concat([pd.read_parquet(Path(obs_dir) / f"observations_QO_{ds}.parquet").assign(dataset=ds) for ds in datasets],
                     ignore_index=True)


def qo_holdout_eval(qo: pd.DataFrame, mod: int = 10, n_jobs: int = -1) -> dict:
    """Fit S_orient on the training complexes (csd_code crc32 % mod != 0) and score the held-out ones.
    Returns {'train_rows', 'heldout_rows', 'n_keys', 'loglik' (orientation.heldout_loglik), 'by_class' DataFrame}."""
    from jacobik.stats import orientation as O
    qo = qo_with_per_sub(qo)
    h = holdout_mask(qo["csd_code"], mod)
    train, test = qo[~h], qo[h]
    entries = O.fit_orientation(train, n_jobs=n_jobs)
    ll = O.heldout_loglik(test, entries)
    tab = {}
    for jk, e in entries.items():
        tab[ast.literal_eval(jk.split(":", 1)[1])] = e
    per = []
    test = test[test["k"].isin(list(O.DIM))]      # per-class table: kappa donors (haptic rows are in ll above)
    for r in test.itertuples(index=False):
        k = int(r.k); d = O.DIM[k]; g = np.array([getattr(r, f"g{i}") for i in range(d)], float)
        f = next((tab[kk] for kk in O.level_keys(r.sub, r.metal, r.geom, int(r.anion), k, int(r.ring)) if kk in tab), None)
        per.append(np.nan if f is None else O.mixture_logpdf(g, f) / d)
    t = test.assign(lp=per, hbonded=test["hb_min"] < O.HB_CUT)
    by_class = t.groupby("sub").agg(n=("lp", "size"), loglik_per_dim=("lp", "mean")).sort_values("n", ascending=False)
    return {"train_rows": int(len(train)), "heldout_rows": int(len(test)), "n_keys": len(entries), "loglik": ll,
            "by_class": by_class}


def qo_with_per_sub(qo: pd.DataFrame) -> pd.DataFrame:
    """QO rows plus the per-substituent haptic rows (orientation.per_sub_rows: one row per face substituent, class
    k + PER_SUB_OFFSET[H|X]) that the scorer uses when HAPTIC_PER_SUB is on (2026-09-29 v5 DB: fitted separately by
    resources/db_provenance/hapt_ps_fit.py; now part of the refit)."""
    from jacobik.stats import orientation as O
    try:
        ps = O.per_sub_rows(qo)
    except Exception as exc:  # pragma: no cover - older QO files without raw face geometry
        print(f"  per-substituent haptic rows unavailable ({exc}); face entries only")
        return qo
    return pd.concat([qo, ps], ignore_index=True) if len(ps) else qo


def qo_fit_full(qo: pd.DataFrame, n_jobs: int = -1) -> Dict[str, dict]:
    """Final S_orient fit on every row incl. the per-substituent haptic rows (entries for constraints_refit.json)."""
    from jacobik.stats import orientation as O
    return O.fit_orientation(qo_with_per_sub(qo), n_jobs=n_jobs)


def add_entries_to_db(db_json: Path, entries: Dict[str, dict], quantity: str = "QO") -> Dict[str, int]:
    """Replace every <quantity>: entry of a constraints JSON with ``entries`` (in place). Returns key counts."""
    db_json = Path(db_json)
    raw = json.loads(db_json.read_text())
    raw = {k: v for k, v in raw.items() if not k.startswith(quantity + ":")}
    raw.update(entries)
    db_json.write_text(json.dumps(raw))
    return {"removed_old_and_added": len(entries), "total_keys": len(raw)}


def qo_level_audit(entries: Dict[str, dict]) -> pd.DataFrame:
    """Number of fitted QO keys and training rows per (k, key length) = ladder depth."""
    rows = []
    for jk, e in entries.items():
        key = ast.literal_eval(jk.split(":", 1)[1])
        rows.append(dict(k=key[0], key_len=len(key), n=e["n"], n_comp=len(e["weights"])))
    d = pd.DataFrame(rows)
    return d.groupby(["k", "key_len"]).agg(keys=("n", "size"), rows=("n", "sum"), mean_components=("n_comp", "mean"))



# ---------------------------------------------------------------------------------------------------------------------
# 2026-10-01: benchmark hold-out, pooled held-out selection, priors, DB metadata
# ---------------------------------------------------------------------------------------------------------------------
HOLDOUT_CODES_JSON = Path(__file__).with_name("benchmark_holdout_codes.json")


def benchmark_codes(path: Path = HOLDOUT_CODES_JSON) -> List[str]:
    """CSD codes of every benchmark complex (N48 screen + RING10); never fitted on."""
    d = json.loads(Path(path).read_text())
    return sorted({c for k, v in d.items() if not k.startswith("_") for c in v})


def exclude_codes_in_place(obs_dir: Path, codes: Optional[Iterable[str]] = None, pattern: str = "observations_*.parquet") -> Dict[str, int]:
    """Drop the given CSD codes (default benchmark_codes()) from every observation file in obs_dir (for observation
    sets merged before the exclusion existed). Returns rows removed per file."""
    codes = set(benchmark_codes() if codes is None else codes)
    out = {}
    for p in sorted(Path(obs_dir).glob(pattern)):
        df = pd.read_parquet(p)
        if "csd_code" not in df.columns:
            continue
        m = df["csd_code"].isin(codes)
        if m.any():
            df[~m].to_parquet(p, index=False, compression="zstd")
        out[p.name] = int(m.sum())
    return out


def _l0_query_keys(q: str, keys: pd.Series) -> List[tuple]:
    return [ast.literal_eval(k) for k in keys]


def pooled_holdout(obs_paths: Sequence[Path], q: str, work_dir: Path, rs=(4.0, 10.0, 25.0), variants=None,
                   n_workers: int = 1, mod: int = 10, max_rows: int = 300_000, seed: int = SEED) -> pd.DataFrame:
    """Held-out selection for a partially pooled quantity (Q1 / Q2). Complexes are split by CSD-code hash (1 in
    ``mod`` held out); the training rows are expanded and pooled-fitted (independent fits once, then MAP adaptation
    for each relevance factor r and, for Q1, each ladder variant); every held-out row is resolved EXACTLY as the
    runtime does (StatsLookup) and scored. Returns one row per (variant, r) with the mean held-out log p per complex,
    its 5th percentile, the same for complexes whose exact L0 key had < 10 training complexes ("sparse"), and the
    number of unresolved rows. Pick the best mean (pick_pooling)."""
    from jacobik.stats import expander, pooling, mixture as MX
    from jacobik.stats import ladders as L
    from jacobik.stats.stats import StatsLookup, load_db

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    n_tr, n_ho = split_observations(obs_paths, q, work_dir, mod)
    expander.expand_quantity(work_dir, work_dir / "exp", quantities=[q])
    pf = pooling.PooledFitter(q, pooling.load_expanded(work_dir / "exp", q))
    pf.fit_independent(n_workers=n_workers, progress=print)
    ho = pd.read_parquet(work_dir / f"heldout_{q}.parquet")
    if len(ho) > max_rows:
        ho = ho.sample(max_rows, random_state=seed)
    keys = [ast.literal_eval(k) for k in ho["key"]]
    vals = ho["value"].to_numpy(float)
    if q == "Q2":
        vals = np.cos(np.deg2rad(vals))
    codes = ho["csd_code"].to_numpy()
    train_cx = pf.n_cx
    sparse = np.array([train_cx.get(repr(k), 0) < 10 for k in keys])
    groups: Dict[tuple, list] = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    groups = {k: np.asarray(v) for k, v in groups.items()}
    rows = []
    for var in (variants or ([L.DEFAULT_Q1_VARIANT] if q == "Q1" else [None])):
        for r in rs:
            ent = pf.entries(r, var)
            db = load_db({f"{q}:{k}": e for k, e in ent.items()}, meta={"q1_variant": var} if var else {})
            st = StatsLookup(db, q1_variant=var)
            lp = np.full(len(keys), np.nan)
            for k, idx in groups.items():
                try:
                    fd = st.dist(q, k)
                except Exception:
                    continue
                f = MX.from_entry(dict(means=fd.means, vars=fd.vars, weights=fd.weights, nu=fd.nu))
                lp[idx] = MX.logp(f, vals[idx])
            d = pd.DataFrame({"cx": codes, "lp": lp, "sparse": sparse}).dropna()
            per = d.groupby("cx")["lp"].mean()
            per_s = d[d["sparse"]].groupby("cx")["lp"].mean()
            rows.append(dict(quantity=q, variant=var, r=r, n_cx=len(per), mean=per.mean(), p5=per.quantile(0.05),
                             sparse_n_cx=len(per_s), sparse_mean=per_s.mean() if len(per_s) else np.nan,
                             sparse_p5=per_s.quantile(0.05) if len(per_s) else np.nan,
                             unresolved=int(np.isnan(lp).sum()), train_rows=n_tr, heldout_rows=n_ho))
            print(f"  {q} variant={var} r={r:g}: mean {rows[-1]['mean']:.3f}  p5 {rows[-1]['p5']:.3f}  "
                  f"sparse {rows[-1]['sparse_mean']:.3f}/{rows[-1]['sparse_p5']:.3f}", flush=True)
    return pd.DataFrame(rows)


def pick_pooling(ev: pd.DataFrame, tol: float = 0.01, require_state: bool = False) -> dict:
    """Best (variant, r) by mean held-out log p per complex; among variants within ``tol`` of the best, a
    state-keeping Q1 variant is preferred (as pick_q1_variant). ``require_state``: Q1 picks only among state-keeping
    variants (owner 2026-10-15: the oxidation / spin state stays in the keys; the full refit's best state-free
    variant led by 0.013, just outside ``tol``)."""
    e = ev.sort_values("mean", ascending=False).reset_index(drop=True)
    best = e.iloc[0]
    pick = best
    if best["quantity"] == "Q1" and best["variant"] not in STATE_KEEPING:
        near = e[((e["mean"] >= best["mean"] - tol) | require_state) & e["variant"].isin(STATE_KEEPING)]
        if len(near):
            pick = near.iloc[0]
    return {"variant": pick["variant"], "r": float(pick["r"]), "mean": float(pick["mean"]),
            "best_raw": (best["variant"], float(best["r"]), float(best["mean"]))}


def fit_priors(obs_dir: Path, datasets: Sequence[str]) -> Dict[str, dict]:
    """QG (geometry | metal, OS, CN) and QP (haptic arc partition) categorical priors from the extracted rows
    (stats/hypothesis_tables.py). QB (candidate donors) is extracted but not fitted (ligands-only mode)."""
    from jacobik.stats import hypothesis_tables as HT

    out = {}
    for q, fit in (("QG", HT.fit_geometry_prior), ("QP", HT.fit_partition_prior)):
        paths = [Path(obs_dir) / f"observations_{q}_{ds}.parquet" for ds in datasets]
        paths = [p for p in paths if p.exists()]
        if not paths:
            print(f"  no {q} observation files -- skipped")
            continue
        out.update(fit(pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)))
    return out


def write_meta(final_dir: Path, q1_variant: str, pooling: Dict[str, float], extra: Optional[dict] = None,
               name: str = "constraints_refit") -> Path:
    """The DB sidecar. The runtime keys model-dependent defaults on it (scorer: Q5 folding when q5_model ends in
    'folded'; stats: Q2 pair-class lookup when the DB's Q2 keys carry a pair class)."""
    meta = {"q1_variant": q1_variant, "oxidation_keyed": True,
            "ladders": "jacobik.stats.ladders", "fitter": "2026-10-01",
            "q1_model": "gmm_pooled", "q2_model": "tmix_pooled_pairclass", "q5_model": "wells_folded",
            "q6_model": "onset_hp", "pooling": {k: float(v) for k, v in pooling.items()},
            "benchmark_holdout": benchmark_codes()}
    meta.update(extra or {})
    p = Path(final_dir) / f"{name}.meta.json"
    p.write_text(json.dumps(meta, indent=1))
    return p


def db_summary(db_json: Path) -> pd.DataFrame:
    """Entries per quantity, with the pooled / adapted shares and median components."""
    raw = json.loads(Path(db_json).read_text())
    rows = []
    for k, e in raw.items():
        q = k.split(":", 1)[0]
        rows.append(dict(q=q, pooled=bool(e.get("pooled")), adapted=e.get("src") == "adapted",
                         k=e.get("k", len(e.get("weights") or [0])), typ=e.get("type")))
    d = pd.DataFrame(rows)
    return d.groupby("q").agg(entries=("q", "size"), pooled=("pooled", "mean"), adapted=("adapted", "mean"),
                              median_k=("k", "median"), types=("typ", lambda s: ",".join(sorted(set(map(str, s))))))
