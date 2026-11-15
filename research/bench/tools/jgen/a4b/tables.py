"""Build the frozen A4b row, unit, parent, and bootstrap summary tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import os

#: production fingerprint the A4 records must carry (A4b ran on 273f2ccbf892; reruns set JACOBIK_A4_FINGERPRINT)
A4_FINGERPRINT = os.environ.get("JACOBIK_A4_FINGERPRINT", "273f2ccbf892")

import numpy as np

from .relax_set import select

#: unit seeds; the P2R fold-in gate runs fresh seeds 31-33 (JACOBIK_A4_SEEDS="31,32,33")
SEEDS = tuple(int(x) for x in os.environ.get("JACOBIK_A4_SEEDS", "11,12,13").split(","))
BUCKETS_U = ("U-TOP1", "U-POOL", "U-NONE")
BUCKETS_R = ("R-TOP1", "R-RANK", "R-MISS")


def unrel_bucket(rmsd, top):
    if rmsd[top] is not None and rmsd[top] <= 1.0:
        return "U-TOP1"
    if any(x is not None and x <= 1.0 for x in rmsd):
        return "U-POOL"
    return "U-NONE"


def rel_bucket(rows, top):
    good = lambda x: bool(x.get("hit") or x.get("equivalent"))
    if top in rows and good(rows[top]):
        return "R-TOP1"
    if any(i != top and good(x) for i, x in rows.items()):
        return "R-RANK"
    return "R-MISS"


def unrel_metrics(scores, energies, rmsd, top, floor, e_ref=None):
    e, s = np.asarray(energies, float), np.asarray(scores, float)
    d = e - e[floor]
    garb = d > 100.0
    pairs = [(g, r) for g in range(len(e)) if garb[g] for r in range(len(e)) if not garb[r]]
    return dict(
        ranking_loss=float(e[top] - e[floor]),
        obscene=bool(e[top] - e_ref > 200) if e_ref is not None else None,
        top1_inversion=bool(garb[top]),
        pair_inversion_rate=(float(sum(s[g] > s[r] for g, r in pairs) / len(pairs)) if pairs else float("nan")),
        u_bucket=unrel_bucket(rmsd, top),
    )


def relaxed_values(E, Erel, Eref, Erefrel):
    if not all(np.isfinite(v) for v in (E, Erel, Eref, Erefrel)):
        return dict(strain=float("nan"), offset=float("nan"), offset_raw=float("nan"))
    strain, offset, raw = E - Erel, Erel - Erefrel, Erel - Eref
    if abs((E - Eref) - (strain + raw)) > 1e-6:
        raise ValueError("strain/offset identity failed")
    return dict(strain=float(strain), offset=float(offset), offset_raw=float(raw))


def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _finite(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


def _parent_summary(by_parent):
    """Mean, percentile parent bootstrap CI, and median over parent means."""
    by_parent = {k: np.asarray(v, float) for k, v in by_parent.items() if len(v)}
    means = np.asarray([v.mean() for _, v in sorted(by_parent.items())], float)
    if len(means) == 0:
        return dict(n_parents=0, mean=None, lo=None, hi=None, median=None)
    rng = np.random.default_rng(11)
    draws = means[rng.integers(0, len(means), size=(10000, len(means)))].mean(axis=1)
    return dict(n_parents=len(means), mean=float(means.mean()),
                lo=float(np.percentile(draws, 2.5)), hi=float(np.percentile(draws, 97.5)),
                median=float(np.median(means)))


def _write_csv(path, records, fields):
    with Path(path).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)


def _panel_members(root):
    split = _read_json(root / "vm_split.json", {}) or {}
    cal = set(split.get("cal_parents", []))
    tail = set(split.get("tail_parents", []))
    if not cal or not tail:
        raise ValueError("vm_split.json must define cal_parents and tail_parents")
    return {"cal": cal, "tail": tail}


def _unit_relax(root, code, seed):
    p = root / "relax" / f"{code}_{seed}.json"
    d = _read_json(p, {}) or {}
    return d.get("rows", {})


def build(data):
    root = Path(data)
    members = _panel_members(root)
    charge_mult = _read_json(root / "charge_mult.json", {}) or {}
    fallback_codes = set(_read_json(root / "fallback_codes.json", []) or [])
    recs, manifests = {}, {}
    for p in sorted((root / "records").rglob("*.json")):
        d = _read_json(p)
        if d and d.get("code") and d.get("seed") is not None:
            recs[(d["code"], int(d["seed"]))] = d
    for p in sorted((root / "xyz").rglob("*_pool.json")):
        d = _read_json(p)
        if d and d.get("code") and d.get("seed") is not None:
            manifests[(d["code"], int(d["seed"]))] = (p, d)

    raw_units, raw_rows, parent_data = [], [], defaultdict(list)
    coverage_disagreements = []
    missing = []
    for key, rec in sorted(recs.items()):
        code, seed = key
        mpair = manifests.get(key)
        if mpair is None:
            missing.append(dict(code=code, seed=seed, reason="pool manifest missing"))
            raw_units.append(dict(code=code, seed=seed, status=rec.get("status", "failed"), reason=rec.get("reason", "pool manifest missing"),
                                  panel=rec.get("panel"), embed_fallback=code in fallback_codes))
            continue
        mp, manifest = mpair
        if rec.get("status") != "ok" or rec.get("fingerprint") != A4_FINGERPRINT:
            # an ok record with another fingerprint is NOT an ok unit (it has no metrics): label it explicitly
            st = rec.get("status", "failed")
            st = "fingerprint_mismatch" if st == "ok" else st
            raw_units.append(dict(code=code, seed=seed, panel=rec.get("panel"), status=st,
                                  reason=rec.get("reason") or f"record fingerprint {rec.get('fingerprint')} != {A4_FINGERPRINT}",
                                  embed_fallback=code in fallback_codes))
            continue
        if int(manifest.get("n_rows", -1)) != len(manifest.get("rows", [])):
            raise ValueError(f"manifest row count malformed for {code} seed {seed}")
        en = np.asarray(rec.get("energies_kj", []), float)
        scores = np.asarray(rec.get("scores", []), float)
        rmsd = rec.get("coverage", {}).get("rmsd_A", [])
        if len(en) != len(manifest["rows"]) or len(scores) != len(en) or len(rmsd) != len(en):
            raise ValueError(f"record/manifest row lengths differ for {code} seed {seed}")
        mrows = manifest["rows"]
        for i, row in enumerate(mrows):
            me = row.get("mace_energy_kj")
            if not _finite(me) or abs(float(me) - en[i]) > 1e-6:
                raise ValueError(f"record/manifest energy mismatch for {code} seed {seed} row {i}")
        top = next(i for i, r in enumerate(mrows) if r.get("is_top1"))
        floor = next(i for i, r in enumerate(mrows) if r.get("is_floor"))
        selected = set(select(manifest, rec))
        rr = _unit_relax(root, code, seed)
        refs = _read_json(root / "relax" / "references.json", {}) or {}
        ref = refs.get(code, {})
        eref = float(rec["ref_energy_kj"])
        um = unrel_metrics(scores, en, rmsd, top, floor, eref)
        cov = rec.get("coverage", {})
        expected_bucket = ("U-TOP1" if cov.get("top1_within_1A") else ("U-POOL" if cov.get("full_pool_within_1A") else "U-NONE"))
        if um["u_bucket"] != expected_bucket:
            coverage_disagreements.append(dict(code=code, seed=seed, table=um["u_bucket"], record=expected_bucket))
        relrows = {}
        top_rr = rr.get(str(top), {})
        for i, (mr, energy, score, rd) in enumerate(zip(mrows, en, scores, rmsd)):
            rrel = rr.get(str(i), {})
            er = rrel.get("E_mace_kj")
            er = float(er) if _finite(er) else float("nan")
            vals = relaxed_values(float(energy), er, eref, float(ref["E_mace_kj"])) if _finite(ref.get("E_mace_kj")) else dict(strain=float("nan"), offset=float("nan"), offset_raw=float("nan"))
            if i in selected:
                relrows[i] = dict(hit=bool(rrel.get("hit")), equivalent=bool(rrel.get("equivalent")),
                                  rmsd_refrel_A=rrel.get("rmsd_refrel_A"), offset=vals["offset"])
            raw_rows.append(dict(code=code, seed=seed, row_index=i, score=float(score), E_kj=float(energy),
                                 dE_unrel=float(energy - eref), dE_vs_floor=float(energy - en[floor]),
                                 rmsd_ref_A=rd, is_top1=i == top, is_floor=i == floor,
                                 ugarb=bool(energy - en[floor] > 100), in_relax_set=i in selected,
                                 relax_status=rrel.get("status", "not_selected"), cycles=rrel.get("cycles"),
                                 E_rel=er if np.isfinite(er) else None,
                                 strain=vals["strain"] if np.isfinite(vals["strain"]) else None,
                                 offset=vals["offset"] if np.isfinite(vals["offset"]) else None,
                                 hit=rrel.get("hit"), equivalent=rrel.get("equivalent"),
                                 contact_change=rrel.get("contact_change")))
        um.update(code=code, seed=seed, status="ok", reason="", panel=rec.get("panel"),
                  top1_dE=float(en[top] - eref), embed_fallback=code in fallback_codes,
                  ref_relax_status=ref.get("status", "missing"),
                  n_relax_selected=len(selected),
                  n_relax_failed=sum(rr.get(str(i), {}).get("status") in ("failed", "not_selected", None) for i in selected),
                  n_relax_capped=sum(rr.get(str(i), {}).get("status") == "capped" for i in selected),
                  r_bucket=rel_bucket(relrows, top), charge=charge_mult.get(code, {}).get("charge"),
                  multiplicity=charge_mult.get(code, {}).get("mult"), n_atoms=len(manifest["ref_symbols"]),
                  top1_relax_status=top_rr.get("status", "not_selected"),
                  strain_top1=float("nan"), offset_top1=float("nan"), strain_share=float("nan"),
                  contact_change_top1=(bool(top_rr["contact_change"]) if top_rr.get("contact_change") is not None else None))
        top_er = top_rr.get("E_mace_kj")
        topvals = relaxed_values(float(en[top]), float(top_er), eref, float(ref["E_mace_kj"])) if _finite(top_er) and _finite(ref.get("E_mace_kj")) else dict(strain=float("nan"), offset=float("nan"), offset_raw=float("nan"))
        um["strain_top1"], um["offset_top1"] = topvals["strain"], topvals["offset"]
        um["strain_share"] = topvals["strain"] / (en[top] - eref) if np.isfinite(topvals["strain"]) and en[top] - eref > 10 else float("nan")
        parent_data[code].append((um, relrows, top, rr))
        raw_units.append(um)

    # Preserve attempted parents with no record as explicit failed units.
    seen = set(recs)
    for code, seed in sorted((c, s) for c in members["cal"] | members["tail"] for s in SEEDS):
        if (code, seed) not in seen:
            raw_units.append(dict(code=code, seed=seed, status="missing", reason="record missing", embed_fallback=code in fallback_codes))
            missing.append(dict(code=code, seed=seed, reason="record missing"))

    # Expand the five shared parents into tail without solving or relaxing twice.
    units, rows = [], []
    for u in raw_units:
        code, seed = u["code"], u["seed"]
        if u.get("status") != "ok":
            for panel in ("cal", "tail"):
                if code in members[panel]:
                    units.append({**u, "panel": panel})
            continue
        rowset = [r for r in raw_rows if r["code"] == code and r["seed"] == seed]
        for panel in ("cal", "tail"):
            if code in members[panel]:
                units.append({**u, "panel": panel})
                rows.extend({**r, "panel": panel} for r in rowset)

    parents = []
    for code in sorted(members["cal"] | members["tail"]):
        good = [u for u in raw_units if u["code"] == code and u.get("status") == "ok"]
        panels = [p for p in ("cal", "tail") if code in members[p]]
        cm = charge_mult.get(code, {})
        if not good:
            parents.append(dict(code=code, panels=",".join(panels), n_seeds_ok=0, charge=cm.get("charge"), multiplicity=cm.get("mult")))
            continue
        d = dict(code=code, panels=",".join(panels), n_seeds_ok=len({u["seed"] for u in good}),
                 dE_unrel_top1=_mean_finite([u["top1_dE"] for u in good]),
                 ranking_loss=_mean_finite([u["ranking_loss"] for u in good]),
                 strain_top1=_mean_finite([u["strain_top1"] for u in good]),
                 offset_top1=_mean_finite([u["offset_top1"] for u in good]),
                 obscene_seeds=sum(bool(u["obscene"]) for u in good), charge=cm.get("charge"),
                 multiplicity=cm.get("mult"), n_atoms=good[0].get("n_atoms"),
                 embed_fallback=any(u.get("embed_fallback") for u in good))
        for b in BUCKETS_U + BUCKETS_R:
            d[b.replace("-", "_") + "_seed_count"] = sum(u.get("u_bucket" if b.startswith("U") else "r_bucket") == b for u in good)
        parents.append(d)

    # Dataset has 125 parents x three seeds. Keep failed/missing units in denominator metadata.
    expected = {(c, s) for c in members["cal"] | members["tail"] for s in SEEDS}
    observed = set(recs)
    missing_units = [dict(code=c, seed=s, status="missing", reason="record missing") for c, s in sorted(expected - observed)]
    valid_units = [u for u in units if u.get("status") == "ok" and not u.get("embed_fallback")]
    for u in valid_units:
        # The panel contract is authoritative: the duplicated unit row is already assigned per panel.
        pass

    _write_csv(root / "rows.csv", rows, ["code", "panel", "seed", "row_index", "score", "E_kj", "dE_unrel", "dE_vs_floor", "rmsd_ref_A", "is_top1", "is_floor", "ugarb", "in_relax_set", "relax_status", "cycles", "E_rel", "strain", "offset", "hit", "equivalent", "contact_change"])
    _write_csv(root / "units.csv", units, ["code", "panel", "seed", "status", "reason", "embed_fallback", "top1_dE", "ranking_loss", "obscene", "top1_inversion", "pair_inversion_rate", "u_bucket", "r_bucket", "strain_top1", "offset_top1", "strain_share", "contact_change_top1", "ref_relax_status", "n_relax_selected", "n_relax_failed", "n_relax_capped", "top1_relax_status", "charge", "multiplicity", "n_atoms"])
    _write_csv(root / "parents.csv", parents, None if not parents else list(parents[0].keys()))

    summaries = {}
    for panel in ("cal", "tail"):
        pu = [u for u in units if u.get("panel") == panel]
        analyzable = [u for u in pu if u.get("status") == "ok" and not u.get("embed_fallback")]
        sm = {"denominators": {"units_attempted": len(pu), "units_ok": sum(u.get("status") == "ok" for u in pu),
                               "units_failed": sum(u.get("status") != "ok" for u in pu),
                               "fallback_units_excluded": sum(u.get("embed_fallback", False) for u in pu),
                               "parents_excluded": sorted({u["code"] for u in pu if u.get("embed_fallback")})}}
        for key, buckets in (("u_bucket", BUCKETS_U), ("r_bucket", BUCKETS_R)):
            sm[key] = {b: _summary(analyzable, key, lambda x, b=b: float(x.get(key) == b)) for b in buckets}
        for key in ("obscene", "top1_inversion", "pair_inversion_rate", "ranking_loss", "top1_dE", "strain_top1", "offset_top1", "strain_share", "contact_change_top1"):
            sm[key] = _summary(analyzable, key, lambda x, key=key: float(x[key]) if _finite(x.get(key)) else float("nan"))
        sm["seed_consistency"] = {}
        for key, buckets in (("u_bucket", BUCKETS_U), ("r_bucket", BUCKETS_R)):
            consistency_units = [u for u in pu if not u.get("embed_fallback")]
            sm["seed_consistency"][key] = {b: {f"{n}/3": _consistency(consistency_units, key, b, n) for n in (3, 2, 1, 0)} for b in buckets}
        sm["failures"] = _failure_counts(pu)
        sm["sensitivity"] = _sensitivity(root, analyzable, panel, rows)
        summaries[panel] = sm
    (root / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False))
    (root / "missing_units.json").write_text(json.dumps(missing_units + missing, indent=2))
    (root / "coverage_disagreements.json").write_text(json.dumps(coverage_disagreements, indent=2))
    return rows, units, parents, summaries


def _mean_finite(values):
    a = np.asarray([float(v) for v in values if _finite(v)], float)
    return float(a.mean()) if len(a) else None


def _summary(units, key, get):
    by = defaultdict(list)
    for u in units:
        v = get(u)
        if _finite(v):
            by[u["code"]].append(float(v))
    return _parent_summary(by)


def _consistency(units, key, bucket, n):
    by = defaultdict(list)
    for u in units:
        by[u["code"]].append(u.get(key) == bucket)
    return sum(sum(v) == n and len(v) == 3 for v in by.values())


def _failure_counts(units):
    status = Counter(u.get("status", "failed") for u in units)
    reasons = Counter(u.get("reason") or "unspecified" for u in units if u.get("status") != "ok")
    relax_failed = sum(int(u.get("n_relax_failed", 0) or 0) for u in units)
    relax_capped = sum(int(u.get("n_relax_capped", 0) or 0) for u in units)
    ref_failed = sum(u.get("ref_relax_status") not in ("ok", "not_converged") for u in units if u.get("status") == "ok")
    return dict(unit_status=dict(status), unit_failure_reasons=dict(reasons), relax_failed=relax_failed,
                relax_capped=relax_capped, reference_relax_failed=ref_failed)


def _sensitivity(root, units, panel, rows):
    # Reclassify the exact measured relax-set rows using the directive's alternate thresholds.
    results = {}
    for hit_radius in (0.3, 0.5):
        for eq_threshold in (2.0, 5.0, 10.0):
            classifications = defaultdict(list)
            for u in units:
                code, seed = u["code"], u["seed"]
                unit_rows = [r for r in rows if r["code"] == code and r["seed"] == seed and r["panel"] == panel]
                top_row = next((r for r in unit_rows if r["is_top1"]), None)
                selected = [r for r in unit_rows if r["in_relax_set"]]
                rr = _unit_relax(root, code, seed)
                relaxed = {}
                for row in selected:
                    item = rr.get(str(row["row_index"]), {})
                    rmsd = item.get("rmsd_refrel_A")
                    off = row.get("offset")
                    hit = _finite(rmsd) and float(rmsd) <= hit_radius
                    equivalent = not hit and _finite(off) and float(off) <= eq_threshold
                    relaxed[row["row_index"]] = dict(hit=bool(hit), equivalent=bool(equivalent))
                top = top_row["row_index"] if top_row else -1
                bucket = rel_bucket(relaxed, top)
                classifications[code].append(bucket)
            results[f"hit{hit_radius:g}_eq{eq_threshold:g}"] = {
                b: _parent_summary({c: [float(x == b) for x in vals] for c, vals in classifications.items()})
                for b in BUCKETS_R
            }
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    a = p.parse_args()
    build(a.data)
