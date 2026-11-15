"""Step 6 read-out: corrected DB versus production (current start), and existing P/S versus the current start on the corrected DB and the damaged-trial DBs.

    python step6_analyse.py JOB_DIR OUT_DIR [--c2b-settings research/campaigns/c2/C2B_SETTINGS.json]

JOB_DIR is the fetched ``ds_step6`` job directory (``<phase>/g<i>/<arm>/{records,xyz}``). Arms: C = production DB, current start; D = candidate DB, current start;
DS = candidate DB + P/S start (``start_kabsch`` + ``start_kabsch_pivot``). The comparison arithmetic is the D3 one (``tools.jgen.d3.comparison.compare``: per-parent mean over seeds,
parent-clustered bootstrap, failures stay in the denominator). The candidate rule of contract v2 C is evaluated on the development parents:
  upper 95% bound of the paired top-1 dE difference (DS minus D) below +5 kJ/mol, no lower shell05 (coordination) than D, no development parent worse by more than 25 kJ/mol on two seeds.
JEGPUJ and SIVBUV are the diagnostic supplement: reported, not in the rule. The coordination column is the shell-RMSD fraction within 0.5 A (``shell05``), the bench's own coordination metric;
the fixed-pool 1.25 x source-distance rule is not applied to generated pools here.
"""
import argparse, json, shutil, sys
from pathlib import Path

DEV = "LUQVOK SESWIZ LIPDEU YABYEJ LOPJEF ZAQNUE HOXJUZ USAPOU UMUBAF MIGTEE EFOXII".split()
SUPPLEMENT = ["JEGPUJ", "SIVBUV"]
DAMAGED = {"LUQVOK_Q1_obs": "LUQVOK", "LUQVOK_Q1_rem": "LUQVOK", "LIPDEU_QO_obs": "LIPDEU", "LIPDEU_QO_rem": "LIPDEU"}


PM_PREFIX = None  # the job directory on Perlmutter (set from --pm-prefix); pool_manifest paths are remapped to the fetched copy


def link(job: Path, phase: str, arm: str, root: Path, rename: str = None):
    """Copy the records of ``phase/g*/arm`` into ``root/<rename or arm>/records`` (small files); raises on a duplicated record.
    Each ok record is enriched (``tools.jgen.d3.worker.enrich_record``: shell RMSD, edge-on counts) from the fetched pool, as ``gate_pm assemble`` does."""
    from tools.jgen.d3.worker import enrich_record
    n = 0
    for gdir in sorted((job / phase).glob("g[0-9]")):
        src = gdir / arm / "records"
        if not src.is_dir():
            continue
        dst = root / (rename or arm) / "records"
        dst.mkdir(parents=True, exist_ok=True)
        for rec in src.glob("*.json"):
            if (dst / rec.name).exists():
                raise FileExistsError(f"{dst / rec.name}: two GPUs wrote the same record")
            r = json.loads(rec.read_text())
            if r.get("pool_manifest") and PM_PREFIX:
                r["pool_manifest"] = r["pool_manifest"].replace(f"{PM_PREFIX}/{phase}/{gdir.name}/{arm}", str((gdir / arm).resolve()), 1)
            (dst / rec.name).write_text(json.dumps(r)); n += 1
            if r.get("status") == "ok":
                try:
                    enrich_record(dst / rec.name)
                except Exception as exc:  # noqa: BLE001 - recorded as a failed record, as the D3 worker does
                    r.update(status="failed", reason="enrichment: " + repr(exc)); (dst / rec.name).write_text(json.dumps(r))
    return n


def summarize(c):
    return {k: dict(delta=v.get("mean"), ci=v.get("ci"), n=v.get("n")) for k, v in c["metrics"].items() if k in ("top1", "floor", "shell05", "min_rmsd", "distinct", "hit1")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--pm-prefix", default="/pscratch/sd/a/achen310/jacobik/runs/jk-ds-step6-20261111-2344", help="the job directory on Perlmutter, for remapping pool paths")
    ap.add_argument("--c2b-settings", type=Path, default=Path("research/campaigns/c2/C2B_SETTINGS.json"))
    a = ap.parse_args()
    global PM_PREFIX
    PM_PREFIX = a.pm_prefix
    from tools.jgen.d3.comparison import compare
    settings = json.loads(a.c2b_settings.read_text())
    settings["seeds"] = [11, 12, 13]
    a.out.mkdir(parents=True, exist_ok=True)
    result = {}
    # main root: C (production), D and DS (candidate DB)
    root = a.out / "main"
    counts = dict(C=link(a.job, "s6_C", "C", root), D=link(a.job, "s6_DF", "D", root), DS=link(a.job, "s6_DF", "DS", root))
    codes = DEV + SUPPLEMENT
    # primary exclusions of the C-2b settings do not apply to this development set; keep every listed parent in the denominators
    settings = dict(settings, primary_exclusions=[], guard_artifacts=[])
    d_vs_c = compare(root, "D", "D", settings, codes=DEV, reference="C", proof=True)
    ds_vs_d = compare(root, "DS", "D", settings, codes=DEV, reference="D", proof=True)
    supp = {n: compare(root, n, "D", settings, codes=SUPPLEMENT, reference=ref, proof=True) for n, ref in (("D", "C"), ("DS", "D"))}
    per = {c: {m: ds_vs_d["per_parent"][c][m].get("delta") for m in ("top1", "floor", "shell05")} for c in DEV}
    top1 = ds_vs_d["metrics"]["top1"]
    worse = [c for c in DEV if (lambda d: d is not None and d > 25)(ds_vs_d["per_parent"][c]["top1"].get("delta"))]
    ci_hi = top1["ci"][1] if top1.get("ci") else None
    shell = ds_vs_d["metrics"].get("shell05", {})
    rule = dict(top1_upper_below_5=ci_hi is not None and ci_hi < 5.0, shell05_not_lower=shell.get("mean") is not None and shell["mean"] >= 0,
                parents_worse_than_25_on_mean=worse, passes=bool(ci_hi is not None and ci_hi < 5.0 and (shell.get("mean") or 0) >= 0 and not worse),
                note="the 25 kJ/mol clause is evaluated on the three-seed mean here; per-seed counts are in ds_vs_d.guard.trip_parents")
    result["record_counts"] = counts
    result["D_vs_C_current_start"] = dict(summary=summarize(d_vs_c), failures=d_vs_c["failures"], per_parent={c: d_vs_c["per_parent"][c]["top1"].get("delta") for c in DEV})
    result["DS_vs_D_on_candidate_DB"] = dict(summary=summarize(ds_vs_d), failures=ds_vs_d["failures"], per_parent=per, guard=ds_vs_d["guard"]["with_artifacts"])
    result["P_candidate_rule"] = rule
    result["supplement_JEGPUJ_SIVBUV"] = {n: dict(summary=summarize(c), failures=c["failures"], per_parent={p: c["per_parent"][p]["top1"].get("delta") for p in SUPPLEMENT}) for n, c in supp.items()}
    # damaged phases: D and DS on the damaged DB, for the stressed parent only
    dam = {}
    for name, code in DAMAGED.items():
        r = a.out / name
        n = dict(D=link(a.job, f"s6_{name}", "D", r), DS=link(a.job, f"s6_{name}", "DS", r))
        entry = dict(parent=code, records=n)
        try:
            c = compare(r, "DS", "D", settings, codes=[code], reference="D", proof=True)
            entry.update(DS_minus_D_top1=c["per_parent"][code]["top1"].get("delta"), failures=c["failures"])
        except Exception as exc:  # noqa: BLE001 - a missing phase is reported, never raised
            entry.update(error=repr(exc))
        # the damaged D against the clean D of the main root: how much the damage costs the generated winner
        clean = {s: json.loads((root / "D" / "records" / f"{code}_{s}.json").read_text()).get("top1_dE") if (root / "D" / "records" / f"{code}_{s}.json").exists() else None for s in (11, 12, 13)}
        dmg = {s: json.loads((r / "D" / "records" / f"{code}_{s}.json").read_text()).get("top1_dE") if (r / "D" / "records" / f"{code}_{s}.json").exists() else None for s in (11, 12, 13)}
        entry.update(top1_dE_clean_D=clean, top1_dE_damaged_D=dmg,
                     damage_cost_kj={s: (None if clean[s] is None or dmg[s] is None else dmg[s] - clean[s]) for s in clean})
        dam[name] = entry
    result["damaged"] = dam
    (a.out / "step6_summary.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps(dict(rule=rule, D_vs_C_top1=result["D_vs_C_current_start"]["summary"]["top1"], DS_vs_D_top1=result["DS_vs_D_on_candidate_DB"]["summary"]["top1"]), default=str))


if __name__ == "__main__":
    main()
