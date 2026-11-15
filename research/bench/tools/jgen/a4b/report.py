"""Render the concise A4b checkpoint report from measured tables and provenance."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path

DB_REL = "data/db/constraints_2026-11-12.npz"  # production DB since the DS fold-in (2026-11-12)

def sha(path):
    p=Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"

def load_csv(path):
    with Path(path).open(newline="") as f: return list(csv.DictReader(f))

def fmt(x, digits=2):
    if x is None: return "NA"
    try: return f"{float(x):.{digits}f}"
    except (ValueError,TypeError): return "NA"

def stat(x):
    if not x or x.get("mean") is None: return "NA"
    return f"{fmt(x['mean'])} [{fmt(x.get('lo'))}, {fmt(x.get('hi'))}], median {fmt(x.get('median'))}"

def build_report(data, repo, out):
    root=Path(data); repo=Path(repo)
    summary=json.loads((root/'summary.json').read_text())
    units=load_csv(root/'units.csv'); rows=load_csv(root/'rows.csv')
    split=json.loads((root/'vm_split.json').read_text())
    meta=json.loads((root/'run_meta.json').read_text()) if (root/'run_meta.json').exists() else {}
    recs=[]
    for p in (root/'records').rglob('*.json'):
        try: recs.append(json.loads(p.read_text()))
        except Exception: pass
    fingerprints=sorted({r.get('fingerprint','missing') for r in recs})
    # the DB that actually ran, as the records state it; the repo default only if no record carries it
    run_dbs=sorted({r.get('constraints_db_sha256') for r in recs if r.get('constraints_db_sha256')})
    db=repo/DB_REL; dbmeta=db.with_name(db.stem+'.meta.json')
    db_line=(f"- Constraints DB SHA-256 (from the records): `{', '.join(run_dbs)}`." if run_dbs else
             f"- Constraints DB SHA-256 (repo default `{DB_REL}`, no record states it): `{sha(db)}`; meta: "
             f"`{sha(dbmeta) if dbmeta.exists() else 'embedded in the container'}`.")
    out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    lines=["# A4b production baseline", "", "## Provenance", "",
      f"- Commit: `{meta.get('commit','unknown')}`; fingerprint(s): `{', '.join(fingerprints) or 'none'}`.",
      db_line,
      f"- VM: {meta.get('vm_type','NVIDIA L4')}; seeds 11, 12, 13; panels cal and tail; arm control; production flags/defaults.",
      f"- Unique parents attempted: {len(split['cal_parents']) + len(split['tail_only_A']) + len(split['tail_only_B'])}; units attempted by panel: cal {summary['cal']['denominators']['units_attempted']}, tail {summary['tail']['denominators']['units_attempted']}; ok/failed: cal {summary['cal']['denominators']['units_ok']}/{summary['cal']['denominators']['units_failed']}, tail {summary['tail']['denominators']['units_ok']}/{summary['tail']['denominators']['units_failed']}.",
      f"- Colab spend: {fmt(meta.get('full_run_colab_units'),3)} units for the successful full run; {fmt(meta.get('task_total_colab_units'),3)} units for this task including smoke and the discarded attempt. Aggregate VM-active time: {fmt(meta.get('vm_wall_min_sum'),1)} min; elapsed full-run wall: {fmt(meta.get('elapsed_wall_min'),1)} min; solve wall sum: {fmt(meta.get('solve_wall_s'),1)} s; xTB relaxation wall sum: {fmt(meta.get('relax_wall_s'),1)} s; CUDA MACE wall: {fmt(meta.get('mace_wall_s'),1)} s.",
      "", "## Q1: coverage and relaxed ranking", ""]
    for panel in ('cal','tail'):
        sm=summary[panel]
        lines += [f"### {panel}", "", "| Bucket | Share (mean, 95% parent-bootstrap CI; parent median) |", "|---|---:|"]
        for key in ('u_bucket','r_bucket'):
            for b,v in sm[key].items(): lines.append(f"| {b} | {stat(v)} |")
        lines += ["", "Seed consistency counts per parent (number of seeds in each bucket):", "",
                  "| Bucket | 3/3 | 2/3 | 1/3 | 0/3 |", "|---|---:|---:|---:|---:|"]
        for key in ('u_bucket','r_bucket'):
            for b,counts in sm['seed_consistency'][key].items():
                lines.append(f"| {b} | {counts['3/3']} | {counts['2/3']} | {counts['1/3']} | {counts['0/3']} |")
        lines.append("")
    lines += ["## Q2: score inversions", ""]
    for panel in ('cal','tail'):
        sm=summary[panel]
        lines += [f"### {panel}", "", f"- Obscene share: {stat(sm['obscene'])}.",
          f"- Top-1 inversion share: {stat(sm['top1_inversion'])}.",
          f"- Pair inversion rate: {stat(sm['pair_inversion_rate'])}.",
          f"- Ranking loss: {stat(sm['ranking_loss'])} kJ/mol.", "",
          "Top-1 inversion units:", ""]
        inv=[u for u in units if u.get('panel')==panel and u.get('top1_inversion') in ('True','true','1')]
        lines.extend([f"- {u['code']} seed {u['seed']}" for u in inv] or ["- None"])
        ranked=[r for r in rows if r.get('panel')==panel and r.get('is_top1') in ('True','true','1')]
        ranked.sort(key=lambda r: float(r.get('dE_vs_floor') or float('-inf')),reverse=True)
        lines += ["", "20 largest unit ranking losses:", "", "| Code | Seed | Ranking loss | Top-1 dE |", "|---|---:|---:|---:|"]
        # Ranking loss is unit-level; top row dE supplies the matching top-1 error.
        unitmap={(u['code'],u['seed'],u['panel']):u for u in units}
        candidates=[]
        for u in units:
            if u.get('panel')==panel and u.get('ranking_loss') not in ('',None):
                candidates.append(u)
        candidates.sort(key=lambda u: float(u['ranking_loss']),reverse=True)
        for u in candidates[:20]:
            lines.append(f"| {u['code']} | {u['seed']} | {fmt(u['ranking_loss'])} | {fmt(u.get('top1_dE'))} |")
        lines.append("")
    lines += ["## Q3: strain and basin offset", ""]
    for panel in ('cal','tail'):
        sm=summary[panel]
        top_changes=sum(u.get('panel')==panel and u.get('contact_change_top1') in ('True','true','1') for u in units)
        lines += [f"### {panel}", "", f"- Top-1 strain: {stat(sm['strain_top1'])} kJ/mol.",
          f"- Top-1 basin offset: {stat(sm['offset_top1'])} kJ/mol.",
          f"- Top-1 strain share: {stat(sm['strain_share'])}.",
          f"- Top-1 contact changes: {top_changes} units.", ""]
    lines += ["## Sensitivity", ""]
    for panel in ('cal','tail'):
        lines += [f"### {panel}", "", "| Hit radius / equivalent offset | R-TOP1 share | R-RANK share | R-MISS share |", "|---|---:|---:|---:|"]
        for name,buckets in summary[panel]['sensitivity'].items():
            lines.append(f"| {name} | " + " | ".join(stat(buckets[b]) for b in ('R-TOP1','R-RANK','R-MISS')) + " |")
        lines.append("")
    lines += ["## Failures and deviations", ""]
    for panel in ('cal','tail'):
        lines.append(f"- {panel}: `{json.dumps(summary[panel]['failures'],sort_keys=True)}`; fallback exclusions: `{json.dumps(summary[panel]['denominators']['parents_excluded'])}`.")
    fallback=json.loads((root/'fallback_codes.json').read_text()) if (root/'fallback_codes.json').exists() else []
    lines.append(f"- Embed fallback parents: {', '.join(fallback) if fallback else 'none detected in saved logs'}.")
    for code in fallback:
        bad=[r for r in recs if r.get('code')==code and r.get('status')!='ok']
        seeds=','.join(str(r.get('seed')) for r in sorted(bad,key=lambda x:int(x.get('seed',0)))) or 'none'
        lines.append(f"- {code}: {len(bad)} failed of 3 attempted seeds ({seeds}); no valid pool rows/manifests were produced, per Amendment 2.")
    if meta.get('interrupted_attempt_spend_units'):
        lines.append(f"- A first full-run attempt was discarded before fetch after a controller exception; it consumed {fmt(meta['interrupted_attempt_spend_units'],3)} units and is not included in these measurements. The successful retry used the same committed code.")
    if meta.get('relax_overlapped_with_solve') is False:
        lines.append("- Scheduling deviation: each VM began the CPU xTB relaxation batch after its complete GPU solve batch, so xTB did not overlap the GPU solves as §3 Step 3 specified. CUDA MACE evaluation also began after each VM's solves.")
    disagreements=json.loads((root/'coverage_disagreements.json').read_text()) if (root/'coverage_disagreements.json').exists() else []
    lines.append(f"- U-bucket disagreements with bench coverage fields: {len(disagreements)}; see `coverage_disagreements.json`.")
    charge_failures={k:v for k,v in json.loads((root/'charge_mult.json').read_text()).items() if v.get('status')=='failed'}
    lines.append(f"- Charge/multiplicity lookup failures: {json.dumps(charge_failures,sort_keys=True) if charge_failures else 'none'}.")
    lines.append(f"- Missing or incomplete records: see `{root/'missing_units.json'}`.")
    out.write_text("\n".join(lines)+"\n")
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--repo',required=True); p.add_argument('--out',required=True); a=p.parse_args(); print(build_report(a.data,a.repo,a.out))
