"""Render frozen G-3c0 measures from the retained analysis artifact."""
import argparse
import json
from pathlib import Path


def estimate(p, scale=1, digits=2):
    if not p.get('n_parents'):
        return 'unavailable'
    return f"{p['mean']*scale:+.{digits}f} [{p['ci95'][0]*scale:+.{digits}f}, {p['ci95'][1]*scale:+.{digits}f}]"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--analysis', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); data = json.loads(Path(a.analysis).read_text()); arms = data['arms']
    lines = ['### Completed results', '',
             f"Records **{data['actual_records']}/{data['expected_records']}**; missing {len(data['missing'])}; failed {len(data['failures'])}; starting-θ digest mismatches {len(data['start_digest_mismatches'])}. All 24 fixed parents remain in the ledger.", '',
             'All energy changes are arm minus paired C1, kJ/mol; negative is better. CIs resample parents, with seeds averaged within parent. Median is the median paired parent/seed-unit difference.', '',
             '| Arm | Paired parents / units | Top1 mean [95% CI] | Top1 median | Parents improved | Floor mean [95% CI] | Floor median |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name,v in arms.items():
        t,f = v['paired']['top1_dE'],v['paired']['floor_dE']
        lines.append(f"| {name} | {t.get('n_parents',0)} / {t.get('n_paired_units',0)} | {estimate(t)} | {t.get('median_unit_difference',float('nan')):+.2f} | {t.get('parents_improved',0)} | {estimate(f)} | {f.get('median_unit_difference',float('nan')):+.2f} |")
    lines += ['', '| Arm | Mean top1 score | Paired score change [95% CI] | Solver/assembly wall (s) | Wall / C1 | Returned rows |',
              '|---|---:|---:|---:|---:|---:|']
    for name,v in arms.items():
        lines.append(f"| {name} | {v['absolute_unit_means']['top1_score']:.6f} | {estimate(v['paired']['top1_score'], digits=6)} | {v['wall_solve_s']:.2f} | {v['wall_solve_s']/arms['C1']['wall_solve_s']:.3f} | {v['n_rows']} |")
    lines += ['', 'Wall sums each parent once (its three seeds were solved together). Native score-only arms execute eagerly; C1/P2/R1 search uses CUDA replay. These are implementation timings, not a clean optimizer-cost comparison.', '',
              '| Arm | Row class-change rate | Geometry-label change | Class unavailable | Obscene rows (>200 kJ/mol) | Paired obscene-rate change (pp) [95% CI] |',
              '|---|---:|---:|---:|---:|---:|']
    for name,v in arms.items():
        lines.append(f"| {name} | {100*v['class_changed_rate']:.2f}% | {100*v['geometry_changed_rate']:.2f}% | {v['class_unavailable']} | {100*v['obscene_rate']:.2f}% | {estimate(v['paired']['obscene_rate'],100)} |")
    lines += ['', 'Row-pooled fractions weight parents by their returned counts; the paired rate change instead gives parents equal weight. They need not subtract to the same number.', '']
    for key,title in [('shape','Shape deviation (the shape constraint statistic, dimensionless)'),
                      ('max_q2_angle_deg','Maximum angle deviation from nearest Q2 mean (degrees)')]:
        lines += [f'### {title}', '', '| Arm | Mean | Min | p05 | p25 | Median | p75 | p95 | Max |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for name,v in arms.items():
            d=v[key]; lines.append('| '+name+' | '+' | '.join(f'{d[k]:.4f}' for k in ['mean','min','p05','p25','median','p75','p95','max'])+' |')
        lines += ['']
    lines += ['| Arm | Top1 shape median / p95 / max | Top1 max-angle median / p95 / max (degrees) |',
              '|---|---:|---:|']
    for name,v in arms.items():
        cols = [' / '.join(f'{v[key][k]:.4f}' for k in ['median','p95','max'])
                for key in ['top1_shape','top1_max_q2_angle_deg']]
        lines.append('| '+name+' | '+' | '.join(cols)+' |')
    lines += ['', '### Control-energy strata', '',
              'Strata are defined per seed by C1 top1 ≤200 or >200 kJ/mol. A parent can contribute to both; within each stratum, available constituent seeds are averaged before resampling parents.', '',
              '| Stratum | Arm | Parents / units | Top1 mean [95% CI] | Floor mean [95% CI] | Top1 median |',
              '|---|---|---:|---:|---:|---:|']
    for s,label in [('control_non_obscene','C1 non-obscene'),('control_obscene','C1 obscene')]:
        for name,v in arms.items():
            t,f=v['strata'][s]['top1_dE'],v['strata'][s]['floor_dE']
            lines.append(f"| {label} | {name} | {t.get('n_parents',0)} / {t.get('n_paired_units',0)} | {estimate(t)} | {estimate(f)} | {t.get('median_unit_difference',float('nan')):+.2f} |")
    lines += ['', '### Fixed-panel failures and completeness', '',
              '```json', json.dumps(dict(missing=data['missing'],failures=data['failures'],
                                          start_digest_mismatches=data['start_digest_mismatches']),indent=2), '```', '']
    Path(a.out).write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
