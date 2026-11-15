"""Fixed-panel G-3c0 summaries, retaining missing/failed units explicitly."""
import argparse
import json
from pathlib import Path
import numpy as np


def distribution(values):
    a = np.asarray([v for v in values if v is not None], dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {'n': 0}
    return dict(n=len(a), mean=float(a.mean()), **dict(zip(
        ['min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'],
        map(float, np.quantile(a, [0, .05, .25, .5, .75, .95, 1])))))


def paired(records, control, parents, seeds, field, selected_keys=None):
    units = []; means = []; per_parent = {}; unavailable = []
    for code in parents:
        expected_seeds = [s for s in seeds if selected_keys is None or (code, s) in selected_keys]
        if not expected_seeds:
            continue
        v = []
        for seed in expected_seeds:
            a = records.get((code, seed)); c = control.get((code, seed))
            if a is None or c is None:
                unavailable.append([code, seed]); continue
            if field == 'obscene_rate':
                x = np.mean(np.asarray(a['energies_kj']) - a['ref_energy_kj'] > 200)
                y = np.mean(np.asarray(c['energies_kj']) - c['ref_energy_kj'] > 200)
            else:
                x, y = a[field], c[field]
            v.append(float(x-y)); units.append(float(x-y))
        per_parent[code] = dict(differences=v, mean=float(np.mean(v)) if v else None,
                                complete=len(v) == len(expected_seeds))
        # A parent's CI contribution requires all three paired seeds; missing
        # parents remain present in per_parent and unavailable, never replaced.
        if len(v) == len(expected_seeds):
            means.append(np.mean(v))
    means = np.asarray(means)
    if not len(means):
        return dict(n_parents=0, unavailable=unavailable, per_parent=per_parent)
    rng = np.random.default_rng(11)
    boot = means[rng.integers(0, len(means), size=(10000, len(means)))].mean(1)
    return dict(mean=float(means.mean()), ci95=list(map(float, np.quantile(boot, [.025, .975]))),
                median_unit_difference=float(np.median(units)), median_parent_difference=float(np.median(means)),
                n_parents=len(means), n_paired_units=len(units),
                parents_improved=int(((means > 0) if field == 'top1_score' else (means < 0)).sum()),
                parents_worsened=int(((means < 0) if field == 'top1_score' else (means > 0)).sum()), parents_tied=int((means == 0).sum()),
                unavailable=unavailable, per_parent=per_parent)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--roots', nargs='+', required=True)
    ap.add_argument('--out', required=True); args = ap.parse_args()
    contract = json.loads(Path('research/campaigns/g/evaluation_contract_g3c0.json').read_text())
    parents, seeds, arms = contract['parents'], contract['seeds'], list(contract['arms'])
    all_records = {arm: {} for arm in arms}; failures = []; missing = []
    for root in args.roots:
        for arm in arms:
            for path in (Path(root)/arm/'records').glob('*.json'):
                r = json.loads(path.read_text()); key = (r['code'], r['seed'])
                assert key[0] in parents and key[1] in seeds
                assert key not in all_records[arm], (arm, key, 'duplicate record')
                all_records[arm][key] = r
    valid = {arm: {} for arm in arms}
    for arm in arms:
        for code in parents:
            for seed in seeds:
                r = all_records[arm].get((code, seed))
                if r is None:
                    missing.append([arm, code, seed]); continue
                if r['status'] != 'ok' or r['n_finite'] != r['n_rows']:
                    failures.append(dict(arm=arm, code=code, seed=seed, reason=r.get('reason'),
                                         n_finite=r['n_finite'], n_rows=r['n_rows']))
                else:
                    valid[arm][(code, seed)] = r
    result = dict(contract=contract, expected_records=len(parents)*len(seeds)*len(arms),
                  actual_records=sum(map(len, all_records.values())), missing=missing, failures=failures,
                  collection_complete=not missing, arms={}, start_digest_mismatches=[])
    for arm in arms:
        rows = []; unit_fractions = []; walls = {}; top_rows = []
        for (code, seed), r in valid[arm].items():
            c = valid['C1'].get((code, seed))
            if c and r['initial_theta_sha256'] != c['initial_theta_sha256']:
                result['start_digest_mismatches'].append([arm, code, seed])
            de = np.asarray(r['energies_kj']) - r['ref_energy_kj']
            diagnostics = r.get('distortion') or []
            assert len(diagnostics) == len(de), (arm, code, seed, 'missing distortion rows')
            annotated = [dict(d, dE=float(e), code=code, seed=seed) for d,e in zip(diagnostics,de)]
            rows.extend(annotated); top_rows.append(annotated[int(np.argmax(r['scores']))])
            changes = [d['class_changed'] for d in diagnostics if d['class_changed'] is not None]
            unit_fractions.append(dict(code=code, seed=seed, obscene=float(np.mean(de > 200)),
                                       class_changed=float(np.mean(changes)) if changes else None))
            walls[code] = r['wall_solve_s']
        changes = [d['class_changed'] for d in rows if d['class_changed'] is not None]
        wall = sum(walls.values())
        result['arms'][arm] = dict(n_units=len(valid[arm]), n_rows=len(rows), wall_solve_s=wall,
            wall_by_parent=walls,
            absolute_unit_means={f: float(np.mean([r[f] for r in valid[arm].values()])) if valid[arm] else None
                                 for f in ['top1_dE', 'floor_dE', 'top1_score']},
            shape=distribution([r['shape_deviation'] for r in rows]),
            max_q2_angle_deg=distribution([r['max_q2_deviation_deg'] for r in rows]),
            top1_shape=distribution([r['shape_deviation'] for r in top_rows]),
            top1_max_q2_angle_deg=distribution([r['max_q2_deviation_deg'] for r in top_rows]),
            class_changed_rate=float(np.mean(changes)) if changes else None,
            class_unavailable=sum(d['class_changed'] is None for d in rows),
            geometry_changed_rate=float(np.mean([r['starting_geometry'] != r['returned_geometry'] for r in rows])) if rows else None,
            obscene_rate=float(np.mean([r['dE'] > 200 for r in rows])) if rows else None,
            dE_distribution=distribution([r['dE'] for r in rows]),
            unit_fractions=unit_fractions,
            paired={f: paired(valid[arm], valid['C1'], parents, seeds, f)
                    for f in ['top1_dE','floor_dE','top1_score','obscene_rate']},
            strata={label: {f: paired(valid[arm], valid['C1'], parents, seeds, f,
                                      selected_keys={key for key,r in valid['C1'].items()
                                                     if (r['top1_dE'] > 200) == obscene})
                             for f in ['top1_dE','floor_dE','top1_score','obscene_rate']}
                    for label, obscene in [('control_non_obscene',False),('control_obscene',True)]})
    dest = Path(args.out); dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2)+'\n')
    print('records',result['actual_records'],'/',result['expected_records'],'failures',failures,'missing',len(missing))
    for arm, v in result['arms'].items():
        print(arm, json.dumps({k:v[k] for k in ['n_units','n_rows','wall_solve_s','shape','max_q2_angle_deg','class_changed_rate','geometry_changed_rate','obscene_rate']}))
        for f,p in v['paired'].items():
            print(' ',f,json.dumps({k:x for k,x in p.items() if k not in ['per_parent','unavailable']}))


if __name__ == '__main__':
    main()
