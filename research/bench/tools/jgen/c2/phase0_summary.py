"""Summarise registered placement and topology measurements without tuning."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from .phase0 import quantiles
from .rings import population


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    settings=json.loads(Path('research/campaigns/c2/C2_SETTINGS.json').read_text())
    codes, meta = population()
    parents = []; failures = []; groups = defaultdict(lambda: defaultdict(list))
    large = []; aromatic = []
    all_max = defaultdict(list)
    for code in codes:
        path = a.root / f'{code}.json'
        if not path.exists():
            failures.append(dict(code=code, error='missing_record')); continue
        data = json.loads(path.read_text())
        if data['status'] != 'ok':
            failures.append(dict(code=code, error=data['error'])); continue
        maxima = defaultdict(list)
        for seed in data['seeds']:
            for mode in seed['placement']:
                label = mode['kind']; maxima[label].extend(mode['row_max'])
                all_max[label].extend(mode['row_max'])
                for ligand in mode['ligands']:
                    for group in [ligand['kind'], 'all']:
                        for metric in ['offsets', 'row_max', 'md', 'md_error']:
                            groups[(label, group)][metric].extend(np.asarray(ligand[metric]).ravel().tolist())
        parents.append(dict(code=code,n_initial_rows=len(maxima['production']), **{k: quantiles(v) for k, v in maxima.items()}))
        large.extend(dict(code=code, **r) for r in data['large_rings'] if not r['on_macro_path'])
        if data.get('aromatic'): aromatic.append(dict(code=code, **data['aromatic']))
    n_bad = sum(p['production']['0.5'] > settings['placement_bad_offset_A'] for p in parents)
    old = float(np.median(all_max['production'])) if parents else None
    new = float(np.median(all_max['per_row_fit'])) if parents else None
    reduction = (old-new)/old if old else None
    result = dict(n_expected=len(codes), n_success=len(parents),n_initial_rows=len(all_max['production']), failures=failures,
        population=meta, parents=parents,
        class_quantiles=[dict(mode=k[0], ligand_class=k[1], metrics={m: quantiles(v) for m, v in ms.items()})
                         for k, ms in groups.items()],
        placement_decision=dict(parents_above_03=n_bad, fraction=n_bad/len(codes),
            original_median=old, fitted_median=new, reduction=reduction,
            build_start_kabsch=(n_bad/len(codes) >= settings['placement_parent_fraction'] or
                (reduction is not None and reduction >= settings['placement_relative_improvement']))),
        large_rings=dict(n_rings=len(large), n_ligands=len({(r['code'],r['ligand']) for r in large}),
            n_parents=len({r['code'] for r in large}), rings=large), aromatic=aromatic)
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result['placement_decision']))
    print('success',len(parents),'failures',len(failures),'large ligands',result['large_rings']['n_ligands'])


if __name__ == '__main__': main()
