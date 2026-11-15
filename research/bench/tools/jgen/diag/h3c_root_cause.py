"""H-3c Step 0: read saved NUWFAM geometry; no solve or GPU use.

The generated-order indices were independently verified against a freshly
prepared CPU scorer: metal=0, face=[1,2], pivot=1, per-H sx=40 and 41.
"""
import csv
import json
from pathlib import Path

import numpy as np

from jacobik.ik.angle_bends import lean_free_deg
from jacobik.stats.orientation import haptic_features


def measure(x):
    u, v = x[[40, 41]] - x[1]
    angle = float(np.degrees(np.arccos(np.clip(
        u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))))
    leans = [float(np.degrees(np.arcsin(np.clip(haptic_features(
        x[0], x[[1, 2]], False, [(x[1], x[h])])[1], -1, 1))))
        for h in (40, 41)]
    return angle, leans


def main():
    base = Path('research/data/jgen/campaign/H')
    sources = {
        'h3x_cap': base / 'h3x/nuwfam_2060/cap/xyz/NUWFAM',
        'h3x_control': base / 'h3x/nuwfam_2060/control/xyz/NUWFAM',
        'h3b_cap_normal': base / 'h3b/cap_normal/xyz/NUWFAM',
        'h3b_cap_allyl': base / 'h3b/cap_allyl/xyz/NUWFAM',
    }
    width = lean_free_deg()[2]
    rows = []
    for source, root in sources.items():
        files = sorted(root.glob('*.xyz'))
        assert files, root
        for p in files:
            lines = p.read_text().splitlines()
            header = json.loads(lines[1])
            symbols = [line.split()[0] for line in lines[2:]]
            assert symbols[0] == 'Zr'
            assert [symbols[i] for i in (1, 2, 40, 41)] == ['C', 'C', 'H', 'H']
            x = np.asarray([[float(v) for v in line.split()[1:4]] for line in lines[2:]])
            kind = 'pool' if '_row' in p.name else 'top1' if '_top1' in p.name else 'floor'
            angle, leans = measure(x)
            rows.append(dict(source=source, path=str(p), kind=kind,
                             angle_deg=angle, h40_face_lean_deg=leans[0],
                             h41_face_lean_deg=leans[1], width_deg=width,
                             collapsed_lt15=angle < 15,
                             any_exceeds_width=max(map(abs, leans)) > width,
                             row_index=header.get('row_index')))
        for p in sorted(root.glob('*_pool.json')):
            m = json.loads(p.read_text())
            order = np.argsort(m['gen_to_ref_idx'])
            assert np.asarray(m['ref_symbols'])[order].tolist() == symbols
            angle, leans = measure(np.asarray(m['ref_X'])[order])
            rows.append(dict(source=source, path=str(p), kind='crystal',
                             angle_deg=angle, h40_face_lean_deg=leans[0],
                             h41_face_lean_deg=leans[1], width_deg=width,
                             collapsed_lt15=angle < 15,
                             any_exceeds_width=max(map(abs, leans)) > width,
                             row_index=None))
    summary = {}
    for source in sources:
        selected = [r for r in rows if r['source'] == source]
        pool = [r for r in selected if r['kind'] == 'pool']
        bad = [r for r in pool if r['collapsed_lt15']]
        checks = [r for r in selected if r['kind'] in ('top1', 'floor', 'crystal')]
        summary[source] = dict(
            pool_rows=len(pool), collapsed_rows=len(bad),
            collapsed_prediction_passes=sum(r['any_exceeds_width'] for r in bad),
            collapsed_max_abs_lean_deg=max((max(abs(r['h40_face_lean_deg']), abs(r['h41_face_lean_deg'])) for r in bad), default=None),
            min_geminal_angle_deg=min(r['angle_deg'] for r in pool),
            top1_floor_crystal_rows=len(checks),
            top1_floor_crystal_outside_width=sum(r['any_exceeds_width'] for r in checks))
    out = base / 'h3c'
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'step0_rows.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {'coordinate': 'degrees(asin(orientation.haptic_features per-H bend))',
              'indices': {'metal': 0, 'face': [1, 2], 'pivot': 1, 'hydrogens': [40, 41]},
              'width_deg': width, 'summary': summary,
              'step0_pass': all(s['collapsed_prediction_passes'] == s['collapsed_rows'] for s in summary.values())}
    (out / 'step0_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
