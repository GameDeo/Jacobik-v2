"""eta2 face-plane package, Step 4: draw the pre-registered holdout (rule fixed here, before any solve).

Universe: representatives of the frozen frame's `surface_fit` split (cleared for jgen development, LOCKED_SETS.md).
Excluded: cal, val (motion_validation), tail, the reserve, locked_exclusions.json, and every CSD code named in an
H-track or eta2 file (the auditor's H audit and directives, research/campaigns/H*, the eta2 diag scripts).
Strata, from the stored QO observations of the parent (geometry only):
  A  >= 1 eta2 face where each face atom carries >= 2 substituents (the shared plane is used)        up to 40
  B  eta2 faces, none of them qualifying (metal-normal fallback; only the refit entries change)       up to 20
  C  no haptic face at all (collateral: must be bit-identical between arms)                            10
Seed 31, sorted codes, numpy default_rng choice without replacement.
Output: research/data/jgen/campaign/eta2/holdout/ETA2_HOLDOUT.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from tools.jgen import inputs
from tools.jgen.bench import panels

MAIN = Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full')
OBS = MAIN / 'research/archive/data/refit/observations'
OUT = Path('research/data/jgen/campaign/eta2/holdout/ETA2_HOLDOUT.json')
SEED, N = 31, {'A': 40, 'B': 20, 'C': 10}
NAMED_FILES = ([MAIN / '.claude/audit_jgen_H0_H1_H2_2026-10-30.md'] + sorted((MAIN / '.claude').glob('directive_jgen_H*'))
               + sorted(Path('research/campaigns').glob('H*')) + sorted(Path('research/bench/tools/jgen/diag').glob('eta2_*.py'))
               + [MAIN / '.claude/PLAN_jgen.md'])


def named_codes():
    out = set()
    for p in NAMED_FILES:
        try:
            out |= set(re.findall(r'\b[A-Z]{6}(?:\d{2})?\b', p.read_text(errors='ignore')))
        except (OSError, UnicodeError):
            pass
    return out


def main():
    frame = json.load(open(inputs.FRAME_V1 / 'frame.json'))['main_allocation']['splits']
    manifest = json.load(open(inputs.FRAME_V1 / 'manifest.json'))
    locked = json.load(open(MAIN / 'research/bench/benchmarks/locked_exclusions.json'))
    universe = sorted({g['representative_code'] for g in frame['surface_fit']})
    excl = ({t['code'] for t in manifest['tasks'].values() if t.get('split') in ('calibration', 'motion_validation')}
            | {g['representative_code'] for s in ('calibration', 'motion_validation', 'reserve') for g in frame[s]}
            | set(panels.TAIL) | set(locked['reserve_all_codes']) | set(locked['motion_validation_codes']) | named_codes())
    pool = [c for c in universe if c not in excl]
    qo = pd.concat([pd.read_parquet(OBS / f'observations_QO_{s}.parquet', columns=['csd_code', 'k', 'raw_subs'])
                    for s in ('X1', 'X2', 'X3', 'LNf')], ignore_index=True)
    qo = qo[qo.csd_code.isin(pool)]
    seen = set(qo.csd_code)
    hap = qo[qo.k >= 12]
    e2 = hap[hap.k == 12].copy()
    e2['q'] = e2.raw_subs.map(lambda s: (lambda S: len(S) > 0 and min((S[:, 0] == 0).sum(), (S[:, 0] == 1).sum()) >= 2)(
        np.asarray(s, float).reshape(-1, 4)))
    A = sorted(set(e2[e2.q].csd_code))
    B = sorted(set(e2.csd_code) - set(A))
    C = sorted(seen - set(hap.csd_code))
    rng = np.random.default_rng(SEED)
    draw = {k: sorted(rng.choice(v, size=min(N[k], len(v)), replace=False).tolist()) for k, v in (('A', A), ('B', B), ('C', C))}
    res = dict(rule=__doc__, universe=len(universe), excluded=len(set(universe) & excl), pool=len(pool), with_qo=len(seen),
               stratum_sizes={'A': len(A), 'B': len(B), 'C': len(C)}, draw=draw)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    assert not OUT.exists(), f'{OUT} exists: the holdout is drawn once'
    OUT.write_text(json.dumps(res, indent=1) + '\n')
    print({k: v for k, v in res.items() if k != 'rule'})


if __name__ == '__main__':
    main()
