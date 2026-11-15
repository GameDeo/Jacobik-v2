"""Size-matched rho on seen C-2 pools, fixed 20 draws per parent/seed."""
import argparse
import json
from pathlib import Path
import numpy as np
from .analyse import rank,bootstrap
from .rings import population


def matched_rho(scores_a, energy_a, scores_c, energy_c, draws=20, seed=11):
    arrays=[np.asarray(v,float) for v in [scores_a,energy_a,scores_c,energy_c]]
    sa,ea,sc,ec=arrays
    n=min(len(sa),len(sc));rng=np.random.default_rng(seed)
    if n<2:return None
    diffs=[]
    for _ in range(draws):
        ia=rng.choice(len(sa),n,replace=False) if len(sa)>n else np.arange(n)
        ic=rng.choice(len(sc),n,replace=False) if len(sc)>n else np.arange(n)
        a,c=rank(sa[ia],ea[ia]),rank(sc[ic],ec[ic])
        if a is not None and c is not None:diffs.append(a-c)
    return float(np.mean(diffs)) if diffs else None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    settings=json.loads(Path('research/campaigns/c2/C2_SETTINGS.json').read_text())
    results={};failures=[]
    for code in population()[0]:
        try:
            cr=[json.loads((a.root/'C/records'/f'{code}_{s}.json').read_text()) for s in [11,12,13]]
            ar=[json.loads((a.root/'S1/records'/f'{code}_{s}.json').read_text()) for s in [11,12,13]]
            if any(r['status']!='ok' for r in cr+ar):raise ValueError('failed generation')
            sc=[v for r in cr for v in r['scores']];ec=[v for r in cr for v in r['energies_kj']]
            vals=[matched_rho(r['scores'],r['energies_kj'],sc,ec) for r in ar]
            good=[v for v in vals if v is not None]
            results[code]=dict(delta=float(np.mean(good)) if good else None,n_seed_pairs=len(good),
                control_rows=len(sc),wide_rows=[len(r['scores']) for r in ar])
        except Exception as exc:failures.append(dict(code=code,reason=repr(exc)))
    summary=bootstrap([r['delta'] for c,r in results.items() if c not in settings['primary_exclusions'] and r['delta'] is not None],settings)
    a.out.write_text(json.dumps(dict(summary=summary,per_parent=results,failures=failures,
        draws=20,seed=11,status='seen-data informational reanalysis'),indent=2)+'\n')
    print(summary)


if __name__=='__main__':main()
