"""Cost and usefulness of T from the immutable, seen C-2 exports."""
import argparse
import json
from pathlib import Path
from .rings import population


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    extra=[];sizes={k:0 for k in ['C','T']};seen={k:set() for k in sizes};failures=[]
    for code in population()[0]:
        for seed in [11,12,13]:
            try:
                records={k:json.loads((a.root/k/'records'/f'{code}_{seed}.json').read_text()) for k in sizes}
                if any(r['status']!='ok' for r in records.values()):
                    failures.append(dict(code=code,seed=seed,reason='generation_failed'));continue
                for k,r in records.items():
                    p=Path(r['pool_manifest']);m=json.loads(p.read_text())
                    for x in [p]+[p.parent/row['file'] for row in m['rows']]:
                        if x not in seen[k]:sizes[k]+=x.stat().st_size;seen[k].add(x)
                c,t=records['C'],records['T'];n=len(c['scores'])
                if len(t['scores'])>n:
                    assert t['scores'][:n]==c['scores']
                    e=t['energies_kj'][n:];ref=t['ref_energy_kj']
                    extra.append(dict(code=code,seed=seed,added=len(e),best_extra_dE=min(e)-ref,
                        control_floor=c['floor_dE'],extra_beats_control_floor=min(e)-ref<c['floor_dE'],
                        best_extra_rmsd=min(t['coverage']['rmsd_A'][n:]),control_min_rmsd=min(c['coverage']['rmsd_A'])))
            except Exception as exc:failures.append(dict(code=code,seed=seed,reason=repr(exc)))
    a.out.write_text(json.dumps(dict(storage_bytes=sizes,ratio=sizes['T']/sizes['C'],extra=extra,failures=failures),indent=2)+'\n')


if __name__=='__main__':main()
