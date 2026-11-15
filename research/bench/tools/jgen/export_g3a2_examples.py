"""Reproduce requested XYZ exports and diagnose saved FEKKAJ/QEZVEY winners."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root
    src=json.loads((root/'g3a2_filter_structures.json').read_text())
    energies=json.loads((root/'g3a2_all_energies.json').read_text())
    for arm in ('P2','C1'):
        rows=src['parents']['DAMVAQ'][arm]['conformers'];i=int(np.argmax([r['score'] for r in rows]));row=rows[i]
        symbols=row['symbols'];X=np.asarray(row['X_A']);assert row['symbols_order']=='builder/conf.complex.atoms' and X.shape==(len(symbols),3) and np.isfinite(X).all()
        energy=energies['parents']['DAMVAQ']['arms'][arm]['all_delta_E_unrel_kj'][i]
        comment=f'DAMVAQ {arm} seed=11 CPU pre-Li-cap production=615ae97d2350 key={row["key"]} score={row["score"]:.12g} MACE_deltaE_unrel_kJmol={energy:.9f} symbols_order=builder'
        path=root/f'DAMVAQ_G3a2_{arm}_s11_top1.xyz'
        path.write_text('\n'.join([str(len(symbols)),comment]+[f'{s:2s} {x:.12f} {y:.12f} {z:.12f}' for s,(x,y,z) in zip(symbols,X)])+'\n')
        back=path.read_text().splitlines();assert len(back)==len(symbols)+2
        assert [r.split()[0] for r in back[2:]]==symbols
        assert np.allclose([[float(v) for v in r.split()[1:]] for r in back[2:]],X,rtol=0,atol=1e-11)
    diagnostic={}
    for code in ('FEKKAJ','QEZVEY'):
        diagnostic[code]={}
        for arm in ('C0','C1','P2'):
            rows=src['parents'][code][arm]['conformers'];e=energies['parents'][code]['arms'][arm];i=int(np.argmax(e['scores']));row=rows[i]
            rec={'key':row['key'],'score':row['score'],'energy_kj':e['all_delta_E_unrel_kj'][i],'floor_kj':e['floor_delta_E_unrel_kj']}
            if code=='FEKKAJ':
                symbols=row['symbols'];X=np.asarray(row['X_A']);metal=symbols.index('Ru');cl=[i for i,z in enumerate(symbols) if z=='Cl'];assert len(cl)==2
                a,b=[X[i]-X[metal] for i in cl]
                rec.update(Cl_Ru_Cl_deg=float(np.degrees(np.arccos(np.clip(np.dot(a,b)/np.linalg.norm(a)/np.linalg.norm(b),-1,1)))),Cl_Cl_A=float(np.linalg.norm(X[cl[0]]-X[cl[1]])))
            diagnostic[code][arm]=rec
        for arm,key,name in [('P2',diagnostic[code]['C1']['key'],'C1_top1_identity_in_P2'),('C1',diagnostic[code]['P2']['key'],'P2_top1_identity_in_C1')]:
            rows=src['parents'][code][arm]['conformers'];e=energies['parents'][code]['arms'][arm]
            diagnostic[code][name]=[{'index':i,'score':r['score'],'energy_kj':e['all_delta_E_unrel_kj'][i]} for i,r in enumerate(rows) if r['key']==key]
    (root/'g3a2_failure_diagnostic.json').write_text(json.dumps(diagnostic,indent=2))

if __name__=='__main__':main()
