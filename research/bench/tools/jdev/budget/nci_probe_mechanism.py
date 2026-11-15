"""Read-only CH-pi direction/multiplicity falsifications on physical probes.

Post-hoc ablations test mechanisms; they do not calibrate or install a term.
Changing candidate counting would require consistent extraction and refitting.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
from common import REPO,stats_from_db
from pools import setup
from nci_targeted_rescore import CANDIDATE
from nci_calibration import OUT,MONOMERS,topology,write


def run():
    import torch
    from rdkit import Chem
    from research.archive.code.interactions import nci
    from jacobik.topology.graph import get_ligand_neighbors
    torch.set_num_threads(1);stats=stats_from_db(CANDIDATE);report={}
    for name in ('methane','ethane'):
        path=OUT/f'chp_{name}_benzene.npz';meta=json.loads(path.with_suffix('.json').read_text())
        rdA=Chem.AddHs(Chem.MolFromSmiles(MONOMERS[name]));rdB=Chem.AddHs(Chem.MolFromSmiles(MONOMERS['benzene']))
        mol,_=topology(rdA,rdB);nb=get_ligand_neighbors(mol.atoms,mol.bonds,mol.metal_idx)
        rt=nci.build_runtime(mol,stats.nci_tables(),'cpu',types=('chp',))['chp']
        with np.load(path) as z:x=z['X'];old=z['q6_old'];cached=z['score_chp']
        xt=torch.tensor(x,dtype=torch.float32);r,a=nci.features('chp',xt,rt['idx'])
        score=nci._bilinear(rt['T'],rt['row'],r,a,rt['grid']).numpy()
        np.testing.assert_allclose(score.sum(1),cached,atol=2e-6)
        pos={atom.idx:i for i,atom in enumerate(mol.atoms)}
        parent=np.array([pos[nb[mol.atoms[int(h)].idx][0]] for h in rt['idx']['h']])
        h=rt['idx']['h'].numpy();ring=rt['idx']['ring'].numpy()
        centers=nci._ring_frame(xt,rt['idx']['rings'])[0].numpy()[:,ring]
        u=x[:,parent]-x[:,h];v=centers-x[:,h]
        cos=np.einsum('bni,bni->bn',u,v)/np.linalg.norm(u,axis=-1)/np.linalg.norm(v,axis=-1)
        positive=score.clip(min=0);negative=score.clip(max=0)
        directional=negative+positive*np.clip(-cos,0,1)**2
        groups=sorted(set(zip(parent,ring)))
        sibling=np.stack([np.max(directional[:,(parent==p)&(ring==r)],axis=1) for p,r in groups],axis=1).sum(1)
        scores=dict(current=cached,halfspace=(negative+positive*(cos<0)).sum(1),directional=directional.sum(1),directional_sibling_max=sibling)
        predictions={k:old-2.479*s for k,s in scores.items()};predictions['off']=old
        labels=meta['labels'];scans=sorted(set(l['scan'] for l in labels));result={}
        for ext in ('mace64','gfn2'):
            e=np.array(json.loads(path.with_suffix('.'+ext+'.json').read_text())['interaction']);metrics={}
            for key,pred in predictions.items():
                de=e.copy();dp=pred.copy();regret=[]
                for scan in scans:
                    ids=np.array([i for i,l in enumerate(labels) if l['scan']==scan]);far=next(i for i in ids if labels[i]['far'])
                    de[ids]-=e[far];dp[ids]-=pred[far]
                    near=ids[[not labels[i]['far'] for i in ids]]
                    regret.append(float(e[near[np.argmin(pred[near])]]-e[near].min()))
                good=(de<50)&np.array([not l['far'] for l in labels])
                metrics[key]=dict(rmse=float(np.sqrt(np.mean((de[good]-dp[good])**2))),radial_regret=float(np.mean(regret)))
            result[ext]=metrics
        # Nominal H remains at exactly the same r/normal angle as its parent
        # rotates: the runtime descriptor cannot see the tested C-H direction.
        ids=[i for i,l in enumerate(labels) if l['approach']==0 and abs(l['r']-3.)<1e-8]
        result['matched_nominal_contact']=[dict(direction=labels[i]['angle'],score_nominal=float(score[i,0]),score_total=float(cached[i]),
            **{ext:float(json.loads(path.with_suffix('.'+ext+'.json').read_text())['interaction'][i]) for ext in ('mace64','gfn2')}) for i in ids]
        report[name]=result
    write(OUT/'chp_mechanism.json',dict(cases=report,scope='Post-hoc mechanism diagnostics on independent rigid probes. No fitted coefficients. Halfspace gates positive rewards at DHA>=90 degrees; directional uses max(-cos(DHA),0)^2. Sibling max is per carbon/ring, not per hydrogen. Neither constitutes a calibrated refit.'))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':setup();run()
