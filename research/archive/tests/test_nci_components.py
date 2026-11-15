"""Independent NCI component configuration and purely repulsive Q6 checks."""
from types import SimpleNamespace
from pathlib import Path
import json
import os
import subprocess
import sys
import numpy as np
import pytest
import torch
from jacobik.core.types import Atom,Bond
from jacobik.stats import nci,q6pmf


def contact():
    mol=SimpleNamespace(atoms=[Atom(1,'O',np.array([-1.,0,0])),Atom(2,'H',np.zeros(3)),Atom(3,'O',np.array([2.,0,0]))],
                        bonds=[Bond(1,2,1.)],metal_idx=-1,coord_sites=[])
    ge=SimpleNamespace(_q6_a=torch.tensor([1]),_q6_b=torch.tensor([2]),_q6_elem_keys=[('Hp','O')])
    nr=int(round((q6pmf.R_HI-q6pmf.R_LO)/q6pmf.DR_TAB))
    tab={('Hp','O','5',''):dict(w=[-2.]*nr,r0=1.)}
    rt=q6pmf.build_runtime(mol,ge,tab,torch.tensor([1.]),torch.tensor([.3]),'cpu')
    return mol,ge,rt,torch.tensor([[a.position for a in mol.atoms]],dtype=torch.float32)


def test_q6_is_purely_repulsive_and_nci_component_flags(monkeypatch):
    _,ge,rt,x=contact()
    for nci_on in (False, True):
        monkeypatch.setattr(nci,'NCI_ON',nci_on)
        for types in ((),('pp',),('chp',),('pp','chp'),('hb',)):
            monkeypatch.setattr(nci,'ACTIVE_TYPES',types)
            assert q6pmf.evaluate(rt,x,ge._q6_a,ge._q6_b).item()==0.0
            assert nci.hbond_replacement_enabled() == (nci_on and 'hb' in types)


def test_component_filter_does_not_change_extraction_candidates(monkeypatch):
    mol,_,_,_=contact();original=nci.candidates(mol)
    _,_,nr,_,_,na=nci._grid('hb')
    tables={('hb','*','*'):dict(s=np.ones((nr,na)).tolist())}
    monkeypatch.setattr(nci,'ACTIVE_TYPES',('pp',))
    assert nci.build_runtime(mol,tables,'cpu') is None
    assert nci.candidates(mol)==original
    rt=nci.build_runtime(mol,tables,'cpu',types=('hb',))
    assert rt['hb']['n']>0


@pytest.mark.parametrize('value',[['pp','pp'],['unknown'],'pp',None])
def test_profile_rejects_invalid_component_arrays(value):
    env=dict(os.environ,JACOBIK_PROFILE_OVERRIDES=json.dumps({'nci_types':value}),PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    p=subprocess.run([sys.executable,'-c','import jacobik.profile'],env=env,capture_output=True,text=True)
    assert p.returncode!=0 and 'nci_types must be an array' in p.stderr


def test_profile_snapshots_canonical_components():
    env=dict(os.environ,JACOBIK_PROFILE_OVERRIDES=json.dumps({'nci_types':['chp','pp']}),PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    p=subprocess.run([sys.executable,'-c',"from jacobik import profile; print(profile.get('nci_types'))"],env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert "('pp', 'chp')" in p.stdout
