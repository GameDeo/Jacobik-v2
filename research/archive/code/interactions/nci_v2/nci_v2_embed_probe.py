"""Isolate embed preprocessing chemistry at the same ETKDG seed, no solve."""
from research.archive.code.interactions.nci_v2.nci_v2_followup_parity import deterministic_setup
import argparse,sys,tempfile,os
from pathlib import Path
import numpy as np
import torch
from research.archive.code.interactions.nci_v2.nci_v2_devio import MAIN,DEV,Run,load_json,write_json
from research.archive.code.interactions.nci_v2.nci_v2_feasibility import reference_stream
from research.archive.code.interactions.nci_v2.nci_v2_controls import budget


def run(spec_path,frame_path,out):
    import jacobik.builder.engine as engine
    from jacobik.inputs import build_builder_and_inputs
    from jacobik.topology.decompose import decompose_ligand
    from jacobik.ik.geometry import GeometryExtractor
    budget(start=True);temp=DEV/'tmp';temp.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(temp);os.environ['TMPDIR']=str(temp)
    engine.XTB_BIN=str(Path(sys.executable).parent/'xtb')
    frame=load_json(frame_path);parents=[p for p in frame['pilot_cohort']['parents'] if p['code'] in ('ICEYAQ','KOTJIP')];clean=MAIN/'data/refit/clean'
    inputs=[frame_path,Path(engine.XTB_BIN)]
    for p in parents:
        b=clean/('lnQM' if p['dataset']=='LNf' else 'tmQM');inputs += [b/'xyz'/f"tmQM_{p['shard']}.xyz",b/'bo'/f"tmQM_{p['shard']}.BO",b/'tmQM_y.csv']
    r=Run('followup-embed-chemistry',spec_path,inputs,dict(methods=['etkdg','mmff','gfn2'],seed=42,no_solve=True),out)
    if r.resumed:return r.path
    result={}
    for p,ref,error in reference_stream([dict(p,representative_code=p['code'],representative_shard=p['shard']) for p in parents],clean):
        if error:raise ValueError(error)
        builder,_,error=build_builder_and_inputs(ref)
        if error:raise ValueError(error)
        methods={}
        for method in ('etkdg','mmff','gfn2'):
            budget();shell,_,_=builder._build_shell(False,False,embed_seed=None,chelate_embed=True,embed=method)
            for lig in shell.ligands:decompose_ligand(shell,lig)
            ge=GeometryExtractor(shell,torch.device('cpu'))
            methods[method]=dict(bonds=[[b.i,b.j,float(b.order)] for b in shell.bonds],q5_keys=[repr(k) for k in ge._q5_keys])
        result[p['code']]=dict(methods=methods,ligands=[dict(smiles=l.smiles,donors=l.donors) for l in builder.ligands],
            mmff_equals_gfn2_bonds=methods['mmff']['bonds']==methods['gfn2']['bonds'],mmff_equals_gfn2_Q5=methods['mmff']['q5_keys']==methods['gfn2']['q5_keys'],
            etkdg_equals_mmff_bonds=methods['etkdg']['bonds']==methods['mmff']['bonds'])
        print(p['code'],{k:v for k,v in result[p['code']].items() if k not in ('methods','ligands')},flush=True)
    write_json(r.path/'reports/embed_chemistry.json',result);r.finish(dict(parents=len(parents),solve_calls=0))
    return r.path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--frame',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(run(a.spec,a.frame,a.out))
