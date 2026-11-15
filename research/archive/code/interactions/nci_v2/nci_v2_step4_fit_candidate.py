"""Fit response scales from frozen Step 4 labels and write candidate DB/sidecar."""
from __future__ import annotations
import copy,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from research.archive.code.interactions.nci_v2.nci_v2 import SurfaceArtifact
from research.archive.code.interactions.nci_v2.nci_v2_response import fit_response_scales,response_diagnostics,jsonable
from research.archive.code.interactions.nci_v2.nci_v2_step4_label_gpu import safe_load_mol,safe_smiles_index
from tools.jgen.diag import d14_input_consistency as d14


def run(frame_path,spec_path,clean_root,frozen_surface_path,labels_path,db_path,out_dir):
    frame=json.loads(Path(frame_path).read_text());spec=json.loads(Path(spec_path).read_text())
    labels_doc=json.loads(Path(labels_path).read_text()); rows=labels_doc['rows']
    parents=frame['main_allocation']['splits']['calibration']
    by_code={p['representative_code']:p for p in parents}
    groups={str(p['group_id']) for p in parents}
    frozen=json.loads(Path(frozen_surface_path).read_text())
    surfaces={f:SurfaceArtifact(**a) for f,a in frozen.get('surface_artifacts',frozen).items()}
    a3=json.loads(Path('data_dev/nci_v2_real/step4/a3_coverage_floored.json').read_text())
    disabled_components=[f'{r["family"]}:{r["root"]}:{c["component"]}' for r in a3['root_coverage']
              for c in r['ref_only_components'] if not c['passed']]
    artifact=fit_response_scales(surfaces,rows,groups,dict(lambda_F=1.,seed=0,bootstrap_samples=4000,
        coverage_frozen=True,coverage_provenance='data_dev/nci_v2_real/step4/a3_coverage_floored.json',
        a3_disabled_ref_only_components=disabled_components,
        a3_disabled_ref_only_cells=sum(r['disabled_ref_only_cells'] for r in a3['root_coverage'])),
        progress_callback=lambda i,n: print(f'bootstrap {i}/{n}',flush=True))
    diag=response_diagnostics(rows,artifact,groups=groups)

    smiles=safe_smiles_index(Path(clean_root)/'tmQM'/'tmQM_y.csv')
    d14_labels={}
    for code,parent in sorted(by_code.items()):
        mol=safe_load_mol(code,parent['representative_shard'],parent['dataset'],Path(clean_root),smiles)
        if mol is not None: d14_labels[code]=d14.classify(mol)
    fam_scales=artifact.family_scales
    grouped=defaultdict(list)
    for r in rows:
        if 'y' not in r: continue
        code=r.get('source_code')
        lab=d14_labels.get(code,{}).get('label','unavailable')
        pred=sum(float(v)*fam_scales.get(f,artifact.gamma_shared) for f,v in r.get('D_f',{}).items())
        grouped[lab].append((float(r['y']),pred))
    d14_residual={}
    for lab,values in grouped.items():
        actual=np.asarray([a for a,_ in values]);pred=np.asarray([p for _,p in values]);res=actual-pred
        d14_residual[lab]=dict(n=len(values),bias_kj_mol=float(res.mean()),rmse_kj_mol=float(np.sqrt(np.mean(res**2))))

    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    sidecar=out/'candidate_nci_v2.json'
    sidecar.write_text(json.dumps(jsonable(artifact),separators=(',',':'))+'\n')
    db_src=Path(db_path);candidate_db=out/'candidate_constraints.json'
    candidate_db.write_bytes(db_src.read_bytes())
    report=dict(schema='nci_v2_step4_part_c_v1',status='calibration_only_descriptive',
        groups=len(groups),motion_label_rows=len(rows),gamma_shared=artifact.gamma_shared,
        family_scales=artifact.family_scales,selected_mode=artifact.selected_mode,
        response_metadata=artifact.metadata,singular_values=artifact.singular_values,
        condition_number=jsonable(artifact.condition_number),ref_only_traversal_passed=artifact.ref_only_traversal_passed,
        diagnostics=diag,d14_residual_split=d14_residual,d14_labels=d14_labels,
        a3_coverage_sha256=hashlib.sha256(Path('data_dev/nci_v2_real/step4/a3_coverage_floored.json').read_bytes()).hexdigest(),
        frozen_surface_sha256=hashlib.sha256(Path(frozen_surface_path).read_bytes()).hexdigest(),
        candidate_db_path=str(candidate_db),candidate_db_sha256=hashlib.sha256(candidate_db.read_bytes()).hexdigest(),
        sidecar_path=str(sidecar),sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        candidate_db_is_byte_identical_to_base=True,
        calibration_groups_only=True)
    Path(out/'part_c_report.json').write_text(json.dumps(jsonable(report),indent=2,sort_keys=True)+'\n')
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for x in ('frame','spec','clean-root','frozen-surface','labels','db','out-dir'):p.add_argument('--'+x,required=True)
    a=p.parse_args();r=run(a.frame,a.spec,a.clean_root,a.frozen_surface,a.labels,a.db,a.out_dir)
    print(json.dumps({k:r[k] for k in ('groups','motion_label_rows','gamma_shared','family_scales','selected_mode','sidecar_sha256')}))
