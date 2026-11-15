"""Localize energy changes and forces in selected stored structures, without optimization.

Uses exact engine topology, float64 MACE, atom-matched model-local energy changes,
explicit global energy residual, ligand forces/torques, and finite-displacement
energy/score scans along named motions. These are diagnostic directions, not
mass-weighted vibrational normal modes. Outputs stay in reports, not inspect/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, pool_path, stats_from_db
from pools import setup, set_cases
from nci_targeted_rescore import CANDIDATE

EV = 96.4853321233


def orthogonal_basis(vectors, ncoords):
    if not len(vectors):
        return np.zeros((ncoords, 0))
    a = np.asarray(vectors).reshape(len(vectors), ncoords).T
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    return u[:, s > max(s[0]*1e-9, 1e-10)]


def project_forces(forces, modes, metal_row):
    """Disjoint Cartesian force-squared partitions; individual modes overlap."""
    f = np.asarray(forces).ravel()
    rigid = orthogonal_basis([m['v'] for m in modes if m['kind'] in ('translation', 'rotation')], len(f))
    tors = [np.asarray(m['v']).ravel() for m in modes if m['kind']=='torsion']
    tors = [v-rigid@(rigid.T@v) for v in tors]
    tb = orthogonal_basis(tors, len(f))
    fr = rigid@(rigid.T@f)
    ft = tb@(tb.T@f)
    fm = np.zeros_like(forces)
    fm[metal_row] = forces[metal_row]
    fm = fm.ravel()
    residual = f-fr-ft-fm
    parts = dict(ligand_rigid=float(fr@fr), ligand_torsion=float(ft@ft),
                 ligand_internal_residual=float(residual@residual), metal=float(fm@fm))
    total = float(f@f)
    assert abs(sum(parts.values())-total) < 1e-7*max(1,total)
    return dict(force_squared=parts, fraction={k:v/max(total,1e-30) for k,v in parts.items()}, total=total)


def moved(X, mode, step):
    """Exact rotations preserve internal bonds; step is tangent norm in Angstrom."""
    if 'axis' not in mode:
        return X+step*np.asarray(mode['v'])
    y = X.copy()
    rows = mode['rows']
    p = X[rows]-np.asarray(mode['origin'])
    axis = np.asarray(mode['axis'])
    angle = step/mode['rotation_scale']
    y[rows] = (p*np.cos(angle)+np.cross(axis,p)*np.sin(angle)
               +np.outer(p@axis,axis)*(1-np.cos(angle))+mode['origin'])
    return y


def motion_modes(mol, X):
    """Exact topology ligand translations, rotations and non-ring single-bond torsions."""
    import networkx as nx
    from jacobik.topology.graph import get_ligand_neighbors
    pos = {a.idx:i for i,a in enumerate(mol.atoms)}
    nb = get_ligand_neighbors(mol.atoms,mol.bonds,mol.metal_idx)
    modes = []

    def add(kind, label, rows, axis, origin=None):
        v = np.zeros_like(X)
        if origin is None:
            v[rows] = axis
        else:
            v[rows] = np.cross(axis,X[rows]-origin)
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            return
        mode = dict(kind=kind,label=label,rows=rows,v=v/norm)
        if origin is not None:
            mode.update(axis=np.asarray(axis),origin=origin,rotation_scale=float(norm))
        modes.append(mode)

    for lig in mol.ligands:
        rows = [pos[a] for a in lig.atom_indices]
        center = X[rows].mean(0)
        for j,axis in enumerate(np.eye(3)):
            add('translation',f'L{lig.ligand_id}_translate_{"xyz"[j]}',rows,axis)
            add('rotation',f'L{lig.ligand_id}_rotate_{"xyz"[j]}',rows,axis,center)
        metal = X[pos[mol.metal_idx]]
        radial = center-metal
        if np.linalg.norm(radial)>1e-9:
            add('coordination',f'L{lig.ligand_id}_metal_radial',rows,radial/np.linalg.norm(radial))
        graph = nx.Graph()
        graph.add_nodes_from(lig.atom_indices)
        graph.add_edges_from((a,b) for a in lig.atom_indices for b in nb.get(a,()) if b in lig.atom_indices)
        for bond in lig.rotatable_bonds:
            g = graph.copy()
            g.remove_edge(bond.i,bond.j)
            if nx.has_path(g,bond.i,bond.j):
                continue
            sides = [nx.node_connected_component(g,a) for a in (bond.i,bond.j)]
            side = min(sides,key=len)
            axis = X[pos[bond.j]]-X[pos[bond.i]]
            axis /= np.linalg.norm(axis)
            add('torsion',f'L{lig.ligand_id}_bond_{bond.i}_{bond.j}',[pos[a] for a in sorted(side)],axis,X[pos[bond.i]])
    return modes


def energy_accounting(properties):
    """Never spread PolarMACE's nonlocal energy over atoms arbitrarily."""
    local = np.asarray(properties['energies'],float)*EV
    total = float(properties['energy'])*EV
    return local, dict(total=total,model_local_sum=float(local.sum()),
                       nonlocal_residual=total-float(local.sum()),
                       **{k:float(properties[k])*EV for k in ('interaction_energy','electrostatic_energy','electron_energy') if k in properties})


def score_values(scorer, runtime, coordinates):
    """Only assignment-independent Q6 and NCI; no reconstructed Q1/Q2 claim."""
    import torch
    from research.archive.code.interactions import nci, q6pmf
    X = torch.tensor(coordinates,dtype=torch.float32)
    old_flag = nci.NCI_ON
    try:
        with torch.no_grad():
            nci.NCI_ON = False
            base = -q6pmf.evaluate(scorer._q6_pmf,X,scorer._ge._q6_a,scorer._ge._q6_b).numpy()
            nci.NCI_ON = True
            core = -q6pmf.evaluate(scorer._q6_pmf,X,scorer._ge._q6_a,scorer._ge._q6_b).numpy()
            out = dict(Q6_before=base,Q6_after=core)
            for t in nci.TYPES:
                out[f'NCI_{t}'] = nci.evaluate({t:runtime[t]},X).numpy() if t in runtime else np.zeros(len(X))
            return out
    finally:
        nci.NCI_ON = old_flag


def contact_scores(kind, entry, X, model='legacy'):
    """Per-contact display scores with an explicit historical/current evaluator."""
    from research.archive.code.interactions import nci
    if model == 'legacy':
        r, a = nci.features(kind, X, entry['idx'])
        return nci._bilinear(entry['T'],entry['row'],r,a,entry['grid'])
    if model != 'calibrated_hb' or kind != 'hb':
        raise ValueError('per-contact C2 diagnostics currently support hb only')
    r, z = nci.features_cos_torch(kind, X, entry['idx'])
    return float(entry.get('scale',1.0))*nci.evaluate_c2_from_rz_torch(
        entry['C'],entry['M'],entry['row'],r,z,entry['grid'],entry['win'],entry['s_clip'])


def add_contact_modes(modes,mol,X,runtime,limit,contact_model='legacy'):
    """Ligand opening motions for strongest contacts, selected without energy information."""
    import torch
    from research.archive.code.interactions import nci
    pos = {a.idx:i for i,a in enumerate(mol.atoms)}
    ligs = {pos[a]:[pos[u] for u in lig.atom_indices] for lig in mol.ligands for a in lig.atom_indices}
    contacts=[]
    for t,rt in runtime.items():
        xt = torch.tensor(X[None],dtype=torch.float32)
        s = contact_scores(t,rt,xt,contact_model)[0].detach().numpy()
        ix=rt['idx']
        for j in np.argsort(-np.abs(s))[:limit]:
            if abs(s[j])<.05:continue
            if t=='hb':
                moving,target=int(ix['h'][j]),X[int(ix['a'][j])]
                other=int(ix['a'][j])
            elif t=='chp':
                moving=int(ix['h'][j]); ring=ix['rings'][int(ix['ring'][j])].numpy()
                ring=ring[ring>=0];target=X[ring].mean(0);other=int(ring[0])
            else:
                ri=ix['rings'][int(ix['ri'][j])].numpy();ri=ri[ri>=0]
                rj=ix['rings'][int(ix['rj'][j])].numpy();rj=rj[rj>=0]
                moving=int(ri[0]);other=int(rj[0]);target=X[rj].mean(0)
            rows=ligs[moving]
            if other in rows:continue  # no artificial rigid opening of an intraligand contact
            source=X[moving] if t!='pp' else X[ri].mean(0)
            axis=source-target
            axis/=np.linalg.norm(axis)
            v=np.zeros_like(X);v[rows]=axis/np.sqrt(len(rows))
            contacts.append((abs(float(s[j])),dict(kind='contact',label=f'{t}_contact_{j}_ligand_open',rows=rows,v=v)))
    modes.extend(m for _,m in sorted(contacts,key=lambda p:-p[0])[:limit])


def analyze_pose(ev,mol,scorer,runtime,X,symbols,charge,mult,step,max_contacts,
                 score_fn=None,contact_model='legacy'):
    properties=ev.properties(symbols,X,charge,mult)
    force=np.asarray(properties['forces'],float)*EV
    if not np.isfinite(force).all():raise ValueError('nonfinite MACE force')
    local,account=energy_accounting(properties)
    modes=motion_modes(mol,X)
    partition=project_forces(force,modes,next(i for i,a in enumerate(mol.atoms) if a.idx==mol.metal_idx))
    add_contact_modes(modes,mol,X,runtime,max_contacts,contact_model)
    V=np.stack([m['v'] for m in modes])
    # Two step sizes establish that derivatives reflect local response, not one displacement choice.
    displaced=[X]+[moved(X,m,h) for h in (step,-step,step/2,-step/2) for m in modes]
    energies=ev.energies([(symbols,y,charge,mult) for y in displaced])*EV
    scores=(score_fn or (lambda x:score_values(scorer,runtime,x)))(np.asarray(displaced))
    n=len(modes)
    ep,em,ehp,ehm=[energies[1+j*n:1+(j+1)*n] for j in range(4)]
    g=(ep-em)/(2*step);gh=(ehp-ehm)/step
    analytic=-np.einsum('nk,dnk->d',force,V)
    k=(ep+em-2*energies[0])/step**2
    kh=(ehp+ehm-2*energies[0])/(step/2)**2
    fd_error=np.abs(gh-analytic)
    # Full properties and fast batch paths must be the same energy function.
    energy_error=abs(float(energies[0])-account['total'])
    assert energy_error<.01, energy_error
    rows=[]
    for j,m in enumerate(modes):
        row=dict(label=m['label'],kind=m['kind'],dE_dh=float(analytic[j]),fd_dE_dh=float(gh[j]),
                 fd_force_error=float(fd_error[j]),curvature=float(kh[j]),
                 gradient_step_change=float(abs(gh[j]-g[j])),curvature_step_change=float(abs(kh[j]-k[j])))
        for t,s in scores.items():
            row[f'dS_{t}_dh']=float((s[1+2*n+j]-s[1+3*n+j])/step)
            row[f'negative_curvature_S_{t}']=float(-(s[1+2*n+j]+s[1+3*n+j]-2*s[0])/(step/2)**2)
        rows.append(row)
    pos={a.idx:i for i,a in enumerate(mol.atoms)}
    ligands=[]
    for lig in mol.ligands:
        ix=[pos[a] for a in lig.atom_indices]
        ligands.append(dict(ligand=lig.ligand_id,force=force[ix].sum(0).tolist(),
                            torque=np.cross(X[ix]-X[ix].mean(0),force[ix]).sum(0).tolist(),
                            force_norm=float(np.linalg.norm(force[ix])),model_local_energy=float(local[ix].sum())))
    summary=dict(energy=account,force_rms=float(np.sqrt(np.mean(np.sum(force**2,axis=1)))),
                 force_max=float(np.linalg.norm(force,axis=1).max()),net_force=force.sum(0).tolist(),
                 net_torque=np.cross(X-X.mean(0),force).sum(0).tolist(),force_partition=partition,ligands=ligands,
                 checks=dict(full_vs_fast_energy_error=energy_error,force_fd_max_error=float(fd_error.max()),
                             force_fd_median_error=float(np.median(fd_error)),
                             force_fd_relative_max=float(np.max(fd_error/np.maximum(1,np.abs(analytic))))),
                 modes=rows,score_at_pose={t:float(v[0]) for t,v in scores.items()})
    arrays=dict(X=X,forces=force,model_local_energy=local,mode_vectors=V,
                mode_labels=np.array([m['label'] for m in modes]),scan_energy=energies)
    return summary,arrays


def write_plot(folder,labels,summaries,arrays,symbols):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    poses=[k for k in labels if k!='reference']
    fig,axes=plt.subplots(len(poses),2,figsize=(13,3*len(poses)),squeeze=False,layout='constrained')
    for axes_row,label in zip(axes,poses):
        delta=arrays[label]['model_local_energy']-arrays['reference']['model_local_energy']
        ix=np.argsort(-np.abs(delta))[:12]
        ax=axes_row[0]
        ax.bar(np.arange(len(ix)),delta[ix],color=np.where(delta[ix]>0,'#cc553d','#3280a0'))
        ax.set_xticks(np.arange(len(ix)),[f'{symbols[i]}{i+1}' for i in ix],rotation=45)
        ax.set_ylabel('Model-local ΔE vs reference (kJ/mol)')
        e=summaries[label]['energy'];ref=summaries['reference']['energy']
        ax.set_title(f"{label}: total ΔE {e['total']-ref['total']:.1f}; global residual Δ {e['nonlocal_residual']-ref['nonlocal_residual']:.1f}")
        modes=sorted(summaries[label]['modes'],key=lambda r:-abs(r['dE_dh']))[:8]
        ax=axes_row[1]
        ax.barh([m['label'] for m in modes][::-1],[m['dE_dh'] for m in modes][::-1],color='#3280a0')
        ax.set_xlabel('Energy derivative along named motion (kJ/mol/Å tangent norm)')
        ax.set_title('Largest projected forces; +motion follows label')
    fig.savefig(folder/'localization.png',dpi=150)
    plt.close(fig)


def run(a):
    import pandas as pd
    import torch
    from jacobik.core import profile
    from jacobik.core.energy import MACEPolarEvaluator
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.scorer import Scorer
    from research.archive.code.interactions import nci
    from jacobik.topology.decompose import decompose_ligand
    from tools.jdev.cases import get_case
    from tools.jdev.start import starting_geometry,_crystal_ligand_coords
    from research.archive.code.refit.nci_refit import digest

    assert profile.ACTIVE_NAME=='u1' and not nci.NCI_ON
    torch.set_num_threads(a.threads)
    start=time.perf_counter()
    selection=json.loads(a.selection.read_text())
    names={c:(n,s) for s in ('dev32','n48') for c,n in set_cases(s)}
    stats=stats_from_db(a.db)
    model=REPO/'tools/jdev/MACE-POLAR-1-S.model'
    ev=MACEPolarEvaluator(model='polar-1-s',device=a.device,dtype='float64',model_path=model,threads=a.threads)
    config=dict(profile=profile.ACTIVE_NAME,db_sha256=digest(a.db),model_sha256=digest(model),
                selection_sha256=digest(a.selection),driver_sha256=digest(Path(__file__)),step=a.step,
                max_contact_modes=a.max_contact_modes,units=dict(energy='kJ/mol',forces='kJ/mol/Angstrom',
                curvature='kJ/mol/Angstrom^2',score='raw nats'),
                semantics='Model-local atomic energies omit global PolarMACE terms; residual retained explicitly. Motion directions are not Hessian normal modes. Local Q6/NCI derivatives only; stored other score terms retained in pair reports. No optimization, fits or default changes.')
    a.out.mkdir(parents=True,exist_ok=True)
    done=[]
    for code in a.cases:
        name,split=names[code];case=get_case(name)
        with starting_geometry(case,'full_oracle'):
            mol,_,_=case.make_builder()._build_shell(False,False)
        for lig in mol.ligands:decompose_ligand(mol,lig)
        ge=GeometryExtractor(mol,torch.device('cpu'));scorer=Scorer(mol,ge,stats,torch.device('cpu'))
        runtime=nci.build_runtime(mol,stats.nci_tables(),'cpu') or {}
        pos={at.idx:i for i,at in enumerate(mol.atoms)}
        symbols=[at.symbol for at in mol.atoms]
        crystal=_crystal_ligand_coords(case);reference=np.zeros((len(symbols),3));seen={}
        for lig,li in zip(sorted(mol.ligands,key=lambda l:l.ligand_id),case.ligand_inputs):
            j=seen.get(li.smiles,0);seen[li.smiles]=j+1
            reference[[pos[u] for u in lig.atom_indices]]=crystal[li.smiles][j]
        reference[pos[mol.metal_idx]]=case.ref_X[case.ref_metal_row]
        poses={'reference':reference};pairs=[];pool_hashes={}
        for row in selection['pools']:
            if row['case']!=code or row['seed'] not in a.seeds:continue
            path=pool_path(a.arm,split,code,row['seed']);pool_hashes[str(path)]=digest(path)
            with np.load(path) as z:
                assert symbols==z['symbols'].tolist()
                for which,index in [('before',row['base_index']),('after',row['cand_index'])]:
                    poses[f"s{row['seed']}_{which}"]=z['X'][index]
                pair=dict(seed=row['seed'],expected_energy_change=row['energy_change'],
                          stored_score_terms={key:{which:float(z[key][row['base_index'] if which=='before' else row['cand_index']]) for which in ('before','after')} for key in z.files if key.startswith('terms_')},
                          base_index=row['base_index'],candidate_index=row['cand_index'])
                charge,mult=int(z['charge']),int(z['mult'])
                pairs.append(pair)
        if not pairs:raise ValueError(f'no selected pools for {code}')
        folder=a.out/code;folder.mkdir(exist_ok=True)
        summaries,arrays={},{}
        for label,X in poses.items():
            stem=folder/label
            provenance=dict(config=config,geometry_sha256=hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
                            pools=pool_hashes,charge=charge,multiplicity=mult)
            cache=stem.with_suffix('.json')
            if cache.exists():
                loaded=json.loads(cache.read_text())
                if loaded['provenance']!=provenance:raise ValueError(f'cache provenance changed: {cache}; choose a new --out')
                summaries[label]=loaded['summary']
                with np.load(stem.with_suffix('.npz')) as z:arrays[label]={k:z[k] for k in z.files}
                print(f'{code}/{label}: cached',flush=True);continue
            summary,arr=analyze_pose(ev,mol,scorer,runtime,X,symbols,charge,mult,a.step,a.max_contact_modes)
            np.savez_compressed(stem.with_suffix('.npz'),**arr)
            pd.DataFrame(summary['modes']).to_csv(stem.with_name(stem.name+'_modes.csv'),index=False)
            pd.DataFrame(dict(atom=np.arange(1,len(symbols)+1),symbol=symbols,
                local_energy=arr['model_local_energy'],Fx=arr['forces'][:,0],Fy=arr['forces'][:,1],Fz=arr['forces'][:,2],
                force_norm=np.linalg.norm(arr['forces'],axis=1))).to_csv(stem.with_name(stem.name+'_atoms.csv'),index=False)
            cache.write_text(json.dumps(dict(provenance=provenance,summary=summary),indent=2)+'\n')
            summaries[label],arrays[label]=summary,arr
            print(f"{code}/{label}: force RMS {summary['force_rms']:.2f}, max {summary['force_max']:.2f}; FD max error {summary['checks']['force_fd_max_error']:.4g}",flush=True)
        for pair in pairs:
            before,after=[f"s{pair['seed']}_{kind}" for kind in ('before','after')]
            delta=arrays[after]['model_local_energy']-arrays[before]['model_local_energy']
            pair['total_energy_change']=summaries[after]['energy']['total']-summaries[before]['energy']['total']
            pair['local_energy_change']=float(delta.sum())
            pair['nonlocal_energy_change']=summaries[after]['energy']['nonlocal_residual']-summaries[before]['energy']['nonlocal_residual']
            assert abs(pair['total_energy_change']-pair['expected_energy_change'])<.02, pair
            pd.DataFrame(dict(atom=np.arange(1,len(symbols)+1),symbol=symbols,model_local_energy_change=delta)).to_csv(folder/f"s{pair['seed']}_atom_changes.csv",index=False)
            pair['top_local_increases']=[dict(atom=int(i+1),symbol=symbols[i],change=float(delta[i])) for i in np.argsort(-delta)[:8]]
        (folder/'comparison.json').write_text(json.dumps(dict(config=config,pairs=pairs),indent=2)+'\n')
        write_plot(folder,list(poses),summaries,arrays,symbols)
        done.append(code)
        (a.out/'manifest.json').write_text(json.dumps(dict(config=config,completed=done,elapsed_s=time.perf_counter()-start,backend=ev.describe()),indent=2)+'\n')
    print(f'Completed {done} in {time.perf_counter()-start:.1f}s: {a.out}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[11,12,13])
    p.add_argument('--selection',type=Path,default=REPO/'data/refit/reports/n2_targeted_rescore_2026-10-18.json')
    p.add_argument('--arm',default='etkdg_u1_os')
    p.add_argument('--db',type=Path,default=CANDIDATE)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--step',type=float,default=.04)
    p.add_argument('--max-contact-modes',type=int,default=6)
    a=p.parse_args()
    if a.step<=0 or a.threads<1:p.error('step and threads must be positive')
    setup();run(a)


if __name__=='__main__':main()
