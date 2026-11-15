"""Read-only contact diagnosis on existing N2 pools; never solves or changes defaults."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, pool_path, stats_from_db
from pools import setup, set_cases
from nci_targeted_rescore import CANDIDATE


def run():
    import torch
    from jacobik.core import profile
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.scorer import Scorer
    from research.archive.code.interactions import nci, q6pmf as q
    from jacobik.topology.decompose import decompose_ligand
    from jacobik.topology.graph import get_ligand_neighbors, donor_indices_of
    from tools.jdev.cases import get_case
    from tools.jdev.start import starting_geometry, _crystal_ligand_coords

    assert profile.ACTIVE_NAME == 'u1' and not nci.NCI_ON
    torch.set_num_threads(1)
    report_dir = REPO / 'data/refit/reports'
    previous = json.loads((report_dir / 'n2_targeted_rescore_2026-10-18.json').read_text())
    names = {c: (n, s) for s in ('dev32', 'n48') for c, n in set_cases(s)}
    stats = stats_from_db(CANDIDATE)
    result = {}
    for code in ('FETSUU', 'PEWBOL', 'XEPKAF', 'JEFJUZ', 'BULGAT', 'YIYXEJ', 'CBTYCO'):
        name, split = names[code]
        case = get_case(name)
        with starting_geometry(case, 'full_oracle'):
            mol, _, _ = case.make_builder()._build_shell(False, False)
        for lig in mol.ligands:
            decompose_ligand(mol, lig)
        ge = GeometryExtractor(mol, torch.device('cpu'))
        scorer = Scorer(mol, ge, stats, torch.device('cpu'))
        rt = nci.build_runtime(mol, stats.nci_tables(), 'cpu') or {}
        allc = nci.candidates(mol)
        tables = stats.nci_tables()
        cands = {t: [c for c in allc[t] if any(k in tables for k in c[-1])] for t in nci.TYPES}
        nb = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
        pos = {a.idx: i for i, a in enumerate(mol.atoms)}
        sym = {a.idx: a.symbol for a in mol.atoms}
        bound = set(donor_indices_of(mol))
        lig_id = {a: lig.ligand_id for lig in mol.ligands for a in lig.atom_indices}
        crystal = _crystal_ligand_coords(case)
        reference = np.zeros((len(mol.atoms), 3))
        seen = {}
        for lig, li in zip(sorted(mol.ligands, key=lambda l: l.ligand_id), case.ligand_inputs):
            j = seen.get(li.smiles, 0)
            seen[li.smiles] = j + 1
            reference[[pos[a] for a in lig.atom_indices]] = crystal[li.smiles][j]
        reference[pos[mol.metal_idx]] = case.ref_X[case.ref_metal_row]
        poses, labels = [reference], ['reference']
        for row in previous['pools']:
            if row['case'] != code:
                continue
            with np.load(pool_path('etkdg_u1_os', split, code, row['seed'])) as z:
                for kind, index in [('before', row['base_index']), ('after', row['cand_index'])]:
                    poses.append(z['X'][index])
                    labels.append(f"s{row['seed']}_{kind}")
        X = torch.tensor(np.array(poses), dtype=torch.float32)
        with torch.no_grad():
            scorer.score(X)

        def ring_info(x, ring):
            p = x[[pos[a] for a in ring]]
            center = p.mean(0)
            _, _, vh = np.linalg.svd(p-center)
            normal = vh[-1]
            return center, normal, float(np.sqrt(np.mean(((p-center) @ normal)**2)))

        contacts = {}
        for t, e in rt.items():
            r, a = nci.features(t, X, e['idx'])
            scores = nci._bilinear(e['T'], e['row'], r, a, e['grid']).numpy()
            assert scores.shape[1] == len(cands[t])
            contacts[t] = []
            for j, c in enumerate(cands[t]):
                if np.max(np.abs(scores[:, j])) < .01:
                    continue
                item = dict(candidate=j, atoms=c[:-1], values={})
                for b, (label, x) in enumerate(zip(labels, poses)):
                    v = dict(score=float(scores[b, j]), r=float(r[b, j]), angle=float(a[b, j]),
                             below_fit_range=bool(r[b, j] < nci.SPEC[t]['r'][0]))
                    if t == 'chp':
                        h, ring = c[:2]
                        center, normal, rms = ring_info(x, ring)
                        parent = nb.get(h, [])
                        v.update(h=h, parent=parent, parent_symbols=[sym[p] for p in parent], ring_rms=rms,
                                 h_ligand=lig_id.get(h), ring_ligand=lig_id.get(ring[0]))
                        if parent:
                            u, w = x[pos[parent[0]]]-x[pos[h]], center-x[pos[h]]
                            v['C_H_centroid_angle'] = float(np.degrees(np.arccos(np.clip(np.dot(u,w)/np.linalg.norm(u)/np.linalg.norm(w),-1,1))))
                    elif t == 'pp':
                        ci, ni, rmsi = ring_info(x, c[0])
                        cj, nj, rmsj = ring_info(x, c[1])
                        vec = ci-cj
                        height = abs(float(vec @ ni))
                        v.update(height=height, lateral=float(np.sqrt(max(0, float(vec@vec)-height**2))),
                                 ring_rms=[rmsi,rmsj])
                    item['values'][label] = v
                contacts[t].append(item)

        # Recover each removed Q6 well using precisely the runtime interpolation.
        # Validate its sum against the actual old/new evaluator before interpreting it.
        qr = scorer._q6_pmf
        ai, bi = ge._q6_a, ge._q6_b
        d = (X[:, ai]-X[:, bi]).norm(dim=-1)
        r0 = qr['r0'][qr['tot']][None].expand_as(d)
        w = q._interp(qr['T'],qr['tot'],d,qr['r_lo'],qr['dr'],qr['G'])
        w0 = q._interp(qr['T'],qr['tot'],r0,qr['r_lo'],qr['dr'],qr['G'])
        if qr['hb_any']:
            u, v = X[:,qr['d']]-X[:,qr['h']], X[:,qr['acc']]-X[:,qr['h']]
            cos = (u*v).sum(-1)/(u.norm(dim=-1)*v.norm(dim=-1)).clamp(min=1e-8)
            mix = torch.sigmoid((q.ANGLE_SPLIT_COS-cos)/.08)
            for dst, dist in [(w,d),(w0,r0)]:
                wl = q._interp(qr['T'],qr['lin'],dist,qr['r_lo'],qr['dr'],qr['G'])
                wb = q._interp(qr['T'],qr['bent'],dist,qr['r_lo'],qr['dr'],qr['G'])
                dst.copy_(torch.where(qr['hb'][None],mix*wl+(1-mix)*wb,dst))
        wells = torch.where(d<r0,w0,w).clamp(max=0)*q._taper_t(d,*q.HBOND_TAPER)
        wells *= (qr['hp'] & qr['has'] & ~qr['rep'])[None]
        nci.NCI_ON = False
        old = q.evaluate(qr,X,ai,bi)
        nci.NCI_ON = True
        new = q.evaluate(qr,X,ai,bi)
        nci.NCI_ON = False
        error = float((new-old+wells.sum(-1)).abs().max())
        assert error < 1e-3, (code,error)
        allowed = {frozenset((c[1],c[2])) for c in allc['hb']}
        groups = {k: np.zeros(len(X)) for k in ('NCI_covered','co_bound','other_uncovered')}
        removed = []
        for j, (ia, ib) in enumerate(zip(ai.tolist(),bi.tolist())):
            if wells[:,j].abs().max() < .01:
                continue
            aa, bb = mol.atoms[ia].idx, mol.atoms[ib].idx
            h, acceptor = (aa, bb) if sym[aa]=='H' else (bb,aa)
            parent = nb.get(h,[])
            category = 'NCI_covered' if frozenset((aa,bb)) in allowed else ('co_bound' if parent and parent[0] in bound and acceptor in bound else 'other_uncovered')
            vals = (-wells[:,j]).numpy()
            groups[category] += vals
            removed.append(dict(h=h, d=parent, acceptor=acceptor, acceptor_symbol=sym[acceptor],category=category,
                                removed_reward=dict(zip(labels,map(float,vals)))))
        result[code] = dict(ligands=[l.smiles for l in case.ligand_inputs], labels=labels,contacts=contacts,
                            q6_well_partition={k:dict(zip(labels,map(float,v))) for k,v in groups.items()},
                            q6_wells=removed, q6_partition_error=error)
        # Post-hoc mechanism ablations, never installed as solver objectives.
        # Natural half-space test (C-H-centroid >=90 degrees) and one positive
        # reward per parent-carbon/ring isolate direction and sibling multiplicity.
        pair_fields = 'hp tot lin bent has hb h d acc mu sigma rep rep_A rep_b rep_rm rep_rl'.split()
        mask = torch.tensor([frozenset((mol.atoms[ia].idx,mol.atoms[ib].idx)) in allowed
                             for ia,ib in zip(ai.tolist(),bi.tolist())],dtype=torch.bool)
        subset = dict(qr)
        subset.update({k:qr[k][mask] for k in pair_fields})
        subset.update(hb_any=bool(subset['hb'].any()),rep_any=bool(subset['rep'].any()),n_m13=0)
        ablations=[]
        for row in previous['pools']:
            if row['case']!=code:continue
            with np.load(pool_path('etkdg_u1_os',split,code,row['seed'])) as z:
                P=torch.tensor(z['X'],dtype=torch.float32);Z=float(z['Z']);base=z['score']
            nci.NCI_ON=False
            full_before=q.evaluate(qr,P,ai,bi)
            covered_before=q.evaluate(subset,P,ai[mask],bi[mask])
            nci.NCI_ON=True
            full_delta=full_before-q.evaluate(qr,P,ai,bi)
            covered_delta=covered_before-q.evaluate(subset,P,ai[mask],bi[mask])
            nci.NCI_ON=False
            total=nci.evaluate(rt,P)
            changed=dict(complete=base+(total+full_delta).numpy()/Z,
                         covered_Q6_replacement=base+(total+covered_delta).numpy()/Z)
            if 'chp' in rt:
                e=rt['chp'];r,a=nci.features('chp',P,e['idx'])
                scores=nci._bilinear(e['T'],e['row'],r,a,e['grid'])
                center,_=nci._ring_frame(P,e['idx']['rings'])
                H=P[:,e['idx']['h']];target=center[:,e['idx']['ring']]
                parent=[nb.get(c[0],[]) for c in cands['chp']]
                valid=torch.tensor([len(p)==1 and sym[p[0]]=='C' for p in parent])
                parent_pos=torch.tensor([pos[p[0]] if p else 0 for p in parent])
                cosine=((P[:,parent_pos]-H)*(target-H)).sum(-1)
                directed=torch.where((cosine<=0)&valid[None],scores,scores.clamp(max=0))
                groups={}
                for j,c in enumerate(cands['chp']):
                    groups.setdefault((tuple(parent[j]),tuple(c[1])),[]).append(j)
                grouped=scores.clamp(max=0).sum(-1)
                for ix in groups.values():grouped+=scores[:,ix].clamp(min=0).max(-1).values
                changed.update(CH_direction_halfspace=base+(total-scores.sum(-1)+directed.sum(-1)+full_delta).numpy()/Z,
                               CH_parent_ring_max=base+(total-scores.sum(-1)+grouped+full_delta).numpy()/Z,
                               no_CH=base+(total-scores.sum(-1)+full_delta).numpy()/Z)
            from common import pool_energy
            energy=pool_energy(pool_path('etkdg_u1_os',split,code,row['seed']),'mace64')
            picks={k:int(v.argmax()) for k,v in changed.items()}
            assert picks['complete']==row['cand_index']
            ablations.append(dict(seed=row['seed'],variants={k:dict(index=i,energy_change=float(energy[i]-energy[row['base_index']])) for k,i in picks.items()}))
        result[code]['posthoc_ablations']=ablations
        print(code, json.dumps(result[code]['q6_well_partition']),flush=True)
    out = report_dir / 'n2_contact_diagnosis_2026-10-18.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(out,flush=True)


if __name__ == '__main__':
    setup()
    run()
