"""A4t per-row, per-seed and per-parent coverage triage."""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_var] = "1"

import numpy as np

from tools.jgen import inputs
from tools.jgen.a4b.relax_run import to_ref_order
from tools.jgen.bench.pool_export import _read_frames
from tools.runkit import geom
from .metrics import (CLASSES, csm_by_geometry, decision, ring_share, shell_rmsd,
                      shell_radial_lower_bound, stereo_diff, stereo_diff_causes,
                      stereo_graph)

PARENTS = "BABNAV BOCRAN CAQLEP CIFZEZ CIXYUD DAMVAQ DOHTOM DUHHIX EFOXII EZIBUL FEKKAJ FETTAZ FEXWUC FUTSOD GOTWET HOHZAG HOXJUZ HOYBED LIPDEU LUQVOK MIDQEY NUCKUS NUWFAM QAPFAS SILROY SIVBUV USAPOU UYOVUA XACMOE XEVDAE XOJHIN YOVNEC ZAFPOP ZAQNUE ZAZREZ".split()
SEEDS = (11, 12, 13)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def _load_parent(code):
    shard, ds = inputs.frame_shards()[code]
    mol = inputs.load_mol(code, shard, ds)
    if mol is None: raise ValueError(f"input parent {code} not found")
    from jacobik.core.env import build_builder_and_inputs
    builder, lig_inputs, err = build_builder_and_inputs(mol)
    if builder is None: raise ValueError(f"builder reconstruction failed: {err}")
    donors = sorted({int(i)-1 for lig in mol.ligands for i in lig.donor_atoms})
    if not donors: raise ValueError("builder input has no donor atoms")
    return mol, builder, donors


def _relaxed_rows(root, code, seed):
    p=Path(root)/"relax"/f"{code}_{seed}.json"
    return json.loads(p.read_text()).get("rows", {})


def _single_link_count(coords, symbols, metal, donors, threshold=.3):
    n=len(coords)
    if not n:return 0
    parent=list(range(n))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb:parent[rb]=ra
    for i in range(n):
        components=defaultdict(list)
        for j in range(i):components[find(j)].append(j)
        # For each existing component, test rows until one edge is found. This
        # preserves exact connected components while avoiding a full pair matrix.
        for members in components.values():
            for j in members:
                if shell_radial_lower_bound(coords[i],coords[j],symbols,metal,donors)<threshold and shell_rmsd(coords[i],coords[j],symbols,metal,donors)<threshold:
                    union(i,j);break
    return len({find(i) for i in range(n)})


def analyse(a4b_root, a4t_root, out, parents=None, workers=None):
    parents = list(parents if parents is not None else PARENTS)
    base, root, out = Path(a4b_root), Path(a4t_root), Path(out)
    out.mkdir(parents=True,exist_ok=True)
    caps_path=base.parent/"census_caps"/"caltail.jsonl"
    caps={}
    if caps_path.exists():
        for line in caps_path.read_text().splitlines():
            try:
                d=json.loads(line); caps[d["code"]]=d
            except Exception: pass
    row_records=[]; seed_records=[]; parent_records=[]; allclasses=defaultdict(dict)
    allrefs=json.loads((root/"relax"/"references.json").read_text())
    skipped=[c for c in parents if c not in allrefs]
    for c in skipped: print(f"A4t {c}: no relaxed reference (no ok unit in this arm), skipped", flush=True)
    parents=[c for c in parents if c in allrefs]
    tasks = [(code, base, root, allrefs, caps) for code in parents]
    workers = workers if workers is not None else len(os.sched_getaffinity(0))
    if workers < 1: raise ValueError("workers must be positive")
    with ProcessPoolExecutor(max_workers=workers, initializer=_pin_worker) as pool:
        for pr, sr, rr, classes in pool.map(_analyse_parent, tasks):
            parent_records.extend(pr); seed_records.extend(sr); row_records.extend(rr)
            for key, values in classes.items(): allclasses[key].update(values)
    _write(out/"rows.csv",row_records); _write(out/"seeds.csv",seed_records); _write(out/"parents.csv",parent_records)
    counts={k:dict(Counter(v.values())) for k,v in allclasses.items()}
    cross=defaultdict(Counter)
    for r in parent_records: cross[r["classification"]][str(r.get("dedup_capped"))]+=1
    meta=dict(n_parents=len(parents),n_rows=len(row_records),class_counts=counts,class_by_dedup_capped={k:dict(v) for k,v in cross.items()},
        a4b_units_sha256=sha256(base/"units.csv"),pool_manifest_sha256={f"{c}_{s}":sha256(base/"xyz"/c/f"{c}_control_s{s}_pool.json") for c in parents for s in SEEDS},
        relaxed_row_count=sum(1 for r in row_records if r["relax_status"]=="ok"),
        capped_row_count=sum(1 for r in row_records if r["relax_status"]=="capped"),
        failed_row_count=sum(1 for r in row_records if r["relax_status"] not in ("ok","capped")))
    (out/"summary.json").write_text(json.dumps(meta,indent=2))
    return parent_records,seed_records,row_records,meta


def _pin_worker():
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def _analyse_parent(args):
    code, base, root, allrefs, caps = args
    row_records=[]; seed_records=[]; parent_records=[]; allclasses=defaultdict(dict)
    mol,builder,builder_donors=_load_parent(code)
    symbols=[a.symbol for a in mol.atoms]; metal=int(mol.metal_idx)-1
    # Reference geometry/donors are defined from the VM-relaxed reference.
    refout=allrefs[code]
    refX=np.asarray(refout["X"],float)
    ref_donors=geom.donors_of(refX,symbols,metal,pad=.6).tolist()
    donor_mismatch=set(builder_donors)!=set(map(int,ref_donors))
    ref_donors_03=geom.donors_of(refX,symbols,metal,pad=.3).tolist()
    ideal_cns={len(shape) for shape in __import__("jacobik.ik.targets",fromlist=["IDEAL_GEOMETRIES"]).IDEAL_GEOMETRIES.values()}
    geometry_evaluable=len(ref_donors) in ideal_cns
    csm=csm_by_geometry(refX,metal,ref_donors) if geometry_evaluable else {}
    ref_geom=min(csm,key=csm.get) if csm else None
    builder_geom=str(builder.geometry)
    builder_measure=csm.get(builder_geom)
    geom_mismatch=(builder_geom!=ref_geom and builder_measure is not None and builder_measure-csm[ref_geom]>1.0) if geometry_evaluable else None
    perms,_=geom.automorphisms(refX,symbols,metal)
    ref_graph=geom.heavy_graph(refX,symbols,metal)
    try: ref_stereo=stereo_graph(mol,symbols,refX)
    except Exception as exc: raise RuntimeError(f"reference stereo could not be computed for {code}: {exc}") from exc
    parent_rows=[]; seed_classes={}; shell_rel_all=[]; shell_unrel_all=[]; row_counts={};
    row_data_by_seed={}
    for seed in SEEDS:
        mp=base/"xyz"/code/f"{code}_control_s{seed}_pool.json"
        manifest=json.loads(mp.read_text())
        rec=json.loads((base/"records"/f"{code}_{seed}.json").read_text())
        frames=list(_read_frames(mp.parent/manifest["rows"][0]["file"]))
        relax=_relaxed_rows(root,code,seed)
        rawrows=[]; row_counts[str(seed)]=len(frames)
        for i,(header,_,Xgen) in enumerate(frames):
            X=to_ref_order(Xgen,manifest["gen_to_ref_idx"])
            rr=relax.get(str(i),{})
            xr=np.asarray(rr["X"],float) if rr.get("status")=="ok" and rr.get("X") is not None else None
            smrel=shell_rmsd(xr,refX,symbols,metal,ref_donors) if xr is not None else None
            smun=shell_rmsd(X,np.asarray(manifest["ref_X"],float),symbols,metal,ref_donors)
            E=rr.get("E_mace_kj")
            offset=float(E)-float(refout["E_refrel_mace_kj"]) if E is not None else None
            rmsd=geom.sym_rmsd_to_ref(xr[np.asarray([s!="H" for s in symbols])][None,:,:],refX[np.asarray([s!="H" for s in symbols])],perms,graph=ref_graph)[0] if xr is not None else None
            hit=bool(rmsd is not None and rmsd<=.3)
            equivalent=bool(not hit and offset is not None and offset<=5.0)
            rrdata=dict(code=code,seed=seed,row_index=i,score=header.get("score"),E_rel=E,
                        relax_status=rr.get("status","missing"),offset=offset,rmsd_refrel_A=float(rmsd) if rmsd is not None else None,
                        hit=hit,equivalent=equivalent,shell_rmsd_rel_A=smrel,shell_rmsd_unrel_A=smun,
                        shell_match=bool(smrel is not None and smrel<=.3),shell_match_05=bool(smrel is not None and smrel<=.5),
                        unrel_shell_match=bool(smun<=.3),unrel_shell_match_05=bool(smun<=.5),pool_rank=i+1)
            rawrows.append(rrdata); parent_rows.append(rrdata)
            if rrdata["shell_match"]: shell_rel_all.append((float(rmsd),xr,rrdata))
            if rrdata["unrel_shell_match"]: shell_unrel_all.append((i,X,rrdata))
            row_records.append(rrdata)
            if (i+1)%10==0: print(f"A4t {code} seed {seed}: {i+1}/{len(frames)} rows",flush=True)
        row_data_by_seed[seed]=rawrows
        print(f"A4t {code} seed {seed} complete: {len(rawrows)} rows",flush=True)
        seed_classes[seed]=None

    best_shell=min(shell_rel_all,key=lambda x:x[0]) if shell_rel_all else None
    if best_shell:
        try:
            row_stereo=stereo_graph(mol,symbols,best_shell[1])
            ring,row_to_ref=ring_share(best_shell[1],refX,symbols,perms,row_stereo[2],return_atom_map=True)
            stereo_mismatch_old=stereo_diff(ref_stereo,row_stereo,allow_mirror=False,
                                            require_reference_centres=True,
                                            ignore_undetermined=False)
            stereo_mismatch=stereo_diff(ref_stereo,row_stereo,row_to_ref,metal=metal)
            stereo_old_categories,stereo_change_causes,stereo_remaining=stereo_diff_causes(
                ref_stereo,row_stereo,row_to_ref,metal,ref_donors)
        except Exception as exc: raise RuntimeError(f"row stereo/ring could not be computed for {code}: {exc}") from exc
    else:
        stereo_mismatch_old=False;stereo_mismatch=False;stereo_old_categories=[];stereo_change_causes=[];stereo_remaining=[];ring=None
    overall_rows=[r for s in SEEDS for r in row_data_by_seed[s]]
    baseclass=decision(overall_rows,-1,donor_mismatch,geom_mismatch,
                       any(r["shell_match"] for r in overall_rows),stereo_mismatch,ring,code,
                       geometry_evaluable=geometry_evaluable)
    allclasses["base"][code]=baseclass
    oldclass=decision(overall_rows,-1,donor_mismatch,geom_mismatch,
                      any(r["shell_match"] for r in overall_rows),stereo_mismatch_old,ring,code,
                      geometry_evaluable=geometry_evaluable)
    allclasses["amend1"][code]=oldclass
    for s in SEEDS:
        rows=row_data_by_seed[s]
        cl=decision(rows,-1,donor_mismatch,geom_mismatch,any(r["shell_match"] for r in rows),stereo_mismatch,ring,code,
                    geometry_evaluable=geometry_evaluable)
        seed_classes[s]=cl; allclasses[f"seed{s}"][code]=cl
    # Sensitivity decision counts: shell radius 0.5, and unrelaxed shell at 0.3/0.5.
    sensitivity={}
    for label,key in (("relaxed_shell_05","shell_match_05"),("unrelaxed_shell_03","unrel_shell_match"),("unrelaxed_shell_05","unrel_shell_match_05")):
        sensitivity[label]=decision(overall_rows,-1,donor_mismatch,geom_mismatch,any(r[key] for r in overall_rows),stereo_mismatch,ring,code,
                                    geometry_evaluable=geometry_evaluable)
        allclasses[label][code]=sensitivity[label]
    shell_coords=[r["X"] for s in SEEDS for r in []]
    # Keep exact relaxed pool geometries from the per-row output for shell clustering.
    coords=[]
    for s in SEEDS:
        for rr in _relaxed_rows(root,code,s).values():
            if rr.get("status")=="ok" and rr.get("X") is not None: coords.append(np.asarray(rr["X"],float))
    n_shells=_single_link_count(coords,symbols,metal,ref_donors)
    cap=caps.get(code,{})
    offsets=[r["offset"] for r in overall_rows if r["offset"] is not None]
    rmsds=[r["rmsd_refrel_A"] for r in overall_rows if r["rmsd_refrel_A"] is not None]
    if geometry_evaluable:
        geom_evidence=f"ref geom={ref_geom} CSM={csm[ref_geom]:.3f}; builder={builder_geom} CSM={builder_measure if builder_measure is not None else 'n/e at reference CN'}"
    else:
        geom_evidence=f"ref geometry undefined at CN={len(ref_donors)}; donors={','.join(symbols[i] for i in ref_donors)}"
    report_evidence=f"{geom_evidence}; donors builder={builder_donors} ref={ref_donors}; best shell={best_shell[0]:.3f} Å" if best_shell else f"{geom_evidence}; donors builder={builder_donors} ref={ref_donors}; no relaxed shell match"
    f_ranks={str(s):[r["pool_rank"] for r in row_data_by_seed[s] if r["hit"] or r["equivalent"]] for s in SEEDS}
    parent_records.append(dict(code=code,classification=baseclass,classification_amend1=oldclass,evidence=report_evidence,
        geometry_label_builder=builder_geom,geometry_ref=ref_geom,geometry_evaluable=geometry_evaluable,geometry_csm=json.dumps(csm,sort_keys=True),
        builder_geometry_measure=builder_measure,minimum_geometry_measure=csm.get(ref_geom),geometry_mismatch=geom_mismatch,
        donor_set_mismatch=donor_mismatch,builder_donors=json.dumps(builder_donors),reference_donors=json.dumps(ref_donors),
        reference_donor_elements=json.dumps([symbols[i] for i in ref_donors]),reference_donors_pad03=json.dumps(ref_donors_03),
        donor_mismatch_pad03=set(builder_donors)!=set(map(int,ref_donors_03)),
        f_row_ranks_by_seed=json.dumps(f_ranks,sort_keys=True),
        stereo_mismatch_amend1=stereo_mismatch_old,stereo_mismatch=stereo_mismatch,
        stereo_old_differing_elements=json.dumps(stereo_old_categories),
        stereo_amend2_causes=json.dumps(stereo_change_causes),
        stereo_amend2_remaining=json.dumps(stereo_remaining),
        ring_share=ring,ring_conformer=bool(ring is not None and ring>.5),
        per_seed_classes=json.dumps({str(k):v for k,v in seed_classes.items()},sort_keys=True),
        pool_rows_per_seed=json.dumps(row_counts,sort_keys=True),n_distinct_relaxed_shells=n_shells,
        lowest_offset=min(offsets) if offsets else None,best_rmsd_refrel=min(rmsds) if rmsds else None,
        dedup_capped=cap.get("dedup_capped"),n_ez_combinations=cap.get("n_ez_combinations"),
        embed_fallback=cap.get("embed_fallback"),**{f"class_{k}":v for k,v in sensitivity.items()}))
    for s in SEEDS:
        per= row_data_by_seed[s]
        seed_records.append(dict(code=code,seed=s,classification=seed_classes[s],n_pool_rows=len(per),
            n_ok=sum(r["relax_status"]=="ok" for r in per),n_capped=sum(r["relax_status"]=="capped" for r in per),
            n_failed=sum(r["relax_status"] not in ("ok","capped") for r in per),
            n_hits=sum(r["hit"] for r in per),n_equivalent=sum(r["equivalent"] for r in per),
            n_shell_match=sum(r["shell_match"] for r in per),best_rmsd_refrel=min((r["rmsd_refrel_A"] for r in per if r["rmsd_refrel_A"] is not None),default=None),
            lowest_offset=min((r["offset"] for r in per if r["offset"] is not None),default=None)))
    print(f"A4t analysed {code}: {baseclass}; {sum(len(row_data_by_seed[s]) for s in SEEDS)} pool rows",flush=True)


    return parent_records, seed_records, row_records, allclasses


def _write(path, rows):
    with Path(path).open("w",newline="") as f:
        cols=list(dict.fromkeys(k for row in rows for k in row))
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument("--a4b",required=True);p.add_argument("--a4t",required=True);p.add_argument("--out",required=True)
    p.add_argument("--workers", type=int, default=None); a=p.parse_args(); _,_,_,m=analyse(a.a4b,a.a4t,a.out, workers=a.workers); print(json.dumps(m,indent=2))


if __name__=="__main__":main()
