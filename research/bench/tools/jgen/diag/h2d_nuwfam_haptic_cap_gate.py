"""Paired CPU H-2d NUWFAM haptic-cap geometry/MACE gate."""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
sys.path[:0] = ["src", ".", "research/bench"]

from tools.jgen import inputs
from jacobik.core.env import build_builder_and_inputs
from jacobik.core.env import MACE_MODEL_PATH
from tools.jdev.budget.common import stats_from_db
from tools.runkit import xtb
from jacobik.builder.atom_map import build_atom_mapping
from jacobik.builder import engine
from jacobik.core import profile
from jacobik.core.energy import MACEPolarEvaluator

OUT_DEFAULT = Path("research/data/jgen/campaign/H/h2_census/h2d_nuwfam_haptic_cap.json")
# Source/reference atom indices, validated from the D-1 shell/source atom map.
ALLYL_C3 = (1, 2, 3)
F_ATOM, C8_ATOM = 38, 37
BONDS = {"C-F": (37, 38), "C-C": (2, 3), "C-B": (4, 16)}

class FallbackWatch(logging.Handler):
    def __init__(self): super().__init__(level=logging.WARNING); self.messages=[]
    def emit(self, record):
        msg=record.getMessage()
        if "GFN2 embed refinement failed" in msg: self.messages.append(msg)

def xyz(mol): return np.asarray([a.position for a in mol.atoms],dtype=float)

def plane_metrics(X, plane_atoms=ALLYL_C3, f_atom=F_ATOM):
    p=X[list(plane_atoms)]; origin=p[0]
    normal=np.cross(p[1]-p[0],p[2]-p[0]); norm=np.linalg.norm(normal)
    if norm < 1e-10: raise RuntimeError("degenerate allyl C3 plane")
    normal/=norm; vec=X[f_atom]-origin
    h=float(np.dot(vec,normal)); inplane=vec-h*normal
    elevation=float(np.degrees(np.arctan2(abs(h),np.linalg.norm(inplane))))
    cf=X[f_atom]-X[BONDS["C-F"][0]]
    cf_angle=float(np.degrees(np.arcsin(np.clip(abs(np.dot(cf,normal))/np.linalg.norm(cf),0,1))))
    return {"signed_F_height_A":h,"F_out_of_plane_distance_A":abs(h),
            "F_position_elevation_from_plane_deg":elevation,"C-F_bond_angle_to_plane_deg":cf_angle}

def angle_sum(X, center, adj):
    ns=adj[center]
    if len(ns)!=3: return None
    v=[X[j]-X[center] for j in ns]; v=[q/np.linalg.norm(q) for q in v]
    return float(sum(np.degrees(np.arccos(np.clip(np.dot(v[i],v[j]),-1,1)))
                     for i in range(3) for j in range(i+1,3)))

def source_map(ref, mol):
    Xr,Xg=xyz(ref),xyz(mol)
    amap=build_atom_mapping(ref,gen_mol=mol,X_ref=Xr,X_gen=Xg)
    src_to_gen=np.asarray(amap.gen_to_ref_idx,dtype=int)
    if not amap.verify_symbols([a.symbol for a in ref.atoms],[a.symbol for a in mol.atoms]):
        raise RuntimeError("generated shell symbols do not map to reference symbols")
    if len(src_to_gen)!=len(ref.atoms) or len(Xg)!=len(mol.atoms): raise RuntimeError("atom mapping count mismatch")
    return Xr,Xg,src_to_gen,amap

def geometry_row(Xsrc, adj):
    lengths={k:float(np.linalg.norm(Xsrc[a]-Xsrc[b])) for k,(a,b) in BONDS.items()}
    return {"bond_lengths_A":lengths,"F_vs_allyl_C3_plane":plane_metrics(Xsrc),
            "sp2_angle_sums_deg":{"allyl_C1_source1":angle_sum(Xsrc,1,adj),
                "allyl_C2_source2":angle_sum(Xsrc,2,adj),"allyl_C3_source3":angle_sum(Xsrc,3,adj),
                "C8_source37":angle_sum(Xsrc,C8_ATOM,adj)}}

def internal_adj(ref):
    adj=[[] for _ in ref.atoms]
    metals={"Zr","Hf","Nb","Ta","Mo","W","Re","Ru","Rh","Ir","Fe","Co","Ni","Pd","Pt","Cu","Ag","Au"}
    for b in ref.bonds:
        i,j=int(b.i)-1,int(b.j)-1
        if min(i,j)<0 or max(i,j)>=len(ref.atoms):continue
        if ref.atoms[i].symbol in metals or ref.atoms[j].symbol in metals: continue
        adj[i].append(j);adj[j].append(i)
    return adj

def align_ref_to_target(ref_pts,target_pts):
    cr=ref_pts.mean(axis=0);ct=target_pts.mean(axis=0)
    u,_,vt=np.linalg.svd((ref_pts-cr).T@(target_pts-ct))
    fix=np.eye(3);fix[-1,-1]=np.linalg.det(u@vt)
    R=u@fix@vt
    return R,ct-cr@R

def reset_bond_lengths(Xsrc,pairs,targets):
    X=Xsrc.copy()
    for pair,target in zip(pairs,targets):
        i,j=pair;v=X[j]-X[i];d=np.linalg.norm(v)
        if d<1e-10:continue
        unit=v/d;shift=(float(target)-d)/2
        X[i]-=shift*unit;X[j]+=shift*unit
    return X

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",default=str(OUT_DEFAULT));a=ap.parse_args()
    if not __import__("shutil").which("xtb"):raise SystemExit("xtb is required")
    index=inputs.frame_shards()
    sh,ds=index["NUWFAM"];ref=inputs.load_mol("NUWFAM",sh,ds);builder,_,err=build_builder_and_inputs(ref)
    if builder is None:raise RuntimeError(err)
    Xr=xyz(ref);adj=internal_adj(ref)
    expected={"C-F":("C","F"),"C-C":("C","C"),"C-B":("C","B")}
    for key,(i,j) in BONDS.items():
        if (ref.atoms[i].symbol,ref.atoms[j].symbol)!=expected[key]:raise RuntimeError(f"atom-map validation failed for {key}: {i,j}")
    for i in ALLYL_C3:
        if ref.atoms[i].symbol!="C":raise RuntimeError(f"allyl plane atom {i} is not carbon")
    if ref.atoms[F_ATOM].symbol!="F" or ref.atoms[C8_ATOM].symbol!="C":raise RuntimeError("F/C8 atom-map validation failed")
    variants=builder._ez_combinations()
    watch=FallbackWatch();logging.getLogger("jacobik.builder.engine").addHandler(watch)
    embedded=[]
    for vi,fixes in enumerate(variants):
        engine._GFN2_CACHE.clear();engine._GFN2_BOND_CHECK_CACHE.clear();watch.messages.clear()
        for lig,fix in zip(builder.ligands,fixes):lig.ez_fix=fix
        try:
            with xtb.FallbackWatch(raise_on_exit=True):
                shell,_,_=builder._build_shell(False,False,embed_seed=42,chelate_embed=True,embed="gfn2")
            Xr0,Xg,src_to_gen,_=source_map(ref,shell);Xsrc=Xg[src_to_gen]
            embedded.append({"variant":vi,"fixes":fixes,"geometry":geometry_row(Xsrc,adj)})
        finally:
            for lig in builder.ligands:lig.ez_fix=None
    if watch.messages:raise RuntimeError(f"GFN2 fallback during variant embed: {watch.messages}")
    # Full solve under the active flag. The retained conformer rows carry embed_idx (E/Z combination index).
    engine._GFN2_CACHE.clear();engine._GFN2_BOND_CHECK_CACHE.clear();watch.messages.clear()
    result=builder.solve(seed=11,stats=stats_from_db(inputs.DB_PATH),angle_bends=True,device="cpu",
                         n_ik_steps=250,n_conformers=16,embed_seed=42)
    if watch.messages:raise RuntimeError(f"GFN2 fallback during NUWFAM solve: {watch.messages}")
    if not result.conformers:raise RuntimeError(f"NUWFAM returned no conformers: {result.failure}")
    calc=MACEPolarEvaluator(model="polar-1-s",device="cpu",dtype="float64",model_path=str(MACE_MODEL_PATH),threads=1)
    mapped=[]; batch=[]
    ref_shell=None
    for rank,conf in enumerate(result.conformers,1):
        mol=conf.complex;Xr0,Xg,src_to_gen,amap=source_map(ref,mol)
        if "Li" in [z.symbol for z in mol.atoms]:raise RuntimeError("temporary Li cap leaked")
        if int(ref.charge)!=int(mol.charge) or int(ref.spin_state)!=int(mol.spin_state):raise RuntimeError("charge/spin mismatch")
        Xsrc=Xg[src_to_gen]
        embed_idx=getattr(conf,"embed_idx",None)
        row={"rank":rank,"variant":int(embed_idx) if embed_idx is not None else 0,"score":float(conf.score),
             "geometry":geometry_row(Xsrc,adj),"atom_count":len(Xg),"hydrogen_count":sum(a.symbol=="H" for a in mol.atoms),
             "charge":int(mol.charge),"spin":int(mol.spin_state),"li_count":sum(a.symbol=="Li" for a in mol.atoms)}
        mapped.append((row,mol,Xg,src_to_gen,amap));batch.append(([z.symbol for z in mol.atoms],Xg,int(mol.charge),int(mol.spin_state)))
    ref_shell=None
    # Crystal in generated-shell atom order from the top row mapping (all mappings have same symbols/order).
    top=mapped[0];ref_shell=Xr[np.asarray(top[4].ref_to_gen_idx,dtype=int)]
    batch.append(([a.symbol for a in top[1].atoms],ref_shell,int(ref.charge),int(ref.spin_state)))
    energies=calc.energies(batch,batch_size=1)
    for k,(row,*_) in enumerate(mapped):row["mace_eV"]=float(energies[k]);row["mace_delta_vs_crystal_kJ_mol"]=float((energies[k]-energies[-1])*96.4853321233)
    top1_by_variant={}
    for vi in range(len(variants)):
        rows=[x[0] for x in mapped if x[0]["variant"]==vi]
        top1_by_variant[str(vi)]=max(rows,key=lambda x:x["score"]) if rows else None
    crystal_geom=geometry_row(Xr,adj)
    # F-only local crystal placement and independent bond-only reset on the best retained variant-1 row.
    variant1=[x for x in mapped if x[0]["variant"]==1]
    decomposition=None
    if variant1:
        base=sorted(variant1,key=lambda x:x[0]["score"],reverse=True)[0]
        row,mol,Xg,src_to_gen,amap=base;Xsrc=Xg[src_to_gen].copy();pairs=list(BONDS.values())
        R,t=align_ref_to_target(Xr[list(ALLYL_C3)],Xsrc[list(ALLYL_C3)])
        # Keep generated C8 and the current C-F length fixed; restore the crystal
        # C8-F internal direction in the allyl-C3 frame.
        f_vec=(Xr[F_ATOM]-Xr[C8_ATOM])@R
        f_target=Xsrc[C8_ATOM]+np.linalg.norm(Xsrc[F_ATOM]-Xsrc[C8_ATOM])*f_vec/np.linalg.norm(f_vec)
        Xf=Xsrc.copy();Xf[F_ATOM]=f_target
        # Reset just the three selected internal bond lengths to the crystal values.
        crystal_targets=[float(np.linalg.norm(Xr[i]-Xr[j])) for i,j in pairs]
        Xb=reset_bond_lengths(Xsrc,pairs,crystal_targets)
        shell_syms=[a.symbol for a in mol.atoms]
        def to_shell(Xs):
            Xout=Xg.copy();Xout[src_to_gen]=Xs;return Xout
        coords=[Xg,to_shell(Xf),to_shell(Xb)]
        eb=calc.energies([(shell_syms,X,int(mol.charge),int(mol.spin_state)) for X in coords],batch_size=1)
        decomposition={"source_row":{"rank":row["rank"],"variant":row["variant"],"score":row["score"],"mace_eV":row["mace_eV"]},
            "crystal_internal_bond_targets_A":dict(zip(BONDS.keys(),crystal_targets)),
            "f_only_crystal_local_placement":{"mace_eV":float(eb[1]),"delta_vs_variant1_kJ_mol":float((eb[1]-eb[0])*96.4853321233),
                "F_target_source_coords_A":f_target.tolist(),"F_target_geometry":plane_metrics(Xf)},
            "bonds_only_to_crystal_lengths":{"mace_eV":float(eb[2]),"delta_vs_variant1_kJ_mol":float((eb[2]-eb[0])*96.4853321233)}}
    out={"code":"NUWFAM","profile":profile.snapshot(),"seed":11,"embed_seed":42,"steps":250,
        "variant_count":len(variants),"embedded_variants":embedded,"crystal_geometry":crystal_geom,
        "returned_rows":[x[0] for x in mapped],
        "top1_by_variant":top1_by_variant,
        "top1":{"rank":mapped[0][0]["rank"],"variant":mapped[0][0]["variant"],"score":mapped[0][0]["score"],
                "mace_eV":mapped[0][0]["mace_eV"],"mace_delta_vs_crystal_kJ_mol":mapped[0][0]["mace_delta_vs_crystal_kJ_mol"]},
        "variant1_rows":[x[0] for x in mapped if x[0]["variant"]==1],"variant1_decomposition":decomposition,
        "geometry_atom_indices_source_0based":{"allyl_C3":list(ALLYL_C3),"F":F_ATOM,"C8":C8_ATOM,"bonds":BONDS},
        "notes":"F-only counterfactual rigidly aligns the crystal local allyl-C3 frame to the generated allyl-C3 frame, transfers only the C8-F bond direction, and preserves the generated C-F length. Bond-only counterfactual independently restores C-F, C-C and C-B lengths to crystal values."}
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k] for k in ("profile","variant_count","top1","top1_by_variant","variant1_rows","variant1_decomposition")},indent=2))

if __name__=="__main__":main()
