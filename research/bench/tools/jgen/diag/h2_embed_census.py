"""CPU census for Package H-2. Reads cal and named-tail geometries only; writes under research/data/jgen/campaign/H."""
from __future__ import annotations
import argparse, json, logging, os, shutil, sys
from pathlib import Path
import numpy as np
sys.path[:0]=["src",".","research/bench"]

from tools.jgen import inputs
from tools.jgen.bench import panels
from tools.runkit.policy import get as get_policy
from tools.runkit import xtb
from jacobik.core.env import build_builder_and_inputs
from jacobik.builder.atom_map import build_atom_mapping
from jacobik.builder.embed import ez_variants
from rdkit import Chem

OUT=Path("research/data/jgen/campaign/H/h2_census")
def xyz(m): return np.asarray([a.position for a in m.atoms],float)

class FallbackWatch(logging.Handler):
    def __init__(self): super().__init__(level=logging.WARNING); self.messages=[]
    def emit(self,record):
        msg=record.getMessage()
        if "GFN2 embed refinement failed" in msg: self.messages.append(msg)

def angle(a,b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a@b,-1,1))))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--panel",choices=["cal","tail"],action="append")
    ap.add_argument("--limit",type=int,default=0)
    ap.add_argument("--shard",type=int,default=0,help="zero-based worker index")
    ap.add_argument("--nshards",type=int,default=1)
    ap.add_argument("--codes",default="",help="optional comma-separated parent codes to rerun")
    a=ap.parse_args()
    xtb.require()
    panels_req=a.panel or ["cal","tail"]
    manifest=json.load(open(inputs.FRAME_V1/"manifest.json"))
    codes=[]
    for pn in panels_req:
        units=panels.select(manifest,pn)["units"]
        cc=sorted({u["code"] for u in units})
        get_policy().assert_panel(pn,cc,cal_codes=[u["code"] for u in panels.select(manifest,"cal")["units"]] if pn=="cal" else None)
        codes.extend((code,pn) for code in cc if (code,pn) not in codes)
    if a.limit: codes=codes[:a.limit]
    if a.codes:
        wanted={x.strip().upper() for x in a.codes.split(",") if x.strip()}
        codes=[x for x in codes if x[0] in wanted]
    if not 0 <= a.shard < a.nshards: raise SystemExit("invalid shard index")
    codes=codes[a.shard::a.nshards]
    index=inputs.frame_shards()
    OUT.mkdir(parents=True,exist_ok=True)
    watch=FallbackWatch(); logging.getLogger("jacobik.builder.engine").addHandler(watch)
    records=[]
    from jacobik.builder import engine as engine_mod
    for ix,(code,panel) in enumerate(codes,1):
        watch.messages.clear()
        # Isolate fallback provenance per parent: GFN2 geometries are normally cached by ligand SMILES.
        # Clearing between parents lets this census detect a failed refinement even when a later parent
        # contains the same ligand. Cache remains active within each parent/build.
        engine_mod._GFN2_CACHE.clear()
        engine_mod._GFN2_BOND_CHECK_CACHE.clear()
        rec={"code":code,"panel":panel}
        try:
            sh,ds=index[code]; mol=inputs.load_mol(code,sh,ds)
            if mol is None: raise RuntimeError("input mol unavailable")
            builder,_,err=build_builder_and_inputs(mol)
            if builder is None: raise RuntimeError(f"builder: {err}")
            shell,_,_=builder._build_shell(False,False,chelate_embed=True)
            Xr,xg=xyz(mol),xyz(shell)
            amap=build_atom_mapping(mol,gen_mol=shell,X_ref=Xr,X_gen=xg)
            ref_to_gen=np.asarray(amap.gen_to_ref_idx,int)
            if len(ref_to_gen)!=len(mol.atoms): raise RuntimeError("atom map length mismatch")
            gen_to_ref=np.asarray(amap.ref_to_gen_idx,int)
            donor_type_ref={}
            for site in shell.coord_sites:
                for gd in site.donor_indices:
                    donor_type_ref[int(gen_to_ref[int(gd)-1])]=str(site.kind)
            metal_ref=int(gen_to_ref[int(shell.metal_idx)-1])
            # Bond displacement by reference bond class; use topology supplied by the input geometry.
            atom_bonds=[]
            for b in mol.bonds:
                i,j=int(b.i)-1,int(b.j)-1
                if min(i,j)<0 or max(i,j)>=len(mol.atoms): continue
                dref=float(np.linalg.norm(Xr[i]-Xr[j]))
                dgen=float(np.linalg.norm(xg[ref_to_gen[i]]-xg[ref_to_gen[j]]))
                si,sj=mol.atoms[i].symbol,mol.atoms[j].symbol
                donor_kinds=[donor_type_ref[k] for k in (i,j) if k in donor_type_ref]
                donor_class=("metal-"+donor_type_ref.get(j if i==metal_ref else i,"unknown")) if metal_ref in (i,j) else ("ligand-near-"+"+".join(sorted(set(donor_kinds))) if donor_kinds else "ligand-internal")
                atom_bonds.append({"atoms":[i,j],"elements":"-".join(sorted([si,sj])),"order":float(b.order),
                    "donor_class":donor_class,
                    "reference_A":dref,"embed_A":dgen,"delta_A":dgen-dref if np.isfinite(dgen) else None,
                    "stretched_gt_0p1":bool(np.isfinite(dgen) and abs(dgen-dref)>.1)})
            # Three-coordinate ligand centres: sum of the three heavy-neighbour angles.
            adj=[[] for _ in mol.atoms]
            for b in mol.bonds:
                i,j=int(b.i)-1,int(b.j)-1
                if 0<=i<len(adj) and 0<=j<len(adj): adj[i].append(j);adj[j].append(i)
            angle_sums=[]
            for i,ns in enumerate(adj):
                ns=[j for j in ns if mol.atoms[j].symbol!="H" and mol.atoms[i].symbol!="H"]
                if len(ns)==3 and mol.atoms[i].symbol in ("C","N","B","P","S"):
                    angles=[]
                    for u,v in ((0,1),(0,2),(1,2)):
                        a1=Xr[ns[u]]-Xr[i]; b1=Xr[ns[v]]-Xr[i]
                        angles.append(angle(a1,b1))
                    ang_embed=[]
                    if i<len(ref_to_gen):
                        ci=int(ref_to_gen[i])
                        for u,v in ((0,1),(0,2),(1,2)):
                            if ns[u]<len(ref_to_gen) and ns[v]<len(ref_to_gen):
                                ang_embed.append(angle(xg[int(ref_to_gen[ns[u]])]-xg[ci],xg[int(ref_to_gen[ns[v]])]-xg[ci]))
                    angle_sums.append({"atom":i,"element":mol.atoms[i].symbol,"reference_sum_deg":sum(angles),
                        "embed_sum_deg":sum(ang_embed) if len(ang_embed)==3 else None,
                        "reference_planarity_lt345":sum(angles)<345,
                        "embed_pyramidal_lt345":len(ang_embed)==3 and sum(ang_embed)<345})
            # Donor-aware features and E/Z variants on each ligand.
            haptic_anions=[]; ez=[]
            shell_sites={}
            for s in shell.coord_sites:
                for d in s.donor_indices: shell_sites.setdefault(int(d)-1,[]).append(s.kind)
            for li,lig in enumerate(builder.ligands):
                variants=ez_variants(lig.smiles,lig.donors)
                base=Chem.MolFromSmiles(lig.smiles)
                for vi,var in enumerate(variants):
                    for b,c,a0,e,cis in var:
                        same_haptic=any(b in grp and c in grp and len(grp)>=3 for grp in lig.donors)
                        ez.append({"ligand":li,"variant":vi,"bond":[b,c],"cis":bool(cis),"inside_haptic_group":same_haptic})
                if base is not None:
                    for gi,grp in enumerate(lig.donors):
                        if len(grp)<2: continue
                        q=sum(base.GetAtomWithIdx(k).GetFormalCharge() for k in grp)
                        if q<0: haptic_anions.append({"ligand":li,"group":list(grp),"net_formal_charge":q,
                            "donor_symbols":[base.GetAtomWithIdx(k).GetSymbol() for k in grp]})
            # Capture true solver step 0 with one CPU optimization step. This is a placement diagnostic only;
            # it records no energy comparison and does not tune on the result.
            from jacobik.ik.ik_solver import JointIKSolver
            from tools.jdev.budget.common import stats_from_db
            starts=[]
            metal=int(shell.metal_idx)-1
            donor_groups=[[int(d)-1 for d in s.donor_indices] for s in shell.coord_sites]
            targets=np.asarray([np.asarray(s.position,float) for s in shell.coord_sites],float)
            def hook(step,detail):
                if step==0:
                    starts.append(np.array(detail["X"].detach().cpu().numpy(),copy=True))
            JointIKSolver.trace_hook=staticmethod(hook)
            try:
                builder.solve(seed=11,stats=stats_from_db(inputs.DB_PATH),angle_bends=True,device="cpu",n_ik_steps=1,n_conformers=1)
            finally:
                JointIKSolver.trace_hook=None
            placement={"rows":0,"share_rows_any_donor_gt_0p5A":None,"per_donor_median_A":[],"per_donor_p95_A":[]}
            if starts:
                X0=starts[0]
                # One donor vector per site; haptic groups use their centroid, not each donor atom separately.
                dev=[]; vang=[]; md=[]; row_details=[]
                mpos=np.mean(X0[:,metal,:],axis=0)
                target_vec=targets-mpos[None,:]
                target_vec=target_vec/np.maximum(np.linalg.norm(target_vec,axis=1,keepdims=True),1.e-12)
                for row in X0:
                    rowdev=[]; rowang=[]; rowmd=[]
                    for si,group in enumerate(donor_groups):
                        pos=row[group].mean(axis=0)
                        rowdev.append(float(np.linalg.norm(targets[si]-pos)))
                        dv=pos-row[metal]; dv=dv/max(float(np.linalg.norm(dv)),1.e-12)
                        rowang.append(angle(dv,target_vec[si]))
                        rowmd.append(float(np.linalg.norm(pos-row[metal])))
                    dev.append(rowdev);vang.append(rowang);md.append(rowmd)
                    row_details.append({"row":int(len(row_details)),"donor_site_deviation_A":[round(x,5) for x in rowdev],
                        "donor_metal_distance_A":[round(x,5) for x in rowmd],"donor_vector_error_deg":[round(x,4) for x in rowang]})
                dev=np.asarray(dev,float);vang=np.asarray(vang,float);md=np.asarray(md,float)
                placement={"rows":int(len(dev)),"share_rows_any_donor_gt_0p5A":float(np.mean((dev>.5).any(axis=1))),
                    "per_donor_median_A":np.median(dev,axis=0).round(4).tolist(),
                    "per_donor_p95_A":np.quantile(dev,.95,axis=0).round(4).tolist(),
                    "per_donor_metal_distance_median_A":np.median(md,axis=0).round(4).tolist(),
                    "per_donor_metal_distance_p95_A":np.quantile(md,.95,axis=0).round(4).tolist(),
                    "per_donor_vector_error_median_deg":np.median(vang,axis=0).round(3).tolist(),
                    "per_donor_vector_error_p95_deg":np.quantile(vang,.95,axis=0).round(3).tolist(),
                    "row_metrics":row_details}
            rec.update({"status":"ok","n_atoms":len(shell.atoms),"n_bonds":len(atom_bonds),"bonds":atom_bonds,
                "mapping_is_ambiguous":bool(amap.is_ambiguous),"mapping_num_automorphisms":int(amap.num_automorphisms),
                "angle_sums":angle_sums,"ez_variants":ez,"haptic_anions":haptic_anions,
                "step0_placement":placement,"fallbacks":list(watch.messages)})
            if watch.messages: rec["status"]="gfn2_fallback"
        except Exception as exc:
            rec.update(status="error",error=f"{type(exc).__name__}: {exc}",fallbacks=list(watch.messages))
        records.append(rec)
        # per-parent logging and durable checkpointing
        (OUT/f"{code}_{panel}.json").write_text(json.dumps(rec,indent=1))
        (OUT/f"census_partial_{a.shard}.json").write_text(json.dumps({"panels":panels_req,"completed":ix,"total":len(codes),
            "parents":records},indent=1))
        print(f"[{ix}/{len(codes)}] {panel} {code} {rec['status']}",flush=True)
    print(f"wrote {OUT/f'census_partial_{a.shard}.json'}; rows={len(records)}",flush=True)
if __name__=="__main__": main()
