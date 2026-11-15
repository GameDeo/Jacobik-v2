"""Compare fixed ligand ring torsions in crystal and GFN2 embed for H-2 cal/tail parents."""
from __future__ import annotations
import argparse,json,logging,sys
from pathlib import Path
import numpy as np
sys.path[:0]=["src",".","research/bench"]
from tools.jgen import inputs
from tools.jgen.bench import panels
from tools.runkit.policy import get as get_policy
from tools.runkit import xtb
from jacobik.core.env import build_builder_and_inputs
from jacobik.builder.atom_map import build_atom_mapping
from jacobik.builder import engine
import networkx as nx

OUT=Path("research/data/jgen/campaign/H/h2_census")
class Watch(logging.Handler):
 def __init__(self):super().__init__(level=logging.WARNING);self.messages=[]
 def emit(self,r):
  if "GFN2 embed refinement failed" in r.getMessage():self.messages.append(r.getMessage())
def dih(p):
 p=np.asarray(p,float);b0=-(p[1]-p[0]);b1=p[2]-p[1];b2=p[3]-p[2];b1/=np.linalg.norm(b1)
 v=b0-np.dot(b0,b1)*b1;w=b2-np.dot(b2,b1)*b1
 return float(np.degrees(np.arctan2(np.dot(np.cross(b1,v),w),np.dot(v,w))))
def wrap(x): return abs((x+180)%360-180)
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--shard",type=int,default=0);ap.add_argument("--nshards",type=int,default=1);a=ap.parse_args()
 xtb.require();manifest=json.load(open(inputs.FRAME_V1/"manifest.json"));tasks=[]
 for pn in ("cal","tail"):
  cc=sorted({r["code"] for r in panels.select(manifest,pn)["units"]})
  get_policy().assert_panel(pn,cc,cal_codes=[r["code"] for r in panels.select(manifest,"cal")["units"]] if pn=="cal" else None)
  tasks.extend((c,pn) for c in cc)
 if not 0<=a.shard<a.nshards:raise SystemExit("invalid shard")
 tasks=tasks[a.shard::a.nshards]
 OUT.mkdir(parents=True,exist_ok=True);watch=Watch();logging.getLogger("jacobik.builder.engine").addHandler(watch)
 for k,(code,panel) in enumerate(tasks,1):
  rec={"code":code,"panel":panel,"rings":[]}
  try:
   watch.messages.clear();engine._GFN2_CACHE.clear();engine._GFN2_BOND_CHECK_CACHE.clear()
   sh,ds=inputs.frame_shards()[code];mol=inputs.load_mol(code,sh,ds)
   builder,_,err=build_builder_and_inputs(mol)
   if builder is None:raise RuntimeError(str(err))
   shell,_,_=builder._build_shell(False,False,chelate_embed=True)
   xr=np.asarray([x.position for x in mol.atoms],float);xg=np.asarray([x.position for x in shell.atoms],float)
   amap=build_atom_mapping(mol,gen_mol=shell,X_ref=xr,X_gen=xg)
   ref_to_gen=np.asarray(amap.gen_to_ref_idx,int);xe=xg[ref_to_gen]
   metal_ref=int(amap.ref_to_gen_idx[int(shell.metal_idx)-1])
   g=nx.Graph()
   for i,a in enumerate(mol.atoms):
    if i!=metal_ref:g.add_node(i)
   for b in mol.bonds:
    i,j=int(b.i)-1,int(b.j)-1
    if i!=metal_ref and j!=metal_ref:g.add_edge(i,j)
   rings=sorted((c for c in nx.cycle_basis(g) if len(c)>=5),key=lambda c:(len(c),c))
   for ri,cycle in enumerate(rings):
    rr=[];ee=[]
    n=len(cycle)
    for i in range(n):
     ids=[cycle[(i+j)%n] for j in (-1,0,1,2)]
     rr.append(dih(xr[ids]));ee.append(dih(xe[ids]))
    dif=[wrap(a-b) for a,b in zip(rr,ee)]
    rec["rings"].append({"ring":ri,"n_atoms":n,"atoms":cycle,"reference_torsions_deg":[round(x,2) for x in rr],
     "embed_torsions_deg":[round(x,2) for x in ee],"median_abs_difference_deg":round(float(np.median(dif)),2),
     "max_abs_difference_deg":round(float(max(dif)),2),"any_torsion_diff_gt30":bool(max(dif)>30)})
   rec["status"]="gfn2_fallback" if watch.messages else "ok";rec["fallbacks"]=list(watch.messages)
   rec["mapping_is_ambiguous"]=bool(amap.is_ambiguous)
  except Exception as e:rec.update(status="error",error=f"{type(e).__name__}: {e}",fallbacks=list(watch.messages))
  (OUT/f"rings_{code}_{panel}.json").write_text(json.dumps(rec,indent=1))
  print(f"[{k}/{len(tasks)}] {panel} {code} {rec['status']} rings={len(rec['rings'])}",flush=True)
if __name__=="__main__":main()
