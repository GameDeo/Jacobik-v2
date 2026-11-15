"""Single-point MACE cost of census-flagged bond defects on selected top-1 rows."""
import json
from pathlib import Path
import numpy as np
from tools.jgen import inputs
from jacobik.core.env import MACE_MODEL_PATH
from jacobik.core.energy import MACEPolarEvaluator

ROOT=Path("research/data/jgen/campaign")
OUT=Path("research/data/jgen/campaign/H/h2_census/bond_reset_cost.json")
CASES=[
 ("NUWFAM","D1R2plus","haptic-internal"),
 ("QEZVEY","C1","haptic-nearby"),
 ("RIQDEB","C1","haptic-nearby"),
 ("XOJHIN","C1","coordinated-pi"),
]
def getxyz(path):
 l=path.read_text().splitlines();n=int(l[0]);sy=[];X=[]
 for z in l[2:2+n]: 
  a=z.split();sy.append(a[0]);X.append([float(v) for v in a[1:4]])
 return sy,np.asarray(X,float)
def main():
 calc=MACEPolarEvaluator(model="polar-1-s",device="cpu",dtype="float64",model_path=str(MACE_MODEL_PATH),threads=1)
 results=[]
 for code,arm,kind in CASES:
  census=json.load(open(f"research/data/jgen/campaign/H/h2_census/{code}_cal.json"))
  xyz=(Path("research/data/jgen/campaign/D/nuwfam/xyz/R2_top1.xyz") if arm=="D1R2plus" else
       ROOT/f"B/g/g3c0_remaining/{arm}/xyz/{code}/{code}_{arm}_s11_top1.xyz")
  sy,X=getxyz(xyz);sh,ds=inputs.frame_shards()[code];mol=inputs.load_mol(code,sh,ds)
  bonds=census["bonds"]
  if kind=="haptic-internal":
   chosen=[b for b in bonds if not b["donor_class"].startswith("metal-") and abs(b["delta_A"])>.1]
  elif kind=="haptic-nearby":
   chosen=[b for b in bonds if b["donor_class"]=="ligand-near-haptic" and abs(b["delta_A"])>.1]
  else:
   ring=json.load(open(f"research/data/jgen/campaign/H/h2_census/rings_{code}_cal.json"))["rings"][0]
   cycle=set(ring["atoms"])
   chosen=[b for b in bonds if b["elements"]=="C-C" and b["order"]>1.05 and set(b["atoms"])<=cycle]
  Xr=X.copy()
  for b in chosen:
   i,j=b["atoms"];v=Xr[j]-Xr[i];d=np.linalg.norm(v)
   if d<1e-8: continue
   dv=(b["reference_A"]-d)*v/d/2
   Xr[i]-=dv;Xr[j]+=dv
  energies=calc.energies([(sy,X,int(mol.charge),int(mol.spin_state)),(sy,Xr,int(mol.charge),int(mol.spin_state))],batch_size=1)
  results.append({"code":code,"arm":arm,"kind":kind,"bonds_reset":[{"atoms":b["atoms"],"elements":b["elements"],"embed_delta_A":b["delta_A"],"reference_A":b["reference_A"]} for b in chosen],
   "mace_delta_reset_kj_mol":float((energies[1]-energies[0])*96.4853321233),"energy_ev_before_after":energies.tolist()})
  print(code,kind,len(chosen),results[-1]["mace_delta_reset_kj_mol"],flush=True)
 OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(results,indent=2))
if __name__=="__main__":main()
