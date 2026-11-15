"""Census of saturated rings of size 5, 7, 8+ (and 6 for reference): share of unrelaxed pool rows (seed 11) whose ring
torsions differ from the reference by >= 30 deg after the best ring relabelling (dihedral group, conservative)."""
import json, sys, collections
from pathlib import Path
import numpy as np
from tools.jgen.a4t import metrics2 as m2
from tools.jgen.bench.pool_export import _read_frames
from tools.runkit import geom
XYZ=Path("/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/data/audit/2026-11-08/rc_diagnosis/pm/a4r-solve/control/xyz")
def tors_min_diff(refX, X, ring):
    n=len(ring); tr=np.array(m2.ring_torsions(refX,ring)); tw=np.array(m2.ring_torsions(X,ring))
    best=1e9
    for s in range(n):
        for rev in (False,True):
            idx=[(s+k)%n for k in range(n)]
            t=tw[idx][::-1] if rev else tw[idx]
            for sg in (1,-1):
                d=np.abs((tr-sg*t+180)%360-180).max(); best=min(best,d)
    return best
out=[]
for d in sorted(XYZ.iterdir()):
    code=d.name
    mp=d/f"{code}_control_s11_pool.json"
    if not mp.exists(): continue
    man=json.loads(mp.read_text()); sy=man["ref_symbols"]; R=np.asarray(man["ref_X"],float)
    try:
        metal=geom.find_metal(sy)
        donors=geom.donors_of(R,sy,metal,pad=.6).tolist()
        rings=m2.ring_atoms(R,sy,metal,donors,min_size=5)
    except Exception as e:
        continue
    sats=[r for r in rings["sat"] if len(r)!=6]
    if not sats: continue
    diffs=collections.defaultdict(list)
    for hdr,_,Xg in _read_frames(d/man["rows"][0]["file"]):
        X=np.asarray(Xg)[np.asarray(man["gen_to_ref_idx"],int)]
        for r in sats: diffs[len(r)].append(tors_min_diff(R,X,r))
    rec=dict(code=code,sizes={k:dict(n_rings=sum(1 for r in sats if len(r)==k),rows=len(v)//max(1,sum(1 for r in sats if len(r)==k)),share_ge30=float(np.mean(np.array(v)>=30))) for k,v in diffs.items()})
    out.append(rec); print(rec,flush=True)
Path("/tmp/cov/ring_census.json").write_text(json.dumps(out,indent=1))
