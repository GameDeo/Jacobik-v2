"""Bounded A4 MACE timing sample; discards energies and writes timings only."""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
import numpy as np
import torch
from jacobik.benchmark.energy import MACEPolarEvaluator


def run(centers_dir, paths_dir, pools_dir, model_path, out_path, count=20):
    if not torch.cuda.is_available():
        raise RuntimeError("A4 timing requires the GPU listed by nvidia-smi")
    torch.cuda.set_per_process_memory_fraction(0.45, device=0)
    centers=[]
    for center in sorted(Path(centers_dir).glob("*_baseline.npz")):
        code,seed,_=center.stem.split("_")
        with np.load(center,allow_pickle=False) as c:
            n=len(c["X"])
        pool=Path(pools_dir)/f"{code}_{seed}_control.npz"
        if not pool.exists(): continue
        with np.load(pool,allow_pickle=False) as p:
            n=len(p["ref_symbols"])
        motion_file=Path(paths_dir)/f"{code}_baseline_{center.stem}.json"
        if not motion_file.exists(): continue
        centers.append((n,code,int(seed),center,motion_file,pool))
    centers.sort(key=lambda x:(x[0],x[1],x[2]))
    if len(centers)<count: raise RuntimeError(f"only {len(centers)} eligible baseline centers for {count} timing samples")
    picks=[centers[round(i*(len(centers)-1)/(count-1))] for i in range(count)] if count>1 else [centers[len(centers)//2]]
    evaluator=MACEPolarEvaluator(model="polar-1-s",device="cuda",dtype="float64",
                                 model_path=model_path,mode="energy-only",threads=1)
    load_t=time.perf_counter();evaluator.load();torch.cuda.synchronize();load_s=time.perf_counter()-load_t
    rows=[]
    for i,(n,code,seed,center_path,motion_path,pool_path) in enumerate(picks):
        doc=json.loads(motion_path.read_text())
        motion=next((m for m in doc.get("motions",[]) if m.get("selected") and m.get("valid") and "positions" in m),None)
        if motion is None: raise RuntimeError(f"no selected valid motion for timing center {code}_{seed}")
        point=i%len(motion["positions"])
        with np.load(pool_path,allow_pickle=False) as p:
            symbols=[str(x) for x in p["ref_symbols"]]
            charge=int(np.asarray(p["charge"]).item()); multiplicity=int(np.asarray(p["mult"]).item())
        X=np.asarray(motion["positions"][point],dtype=np.float64)
        torch.cuda.synchronize();t=time.perf_counter()
        energy=evaluator.energy(symbols,X,charge,multiplicity)
        torch.cuda.synchronize();elapsed=time.perf_counter()-t
        # Explicitly discard the timing-sample energy; it is not a Step 4 label.
        rows.append(dict(index=i,code=code,seed=seed,atoms=n,motion_id=motion["motion_id"],t=motion["t"][point],wall_s=elapsed))
        del energy
    result=dict(schema="nci_v2_step4_a4_timing_v1",gpu=torch.cuda.get_device_name(0),
                mem_fraction=0.45,model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
                model_load_s=load_s,points=count,point_rows=rows,
                median_eval_s=float(np.median([r["wall_s"] for r in rows])),
                p95_eval_s=float(np.quantile([r["wall_s"] for r in rows],.95)),
                mean_eval_s=float(np.mean([r["wall_s"] for r in rows])),
                timings_only=True,energies_retained=False)
    Path(out_path).parent.mkdir(parents=True,exist_ok=True);Path(out_path).write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--centers',required=True);p.add_argument('--paths',required=True);p.add_argument('--pools',required=True);p.add_argument('--model',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    r=run(a.centers,a.paths,a.pools,a.model,a.out);print(json.dumps({k:r[k] for k in ('gpu','points','model_load_s','median_eval_s','p95_eval_s','mean_eval_s')}))
