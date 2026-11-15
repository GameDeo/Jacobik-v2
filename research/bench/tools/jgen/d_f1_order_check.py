"""CPU validation of the frozen A4b generator/reference atom maps used by F1."""
from __future__ import annotations
import io, json, logging, shutil
from pathlib import Path
import numpy as np
import torch

from tools.jgen import inputs
from tools.jdev import multiseed
from tools.jdev.budget.common import stats_from_db
from jacobik.core.env import build_builder_and_inputs
from tools.jgen.bench.pool_export import _read_frames

ROOT=Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full')
A4B=ROOT/'research/data/jgen/campaign/A4b/xyz'
OUT=ROOT/'research/data/jgen/campaign/D/f1_order_check.json'
CODES=['QEZVEY','VAVBIE','USAPOU','AZUSIA']

def main():
    torch.set_num_threads(1)
    if shutil.which('xtb') is None:
        raise RuntimeError('xtb is not on PATH; refusing to validate fallback embed shells')
    records=[]; failures=[]; stats=stats_from_db(inputs.DB_PATH)
    for code in CODES:
        try:
            mp=A4B/code/f'{code}_control_s11_pool.json'
            manifest=json.loads(mp.read_text())
            symbols=list(manifest['ref_symbols']); r2g=np.asarray(manifest['gen_to_ref_idx'],int)
            ref_X=np.asarray(manifest['ref_X'],float)
            if sorted(r2g.tolist())!=list(range(len(symbols))) or len(r2g)!=len(ref_X):
                raise ValueError('manifest map is not a bijection matching reference coordinates')
            shard,dataset=inputs.frame_shards()[code]
            input_mol=inputs.load_mol(code,shard,dataset)
            if [a.symbol for a in input_mol.atoms] != symbols:
                raise ValueError('parsed input symbols differ from the A4b reference symbols')
            builder,_,err=build_builder_and_inputs(input_mol)
            if builder is None: raise RuntimeError(f'builder failed: {err}')
            stream=io.StringIO(); handler=logging.StreamHandler(stream); logger=logging.getLogger('jacobik.builder.engine'); logger.addHandler(handler)
            try: ctx,assign,bw,_=multiseed.prepare(builder,11,device='cpu',angle_bends=True,stats=stats)
            finally: logger.removeHandler(handler)
            if 'GFN2 embed refinement failed' in stream.getvalue():
                raise RuntimeError('GFN2 embed refinement failed; rejecting MMFF fallback shell')
            gen_symbols=[a.symbol for a in ctx.mol.atoms]
            if [gen_symbols[j] for j in r2g] != symbols:
                raise ValueError('builder-order symbols do not match reference symbols through gen_to_ref_idx')
            gen_to_ref=np.argsort(r2g)
            gen_X=ref_X[gen_to_ref]
            if not np.array_equal(gen_X[r2g],ref_X):
                raise ValueError('coordinate round-trip failed')
            frames=list(_read_frames(mp.parent/manifest['rows'][0]['file']))
            if not frames or np.asarray(frames[0][2]).shape != gen_X.shape:
                raise ValueError('stored A4b pool frame shape does not match builder-order map')
            records.append({'code':code,'status':'ok','n_atoms':len(symbols),'map_bijective':True,
                            'symbol_map_ok':True,'coord_roundtrip':True,'pool_frame_shape':list(frames[0][2].shape)})
        except Exception as exc:
            failures.append({'code':code,'error':f'{type(exc).__name__}: {exc}'[:500]})
        print(f'{code}: {"ok" if not failures or failures[-1]["code"] != code else "failed"}',flush=True)
    result={'parents_requested':len(CODES),'parents_ok':len(records),'failures':failures,'records':records}
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
    if failures: raise SystemExit(1)
if __name__=='__main__': main()
