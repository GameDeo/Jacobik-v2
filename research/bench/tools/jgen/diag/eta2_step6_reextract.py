"""eta2 package, Step 6: re-extract the QO observations with raw_face_elems (orientation.donor_rows, the extractor's QO
path) for every tmQM / LnQM shard, and check that every archived column reproduces.

  run [N]   N worker processes (default 4): one parquet per shard under OUT/shards (skips shards already written)
  merge     per archive split (X1, X2, X3, LNf): the archived rows (research/archive/data/refit/observations, what the
            production DB was fitted on) unchanged, plus raw_face_elems taken from the re-extracted row with the same
            csd_code, k and bit-identical raw_face. Written only if every archived haptic row is matched. The
            current extractor's drift from the archive (complexes / rows it now adds) is reported, not used.
Parser: fit.parser.TMQMParser (CSD_code and SMILES only; no QM columns). The archive is only read.
"""
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REF = Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/data/ref')
ARCH = Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full/research/archive/data/refit/observations')
OUT = Path('research/data/jgen/campaign/eta2/observations')
COLS = (["csd_code", "hb_min", "ring", "sub", "metal", "geom", "anion", "k", "sub_elems"]
        + [f"g{i}" for i in range(6)] + ["raw_face", "raw_face_elems", "raw_closed", "raw_subs"])
SPLITS = {'X1': 'tmQM', 'X2': 'tmQM', 'X3': 'tmQM', 'LNf': 'lnQM'}


def shards():
    out = []
    for split, ds in SPLITS.items():
        for p in sorted((REF / ds / 'xyz').glob(f'tmQM_{split}s*.xyz')):
            out.append((split, ds, p.stem[len('tmQM_'):]))
    return out


def one(args):
    split, ds, shard = args
    dst = OUT / 'shards' / f'{shard}.parquet'
    if dst.exists():
        return shard, 'skip', 0
    logging.disable(logging.ERROR)
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    from fit.qo import donor_rows
    from fit.parser import TMQMParser
    rows, n, err = [], 0, 0
    for mol in TMQMParser(REF / ds).stream_split(shard):
        n += 1
        try:
            rows.extend(donor_rows(mol))
        except Exception:  # as the extractor: count and continue
            err += 1
    qo = pd.DataFrame(rows, columns=COLS)
    qo["raw_closed"] = qo["raw_closed"].astype("boolean")
    tmp = dst.with_suffix('.tmp')
    qo.to_parquet(tmp)
    tmp.rename(dst)
    return shard, f'{n} mols, {len(qo)} rows, {err} errors', n


def run(nproc):
    (OUT / 'shards').mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(nproc) as ex:
        for shard, msg, _ in ex.map(one, shards()):
            print(shard, msg, flush=True)



def merge():
    rep = {}
    key = lambda d: (d.csd_code + '|' + d.k.astype(str) + '|'  # noqa: E731
                     + d.raw_face.map(lambda f: '' if f is None else np.asarray(f, float).tobytes().hex()))
    for split in SPLITS:
        new = pd.concat([pd.read_parquet(OUT / 'shards' / f'{s}.parquet') for sp, _, s in shards() if sp == split],
                        ignore_index=True)
        old = pd.read_parquet(ARCH / f'observations_QO_{split}.parquet')
        nh, hap = new[new.k >= 10], old.k >= 10
        el = dict(zip(key(nh), nh.raw_face_elems))
        assert len(el) == len(nh), 'duplicate haptic row keys'
        ko = key(old[hap])
        hit = ko.map(lambda x: x in el)
        out = old.copy()
        out.insert(out.columns.get_loc('raw_face') + 1, 'raw_face_elems', None)
        out['raw_face_elems'] = out['raw_face_elems'].astype(object)
        out.loc[hap, 'raw_face_elems'] = pd.Series([list(el[x]) if x in el else None for x in ko], index=ko.index)
        cn, co = new.groupby(['csd_code', 'k']).size(), old.groupby(['csd_code', 'k']).size()
        drift = cn.sub(co, fill_value=0)
        r = dict(rows_archive=len(old), haptic_rows=int(hap.sum()), matched=int(hit.sum()), unmatched=int((~hit).sum()),
                 extractor_drift=dict(rows_new=len(new), codes_only_new=len(set(new.csd_code) - set(old.csd_code)),
                                      codes_only_archive=len(set(old.csd_code) - set(new.csd_code)),
                                      changed_code_k_groups=int((drift != 0).sum())))
        e = out[out.k == 12]
        r['eta2_elems'] = pd.Series(['='.join(sorted(x)) for x in e.raw_face_elems]).value_counts().head(12).to_dict()
        if r['unmatched'] == 0:
            out.to_parquet(OUT / f'observations_QO_{split}.parquet')
        rep[split] = r
        print(split, json.dumps(r), flush=True)
    (OUT / 'reextract_check.json').write_text(json.dumps(rep, indent=1) + '\n')

if __name__ == '__main__':
    merge() if sys.argv[1:2] == ['merge'] else run(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
