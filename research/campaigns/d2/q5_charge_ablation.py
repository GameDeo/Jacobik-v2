"""Targeted Q5 charge-filter ablation; locked parents remain excluded in both fits."""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from fit import q5
from fit.pipeline.compare import load_container
from fit.sanity import excluded_charge_codes, locked_codes
from fit.fitter import _evidence_weights

ap = argparse.ArgumentParser()
ap.add_argument('--obs-dir', type=Path, required=True)
ap.add_argument('--census', type=Path, required=True)
ap.add_argument('--locked-path', type=Path, required=True)
ap.add_argument('--db', type=Path, required=True)
ap.add_argument('--keys-json', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
capture = json.loads(a.keys_json.read_text())['changed']
keys = set()
for code in ('AJIXUN', 'AFOWIB'):
    for row in capture[code]:
        q, key = row['quantity_key']
        if q == 'Q5': keys.add(repr(tuple(key)))
locked = locked_codes(a.locked_path)
census = json.loads(a.census.read_text())
charge_bad = excluded_charge_codes(census)
db, meta, _ = load_container(a.db)
frames = []
for split in ('X1', 'X2', 'X3', 'LNf'):
    p = a.obs_dir / f'observations_Q5_{split}.parquet'
    frame = pd.read_parquet(p, columns=['key','value','csd_code'], filters=[('key','in',sorted(keys))])
    frame = frame.loc[~frame['csd_code'].astype(str).isin(locked)]
    frames.append(frame)
raw = pd.concat(frames, ignore_index=True)
results = []
for key in sorted(keys):
    df = raw.loc[raw['key'] == key].copy()
    filtered = df.loc[~df['csd_code'].astype(str).isin(charge_bad)].copy()
    fits = {}
    for name, part in (('charge_excluded',filtered),('charge_included',df)):
        if len(part) < q5.N_MIN:
            fits[name] = None
            continue
        weights = _evidence_weights(part)
        fits[name] = q5.fit_from_observations({key: (part['value'].to_numpy(np.float64), weights)}).get('Q5:'+key)
    actual = db.get('Q5:'+key)
    def model_fields(entry):
        return None if entry is None else {k:entry.get(k) for k in ('type','model','means','kappas','weights')}
    results.append(dict(key=ast.literal_eval(key), n_locked_excluded=len(df), n_charge_excluded=len(filtered),
                        n_charge_removed=len(df)-len(filtered), fitted_filtered=model_fields(fits['charge_excluded']),
                        fitted_with_charge=model_fields(fits['charge_included']), recorded_d2=model_fields(actual),
                        filtered_matches_recorded=model_fields(fits['charge_excluded']) == model_fields(actual)))
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps({'note':'Typed raw extraction; locked parents excluded in both arms. No held-out results evaluated.',
                            'source_obs':str(a.obs_dir), 'keys':results}, indent=2, default=str))
print(json.dumps([{'key':x['key'], 'n_charge_removed':x['n_charge_removed'],
                   'filtered_matches_recorded':x['filtered_matches_recorded']} for x in results]))
