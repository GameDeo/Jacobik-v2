"""eta2 + haptic cap, Step 4b: which parents the haptic Li cap acts on (CPU, no embed, no solve).

Eligible = some ligand has a haptic donor group (>= 2 atoms) with net negative formal charge that is not locally
neutral (group + neighbours) -- the gate in engine._gfn2_refine under embed_haptic_li_cap (both the carbon pi cap and
the heteroatom kappa placement count). Run over the same surface_fit pool and exclusions as eta2_step4_holdout.py.
Output: research/data/jgen/campaign/eta2/holdout/cap_eligible.json {code: [n_groups, error]}.
`draw`: stratum D = up to 40 eligible parents not in ETA2_HOLDOUT.json and not excluded by its rule (seed 31, sorted
codes, numpy default_rng choice) -> ETA2_HOLDOUT_D.json.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

OUT = Path('research/data/jgen/campaign/eta2/holdout/cap_eligible.json')


def check(args):
    code, shard, ds = args
    try:
        from rdkit import Chem
        from jacobik.core.env import build_builder_and_inputs
        from tools.jgen import inputs
        mol = inputs.load_mol(code, shard, ds)
        if mol is None:
            return code, [0, 'no geometry']
        builder, _, err = build_builder_and_inputs(mol)
        if builder is None:
            return code, [0, str(err)[:120]]
        n = 0
        for lig in builder.ligands:
            m = Chem.MolFromSmiles(lig.smiles)
            if m is None:
                continue
            for g in lig.donors:
                g = sorted({int(a) for a in g})
                if len(g) < 2 or sum(m.GetAtomWithIdx(a).GetFormalCharge() for a in g) >= 0:
                    continue
                loc = set(g) | {nb.GetIdx() for a in g for nb in m.GetAtomWithIdx(a).GetNeighbors()}
                if sum(m.GetAtomWithIdx(a).GetFormalCharge() for a in loc) < 0:
                    n += 1
        return code, [n, None]
    except Exception as exc:  # a loop over complexes never fails the batch
        return code, [0, f'{type(exc).__name__}: {str(exc)[:100]}']


def main():
    sys.path.insert(0, 'research/bench/tools/jgen/diag')
    import eta2_step4_holdout as H
    from tools.jgen import inputs
    frame = json.load(open(inputs.FRAME_V1 / 'frame.json'))['main_allocation']['splits']
    shards = inputs.frame_shards()
    codes = sorted({g['representative_code'] for g in frame['surface_fit']})
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        res = dict(ex.map(check, [(c, *shards[c]) for c in codes], chunksize=8))
    OUT.write_text(json.dumps(res, indent=1) + '\n')
    print('eligible', sum(v[0] > 0 for v in res.values()), 'errors', sum(v[1] is not None for v in res.values()), 'of', len(res))


def draw():
    import numpy as np
    sys.path.insert(0, 'research/bench/tools/jgen/diag')
    import eta2_step4_holdout as H
    from tools.jgen import inputs
    from tools.jgen.bench import panels
    frame = json.load(open(inputs.FRAME_V1 / 'frame.json'))['main_allocation']['splits']
    locked = json.load(open(H.MAIN / 'research/bench/benchmarks/locked_exclusions.json'))
    excl = ({g['representative_code'] for s in ('calibration', 'motion_validation', 'reserve') for g in frame[s]}
            | set(panels.TAIL) | set(locked['reserve_all_codes']) | set(locked['motion_validation_codes']) | H.named_codes())
    held = {c for cs in json.load(open(H.OUT))['draw'].values() for c in cs}
    el = json.load(open(OUT))
    pool = sorted(c for c, v in el.items() if v[0] > 0 and c not in held and c not in excl)
    d = sorted(np.random.default_rng(31).choice(pool, size=min(40, len(pool)), replace=False).tolist())
    out = OUT.parent / 'ETA2_HOLDOUT_D.json'
    assert not out.exists(), f'{out} exists: drawn once'
    out.write_text(json.dumps(dict(rule=__doc__, pool=len(pool), draw={'D': d}), indent=1) + '\n')
    print(len(pool), d)


if __name__ == '__main__':
    draw() if sys.argv[1:2] == ['draw'] else main()
