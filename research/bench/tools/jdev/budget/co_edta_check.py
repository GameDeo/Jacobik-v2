"""Diagnose Co(III)-EDTA binding for both carboxylate donor representations.

No complex relaxation; only the standard solver and free-ligand embeds.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import DB, stats_from_db
from pools import setup


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 11])
    a = p.parse_args()
    setup()
    from jacobik import ComplexBuilder, LigandInput
    from jacobik.core.profile import snapshot
    from tools.jdev.cases import EDTA

    stats = stats_from_db(DB)
    rows = []
    for label, donor_ids in [('mixed_oxygen', [0, 4, 7, 11, 15, 19]), ('anionic_oxygen', [0, 4, 8, 11, 15, 19])]:
        for embed in ('gfn2', 'mmff'):
            for bends in (False, True):
                for seed in a.seeds:
                    t = time.monotonic()
                    builder = ComplexBuilder('Co', 'octahedral', charge=-1, spin=1,
                                             ligands=[LigandInput(EDTA, donors=[[i] for i in donor_ids])])
                    r = builder.solve(stats=stats, seed=seed, embed=embed, angle_bends=bends, device=a.device)
                    entries = []
                    for conf in r.conformers:
                        m = conf.complex
                        xyz = {at.idx: np.array(at.position) for at in m.atoms}
                        d = [float(np.linalg.norm(xyz[i]-xyz[m.metal_idx])) for i in m.ligands[0].donor_atoms]
                        entries.append(dict(score=float(conf.score), distances=d, max_distance=max(d),
                                            assignment=conf.assignment_idx))
                    row = dict(donors=label, embed=embed, bends=bends, seed=seed, oxidation_state=builder.oxidation_state,
                               seconds=time.monotonic()-t, conformers=entries)
                    rows.append(row)
                    print(label, embed, bends, seed, 'OS', builder.oxidation_state, 'best', entries[0], flush=True)
                    Path(a.out).write_text(json.dumps(dict(profile=snapshot(), rows=rows), indent=2)+'\n')


if __name__ == '__main__':
    main()
