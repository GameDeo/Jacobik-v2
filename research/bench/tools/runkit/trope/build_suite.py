"""Build the trope suite: candidates (frame splits surface_fit + calibration, locked exclusions removed), tags, set cover.

    python -m tools.runkit.trope.build_suite [--features FEATURES.json] [--out research/archive/code/trope_suite/suite.json]
FEATURES.json is made by `--extract` (reads geometry, bond orders and SMILES of the candidates; about 80 s with 1 process)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .. import resources
from . import cover, features, tags

HERE = Path(__file__).resolve()
DEFAULT_OUT = HERE.parents[5] / "research/archive/code/trope_suite/suite.json"
DEFAULT_FEAT = resources.RUN_DIR / "tropes/features.json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--features", default=str(DEFAULT_FEAT))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--max-parents", type=int, default=12)
    a = ap.parse_args(argv)
    from tools.jgen import inputs
    groups, info = features.candidate_groups(inputs.FRAME_V1 / "frame.json")
    fp = Path(a.features)
    if a.extract or not fp.exists():
        feats = features.collect(groups, workers=1)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(json.dumps(dict(info=info, features=feats)))
    F = json.loads(fp.read_text())["features"]
    errs = [f for f in F if "error" in f]
    F = [f for f in F if "error" not in f]
    P = [dict(code=f["code"], n_atoms=f["n_atoms"], tags=sorted(tags.tags_of(f)), meta=f) for f in F]
    ch, unc, freq, params = cover.best_cover(P, a.max_parents)
    by = {p["code"]: p for p in P}
    parents = []
    for c in ch:
        f = by[c["code"]]["meta"]
        parents.append(dict(code=c["code"], n_atoms=c["n_atoms"], split=f["split"], stratum=f["stratum"], metal=f["metal"], cn=f["cn"],
                            coord_geom=f["coord_geom"], new_tags=c["new_tags"], all_tags=c["all_tags"]))
    out = dict(created=time.strftime("%Y-%m-%d"), rules_version=tags.RULES_VERSION, candidates=info,
               candidates_featurised=len(F), feature_errors=len(errs), cover_params=params, max_parents=a.max_parents,
               tag_frequency=dict(sorted(freq.items(), key=lambda x: x[1])), uncovered_tags=unc, parents=parents,
               exclusion_policy="runkit.policy.DataPolicy: reserve_all_codes + motion_validation_codes removed (groups with any excluded member dropped)")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{len(parents)} parents, uncovered {unc}, atoms {sum(p['n_atoms'] for p in parents)} -> {a.out}")


if __name__ == "__main__":
    main()
