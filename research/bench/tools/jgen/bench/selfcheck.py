"""B1 self-checks of the simple mode (research/archive/docs/roadmap/roadmap_generation_v1.md 8.3): run on the stored pools, CPU only.

1. Frozen control against itself must give zero change everywhere.
2. Frozen control against the S2b regeneration must reproduce the repeatability of [D12] (696 units, 0.65 kJ/mol band).
"""
from __future__ import annotations

import os
from typing import Dict

from tools.jgen.bench import simple
from tools.jgen.bench.panels import select, unit_keys


def control_vs_self(reference_root: str, manifest_path: str, repair_path: str, panel: str = "all") -> Dict:
    import json
    manifest = json.load(open(manifest_path))
    keys = unit_keys(select(manifest, panel)["units"])
    frozen = manifest.get("base_provenance", {})
    rep = simple.Repairs(reference_root)
    cand = simple.records_from_pool_dir(os.path.join(reference_root, "pools"), keys, rep, effective=True)
    for r in cand.values():
        r.update(frame_sha256=frozen.get("frame_sha256"), constraints_db_sha256=frozen.get("constraints_db_sha256"),
                 code_git_commit=frozen.get("code_git_commit"))
    return simple.run(panel, cand, reference_root, manifest_path, repair_path)


def control_vs_s2b(reference_root: str, manifest_path: str, repair_path: str, panel: str = "all") -> Dict:
    import json
    manifest = json.load(open(manifest_path))
    keys = unit_keys(select(manifest, panel)["units"])
    frozen = manifest.get("base_provenance", {})
    cand = simple.records_from_pool_dir(os.path.join(reference_root, "s2b_eval_state_v1", "pools"), keys,
                                        effective=False)
    for r in cand.values():
        r.update(frame_sha256=frozen.get("frame_sha256"), constraints_db_sha256=frozen.get("constraints_db_sha256"),
                 code_git_commit="f71f196 (S2b, registry record 22)")
    return simple.run(panel, cand, reference_root, manifest_path, repair_path)
