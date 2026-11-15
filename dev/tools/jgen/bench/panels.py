"""Panels (research/archive/docs/roadmap/roadmap_generation_v1.md 1.4.2), defined from the frozen manifest, never from outcomes.

A panel never removes a failed unit: failures stay in the denominator and appear in the failure ledger.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

#: the 10 parents with any top-1 dE > 500 kJ/mol in the stored pools (d01)
TAIL = ["DAMVAQ", "HOXJUZ", "IFUROQ", "JEGPUJ", "NEPNEF", "SOBHAS", "USAPOU", "XEYTUT", "YUCZUS", "ZELTAP"]
#: named failure cases of roadmap section 1.4.2
FAILURES = ["IRAQUO", "RELCOA", "ACUYUV", "NALJOZ", "IFAPUD", "KADCUN", "LAWRAC", "ROJMEJ", "SUQWEG", "NEPNEF",
            "IWOLEN", "UQUNUQ", "VENGUR", "ZELTAP", "BODCAY", "LIYTAQ", "OBABAX", "ADINOR"]
PANELS = ("tail", "cal", "val", "failures", "all")
SPLIT_OF_PANEL = {"cal": "calibration", "val": "motion_validation"}


def load_manifest(path: Path) -> Dict:
    return json.load(open(path))


def tasks(manifest: Dict) -> List[Dict]:
    """Every task record of the manifest (sorted by code, seed)."""
    return sorted(manifest["tasks"].values(), key=lambda r: (r["code"], int(r["seed"])))


def select(manifest: Dict, panel: str) -> Dict[str, List[Dict]]:
    """Units of a panel, split into 'units' (all tasks in the panel, any status) and 'not_in_cohort' (named cases the
    manifest does not contain). Only tasks of the splits named by the panel are returned; the status is kept."""
    if panel not in PANELS:
        raise ValueError(f"unknown panel {panel!r}; panels: {PANELS}")
    all_tasks = tasks(manifest)
    if panel in SPLIT_OF_PANEL:
        units = [t for t in all_tasks if t.get("split") == SPLIT_OF_PANEL[panel]]
        return {"units": units, "not_in_cohort": []}
    if panel == "all":
        units = [t for t in all_tasks if t.get("split") in ("calibration", "motion_validation")]
        return {"units": units, "not_in_cohort": []}
    if panel == "tail":
        units = [t for t in all_tasks if t["code"] in TAIL and t.get("split") in ("calibration", "motion_validation")]
        return {"units": units, "not_in_cohort": sorted(set(TAIL) - {t["code"] for t in units})}
    # failures: every named case, found in any split of the manifest
    found = {t["code"] for t in all_tasks}
    units = [t for t in all_tasks if t["code"] in FAILURES]
    return {"units": units, "not_in_cohort": sorted(c for c in FAILURES if c not in found)}


def unit_keys(units: List[Dict]) -> List[tuple]:
    return [(t["code"], int(t["seed"])) for t in units]
