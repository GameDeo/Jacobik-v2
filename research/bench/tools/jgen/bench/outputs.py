"""JSON and Markdown writers for the bench (never writes under data/)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)


def fmt(x, nd=2) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        if x != x:
            return "nan"
        return f"{x:.{nd}f}"
    return str(x)


def md_table(header: List[str], rows: List[List]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(fmt(c) if not isinstance(c, str) else c for c in r) + " |")
    return "\n".join(out)


def arm_block(name: str, s: Dict) -> str:
    """Markdown block of one arm on one panel (from metrics.summarize plus the simple-mode fields)."""
    lines = [f"### {name} / panel `{s.get('panel', '?')}`", ""]
    lines.append(md_table(["quantity", "mean (kJ/mol)", "95% CI (parent bootstrap)"], [
        ["e (top-1 change)", s["e"]["mean"], f"[{fmt(s['e']['ci95'][0])}, {fmt(s['e']['ci95'][1])}]"],
        ["f (floor change)", s["f"]["mean"], f"[{fmt(s['f']['ci95'][0])}, {fmt(s['f']['ci95'][1])}]"],
        ["r (regret change)", s["r"]["mean"], f"[{fmt(s['r']['ci95'][0])}, {fmt(s['r']['ci95'][1])}]"],
    ]))
    lines += ["", f"- units in panel {s['units_panel']}, paired {s['units_paired']}, parents {s['parents_paired']}",
              f"- valid top-1 share {fmt(s['valid_top1_share'], 3)}; top-1 within 4.184 of floor "
              f"{fmt(s['top1_within_4184_share'], 3)}",
              f"- parents with |d_p^(e)| above the 0.65 band: {s['parent_band_exceed_count']} "
              f"{s['parent_band_exceed']}",
              f"- parent mean losses above 10 kJ/mol: {s['parent_mean_losses_gt10']}",
              f"- units above 10 kJ/mol (|change| of top-1): {len(s['units_gt10'])}",
              f"- worst single-unit loss: {s['worst_single_seed_loss']}",
              f"- tail counts (valid top-1 above 100/200/500/1000): {s['tail_counts']}",
              f"- ledger entries: {len(s['ledger'])}; not in cohort: {s.get('not_in_cohort', [])}", ""]
    return "\n".join(lines)
