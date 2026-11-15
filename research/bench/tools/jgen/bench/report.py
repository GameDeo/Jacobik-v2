"""Report of finished arms: simple-mode summaries per arm and panel, with the input-label split (d14 v3).

    python -m tools.jgen bench report --labels LABELS.json --reference ROOT --manifest F --repair F \
        --arm NAME=RECORDS_DIR [--arm ...] --panel P --out DIR [--control-dir RECORDS_DIR]

--control-dir: the paired baseline is a control bench run (records), not the stored pools (--reference is then unused).

The labels file is the output of tools/jgen/diag/d14_input_consistency.py (parents -> label). Read only.
Writes DIR/report_<panel>.json and DIR/report_<panel>.md. Nothing is written under data/.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

from tools.jgen.bench import outputs, simple
from tools.jgen.bench.panels import select, unit_keys


def label_split(parents_labels: Dict[str, str], keys: List, summary: Dict) -> Dict:
    """Per input label: parents, units paired, mean e and f change (paired units only)."""
    out: Dict[str, Dict] = {}
    pc = summary["parent_changes"]
    for lab in ("consistent", "ambiguous", "inconsistent", "unlabelled"):
        codes = [c for c in pc if parents_labels.get(c, "unlabelled") == lab]
        if not codes:
            continue
        e = [pc[c]["e"] for c in codes]
        f = [pc[c]["f"] for c in codes]
        out[lab] = dict(parents=len(codes), e_mean=round(sum(e) / len(e), 3), f_mean=round(sum(f) / len(f), 3),
                        band_exceed=[c for c in codes if abs(pc[c]["e"]) > 0.65])
    return out


def main(a) -> int:
    labels_raw = json.load(open(a.labels))
    labels = {p["code"]: p["label"] for p in labels_raw["parents"]} if isinstance(labels_raw, dict) else {}
    manifest = json.load(open(a.manifest))
    keys = unit_keys(select(manifest, a.panel)["units"])
    ref_root = os.path.dirname(os.path.abspath(a.reference)) if os.path.basename(a.reference.rstrip("/")) == "pools" \
        else a.reference
    base_keys = keys
    arms = {}
    control = simple.read_candidate_dir(a.control_dir) if getattr(a, "control_dir", "") else None
    for spec in a.arm:
        name, path = spec.split("=", 1)
        cand = simple.read_candidate_dir(path)
        s = simple.run(a.panel, cand, None if control is not None else ref_root, a.manifest, a.repair,
                       control=control)
        s["label_split"] = label_split(labels, base_keys, s)
        arms[name] = s
    os.makedirs(a.out, exist_ok=True)
    outputs.write_json(os.path.join(a.out, f"report_{a.panel}.json"), arms)
    md = [f"# Arms on panel `{a.panel}`", ""]
    for name, s in arms.items():
        md.append(outputs.arm_block(name, s))
        md.append("Input-label split (d14 v3):")
        md.append(outputs.md_table(["label", "parents", "mean e", "mean f", "band exceedances"], [
            [k, v["parents"], v["e_mean"], v["f_mean"], ", ".join(v["band_exceed"]) or "-"]
            for k, v in s["label_split"].items()]))
        md.append("")
    with open(os.path.join(a.out, f"report_{a.panel}.md"), "w") as fh:
        fh.write("\n".join(md))
    print(json.dumps({k: dict(units_paired=v["units_paired"], e=round(v["e"]["mean"], 2),
                              f=round(v["f"]["mean"], 2)) for k, v in arms.items()}))
    return 0
