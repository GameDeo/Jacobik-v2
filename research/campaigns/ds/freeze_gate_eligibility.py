"""Freeze source-based gate eligibility from the independent input audit, before any new performance result.

Reads only the audit dispositions (input-only evidence) and writes research/campaigns/ds/gate_eligibility_v1.json.
No performance result is read. Historical D3 verdicts are untouched; strata are reported side by side.

  python research/campaigns/ds/freeze_gate_eligibility.py AUDIT_DISPOSITIONS.csv [OUT.json]
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

VERSION = "gate_eligibility.1"
STRATUM = {"no_input_discrepancy_detected": "valid", "confirmed_error_repaired_input": "repaired",
           "ambiguous_target": "ambiguous", "confirmed_error_unrepaired": "unrepaired"}
#: strata that make up the corrected-gate comparison; the others stay visible as separate strata, never dropped silently
PRIMARY_STRATA = ("valid", "repaired")
#: owner decision 2026-11-10 (source_review.1): single declaration is kappa1 on the near O; the repaired target for
#: these follows the generic reconciliation, not the earlier kappa2 overlay
TARGET_AMENDMENTS = {"AGAPEG": "kappa1_near_O", "ILEPOE": "kappa1_near_O"}


def build(csv_path: Path) -> dict:
    raw = Path(csv_path).read_bytes()
    rows = list(csv.DictReader(raw.decode().splitlines()))
    parents = []
    for r in sorted(rows, key=lambda r: (r["stage"], r["code"])):
        stratum = STRATUM[r["disposition"]]
        parents.append(dict(code=r["code"], stage=int(r["stage"]), stage1_primary=r["stage1_primary"] == "True",
                            stratum=stratum, in_corrected_primary=stratum in PRIMARY_STRATA,
                            reference_compatible=r["reference_compatible"] == "True",
                            target_amendment=TARGET_AMENDMENTS.get(r["code"])))
    counts = Counter((p["stage"], p["stratum"]) for p in parents)
    return dict(version=VERSION, source="gate-input audit dispositions (input-only evidence)",
                source_sha256=hashlib.sha256(raw).hexdigest(), primary_strata=list(PRIMARY_STRATA),
                rule="in_corrected_primary = stratum in primary_strata; ambiguous and unrepaired parents stay visible "
                     "as their own strata; failed or missing parents stay in every denominator",
                counts={f"stage{k[0]}_{k[1]}": v for k, v in sorted(counts.items())}, total=len(parents), parents=parents)


if __name__ == "__main__":
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).with_name("gate_eligibility_v1.json")
    doc = build(Path(sys.argv[1]))
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(out, hashlib.sha256(out.read_bytes()).hexdigest(), doc["counts"], doc["total"])
