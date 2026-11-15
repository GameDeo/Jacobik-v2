"""Reproduction stage: entry-by-entry comparison of two constraint DBs (tolerance and exact modes)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

PRODUCTION_QUANTITIES = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "QO", "Q6P"]


def _align_mixture_components(test: Dict[str, Any], target: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Align mixture components by minimum parameter distance, independent of storage order."""
    test, target = dict(test), dict(target)
    means_test, means_target = test.get("means"), target.get("means")
    if (not isinstance(means_test, list) or not isinstance(means_target, list)
            or len(means_test) != len(means_target) or len(means_test) < 2):
        return test, target

    from scipy.optimize import linear_sum_assignment

    k = len(means_test)
    fields = ("means", "vars", "covs", "covars", "kappas", "weights")
    cost = np.zeros((k, k), dtype=float)
    for field in fields:
        a, b = test.get(field), target.get(field)
        if not (isinstance(a, list) and isinstance(b, list) and len(a) == len(b) == k):
            continue
        for i in range(k):
            for j in range(k):
                av, bv = np.asarray(a[i], dtype=float), np.asarray(b[j], dtype=float)
                if av.shape != bv.shape:
                    cost[i, j] += 1e12
                else:
                    cost[i, j] += float(np.mean(np.abs(av - bv)) / max(1.0, float(np.max(np.abs(bv)))))
    rows, cols = linear_sum_assignment(cost)
    test_order = [int(rows[np.where(cols == j)[0][0]]) for j in range(k)]
    for field in fields:
        value = test.get(field)
        if isinstance(value, list) and len(value) == k:
            test[field] = [value[i] for i in test_order]
    return test, target


def compare_databases(
    db_test: Dict[str, Any],
    db_target: Dict[str, Any],
    tolerance: float = 1e-4,
    only_quantities: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Entry-by-entry comparison between generated DB and target DB."""
    quantities = sorted(list({k.split(":")[0] for k in list(db_test.keys()) + list(db_target.keys())}))
    # Diagnostics and historical annotations in the target JSON are not
    # production fit quantities.  An omitted filter means the complete
    # production DB surface, not every key family present in either file.
    requested = PRODUCTION_QUANTITIES if only_quantities is None else only_quantities
    allowed = {str(q).upper() for q in requested}
    quantities = [q for q in quantities if q.upper() in allowed]
    report = {}

    for q in quantities:
        test_keys = {k for k in db_test if k.startswith(f"{q}:")}
        target_keys = {k for k in db_target if k.startswith(f"{q}:")}

        common = sorted(test_keys & target_keys)
        missing = sorted(target_keys - test_keys)
        extra = sorted(test_keys - target_keys)

        equal_count = 0
        within_tol_count = 0
        diff_count = 0
        max_abs_diff = 0.0
        model_label_differences = 0

        for k in common:
            e_test = db_test[k]
            e_tgt = db_target[k]
            if isinstance(e_test, dict) and isinstance(e_tgt, dict):
                # ref_ll/ref_sd are restored writer fields, not fit parameters;
                # compare fitted distributions independently of these annotations.
                if e_test.get("model") != e_tgt.get("model"):
                    model_label_differences += 1
                e_test = {a: b for a, b in e_test.items() if a not in {"ref_ll", "ref_sd"}}
                e_tgt = {a: b for a, b in e_tgt.items() if a not in {"ref_ll", "ref_sd"}}
                e_test.pop("model", None)
                e_tgt.pop("model", None)
                e_test, e_tgt = _align_mixture_components(e_test, e_tgt)

            if e_test == e_tgt:
                equal_count += 1
                continue

            # Check numeric tolerance on floats
            is_close = True
            diff_entry = 0.0

            def _compare_vals(v1, v2):
                nonlocal is_close, diff_entry
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    d = abs(float(v1) - float(v2))
                    if d > diff_entry:
                        diff_entry = d
                    denom = max(abs(float(v2)), 1.0)
                    if d / denom > tolerance and d > tolerance:
                        is_close = False
                elif isinstance(v1, list) and isinstance(v2, list):
                    if len(v1) != len(v2):
                        is_close = False
                    else:
                        for a, b in zip(v1, v2):
                            _compare_vals(a, b)
                elif isinstance(v1, dict) and isinstance(v2, dict):
                    if set(v1.keys()) != set(v2.keys()):
                        is_close = False
                    else:
                        for sk in v1:
                            _compare_vals(v1[sk], v2[sk])
                else:
                    if v1 != v2:
                        is_close = False

            _compare_vals(e_test, e_tgt)
            if diff_entry > max_abs_diff:
                max_abs_diff = diff_entry

            if is_close:
                within_tol_count += 1
            else:
                diff_count += 1

        report[q] = {
            "target_keys": len(target_keys),
            "generated_keys": len(test_keys),
            "common_keys": len(common),
            "equal_exact": equal_count,
            "within_tolerance": within_tol_count,
            "different": diff_count,
            "model_label_differences": model_label_differences,
            "missing": len(missing),
            "extra": len(extra),
            "max_abs_diff": max_abs_diff,
        }

    return report




def load_container(path: Path | str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """(entries, _meta, _tables) of a constraints.json / .npz; a legacy DB has empty meta and tables."""
    path = Path(path)
    if str(path).endswith(".npz"):
        from jacobik.core.container import load_lossless_npz

        raw, _ = load_lossless_npz(path)
    else:
        raw = json.loads(path.read_text())
    meta = raw.pop("_meta", {}) if isinstance(raw.get("_meta"), dict) else {}
    tables = raw.pop("_tables", {}) if isinstance(raw.get("_tables"), dict) else {}
    return raw, meta, tables


def canonical(entry: Any) -> str:
    return json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)


def compare_exact(entries_a: Dict[str, Any], entries_b: Dict[str, Any],
                  tables_a: Optional[Dict[str, Any]] = None, tables_b: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Entry-for-entry equality of two DBs (canonical JSON of every entry and table); ``_meta`` is not compared."""
    ka, kb = set(entries_a), set(entries_b)
    differing = sorted(k for k in ka & kb if canonical(entries_a[k]) != canonical(entries_b[k]))
    tables_equal = canonical(tables_a or {}) == canonical(tables_b or {})
    return {"entries_a": len(ka), "entries_b": len(kb), "only_in_a": sorted(ka - kb)[:20], "only_in_b": sorted(kb - ka)[:20],
            "n_only_in_a": len(ka - kb), "n_only_in_b": len(kb - ka), "n_different": len(differing),
            "different_examples": differing[:20], "tables_equal": tables_equal,
            "identical": not (ka ^ kb) and not differing and tables_equal}
