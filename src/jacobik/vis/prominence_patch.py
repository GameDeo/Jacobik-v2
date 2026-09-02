"""
prominence_patch.py
====================
STANDALONE, THROW-AWAY UTILITY — not part of the jacobik package/build.

Purpose
-------
`fitter.py` fits GMM/VMM mixtures per Q1-Q6 key with BIC-selected K. Some of
those fitted components are real (a genuine secondary chemical mode) and
some are fitting artefacts: a handful of outlier/mislabelled observations
that BIC still rewards modelling as their own low-weight component. The
existing filter (`_Q1_PROMINENCE_FLOOR = 0.10` in engine.py) only screens
*mixture weight*, and only at coordination-sphere target-sampling time — it
never touches the score/gradient used during the actual 200-step IK solve
(scorer.py reads the raw, unfiltered stats DB). Mixture weight is also not
the same thing as *visual* prominence: a wide component can carry a weight
above a weight-only floor while its peak density is a fraction of the main
mode's, because peak height for a Gaussian component is w/sqrt(2*pi*var),
not just w. A component that's broad AND low-weight is exactly the shape
you'd expect from scattered outlier data rather than a real secondary mode.

This script:
  1. `diagnose()` — reports, for every GMM/VMM entry in a constraints.json,
     which components would survive a peak-HEIGHT-based prominence filter
     vs. the old weight-only filter, so you can see the disagreement before
     touching anything.
  2. `patch()` — writes a new constraints.json with low-peak-height
     components removed and remaining weights renormalised. Drop-in
     replacement: same "Q<n>:(key repr)" string-keyed JSON format that
     stats.py's load_db() already expects, so no loader changes needed.

This is explicitly a stopgap. The real fix is to move this pruning into
fitter.py itself (prune before the distribution is ever written to
constraints.json), so engine.py's target sampling and scorer.py's score
construction can't disagree about which modes are real. Do that on the next
full refit; this script exists only to unblock benchmarking before then.

Version history
----------------
1.0.0 : initial release. Peak-height prominence filter for GMM (Q1, Q4) and
        VMM (Q2, Q3, Q5) mixture components. Q6 (halfgauss) has no mixture
        structure and is untouched. CLI: diagnose / patch subcommands.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.special import i0e

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD: float = 0.10  # component kept if peak_height/max_peak_height >= this
GMM_QUANTITIES = ("Q1", "Q4")
VMM_QUANTITIES = ("Q2", "Q3", "Q5")
_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Peak-height computation
# ---------------------------------------------------------------------------


def gmm_peak_heights(weights: np.ndarray, vars_: np.ndarray) -> np.ndarray:
    """
    Peak density of each Gaussian component: w_k / sqrt(2*pi*var_k).

    This is what you'd actually see plotted, unlike raw weight, which
    ignores how spread-out (and therefore how visually prominent) a
    component is.
    """
    weights = np.asarray(weights, dtype=float)
    vars_ = np.clip(np.asarray(vars_, dtype=float), 1e-10, None)
    return weights / np.sqrt(_TWO_PI * vars_)


def vmm_peak_heights(weights: np.ndarray, kappas: np.ndarray) -> np.ndarray:
    """
    Peak density of each von Mises component at its mode:
        f_peak = exp(kappa) / (2*pi*I0(kappa)) = 1 / (2*pi*i0e(kappa))
    scaled by the component weight. Uses the exponentially-scaled Bessel
    function i0e for numerical stability at large kappa (same trick already
    used in scorer.py's VMM log-norm precomputation).
    """
    weights = np.asarray(weights, dtype=float)
    kappas = np.clip(np.asarray(kappas, dtype=float), 0.0, None)
    return weights / (_TWO_PI * np.clip(i0e(kappas), 1e-300, None))


def prominence_ratios(peak_heights: np.ndarray) -> np.ndarray:
    """Normalise peak heights to the tallest component (ratio in [0, 1])."""
    peak_heights = np.asarray(peak_heights, dtype=float)
    top = peak_heights.max() if peak_heights.size else 0.0
    if top <= 0:
        return np.ones_like(peak_heights)
    return peak_heights / top


def weight_ratios(weights: np.ndarray) -> np.ndarray:
    """The OLD filter's criterion, for comparison: weight / max(weight)."""
    weights = np.asarray(weights, dtype=float)
    top = weights.max() if weights.size else 0.0
    if top <= 0:
        return np.ones_like(weights)
    return weights / top


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------


def prune_gmm(
    means: List[float], vars_: List[float], weights: List[float], threshold: float
) -> Tuple[List[float], List[float], List[float], np.ndarray]:
    """
    Drop GMM components with peak height < threshold * max peak height,
    renormalise remaining weights. Always keeps at least the top component.

    Returns (means_kept, vars_kept, weights_kept, keep_mask) where keep_mask
    is over the ORIGINAL component order (for diagnostics/reporting).
    """
    w = np.asarray(weights, dtype=float)
    v = np.asarray(vars_, dtype=float)
    m = np.asarray(means, dtype=float)
    ratios = prominence_ratios(gmm_peak_heights(w, v))
    keep = ratios >= threshold
    if not keep.any():
        keep[int(np.argmax(ratios))] = True
    w_kept = w[keep]
    w_kept = w_kept / w_kept.sum()
    return m[keep].tolist(), v[keep].tolist(), w_kept.tolist(), keep


def prune_vmm(
    means: List[float], kappas: List[float], weights: List[float], threshold: float
) -> Tuple[List[float], List[float], List[float], np.ndarray]:
    """Same as prune_gmm, for von Mises mixtures (means in degrees, as stored)."""
    w = np.asarray(weights, dtype=float)
    k = np.asarray(kappas, dtype=float)
    m = np.asarray(means, dtype=float)
    ratios = prominence_ratios(vmm_peak_heights(w, k))
    keep = ratios >= threshold
    if not keep.any():
        keep[int(np.argmax(ratios))] = True
    w_kept = w[keep]
    w_kept = w_kept / w_kept.sum()
    return m[keep].tolist(), k[keep].tolist(), w_kept.tolist(), keep


# ---------------------------------------------------------------------------
# JSON I/O — mirrors stats.py's key format exactly (no loader changes needed)
# ---------------------------------------------------------------------------


def _split_json_key(json_key: str) -> Tuple[str, str]:
    colon = json_key.index(":")
    return json_key[:colon], json_key[colon + 1 :]


def load_raw(path: str | Path) -> Dict[str, dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_raw(data: Dict[str, dict], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)


# ---------------------------------------------------------------------------
# diagnose()
# ---------------------------------------------------------------------------


def diagnose(
    path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    quantities: Tuple[str, ...] = GMM_QUANTITIES + VMM_QUANTITIES,
    top_n: int = 40,
    verbose: bool = True,
) -> List[dict]:
    """
    Report every entry where the NEW peak-height filter would drop a
    component that the OLD weight-only filter kept (the "ghost mode" case),
    sorted by how prominent-by-weight-but-not-by-height the dropped
    component is (worst offenders first).

    Returns the list of finding dicts (also printed if verbose).
    """
    raw = load_raw(path)
    findings: List[dict] = []

    for json_key, entry in raw.items():
        quantity, key_repr = _split_json_key(json_key)
        if quantity not in quantities:
            continue
        etype = entry.get("type")
        if etype not in ("gmm", "vmm"):
            continue

        weights = entry.get("weights")
        if not weights or len(weights) < 2:
            continue  # nothing to prune with a single component

        if etype == "gmm":
            w_ratio = weight_ratios(np.asarray(weights))
            h_ratio = prominence_ratios(
                gmm_peak_heights(np.asarray(weights), np.asarray(entry["vars"]))
            )
        else:
            w_ratio = weight_ratios(np.asarray(weights))
            h_ratio = prominence_ratios(
                vmm_peak_heights(np.asarray(weights), np.asarray(entry["kappas"]))
            )

        old_keep = w_ratio >= threshold
        new_keep = h_ratio >= threshold
        disagreement = old_keep & ~new_keep  # old filter kept it, new filter would drop it
        if not disagreement.any():
            continue

        try:
            key_tuple = ast.literal_eval(key_repr)
        except (ValueError, SyntaxError):
            key_tuple = key_repr

        for idx in np.where(disagreement)[0]:
            findings.append(
                {
                    "quantity": quantity,
                    "key": key_tuple,
                    "component_index": int(idx),
                    "mean": entry["means"][idx],
                    "weight": float(weights[idx]),
                    "weight_ratio": float(w_ratio[idx]),
                    "height_ratio": float(h_ratio[idx]),
                    "n": entry.get("n"),
                }
            )

    findings.sort(key=lambda f: f["weight_ratio"] - f["height_ratio"], reverse=True)

    if verbose:
        print(f"constraints file: {path}")
        print(f"threshold        : {threshold}")
        print(f"quantities       : {quantities}")
        print(
            f"entries with a ghost-mode disagreement: {len(findings)}"
            f"  (old weight-floor keeps it, new peak-height-floor would drop it)"
        )
        print()
        header = f"{'Q':<3} {'mean':>9} {'w':>7} {'w_ratio':>8} {'h_ratio':>8} {'n':>7}  key"
        print(header)
        print("-" * len(header))
        for f in findings[:top_n]:
            print(
                f"{f['quantity']:<3} {f['mean']:>9.3f} {f['weight']:>7.3f} "
                f"{f['weight_ratio']:>8.3f} {f['height_ratio']:>8.3f} "
                f"{str(f['n']):>7}  {f['key']}"
            )
        if len(findings) > top_n:
            print(f"... ({len(findings) - top_n} more, not shown)")

    return findings


# ---------------------------------------------------------------------------
# patch()
# ---------------------------------------------------------------------------


def patch(
    in_path: str | Path,
    out_path: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    quantities: Tuple[str, ...] = GMM_QUANTITIES + VMM_QUANTITIES,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Write a patched constraints.json with low-peak-height components
    removed from every GMM/VMM entry, weights renormalised.

    Entries are otherwise untouched byte-for-byte (Q6 halfgauss entries,
    single-component entries, and any entry for a quantity not in
    `quantities` pass through unmodified). Patched entries get two extra
    metadata fields (`_prominence_patched`, `_original_k`) so patched
    entries stay identifiable — stats.py's FittedDist ignores unknown
    fields, so this is safe to load with no loader changes.

    Returns a summary dict; writes nothing if dry_run=True.
    """
    raw = load_raw(in_path)
    out = copy.deepcopy(raw)

    n_entries_seen = 0
    n_entries_patched = 0
    n_components_dropped = 0

    for json_key, entry in raw.items():
        quantity, _ = _split_json_key(json_key)
        if quantity not in quantities:
            continue
        etype = entry.get("type")
        if etype not in ("gmm", "vmm"):
            continue

        weights = entry.get("weights")
        if not weights or len(weights) < 2:
            continue
        n_entries_seen += 1

        if etype == "gmm":
            means, vars_, w_new, keep = prune_gmm(
                entry["means"], entry["vars"], weights, threshold
            )
            n_dropped = int((~keep).sum())
            if n_dropped == 0:
                continue
            out[json_key]["means"] = means
            out[json_key]["vars"] = vars_
            out[json_key]["weights"] = w_new
        else:
            means, kappas, w_new, keep = prune_vmm(
                entry["means"], entry["kappas"], weights, threshold
            )
            n_dropped = int((~keep).sum())
            if n_dropped == 0:
                continue
            out[json_key]["means"] = means
            out[json_key]["kappas"] = kappas
            out[json_key]["weights"] = w_new

        out[json_key]["k"] = len(w_new)
        out[json_key]["_prominence_patched"] = True
        out[json_key]["_original_k"] = len(weights)
        n_entries_patched += 1
        n_components_dropped += n_dropped

    summary = {
        "in_path": str(in_path),
        "out_path": str(out_path),
        "threshold": threshold,
        "quantities": quantities,
        "entries_scanned": n_entries_seen,
        "entries_patched": n_entries_patched,
        "components_dropped": n_components_dropped,
        "dry_run": dry_run,
    }

    if verbose:
        print(
            f"scanned {n_entries_seen} multi-component GMM/VMM entries in "
            f"{quantities}; patched {n_entries_patched} entries, dropped "
            f"{n_components_dropped} ghost components (threshold={threshold})."
        )

    if not dry_run:
        save_raw(out, out_path)
        if verbose:
            print(f"wrote patched constraints file to {out_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnose", help="report ghost-mode disagreements, no writes")
    d.add_argument("constraints_json")
    d.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    d.add_argument(
        "--quantities", nargs="+", default=list(GMM_QUANTITIES + VMM_QUANTITIES)
    )
    d.add_argument("--top-n", type=int, default=40)

    pa = sub.add_parser("patch", help="write a patched constraints.json")
    pa.add_argument("constraints_json")
    pa.add_argument("out_json")
    pa.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    pa.add_argument(
        "--quantities", nargs="+", default=list(GMM_QUANTITIES + VMM_QUANTITIES)
    )
    pa.add_argument("--dry-run", action="store_true")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "diagnose":
        diagnose(
            args.constraints_json,
            threshold=args.threshold,
            quantities=tuple(args.quantities),
            top_n=args.top_n,
        )
    elif args.cmd == "patch":
        patch(
            args.constraints_json,
            args.out_json,
            threshold=args.threshold,
            quantities=tuple(args.quantities),
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())