"""
fitter.py
=========
Step 5 -- Distribution fitter.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import multiprocessing
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.special import i0 as _i0
from sklearn.mixture import GaussianMixture

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_MIN: int = 30
K_MAX_GMM: int = 4
K_MAX_VMM: int = 6
VMM_MAX_ITER: int = 200
VMM_TOL: float = 1e-5

_DIST_TYPE: Dict[str, str] = {
    "Q1": "gmm",
    "Q2": "vmm",
    "Q3": "vmm",  # approach angle — angular quantity, must be VMM not GMM
    "Q4": "gmm",
    "Q5": "vmm",
    "Q6": "halfgauss",
}

_K_MAX: Dict[str, int] = {
    "Q1": K_MAX_GMM,
    "Q2": K_MAX_VMM,
    "Q3": K_MAX_VMM,  # matches VMM now
    "Q4": K_MAX_GMM,
    "Q5": K_MAX_VMM,
}

# ---------------------------------------------------------------------------
# GMM
# ---------------------------------------------------------------------------


def fit_gmm(values: np.ndarray, k_max: int = K_MAX_GMM) -> Optional[Dict]:
    """
    Fit GMM with BIC-selected K in 1..k_max.
    """
    n = len(values)
    if n < N_MIN:
        return None

    n_init = 1 if n > 500 else 3
    x = values.reshape(-1, 1)
    k_upper = min(k_max, n // max(N_MIN, 5))
    best_bic = np.inf
    best_gm = None
    best_k = 1

    for k in range(1, k_upper + 1):
        try:
            gm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                n_init=n_init,
                max_iter=200,
                random_state=42,
            )
            gm.fit(x)
            bic = gm.bic(x)
            if bic < best_bic:
                best_bic = bic
                best_gm = gm
                best_k = k
        except Exception as exc:
            log.debug("GMM k=%d failed: %s", k, exc)
            break

    if best_gm is None:
        return None

    order = np.argsort(best_gm.means_.ravel())
    means = best_gm.means_.ravel()[order].tolist()
    vars_ = best_gm.covariances_.ravel()[order].tolist()
    weights = best_gm.weights_[order].tolist()

    return {"type": "gmm", "n": n, "k": best_k, "means": means, "vars": vars_, "weights": weights}


# ---------------------------------------------------------------------------
# VMM (von Mises Mixture, EM)
# ---------------------------------------------------------------------------


def _kappa_from_rbar(R_bar: np.ndarray) -> np.ndarray:
    """
    Closed-form kappa estimate from mean resultant length R_bar in [0,1).
    Mardia & Jupp (2000) approximation, vectorised over an array of R_bar.
    """
    R = np.clip(R_bar, 0.0, 1.0 - 1e-6)
    return np.where(
        R < 0.53,
        R * (2.0 - R**2) / (1.0 - R**2),
        np.where(
            R < 0.85,
            -0.4 + 1.39 * R + 0.43 / (1.0 - R),
            1.0 / (R**3 - 4.0 * R**2 + 3.0 * R + 1e-8),
        ),
    )


def _vmm_bic(ll: float, n: int, k: int) -> float:
    return -2.0 * ll + (3 * k - 1) * math.log(n)


def _fit_vmm_k(
    x: np.ndarray,  # (n,) radians, pre-wrapped to [-pi, pi]
    sinx: np.ndarray,  # (n,) sin(x)  -- precomputed
    cosx: np.ndarray,  # (n,) cos(x)  -- precomputed
    k: int,
    n_init: int,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit VMM with exactly k components.  Returns (ll, means, kappas, weights).
    sinx/cosx are precomputed outside to avoid redundant computation across K.
    """
    n = len(x)
    best_ll = -np.inf
    best_out = None

    for _ in range(n_init):
        idx = np.random.choice(n, k, replace=(n < k))
        means = x[idx] + np.random.uniform(-0.3, 0.3, k)
        kappas = np.ones(k)
        weights = np.full(k, 1.0 / k)
        prev_ll = -np.inf

        for _iter in range(VMM_MAX_ITER):
            # E-step: log responsibilities
            # log_p[i,j] = kappa_j * cos(x_i - mu_j) - log(2pi*I0(kappa_j))
            log_norm = np.log(2.0 * np.pi * _i0(kappas))  # (k,)
            # cos(x - mu) = cos(x)cos(mu) + sin(x)sin(mu)
            cos_diff = (
                cosx[:, None] * np.cos(means)[None, :] + sinx[:, None] * np.sin(means)[None, :]
            )  # (n,k)
            log_p = kappas[None, :] * cos_diff - log_norm[None, :]
            log_r = log_p + np.log(weights + 1e-300)  # (n,k)

            # log-sum-exp for LL (computed here, not in a separate call)
            log_rmax = log_r.max(axis=1, keepdims=True)  # (n,1)
            lse = log_rmax.ravel() + np.log(np.exp(log_r - log_rmax).sum(axis=1))  # (n,)
            ll = float(lse.sum())

            # Responsibilities
            r = np.exp(log_r - log_rmax)
            r /= r.sum(axis=1, keepdims=True) + 1e-300  # (n,k)

            # M-step
            r_sum = r.sum(axis=0) + 1e-300  # (k,)
            weights = r_sum / r_sum.sum()

            sin_sum = (r * sinx[:, None]).sum(axis=0)  # (k,)
            cos_sum = (r * cosx[:, None]).sum(axis=0)  # (k,)
            means = np.arctan2(sin_sum, cos_sum)
            R_bar = np.sqrt(sin_sum**2 + cos_sum**2) / r_sum
            kappas = np.clip(_kappa_from_rbar(R_bar), 0.01, 500.0)

            if abs(ll - prev_ll) < VMM_TOL:
                break
            prev_ll = ll

        if prev_ll > best_ll:
            best_ll = prev_ll
            best_out = (means.copy(), kappas.copy(), weights.copy())

    if best_out is None:
        raise RuntimeError(f"VMM k={k} EM failed")
    return best_ll, best_out[0], best_out[1], best_out[2]


def fit_vmm(values_deg: np.ndarray, k_max: int = K_MAX_VMM) -> Optional[Dict]:
    """Fit VMM with BIC-selected K. Input in degrees, output means in degrees."""
    n = len(values_deg)
    if n < N_MIN:
        return None

    n_init = 1 if n > 500 else 3
    x = np.deg2rad(values_deg)
    x = (x + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    sinx = np.sin(x)  # precompute once for all K
    cosx = np.cos(x)

    k_upper = min(k_max, n // max(N_MIN, 5))
    best_bic = np.inf
    best_result = None
    best_k = 1
    prev_bic = np.inf  # for early stopping

    for k in range(1, k_upper + 1):
        try:
            ll, means, kappas, weights = _fit_vmm_k(x, sinx, cosx, k, n_init)
            bic = _vmm_bic(ll, n, k)
            if bic < best_bic:
                best_bic = bic
                best_result = (means, kappas, weights)
                best_k = k
            # Early stopping: if BIC increased vs previous K, more components
            # won't help (BIC penalises complexity monotonically beyond the
            # true K for well-separated mixtures).
            if k > 1 and bic > prev_bic:
                break
            prev_bic = bic
        except Exception as exc:
            log.debug("VMM k=%d failed: %s", k, exc)
            break

    if best_result is None:
        return None

    means, kappas, weights = best_result
    order = np.argsort(means)
    means_d = np.rad2deg(means[order]).tolist()
    kappas = kappas[order].tolist()
    weights = weights[order].tolist()

    return {
        "type": "vmm",
        "n": n,
        "k": best_k,
        "means": means_d,
        "kappas": kappas,
        "weights": weights,
    }


# ---------------------------------------------------------------------------
# Half-Gaussian (Q6)
# ---------------------------------------------------------------------------


def fit_halfgauss(values: np.ndarray) -> Optional[Dict]:
    """
    Fit half-Gaussian clash wall to a non-bonded distance distribution.
    """
    n = len(values)
    if n < N_MIN:
        return None

    mu = float(np.percentile(values, 5))

    left = values[values <= mu]
    if len(left) < 3:
        # Fallback: use 10th–90th percentile spread / 4 as a width proxy
        sigma = float((np.percentile(values, 90) - np.percentile(values, 10)) / 4.0)
    else:
        sigma = float(np.std(np.concatenate([left, 2.0 * mu - left])))

    sigma = max(sigma, 0.1)  # minimum 0.1 Å — wall must have finite width
    return {"type": "halfgauss", "n": n, "k": 1, "mu": mu, "sigma": sigma}


# ---------------------------------------------------------------------------
# Worker initialiser (imports happen once per process, not once per task)
# ---------------------------------------------------------------------------

# Module-level globals set by pool initialiser -- avoids re-importing in child
_POOL_DIST_TYPE: Dict[str, str] = {}
_POOL_K_MAX: Dict[str, int] = {}


def _pool_init(dist_type: Dict, k_max: Dict) -> None:
    """Called once per worker process to set up shared state."""
    global _POOL_DIST_TYPE, _POOL_K_MAX
    _POOL_DIST_TYPE = dist_type
    _POOL_K_MAX = k_max
    # Warm up numpy RNG so first call isn't slower
    np.random.seed()


def _fit_chunk(chunk: List[Tuple[str, str, np.ndarray]]) -> List[Tuple[str, str, Optional[Dict]]]:
    """
    Fit a chunk of (quantity, key_str, values) triples in one worker call.
    Processing a chunk rather than one item per call amortises IPC overhead.
    """
    results = []
    for quantity, key_str, values in chunk:
        dist_type = _POOL_DIST_TYPE.get(quantity)
        k_max = _POOL_K_MAX.get(quantity, K_MAX_GMM)
        try:
            if dist_type == "gmm":
                params = fit_gmm(values, k_max)
            elif dist_type == "vmm":
                params = fit_vmm(values, k_max)
            elif dist_type == "halfgauss":
                params = fit_halfgauss(values)
            else:
                params = None
        except Exception as exc:
            log.debug("Fit failed %s %r: %s", quantity, key_str, exc)
            params = None
        results.append((quantity, key_str, params))
    return results


# ---------------------------------------------------------------------------
# Data loading  (pandas groupby -- C-level, much faster than Python loop)
# ---------------------------------------------------------------------------


def _load_observations(
    expanded_dir: Path,
    quantity: str,
    n_min: int,
) -> Dict[str, np.ndarray]:
    """
    Load the aggregated expanded_<Q>.parquet for one quantity via pandas,
    group by key, and filter to n >= n_min.
    """
    path = expanded_dir / f"expanded_{quantity}.parquet"
    if not path.exists():
        log.warning("%s: expanded file not found: %s", quantity, path)
        return {}

    df = pd.read_parquet(str(path), columns=["key", "value"])
    grouped = df.groupby("key", sort=False)["value"]
    counts = grouped.count()
    valid = counts[counts >= n_min].index

    result: Dict[str, np.ndarray] = {}
    for key_str in valid:
        result[key_str] = grouped.get_group(key_str).to_numpy(dtype=np.float64)

    log.info(
        "%s: %d keys with n>=%d  (total keys seen: %d)", quantity, len(result), n_min, len(counts)
    )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_all(
    expanded_dir: str | Path,
    out_path: str | Path,
    quantities: Optional[List[str]] = None,
    n_min: int = N_MIN,
    n_workers: Optional[int] = None,
    chunk_size: int = 64,
) -> Dict[str, Dict]:
    """
    Fit distributions for all quantities and write the constraint database.


    Parameters
    ----------
    expanded_dir : directory with expanded_<Q>.parquet files
    out_path     : path for constraints.json
    quantities   : quantities to fit (default: Q1-Q6)
    n_min        : minimum observations per key
    n_workers    : parallel workers (default: cpu_count)
    chunk_size   : keys per worker call (amortises IPC overhead)

    Returns
    -------
    dict mapping quantity -> {fitted, skipped, elapsed_s}

    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    expanded_dir = Path(expanded_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if quantities is None:
        quantities = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]

    if n_workers is None:
        n_workers = multiprocessing.cpu_count()

    database: Dict[str, Dict] = {}
    summary: Dict[str, Dict] = {}

    for q in quantities:
        t0 = time.perf_counter()
        log.info("=== Fitting %s ===", q)

        obs = _load_observations(expanded_dir, q, n_min)
        if not obs:
            log.warning("%s: no observations above n_min=%d", q, n_min)
            summary[q] = {"fitted": 0, "skipped": 0, "elapsed_s": 0.0}
            continue

        # Build work list and split into chunks
        items = [(q, k, v) for k, v in obs.items()]
        chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

        fitted = 0
        skipped = 0

        n_chunks = len(chunks)
        n_keys = len(items)
        done_chunks = 0
        report_every = max(1, n_chunks // 20)  # progress every ~5%
        log.info("%s: %d keys → %d chunks  (%d workers)", q, n_keys, n_chunks, n_workers)

        # Submit chunks lazily: keep at most (n_workers * 2) in-flight at
        # once.  Submitting all chunks upfront fills the IPC pipe buffer on
        # Windows and deadlocks when total data size is large (e.g. Q6).
        max_inflight = n_workers * 2
        chunk_iter = iter(chunks)
        inflight: dict = {}

        def _submit_next() -> bool:
            """Submit one more chunk if available. Returns True if submitted."""
            try:
                c = next(chunk_iter)
                inflight[pool.submit(_fit_chunk, c)] = c
                return True
            except StopIteration:
                return False

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_pool_init,
            initargs=(_DIST_TYPE, _K_MAX),
        ) as pool:
            # Seed the pool with initial work
            for _ in range(min(max_inflight, n_chunks)):
                _submit_next()

            while inflight:
                for future in as_completed(list(inflight)):
                    chunk_for_future = inflight.pop(future)
                    # Immediately submit next chunk to keep workers busy
                    _submit_next()

                    try:
                        chunk_results = future.result()
                    except Exception as exc:
                        log.warning("Chunk worker error: %s", exc)
                        skipped += len(chunk_for_future)
                        done_chunks += 1
                        continue

                    for qty, key_str, params in chunk_results:
                        if params is None:
                            skipped += 1
                        else:
                            database[f"{qty}:{key_str}"] = params
                            fitted += 1

                    done_chunks += 1
                    if done_chunks % report_every == 0 or done_chunks == n_chunks:
                        pct = 100.0 * done_chunks / n_chunks
                        elapsed_so_far = time.perf_counter() - t0
                        log.info(
                            "%s: %d/%d chunks (%.0f%%)  fitted=%d  skipped=%d  %.1fs",
                            q,
                            done_chunks,
                            n_chunks,
                            pct,
                            fitted,
                            skipped,
                            elapsed_so_far,
                        )
                    break  # re-enter as_completed with updated inflight

        elapsed = time.perf_counter() - t0
        log.info("%s: fitted=%d  skipped=%d  %.1fs", q, fitted, skipped, elapsed)
        summary[q] = {"fitted": fitted, "skipped": skipped, "elapsed_s": elapsed}

    log.info("Writing %s  (%d entries) ...", out_path, len(database))
    t_write = time.perf_counter()
    serialised = json.dumps(database, separators=(",", ":"))
    with open(out_path, "w") as fh:
        fh.write(serialised)
    del serialised  # release memory immediately
    log.info("Write complete in %.1fs", time.perf_counter() - t_write)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit distributions to expanded observation parquets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--expanded-dir", required=True)
    p.add_argument("--out-path", required=True)
    p.add_argument("--quantities", nargs="+", default=None)
    p.add_argument("--n-min", type=int, default=N_MIN)
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--chunk-size", type=int, default=64)
    from jacobik.stats.cli_utils import add_logging_args

    add_logging_args(p)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    from jacobik.stats.cli_utils import setup_logging

    setup_logging(args)
    summary = fit_all(
        expanded_dir=args.expanded_dir,
        out_path=args.out_path,
        quantities=args.quantities,
        n_min=args.n_min,
        n_workers=args.n_workers,
        chunk_size=args.chunk_size,
    )
    print(f"\n{'Quantity':<8}  {'Fitted':>8}  {'Skipped':>8}  {'Time s':>7}")
    print("-" * 40)
    for q, s in summary.items():
        print(f"  {q:<6}  {s['fitted']:>8,}  {s['skipped']:>8,}  {s['elapsed_s']:>7.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
