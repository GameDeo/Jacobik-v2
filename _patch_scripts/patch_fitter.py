#!/usr/bin/env python3
"""
Patch src/jacobik/stats/fitter.py: Q2/Q3 move from VMM to GMM-on-cos(angle).

Root cause (confirmed by code inspection, approved by user earlier):
Q2 (D-M-D angle) and Q3 (donor approach angle) are both computed via
arccos() and are bounded [0 deg, 180 deg] -- they are NOT circular
quantities (0 deg != 360 deg is moot; they never even reach 360 deg).
Fitting them with a von Mises mixture assumes periodicity that doesn't
exist, and -- more concretely -- a VMM component whose mean sits near a
real physical boundary (e.g. CO's M-C-O approach angle piling up near
180 deg, a hard physical bound) gets pulled to systematically understate
peakedness, because EM has to "spend" some of the component's assumed-
symmetric spread on the non-existent far side of the boundary. Fitting
on cos(angle) instead respects the bound (cos maps [0,180] deg
monotonically onto [-1,1]) and lets a component legitimately sit sharply
peaked right at a boundary.
"""
import sys

PATH = "src/jacobik/stats/fitter.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

steps_done = []


def do_replace(label, old, new, count=1):
    global src
    n = src.count(old)
    if n == 0:
        print(f"FATAL: pattern for step [{label}] not found", file=sys.stderr)
        sys.exit(1)
    if n != count:
        print(f"FATAL: pattern for step [{label}] found {n} times, expected {count}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, count)
    steps_done.append(label)


OLD_DIST_TYPE = '''_DIST_TYPE: Dict[str, str] = {
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
}'''
NEW_DIST_TYPE = '''_DIST_TYPE: Dict[str, str] = {
    "Q1": "gmm",
    "Q2": "gmm_cos",  # D-M-D angle, bounded [0,180] deg, NOT circular -- see
    "Q3": "gmm_cos",  # fit_gmm_cosangle() docstring below for the rationale.
    "Q4": "gmm",
    "Q5": "vmm",  # torsion dihedral -- genuinely circular (0 == 360), VMM is correct here.
    "Q6": "halfgauss",
}

_K_MAX: Dict[str, int] = {
    "Q1": K_MAX_GMM,
    "Q2": K_MAX_VMM,  # component budget unrelated to model family; kept at 6
    "Q3": K_MAX_VMM,  # (Q2/Q3 legitimately need more components than Q1/Q4 --
    "Q4": K_MAX_GMM,  # e.g. Q2 for a homoleptic monodentate complex pools BOTH
    "Q5": K_MAX_VMM,  # cis-like and trans-like angles under one key -- see
    # ik/scorer.py Q2 section for how that pooling works at score time).
}'''
do_replace("F1-dist-type-and-kmax", OLD_DIST_TYPE, NEW_DIST_TYPE)

OLD_FIT_GMM_TAIL = '''    return {"type": "gmm", "n": n, "k": best_k, "means": means, "vars": vars_, "weights": weights}


# ---------------------------------------------------------------------------
# VMM (von Mises Mixture, EM)
# ---------------------------------------------------------------------------'''
NEW_FIT_GMM_TAIL = '''    return {"type": "gmm", "n": n, "k": best_k, "means": means, "vars": vars_, "weights": weights}


def fit_gmm_cosangle(values_deg: np.ndarray, k_max: int = K_MAX_GMM) -> Optional[Dict]:
    """
    Fit a GMM on cos(angle) for a bounded, non-circular angular quantity
    (Q2 D-M-D angle, Q3 donor approach angle -- both computed via arccos(),
    range [0,180] deg, never wrapping to 360). See the module-level note in
    _DIST_TYPE above for why VMM (which assumes periodicity) is the wrong
    model family for these.

    cos() maps [0,180] deg monotonically onto [-1,1], so a component fit
    here can legitimately sit sharply peaked right at a physical boundary
    (e.g. a linear M-C-O approach angle piling up near 180 deg) without
    the VMM's "symmetric spread on both sides of the mean" assumption
    forcing it to understate that peakedness. This is not a perfectly
    boundary-exact fix either (a 1-D Gaussian component near cos=-1 or
    cos=+1 still has unbounded support in principle), but it is a
    substantially better approximation than a periodic model for a
    quantity that never wraps, and it reuses the exact same well-tested
    GMM/BIC machinery as Q1/Q4 (fit_gmm above) rather than adding a new
    numerical method.

    Output means/vars/weights are stored in cos-space (dimensionless,
    means in [-1,1]) -- NOT degrees. ik/scorer.py evaluates these directly
    against cos(current_angle), so no unit conversion is needed at score
    time (simpler than the old VMM path, which had to convert degrees to
    radians for both fitting and scoring).

    Parameters
    ----------
    values_deg : raw observed angles in degrees (same array fit_vmm used
                 to receive -- no change needed upstream in key_builder.py
                 or the expand/aggregate pipeline).
    k_max      : as fit_gmm.

    Returns
    -------
    Same dict shape as fit_gmm(), but with "type": "gmm_cos" so
    ik/scorer.py (and any stats introspection / visualisation code) can
    tell it apart from a Q1/Q4-style plain-degree/Å GMM.
    """
    x = np.cos(np.deg2rad(values_deg))
    result = fit_gmm(x, k_max)
    if result is not None:
        result["type"] = "gmm_cos"
    return result


# ---------------------------------------------------------------------------
# VMM (von Mises Mixture, EM)
# ---------------------------------------------------------------------------'''
do_replace("F2-fit-gmm-cosangle", OLD_FIT_GMM_TAIL, NEW_FIT_GMM_TAIL)

OLD_DISPATCH = '''        try:
            if dist_type == "gmm":
                params = fit_gmm(values, k_max)
            elif dist_type == "vmm":
                params = fit_vmm(values, k_max)
            elif dist_type == "halfgauss":
                params = fit_halfgauss(values)
            else:
                params = None'''
NEW_DISPATCH = '''        try:
            if dist_type == "gmm":
                params = fit_gmm(values, k_max)
            elif dist_type == "gmm_cos":
                params = fit_gmm_cosangle(values, k_max)
            elif dist_type == "vmm":
                params = fit_vmm(values, k_max)
            elif dist_type == "halfgauss":
                params = fit_halfgauss(values)
            else:
                params = None'''
do_replace("F3-chunk-dispatch", OLD_DISPATCH, NEW_DISPATCH)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)
