"""
stats.py
========
Constraints database loader and key-resolution interface for the IK engine.

Version history
----------------
1.0.0 : initial release
1.1.0 : _fallback_q6 no longer uses raw covalent-radii-sum for metal-
        involving pairs. The constraint DB has no fitted Q6 statistics for
        ANY (metal, element) pair (Q6 was only ever fit on organic
        element pairs), so every metal-ligand-atom contact was hitting this
        hard fallback. Covalent radii describe bonding distance, so the old
        fallback only penalised a non-donor atom once it was nearly close
        enough to covalently bond to the metal -- confirmed on EDTA: a
        backbone carbon at 1.71 A from Co (inside typical covalent contact)
        cost only -0.32 nats out of a ~-13 total score. Metal-involving
        pairs now use a VDW-like scale factor (1.18x covalent-sum, vs.
        covalent-sum alone -- VDW radii are the standard tool for steric-
        clash walls, not covalent radii) and a tighter sigma (0.12 vs.
        0.25), so a severe violation isn't washed out. Organic-pair
        fallback behaviour is unchanged.
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from importlib.resources.abc import Traversable
except ImportError:
    from importlib.abc import Traversable

from jacobik.topology.donor_hierarchy import (
    ELEMENT_ANY,
    full_path,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METAL_ANY = "metal_any"
GEOM_ANY = "geom_any"
DIST_ANY = "dist_any"  # graph-distance catch-all

_METAL_PERIOD_GROUP: Dict[str, str] = {
    **{s: "period4_3d" for s in ("Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn")},
    **{s: "period5_3d" for s in ("Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd")},
    **{s: "period6_3d" for s in ("Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg")},
    **{
        s: "lanthanide"
        for s in (
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
        )
    },
    **{s: "actinide" for s in ("Ac", "Th", "Pa", "U", "Np", "Pu")},
}

_GEOM_TO_CN: Dict[str, str] = {
    "linear": "CN2",
    "t_shaped": "CN3",
    "trigonal": "CN3",
    "square_plan": "CN4",
    "see_saw": "CN4",
    "tetrahedral": "CN4",
    "square_pyr": "CN5",
    "trig_bipyr": "CN5",
    "octahedral": "CN6",
    "trig_pris": "CN6",
    "pent_bipyr": "CN7",
    "square_anti": "CN8",
}

#: Graph-distance coarsening: specific → same_or_different → dist_any
_DIST_TO_SAME_OR_DIFF: Dict[str, str] = {
    "chelate_short": "same_ligand",
    "chelate_long": "same_ligand",
    "same_remote": "same_ligand",
    "different_ligand": "different_ligand",
}

#: Q5 atom-type coarsening: type → element → heavy → any
_Q5_TYPE_TO_ELEM: Dict[str, str] = {
    "C_ar": "C",
    "C_sp2": "C",
    "C_sp3": "C",
    "N_ar": "N",
    "N_sp2": "N",
    "N_sp3": "N",
    "O_sp2": "O",
    "O_sp3": "O",
    "S_any": "S",
    "P_any": "P",
    "X_any": "X",
}

#: Covalent radii in Å for Q6 hard fallback (Alvarez 2008 values)
_COVALENT_RADIUS: Dict[str, float] = {
    "H": 0.31,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "As": 1.19,
    "Se": 1.20,
    "Br": 1.20,
    "Te": 1.38,
    "I": 1.39,
    # Transition metals (typical values)
    "Sc": 1.70,
    "Ti": 1.60,
    "V": 1.53,
    "Cr": 1.39,
    "Mn": 1.61,
    "Fe": 1.52,
    "Co": 1.50,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Y": 1.90,
    "Zr": 1.75,
    "Nb": 1.64,
    "Mo": 1.54,
    "Tc": 1.47,
    "Ru": 1.46,
    "Rh": 1.42,
    "Pd": 1.39,
    "Ag": 1.45,
    "Cd": 1.44,
    "Hf": 1.75,
    "Ta": 1.70,
    "W": 1.62,
    "Re": 1.51,
    "Os": 1.44,
    "Ir": 1.41,
    "Pt": 1.36,
    "Au": 1.36,
    "Hg": 1.32,
    "La": 2.07,
    "Ce": 2.04,
    "Pr": 2.03,
    "Nd": 2.01,
    "Sm": 1.98,
    "Eu": 1.98,
    "Gd": 1.96,
    "Tb": 1.94,
    "Dy": 1.92,
    "Ho": 1.92,
    "Er": 1.89,
    "Tm": 1.90,
    "Yb": 1.87,
    "Lu": 1.87,
}
_DEFAULT_RADIUS = 1.50  # fallback for unknown elements

# Symbols covered by the "Transition metals" section of _COVALENT_RADIUS
# above. Used by _fallback_q6 to detect metal-involving pairs and apply the
# VDW-like scale/tighter sigma instead of raw covalent-radii-sum.
_TRANSITION_METALS = frozenset(
    {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
        "Er", "Tm", "Yb", "Lu",
    }
)
#: VDW radii run ~1.15-1.2x covalent radii; applied to covalent-radii-sum
#: for metal-involving Q6 fallback pairs only (see _fallback_q6 1.1.0).
_Q6_METAL_VDW_SCALE: float = 1.18
#: Tighter than the generic _Q6_DEFAULT_SIGMA (0.25) so a severe
#: metal-proximity violation isn't washed out by soft-min aggregation.
_Q6_METAL_SIGMA: float = 0.12


_Q6_DEFAULT_SIGMA = 0.25  # Å — width of hard-fallback half-Gaussian


# ---------------------------------------------------------------------------
# FittedDist
# ---------------------------------------------------------------------------


@dataclass
class FittedDist:
    """
    One fitted distribution entry from constraints.json.
    """

    type: str
    n: int
    level: int
    means: Optional[List[float]] = None
    vars: Optional[List[float]] = None
    kappas: Optional[List[float]] = None
    weights: Optional[List[float]] = None
    mu: Optional[float] = None
    sigma: Optional[float] = None


# ---------------------------------------------------------------------------
# ConstraintDB
# ---------------------------------------------------------------------------


class ConstraintDB:
    """
    Parsed in-memory representation of constraints.json.
    """

    def __init__(self, data: Dict[str, Dict[tuple, FittedDist]]) -> None:
        self.data = data

    def __contains__(self, item: Tuple[str, tuple]) -> bool:
        """(quantity, key_tuple) in db"""
        q, k = item
        return q in self.data and k in self.data[q]

    def get(self, quantity: str, key: tuple) -> Optional[FittedDist]:
        """Return FittedDist or None if not found."""
        return self.data.get(quantity, {}).get(key)

    def sizes(self) -> Dict[str, int]:
        """Return number of fitted keys per quantity."""
        return {q: len(v) for q, v in self.data.items()}


def load_db(path: str | Path | Traversable, n_min: int = 1) -> ConstraintDB:
    """
    Load constraints.json from disk and return a ConstraintDB.


    Parameters
    ----------
    path  : path to constraints.json or a Traversable resource
    n_min : only include entries with n ≥ n_min (default 1 = keep all)

    Returns
    -------
    ConstraintDB

    """
    if isinstance(path, (str, Path)):
        path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        raw: dict = json.load(fh)

    data: Dict[str, Dict[tuple, FittedDist]] = {q: {} for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")}
    n_loaded = 0
    n_skipped = 0

    for json_key, entry in raw.items():
        # Split "Q1:('Fe', ...)" into quantity and key_repr
        colon = json_key.index(":")
        quantity = json_key[:colon]
        key_repr = json_key[colon + 1 :]

        if quantity not in data:
            log.debug("Unknown quantity in constraints.json: %s", quantity)
            continue

        n_obs = int(entry.get("n", 0))
        if n_obs < n_min:
            n_skipped += 1
            continue

        try:
            key_tuple = ast.literal_eval(key_repr)
        except (ValueError, SyntaxError) as exc:
            log.warning("Failed to parse key %r: %s", key_repr, exc)
            continue

        if not isinstance(key_tuple, tuple):
            # Single-element tuples with a trailing comma parse correctly;
            # non-tuples indicate a format error.
            key_tuple = (key_tuple,)

        # Determine fallback level from the number of elements in the key
        # relative to the L0 key for this quantity.  The fitter doesn't
        # store the level explicitly in all versions — derive it from context.
        # Use the 'level' field if present, else 0.
        level = int(entry.get("level", 0))

        fd = FittedDist(
            type=entry["type"],
            n=n_obs,
            level=level,
            means=entry.get("means"),
            vars=entry.get("vars"),
            kappas=entry.get("kappas"),
            weights=entry.get("weights"),
            mu=entry.get("mu"),
            sigma=entry.get("sigma"),
        )
        data[quantity][key_tuple] = fd
        n_loaded += 1

    log.info(
        "Loaded %d distributions from %s (%d skipped, n < %d)",
        n_loaded,
        path,
        n_skipped,
        n_min,
    )
    return ConstraintDB(data)


# ---------------------------------------------------------------------------
# Internal coarsening helpers
# ---------------------------------------------------------------------------


def _metal_coarsen(metal: str) -> List[str]:
    """Ordered list: [metal, period_group, metal_any]."""
    pg = _METAL_PERIOD_GROUP.get(metal)
    if pg:
        return [metal, pg, METAL_ANY]
    return [metal, METAL_ANY]


def _donor_coarsen(subclass: str) -> List[str]:
    """
    Ordered full coarsening path for a donor subclass.
    Returns [subclass, bucket_or_family, ..., element_any].
    Mirrors donor_hierarchy.full_path().
    """
    try:
        return full_path(subclass)
    except KeyError:
        # Unknown subclass — fall back to element_any immediately
        log.debug("Unknown donor subclass in coarsening: %r", subclass)
        return [subclass, ELEMENT_ANY]


def _geom_coarsen(geom: str) -> List[str]:
    """Ordered: [geom, CN_label, geom_any]."""
    cn = _GEOM_TO_CN.get(geom)
    if cn:
        return [geom, cn, GEOM_ANY]
    return [geom, GEOM_ANY]


def _dist_coarsen(dist_class: str) -> List[str]:
    """Ordered: [dist_class, same_or_diff, dist_any]."""
    sod = _DIST_TO_SAME_OR_DIFF.get(dist_class, "same_ligand")
    return [dist_class, sod, DIST_ANY]


def _q5_type_coarsen(t: str) -> List[str]:
    """Ordered: [type, element, 'heavy', 'any']."""
    elem = _Q5_TYPE_TO_ELEM.get(t, None)
    if elem:
        return [t, elem, "heavy", "any"]
    # Already a short element symbol or 'other'
    return [t, "heavy", "any"]


def _is_empty_dist(fd: "FittedDist") -> bool:
    """
    True if a FittedDist carries no usable parameters and must not be returned
    by the resolver, regardless of its observation count ``n``.
    """
    t = fd.type
    if t in ("gmm", "vmm"):
        return (not fd.means) or (not fd.weights)
    if t == "halfgauss":
        return fd.mu is None or fd.sigma is None
    return True


# ---------------------------------------------------------------------------
# L0 key arities — used to validate inputs to dist()
# ---------------------------------------------------------------------------

#: Expected tuple length of an L0 (most-specific) key for each quantity.
#: Keys shorter than this are coarsened / fallback-level keys that were
#: stored in the database by the fitter but are NOT valid inputs to dist().
_L0_ARITY: Dict[str, int] = {
    "Q1": 6,  # (metal, spin, charge, donor_subclass, coord_geom, trans_partner)
    "Q2": 5,  # (coord_geom, graph_dist, metal, donor_A_subclass, donor_B_subclass)
    "Q3": 3,  # (donor_subclass, metal, coord_geom)
    "Q4": 2,  # (donor_subclass, coord_geom)
    "Q5": 5,  # (type_j, type_k, type_i, type_l, ring_ctx)
    "Q6": 2,  # (elem_A, elem_B)
}


# ---------------------------------------------------------------------------
# StatsLookup
# ---------------------------------------------------------------------------


class StatsLookup:
    """
    The stats module interface consumed by Scorer.


    Parameters
    ----------
    db    : ConstraintDB loaded from constraints.json
    n_min : minimum observations required to use a distribution (default 30)

    Usage
    -----
        stats = StatsLookup(db, n_min=30)
        dist  = stats.dist("Q1", ("Fe", 1, 0, "N_pyridine", "octahedral", "N_pyridine"))

    """

    def __init__(self, db: ConstraintDB, n_min: int = 30) -> None:
        self._db = db
        self._n_min = n_min
        # Diagnostic trace state (populated by _resolve, read by dist_trace).
        self._last_resolved_key = None
        self._last_candidates = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dist(self, quantity: str, key: tuple) -> FittedDist:
        """
        Return the most-specific fitted distribution for (quantity, key).


        Parameters
        ----------
        quantity : "Q1" | "Q2" | "Q3" | "Q4" | "Q5" | "Q6"
        key      : L0 key tuple for this quantity (as produced by key_builder)

        Returns
        -------
        FittedDist

        """
        _dispatch = {
            "Q1": self._fallback_q1,
            "Q2": self._fallback_q2,
            "Q3": self._fallback_q3,
            "Q4": self._fallback_q4,
            "Q5": self._fallback_q5,
            "Q6": self._fallback_q6,
        }
        fn = _dispatch.get(quantity)
        if fn is None:
            raise ValueError(f"Unknown quantity: {quantity!r}")

        expected = _L0_ARITY.get(quantity, 0)
        if len(key) != expected:
            raise ValueError(
                f"stats.dist({quantity!r}, ...) requires an L0 key of length "
                f"{expected}, got length {len(key)}: {key!r}.  "
                f"Only full L0 keys are valid inputs; coarsened / fallback-level "
                f"keys stored in the database must not be passed directly to dist()."
            )

        # Reset the diagnostic record (populated by _resolve).  Read back by
        # dist_trace(); harmless overhead for normal dist() calls.
        self._last_resolved_key = None
        self._last_candidates = None
        return fn(key)

    def dist_trace(self, quantity: str, key: tuple):
        """
        Diagnostic variant of dist().  Returns the resolved FittedDist along
        with the candidate key it resolved to and the full ordered candidate
        list that was tried.

        Returns
        -------
        (fd, resolved_key, candidates) where:
          fd            : the FittedDist dist() would return
          resolved_key  : the candidate tuple that produced fd (None if the
                          hard/empty fallback path was taken)
          candidates    : the ordered list of candidate keys that were tried

        Intended for notebooks / debugging only — it shows exactly which
        fallback level a given L0 key landed on, so you can see whether a
        distribution came from a specific fit or a coarse global fallback.
        """
        fd = self.dist(quantity, key)
        return fd, self._last_resolved_key, self._last_candidates

    # See exp.md for detailed rationale

    def _resolve(
        self,
        quantity: str,
        candidates: List[tuple],
        hard_fallback: Optional[FittedDist] = None,
    ) -> FittedDist:
        """
        Try each candidate key in order; return first with n ≥ n_min.
        If nothing meets n_min, return the last candidate unconditionally
        (the global fallback row).  If even the global fallback is absent,
        raise RuntimeError.
        """
        q_data = self._db.data.get(quantity, {})
        last_found: Optional[FittedDist] = None
        last_level: int = 0
        last_candidate: Optional[tuple] = None

        # Record candidate list for dist_trace() diagnostics.
        self._last_candidates = list(candidates)

        for level_idx, candidate in enumerate(candidates):
            fd = q_data.get(candidate)
            if fd is None:
                continue
            # See exp.md for detailed rationale
            if _is_empty_dist(fd):
                log.debug(
                    "%s: candidate %s has n=%d but empty parameters; skipping.",
                    quantity,
                    candidate,
                    fd.n,
                )
                continue
            # Overwrite level with actual resolution depth (candidate index),
            # regardless of what the fitter stored in the JSON.
            fd.level = level_idx
            last_found = fd
            last_level = level_idx
            last_candidate = candidate
            if fd.n >= self._n_min:
                self._last_resolved_key = candidate
                return fd

        # Nothing met n_min — return whatever we found last (coarsest populated)
        if last_found is not None:
            log.debug(
                "%s: all candidates below n_min=%d; using n=%d (level %d)",
                quantity,
                self._n_min,
                last_found.n,
                last_level,
            )
            self._last_resolved_key = last_candidate
            return last_found

        # Hard fallback (used for Q6 covalent-radius approximation)
        if hard_fallback is not None:
            self._last_resolved_key = None
            return hard_fallback

        raise RuntimeError(
            f"No distribution found for {quantity} after full fallback. "
            f"Tried: {candidates[:3]}... "
            f"Database may be missing the global fallback node."
        )

    # See exp.md for detailed rationale

    def _fallback_q1(self, key: tuple) -> FittedDist:
        metal, spin, charge, subclass, geom, trans = key
        donor_path = _donor_coarsen(subclass)  # [subclass, bucket?, family, element_any]
        metal_path = _metal_coarsen(metal)  # [metal, period_group?, metal_any]

        candidates: List[tuple] = []

        # L0 — only when trans != "none" (expander rule)
        if trans != "none":
            candidates.append((metal, spin, charge, subclass, geom, trans))

        # L1: drop trans
        candidates.append((metal, spin, charge, subclass, geom))

        # L2: drop geom
        candidates.append((metal, spin, charge, subclass))

        # L3–L4: coarsen donor (skip duplicate bucket==family for shallow families)
        seen_donor = {subclass}
        for d in donor_path:
            if d == ELEMENT_ANY:
                break
            if d not in seen_donor:
                candidates.append((metal, spin, charge, d))
                seen_donor.add(d)

        # The family label is the last non-element_any in donor_path
        donor_family = donor_path[-2] if len(donor_path) >= 2 else donor_path[-1]
        if donor_family == ELEMENT_ANY:
            donor_family = subclass  # pathological: subclass is element_any

        # L5: drop spin
        candidates.append((metal, charge, donor_family))

        # L6: drop charge
        candidates.append((metal, donor_family))

        # L7–L8: coarsen metal
        seen_metal = {metal}
        for m in metal_path[1:]:  # skip metal itself (already added)
            if m not in seen_metal:
                candidates.append((m, donor_family))
                seen_metal.add(m)

        # L9: drop metal
        candidates.append((donor_family,))

        # L10: global fallback
        candidates.append((ELEMENT_ANY,))

        return self._resolve("Q1", candidates)

    # See exp.md for detailed rationale

    def _fallback_q2(self, key: tuple) -> FittedDist:
        geom, dist_class, metal, sub_A, sub_B = key

        donor_path_A = _donor_coarsen(sub_A)
        donor_path_B = _donor_coarsen(sub_B)
        geom_path = _geom_coarsen(geom)  # [geom, CN_label?, geom_any]

        # Extract levels of A and B in lockstep — use the shorter path
        def _level(path: List[str], lvl: int) -> str:
            return path[min(lvl, len(path) - 1)]

        fam_A = donor_path_A[-2] if len(donor_path_A) >= 2 else donor_path_A[-1]
        fam_B = donor_path_B[-2] if len(donor_path_B) >= 2 else donor_path_B[-1]
        if fam_A == ELEMENT_ANY:
            fam_A = sub_A
        if fam_B == ELEMENT_ANY:
            fam_B = sub_B

        bkt_A = donor_path_A[1] if len(donor_path_A) >= 3 else fam_A
        bkt_B = donor_path_B[1] if len(donor_path_B) >= 3 else fam_B

        cn_label = geom_path[1] if len(geom_path) >= 3 else GEOM_ANY

        candidates: List[tuple] = [
            # L0
            (geom, dist_class, metal, sub_A, sub_B),
            # L1: coarsen donors to bucket
            (geom, dist_class, metal, bkt_A, bkt_B),
            # L2: coarsen donors to family
            (geom, dist_class, metal, fam_A, fam_B),
            # L3: drop metal
            (geom, dist_class, fam_A, fam_B),
            # L4: drop donors
            (geom, dist_class),
            # L5: coarsen geom to CN_only
            (cn_label, dist_class),
            # L6: drop dist
            (geom,),
            # L7: CN_only only
            (cn_label,),
            # L8: global fallback
            (GEOM_ANY,),
        ]
        return self._resolve("Q2", candidates)

    # See exp.md for detailed rationale

    def _fallback_q3(self, key: tuple) -> FittedDist:
        subclass, metal, geom = key
        donor_path = _donor_coarsen(subclass)
        metal_path = _metal_coarsen(metal)

        candidates: List[tuple] = [
            (subclass, metal, geom),  # L0
            (subclass, metal),  # L1
        ]
        for m in metal_path[1:]:
            candidates.append((subclass, m))  # L2, L3

        candidates.append((subclass,))  # L4

        # L5–L6: coarsen donor (skip subclass itself, already added)
        seen = {subclass}
        for d in donor_path:
            if d not in seen:
                candidates.append((d,))
                seen.add(d)

        # L7: global fallback
        if (ELEMENT_ANY,) not in candidates:
            candidates.append((ELEMENT_ANY,))

        return self._resolve("Q3", candidates)

    # See exp.md for detailed rationale

    def _fallback_q4(self, key: tuple) -> FittedDist:
        subclass, geom = key
        donor_path = _donor_coarsen(subclass)

        candidates: List[tuple] = [
            (subclass, geom),  # L0
            (subclass,),  # L1
        ]
        seen = {subclass}
        for d in donor_path:
            if d not in seen:
                candidates.append((d,))
                seen.add(d)

        if (ELEMENT_ANY,) not in candidates:
            candidates.append((ELEMENT_ANY,))

        return self._resolve("Q4", candidates)

    # See exp.md for detailed rationale

    def _fallback_q5(self, key: tuple) -> FittedDist:
        type_j, type_k, type_i, type_l, ring_ctx = key

        # Coarsen each type to [type, elem, 'heavy', 'any']
        path_j = _q5_type_coarsen(type_j)
        path_k = _q5_type_coarsen(type_k)
        path_i = _q5_type_coarsen(type_i)
        path_l = _q5_type_coarsen(type_l)

        elem_j = path_j[1] if len(path_j) >= 2 else type_j
        elem_k = path_k[1] if len(path_k) >= 2 else type_k

        candidates: List[tuple] = [
            (type_j, type_k, type_i, type_l, ring_ctx),  # L0
            (type_j, type_k, type_i, type_l),  # L1 — strip ring_ctx
            (type_j, type_k, type_i),  # L2 — drop tail flanking
            (type_j, type_k),  # L3 — drop both flanking
            (elem_j, elem_k),  # L4 — coarsen to element
            (ELEMENT_ANY,),  # L5 — global fallback
        ]
        return self._resolve("Q5", candidates)

    # ── Q6 ──────────────────────────────────────────────────────────────────
    # No fallback tree. Exact key only; hard fallback = sum of covalent radii.

    def _fallback_q6(self, key: tuple) -> FittedDist:
        elem_A, elem_B = key
        # Canonical ordering (should already be sorted by scorer, but defensive)
        if elem_A > elem_B:
            elem_A, elem_B = elem_B, elem_A
        canonical = (elem_A, elem_B)

        fd = self._db.data.get("Q6", {}).get(canonical)
        if fd is not None and fd.n >= self._n_min:
            return fd

        # Hard fallback: sum of covalent radii as wall position.
        r_A = _COVALENT_RADIUS.get(elem_A, _DEFAULT_RADIUS)
        r_B = _COVALENT_RADIUS.get(elem_B, _DEFAULT_RADIUS)
        mu = r_A + r_B
        sigma = _Q6_DEFAULT_SIGMA

        # BUGFIX (1.1.0): this hard fallback only fires when no fitted Q6
        # statistic exists for the pair -- true for every (metal, element)
        # pair, since the constraint DB has no metal entries at all (Q6 was
        # only ever fit on organic element pairs). Covalent-radii-sum is the
        # wrong physical model here: it describes BONDING distance, so a
        # non-donor ligand atom (a backbone carbon, say) only gets penalised
        # once it's nearly close enough to covalently bond to the metal --
        # far too permissive for "how close should an atom that ISN'T
        # coordinating ever get to an already 6-donor-saturated centre."
        # Confirmed on EDTA: a backbone carbon measured at 1.71 A from Co
        # (inside typical covalent contact) cost only -0.32 nats out of a
        # ~-13 total score, barely distinguishable from a conformer whose
        # closest such contact was a normal 2.33 A. Van der Waals radii (the
        # standard tool for steric-clash walls in molecular modelling, as
        # opposed to covalent radii for bonded distances) are typically
        # ~1.15-1.2x covalent radii; applied here only for metal-involving
        # pairs (organic-pair fallbacks are unaffected -- not the reported
        # problem, and covalent-sum is a reasonable model for two organic
        # atoms that could plausibly almost-bond). Sigma is also tightened
        # for metal pairs so the wall is steep enough that a severe
        # violation isn't washed out by soft-min aggregation across the
        # (typically much larger) count of organic-organic Q6 pairs.
        if elem_A in _TRANSITION_METALS or elem_B in _TRANSITION_METALS:
            mu *= _Q6_METAL_VDW_SCALE
            sigma = _Q6_METAL_SIGMA

        log.debug(
            "Q6 hard fallback for %s: mu=%.3f (%.3f + %.3f, scale=%s), sigma=%.3f",
            canonical,
            mu,
            r_A,
            r_B,
            elem_A in _TRANSITION_METALS or elem_B in _TRANSITION_METALS,
            sigma,
        )
        return FittedDist(
            type="halfgauss",
            n=0,
            level=99,
            mu=mu,
            sigma=sigma,
        )