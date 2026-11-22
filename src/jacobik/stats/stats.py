"""
stats.py
========
Constraints database loader and key-resolution interface for the IK engine.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np

from jacobik.core.chemistry import Q6_FALLBACK_DEFAULT_RADIUS, Q6_FALLBACK_RADII, TRANSITION_METALS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: VDW radii run ~1.15-1.2x covalent radii; applied to covalent-radii-sum
#: for metal-involving Q6 fallback pairs only (see _fallback_q6).
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
    #: full covariance matrices, one per component (QO / "mvgmm" entries only)
    covs: Optional[List[List[List[float]]]] = None
    #: haptic QO entries: which of orientation.HAPTIC_FEATURES the mixture is over (None: legacy [tilt, bend])
    feats: Optional[List[int]] = None
    #: per-substituent haptic entries: "H" / "X" (orientation.PER_SUB_OFFSET); None for face entries
    per_sub: Optional[str] = None
    #: B2.6 reference log-likelihood of a typical real observation under this entry, as the scorer evaluates it
    #: (stats/reference_ll.py); None in DBs fitted before 2026-09-27
    ref_ll: Optional[float] = None
    ref_sd: Optional[float] = None
    #: Student-t mixtures ("tmix_cos", Q2 since 2026-10-01): degrees of freedom shared by the components; ``vars``
    #: then holds the squared scales
    nu: Optional[float] = None
    #: partial pooling (2026-10-01): the entry was MAP-adapted from its ladder parent, so it is valid at ANY n (the
    #: lookup takes the most specific pooled entry instead of walking on to a level with n >= n_min)
    pooled: bool = False
    #: number of independent complexes behind the entry (evidence weight; the fit's BIC count)
    n_eff: Optional[float] = None


# ---------------------------------------------------------------------------
# ConstraintDB
# ---------------------------------------------------------------------------


class ConstraintDB:
    """
    Parsed in-memory representation of constraints.json.
    """

    q1_variant: Optional[str] = None
    orientation_variant: Optional[str] = None
    #: names of the runtime tables the container carries (``_tables``, or a legacy ``.bends.json``)
    tables: Sequence[str] = ()

    def __init__(self, data: Dict[str, Dict[tuple, FittedDist]]) -> None:
        self.data = data
        self.priors: Dict[str, Dict[tuple, dict]] = {}  # B2.5 QG / QP categorical priors
        self._q2_legacy: Optional[bool] = None
        #: the refit's sidecar (constraints.meta.json): ladder variants and, since 2026-10-01, the model versions
        #: ("q5_model": "wells_folded", "q6_model": "onset_hp", "pooling": {...}) the runtime keys its defaults on
        self.meta: Dict[str, object] = {}

    @property
    def q2_legacy(self) -> bool:
        """True for a DB fitted before the Q2 site-pair class (2026-10-01): its Q2 keys carry no pair class, so the
        lookup uses ladders.q2_candidates_legacy."""
        if self._q2_legacy is None:
            from jacobik.stats.ladders import is_pair_class

            q2 = self.data.get("Q2", {})
            self._q2_legacy = bool(q2) and not any(len(k) >= 2 and is_pair_class(k[1]) for k in q2)
        return self._q2_legacy

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


_KEY_TOKEN = re.compile(r"'([^'\\]*)'|(-?\d+)|(None)|(True)|(False)")
_KEY_LITERALS = {"None": None, "True": True, "False": False}


def _parse_key(key_repr: str):
    """The tuple a DB key string spells (as ast.literal_eval would), parsed without the compiler.

    Keys are flat tuples of quoted strings, integers, None and booleans. The token parse is accepted only when the
    tuple's own repr reproduces the string exactly; anything else goes to ast.literal_eval. 420k keys: 0.4 s instead
    of 4 s."""
    if key_repr.startswith("(") and key_repr.endswith(")"):
        vals = []
        for s, i, *lit in _KEY_TOKEN.findall(key_repr):
            if s or (not i and not any(lit)):
                vals.append(s)
            elif i:
                vals.append(int(i))
            else:
                vals.append(_KEY_LITERALS[next(x for x in lit if x)])
        t = tuple(vals)
        if repr(t) == key_repr:
            return t
    return ast.literal_eval(key_repr)


def _fitted(entry: dict) -> FittedDist:
    """The FittedDist of one DB entry (its fallback level is the entry's 'level' field, else 0)."""
    return FittedDist(
        type=entry["type"], n=int(entry.get("n", 0)), level=int(entry.get("level", 0)),
        means=entry.get("means"), vars=entry.get("vars"), kappas=entry.get("kappas"), weights=entry.get("weights"),
        mu=entry.get("mu"), sigma=entry.get("sigma"), covs=entry.get("covs"), feats=entry.get("feats"),
        per_sub=entry.get("per_sub"), ref_ll=entry.get("ref_ll"), ref_sd=entry.get("ref_sd"), nu=entry.get("nu"),
        pooled=bool(entry.get("pooled", False)), n_eff=entry.get("n_eff"),
    )


_PLAIN_KEY_TYPES = (str, int, bool, type(None))


def _key_repr(key) -> str:
    """The canonical string of a lookup key, as the container stores keys (numpy scalars read as Python ones)."""
    if type(key) is not tuple:
        key = (key,)
    if not all(type(x) in _PLAIN_KEY_TYPES for x in key):
        key = tuple(x.item() if isinstance(x, np.generic) else x for x in key)
    return repr(key)


class _LazyTable(Mapping):
    """One quantity of an .npz DB, read on demand.

    Loading builds only a key-string -> row index (0.1 s for the 420k entries); an entry is decoded into its
    FittedDist the first time it is looked up, and lookups (hits and misses) are memoised. A solve reads a few
    thousand entries, so a load costs about 1 s and 0.2 GB instead of 5.5 s and 2 GB. Keys are matched by their
    canonical repr, the form the fit writes them in; iteration parses keys only when asked."""

    def __init__(self, quantity: str, rows, n_min: int) -> None:
        off = len(quantity) + 1                     # stored keys are "Q1:('Fe', ...)"
        keep = np.flatnonzero(rows.n >= n_min).tolist()
        self._rows = rows
        self._index = {rows.keys[i][off:]: i for i in keep}
        self._memo: Dict[object, Optional[FittedDist]] = {}
        self._parsed: Optional[list] = None

    def get(self, key, default=None):
        try:
            fd = self._memo[key]
        except KeyError:
            i = self._index.get(_key_repr(key))
            fd = self._memo[key] = None if i is None else _fitted(self._rows(i))
        return default if fd is None else fd

    def __getitem__(self, key) -> FittedDist:
        fd = self.get(key)
        if fd is None:
            raise KeyError(key)
        return fd

    def __contains__(self, key) -> bool:
        return self.get(key) is not None

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self):
        if self._parsed is None:
            self._parsed = [k if isinstance(k := _parse_key(s), tuple) else (k,) for s in self._index]
        return iter(self._parsed)


#: load_db results by (resolved path, mtime_ns, size, n_min): a ConstraintDB is read-only after loading, and a solve
#: of several configuration classes, or a script calling generate() repeatedly, otherwise re-reads the 28 MB
#: container each time (about 8 s per read).
_DB_CACHE: Dict[tuple, "ConstraintDB"] = {}


def load_db(path=None, n_min: int = 1, meta: Optional[dict] = None) -> ConstraintDB:
    """
    Load a constraints DB (.npz container or constraints.json) from disk and return a ConstraintDB.

    Loads of the same file (same resolved path, size and modification time, no explicit ``meta``) return the cached
    ConstraintDB of the first load.


    Parameters
    ----------
    path  : path to constraints DB, a Traversable resource, the parsed JSON dict itself, or None (default: ``default_db_path()``)
    meta  : sidecar dict to use instead of <stem>.meta.json next to ``path``
    n_min : only include entries with n ≥ n_min (default 1 = keep all)

    Returns
    -------
    ConstraintDB

    """
    if path is None:
        from jacobik.core.env import default_db_path
        path = default_db_path()
        if path is None:
            raise FileNotFoundError("Constraints DB not found (pass path=..., or set JACOBIK_DB)")
    cache_key = None
    if isinstance(path, (str, Path)) and meta is None:
        try:
            p_res = Path(path).resolve()
            st_ = p_res.stat()

            def _side(q):  # a sidecar the load reads beside the path as given: its state is part of the key
                try:
                    s = q.stat()
                    return (s.st_mtime_ns, s.st_size)
                except OSError:
                    return None

            pg = Path(path)
            cache_key = (str(pg), str(p_res), st_.st_mtime_ns, st_.st_size, int(n_min),
                         _side(pg.with_suffix(".bends.json")), _side(pg.with_name(pg.stem + ".meta.json")))
        except OSError:
            cache_key = None
        if cache_key is not None and cache_key in _DB_CACHE:
            return _DB_CACHE[cache_key]
    db = _load_db_uncached(path, n_min, meta)
    if cache_key is not None:
        _DB_CACHE[cache_key] = db
    return db


def _load_db_uncached(path, n_min: int, meta: Optional[dict]) -> ConstraintDB:
    lazy_rows: dict = {}
    if isinstance(path, dict):
        raw: dict = path
        path = None
    else:
        if isinstance(path, (str, Path)):
            path = Path(path)
        if str(path).endswith(".npz"):
            from jacobik.core.container import open_lossless_npz

            lazy_rows, raw, npz_meta = open_lossless_npz(path)
            if meta is None and npz_meta:
                meta = npz_meta
        else:
            with path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)

    # QO = joint donor orientation (stats/orientation.py); absent in DBs fitted before 2026-09-24.
    data: Dict[str, Dict[tuple, FittedDist]] = {q: {} for q in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "QO")}
    n_loaded = 0
    n_skipped = 0
    cats: Dict[str, Dict[tuple, dict]] = {}

    embedded_meta = raw.get("_meta") if isinstance(raw.get("_meta"), dict) else None
    embedded_tables = raw.get("_tables") if isinstance(raw.get("_tables"), dict) else {}

    for json_key, entry in raw.items():
        if json_key.startswith("_"):  # container members (_meta, _tables): dbio.py
            continue
        # Split "Q1:('Fe', ...)" into quantity and key_repr
        colon = json_key.index(":")
        quantity = json_key[:colon]
        key_repr = json_key[colon + 1 :]

        if quantity == "NCI":
            # Legacy retired NCI spline tables in existing DBs are ignored cleanly.
            continue
        if quantity in ("QG", "QP", "Q6P", "QV", "QVR"):
            # B2.5 hypothesis-layer priors: categorical {labels, p, n}; Q6P = Q6 potential-of-mean-force tables
            # (stats/q6.py, 2026-10-14); kept as plain dicts
            try:
                kt = _parse_key(key_repr)
            except (ValueError, SyntaxError):
                continue
            cats.setdefault(quantity, {})[kt if isinstance(kt, tuple) else (kt,)] = entry
            continue
        if quantity not in data:
            log.debug("Unknown quantity in constraints.json: %s", quantity)
            continue

        n_obs = int(entry.get("n", 0))
        if n_obs < n_min:
            n_skipped += 1
            continue

        try:
            key_tuple = _parse_key(key_repr)
        except (ValueError, SyntaxError) as exc:
            log.warning("Failed to parse key %r: %s", key_repr, exc)
            continue

        if not isinstance(key_tuple, tuple):
            # Single-element tuples with a trailing comma parse correctly;
            # non-tuples indicate a format error.
            key_tuple = (key_tuple,)

        data[quantity][key_tuple] = _fitted(entry)
        n_loaded += 1

    for quantity, rows in lazy_rows.items():
        if quantity in data:
            table = data[quantity] = _LazyTable(quantity, rows, n_min)
            n_loaded += len(table)
            n_skipped += len(rows) - len(table)

    log.info(
        "Loaded %d distributions from %s (%d skipped, n < %d)",
        n_loaded,
        path,
        n_skipped,
        n_min,
    )
    db = ConstraintDB(data)
    db.priors = cats

    def _q7_table(q7: dict) -> dict:
        return {
            (str(e["key"][0]), e["key"][1], tuple(e["key"][2]) if e["key"][2] is not None else None,
             e["key"][3], e["key"][4]): {k: v for k, v in e.items() if k != "key"}
            for e in q7.get("entries", [])
        }

    # Runtime tables: the container's own first; a legacy DB's <stem>.bends.json beside the path as given (never
    # resolved to a symlink target).
    db.tables = sorted(embedded_tables)
    if "bends" in embedded_tables:
        try:
            db.priors["Q7"] = _q7_table(embedded_tables["bends"])
        except Exception as exc:
            log.error("Failed to read the embedded Q7 bend table: %s", exc)
    elif path is not None and embedded_meta is None:
        sidecar = Path(path).with_suffix(".bends.json")
        if sidecar.exists():
            try:
                db.priors["Q7"] = _q7_table(json.loads(sidecar.read_text(encoding="utf-8")))
                db.tables = ["bends"]
            except Exception as exc:
                log.warning("Failed to load Q7 bend sidecar %s: %s", sidecar, exc)
    try:
        from jacobik.core.container import missing_tables

        for flag, table in missing_tables(db.tables):
            log.error("profile flag %s needs the %r table, which this DB (%s) does not carry; the term stays inert "
                      "(the bench and tools.run refuse to start in this state)", flag, table, path)
    except Exception:  # pragma: no cover - never fail a load over the check
        pass
    if meta is None and embedded_meta is not None:
        meta = embedded_meta
    if meta is not None:
        db.meta = dict(meta)
        db.q1_variant = meta.get("q1_variant")
        db.orientation_variant = meta.get("orientation_variant")
        return db
    # Optional sidecar written by the refit (chosen Q1 ladder variant etc.)
    try:
        if isinstance(path, Path):
            meta = path.with_name(path.stem + ".meta.json")
            if meta.exists():
                metadata = json.loads(meta.read_text())
                db.meta = metadata
                db.q1_variant = metadata.get("q1_variant")
                db.orientation_variant = metadata.get("orientation_variant")
    except Exception as exc:  # pragma: no cover
        log.warning("could not read constraints meta sidecar: %s", exc)
    return db


# ---------------------------------------------------------------------------
# Internal coarsening helpers
# ---------------------------------------------------------------------------


def _is_empty_dist(fd: "FittedDist") -> bool:
    """
    True if a FittedDist carries no usable parameters and must not be returned
    by the resolver, regardless of its observation count ``n``.

    Covers gmm, vmm and gmm_cos (means/vars/weights in cos-space) entries and
    halfgauss entries; any other type is treated as empty.
    """
    t = fd.type
    if t in ("gmm", "vmm", "gmm_cos"):
        return (not fd.means) or (not fd.weights)
    if t == "tmix_cos":
        return (not fd.means) or (not fd.weights) or fd.nu is None
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
    "Q1": 6,  # (metal, spin, oxidation_state, donor_subclass, coord_geom, trans_partner)
    "Q2": 6,  # (coord_geom, pair_class, graph_dist, metal, donor_A_subclass, donor_B_subclass)
    "Q3": 4,  # (donor_subclass, metal, coord_geom, anion_co_donor_bucket)
    "Q4": 4,  # (donor_subclass, coord_geom, n_substituents, anion_co_donor_bucket)
    "Q5": 5,  # (type_j, type_k, type_i, type_l, ring_ctx)
    "Q6": 2,  # (elem_A, elem_B)
}


# ---------------------------------------------------------------------------
# StatsLookup (moved; re-exported so existing imports keep working)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Key-resolution lookup engine
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

    def __init__(
        self,
        db: ConstraintDB,
        n_min: int = 30,
        q1_variant: Optional[str] = None,
        orientation_variant: Optional[str] = None,
        shannon: bool = False,
    ) -> None:
        from jacobik.stats import ladders as L

        self._db = db
        self._n_min = n_min
        #: Q1 ladder ordering (see ladders.Q1_VARIANTS); the fitted DB may
        #: record its own choice in a sidecar (constraints.meta.json).
        self.q1_variant = q1_variant or getattr(db, "q1_variant", None) or L.DEFAULT_Q1_VARIANT
        #: Q3/Q4 lookup policy.  This is deliberately separate from Q1 and
        #: recorded in the DB sidecar so experimental orientation work cannot
        #: silently alter the validated refit control.
        self.orientation_variant = (
            orientation_variant
            or getattr(db, "orientation_variant", None)
            or L.DEFAULT_ORIENTATION_VARIANT
        )
        # Validate eagerly instead of failing part-way through a solve.
        L._orientation_variant(self.orientation_variant)
        #: Switchable Shannon-radius shift for (metal, OS, spin) states that
        #: have no data of their own (see stats/shannon.py).
        self.shannon = bool(shannon)
        self.last_note: Optional[str] = None
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
        from jacobik.stats.stats import _L0_ARITY

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
        self.last_note = None
        if quantity == "Q1" and self.shannon:
            fd = self._q1_shannon(key)
            if fd is not None:
                return fd
        return fn(key)

    @property
    def has_orientation(self) -> bool:
        """True when the DB carries joint donor-orientation (QO) fits."""
        return bool(self._db.data.get("QO"))

    def q6_pmf_tables(self) -> Dict[tuple, dict]:
        """{(a, b, cls, ang): table entry} of the Q6 potential of mean force (stats/q6pmf.py); {} for DBs without."""
        return getattr(self._db, "priors", {}).get("Q6P", {})

    def q7_bound_angle_table(self) -> Dict[tuple, dict]:
        """Crystal bound-state bend-angle sidecar entries, when present."""
        return getattr(self._db, "priors", {}).get("Q7", {})

    def orientation_dist(self, sub: str, metal: str, geom: str, anion, k: int, ring: int):
        """
        (resolved key, FittedDist) of the joint donor-orientation model for one donor, walking
        ``orientation.level_keys`` from most to least specific; (None, None) if nothing is fitted.
        Keys were only fitted with >= orientation.N_MIN rows, so no n_min check is applied here.
        """
        from jacobik.stats.orientation import level_keys

        table = self._db.data.get("QO", {})
        for key in level_keys(sub, metal, geom, anion, k, ring):
            fd = table.get(key)
            if fd is not None and fd.means:
                return key, fd
        return None, None

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
            if fd.n >= self._n_min or fd.pooled:
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

    def _fallback_q1(self, key: tuple) -> FittedDist:
        from jacobik.stats import ladders as L

        return self._resolve("Q1", L.q1_candidates(key, self.q1_variant))

    def _fallback_q2(self, key: tuple) -> FittedDist:
        from jacobik.stats import ladders as L

        if self._db.q2_legacy:
            return self._resolve("Q2", L.q2_candidates_legacy(key))
        return self._resolve("Q2", L.q2_candidates(key))

    def _fallback_q3(self, key: tuple) -> FittedDist:
        from jacobik.stats import ladders as L

        return self._resolve("Q3", L.q3_candidates(key, self.orientation_variant))

    def _fallback_q4(self, key: tuple) -> FittedDist:
        from jacobik.stats import ladders as L

        return self._resolve("Q4", L.q4_candidates(key, self.orientation_variant))

    def _fallback_q5(self, key: tuple) -> FittedDist:
        from jacobik.stats import ladders as L

        return self._resolve("Q5", L.q5_candidates(key))

    # ── Q6 ──────────────────────────────────────────────────────────────────
    # No fallback tree. Exact key only; hard fallback = sum of covalent radii.

    def _fallback_q6(self, key: tuple) -> FittedDist:
        elem_A, elem_B = key
        if elem_A > elem_B:
            elem_A, elem_B = elem_B, elem_A
        canonical = (elem_A, elem_B)
        q6 = self._db.data.get("Q6", {})

        # exact pair, then (period-group, X) for metal pairs, then hard fallback; a polar-H key ("Hp", 2026-10-01)
        # falls back to the plain H key (DBs fitted before it have none)
        cands = [canonical]
        if "Hp" in canonical:
            plain = tuple(sorted("H" if e == "Hp" else e for e in canonical))
            cands.append(plain)
            elem_A, elem_B = plain
        from jacobik.stats.ladders import PERIOD_GROUP

        for m, x in ((elem_A, elem_B), (elem_B, elem_A)):
            pg = PERIOD_GROUP.get(m)
            if pg is not None:
                cands.append((pg, x))  # the post-fit n-weighted wall (fitter writes repr((pg, x)); ARCHITECTURE.md section 3 (CHANGELOG 2026-09-28))
                cands.append(tuple(sorted((pg, x))))  # the raw-pooled wall
        self._last_candidates = list(cands)
        for c in cands:
            fd = q6.get(c)
            if fd is not None and fd.n >= self._n_min and not _is_empty_dist(fd):
                self._last_resolved_key = c
                return fd

        r_A = Q6_FALLBACK_RADII.get(elem_A, Q6_FALLBACK_DEFAULT_RADIUS)
        r_B = Q6_FALLBACK_RADII.get(elem_B, Q6_FALLBACK_DEFAULT_RADIUS)
        mu = r_A + r_B
        sigma = _Q6_DEFAULT_SIGMA
        if elem_A in TRANSITION_METALS or elem_B in TRANSITION_METALS:
            mu *= _Q6_METAL_VDW_SCALE
            sigma = _Q6_METAL_SIGMA
        return FittedDist(type="halfgauss", n=0, level=99, mu=mu, sigma=sigma)

    # ── Shannon-radius shift (switchable) ───────────────────────────────────
    def _q1_shannon(self, key: tuple) -> Optional[FittedDist]:
        """
        If the query's (metal, spin, OS) state has no specific data, borrow the
        distribution of the best-populated reference state of the same
        (metal, donor, geometry) and shift its component means by the Shannon
        ionic-radius difference.  Returns None to fall through to the normal
        ladder (state has data, OS unknown, no radius data, or no reference).
        """
        from jacobik.stats import ladders as L
        from jacobik.core import chemistry as SH

        metal, spin, os_, sub, geom, trans = key
        if os_ == L.OS_UNKNOWN:
            return None
        q1 = self._db.data.get("Q1", {})
        own = [(metal, spin, os_, sub, geom, trans), (metal, spin, os_, sub, geom),
               (metal, spin, os_, sub)]
        for c in own:
            fd = q1.get(c)
            if fd is not None and fd.n >= self._n_min and not _is_empty_dist(fd):
                return None  # the state is covered: no borrowing
        cn = L.geom_cn(geom)
        best = None
        for k, fd in self._ref_states(metal, sub, geom):
            if fd.n < self._n_min or _is_empty_dist(fd):
                continue
            _, s_ref, o_ref = k[0], k[1], k[2]
            if (s_ref, o_ref) == (spin, os_):
                continue
            d = SH.radius_shift(metal, (o_ref, s_ref), (os_, spin), cn)
            if d is None:
                continue
            score = fd.n
            if best is None or score > best[0]:
                best = (score, fd, d, (s_ref, o_ref))
        if best is None:
            return None
        _, fd, d, ref = best
        out = FittedDist(
            type=fd.type, n=fd.n, level=fd.level, means=[m + d for m in fd.means],
            vars=list(fd.vars) if fd.vars else fd.vars, kappas=fd.kappas,
            weights=list(fd.weights), mu=fd.mu, sigma=fd.sigma,
        )
        self.last_note = f"shannon: ref(spin={ref[0]},OS={ref[1]}) shift {d:+.3f} A"
        self._last_resolved_key = ("shannon", metal, spin, os_, sub, geom)
        return out

    def _ref_states(self, metal: str, sub: str, geom: str):
        cache = getattr(self, "_ref_cache", None)
        if cache is None:
            cache = self._ref_cache = {}
            for k, fd in self._db.data.get("Q1", {}).items():
                if len(k) == 5 and isinstance(k[1], int) and isinstance(k[2], int):
                    cache.setdefault((k[0], k[3], k[4]), []).append((k, fd))
        return cache.get((metal, sub, geom), [])
