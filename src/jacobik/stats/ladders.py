"""
ladders.py
==========
SINGLE source of truth for the fallback ("coarsening") ladders of every
quantity, used by BOTH sides of the pipeline:

* the expander (``ancestor_rows``): which coarser keys does an L0 observation
  also contribute to, and at which level number;
* the runtime lookup (``candidates``): in which order does a query walk the
  coarser keys until one has enough data.

Having one generator removes the whole class of bug where the expander emits
a key the lookup never asks for (or the reverse), and it makes the ordering a
data-selectable *variant* (``Q1_VARIANTS``) rather than code.

Q1 keys carry the metal OXIDATION STATE, not the net charge:

    L0 = (metal, spin, oxidation_state, donor_subclass, coord_geom, trans_subclass)

``OS_UNKNOWN`` marks rows/queries whose oxidation state could not be
established; such rows still feed every state-free level and such queries skip
every level that keys on the oxidation state.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from jacobik.core.chemistry import LANTHANIDES, OS_UNKNOWN
from jacobik.topology.donor_classifier import ELEMENT_ANY, full_path, is_known

# ---------------------------------------------------------------------------
# Sentinels / shared tables
# ---------------------------------------------------------------------------

METAL_ANY = "metal_any"
GEOM_ANY = "geom_any"
TRANS_NONE = "none"

PERIOD_GROUP: Dict[str, str] = {
    **{s: "period4_3d" for s in ("Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn")},
    **{s: "period5_3d" for s in ("Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd")},
    **{s: "period6_3d" for s in ("Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg")},
    **{s: "lanthanide" for s in LANTHANIDES},
    **{s: "actinide" for s in ("Ac", "Th", "Pa", "U", "Np", "Pu")},
}

#: Geometry label -> coordination-number label (single copy; includes CN9).
GEOM_TO_CN: Dict[str, str] = {
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
    "tricap_pris": "CN9",
    # "other" intentionally absent
}


def metal_period_group(symbol: str) -> str:
    return PERIOD_GROUP.get(symbol, METAL_ANY)


def geom_cn(geom: str) -> Optional[str]:
    """CN label of a geometry, or None for 'other'/unknown labels."""
    return GEOM_TO_CN.get(geom)


def donor_path(sub: str) -> List[str]:
    return full_path(sub) if is_known(sub) else [sub, ELEMENT_ANY]


def bucket_family(sub: str) -> Tuple[str, str]:
    """(bucket, family) of a donor subclass.  ONE definition for expander and lookup."""
    p = donor_path(sub)
    fam = p[-2] if len(p) >= 2 else p[-1]
    if fam == ELEMENT_ANY:
        fam = sub
    bkt = p[1] if len(p) >= 3 else fam
    return bkt, fam


def is_anionic_oxygen_donor(sub: str) -> bool:
    """True for any O donor subclass under the O_anionic_any bucket (hydroxide,
    oxo, carboxylate, sulfonate, nitro, phenoxide, ...).  Single definition so
    the co-donor-context feature below and any future consumer agree."""
    return "O_anionic_any" in donor_path(sub)


#: Cap on the "how many anionic-O co-donors share this shell" feature (see
#: ``anionic_co_donor_bucket`` in key_builder.py).  0 / 1 / 2-or-more.  Chosen
#: from the held-out data itself (data/refit/reports/q3-anionic-context.json):
#: mean Q3 drops monotonically and substantially across these three buckets
#: (127.0 -> 113.7/114.1 -> 103.7 deg on the raw LnQM/tmQM water population),
#: and a held-out log-likelihood comparison shows this single feature beats
#: the blanket O_neutral_sigma bucket pool. Not a hand-tuned threshold: it is
#: the coarsest split that still shows monotonic separation in the raw data.
ANIONIC_BUCKET_MAX = 2


def _dedupe(named: List[Tuple[str, tuple]]) -> List[Tuple[str, tuple]]:
    seen, out = set(), []
    for n, k in named:
        if k not in seen:
            seen.add(k)
            out.append((n, k))
    return out


# ---------------------------------------------------------------------------
# Q1
# ---------------------------------------------------------------------------

#: Canonical level order (the integer level stored in the expanded parquet).
Q1_LEVELS: List[str] = [
    "trans", "geom", "cn_sub", "mso_sub", "mo_sub", "m_cn_sub", "m_sub",
    "mso_bkt", "mo_bkt", "m_bkt", "mso_fam", "mo_fam", "m_fam",
    "pg_sub", "pg_bkt", "pg_fam", "any_fam", "fam", "el",
]
Q1_LEVEL_INT: Dict[str, int] = {n: i for i, n in enumerate(Q1_LEVELS)}

#: Levels whose key contains the oxidation state (skipped for unknown OS).
_Q1_OS_LEVELS = frozenset(
    {"trans", "geom", "cn_sub", "mso_sub", "mo_sub", "mso_bkt", "mo_bkt", "mso_fam", "mo_fam"}
)

#: Query-time orderings.  Choose with the held-out log-likelihood evaluation
#: (see refit/), never by taste.  A variant that omits a level simply never asks for it.
Q1_VARIANTS: Dict[str, List[str]] = {
    "orig": [
        "trans", "geom", "mso_sub", "mso_bkt", "mso_fam", "mo_fam", "m_fam",
        "pg_fam", "any_fam", "fam", "el",
    ],
    "m_sub_early": [
        "trans", "geom", "mso_sub", "m_sub", "mso_bkt", "m_bkt", "mso_fam", "mo_fam", "m_fam",
        "pg_sub", "pg_bkt", "pg_fam", "any_fam", "fam", "el",
    ],
    "mo_state": [
        "trans", "geom", "mso_sub", "mo_sub", "m_sub", "mso_bkt", "mo_bkt", "m_bkt",
        "mso_fam", "mo_fam", "m_fam", "pg_sub", "pg_bkt", "pg_fam", "any_fam", "fam", "el",
    ],
    "cn_state": [
        "trans", "geom", "cn_sub", "mso_sub", "mo_sub", "m_cn_sub", "m_sub", "mso_bkt", "mo_bkt",
        "m_bkt", "mso_fam", "mo_fam", "m_fam", "pg_sub", "pg_bkt", "pg_fam", "any_fam", "fam", "el",
    ],
    "metal_first": [
        "trans", "geom", "mso_sub", "m_sub", "m_bkt", "m_fam", "pg_sub", "pg_bkt", "pg_fam",
        "any_fam", "fam", "el",
    ],
    "no_state": [
        "trans", "geom", "m_sub", "m_bkt", "m_fam", "pg_sub", "pg_bkt", "pg_fam", "any_fam", "fam", "el",
    ],
}
DEFAULT_Q1_VARIANT = "mo_state"


def q1_named_levels(key: tuple) -> List[Tuple[str, tuple]]:
    """All Q1 levels for an L0 key, canonical order, NOT deduplicated by key."""
    m, s, o, sub, geom, trans = key
    bkt, fam = bucket_family(sub)
    pg = metal_period_group(m)
    cn = geom_cn(geom)
    lv: List[Tuple[str, tuple]] = []
    if trans != TRANS_NONE:
        lv.append(("trans", (m, s, o, sub, geom, trans)))
    lv.append(("geom", (m, s, o, sub, geom)))
    if cn is not None:
        lv.append(("cn_sub", (m, s, o, sub, cn)))
    lv += [("mso_sub", (m, s, o, sub)), ("mo_sub", (m, o, sub))]
    if cn is not None:
        lv.append(("m_cn_sub", (m, sub, cn)))
    lv += [
        ("m_sub", (m, sub)),
        ("mso_bkt", (m, s, o, bkt)), ("mo_bkt", (m, o, bkt)), ("m_bkt", (m, bkt)),
        ("mso_fam", (m, s, o, fam)), ("mo_fam", (m, o, fam)), ("m_fam", (m, fam)),
    ]
    if pg != METAL_ANY:
        lv += [("pg_sub", (pg, sub)), ("pg_bkt", (pg, bkt)), ("pg_fam", (pg, fam))]
    lv += [("any_fam", (METAL_ANY, fam)), ("fam", (fam,)), ("el", (ELEMENT_ANY,))]
    return lv


def q1_candidates(key: tuple, variant: str = DEFAULT_Q1_VARIANT) -> List[tuple]:
    """Ordered candidate keys for a Q1 query (unknown OS skips OS-keyed levels)."""
    os_known = key[2] != OS_UNKNOWN
    by: Dict[str, tuple] = {}
    for n, k in q1_named_levels(key):
        by.setdefault(n, k)
    out, seen = [], set()
    for n in Q1_VARIANTS[variant]:
        if not os_known and n in _Q1_OS_LEVELS:
            continue
        k = by.get(n)
        if k is not None and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def q1_ancestor_rows(key: tuple) -> List[Tuple[int, tuple]]:
    """(level_int, key) rows an L0 Q1 observation feeds (every level except 'trans')."""
    os_known = key[2] != OS_UNKNOWN
    rows = []
    for n, k in _dedupe(q1_named_levels(key)):
        if n == "trans":
            continue
        if not os_known and n in _Q1_OS_LEVELS:
            continue
        rows.append((Q1_LEVEL_INT[n], k))
    return rows


# ---------------------------------------------------------------------------
# Q2   key = (geom, pair_class, dist_class, metal, subA, subB)
# ---------------------------------------------------------------------------
#
# pair_class (2026-10-01, refit spec §5): the ideal angle class of the two sites in the best-fit assignment of the
# shell to the ideal polyhedron ("c90", "c180", "c120", ...; key_builder.q2_pair_classes). It is kept at EVERY back-off
# level: Q2 pools over chemistry, never over site-pair class (a table pooling cis and trans pairs is a mixture of
# unrelated peaks with an empty trough between them). Haptic faces form their own family ("C_haptic_any"): before,
# the family level merged them with kappa carbon donors (C_donor), so a Cp/arene pair borrowed CO-CO angles.

Q2_LEVELS = ["L0", "bkt", "fam", "pg_fam", "g_fam", "geom_dist", "cn_dist", "geom", "cn", "any"]
Q2_LEVEL_INT = {n: i for i, n in enumerate(Q2_LEVELS)}
PAIR_CLASS_NONE = "na"
HAPTIC_FAMILY = "C_haptic_any"


def _pair(a: str, b: str) -> Tuple[str, str]:
    """Order-free pair: coarsened pairs are sorted so (A,B) and (B,A) share one key."""
    return (a, b) if a <= b else (b, a)


def pair_class(angle_deg: Optional[float]) -> str:
    """Pair-class label of an IDEAL site-site angle (degrees), e.g. 90 -> "c90"; None -> PAIR_CLASS_NONE."""
    return PAIR_CLASS_NONE if angle_deg is None else f"c{int(round(float(angle_deg)))}"


def is_pair_class(x) -> bool:
    return isinstance(x, str) and (x == PAIR_CLASS_NONE or (len(x) > 1 and x[0] == "c" and x[1:].isdigit()))


def q2_bucket_family(sub: str) -> Tuple[str, str]:
    """(bucket, family) for the Q2 ladder: bucket_family, except that a haptic face is its own family."""
    if HAPTIC_FAMILY in donor_path(sub):
        return HAPTIC_FAMILY, HAPTIC_FAMILY
    return bucket_family(sub)


def q2_named_levels(key: tuple) -> List[Tuple[str, tuple]]:
    geom, pc, dist, metal, sa, sb = key
    ba, fa = q2_bucket_family(sa)
    bb, fb = q2_bucket_family(sb)
    cn = geom_cn(geom) or GEOM_ANY
    pg = metal_period_group(metal)
    lv: List[Tuple[str, tuple]] = [
        ("L0", (geom, pc, dist, metal, sa, sb)),
        ("bkt", (geom, pc, dist, metal, *_pair(ba, bb))),
        ("fam", (geom, pc, dist, metal, *_pair(fa, fb))),
    ]
    if pg != METAL_ANY:
        lv.append(("pg_fam", (geom, pc, dist, pg, *_pair(fa, fb))))
    lv += [
        ("g_fam", (geom, pc, dist, *_pair(fa, fb))),
        ("geom_dist", (geom, pc, dist)),
        ("cn_dist", (cn, pc, dist)),
        ("geom", (geom, pc)),
        ("cn", (cn, pc)),
        ("any", (GEOM_ANY, pc)),
    ]
    return lv


def q2_candidates(key: tuple) -> List[tuple]:
    return [k for _, k in _dedupe(q2_named_levels(key))]


def q2_candidates_legacy(key: tuple) -> List[tuple]:
    """Lookup order for DBs fitted before 2026-10-01 (5-tuple keys, no pair class, haptic family = C_donor), so the
    current code can still be run against an old DB as a baseline. ``key`` is a current 6-tuple L0 key."""
    geom, _pc, dist, metal, sa, sb = key
    ba, fa = bucket_family(sa)
    bb, fb = bucket_family(sb)
    cn = geom_cn(geom) or GEOM_ANY
    pg = metal_period_group(metal)
    lv = [(geom, dist, metal, sa, sb), (geom, dist, metal, *_pair(ba, bb)), (geom, dist, metal, *_pair(fa, fb))]
    if pg != METAL_ANY:
        lv.append((geom, dist, pg, *_pair(fa, fb)))
    lv += [(geom, dist, *_pair(fa, fb)), (geom, dist), (cn, dist), (geom,), (cn,), (GEOM_ANY,)]
    out, seen = [], set()
    for k in lv:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def q2_ancestor_rows(key: tuple) -> List[Tuple[int, tuple]]:
    return [(Q2_LEVEL_INT[n], k) for n, k in _dedupe(q2_named_levels(key)) if n != "L0"]


# ---------------------------------------------------------------------------
# Q3   key = (subclass, metal, geom, anion_bucket)
# ---------------------------------------------------------------------------
#
# anion_bucket (see key_builder.anionic_co_donor_bucket) was added after the
# original (subclass, metal, geom) L0 key; existing fits stored under that
# 3-tuple shape remain valid and are referenced below as the "sub_m_geom"
# rung -- they never collide with the new int-tagged rungs because a geom
# string is never equal to an anion_bucket int.

Q3_LEVELS = [
    "L0", "sub_m_anion", "sub_anion", "sub_m_geom", "sub_m", "sub_pg", "sub",
    "bkt", "fam", "any",
]
Q3_LEVEL_INT = {n: i for i, n in enumerate(Q3_LEVELS)}

# Orientation lookup is versioned independently from the fitted database so an
# experimental orientation model can be A/B tested without silently changing
# the established Q3/Q4 control.
#   fine_leaf         : pre-pooling lookup order (per metal+geom).
#   neutral_sigma_pool: blanket-pool O_neutral_sigma at the bucket level.
#   anionic_context    : data-derived -- for O_neutral_sigma donors, prefer
#                        rungs keyed by the co-donor "how many anionic-O
#                        donors share this shell" feature over metal/geom
#                        identity, adaptively falling back through _resolve's
#                        existing n_min gate rather than jumping straight to
#                        the blanket pool. See
#                        data/refit/reports/q3-anionic-context.json for the
#                        held-out log-likelihood comparison that motivated
#                        making this the default.
ORIENTATION_VARIANTS = ("fine_leaf", "neutral_sigma_pool", "anionic_context")
DEFAULT_ORIENTATION_VARIANT = "anionic_context"


def _orientation_variant(variant: str | None) -> str:
    out = variant or DEFAULT_ORIENTATION_VARIANT
    if out not in ORIENTATION_VARIANTS:
        raise ValueError(
            f"unknown orientation variant {out!r}; expected one of {ORIENTATION_VARIANTS}"
        )
    return out


def q3_named_levels(key: tuple) -> List[Tuple[str, tuple]]:
    sub, metal, geom, anion = key
    bkt, fam = bucket_family(sub)
    pg = metal_period_group(metal)
    lv = [
        ("L0", (sub, metal, geom, anion)),
        ("sub_m_anion", (sub, metal, anion)),
        ("sub_anion", (sub, anion)),
        ("sub_m_geom", (sub, metal, geom)),
        ("sub_m", (sub, metal)),
    ]
    if pg != METAL_ANY:
        lv.append(("sub_pg", (sub, pg)))
    lv += [("sub", (sub,)), ("bkt", (bkt,)), ("fam", (fam,)), ("any", (ELEMENT_ANY,))]
    return lv


def q3_candidates(
    key: tuple, variant: str = DEFAULT_ORIENTATION_VARIANT
) -> List[tuple]:
    named = _dedupe(q3_named_levels(key))
    bkt, _ = bucket_family(key[0])
    v = _orientation_variant(variant)
    if bkt == "O_neutral_sigma":
        if v == "neutral_sigma_pool":
            # Blanket-pool: skip straight to the bucket level regardless of
            # how much data the finer rungs have.  Kept only for comparison
            # against anionic_context -- see the module docstring above.
            return [k for name, k in named if name in {"bkt", "fam", "any"}]
        if v == "anionic_context":
            # Prefer the co-donor-context rungs first (most to least
            # specific), THEN fall through to the old metal/geom rungs and
            # finally the blanket pool.  _resolve() still gates every step
            # on n_min, so this only actually uses a rung when it has enough
            # data -- it does not skip metal/geom specificity when a
            # combined (sub, metal, anion) or (sub, anion) fit isn't
            # populated (e.g. before a refit has run with the new key).
            order = ["L0", "sub_m_anion", "sub_anion", "sub_m_geom", "sub_m",
                      "sub_pg", "sub", "bkt", "fam", "any"]
            by_name = dict(named)
            return [by_name[n] for n in order if n in by_name]
        # fine_leaf: behave exactly as before the anion_bucket feature
        # existed -- try metal+geom (old L0 shape) first, ignore anion rungs.
        order = ["sub_m_geom", "sub_m", "sub_pg", "sub", "bkt", "fam", "any"]
        by_name = dict(named)
        return [by_name[n] for n in order if n in by_name]
    return [k for _, k in named]


def q3_ancestor_rows(key: tuple) -> List[Tuple[int, tuple]]:
    return [(Q3_LEVEL_INT[n], k) for n, k in _dedupe(q3_named_levels(key)) if n != "L0"]


# ---------------------------------------------------------------------------
# Q4   key = (subclass, geom, k, anion_bucket)
#      k = number of ligand substituents on the donor
# ---------------------------------------------------------------------------

Q4_LEVELS = ["L0", "sub_k_anion", "sub_geom_k", "sub_k", "bkt_k", "fam_k", "k", "any"]
Q4_LEVEL_INT = {n: i for i, n in enumerate(Q4_LEVELS)}


def q4_named_levels(key: tuple) -> List[Tuple[str, tuple]]:
    sub, geom, k, anion = key
    bkt, fam = bucket_family(sub)
    return [
        ("L0", (sub, geom, k, anion)),
        ("sub_k_anion", (sub, k, anion)),
        ("sub_geom_k", (sub, geom, k)),
        ("sub_k", (sub, k)),
        ("bkt_k", (bkt, k)),
        ("fam_k", (fam, k)),
        ("k", (k,)),
        ("any", (ELEMENT_ANY,)),
    ]


def q4_candidates(
    key: tuple, variant: str = DEFAULT_ORIENTATION_VARIANT
) -> List[tuple]:
    named = _dedupe(q4_named_levels(key))
    bkt, _ = bucket_family(key[0])
    v = _orientation_variant(variant)
    if bkt == "O_neutral_sigma":
        if v == "neutral_sigma_pool":
            return [k for name, k in named if name in {"bkt_k", "fam_k", "k", "any"}]
        if v == "anionic_context":
            order = ["L0", "sub_k_anion", "sub_geom_k", "sub_k", "bkt_k", "fam_k", "k", "any"]
            by_name = dict(named)
            return [by_name[n] for n in order if n in by_name]
        order = ["sub_geom_k", "sub_k", "bkt_k", "fam_k", "k", "any"]
        by_name = dict(named)
        return [by_name[n] for n in order if n in by_name]
    return [k for _, k in named]


def q4_ancestor_rows(key: tuple) -> List[Tuple[int, tuple]]:
    return [(Q4_LEVEL_INT[n], k) for n, k in _dedupe(q4_named_levels(key)) if n != "L0"]


# ---------------------------------------------------------------------------
# Q5   key = (type_j, type_k, type_i, type_l, ring_ctx)
# ---------------------------------------------------------------------------

Q5_LEVELS = ["L0", "no_ortho", "no_ring", "jki", "jk", "elem", "any"]
Q5_LEVEL_INT = {n: i for i, n in enumerate(Q5_LEVELS)}
_HYB_PREFIX = ("C_", "N_", "O_", "S_", "P_")


def q5_elem(token: str) -> str:
    """Local bond type -> element (C_sp3 -> C).  Halogens / H / 'X' stay as they are."""
    for p in _HYB_PREFIX:
        if token.startswith(p):
            return p[:-1]
    return token


def q5_named_levels(key: tuple) -> List[Tuple[str, tuple]]:
    tj, tk, ti, tl, ctx = key
    # ctx = ring context ("free" / "chelate"), for aryl rotors with the ortho count ("free_o2", key_builder.q5_context);
    # "no_ortho" drops the count (and is the L0 key of DBs fitted before 2026-10-01)
    return [
        ("L0", (tj, tk, ti, tl, ctx)),
        ("no_ortho", (tj, tk, ti, tl, str(ctx).split("_o")[0])),
        ("no_ring", (tj, tk, ti, tl)),
        ("jki", (tj, tk, ti)),
        ("jk", (tj, tk)),
        ("elem", (q5_elem(tj), q5_elem(tk))),
        ("any", (ELEMENT_ANY,)),
    ]


def q5_candidates(key: tuple) -> List[tuple]:
    return [k for _, k in _dedupe(q5_named_levels(key))]


def q5_ancestor_rows(key: tuple) -> List[Tuple[int, tuple]]:
    return [(Q5_LEVEL_INT[n], k) for n, k in _dedupe(q5_named_levels(key)) if n != "L0"]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


ANCESTOR_FN = {
    "Q1": q1_ancestor_rows,
    "Q2": q2_ancestor_rows,
    "Q3": q3_ancestor_rows,
    "Q4": q4_ancestor_rows,
    "Q5": q5_ancestor_rows,
}


def l0_bucket(quantity: str, key: tuple) -> str:
    """
    Reservoir bucket id used by the expander to cap each ancestor node per
    L0 subclass.  Lanthanide and d-block rows are capped separately so the far
    larger LnQM population cannot crowd out d-block observations.
    """
    try:
        if quantity == "Q1":
            return f"{key[3]}|{'Ln' if key[0] in LANTHANIDES else 'TM'}"
        if quantity == "Q2":
            return f"{key[4]}|{'Ln' if key[3] in LANTHANIDES else 'TM'}"
        if quantity in ("Q3", "Q4", "Q5"):
            return str(key[0])
    except (IndexError, TypeError):
        pass
    return "unknown"
