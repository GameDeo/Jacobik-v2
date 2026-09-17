"""
donor_hierarchy.py
==================
Machine-readable taxonomy of transition-metal donor subclasses.
"""

from __future__ import annotations

from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Sentinel
# ---------------------------------------------------------------------------

#: Global catch-all — topmost node in the donor dimension.
ELEMENT_ANY = "element_any"

# See exp.md for detailed rationale

_PARENT: Dict[str, str] = {
    # -----------------------------------------------------------------------
    # Carbon
    # -----------------------------------------------------------------------
    "C_donor": ELEMENT_ANY,
    # bucket: sigma donors
    "C_sigma": "C_donor",
    "C_carbonyl": "C_sigma",
    "C_isocyanide": "C_sigma",
    "C_carbene": "C_sigma",
    "C_alkyl": "C_sigma",
    # bucket: kappa pi
    "C_pi_sigma": "C_donor",
    "C_pi": "C_pi_sigma",
    # bucket: haptic faces
    "C_haptic_any": "C_donor",
    "C_allyl": "C_haptic_any",  # η³
    "C_eta4": "C_haptic_any",  # η⁴
    "C_eta5": "C_haptic_any",  # η⁵ (Cp)
    "C_eta6": "C_haptic_any",  # η⁶ (arene)
    # η2 and any unrecognised order — legacy string kept as alias
    "C_haptic": "C_haptic_any",
    # -----------------------------------------------------------------------
    # Nitrogen
    # -----------------------------------------------------------------------
    "N_donor": ELEMENT_ANY,
    # bucket: linear/terminal
    "N_linear": "N_donor",
    "N_dinitrogen": "N_linear",
    "N_cyanide": "N_linear",
    "N_nitrosyl": "N_linear",
    # bucket: aromatic
    "N_aromatic_any": "N_donor",
    "N_NHC": "N_aromatic_any",
    "N_pyridine": "N_aromatic_any",
    "N_azole": "N_aromatic_any",
    # legacy string used as fallback subclass — alias to the bucket node
    "N_aromatic": "N_aromatic_any",
    # bucket: sp3-like (flexible approach geometry)
    "N_sp3_any": "N_donor",
    "N_imine": "N_sp3_any",
    "N_amide": "N_sp3_any",
    "N_amine": "N_sp3_any",
    # -----------------------------------------------------------------------
    # Oxygen
    # -----------------------------------------------------------------------
    "O_donor": ELEMENT_ANY,
    # bucket: anionic / delocalized
    "O_anionic_any": "O_donor",
    "O_carboxylate": "O_anionic_any",
    "O_sulfonate": "O_anionic_any",
    "O_nitro": "O_anionic_any",
    "O_phenoxide": "O_anionic_any",
    # bucket: neutral sp2 (lone pair on pi oxygen)
    "O_neutral_pi": "O_donor",
    "O_carbonyl": "O_neutral_pi",
    "O_hydroxo": "O_neutral_pi",
    # bucket: neutral sp3 (lone pair on sigma oxygen)
    # O_ether_cyclic is collapsed into O_ether at the DESIGN.md bucket level;
    # the legacy subclass string is kept as an alias.
    "O_neutral_sigma": "O_donor",
    "O_water": "O_neutral_sigma",
    "O_ether": "O_neutral_sigma",
    "O_ether_cyclic": "O_neutral_sigma",  # alias; merges into O_ether bucket
    "O_alkoxide": "O_neutral_sigma",
    # -----------------------------------------------------------------------
    # Phosphorus  (no intermediate bucket; all three pool directly to P_donor)
    # -----------------------------------------------------------------------
    "P_donor": ELEMENT_ANY,
    "P_phosphine": "P_donor",
    "P_phosphite": "P_donor",
    "P_phosphinite": "P_donor",
    # -----------------------------------------------------------------------
    # Sulfur
    # -----------------------------------------------------------------------
    "S_donor": ELEMENT_ANY,
    "S_pi_any": "S_donor",
    "S_sulfoxide": "S_pi_any",
    "S_thiocarbonyl": "S_pi_any",
    "S_sigma_any": "S_donor",
    "S_thioether": "S_sigma_any",
    "S_thiolate": "S_sigma_any",
    # -----------------------------------------------------------------------
    # Halogens  (shallow: subclass -> family -> element_any)
    # -----------------------------------------------------------------------
    "Hal_donor": ELEMENT_ANY,
    "Hal_F": "Hal_donor",
    "Hal_Cl": "Hal_donor",
    "Hal_Br": "Hal_donor",
    "Hal_I": "Hal_donor",
    # -----------------------------------------------------------------------
    # Hydrogen  (shallow)
    # -----------------------------------------------------------------------
    "H_donor": ELEMENT_ANY,
    "H_hydride": "H_donor",
    # -----------------------------------------------------------------------
    # Boron  (shallow)
    # -----------------------------------------------------------------------
    "B_donor": ELEMENT_ANY,
    "B_boronate": "B_donor",
    # -----------------------------------------------------------------------
    # Chalcogenides / pnictogens  (no subclasses currently; leaf = family)
    # -----------------------------------------------------------------------
    "Se_donor": ELEMENT_ANY,
    "Te_donor": ELEMENT_ANY,
    "As_donor": ELEMENT_ANY,
}

# ---------------------------------------------------------------------------
# Element symbol -> family node
# ---------------------------------------------------------------------------

_ELEMENT_TO_FAMILY: Dict[str, str] = {
    "C": "C_donor",
    "N": "N_donor",
    "O": "O_donor",
    "P": "P_donor",
    "S": "S_donor",
    "F": "Hal_donor",
    "Cl": "Hal_donor",
    "Br": "Hal_donor",
    "I": "Hal_donor",
    "B": "B_donor",
    "Se": "Se_donor",
    "Te": "Te_donor",
    "As": "As_donor",
    "H": "H_donor",
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def ancestors(node: str) -> List[str]:
    """
    Return the ordered ancestry chain above *node*, from its immediate parent
    up to and including ELEMENT_ANY.

    Raises KeyError if *node* is not in the hierarchy.

    Examples
    --------
    >>> ancestors("N_pyridine")
    ['N_aromatic_any', 'N_donor', 'element_any']

    >>> ancestors("P_phosphine")
    ['P_donor', 'element_any']

    >>> ancestors("C_eta5")
    ['C_haptic_any', 'C_donor', 'element_any']

    >>> ancestors("Se_donor")
    ['element_any']
    """
    if node not in _PARENT and node != ELEMENT_ANY:
        raise KeyError(f"Unknown donor node: {node!r}")
    chain: List[str] = []
    current = node
    while current != ELEMENT_ANY:
        current = _PARENT[current]
        chain.append(current)
    return chain


def coarsen(node: str, n: int = 1) -> str:
    """
    Walk *node* up *n* levels in the hierarchy.  Clamps at ELEMENT_ANY.

    Examples
    --------
    >>> coarsen("N_pyridine")        # n=1
    'N_aromatic_any'
    >>> coarsen("N_pyridine", 2)
    'N_donor'
    >>> coarsen("N_pyridine", 99)    # clamps
    'element_any'
    """
    current = node
    for _ in range(n):
        if current == ELEMENT_ANY:
            break
        if current not in _PARENT:
            raise KeyError(f"Unknown donor node: {current!r}")
        current = _PARENT[current]
    return current


def element_to_family(symbol: str) -> str:
    """
    Return the family node for an element symbol.

    Raises KeyError for elements not in the donor vocabulary.

    Examples
    --------
    >>> element_to_family("Fe")    # metals are not donor families
    KeyError
    >>> element_to_family("N")
    'N_donor'
    """
    if symbol not in _ELEMENT_TO_FAMILY:
        raise KeyError(f"No donor family for element {symbol!r}")
    return _ELEMENT_TO_FAMILY[symbol]


def full_path(node: str) -> List[str]:
    """
    Return the full coarsening path from *node* to ELEMENT_ANY, inclusive.

    Examples
    --------
    >>> full_path("N_pyridine")
    ['N_pyridine', 'N_aromatic_any', 'N_donor', 'element_any']
    """
    return [node] + ancestors(node)


def is_known(node: str) -> bool:
    """Return True if *node* is a registered hierarchy node."""
    return node == ELEMENT_ANY or node in _PARENT


def bucket(node: str) -> Optional[str]:
    """
    Return the mechanism-bucket ancestor of *node*, or None if the node's
    family has no bucket level (e.g. P, halogens, H, B, chalcogenides).


    Examples
    --------
    >>> bucket("N_pyridine")
    'N_aromatic_any'
    >>> bucket("P_phosphine")   # P has no bucket
    None
    >>> bucket("N_aromatic_any")   # already at bucket level
    'N_aromatic_any'

    """
    chain = full_path(node)
    # chain: [node, ..., family, element_any]
    # A bucket exists when len(chain) >= 4: [subclass, bucket, family, element_any]
    # or when node is already a bucket: [bucket, family, element_any]
    for candidate in chain:
        parent = _PARENT.get(candidate)
        if parent is None or parent == ELEMENT_ANY:
            # candidate is a family node; no bucket found
            return None
        grandparent = _PARENT.get(parent)
        if grandparent == ELEMENT_ANY or grandparent is None:
            # See exp.md for detailed rationale
            if _PARENT.get(_PARENT.get(candidate, ""), "") == ELEMENT_ANY:
                # candidate's grandparent is element_any -> candidate is bucket OR
                # it's a shallow subclass (no bucket).  They look the same structurally.
                # Distinguish by checking whether there are any nodes whose parent == candidate.
                # (A bucket has children; a shallow leaf does not.)
                if any(v == candidate for v in _PARENT.values()):
                    return candidate
                return None
        if grandparent is not None and _PARENT.get(grandparent) == ELEMENT_ANY:
            # candidate -> parent (bucket) -> family -> element_any
            return parent
    return None
