"""
visualizer.py
-------------
Interactive py3Dmol visualizer for tmQM metal complexes.
"""

from __future__ import annotations

from typing import Optional

import ipywidgets as widgets
import networkx as nx
import numpy as np
import py3Dmol
from IPython.display import display

from jacobik.core.types import Ligand, TMComplex
from jacobik.stats import key_builder
from jacobik.stats.parser import TMQMParser

# ---------------------------------------------------------------------------
# CPK colour palette
# ---------------------------------------------------------------------------

CPK = {
    "H": "#FFFFFF",
    "C": "#404040",
    "N": "#3050F8",
    "O": "#FF0D0D",
    "F": "#90E050",
    "P": "#FF8000",
    "S": "#FFFF30",
    "Cl": "#1FF01F",
    "Br": "#A62929",
    "I": "#940094",
    "B": "#FFB5B5",
    "Se": "#FFA100",
    "Sc": "#E6E6E6",
    "Ti": "#BFC2C7",
    "V": "#A6A6AB",
    "Cr": "#8A99C7",
    "Mn": "#9C7AC7",
    "Fe": "#E06633",
    "Co": "#F090A0",
    "Ni": "#50D050",
    "Cu": "#C88033",
    "Zn": "#7D80B0",
    "Y": "#94FFFF",
    "Zr": "#94E0E0",
    "Nb": "#73C2C9",
    "Mo": "#54B5B5",
    "Tc": "#3B9E9E",
    "Ru": "#248F8F",
    "Rh": "#0A7D8C",
    "Pd": "#006985",
    "Ag": "#C0C0C0",
    "Cd": "#FFD98F",
    "La": "#70D4FF",
    "Hf": "#4DC2FF",
    "Ta": "#4DA6FF",
    "W": "#2194D6",
    "Re": "#267DAB",
    "Os": "#266696",
    "Ir": "#175487",
    "Pt": "#D0D0E0",
    "Au": "#FFD123",
    "Hg": "#B8B8D0",
}


def _cpk(symbol: str) -> str:
    return CPK.get(symbol, "#FF69B4")


# ---------------------------------------------------------------------------
# Rigid-body colour palette for ligand_view
# ---------------------------------------------------------------------------

BODY_PALETTE = [
    "#E63946",
    "#457B9D",
    "#2A9D8F",
    "#E9C46A",
    "#F4A261",
    "#A8DADC",
    "#C77DFF",
    "#80B918",
    "#FF6B6B",
    "#4CC9F0",
    "#F77F00",
    "#7209B7",
    "#3A86FF",
    "#06D6A0",
    "#FFD166",
]


def _body_color(body_idx: int) -> str:
    return BODY_PALETTE[body_idx % len(BODY_PALETTE)]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + t * (b - a)


def _xyz(v: np.ndarray) -> dict:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


# ---------------------------------------------------------------------------
# Bond-order annotation helpers
# ---------------------------------------------------------------------------

# See exp.md for detailed rationale
_VIZ_BOND_THRESHOLD = 0.5

# Quantized bond orders assigned by mapping.py via SMILES
_BO_AROMATIC = 1.5
_BO_DOUBLE = 2.0
_BO_TRIPLE = 3.0


def _build_ligand_graph(mol: TMComplex, atom_subset: set[int] | None = None) -> nx.Graph:
    """
    Undirected graph of non-metal atoms with 'bo' edge attribute.
    Uses only bonds with quantized order >= _VIZ_BOND_THRESHOLD so that
    raw through-space interactions from the .BO file do not create
    spurious chords inside rings.
    """
    G = nx.Graph()
    for a in mol.atoms:
        if a.idx == mol.metal_idx:
            continue
        if atom_subset and a.idx not in atom_subset:
            continue
        G.add_node(a.idx)
    for b in mol.bonds:
        if b.i == mol.metal_idx or b.j == mol.metal_idx:
            continue
        if b.order < _VIZ_BOND_THRESHOLD:
            continue
        if atom_subset and (b.i not in atom_subset or b.j not in atom_subset):
            continue
        G.add_edge(b.i, b.j, bo=b.order)
    return G


def _find_aromatic_rings(
    mol: TMComplex,
    atom_subset: set[int] | None = None,
) -> list[list[int]]:
    """
    Return a list of aromatic rings (each a list of atom indices).
    """
    G = _build_ligand_graph(mol, atom_subset)
    aromatic: list[list[int]] = []
    for cycle in nx.minimum_cycle_basis(G):
        cycle_set = set(cycle)
        ring_bonds = [(u, v) for u, v in G.edges(cycle) if u in cycle_set and v in cycle_set]
        if not ring_bonds:
            continue
        bos = [G[u][v]["bo"] for u, v in ring_bonds]
        if all(abs(bo - _BO_AROMATIC) < 0.1 for bo in bos):
            aromatic.append(cycle)
    return aromatic


def _ring_circle_points(
    positions: list[np.ndarray],
    n_points: int = 20,
    shrink: float = 0.52,
) -> list[np.ndarray]:
    """
    Return n_points evenly-spaced 3D points forming a circle inscribed in the
    plane of the ring atoms, at shrink * mean_radius distance from the centroid.
    """
    pts = np.array(positions)
    centroid = pts.mean(axis=0)
    v1 = pts[0] - centroid
    v2 = pts[1] - centroid
    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-8:
        return []
    normal /= norm_len
    u = v1 / np.linalg.norm(v1)
    w = np.cross(normal, u)
    w_len = np.linalg.norm(w)
    if w_len < 1e-8:
        return []
    w /= w_len
    r = np.mean([np.linalg.norm(p - centroid) for p in pts]) * shrink
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    return [centroid + r * (np.cos(a) * u + np.sin(a) * w) for a in angles]


def _find_double_bonds(
    mol: TMComplex,
    aromatic_atom_set: set[int],
    atom_subset: set[int] | None = None,
) -> list[tuple[int, int]]:
    """
    Return (i, j) pairs for bonds that should get a double-bond marker.
    """
    doubles = []
    for b in mol.bonds:
        if b.i == mol.metal_idx or b.j == mol.metal_idx:
            continue
        if abs(b.order - _BO_DOUBLE) > 0.1:
            continue
        if atom_subset and (b.i not in atom_subset or b.j not in atom_subset):
            continue
        if b.i in aromatic_atom_set or b.j in aromatic_atom_set:
            continue
        doubles.append((b.i, b.j))
    return doubles


def _double_bond_offset(
    p1: np.ndarray,
    p2: np.ndarray,
    ring_centroid: np.ndarray | None,
    offset: float = 0.16,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the offset line for a double bond.
    If ring_centroid is given, offset inward (toward centroid).
    Otherwise offset perpendicular using the world-up vector.
    """
    if ring_centroid is not None:
        mid = (p1 + p2) * 0.5
        inward = ring_centroid - mid
        n = np.linalg.norm(inward)
        if n > 1e-6:
            inward /= n
            return p1 + inward * offset, p2 + inward * offset

    # Chain / exocyclic: perp to bond in the plane with world up
    bond_dir = p2 - p1
    bond_len = np.linalg.norm(bond_dir)
    if bond_len < 1e-6:
        return p1, p2
    bond_dir /= bond_len
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(bond_dir, up)) > 0.9:
        up = np.array([0.0, 0.0, 1.0])
    perp = np.cross(bond_dir, up)
    perp /= np.linalg.norm(perp)
    return p1 + perp * offset, p2 + perp * offset


# ---------------------------------------------------------------------------
# Render constants
# ---------------------------------------------------------------------------

STICK_R = 0.06
ATOM_SCALE = 0.22
DONOR_GHOST_R = 0.55
DASH_BEADS = 9
DASH_BEAD_R = 0.05

AROM_BEAD_R = 0.030  # radius of each dot in the aromatic circle
AROM_N_POINTS = 20  # dots per aromatic ring
AROM_SHRINK = 0.52  # circle radius as fraction of mean ring radius
AROM_COLOR = "#AAAAFF"  # pale blue-violet — distinct from bonds
AROM_OPACITY = 0.75

DBL_OFFSET = 0.16  # Å offset for double-bond second line
DBL_RADIUS = 0.030  # thinner cylinder for the second line
DBL_OPACITY = 0.85
DBL_DOT_R = 0.022  # dotted-line bead radius for double bonds
DBL_DOT_COUNT = 10  # dots per double-bond marker
DBL_DOT_OPACITY = 0.82

_ROTATABLE_BOND_COLOR = "#FF3EFF"
_DONOR_HIGHLIGHT_COLOR = "#FFD700"
_LABEL_OFFSET = 0.7


# ---------------------------------------------------------------------------
# Primitive drawing helpers
# ---------------------------------------------------------------------------


def _add_atom_sphere(view, pos, symbol, radius=ATOM_SCALE, color=None, opacity=1.0):
    view.addSphere(
        {
            "center": _xyz(pos),
            "radius": radius,
            "color": color or _cpk(symbol),
            "opacity": opacity,
        }
    )


def _add_stick(view, p1, p2, color1, color2, radius=STICK_R):
    mid = _lerp(p1, p2, 0.5)
    view.addCylinder(
        {"start": _xyz(p1), "end": _xyz(mid), "radius": radius, "color": color1, "opacity": 1.0}
    )
    view.addCylinder(
        {"start": _xyz(mid), "end": _xyz(p2), "radius": radius, "color": color2, "opacity": 1.0}
    )


def _add_dashed_line(view, p1, p2, color="#AAAAAA", n_beads=DASH_BEADS, bead_r=DASH_BEAD_R):
    for k in range(n_beads):
        t = (k + 0.5) / n_beads
        view.addSphere(
            {"center": _xyz(_lerp(p1, p2, t)), "radius": bead_r, "color": color, "opacity": 0.7}
        )


def _add_dotted_line(
    view, p1, p2, color="#AAAAAA", n_dots=DBL_DOT_COUNT, dot_r=DBL_DOT_R, opacity=DBL_DOT_OPACITY
):
    for k in range(n_dots):
        t = (k + 0.5) / n_dots
        view.addSphere(
            {"center": _xyz(_lerp(p1, p2, t)), "radius": dot_r, "color": color, "opacity": opacity}
        )


def _add_ghost_sphere(view, pos, color="#FFFFFF", radius=DONOR_GHOST_R):
    view.addSphere({"center": _xyz(pos), "radius": radius, "color": color, "opacity": 0.18})


def _add_donor_label(view, pos, text: str):
    label_pos = pos + np.array([0.0, _LABEL_OFFSET, 0.0])
    view.addLabel(
        text,
        {
            "position": _xyz(label_pos),
            "fontSize": 10,
            "fontColor": "white",
            "backgroundColor": "black",
            "backgroundOpacity": 0.45,
            "borderThickness": 0.0,
            "inFront": True,
            "showBackground": True,
        },
    )


# ---------------------------------------------------------------------------
# Aromatic ring circle
# ---------------------------------------------------------------------------


def _add_aromatic_circle(view, ring_atom_positions: list[np.ndarray], color=AROM_COLOR):
    """Draw a dotted circle of beads inscribed in the ring plane."""
    pts = _ring_circle_points(ring_atom_positions, n_points=AROM_N_POINTS, shrink=AROM_SHRINK)
    for p in pts:
        view.addSphere(
            {"center": _xyz(p), "radius": AROM_BEAD_R, "color": color, "opacity": AROM_OPACITY}
        )


# ---------------------------------------------------------------------------
# Double bond offset line
# ---------------------------------------------------------------------------


def _add_double_bond_line(view, p1, p2, color1, color2, ring_centroid=None):
    """Draw the second double-bond marker as a dotted offset line."""
    q1, q2 = _double_bond_offset(p1, p2, ring_centroid, offset=DBL_OFFSET)
    _add_dotted_line(view, q1, q2, color=AROM_COLOR)


# ---------------------------------------------------------------------------
# Shared annotation driver
# Called by both _render_complex and _render_ligand after bonds are drawn.
# ---------------------------------------------------------------------------


def _add_bond_order_annotations(
    view,
    mol: TMComplex,
    atom_map: dict[int, "Atom"],
    atom_subset: set[int] | None,
    color_fn,  # callable(atom_idx) -> hex color string
):
    """
    Add aromatic ring circles and double-bond offset lines.
    """
    # --- aromatic rings ---
    aromatic_rings = _find_aromatic_rings(mol, atom_subset)
    aromatic_atom_set: set[int] = set()
    for ring in aromatic_rings:
        for idx in ring:
            aromatic_atom_set.add(idx)
        positions = [atom_map[idx].position for idx in ring]
        _add_aromatic_circle(view, positions)

    # --- double bonds (quantized order == 2.0, not in aromatic ring) ---
    double_bonds = _find_double_bonds(mol, aromatic_atom_set, atom_subset)

    # Build ring centroid lookup for non-aromatic rings (for inward offset)
    G = _build_ligand_graph(mol, atom_subset)
    non_ar_ring_centroid: dict[frozenset, np.ndarray] = {}
    for cycle in nx.minimum_cycle_basis(G):
        cycle_set = frozenset(cycle)
        c = np.mean([atom_map[i].position for i in cycle], axis=0)
        non_ar_ring_centroid[cycle_set] = c

    for i, j in double_bonds:
        p1 = atom_map[i].position
        p2 = atom_map[j].position
        c1 = color_fn(i)
        c2 = color_fn(j)
        # Use ring centroid for inward offset if both atoms share a non-aromatic ring
        ring_centroid = None
        for cycle_set, c in non_ar_ring_centroid.items():
            if i in cycle_set and j in cycle_set:
                ring_centroid = c
                break
        _add_double_bond_line(view, p1, p2, c1, c2, ring_centroid)


# ---------------------------------------------------------------------------
# Complex renderer
# ---------------------------------------------------------------------------


def _render_complex(view, mol: TMComplex, show_hydrogens: bool = True):
    atom_map = {a.idx: a for a in mol.atoms}
    metal_pos = atom_map[mol.metal_idx].position

    # Coordination is read from mol.coord_sites — the single verified record.
    # (donor_labels still provides per-atom chemistry, e.g. which H atoms are
    # hydrides and therefore worth drawing.)
    donor_labels = getattr(mol, "donor_labels", {})

    # H atoms to show: only if they are donors (hydride)
    h_donor_set: set[int] = {
        idx for idx, lbl in donor_labels.items() if lbl.subclass == "H_hydride"
    }

    # --- atoms (suppress H unless donor) ---
    for a in mol.atoms:
        if not show_hydrogens and a.symbol == "H" and a.idx not in h_donor_set:
            continue
        _add_atom_sphere(view, a.position, a.symbol)

    # --- bonds ---
    metal_bond_pairs = {
        (min(b.i, b.j), max(b.i, b.j))
        for b in mol.bonds
        if b.i == mol.metal_idx or b.j == mol.metal_idx
    }
    for b in mol.bonds:
        key = (min(b.i, b.j), max(b.i, b.j))
        if key in metal_bond_pairs:
            continue
        if b.order < _VIZ_BOND_THRESHOLD:
            continue
        ai, aj = atom_map[b.i], atom_map[b.j]
        # Suppress bonds to non-donor H
        if not show_hydrogens and (
            (ai.symbol == "H" and b.i not in h_donor_set)
            or (aj.symbol == "H" and b.j not in h_donor_set)
        ):
            continue
        _add_stick(view, ai.position, aj.position, _cpk(ai.symbol), _cpk(aj.symbol))

    # --- bond-order annotations ---
    non_h_ligand = {
        a.idx
        for a in mol.atoms
        if a.idx != mol.metal_idx and (a.symbol != "H" or a.idx in h_donor_set)
    }
    _add_bond_order_annotations(
        view,
        mol,
        atom_map,
        atom_subset=non_h_ligand,
        color_fn=lambda idx: _cpk(atom_map[idx].symbol),
    )

    # See exp.md for detailed rationale
    for site in mol.coord_sites:
        pos = np.asarray(site.position, dtype=float)
        _add_dashed_line(view, pos, metal_pos, color="#CCCCCC")
        if site.is_haptic:
            _add_ghost_sphere(view, pos, color="#88DDFF", radius=DONOR_GHOST_R * 1.2)
        else:
            _add_ghost_sphere(view, pos, color="#FFFFFF")
        if site.label:
            _add_donor_label(view, pos, site.label)

        # Draw Hungarian matching ghostly anchor and vector
        if getattr(site, "anchor", None) is not None:
            anchor_pos = metal_pos + np.asarray(site.anchor, dtype=float)
            # Ghost sphere for anchor
            view.addSphere(
                {
                    "center": _xyz(anchor_pos),
                    "radius": 0.45 * (1.2 if site.is_haptic else 1.0),
                    "color": "#FFA500",  # warm orange/gold color for anchor
                    "opacity": 0.25,
                }
            )
            # Vector arrow from donor position to anchor
            view.addArrow(
                {
                    "start": _xyz(pos),
                    "end": _xyz(anchor_pos),
                    "radius": 0.05,
                    "color": "#FFA500",
                    "opacity": 0.6,
                }
            )


# ---------------------------------------------------------------------------
# Ligand renderer
# ---------------------------------------------------------------------------


def _render_ligand(view, mol: TMComplex, lig: Ligand, show_hydrogens: bool = True):
    atom_map = {a.idx: a for a in mol.atoms}
    decomp = lig.decomposition

    body_of: dict[int, int] = {}
    if decomp is not None:
        body_of = decomp.body_of

    def _atom_color(idx: int) -> str:
        b = body_of.get(idx)
        return _body_color(b) if b is not None else _cpk(atom_map[idx].symbol)

    pivot_set: set[tuple[int, int]] = set()
    if decomp is not None:
        for i, j in decomp.pivot_bonds:
            pivot_set.add((min(i, j), max(i, j)))

    donor_labels = getattr(mol, "donor_labels", {})
    h_donor_set: set[int] = {
        idx for idx, lbl in donor_labels.items() if lbl.subclass == "H_hydride"
    }

    lig_set = set(lig.atom_indices)
    lig_bonds = [
        b for b in mol.bonds if b.i in lig_set and b.j in lig_set and b.order >= _VIZ_BOND_THRESHOLD
    ]

    # --- atoms (suppress non-donor H) ---
    for idx in lig.atom_indices:
        a = atom_map[idx]
        if not show_hydrogens and a.symbol == "H" and idx not in h_donor_set:
            continue
        _add_atom_sphere(view, a.position, a.symbol, color=_atom_color(idx))

    # --- bonds ---
    for b in lig_bonds:
        ai, aj = atom_map[b.i], atom_map[b.j]
        if not show_hydrogens and (
            (ai.symbol == "H" and b.i not in h_donor_set)
            or (aj.symbol == "H" and b.j not in h_donor_set)
        ):
            continue
        key = (min(b.i, b.j), max(b.i, b.j))
        ci = _atom_color(b.i)
        cj = _atom_color(b.j)
        if key in pivot_set:
            view.addCylinder(
                {
                    "start": _xyz(atom_map[b.i].position),
                    "end": _xyz(atom_map[b.j].position),
                    "radius": STICK_R,
                    "color": _ROTATABLE_BOND_COLOR,
                    "opacity": 1.0,
                }
            )
        else:
            _add_stick(view, ai.position, aj.position, ci, cj, radius=STICK_R)

    # --- bond-order annotations ---
    non_h_lig = {idx for idx in lig_set if atom_map[idx].symbol != "H" or idx in h_donor_set}
    _add_bond_order_annotations(
        view,
        mol,
        atom_map,
        atom_subset=non_h_lig,
        color_fn=_atom_color,
    )

    # --- donor ghost spheres + labels ---
    # Collect all haptic donor indices across all groups on this ligand
    haptic_donors: set[int] = set()
    for h in lig.haptic_groups:
        haptic_donors.update(h.donor_indices)

    for d in lig.donor_atoms:
        if d in haptic_donors:
            continue
        pos = atom_map[d].position
        _add_ghost_sphere(view, pos, color=_DONOR_HIGHLIGHT_COLOR, radius=DONOR_GHOST_R)
        lbl = donor_labels.get(d)
        if lbl is not None:
            _add_donor_label(view, pos, lbl.short())

    # One centroid ghost + label per haptic group
    for h in lig.haptic_groups:
        _add_ghost_sphere(view, h.centroid, color="#00FFFF", radius=DONOR_GHOST_R * 1.3)
        first_donor = h.donor_indices[0]
        lbl = donor_labels.get(first_donor)
        if lbl is not None:
            _add_donor_label(view, h.centroid, lbl.short())


# ---------------------------------------------------------------------------
# View factory
# ---------------------------------------------------------------------------


def _make_view(width=600, height=450):
    v = py3Dmol.view(width=width, height=height)
    v.setBackgroundColor("#1A1A2E")
    return v


def _finalise(view, zoom=True):
    if zoom:
        view.zoomTo()
    view.rotate(30, "y")
    view.rotate(15, "x")


# ---------------------------------------------------------------------------
# Quantity annotation constants
# ---------------------------------------------------------------------------

# Per-quantity annotation colours (distinct, legible on dark background)
_Q_COLOR: dict[str, str] = {
    "Q1": "#00E5FF",  # cyan       — bond lengths
    "Q2": "#FF9500",  # orange     — D–M–D angles
    "Q3": "#00FF88",  # green      — approach angles
    "Q4": "#FFE500",  # yellow     — pyramidalization
    "Q5": "#FF3EFF",  # magenta    — torsions (same as rotatable bonds)
}

_Q_LABEL_FONT = 9  # px — slightly smaller than donor labels so they don't crowd
_Q_ARC_BEADS = 14  # beads in a D–M–D arc
_Q_ARC_BEAD_R = 0.035
_Q_MARKER_R = 0.18  # radius of the coloured dot at each annotated site


def _add_qty_label(
    view, pos: np.ndarray, text: str, color: str, offset_dir: np.ndarray | None = None
):
    """
    Draw a small coloured floating label near pos.
    """
    label_pos = pos.copy()
    if offset_dir is not None:
        label_pos = label_pos + offset_dir * 0.5
    else:
        label_pos = label_pos + np.array([0.0, 0.8, 0.0])
    view.addLabel(
        text,
        {
            "position": _xyz(label_pos),
            "fontSize": _Q_LABEL_FONT,
            "fontColor": color,
            "backgroundColor": "#0A0A18",
            "backgroundOpacity": 0.65,
            "borderThickness": 0.0,
            "inFront": True,
            "showBackground": True,
        },
    )


def _add_qty_dot(view, pos: np.ndarray, color: str):
    """Small solid dot marking the annotated position."""
    view.addSphere(
        {
            "center": _xyz(pos),
            "radius": _Q_MARKER_R,
            "color": color,
            "opacity": 0.85,
        }
    )


def _arc_points(
    p_a: np.ndarray,
    p_m: np.ndarray,
    p_b: np.ndarray,
    n: int = _Q_ARC_BEADS,
    r_frac: float = 0.45,
) -> list[np.ndarray]:
    """
    Return n points along a circular arc from p_a to p_b, centered at p_m,
    at r_frac times the mean donor–metal distance.  Used to draw D–M–D angle
    arcs as bead chains.
    """
    va = p_a - p_m
    vb = p_b - p_m
    la = np.linalg.norm(va)
    lb = np.linalg.norm(vb)
    if la < 1e-8 or lb < 1e-8:
        return []
    ua = va / la
    ub = vb / lb
    r = r_frac * (la + lb) * 0.5

    # Gram–Schmidt to get an in-plane perpendicular to ua
    proj = np.dot(ub, ua) * ua
    perp = ub - proj
    perp_len = np.linalg.norm(perp)
    if perp_len < 1e-8:
        return []
    perp /= perp_len

    # Angle of arc
    cos_t = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    theta = np.arccos(cos_t)

    pts = []
    for i in range(n):
        t = theta * i / (n - 1)
        pts.append(p_m + r * (np.cos(t) * ua + np.sin(t) * perp))
    return pts


def _render_quantities(
    view,
    mol: TMComplex,
    show: set[str],
):
    """
    Overlay quantity annotations on a view that already has the base complex
    rendered.  Only quantities whose name appears in *show* are drawn.


    Parameters
    ----------
    show : subset of {"Q1", "Q2", "Q3", "Q4", "Q5"}

    """
    atom_map = {a.idx: a for a in mol.atoms}
    metal_pos = np.asarray(atom_map[mol.metal_idx].position, dtype=float)
    metal_idx = mol.metal_idx

    # Heavy-atom ligand-graph neighbor table (shared by Q3 and Q4)
    heavy_nbs_map: dict[int, list[int]] = {a.idx: [] for a in mol.atoms}
    for b in mol.bonds:
        if b.i == metal_idx or b.j == metal_idx or b.order < 0.5:
            continue
        ai, aj = atom_map.get(b.i), atom_map.get(b.j)
        if ai and aj and ai.symbol != "H" and aj.symbol != "H":
            heavy_nbs_map[b.i].append(b.j)
            heavy_nbs_map[b.j].append(b.i)

    # ------------------------------------------------------------------ Q1
    if "Q1" in show:
        color = _Q_COLOR["Q1"]
        for site in mol.coord_sites:
            pos = np.asarray(site.position, dtype=float)
            dist = float(np.linalg.norm(pos - metal_pos))
            subclass = _subclass_for_site(site, mol)
            _add_qty_dot(view, pos, color)
            _add_qty_label(view, pos, f"{subclass}\n{dist:.2f} Å", color)

    # ------------------------------------------------------------------ Q2
    if "Q2" in show:
        color = _Q_COLOR["Q2"]
        sites = mol.coord_sites
        for i in range(len(sites)):
            for j in range(i + 1, len(sites)):
                pa = np.asarray(sites[i].position, dtype=float)
                pb = np.asarray(sites[j].position, dtype=float)
                va = pa - metal_pos
                vb = pb - metal_pos
                na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                if na < 1e-8 or nb < 1e-8:
                    continue
                cos_t = float(np.clip(np.dot(va / na, vb / nb), -1.0, 1.0))
                angle = float(np.degrees(np.arccos(cos_t)))

                # Arc at metal, between the two donor directions
                arc = _arc_points(pa, metal_pos, pb)
                for p in arc:
                    view.addSphere(
                        {
                            "center": _xyz(p),
                            "radius": _Q_ARC_BEAD_R,
                            "color": color,
                            "opacity": 0.7,
                        }
                    )

                # Label at arc midpoint, offset outward
                if arc:
                    mid = arc[len(arc) // 2]
                    away = mid - metal_pos
                    away_len = np.linalg.norm(away)
                    offset_dir = away / away_len if away_len > 1e-8 else None
                    _add_qty_label(view, mid, f"{angle:.1f}°", color, offset_dir=offset_dir)

    # See exp.md for detailed rationale
    if "Q3" in show:
        c_vM = "#66DDFF"  # pale cyan  — v_M arm
        c_vC = "#00FF88"  # green      — v_C / normal arm
        c_arc = _Q_COLOR["Q3"]
        arm_len = 1.2  # Å — length to draw each vector arm

        for site in mol.coord_sites:
            try:
                if not site.is_haptic:
                    # ── kappa donor ──────────────────────────────────────────
                    donor_idx = site.donor_indices[0]
                    donor_pos = np.asarray(atom_map[donor_idx].position, dtype=float)
                    nbs = heavy_nbs_map.get(donor_idx, [])
                    if not nbs:
                        continue

                    # Substituent centroid
                    centroid = np.mean(
                        [np.asarray(atom_map[n].position, dtype=float) for n in nbs],
                        axis=0,
                    )

                    v_M_raw = metal_pos - donor_pos
                    v_C_raw = centroid - donor_pos
                    len_M = np.linalg.norm(v_M_raw)
                    len_C = np.linalg.norm(v_C_raw)
                    if len_M < 1e-8 or len_C < 1e-8:
                        continue

                    # Unit vectors, scaled to arm_len for drawing
                    u_M = v_M_raw / len_M
                    u_C = v_C_raw / len_C
                    tip_M = donor_pos + u_M * arm_len
                    tip_C = donor_pos + u_C * arm_len

                    # v_M arm (pale cyan cylinder)
                    view.addCylinder(
                        {
                            "start": _xyz(donor_pos),
                            "end": _xyz(tip_M),
                            "radius": 0.045,
                            "color": c_vM,
                            "opacity": 0.85,
                        }
                    )
                    # v_C arm (green cylinder, slightly thinner + dashed feel via opacity)
                    view.addCylinder(
                        {
                            "start": _xyz(donor_pos),
                            "end": _xyz(tip_C),
                            "radius": 0.04,
                            "color": c_vC,
                            "opacity": 0.85,
                        }
                    )

                    # Arc between u_M and u_C, centred at donor_pos
                    cos_t = float(np.clip(np.dot(u_M, u_C), -1.0, 1.0))
                    angle = float(np.degrees(np.arccos(cos_t)))
                    arc_pts = _arc_points(tip_M, donor_pos, tip_C, r_frac=1.0)
                    for p in arc_pts:
                        view.addSphere(
                            {
                                "center": _xyz(p),
                                "radius": _Q_ARC_BEAD_R,
                                "color": c_arc,
                                "opacity": 0.8,
                            }
                        )

                    # Label at arc midpoint, offset along bisector
                    if arc_pts:
                        mid = arc_pts[len(arc_pts) // 2]
                        bisect = mid - donor_pos
                        bl = np.linalg.norm(bisect)
                        off = bisect / bl if bl > 1e-8 else None
                        subclass = _subclass_for_site(site, mol)
                        _add_qty_label(
                            view, mid, f"Q3 {subclass}: {angle:.1f}°", c_arc, offset_dir=off
                        )

                else:
                    # ── haptic site (ring tilt) ───────────────────────────────
                    face_idxs = [
                        idx
                        for idx in site.donor_indices
                        if idx != metal_idx
                        and atom_map.get(idx) is not None
                        and atom_map[idx].symbol != "H"
                    ]
                    if len(face_idxs) < 3:
                        continue

                    face_pts = np.array([atom_map[idx].position for idx in face_idxs], dtype=float)
                    c_pos = np.asarray(site.position, dtype=float)  # centroid

                    # Best-fit ring-plane normal (SVD)
                    centered = face_pts - face_pts.mean(axis=0)
                    _, _, Vt = np.linalg.svd(centered)
                    normal = Vt[-1]

                    # Metal→centroid vector
                    mc_raw = c_pos - metal_pos
                    mc_len = np.linalg.norm(mc_raw)
                    if mc_len < 1e-8:
                        continue
                    mc_unit = mc_raw / mc_len

                    # Orient normal toward metal so the angle is the deviation
                    # from "metal directly overhead" (tilt from ideal)
                    if np.dot(normal, mc_unit) < 0:
                        normal = -normal

                    cos_t = float(np.clip(abs(np.dot(normal, mc_unit)), 0.0, 1.0))
                    tilt = float(np.degrees(np.arccos(cos_t)))

                    # Draw: ring normal arm from centroid
                    tip_n = c_pos + normal * arm_len
                    view.addCylinder(
                        {
                            "start": _xyz(c_pos),
                            "end": _xyz(tip_n),
                            "radius": 0.045,
                            "color": c_vC,
                            "opacity": 0.85,
                        }
                    )
                    # Draw: metal–centroid arm from centroid (toward metal)
                    tip_mc = c_pos + mc_unit * arm_len
                    view.addCylinder(
                        {
                            "start": _xyz(c_pos),
                            "end": _xyz(tip_mc),
                            "radius": 0.04,
                            "color": c_vM,
                            "opacity": 0.85,
                        }
                    )

                    # Arc between normal and mc_unit at centroid
                    arc_pts = _arc_points(tip_n, c_pos, tip_mc, r_frac=1.0)
                    for p in arc_pts:
                        view.addSphere(
                            {
                                "center": _xyz(p),
                                "radius": _Q_ARC_BEAD_R,
                                "color": c_arc,
                                "opacity": 0.8,
                            }
                        )

                    if arc_pts:
                        mid = arc_pts[len(arc_pts) // 2]
                        bisect = mid - c_pos
                        bl = np.linalg.norm(bisect)
                        off = bisect / bl if bl > 1e-8 else None
                        subclass = _subclass_for_site(site, mol)
                        _add_qty_label(
                            view, mid, f"Q3 {subclass} tilt: {tilt:.1f}°", c_arc, offset_dir=off
                        )

            except Exception as exc:
                log.debug("Q3 render site %s: %s", site, exc)

    # ------------------------------------------------------------------ Q4
    # Delegate entirely to key_builder.q4_pyramidalization() so the
    # visualizer always shows exactly the same value as the extractor.
    # (mean pairwise angle = Σθ / C(k+1,2), normalised since v1.4.2)
    if "Q4" in show:
        color = _Q_COLOR["Q4"]
        q4_data = key_builder.q4_pyramidalization(mol)
        # key_builder returns one (key, value) per site in coord_sites order
        # Build a lookup: donor_idx -> mean_angle
        atom_map_local = {a.idx: a for a in mol.atoms}

        # See exp.md for detailed rationale
        eligible_sites = [
            s
            for s in mol.coord_sites
            if not s.is_haptic and len(heavy_nbs_map.get(s.donor_indices[0], [])) >= 2
        ]
        for site, ((subclass, geom), mean_angle) in zip(eligible_sites, q4_data):
            try:
                donor_idx = site.donor_indices[0]
                donor_pos = np.asarray(atom_map_local[donor_idx].position, dtype=float)
                _add_qty_dot(view, donor_pos, color)
                _add_qty_label(
                    view,
                    donor_pos,
                    f"Q4 {subclass}: {mean_angle:.1f}°",
                    color,
                    offset_dir=np.array([0.0, -1.0, 0.0]),
                )
            except Exception as exc:
                log.debug("Q4 render site %s: %s", site, exc)

    # ------------------------------------------------------------------ Q5
    if "Q5" in show:
        color = _Q_COLOR["Q5"]
        q5_by_bond: dict[tuple[int, int], float] = {}
        for lig in mol.ligands:
            for chain in lig.pivot_chains:
                for joint in chain.joints:
                    bk = (min(joint.atom_i, joint.atom_j), max(joint.atom_i, joint.atom_j))
                    if bk in q5_by_bond:
                        continue
                    if joint.atom_i not in atom_map or joint.atom_j not in atom_map:
                        continue
                    pj = np.asarray(atom_map[joint.atom_i].position, dtype=float)
                    pk = np.asarray(atom_map[joint.atom_j].position, dtype=float)
                    heavy_nbs_j = _heavy_nbs(mol, joint.atom_i, exclude=joint.atom_j)
                    heavy_nbs_k = _heavy_nbs(mol, joint.atom_j, exclude=joint.atom_i)
                    if not heavy_nbs_j or not heavy_nbs_k:
                        continue
                    i_idx = max(
                        heavy_nbs_j, key=lambda x: (_ATOMIC_NUM.get(atom_map[x].symbol, 0), -x)
                    )
                    l_idx = max(
                        heavy_nbs_k, key=lambda x: (_ATOMIC_NUM.get(atom_map[x].symbol, 0), -x)
                    )
                    pi = np.asarray(atom_map[i_idx].position, dtype=float)
                    pl = np.asarray(atom_map[l_idx].position, dtype=float)
                    dih = key_builder._dihedral_deg(pi, pj, pk, pl)
                    if dih is not None:
                        q5_by_bond[bk] = dih

        for bk, dih in q5_by_bond.items():
            i_idx, j_idx = bk
            if i_idx not in atom_map or j_idx not in atom_map:
                continue
            pj = np.asarray(atom_map[i_idx].position, dtype=float)
            pk = np.asarray(atom_map[j_idx].position, dtype=float)
            mid = (pj + pk) * 0.5
            bond_dir = pk - pj
            bond_len = np.linalg.norm(bond_dir)
            if bond_len > 1e-8:
                perp = np.cross(bond_dir / bond_len, np.array([0.0, 1.0, 0.0]))
                pl_len = np.linalg.norm(perp)
                offset_dir = perp / pl_len if pl_len > 1e-8 else None
            else:
                offset_dir = None
            _add_qty_label(view, mid, f"φ={dih:.0f}°", color, offset_dir=offset_dir)


def _subclass_for_site(site, mol: TMComplex) -> str:
    """Return the donor subclass string for a CoordinationSite."""
    rep = min(site.donor_indices)
    dl = mol.donor_labels.get(rep)
    return dl.subclass if dl is not None else site.label


def _heavy_nbs(mol: TMComplex, atom_idx: int, exclude: int) -> list[int]:
    """Return heavy-atom neighbor indices of atom_idx, excluding exclude and H."""
    atom_map = {a.idx: a for a in mol.atoms}
    result = []
    for b in mol.bonds:
        if b.order < 0.5:
            continue
        nb = None
        if b.i == atom_idx and b.j != exclude:
            nb = b.j
        elif b.j == atom_idx and b.i != exclude:
            nb = b.i
        if nb is not None and nb != mol.metal_idx:
            a = atom_map.get(nb)
            if a and a.symbol != "H":
                result.append(nb)
    return result


# Atomic number lookup reused from key_builder for neighbour canonicalisation
_ATOMIC_NUM: dict[str, int] = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Te": 52,
    "I": 53,
}


def _quantities_info_html(
    mol: TMComplex,
    results: dict[str, list],
    show: set[str],
) -> str:
    """Info panel for quantities_view: one row per quantity with count and range."""
    rows = ""

    qty_meta = [
        ("Q1", "Bond lengths", "Å", "#00E5FF"),
        ("Q2", "D–M–D angles", "°", "#FF9500"),
        ("Q3", "Approach angles", "°", "#00FF88"),
        ("Q4", "Pyramidalization", "°", "#FFE500"),
        ("Q5", "Torsions", "°", "#FF3EFF"),
        ("Q6", "Steric contacts", "Å", "#AAAAAA"),
    ]

    for qname, label, unit, color in qty_meta:
        data = results.get(qname, [])
        if not data:
            val_str = "—"
        else:
            vals = [v for _, v in data]
            val_str = f"{len(vals)} obs  [{min(vals):.2f} – {max(vals):.2f} {unit}]"
        marker = "●" if qname in show else "○"
        rows += (
            f"<span style='color:{color}'>{marker} {qname}</span>"
            f"<span style='color:#888'> {label}:</span>"
            f"<span style='color:#E8E8E8'> {val_str}</span><br/>"
        )

    return f"<div style='{_PANEL_STYLE}'>{rows}</div>"


# ---------------------------------------------------------------------------
# Public visualizer class
# ---------------------------------------------------------------------------


class ComplexVisualizer:
    """
    Interactive tmQM visualizer built on py3Dmol + ipywidgets.

    Parameters
    ----------
    data_dir : path to the tmQM data directory (contains xyz/, bo/, tmQM_y.csv)
    width    : viewer width in pixels
    height   : viewer height in pixels
    """

    def __init__(self, data_dir: str, width: int = 650, height: int = 480):
        self.data_dir = data_dir
        self.width = width
        self.height = height
        self.parser = TMQMParser(data_dir)

    def complex_view(self, n: int = 20, split: str = "X1"):
        """Step through the first n complexes in split with prev/next buttons."""
        print(f"Loading {n} complexes from {split}…")
        data = self.parser.load_split(split, limit=n)
        codes = list(data.keys())
        if not codes:
            print("No complexes loaded.")
            return

        state = {"idx": 0}
        out = widgets.Output()
        lbl = widgets.Label(value="")
        btn_prev = widgets.Button(description="◀ Prev", layout=widgets.Layout(width="90px"))
        btn_next = widgets.Button(description="Next ▶", layout=widgets.Layout(width="90px"))
        info_html = widgets.HTML(value="")

        def _render():
            mol = data[codes[state["idx"]]]
            lbl.value = (
                f"  {state['idx'] + 1} / {len(codes)}  —  "
                f"{mol.csd_code}  |  {mol.metal_symbol}  |  "
                f"{mol.stoichiometry}"
            )
            info_html.value = _complex_info_html(mol)
            with out:
                out.clear_output(wait=True)
                v = _make_view(self.width, self.height)
                _render_complex(v, mol)
                _finalise(v)
                display(v)

        def on_prev(_):
            if state["idx"] > 0:
                state["idx"] -= 1
                _render()

        def on_next(_):
            if state["idx"] < len(codes) - 1:
                state["idx"] += 1
                _render()

        btn_prev.on_click(on_prev)
        btn_next.on_click(on_next)
        nav = widgets.HBox([btn_prev, lbl, btn_next])
        ui = widgets.VBox(
            [nav, out, info_html],
            layout=widgets.Layout(
                border="1px solid #333", padding="8px", background_color="#0F0F1A"
            ),
        )
        display(ui)
        _render()

    def ligand_view(self, mol: TMComplex):
        """Step through each ligand of a parsed TMComplex."""
        ligands = mol.ligands
        if not ligands:
            print("No ligands found in this complex.")
            return

        state = {"idx": 0}
        out = widgets.Output()
        lbl = widgets.Label(value="")
        btn_prev = widgets.Button(description="◀ Prev", layout=widgets.Layout(width="90px"))
        btn_next = widgets.Button(description="Next ▶", layout=widgets.Layout(width="90px"))
        info_html = widgets.HTML(value="")

        def _render():
            lig = ligands[state["idx"]]
            lbl.value = (
                f"  {state['idx'] + 1} / {len(ligands)}  —  "
                f"Ligand {lig.ligand_id}  |  "
                f"{len(lig.atom_indices)} atoms  |  "
                f"donors={lig.donor_atoms}"
            )
            info_html.value = _ligand_info_html(mol, lig)
            with out:
                out.clear_output(wait=True)
                v = _make_view(self.width, self.height)
                _render_ligand(v, mol, lig)
                _finalise(v)
                display(v)

        def on_prev(_):
            if state["idx"] > 0:
                state["idx"] -= 1
                _render()

        def on_next(_):
            if state["idx"] < len(ligands) - 1:
                state["idx"] += 1
                _render()

        btn_prev.on_click(on_prev)
        btn_next.on_click(on_next)
        title = widgets.HTML(
            value=f"<b style='color:#AAD4FF'>Ligands of {mol.csd_code} ({mol.metal_symbol})</b>"
        )
        nav = widgets.HBox([btn_prev, lbl, btn_next])
        ui = widgets.VBox(
            [title, nav, out, info_html],
            layout=widgets.Layout(border="1px solid #333", padding="8px"),
        )
        display(ui)
        _render()

    def quantities_view(
        self,
        mol: TMComplex,
        show: Optional[set[str]] = None,
    ):
        """
        Render a TMComplex with extracted geometric quantities overlaid.


        Parameters
        ----------
        mol  : a fully-assembled TMComplex (coord_sites and donor_labels
               must be populated; pivot_chains for Q5).
        show : set of quantity names to overlay, e.g. {"Q1", "Q3"}.
               Defaults to all five drawable quantities.

        """
        if show is None:
            show = {"Q1", "Q2", "Q3", "Q4", "Q5"}

        # Compute all keys once so the info panel can show counts for every
        # quantity regardless of which ones are being drawn.
        all_results = key_builder.all_keys(mol)

        v = _make_view(self.width, self.height)
        _render_complex(v, mol)
        _render_quantities(v, mol, show)
        _finalise(v)

        info = widgets.HTML(value=_quantities_info_html(mol, all_results, show))

        # Quantity toggle checkboxes
        checkboxes = []
        for qname, label, color in [
            ("Q1", "Bond lengths", _Q_COLOR["Q1"]),
            ("Q2", "D–M–D angles", _Q_COLOR["Q2"]),
            ("Q3", "Approach angles", _Q_COLOR["Q3"]),
            ("Q4", "Pyramidalization", _Q_COLOR["Q4"]),
            ("Q5", "Torsions", _Q_COLOR["Q5"]),
        ]:
            cb = widgets.Checkbox(
                value=(qname in show),
                description=f"{qname} {label}",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="180px"),
            )
            checkboxes.append((qname, cb))

        out = widgets.Output()

        def _redraw(*_):
            active = {qname for qname, cb in checkboxes if cb.value}
            with out:
                out.clear_output(wait=True)
                v2 = _make_view(self.width, self.height)
                _render_complex(v2, mol)
                _render_quantities(v2, mol, active)
                _finalise(v2)
                display(v2)
            info.value = _quantities_info_html(mol, all_results, active)

        for _, cb in checkboxes:
            cb.observe(_redraw, names="value")

        cb_row = widgets.HBox([cb for _, cb in checkboxes], layout=widgets.Layout(flex_wrap="wrap"))
        title = widgets.HTML(
            value=(
                f"<b style='color:#AAD4FF'>{mol.csd_code} — "
                f"{mol.metal_symbol} — {mol.coord_geom or 'unknown'} "
                f"(CN={mol.coordination_number()})</b>"
            )
        )

        with out:
            display(v)

        ui = widgets.VBox(
            [title, cb_row, out, info],
            layout=widgets.Layout(
                border="1px solid #333", padding="8px", background_color="#0F0F1A"
            ),
        )
        display(ui)

    def print_keys(
        self,
        mol: TMComplex,
        quantities: Optional[set[str]] = None,
    ):
        """
        Print all extracted (key, value) pairs for a TMComplex to stdout,
        one line per observation, grouped by quantity, with timing.


        Parameters
        ----------
        mol        : fully-assembled TMComplex
        quantities : which quantities to print, e.g. {"Q1", "Q4"}.
                     Defaults to all six (Q1–Q6).

        Output format (per line)
        ------------------------
            Q1  ('N_pyridine', 'Fe', 'octahedral')  =  2.0421 Å
            Q2  ('octahedral', 'chelate_short', 'Fe', 'N_amine', 'N_pyridine')  =  78.3°
            ...

        Timing is printed at the end showing wall time for the full
        key extraction pass.

        """
        import time

        if quantities is None:
            quantities = {"Q1", "Q2", "Q3", "Q4", "Q5", "Q6"}

        units = {"Q1": "Å", "Q2": "°", "Q3": "°", "Q4": "°", "Q5": "°", "Q6": "Å"}

        print(f"\n{'─' * 70}")
        print(
            f"  {mol.csd_code}  |  {mol.metal_symbol}  |  "
            f"{mol.coord_geom or 'unknown'}  |  CN={mol.coordination_number()}"
        )
        print(f"{'─' * 70}")

        t0 = time.perf_counter()
        results = key_builder.all_keys(mol)
        elapsed = time.perf_counter() - t0

        for qname in ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6"):
            if qname not in quantities:
                continue
            data = results.get(qname, [])
            unit = units[qname]
            print(f"\n  {qname}  ({len(data)} observations)")
            if not data:
                print("    (none)")
                continue
            for key, val in data:
                print(f"    {str(key):<60s}  =  {val:.4f} {unit}")

        total = sum(len(v) for v in results.values())
        print(f"\n{'─' * 70}")
        print(f"  {total} observations total  |  extracted in {elapsed * 1000:.2f} ms")
        print(f"{'─' * 70}\n")


_PANEL_STYLE = (
    "font-family: monospace; font-size: 12px; "
    "background: #0F0F1A; color: #C8D8E8; "
    "padding: 8px 12px; border-radius: 4px; "
    "margin-top: 4px; line-height: 1.7;"
)


def _row(label, value, color="#AAD4FF"):
    return (
        f"<span style='color:{color}'>{label}:</span> "
        f"<span style='color:#E8E8E8'>{value}</span><br/>"
    )


def _complex_info_html(mol: TMComplex) -> str:
    cn = len(mol.coord_sites)
    haptic_str = f"{len(mol.haptic_sites)}× haptic" if mol.haptic_sites else "none"
    qm_rows = ""
    if mol.hl_gap is not None:
        qm_rows += _row("HL gap", f"{mol.hl_gap:.4f} Ha", "#90EE90")
    if mol.metal_charge is not None:
        qm_rows += _row("Metal q", f"{mol.metal_charge:+.3f}", "#90EE90")
    if mol.dipole_moment is not None:
        qm_rows += _row("Dipole", f"{mol.dipole_moment:.3f} D", "#90EE90")
    return (
        f"<div style='{_PANEL_STYLE}'>"
        + _row("CSD code", mol.csd_code)
        + _row("Stoich", mol.stoichiometry)
        + _row("Charge / S", f"{mol.charge:+d} / {mol.spin_state}")
        + _row("Metal", f"{mol.metal_symbol} (atom {mol.metal_idx})")
        + _row("Coord bonds", f"{cn}  ({haptic_str})")
        + _row("Ligands", str(len(mol.ligands)))
        + qm_rows
        + "</div>"
    )


def _ligand_info_html(mol: TMComplex, lig: Ligand) -> str:
    decomp = lig.decomposition
    n_piv = len(decomp.pivot_bonds) if decomp else 0
    n_body = len(decomp.rigid_bodies) if decomp else 0

    chain_rows = ""
    for k, chain in enumerate(lig.pivot_chains):
        mode = f"eta-{len(chain.donor_atoms)}" if chain.is_haptic else "κ"
        atom_map = {a.idx: a for a in mol.atoms}
        joints_str = (
            "  →  ".join(
                f"{atom_map[j.atom_i].symbol}{j.atom_i}–{atom_map[j.atom_j].symbol}{j.atom_j}"
                for j in chain.joints
            )
            or "(no joints)"
        )
        chain_rows += _row(f"Chain {k} ({mode})", joints_str, "#F4D03F")

    body_legend = ""
    if decomp:
        for k, body in enumerate(decomp.rigid_bodies):
            atom_map = {a.idx: a for a in mol.atoms}
            syms = sorted(set(atom_map[a].symbol for a in body))
            col = _body_color(k)
            body_legend += (
                f"<span style='color:{col}'>■</span> "
                f"<span style='color:#C8D8E8'>body {k}: "
                f"{len(body)} atoms  {syms}</span><br/>"
            )

    haptic_str = ""
    for k, h in enumerate(lig.haptic_groups):
        haptic_str += _row(
            f"Haptic {k}",
            f"eta-{h.haptic_order}  donors={h.donor_indices}  "
            f"centroid=[{', '.join(f'{x:.2f}' for x in h.centroid)}]",
            "#00FFFF",
        )

    return (
        f"<div style='{_PANEL_STYLE}'>"
        + _row("Ligand", f"{lig.ligand_id}  ({len(lig.atom_indices)} atoms)")
        + _row("Donors", str(lig.donor_atoms))
        + _row("Denticity", str(lig.denticity))
        + haptic_str
        + _row("Rigid bodies", str(n_body))
        + _row("Pivot joints", str(n_piv))
        + chain_rows
        + (
            "<br/><b style='color:#AAD4FF'>Rigid body legend</b><br/>" + body_legend
            if body_legend
            else ""
        )
        + "</div>"
    )
