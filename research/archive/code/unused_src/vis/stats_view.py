"""
stats_view.py
-------------
py3Dmol drawing helpers (complex, bond-order annotations, panel styling) used by vis/visualizer.py.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import py3Dmol

from jacobik.core.types import TMComplex

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
# Geometry helpers
# ---------------------------------------------------------------------------


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + t * (b - a)


def _xyz(v: np.ndarray) -> dict:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


# ---------------------------------------------------------------------------
# Bond-order annotation helpers
# ---------------------------------------------------------------------------

_VIZ_BOND_THRESHOLD = 0.5

# Quantized bond orders assigned by mapping.py via SMILES
_BO_AROMATIC = 1.5
_BO_DOUBLE = 2.0


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
DBL_DOT_R = 0.022  # dotted-line bead radius for double bonds
DBL_DOT_COUNT = 10  # dots per double-bond marker
DBL_DOT_OPACITY = 0.82

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
# Bond-order annotation driver (called by _render_complex after the bonds are drawn)
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
