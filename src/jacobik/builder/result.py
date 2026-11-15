"""Builder data structures, option managers, and geometry helper queries."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import contextlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


class LigandInput:
    """
    User-facing description of one ligand.

    Parameters
    ----------
    smiles : str
        SMILES string for the ligand.  May be hydrogen-explicit or
        hydrogen-implicit; the engine auto-detects which.
    donors : list[list[int]]
        Donor groups.  Each inner list is a set of atom indices (0-based
        on the SMILES heavy-atom graph for H-implicit SMILES, or 0-based
        on the full graph for H-explicit SMILES) that form one coordination
        point.  Single-atom kappa donor: ``[[9]]``.  Haptic Cp ring:
        ``[[0,1,2,3,4]]``.  Mixed bidentate: ``[[9], [1,2,3,4,5]]``.
    sites : list[int] | None
        Explicit 0-indexed coordination site assignments, one per donor
        group.  ``sites[i]`` is the site index (from ``geometry_sites()``)
        that ``donors[i]`` should point toward.  If None (default), the
        engine performs Hungarian matching to assign donors to sites
        automatically.

    Raises
    ------
    ValueError
        If ``sites`` is provided but has a different length than ``donors``.
    """

    def __init__(
        self,
        smiles: str,
        donors: List[List[int]],
        sites: Optional[List[int]] = None,
    ) -> None:
        if not donors:
            raise ValueError("donors must be a non-empty list of donor groups")
        if sites is not None and len(sites) != len(donors):
            raise ValueError(
                f"sites length ({len(sites)}) must equal donors length "
                f"({len(donors)}) when explicitly provided"
            )
        self.smiles = smiles
        self.donors = donors
        self.sites = sites

    def __repr__(self) -> str:
        site_str = f", sites={self.sites}" if self.sites is not None else ""
        return f"LigandInput({self.smiles!r}, donors={self.donors}{site_str})"


def attach_solver_state(conformers, thetas, bend) -> None:
    """S2b capture: give each conformer its row's final thetas (one float64 vector per ligand) and the solver's
    angle-bend prior for that row. `thetas` (per-ligand (B, width) tensors) and `bend` ((B,) array or None) are indexed
    by the conformer's trial_idx. Pure storage: nothing is solved, scored or selected from it."""
    if bend is None or thetas is None:
        return
    host = [th.detach().double().cpu().numpy() for th in thetas]
    for c in conformers:
        b = int(c.trial_idx)
        c.solver_state = {"thetas": [h[b].copy() for h in host], "bend_prior": float(bend[b])}


@dataclass
class ConformerResult:
    """One conformer from the ensemble."""

    complex: object  # TMComplex
    score: float
    converged: bool
    rank: int  # 1-based rank (1 = best)
    trial_idx: Optional[int] = None  # row in the batched IK solve
    assignment_idx: Optional[int] = None
    embed_idx: Optional[int] = None
    solver_state: Optional[dict] = None
    stereo_label: Optional[str] = None
    requested_stereo_label: Optional[str] = None
    stereo_variant_idx: Optional[int] = None
    class_label: Optional[str] = None  # "<geometry>|S:<stereo labels>|A:<site assignment>": the isomer class
    stereo_production: bool = False  # the row belongs to the production (input-tagged) stereo class
    mirror_of: Optional[int] = None  # out_mirror: trial row this row is the reflection of
    ring_state: Optional[str] = None  # ring_states: start pucker per ligand ("L0:chair/twist"); a descriptor, not a class
    config_derived: bool = False  # config_remote: derived from a solved class by a local remote-element inversion

    def descriptors(self) -> dict:
        """The row's class tuple and descriptors (plain types, for pool files)."""
        return dict(class_label=self.class_label, stereo_label=self.stereo_label,
                    requested_stereo_label=self.requested_stereo_label, stereo_production=bool(self.stereo_production),
                    assignment_idx=None if self.assignment_idx is None else int(self.assignment_idx),
                    mirror_of=None if self.mirror_of is None else int(self.mirror_of), ring_state=self.ring_state,
                    config_derived=bool(self.config_derived))

    def xyz_block(self, comment: Optional[str] = None) -> str:
        """This conformer as one XYZ frame (text, with the trailing newline)."""
        mol = self.complex
        if comment is None:
            comment = f"{getattr(mol, 'csd_code', '') or ''}  score={self.score:.4f}".strip()
        lines = [str(len(mol.atoms)), comment]
        for a in mol.atoms:
            x, y, z = a.position
            lines.append(f"{a.symbol:<4s} {x:12.6f} {y:12.6f} {z:12.6f}")
        return "\n".join(lines) + "\n"

    def write_xyz(self, path: str | Path) -> None:
        """Write this conformer to an .xyz file."""
        path = Path(path)
        path.write_text(self.xyz_block())
        log.info("Wrote %s", path)

    def __repr__(self) -> str:
        conv = "converged" if self.converged else "not converged"
        return f"ConformerResult(rank={self.rank}, score={self.score:.4f}, {conv})"


@dataclass
class EnsembleResult:
    """Ranked ensemble of conformers returned by ComplexBuilder.solve()."""

    conformers: List[ConformerResult]
    solve_trace: Optional[Dict[str, np.ndarray]] = None
    scorer: Optional[object] = None
    assignment_search: Optional[Dict[str, object]] = None
    failure: Optional[str] = None

    @property
    def best(self) -> Optional[ConformerResult]:
        return self.conformers[0] if self.conformers else None

    @property
    def score_norm(self) -> Optional[float]:
        """Z, the number of scored features: score x Z is the summed log-likelihood in nats. None without a scorer."""
        sc = self.scorer
        try:
            return float(sc._agg_norm()) if sc is not None and hasattr(sc, "_agg_norm") else None
        except Exception:
            return None

    def delta_nats(self, c: ConformerResult) -> Optional[float]:
        """How far a conformer's summed log-likelihood lies below the best row's (>= 0; None without a scorer).
        It orders rows the way the solver does. It is not an energy: within one basin a nat is about RT, but rows in
        different basins can differ by tens of kJ/mol at equal score."""
        z = self.score_norm
        if z is None or not self.conformers:
            return None
        return float((self.conformers[0].score - c.score) * z)

    def by_class(self) -> Dict[Optional[str], List[ConformerResult]]:
        """Per-class pools: class_label -> its conformers in rank order; classes ordered by their best row.
        A class is a configuration that does not interconvert thermally (site arrangement, stereocentres, E/Z); the
        conformers inside one class are its distinct conformations."""
        out: Dict[Optional[str], List[ConformerResult]] = {}
        for c in sorted(self.conformers, key=lambda c: c.rank):
            out.setdefault(c.class_label, []).append(c)
        return out

    def class_ids(self) -> Dict[Optional[str], str]:
        """class_label -> short id within this ensemble: "C1" is the class of the best row, "C2" the next, ..."""
        return {k: f"C{i}" for i, k in enumerate(self.by_class(), start=1)}

    def classes(self) -> List[dict]:
        """One summary per class, best class first: id, label and its parts (geometry, stereo descriptors, site
        assignment), row count, how many rows are reflections, best rank and score, the best row's distance below the
        overall best in nats, whether it is the production (as written) stereo class, the ring states present."""
        out = []
        ids = self.class_ids()
        for k, v in self.by_class().items():
            parts = parse_class_label(k)
            out.append(dict(class_id=ids[k], class_label=k, geometry=parts["geometry"], stereo=parts["stereo"],
                            assignment=parts["assignment"], n_rows=len(v),
                            n_reflected=sum(c.mirror_of is not None for c in v), best_rank=int(v[0].rank),
                            best_score=float(v[0].score), delta_nats=self.delta_nats(v[0]),
                            stereo_production=any(c.stereo_production for c in v),
                            ring_states=sorted({c.ring_state for c in v if c.ring_state})))
        return out

    def to_records(self) -> List[dict]:
        """One plain dict per conformer in rank order (for a table or a DataFrame): rank, score, delta_nats, converged,
        class id and the row descriptors."""
        ids = self.class_ids()
        return [dict(rank=int(c.rank), score=float(c.score), delta_nats=self.delta_nats(c), converged=bool(c.converged),
                     class_id=ids.get(c.class_label), **c.descriptors())
                for c in sorted(self.conformers, key=lambda c: c.rank)]

    def write_xyz(self, path: str | Path, by_class: bool = False) -> List[Path]:
        """Write the ensemble as multi-frame XYZ, rows in rank order. One file at `path`, or with by_class=True one
        file per class in the directory `path` (C1.xyz, C2.xyz, ...). Each frame's comment line carries rank, score,
        class id and class label. Returns the files written."""
        path = Path(path)
        ids = self.class_ids()

        def frame(c):
            dn = self.delta_nats(c)
            extra = "" if dn is None else f" delta_nats={dn:.2f}"
            refl = "" if c.mirror_of is None else " reflected=1"
            return c.xyz_block(f"rank={c.rank} score={c.score:.4f}{extra} class={ids.get(c.class_label)}{refl} label={c.class_label}")
        written = []
        if by_class:
            path.mkdir(parents=True, exist_ok=True)
            for k, v in self.by_class().items():
                f = path / f"{ids[k]}.xyz"
                f.write_text("".join(frame(c) for c in v))
                written.append(f)
        else:
            path.write_text("".join(frame(c) for c in sorted(self.conformers, key=lambda c: c.rank)))
            written.append(path)
        log.info("Wrote %d file(s) under %s", len(written), path)
        return written

    @property
    def converged(self) -> List[ConformerResult]:
        return [c for c in self.conformers if c.converged]

    def summary(self, by_class: bool = True, max_rows: Optional[int] = None) -> None:
        """Print the ensemble: one line per class (when the rows carry class labels), then the ranked conformers.
        `max_rows` limits the conformer table (None = all). by_class=False prints the plain ranked table only."""
        if not self.conformers:
            print(f"empty ensemble{': ' + self.failure if self.failure else ''}")
            return
        ids = self.class_ids()
        labelled = by_class and any(c.class_label for c in self.conformers)
        if labelled:
            print(f"{'Class':>5}  {'Rows':>4}  {'Best':>4}  {'d(nats)':>8}  Configuration")
            print("-" * 64)
            for cl in self.classes():
                dn = "" if cl["delta_nats"] is None else f"{cl['delta_nats']:.1f}"
                desc = cl["geometry"] or "-"
                if cl["stereo"] and cl["stereo"] != "none":
                    desc += f"  stereo {cl['stereo']}"
                if cl["assignment"] is not None:
                    desc += f"  sites #{cl['assignment']}"
                if cl["n_reflected"]:
                    desc += f"  ({cl['n_reflected']} reflected)"
                if not cl["stereo_production"] and any(c.stereo_variant_idx is not None for c in self.conformers):
                    desc += "  [extra stereo class]"
                print(f"{cl['class_id']:>5}  {cl['n_rows']:>4}  {cl['best_rank']:>4}  {dn:>8}  {desc}")
            print()
        head = f"{'Rank':>4}  {'Score':>10}  {'d(nats)':>8}  {'Converged':>9}" + (f"  {'Class':>5}" if labelled else "")
        print(head)
        print("-" * len(head))
        rows = sorted(self.conformers, key=lambda c: c.rank)
        for c in rows[:max_rows]:
            dn = self.delta_nats(c)
            line = f"{c.rank:>4}  {c.score:>10.4f}  {'' if dn is None else format(dn, '.1f'):>8}  {'yes' if c.converged else 'no':>9}"
            if labelled:
                line += f"  {ids.get(c.class_label, ''):>5}" + ("  reflected" if c.mirror_of is not None else "")
            print(line)
        if max_rows is not None and len(rows) > max_rows:
            print(f"... {len(rows) - max_rows} more")
        n_conv = len(self.converged)
        print(f"\n{n_conv}/{len(self.conformers)} converged" + (f", {len(ids)} classes" if labelled else ""))


def parse_class_label(label: Optional[str]) -> dict:
    """Parts of a class label "<geometry>|S:<stereo descriptors>|A:<site assignment index>" (missing parts are None)."""
    out = dict(geometry=None, stereo=None, assignment=None)
    if not label:
        return out
    for i, part in enumerate(str(label).split("|")):
        if part.startswith("S:"):
            out["stereo"] = part[2:]
        elif part.startswith("A:"):
            try:
                out["assignment"] = int(part[2:])
            except ValueError:
                out["assignment"] = None
        elif i == 0:
            out["geometry"] = part
    return out


def _e():
    from jacobik.builder import engine

    return engine


def geometry_sites(geometry: str) -> np.ndarray:
    """
    Return the (CN, 3) array of ideal unit-vector site positions for a
    coordination geometry.


    Parameters
    ----------
    geometry : str — one of the labels in IDEAL_GEOMETRIES
               (e.g. "octahedral", "square_plan", "tetrahedral", ...)

    Returns
    -------
    (CN, 3) float64 array, one row per site.

    Example
    -------
    >>> geometry_sites("octahedral")
    array([[ 1.,  0.,  0.],   # site 0 (+x)
           [-1.,  0.,  0.],   # site 1 (−x)
           [ 0.,  1.,  0.],   # site 2 (+y)
           [ 0., -1.,  0.],   # site 3 (−y)
           [ 0.,  0.,  1.],   # site 4 (+z)
           [ 0.,  0., -1.]])  # site 5 (−z)

    """
    from jacobik.ik.targets import IDEAL_GEOMETRIES

    if geometry not in IDEAL_GEOMETRIES:
        raise ValueError(f"Unknown geometry {geometry!r}. Valid: {sorted(IDEAL_GEOMETRIES)}")
    return IDEAL_GEOMETRIES[geometry].copy()


def _has_tetrahedral_tags(builder) -> bool:
    """Whether any ligand SMILES carries a tetrahedral tag (mirror completion would invert it)."""
    from rdkit import Chem
    for lig in getattr(builder, "ligands", []) or []:
        m = Chem.MolFromSmiles(getattr(lig, "smiles", "") or "")
        if m is None:
            return True  # unknown: do not mirror
        if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in m.GetAtoms()):
            return True
    return False


class _LigandEmbedFailure(ValueError):
    """Internal signal that an individual ligand could not be embedded."""


class _PrepCapture(BaseException):
    """config_states: carries one class's prepared solve (ctx, assignments, beam_width, n_ik_steps) out of solve().
    A BaseException, so no `except Exception` on the way swallows it (the defect that made the bench solve only the
    last stereo class)."""


@dataclass
class _SolveContext:
    """Per-solve() state independent of the donor-to-site assignment candidate."""

    mol: object  # TMComplex
    ideal: np.ndarray  # (n_sites, 3)
    all_donor_groups: List[Tuple[int, int, List[int]]]  # (ligand, group, atom indices)
    n_donors: int
    chelate_pairs_global: List[Tuple[int, int]]
    heavy_graph: object  # nx.Graph, or None if construction failed
    # Equivalence classes of single-atom donor groups (see _symmetric_donor_groups).
    symmetric_donor_groups: List[List[int]]
    stats: object  # StatsLookup
    device: torch.device
    np_rng: np.random.Generator
    lr: float
    quantity_weights: Optional[Dict[str, float]]
    n_coord_samples: int
    ik_config: Optional[object]  # IKSolverConfig
    n_conformers: Optional[int]
    debug: bool
    rdmols: Optional[list] = None
    macro_pool: int = 64
    early_cull: Optional[Tuple[float, float]] = None  # (step fraction, keep fraction), set per solve
    orientation_model: str = "auto"
    angle_bends: bool = False
    angle_bend_penalty: Optional[Tuple[float, float, float]] = None
    bend_prior: Optional[dict] = None
    score_norm: float = 0.0  # the scorer's per-complex constant Z (output score window), set per solve
    dedup_rmsd: Optional[float] = None
    dedup_max_dev: Optional[float] = None


@contextlib.contextmanager
def _output_options(
    out_per_class_k=None,
    out_per_class_window=None,
    out_window=None,
    dedup_rmsd=None,
    dedup_max_dev=None,
):
    """Set the output options of ``ComplexBuilder.solve`` for the duration of one call (None = unchanged), restore
    them afterwards. Validated: a bad value raises before anything is changed."""
    from jacobik.ik import dedup as _dd

    if out_per_class_k is not None and (int(out_per_class_k) != out_per_class_k or out_per_class_k < 1):
        raise ValueError(f"out_per_class_k must be an integer >= 1, got {out_per_class_k!r}")
    if out_per_class_window is not None and not out_per_class_window > 0:
        raise ValueError(f"out_per_class_window must be > 0, got {out_per_class_window!r}")
    if out_window is not None and not out_window >= 0:
        raise ValueError(f"out_window must be >= 0 (0 = off), got {out_window!r}")
    if dedup_rmsd is not None and not dedup_rmsd > 0:
        raise ValueError(f"dedup_rmsd must be > 0, got {dedup_rmsd!r}")
    if dedup_max_dev is not None and not dedup_max_dev >= 0:
        raise ValueError(f"dedup_max_dev must be >= 0 (0 = off), got {dedup_max_dev!r}")
    eng = _e()
    saved = (eng.OUT_PER_CLASS_K, eng.OUT_PER_CLASS_WINDOW, eng.OUT_WINDOW, _dd.DEDUP_RMSD, _dd.DEDUP_MAX_DEV)
    try:
        if out_per_class_k is not None:
            eng.OUT_PER_CLASS_K = int(out_per_class_k)
        if out_per_class_window is not None:
            eng.OUT_PER_CLASS_WINDOW = float(out_per_class_window)
        if out_window is not None:
            eng.OUT_WINDOW = float(out_window)
        if dedup_rmsd is not None:
            _dd.DEDUP_RMSD = float(dedup_rmsd)
        if dedup_max_dev is not None:
            _dd.DEDUP_MAX_DEV = float(dedup_max_dev)
        yield
    finally:
        eng.OUT_PER_CLASS_K, eng.OUT_PER_CLASS_WINDOW, eng.OUT_WINDOW, _dd.DEDUP_RMSD, _dd.DEDUP_MAX_DEV = saved
