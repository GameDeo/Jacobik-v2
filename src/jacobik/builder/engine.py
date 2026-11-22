"""
Public API and orchestration of the Jacobik conformer ensemble engine.

``ComplexBuilder.solve()`` runs the pipeline for one complex: build a 3D shell
from each ligand's SMILES, assign donors to coordination sites (Hungarian
baseline, optionally the symmetry-distinct enumeration of all bijections),
decompose ligands into kinematic trees, beam-search torsion seeds, then jointly
refine all ligands' degrees of freedom with JointIKSolver (ik/ik_solver.py)
against the Scorer's statistical objective plus shape/symmetry constraints.

Public API: ``LigandInput``, ``ComplexBuilder``, ``EnsembleResult``,
``ConformerResult``, ``geometry_sites``.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import numpy as np

from jacobik.core import gpu

if TYPE_CHECKING:
    from jacobik.core.types import TMComplex
    from jacobik.ik.ik_solver import IKSolverConfig

log = logging.getLogger(__name__)


from jacobik.core import profile as _profile  # noqa: E402
from jacobik.builder.shell import (  # noqa: E402,F401  (re-exported: callers use these names on jacobik.engine)
    _BOND_LENGTH_TABLE,
    _DEFAULT_BOND_LENGTH,
    _GEOM_CN,
    _Q1_PROMINENCE_FLOOR,
    _order_sites_like_donor_groups,
    _q1_sample_distances,
    _refresh_coord_geometry,
    _shallow_copy_mol_for_conformer,
)
from jacobik.core.env import check_environment, resolve_device, torch_device  # noqa: E402
EMBED_BOND_CHECK = _profile.get("embed_bond_check")
RING_STATES = _profile.get("ring_states")
CONFIG_STATES = _profile.get("config_states")
CONFIG_NEAR = _profile.get("config_near")
CONFIG_REMOTE = _profile.get("config_remote")
EMBED_HAPTIC_LI_CAP = True  # production since the eta2 / haptic-cap fold-in (2026-11-01)
EMBED_LI_CAP = True  # production since the Li-cap fold-in (2026-10-29)
_EMBED_BOND_CHECK_EVENTS: List[dict] = []



from jacobik.stats.ladders import OS_UNKNOWN  # noqa: E402
from jacobik.ik.dedup import OUT_DEDUP, deduper_for, select_distinct  # noqa: E402

#: tests: take the returned conformers' coordinates from one batched kinematics pass on the CPU too (CUDA always does)
BATCHED_ASSEMBLY = False
import os as _os  # noqa: E402

#: Output score window (2026-10-16, ARCHITECTURE.md): rows whose final score lies more than OUT_WINDOW nats (summed, i.e.
#: score x the per-complex constant Z) below the best row are not returned. Offline on fixed pools (validity.py; refit
#: dev80 240 pools / v6 348 pools): removes 92 % / 78 % of the > 1000 kJ/mol members and 0.6 % / 0.4 % of those within
#: 50 kJ/mol of the pool's lowest, loses no isomer class with a good member and changes no #1. 0 = off.
# Production resolved configuration (hardcoded values)
OUT_WINDOW = 200.0
DEFAULT_ANGLE_BENDS = True
from jacobik.core import profile as _c2_profile
START_KABSCH = _c2_profile.get("start_kabsch")
START_KABSCH_PIVOT = _c2_profile.get("start_kabsch_pivot")
START_KABSCH_DIVERSITY = _c2_profile.get("start_kabsch_diversity")
STEREO_CANDIDATES = _c2_profile.get("stereo_candidates")
STEREO_SCOPE = _c2_profile.get("stereo_scope")
STEREO_COST_CEILING = _c2_profile.get("stereo_cost_ceiling")
OUT_PER_CLASS = _c2_profile.get("out_per_class")
OUT_MIRROR = bool(_c2_profile.get("out_mirror"))
OUT_PER_CLASS_K = _c2_profile.get("out_per_class_k")
OUT_PER_CLASS_WINDOW = _c2_profile.get("out_per_class_window")

from jacobik.builder.result import (  # noqa: F401
    ConformerResult,
    EnsembleResult,
    LigandInput,
    attach_solver_state,
)

EMBED_METHODS = ("gfn2", "mmff", "etkdg")
#: default ligand embed: "gfn2" (ETKDG -> MMFF -> GFN2-xTB, donor-set geometry held), "mmff" (ETKDG -> MMFF),
#: "etkdg" (ETKDG only; the old fast_embed=True). Profile field embed sets the default. gfn2 falls back to mmff when
#: the xtb binary (JACOBIK_XTB, default "xtb" on PATH) is missing or fails.
# Production resolved configuration (hardcoded values)
DEFAULT_EMBED = "gfn2"
XTB_BIN = _os.environ.get("JACOBIK_XTB", "xtb")
DETERMINISTIC_EMBED = True
_GFN2_CACHE: Dict[tuple, np.ndarray] = {}
_GFN2_BOND_CHECK_CACHE: Dict[tuple, dict] = {}



_POOL_TARGET = 16  # distinct good ring folds wanted
#: v1.0b: macrocycle ring conformers embedded around a virtual metal (profile field macro_template)
MACRO_TEMPLATE = _profile.get("macro_template")
#: v1.0b: refinement of the macrocycle pool conformers: None = free-ligand MMFF only, "gfn2" = then GFN2 like the embed
MACRO_POOL_REFINE = _profile.get("macro_pool_refine")
#: v1.0b: every macrocycle ranks its ring pool by all in-ring donor sites (kappa rays + haptic face normals,
#: beam._donor_face_convergence), not only rings with >= 3 sp3 ring nitrogens
MACRO_POOL_RANK = _profile.get("macro_pool_rank")
MACRO_HAPTIC_FILTER = bool(_profile.get("macro_haptic_filter"))
_HAPTIC_SITE_R = 2.05   # A from a haptic face centroid to the metal (eta2 olefin ~2.0-2.1)
_HAPTIC_SITE_TOL = 2.5  # A: largest spread of the faces' metal sites a fold may have (free COD: tub 1.1-1.6, chair 3.5)
#: v1.0b: embed refinement on the MACE surface, batched over every embed of a call (builder.mace_embed): None = GFN2
#: (production), "mace" = the GFN2 semantics (free ligand, Li caps, donor-donor holds) on MACE
EMBED_REFINE = _profile.get("embed_refine")
#: v1.0b: GFN2 embed Li counter-ion caps never coincide (embed._spread_li_caps)
LI_CAP_SPREAD = _profile.get("li_cap_spread")
#: v1.0b: xtb --opt level of the GFN2 embed refinement ("crude" = production)
EMBED_GFN2_OPT = _profile.get("embed_gfn2_opt")
_POOL_RANDOM_COORDS = False
_POOL_BATCH = 32  # conformers embedded per batch before re-checking
_POOL_GOOD = 2.0  # lone-pair convergence score counted as "good"
_MACRO_POOL_CACHE: Dict[tuple, object] = {}

#: How the variants of one graph-level assignment class (embed-aware dedup) share the row budget: "early_cull" = every
#: variant gets full rows, then an extra cull at EARLY_CULL_AT (fraction of the steps) keeps n_classes / n_candidates of
#: the rows (ranked by score with per-candidate niches); "split" = variants take the beam seeds in turn; "none".
CLASS_BUDGET = "split"
EARLY_CULL_AT = 0.05



from jacobik.builder.assemble import (  # noqa: E402,F401
    MIRROR,
    _AssembleMixin,
    _mirror_score_fn,
    mirror_completion,
    output_window,
)
from jacobik.builder.assignment import _AssignmentMixin  # noqa: E402
from jacobik.builder.beam import _BeamMixin  # noqa: E402
from jacobik.builder.configs import _ConfigsBatchedMixin, _ConfigsPrepMixin  # noqa: E402
from jacobik.builder.result import (  # noqa: E402,F401
    _LigandEmbedFailure,
    _PrepCapture,
    _SolveContext,
    _has_tetrahedral_tags,
    _output_options,
    geometry_sites,
)
from jacobik.builder.embed import (  # noqa: E402,F401
    _bond_refinement_violations,
    _embed_etkdg_with_retries,
    _embed_refine,
    _gfn2_refine,
    pop_embed_bond_check_events,
)
from jacobik.builder.shell import (  # noqa: E402,F401
    _build_complex_shell,
    _donor_indices_to_full_graph,
    _smiles_to_conformer,
)
from jacobik.builder.beam import (  # noqa: E402,F401
    _RowsMixin,
    _diverse_ranked,
    _lone_pair_convergence,
    _wrap,
)

class ComplexBuilder(_ConfigsPrepMixin, _ConfigsBatchedMixin, _AssignmentMixin, _RowsMixin, _BeamMixin, _AssembleMixin):
    """
    Build a transition-metal complex from its components and solve for a
    conformer ensemble.

    Parameters
    ----------
    metal    : element symbol, e.g. "Fe", "Pt", "Ru"
    geometry : coordination geometry label, e.g. "octahedral"
               Use ``geometry_sites(geometry)`` to inspect site vectors.
    charge   : formal charge of the complex
    spin     : spin multiplicity (2S+1)
    oxidation_state : optional metal oxidation state override; if omitted it is
               deduced from ``charge`` and the ligand SMILES (see
               core/oxidation.py) and logged at INFO
    ligands  : list of LigandInput objects.  Each ligand is explicitly
               defined — repeat identical LigandInput objects for symmetric
               complexes.  The engine validates that the total donor count
               (summed across all ligands) equals the CN of the geometry.

    Raises
    ------
    ValueError
        If the total donor count does not match the CN of the geometry.
    ValueError
        If any ligand specifies explicit ``sites=`` that conflict (duplicate
        site indices, or site index out of range for the geometry).

    Example
    -------
    >>> cp    = LigandInput("C1=CC=CC=1", donors=[[0,1,2,3,4]])
    >>> phsph = LigandInput("P(c1ccccc1)(c1ccccc1)c1ccccc1", donors=[[0]])
    >>> builder = ComplexBuilder(
    ...     metal="Fe", geometry="octahedral",
    ...     charge=0, spin=1,
    ...     ligands=[cp, cp, phsph],
    ... )
    >>> result = builder.solve(constraints_db="constraints.json")
    >>> result.summary()
    """

    def __init__(
        self,
        metal: str,
        geometry: str,
        charge: int,
        spin: int,
        ligands: List[LigandInput],
        oxidation_state: Optional[int] = None,
        stereo_enum: Optional[bool] = None,
        out_per_class: Optional[bool] = None,
    ) -> None:
        from jacobik.ik.targets import IDEAL_GEOMETRIES

        if geometry not in IDEAL_GEOMETRIES and geometry != "other":
            raise ValueError(f"Unknown geometry {geometry!r}. Valid: {sorted(IDEAL_GEOMETRIES)}")

        ligands = _normalize_ligands(ligands)
        total_donors = sum(len(lig.donors) for lig in ligands)
        expected_cn = _GEOM_CN.get(geometry)
        if expected_cn is not None and total_donors != expected_cn:
            raise ValueError(
                f"Geometry '{geometry}' requires CN={expected_cn}, but "
                f"total donor groups across all ligands = {total_donors}. "
                f"Each donor group (inner list in donors=) counts as one "
                f"coordination site."
            )

        if expected_cn is not None:
            used_sites: set = set()
            for lig in ligands:
                if lig.sites is None:
                    continue
                for s in lig.sites:
                    if s < 0 or s >= expected_cn:
                        raise ValueError(
                            f"Site index {s} is out of range for "
                            f"geometry '{geometry}' (CN={expected_cn}, "
                            f"valid indices 0..{expected_cn - 1})"
                        )
                    if s in used_sites:
                        raise ValueError(
                            f"Site index {s} is assigned to more than one "
                            f"donor group across the ligand list"
                        )
                    used_sites.add(s)

        self.metal = metal
        self.geometry = geometry
        self.charge = charge
        self.spin = spin
        self.ligands = ligands
        self.stereo_enum = stereo_enum
        self.out_per_class = out_per_class
        # Oxidation state: explicit override wins; otherwise deduced as
        # net charge - ionic-model ligand charges.  Low confidence -> unknown,
        # and the stats lookup then skips every oxidation-state-keyed level.
        from jacobik.core.chemistry import deduce_oxidation_state

        if oxidation_state is not None:
            self.oxidation_state = int(oxidation_state)
            self.oxidation_source = "user"
            self.oxidation_notes: List[str] = []
        else:
            os_, conf, notes = deduce_oxidation_state(metal, charge, ligands)
            self.oxidation_state = None if os_ == OS_UNKNOWN else os_
            self.oxidation_source = f"deduced({conf})"
            self.oxidation_notes = notes
        log.info(
            "oxidation state %s (%s) %s", self.oxidation_state, self.oxidation_source,
            "; ".join(self.oxidation_notes),
        )

        log.debug(
            "ComplexBuilder: %s %s CN=%s  %d ligands  %d total donors",
            metal,
            geometry,
            expected_cn,
            len(ligands),
            total_donors,
        )

    def solve(
        self,
        constraints_db: str | Path | None = None,
        beam_width: int = 16,
        n_ik_steps: int = 200,
        lr: float = 0.1,
        ik_config: Optional[IKSolverConfig] = None,
        n_coord_samples: int = 8,
        quantity_weights: Optional[Dict[str, float]] = None,
        device: str = "cuda",
        seed: Optional[int] = None,
        debug: bool = False,
        stats: Optional[object] = None,
        fast_embed: bool = False,
        embed: Optional[str] = None,
        n_conformers: Optional[int] = None,
        use_symmetry_enumeration: bool = True,
        embed_aware_dedup: bool = True,
        macrocycle_pool: int = 64,
        shannon: bool = False,
        embed_seed: Optional[int] = None,
        chelate_embed: bool = True,
        orientation_model: str = "auto",
        angle_bends: Optional[bool] = None,
        bend_free_deg: Optional[float] = None,
        bend_sigma_deg: float = 5.0,
        bend_weight: float = 1.0,
        bend_penalty: str = "energy",
        bend_scale: float = 1.0,
        stereo_enum: Optional[bool] = None,
        out_per_class: Optional[bool] = None,
        out_per_class_k: Optional[int] = None,
        out_per_class_window: Optional[float] = None,
        out_window: Optional[float] = None,
        dedup_rmsd: Optional[float] = None,
        dedup_max_dev: Optional[float] = None,
        settle_steps: Optional[int] = None,
        settle_top: Optional[int] = None,
    ) -> EnsembleResult:
        """
        Solve for a conformer ensemble.

        Parameters
        ----------
        constraints_db   : path to constraints DB (.npz or .json; default: packaged in jacobik/data)
        beam_width       : torsion seeds per ligand from beam search
        n_ik_steps       : gradient-descent iterations of the joint IK solve
        lr               : Adam learning rate
        ik_config        : IKSolverConfig overrides (n_steps and lr are set from
                           the arguments above; shape_tol None -> per geometry)
        n_coord_samples  : coordination-sphere samples (M-D length diversity)
        quantity_weights : optional {"Q1".."Q6": weight}; default all 1.0
        device           : "cuda" (default; GPU-only Triton build, "auto" resolves to "cuda")
        seed             : seeds torch, numpy and the Q1 sampler
        debug            : print diagnostics
        stats            : pre-built StatsLookup (skips loading constraints_db)
        shannon          : borrow + Shannon-radius-shift Q1 stats for (metal, OS, spin)
                           states with no data of their own (default off)
        embed            : ligand embed method, "gfn2" (default, see DEFAULT_EMBED), "mmff" or "etkdg"
        fast_embed       : legacy alias for embed="etkdg" (skip MMFF refinement of the ETKDG starting geometry)
        n_conformers     : max conformers returned (default
                           max(beam_width, n_coord_samples))
        use_symmetry_enumeration : enumerate all symmetry-distinct donor-to-site
                           assignments and solve them together in one batch,
                           instead of only the baseline Hungarian assignment.
                           Falls back to the baseline for fully explicit,
                           unsaturated or "other"-geometry complexes and when
                           enumeration finds nothing.
        embed_aware_dedup : (default True) merge assignments only by proper
                           rotations and swaps of whole identical ligands, the
                           exact symmetries of the embedded ligands; variants of
                           one graph-level class share one candidate's row
                           budget (site_enumeration.dedup_exact). False = legacy
                           dedup (equivalent-donor swaps x reflections), which can
                           keep only an assignment the embed cannot reach.
        chelate_embed    : chelate-aware free-ligand embed (default on; see
                           chelate_embed.py): donor-pair distance bounds for
                           multidentate ligands + temporary H caps on anionic
                           donors. False = plain ETKDG + MMFF.
        orientation_model: "auto" (default): joint donor-orientation score
                           S_orient in place of Q3 + Q4 when the DB has QO
                           fits, else Q3 + Q4; "sorient" requires QO fits;
                           "q3q4" forces the separate terms.
        angle_bends      : bond-angle bend DOFs on chelate backbones with a
                           flat-bottom penalty (ik/angle_bends.py). Default
                           follows the profile (u1 on, legacy off); explicit
                           True/False overrides it. bend_penalty="energy" (default): Boltzmann
                           prior on each bend DOF, -bend_scale * sum_i k_i
                           (|beta_i| - bend_free_deg)_+^2 / 2RT, k from a MACE
                           scan by pivot type and axis (see angle_bends.BEND_K);
                           "angle": flat-bottom penalty Q7 on bond-angle change
                           (bend_free_deg, bend_sigma_deg, bend_weight).
        embed_seed       : RDKit ETKDG random seed for the initial ligand
                           embedding (default: the fixed value 42 used
                           historically). Does not affect ``seed`` above.
                           For rigid multidentate ligands whose donor atoms
                           span more than one rigid sub-structure, this seed
                           can be the deciding factor between a good solve
                           and a catastrophic one, because torsion
                           optimisation cannot correct the *relative*
                           placement of two rigid pieces the embedding fixed
                           in a bad orientation. See ``solve_multi_embed``,
                           which tries several seeds and keeps the best
                           results, instead of gambling on one.
        n_conformers     : (above) the number of conformers returned; with out_per_class the per-class
                           limit below applies instead.
        out_per_class    : return up to ``out_per_class_k`` distinct conformers per stereo/geometry class instead of
                           a global top-``n_conformers`` (default: the instance setting, else the profile field
                           ``out_per_class``, off).
        out_per_class_k  : conformers kept per class with out_per_class (default 8).
        out_per_class_window : a class member is kept only within this many summed nats of the class's best row
                           (default 35).
        out_window       : rows more than this many summed nats (score x the complex's constant Z) below the best
                           row are not returned; 0 switches the window off (default 200).
        dedup_rmsd       : two rows are the same structure when their heavy-atom (metal included), symmetry-aware
                           RMSD is at most this many Angstrom (default 0.5, profile field ``dedup_rmsd``).
        dedup_max_dev    : optional maximum per-atom heavy deviation threshold in Angstrom (default 0.0 = off,
                           profile field ``dedup_max_dev``).
        settle_steps     : k > 0 adds the settle stage (builder.settle): a candidate pool settle_pool x the requested
                           size, k batched MACE-POLAR steps with metal-donor distances held, dedup, then per-class
                           output by settled energy (needs the mace extra). Default: profile field settle_steps.
        settle_top       : with the settle stage, settle and return only the k rows of each configuration class with the
                           lowest MACE energy at the solver geometry (one evaluation per row picks them; the score is
                           a worse pick, builder.settle). Default None: every row (k = 8-12 keeps the class leaders at
                           a third to half of the settle cost).
        The five options out_per_class_k, out_per_class_window, out_window, dedup_rmsd and dedup_max_dev are process-wide
        module settings that this call sets for its own duration and restores afterwards (not thread-safe for concurrent
        solves with different values). Left at None they change nothing.

        Returns
        -------
        EnsembleResult with conformers sorted best to worst by score.
        """
        from jacobik.builder import mace_embed
        from jacobik.builder import settle as _settle_mod

        device = resolve_device(device)
        if not str(device).startswith("cuda"):
            raise RuntimeError(f"GPU-only Triton build requires CUDA (got device={device!r})")

        mace_embed.set_context(device)  # every embed refinement below (prefetches of every class path) on this device
        _settle_k = _settle_mod.SETTLE_STEPS if settle_steps is None else int(settle_steps)
        if settle_top is not None and int(settle_top) < 1:
            raise ValueError("settle_top must be >= 1 or None")
        if _settle_k > 0 and not getattr(self, "_settle_inner", False):
            import inspect

            _loc = locals()
            _kw = {
                k: _loc[k]
                for k, p in inspect.signature(ComplexBuilder.solve).parameters.items()
                if k not in ("self", "settle_steps", "settle_top") and p.kind != inspect.Parameter.VAR_KEYWORD and k in _loc
            }
            _opc = self.out_per_class if out_per_class is None else out_per_class
            if _opc is None:
                _opc = OUT_PER_CLASS
            _k_pc = (out_per_class_k or OUT_PER_CLASS_K) if _opc else None
            if _settle_mod.SETTLE_POOL > 1.0:              # enlarge a cap; no cap (the window limits) stays none
                if _opc:
                    _kw["out_per_class_k"] = _settle_mod.enlarged(_k_pc, _settle_mod.SETTLE_POOL)
                if n_conformers is not None:
                    _kw["n_conformers"] = _settle_mod.enlarged(n_conformers, _settle_mod.SETTLE_POOL)
            self._settle_inner = True
            try:
                _res = self.solve(**_kw)
            finally:
                self._settle_inner = False
            return _settle_mod.settle(_res, _settle_k, device=str(device), pool=_settle_mod.SETTLE_POOL,
                                      dedup_rmsd=dedup_rmsd, n_request=n_conformers, per_class=_k_pc, top=settle_top)
        _opts = {k: v for k, v in (
            ("out_per_class_k", out_per_class_k),
            ("out_per_class_window", out_per_class_window),
            ("out_window", out_window),
            ("dedup_rmsd", dedup_rmsd),
            ("dedup_max_dev", dedup_max_dev),
        ) if v is not None}
        if angle_bends is None:
            angle_bends = DEFAULT_ANGLE_BENDS
        if bend_free_deg is None:
            from jacobik.ik.angle_bends import FREE_DEG as _DEFAULT_BEND_FREE_DEG
            bend_free_deg = _DEFAULT_BEND_FREE_DEG
        kw_common = dict(
            constraints_db=constraints_db, beam_width=beam_width, n_ik_steps=n_ik_steps, lr=lr,
            ik_config=ik_config, n_coord_samples=n_coord_samples, quantity_weights=quantity_weights,
            device=device, seed=seed, debug=debug, stats=stats, fast_embed=fast_embed, embed=embed,
            n_conformers=n_conformers, use_symmetry_enumeration=use_symmetry_enumeration,
            embed_aware_dedup=embed_aware_dedup, macrocycle_pool=macrocycle_pool, shannon=shannon,
            embed_seed=embed_seed, chelate_embed=chelate_embed, orientation_model=orientation_model,
            angle_bends=angle_bends, bend_free_deg=bend_free_deg, bend_sigma_deg=bend_sigma_deg,
            bend_weight=bend_weight, bend_penalty=bend_penalty, bend_scale=bend_scale,
        )
        if _opts:
            with _output_options(**_opts):
                return self.solve(**kw_common, stereo_enum=stereo_enum, out_per_class=out_per_class)
        from jacobik.builder import embed as _ce

        do_stereo = self.stereo_enum if stereo_enum is None else stereo_enum
        if do_stereo is None:
            do_stereo = STEREO_CANDIDATES

        if _ce.EZ_ENUMERATE and not do_stereo and not getattr(self, "_stereo_inner", False) and not getattr(self, "_ez_inner", False):
            variants = self._ez_combinations()
            if len(variants) > 1:
                return self._solve_classes_batched(variants, [], kw_common, "ez") if CONFIG_STATES else self._solve_ez_variants(variants, kw_common)

        do_out_per_class = self.out_per_class if out_per_class is None else out_per_class
        if do_out_per_class is None:
            do_out_per_class = OUT_PER_CLASS

        if do_stereo and not getattr(self, "_stereo_inner", False) and not getattr(self, "_ez_inner", False):
            from jacobik.ik import stereo
            kw_inner = dict(kw_common, stereo_enum=False, out_per_class=do_out_per_class)
            try:
                prod_combos = self._ez_combinations() if _ce.EZ_ENUMERATE else [[None] * len(self.ligands)]
                elements = stereo.perceive_complex_stereo(self, scope=STEREO_SCOPE)
                variants, sinfo = stereo.superset_variants(
                    self, elements, prod_combos,
                    achiral_environment=stereo.is_achiral_geometry(self.geometry),
                    cost_ceiling_multiplier=STEREO_COST_CEILING,
                    near=bool(CONFIG_NEAR and CONFIG_STATES), mirror=OUT_MIRROR,
                    remote=bool(CONFIG_REMOTE and CONFIG_NEAR and CONFIG_STATES))
            except Exception as exc:  # never fail the caller: fall back to the production path
                log.warning("stereo candidates: class construction failed (%s); production classes only", exc)
                variants, sinfo = [], {}
            if sinfo.get("n_extra", 0) > 0 or (CONFIG_STATES and sinfo.get("remote_derived")):
                self._stereo_class_info = sinfo
                if CONFIG_STATES:      # P2R: every stereo class in one batch
                    return self._solve_classes_batched(variants, elements, kw_inner, "stereo",
                                                       out_per_class=do_out_per_class)
                return self._solve_stereo_variants(variants, elements, kw_inner, out_per_class=do_out_per_class)
            self._stereo_class_info = sinfo
            if _ce.EZ_ENUMERATE and not getattr(self, "_ez_inner", False):
                pv = self._ez_combinations()
                if len(pv) > 1:
                    return self._solve_classes_batched(pv, [], kw_common, "ez") if CONFIG_STATES else self._solve_ez_variants(pv, kw_common)
            do_stereo = False
        _device = str(device)
        np_rng = np.random.default_rng(seed)  # Q1 GMM target sampling

        if seed is not None:
            gpu.seed(seed)
            np.random.seed(seed)

        stats = self._load_stats(constraints_db, stats, shannon=shannon)
        if EMBED_REFINE and not getattr(self, "_capture_prep", False):
            with mace_embed.collecting():  # every ligand of this shell refined in one MACE batch
                try:
                    self._build_shell(fast_embed, False, embed_seed=embed_seed, chelate_embed=chelate_embed, embed=embed)
                except Exception as exc:
                    log.debug("embed prefetch failed (%s)", exc)
        try:
            mol, rdmols, donor_idx_full = self._build_shell(
                fast_embed, debug, embed_seed=embed_seed, chelate_embed=chelate_embed, embed=embed
            )
        except _LigandEmbedFailure as exc:
            log.warning("Complex embed failed; returning an empty ensemble: %s", exc)
            return EnsembleResult(conformers=[], failure=str(exc))
        ideal, all_donor_groups, n_donors = self._collect_donor_groups(mol, donor_idx_full)
        chelate_pairs_global, heavy_graph = self._detect_chelate_pairs(
            mol, all_donor_groups, n_donors
        )
        assignment = self._compute_baseline_assignment(
            mol, ideal, all_donor_groups, n_donors, chelate_pairs_global, heavy_graph=heavy_graph
        )
        log.info("Site assignment: %s", assignment.tolist())

        symmetric_donor_groups = self._symmetric_donor_groups(mol, rdmols, all_donor_groups)

        ctx = _SolveContext(
            mol=mol,
            ideal=ideal,
            all_donor_groups=all_donor_groups,
            n_donors=n_donors,
            chelate_pairs_global=chelate_pairs_global,
            heavy_graph=heavy_graph,
            symmetric_donor_groups=symmetric_donor_groups,
            stats=stats,
            device=_device,
            np_rng=np_rng,
            lr=lr,
            quantity_weights=quantity_weights,
            n_coord_samples=n_coord_samples,
            ik_config=ik_config,
            n_conformers=n_conformers,
            debug=debug,
            rdmols=rdmols,
            macro_pool=macrocycle_pool,
            orientation_model=orientation_model,
            angle_bends=bool(angle_bends),
            angle_bend_penalty=(bend_free_deg, bend_sigma_deg, bend_weight) if (angle_bends and bend_penalty == "angle") else None,
            bend_prior=dict(free_deg=float(bend_free_deg), scale=float(bend_scale)) if (angle_bends and bend_penalty == "energy") else None,
        )

        candidates: List[np.ndarray] = [assignment]
        fully_explicit = all(lig_inp.sites is not None for lig_inp in self.ligands)
        if use_symmetry_enumeration and not fully_explicit:
            candidates = self._run_symmetry_enumeration(ctx, assignment, debug, embed_aware=embed_aware_dedup)

        return self._solve_degrading(ctx, candidates, beam_width, n_ik_steps)

    def _solve_degrading(self, ctx, candidates, beam_width, n_ik_steps):
        """_solve_for_assignments; on CUDA out of memory the batch (candidates x beam_width x n_coord_samples rows) is
        re-solved with beam width and coordinate samples halved, down to 1 x 1, then with the baseline assignment
        alone: the generator degrades instead of failing (challenge_D03, 1488 assignments x 16 x 8 rows, 2026-11-20).
        Batches that fit are untouched."""
        import dataclasses

        bw, cands, cur = int(beam_width), list(candidates), ctx
        while True:
            try:
                return self._solve_for_assignments(cur, cands, bw, n_ik_steps)
            except (gpu.OOMError, RuntimeError) as exc:
                if not isinstance(exc, gpu.OOMError) and "out of memory" not in str(exc).lower():
                    raise
                M = int(cur.n_coord_samples or 1)
                if bw > 1 or M > 1:
                    nbw, nM = max(1, bw // 2), max(1, M // 2)
                elif len(cands) > 1:
                    nbw, nM, cands = bw, M, cands[:1]
                else:
                    raise
                log.warning("joint IK batch out of GPU memory (%d assignments x beam %d x %d samples); retrying with "
                            "%d x %d x %d", len(candidates), bw, M, len(cands), nbw, nM)
            import gc

            gc.collect()
            gpu.empty_cache()
            bw, cur = nbw, dataclasses.replace(cur, n_coord_samples=nM)


_LIGAND_LIBRARY_CACHE: Dict[Path, object] = {}


def load_ligand_library():
    """Dynamically load the curated ``interface/ligand_library.py`` module."""
    existing = sys.modules.get("interface.ligand_library")
    if existing is not None:
        return existing
    candidates: List[Path] = []
    env_path = os.environ.get("JACOBIK_LIGAND_LIBRARY", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(Path(__file__).resolve().parents[3] / "interface" / "ligand_library.py")
    candidates.append(Path.cwd() / "interface" / "ligand_library.py")
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            cached = _LIGAND_LIBRARY_CACHE.get(resolved)
            if cached is not None:
                return cached
            spec = importlib.util.spec_from_file_location("_jacobik_ligand_library", resolved)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_jacobik_ligand_library"] = mod
            spec.loader.exec_module(mod)
            _LIGAND_LIBRARY_CACHE[resolved] = mod
            return mod
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not locate interface/ligand_library.py (tried: {tried}). "
        "Pass LigandInput objects directly or set JACOBIK_LIGAND_LIBRARY."
    )


def ligand(spec: str, mode: Optional[str] = None, *, sites: Optional[Sequence[int]] = None) -> LigandInput:
    """Resolve a library ligand name, alias, or ``name@mode`` string into a ``LigandInput``."""
    return load_ligand_library().ligand(spec, mode, sites=sites)


def ligands(*specs: str) -> List[LigandInput]:
    """Expand library ligand strings (supporting ``@mode`` and ``*N`` counts) into ``List[LigandInput]``."""
    return load_ligand_library().ligands(*specs)


def debug_complex(name: str):
    """Return a predefined debug complex tuple unpackable as ``jacobik.generate(*debug_complex('CBTYCO'))``."""
    return load_ligand_library().debug_complex(name)


def _normalize_ligand_items(item) -> List[LigandInput]:
    """Coerce a ``LigandInput``, library spec string, tuple, or dict into ``List[LigandInput]``."""
    if isinstance(item, LigandInput):
        return [item]
    if isinstance(item, str):
        return load_ligand_library().expand_ligand_spec(item)
    if isinstance(item, dict):
        return [LigandInput(item["smiles"], donors=item["donors"], sites=item.get("sites"))]
    if isinstance(item, (tuple, list)) and len(item) in (2, 3):
        return [LigandInput(item[0], donors=item[1], sites=item[2] if len(item) == 3 else None)]
    raise TypeError(
        "Each ligand must be a LigandInput, library name/alias string, (smiles, donors[, sites]) tuple, "
        f"or {{'smiles': ..., 'donors': ...}} mapping; got {type(item).__name__}: {item!r}"
    )


def _normalize_ligands(ligands_input) -> List[LigandInput]:
    """Expand and validate an iterable of ligand inputs or library spec strings."""
    if isinstance(ligands_input, (str, bytes)):
        raise TypeError(
            "ligands must be an iterable of LigandInput or library strings (for example ['bpy*3']), not str"
        )
    out: List[LigandInput] = []
    for item in ligands_input:
        out.extend(_normalize_ligand_items(item))
    return out


def _normalize_ligand(item) -> LigandInput:
    """Coerce a single non-repeated ligand input into ``LigandInput``."""
    items = _normalize_ligand_items(item)
    if len(items) != 1:
        raise ValueError(f"Expected a single ligand specification, got {len(items)} items from {item!r}")
    return items[0]


def generate(
    metal: str,
    geometry: str,
    charge: int,
    spin: int,
    ligands,
    *,
    oxidation_state: Optional[int] = None,
    constraints_db: str | Path | None = None,
    device: str = "auto",
    check_env: bool = True,
    relax: Optional[str] = None,
    relax_binding: str = "keep",
    relax_top: Optional[int] = None,
    relax_etol: Optional[float] = None,
    **solve_kwargs,
) -> EnsembleResult:
    """High-level single-call conformer ensemble generator.

    Validates runtime environment (``xtb`` when ``embed='gfn2'`` and constraints DB unless
    ``stats=`` is passed), resolves ``device='auto'``, builds a ``ComplexBuilder``, and
    returns ``builder.solve(...)``. ``relax="loose"|"normal"|"tight"`` then relaxes every row with MACE-POLAR
    (EnsembleResult.relax; needs the ``mace`` extra); ``relax_top=k`` relaxes and returns only the k best-ranked rows of each configuration class; ``relax_etol``
    (kJ/mol) adds the energy stop of EnsembleResult.relax(etol=...).
    ``settle_top=k`` (a solve option) settles only the k lowest-energy rows of each class (ComplexBuilder.solve).
    """
    norm_ligands = _normalize_ligands(ligands)
    if check_env:
        embed_mode = solve_kwargs.get("embed") or ("etkdg" if solve_kwargs.get("fast_embed") else DEFAULT_EMBED)
        check_environment(
            require_xtb=(embed_mode == "gfn2"),
            require_db=(solve_kwargs.get("stats") is None),
            constraints_db=constraints_db,
            device=device,
        )
    resolved_device = resolve_device(device)
    builder = ComplexBuilder(
        metal=metal,
        geometry=geometry,
        charge=charge,
        spin=spin,
        ligands=norm_ligands,
        oxidation_state=oxidation_state,
        stereo_enum=solve_kwargs.get("stereo_enum"),
        out_per_class=solve_kwargs.get("out_per_class"),
    )
    from jacobik.core import timing

    timing.reset()
    t0 = time.perf_counter()
    res = builder.solve(constraints_db=constraints_db, device=resolved_device, **solve_kwargs)
    if relax:
        res = res.relax(level=relax, device=str(resolved_device), binding=relax_binding, top=relax_top, etol=relax_etol)
    res.timings = dict(timing.snapshot(), total=round(time.perf_counter() - t0, 3))
    return res
