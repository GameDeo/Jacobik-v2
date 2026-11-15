# Archived Solver Flags and Profile Decisions

This document archives all 37 solver profile flags from `jacobik.profile` (`FIELDS`), their legacy defaults, `u1` values, final production values, and the evidentiary basis for their consolidation in Phase 4 of the repository restructure.

---

## Master Table of Profile Flags and Decisions

| Flag Name | Module & Attribute | Legacy (2026-10-16) | Adopted u1 (2026-10-18) | Final Production | Evidence / Decision Rationale |
|---|---|---|---|---|---|
| `oxidation_model` | `jacobik.core.oxidation.OXIDATION_MODEL` | `"legacy"` | `"training"` | `"training"` | Deduce oxidation state using whole-ligand formal charge matching tmQM extraction conventions; removes fragile radical heuristics and haptic size heuristics. |
| `angle_bends` | `jacobik.engine.DEFAULT_ANGLE_BENDS` | `False` | `True` | `True` | Kinematic in-plane/out-of-plane bend degrees of freedom calibrated against MACE-POLAR-1-S energies; resolves acyclic strain and haptic tilt. |
| `score_agg` | `jacobik.ik.scorer.SCORE_AGG` | `"mean"` | `"sum"` | `"sum"` | Unified objective: total summed log-likelihood normalized by single per-complex constant $Z$ (number of internal features). |
| `rank_agg` | `jacobik.ik.scorer.RANK_AGG` | `"sum"` | `"sum"` | `"sum"` | Final ranking uses summed log-likelihood matching the solve objective. |
| `solve_floor` | `jacobik.ik.scorer.GMM_FLOOR_MODE` | `"scorer"` | `"thermal"` | `"thermal"` | 298 K physical thermal widths from GFN2 Hessians (`refit/thermal_widths.py`) replace arbitrary numerical gradient guard floors (sd 0.2 Å). |
| `rank_floor_fit` | `jacobik.ik.ik_solver._RANK_FLOOR` | `True` | `False` | `False` | Resolves solve-ranking objective mismatch by scoring final ranking with the same thermal floors as the solve. |
| `q1_tail_sigma` | `jacobik.ik.scorer.Q1_TAIL_SIGMA` | `0.10` | `0.0` | `0.0` | Eliminates arbitrary 0.10 Å component clipping in Q1 mixtures; full fitted distribution is evaluated. |
| `q1_wall` | `jacobik.ik.scorer.Q1_WALL` | `True` | `True` | `True` | One-sided Gaussian wall beyond table support prevents metal-donor dissociation during optimization. |
| `q5_fold` | `jacobik.ik.scorer.Q5_FOLD_MODE` | `"auto"` | `"auto"` | `"auto"` | Averages density over equivalent rotor flank choices for symmetric groups (CH3, tBu, NH2, phenyl). |
| `q6_model` | `jacobik.ik.scorer.Q6_MODEL` | `"wall"` | `"repulsive"` | `"repulsive"` | Purely repulsive exchange steric model: physical Buckingham exponential on class-5 non-polar pairs + analytical core on class-4/metal/polar-H contacts. |
| `nci` | `jacobik.stats.nci.NCI_ON` | `False` | `True` | `False` | NCI v2 refit outcome: empirical log-enrichment regularized to physical compliance yields $\gamma = 0$; validation ranking showed no benefit. Code moved to `research/interactions/nci_v2/`. |
| `nci_types` | `jacobik.stats.nci.ACTIVE_TYPES` | `("hb", "pp", "chp")` | `("hb",)` | `("hb",)` | Inactive in production (`nci = False`). |
| `bvs` | `jacobik.ik.scorer.BVS` | `False` | `False` | `False` | Bond-valence sum coupling term (QV / QVR) showed no gain over independent Q1 distributions; disabled. |
| `haptic_per_sub` | `jacobik.ik.scorer.HAPTIC_PER_SUB` | `True` | `True` | `True` | Prices haptic donor orientations per substituent rather than face-averaged bend. |
| `bend_per_z` | `jacobik.ik.ik_solver._BEND_PER_Z` | `False` | `True` | `True` | Normalizes bend prior log-probability by $Z$, balancing bend energy with statistical likelihood terms. |
| `bend_release` | `jacobik.ik.ik_solver.BEND_RELEASE` | `None` | `None` | `[0.1, 100]` | Package C/F: angle-bend step-size ramp ($s_0=0.1$ ramping to $1.0$ over $K=100$ steps) resolves premature bend trapping and slashes catastrophic tail ($> 500$ kJ/mol from 14 to 8 units on `cal`) while preserving median $\Delta E$. |
| `embed_li_cap` | `jacobik.engine.EMBED_LI_CAP` | `False` | `False` | `True` | F-4/F-5: temporary Li counter-ions at locally anionic kappa donors during GFN2 embed refinement; on 102 changed cal/tail parents, top-1 ΔE_unrel -26.0 kJ/mol [-43.1, -14.6], floor -25.7, obscene -3.3 pp (L4, seeds 11-13; 2060 agrees). Folded in 2026-10-29 (`6cc586e17ab1`). |
| `q6_donor_pairs` | `jacobik.ik.geometry.Q6_DONOR_PAIR_SCALE` | `None` | `None` | `{"hh": 0.95, "ho": 0.90, "oo": 0.80}` | E-1/H: cross-ligand kappa donor pairs scored by the physical Q6 repulsion, crystal-derived onset scales. H-3b (local 2060, seeds 11-13, paired, `research/bench/jgen_campaign/H3B_REPORT.md`): cal top-1 -2.82 kJ/mol [-8.16, +0.23]; without FEKKAJ -0.43 [-1.37, +0.43] (neutral); no parent > +25 on two seeds; FEKKAJ (two Cl- collapsing, Cl-Ru-Cl 30-55 deg vs crystal 154) -291 kJ/mol on L4 (H-3). Ordinary-CUDA replay differences equal eager-repeat noise on production control and on the donor arm (FEKKAJ eager-repeat 0.27 vs replay 0.20); deterministic CUDA gives exact parity. Folded in 2026-10-31 (`3f75c9525a4e`). |
| `eta2_face_plane` | `jacobik.stats.orientation.ETA2_FACE_PLANE` | `False` | `False` | `True` | eta2 faces with >= 2 substituents on each atom: face normal from the plane through the bond fitting the substituents (S_orient and lean DOF; tilt, bend, slip); goes with `data/db/constraints_2026-11-01.json` (eta2 keys refit on every eta2 face). Holdout (110 `surface_fit` parents, 2060, seeds 11-13, `research/bench/jgen_campaign/ETA2_RESULT.md`): A top-1 -9.6 kJ/mol [-20.5, +0.4], B +17.2 [-6.7, +48.7]; no geminal collapse; edge-on rows on A 172 -> 299 (gate failed, owner accepted; loader M...C contacts in the eta2 class, far-field / typing package). Folded in 2026-11-01 (`273f2ccbf892`, with the two rows below). |
| `eta2_group_bend` | `jacobik.ik.angle_bends.ETA2_GROUP_BEND` | `False` | `False` | `True` | eta2 faces without the shared plane: a face atom's substituents bend as one rigid group (TEBWIH geminal collapse under the haptic cap). Folded in 2026-11-01 (`273f2ccbf892`). |
| `embed_haptic_li_cap` | `jacobik.engine.EMBED_HAPTIC_LI_CAP` | `False` | `False` | `True` | H-2: temporary Li caps on eligible anionic haptic faces during the GFN2 embed. Holdout amendment 3: cap-eligible top-1 -7.5 kJ/mol [-29.7, +14.9] (improvement gate failed, heavy-tailed: 21 better / 11 worse by > 10), no geminal collapse with the group bend. Folded in 2026-11-01 (`273f2ccbf892`). |
| `finish_phase` | `jacobik.ik.ik_solver.FINISH_PHASE` | `None` | `None` | IA-k dictionary (see profile header) | Owner-approved G-3g, 2026-11-02: iRprop+ search and finish, switch 100, symmetry-only release, thermal widths; shape/projection retained. G-3f L4 cal top-1 -10.9 [-18.7,-4.6] kJ/mol, floor -9.5, end-to-end wall 1.22x. Registered regression guard failed (3 vs threshold2); audit found degenerate C1-repeat null, seed-rotation null0-2, mirror improvement16 vs3 regressions. Owner accepted failure. Production `5f74b14b57ed`; old C1 via `finish_phase=null`, identity `273f2ccbf892`. R2s remains available off, reserved for geometry hypotheses. |
| `embed` | `jacobik.engine.DEFAULT_EMBED` | `"gfn2"` | `"gfn2"` | `"gfn2"` | GFN2-xTB crude optimization of free ligand conformers with frozen donor distances. |
| `deterministic_embed` | `jacobik.engine.DETERMINISTIC_EMBED` | `False` | `False` | `True` | Pins `OMP_NUM_THREADS=1` during GFN2 embed refinement, eliminating seed nondeterminism under multithreaded CPU load. |
| `chelate_ez` | `jacobik.chelate_embed.CHELATE_EZ` | `True` | `True` | `True` | Enforces cis double bond configuration in chelate rings $\le 7$ atoms (predicts 22/22 bonds in N48). |
| `carbene_ylide` | `jacobik.chelate_embed.CARBENE_YLIDE` | `True` | `True` | `True` | Embeds heteroatom-stabilized carbenes in their $X^+=C$ ylide form. |
| `eta2_cap` | `jacobik.chelate_embed.ETA2_CAP` | `False` | `False` | `False` | Capping $\eta^2$ sites hurt weakly bound ligands (DUFLAR 567 vs 335 kJ/mol); disabled. |
| `bound_surrogate` | `jacobik.chelate_embed.BOUND_SURROGATE` | `False` | `True` | `True` | Embeds group-13 boryl/aluminyl donors as bound H-capped forms (e.g. H-Bpin instead of unphysical $B=[O^+]$ double bonds). |
| `ez_enumerate` | `jacobik.chelate_embed.EZ_ENUMERATE` | `False` | `True` | `True` | Solves open double-bond E/Z permutations that SMILES leaves open and chelate rule cannot fix (SOKVET, CBTYCO). |
| `branch_bends` | `jacobik.ik.fk.BRANCH_BENDS` | `True` | `True` | `True` | Allows bend degrees of freedom on branch junctions at kappa donors. |
| `branch_prior` | `jacobik.ik.fk.BRANCH_PRIOR` | `False` | `False` | `False` | Prior on branch bends tested, not adopted. |
| `ring_open_adj` | `jacobik.ik.fk._SMALL_RING_OPEN_ADJ` | `True` | `True` | `True` | Opens small donor rings next to kinematic joints for proper torsion degrees of freedom. |
| `ring_joints` | `jacobik.topology.decompose.RING_JOINTS` | `True` | `True` | `True` | Models non-aromatic non-fused small donor rings as torsion joints. |
| `donor_bends` | `jacobik.ik.angle_bends.DONOR_BENDS` | `False` | `False` | `False` | Substituent branch bends at kappa donors tested; neutral on N48, ranking worse; disabled. |
| `haptic_bends` | `jacobik.ik.angle_bends.HAPTIC_BENDS` | `True` | `True` | `True` | Enables bend degrees of freedom at haptic faces. |
| `ring_fold_hinges` | `jacobik.ik.angle_bends.RING_FOLD_HINGES` | `True` | `True` | `False` | Validated fold-hinge removal: Colab L4 tail $\Delta e = -336$ kJ/mol, cal $\Delta e = -13$ kJ/mol, no parent loss $> 10$ kJ/mol (roadmap §2.6). Skips `("hinge", 0)` DOFs. |
| `haptic_free_lean_max` | `jacobik.ik.angle_bends.HAPTIC_FREE_LEAN_MAX` | `6` | `6` | `6` | Face sizes up to 6 heavy atoms have substituent lean statistics-priced by S_orient. |
| `haptic_lean_wall` | `jacobik.ik.angle_bends.HAPTIC_LEAN_WALL` | `True` | `True` | `True` | Restrains haptic substituent lean within empirical data limits, preventing 180-deg substituent flip through ring plane. |
| `haptic_lean_all` | `jacobik.ik.angle_bends.HAPTIC_LEAN_ALL` | `False` | `True` | `True` | Extends lean wall to all face sizes ($\eta^4$–$\eta^8$), fixing inverted Cp/arene hydrogens (YAVFEG 1200 kJ/mol). |
| `local_plane_deg` | `jacobik.ik.angle_bends.LOCAL_PLANE_DEG` | `20.0` | `20.0` | `20.0` | Planarity angle tolerance (deg) for local-plane definition. |
| `beam_mode_score` | `jacobik.ik.beam.BEAM_MODE_SCORE` | `"density"` | `"density"` | `"density"` | Ranks torsion mode seeds by log von Mises density at mode mean ("mass" tested, not adopted). |
| `out_dedup` | `jacobik.ik.dedup.OUT_DEDUP` | `True` | `True` | `True` | Symmetry-aware heavy-atom RMSD deduplication of returned conformers. |
| `dedup_rmsd` | `jacobik.ik.dedup.DEDUP_RMSD` | `0.5` | `0.5` | `0.5` | 0.5 Å RMSD deduplication threshold. |
| `out_window` | `jacobik.engine.OUT_WINDOW` | `200.0` | `200.0` | `200.0` | Drops conformers $> 200$ summed nats below the top conformer; eliminates $>90\%$ of pathological conformers ($>1000$ kJ/mol). |

---

## Retired Environment Switches

The following environment variables previously set solver switches and have been permanently retired:
- `JACOBIK_SCORE_AGG`
- `JACOBIK_RANK_AGG`
- `JACOBIK_GMM_FLOOR`
- `JACOBIK_RANK_FLOOR`
- `JACOBIK_Q1_TAIL_SIGMA`
- `JACOBIK_Q1_WALL`
- `JACOBIK_Q5_FOLD`
- `JACOBIK_Q6_MODEL`
- `JACOBIK_Q6_AGG`
- `JACOBIK_Q6_PMF_WELLS`
- `JACOBIK_HAPTIC_PER_SUB`
- `JACOBIK_EMBED`
- `JACOBIK_CHELATE_EZ`
- `JACOBIK_CARBENE_YLIDE`
- `JACOBIK_ETA2_CAP`
- `JACOBIK_BRANCH_BENDS`
- `JACOBIK_BRANCH_PRIOR`
- `JACOBIK_RING_OPEN_ADJ`
- `JACOBIK_RING_JOINTS`
- `JACOBIK_DONOR_BENDS`
- `JACOBIK_HAPTIC_BENDS`
- `JACOBIK_HAPTIC_FREE_LEAN_MAX`
- `JACOBIK_HAPTIC_LEAN_WALL`
- `JACOBIK_LOCAL_PLANE_DEG`
- `JACOBIK_OUT_DEDUP`
- `JACOBIK_DEDUP_RMSD`
- `JACOBIK_OUT_WINDOW`

---

## Archival of Dead Code Paths

### 1. Legacy Oxidation State Model (`src/jacobik/core/oxidation.py`)
```python
# DELETED from ligand_ionic_charge()
    ring_atoms = set()
    z = 0
    for grp in donors:
        if len(grp) < 2:
            continue
        syms = {mol.GetAtomWithIdx(i).GetSymbol() for i in grp if i < mol.GetNumAtoms()}
        if syms != {"C"}:
            return None, "haptic_non_carbon"
        order = len(grp)
        ring_z = {5: -1, 6: 0, 3: -1, 4: 0, 2: 0}.get(order)
        if ring_z is None:
            return None, f"haptic_order_{order}"
        z += ring_z
        ring_atoms.update(grp)
    for a in mol.GetAtoms():
        if a.GetIdx() in ring_atoms:
            continue
        z += a.GetFormalCharge() - a.GetNumRadicalElectrons()
    return z, "ok"
```

### 2. Ring-Fold Hinges Block (`src/jacobik/ik/angle_bends.py`)
```python
# DELETED from angle_bends.py (RING_FOLD_HINGES = False in production)
                    if RING_FOLD_HINGES:
                        out.append((a1, a2, a2, sorted(mv), ("hinge", 0), tuple(follow)))
```

### 3. NCI Scoring Runtime Calls (`src/jacobik/ik/scorer.py`)
```python
# DELETED from scorer.py (nci = False in production; module moved to research/interactions/nci.py)
        self._nci = None
        from jacobik.stats import nci as _nci
        if _nci.NCI_ON and hasattr(stats, "nci_tables"):
            self._nci = _nci.build_runtime(mol, stats.nci_tables(), device)

        if self._nci is not None:
            from jacobik.stats.nci import evaluate as _nci_eval
            comp["NCI"] = _nci_eval(self._nci, X) / self._agg_norm()
```

### 4. BVS Coupling Term (`src/jacobik/ik/scorer.py`)
```python
# DELETED from scorer.py (bvs = False in production; module moved to research/interactions/bvs.py)
        self._bvs = None
        if BVS and hasattr(stats, "bvs_tables"):
            self._bvs = _bvs_runtime(mol, q1_keys, *stats.bvs_tables(), device)

        if self._bvs is not None:
            sites, r0, c, mu, ref = self._bvs
            q1_bv = self._ge.q1_distances(X, donor_pos=donor_pos)[:, sites]
            x = torch.exp((r0[None] - q1_bv) / BVS_B).sum(-1) / ref
            comp["QV"] = -0.5 * c * (x - mu) ** 2 / self._agg_norm()
```
