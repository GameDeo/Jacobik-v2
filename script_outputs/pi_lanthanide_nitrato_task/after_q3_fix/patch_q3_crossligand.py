import ast, sys

ROOT = sys.argv[1]

# ============================== engine.py ==============================
epath = f"{ROOT}/src/jacobik/engine.py"
with open(epath) as f:
    e = f.read()

old_block = '''        symmetric_groups: List[List[int]] = []
        try:
            from rdkit import Chem as _Chem

            _sym_key_to_donors: Dict[Tuple[str, int], List[int]] = {}
            _rank_cache: Dict[int, list] = {}
            for _di, (_li, _gi, _grp) in enumerate(all_donor_groups):
                if len(_grp) != 1:
                    continue
                if _li not in _rank_cache:
                    _rank_cache[_li] = list(
                        _Chem.CanonicalRankAtoms(rdmols[_li], breakTies=False)
                    )
                _offset = mol.ligands[_li].atom_indices[0]
                _local_idx = _grp[0] - _offset
                if _local_idx < 0 or _local_idx >= len(_rank_cache[_li]):
                    continue
                _rank = _rank_cache[_li][_local_idx]
                _smiles = self.ligands[_li].smiles
                _sym_key_to_donors.setdefault((_smiles, _rank), []).append(_di)
            symmetric_groups = [v for v in _sym_key_to_donors.values() if len(v) >= 2]
            if debug and symmetric_groups:
                print(f"\\n[DEBUG]   Symmetric donor groups (2.3.0): {symmetric_groups}")
        except Exception as exc:
            log.warning(
                "Symmetric-donor detection failed (%s); skipping similarity restraint.", exc
            )
            symmetric_groups = []
'''

new_block = '''        symmetric_groups: List[List[int]] = []
        q3_symmetric_groups: List[List[int]] = []
        try:
            from rdkit import Chem as _Chem

            _sym_key_to_donors: Dict[Tuple[str, int], List[int]] = {}
            _rank_cache: Dict[int, list] = {}
            _donor_ligand: Dict[int, int] = {}
            for _di, (_li, _gi, _grp) in enumerate(all_donor_groups):
                if len(_grp) != 1:
                    continue
                _donor_ligand[_di] = _li
                if _li not in _rank_cache:
                    _rank_cache[_li] = list(
                        _Chem.CanonicalRankAtoms(rdmols[_li], breakTies=False)
                    )
                _offset = mol.ligands[_li].atom_indices[0]
                _local_idx = _grp[0] - _offset
                if _local_idx < 0 or _local_idx >= len(_rank_cache[_li]):
                    continue
                _rank = _rank_cache[_li][_local_idx]
                _smiles = self.ligands[_li].smiles
                _sym_key_to_donors.setdefault((_smiles, _rank), []).append(_di)
            symmetric_groups = [v for v in _sym_key_to_donors.values() if len(v) >= 2]
            # BUGFIX (2.12.0): q3_symmetric_groups -- a within-single-ligand-
            # instance restriction of symmetric_groups, used ONLY for the Q3
            # approach-angle tie/mode-commit (sym_angle_weight, q3_mode_commit),
            # NOT for the M-D bond-length tie (sym_weight), which keeps using
            # the unrestricted symmetric_groups. Root cause this addresses:
            # symmetric_groups groups donors purely by (SMILES, canonical rank),
            # so N separate instances of the same repeated monodentate ligand
            # (e.g. 6 independent water molecules on [M(H2O)6]) land in ONE
            # group exactly like a single ligand's own internally-symmetric
            # donor arms (e.g. EDTA's 4 carboxylate O, all within one ligand
            # instance). For EDTA that grouping is correct: the four arms are
            # mechanically coupled through one shared backbone, so a genuine
            # "which mode do they jointly commit to" question exists, which is
            # exactly what q3_mode_commit (2.5.0) was built to answer. For 6
            # independent water ligands there is no such coupling -- nothing
            # chemically requires separate, freely-rotating monodentate donors
            # to share a single Q3 approach-angle mode, and the real fitted
            # Q3:('O_water', metal, geometry) distribution is itself
            # genuinely multi-modal (pooled over many different real crystal
            # environments). Confirmed empirically (this session): committing
            # all 6 waters of [Fe(H2O)6]3+ to one shared mode via the initial-
            # pose argmax reliably locked onto a LOW-weight minority mode
            # (~4% of the fitted mixture) purely because the (arbitrary,
            # embedding-derived) initial pose happened to sit closest to it --
            # producing a chemically nonsensical, wildly asymmetric M-O-H
            # geometry (one H at ~68 deg from the metal, the other at ~172 deg,
            # instead of the expected symmetric ~120/120) in every trial,
            # despite a HIGHER solve score than the correct, symmetric
            # geometry (q3_mode_commit=False on the same case gives clean
            # ~126/126 deg M-O-H angles). This is very likely the same class
            # of regression as the "fixed the CO/TBP donor-orientation
            # straightness, then a later change regressed simple monodentate
            # ligand orientation again" issue reported this session: 2.5.0
            # (q3_mode_commit) was layered on top of the Q2/Q3 GMM-cos refit
            # that made the CO/TBP fix work (2.4.0 changelog), specifically to
            # fix EDTA's within-ligand carboxylate multi-modality -- it was
            # never validated (see its own changelog) against a live solve of
            # a simple repeated-monodentate case like water, CN-, or CO.
            # Fix: restrict q3_mode_commit's (and the matching Q3 sym_angle_weight
            # tie's) group membership to donors that share a single ligand
            # INSTANCE, leaving cross-instance donors to each independently
            # score against the full (correctly weighted) Q3 mixture instead
            # of being hard-committed to one one arbitrarily-chosen mode.
            # EDTA-style single-ligand multidentate cases are completely
            # unaffected (all their donors already share one instance).
            q3_symmetric_groups = [
                g for g in symmetric_groups
                if len({_donor_ligand[_d] for _d in g}) == 1
            ]
            if debug and symmetric_groups:
                print(f"\\n[DEBUG]   Symmetric donor groups (2.3.0): {symmetric_groups}")
            if debug and symmetric_groups != q3_symmetric_groups:
                print(
                    f"\\n[DEBUG]   Q3-eligible symmetric groups, within-ligand-"
                    f"instance only (2.12.0): {q3_symmetric_groups}"
                )
        except Exception as exc:
            log.warning(
                "Symmetric-donor detection failed (%s); skipping similarity restraint.", exc
            )
            symmetric_groups = []
            q3_symmetric_groups = []
'''

assert old_block in e, "engine.py: symmetric_groups block not found verbatim"
e = e.replace(old_block, new_block, 1)

old_call = '''            symmetric_groups=symmetric_groups if use_shape_alm else None,
            q3_mode_commit=q3_mode_commit,
        )'''
new_call = '''            symmetric_groups=symmetric_groups if use_shape_alm else None,
            q3_symmetric_groups=q3_symmetric_groups if use_shape_alm else None,
            q3_mode_commit=q3_mode_commit,
        )'''
assert old_call in e, "engine.py: JointIKSolver.solve() call site not found verbatim"
e = e.replace(old_call, new_call, 1)

# Changelog entry
old_changelog_end = '''         though the underlying joint-construction math was verified
         standalone with a synthetic numpy test) -- see ik/fk.py's
         ForwardKinematics 2.4.0 changelog and the test-suite notebook for
         the intended EDTA / PtCl4 / Fe(CO)5 comparison against
         enable_bend_dof=False.
"""'''
new_changelog_end = '''         though the underlying joint-construction math was verified
         standalone with a synthetic numpy test) -- see ik/fk.py's
         ForwardKinematics 2.4.0 changelog and the test-suite notebook for
         the intended EDTA / PtCl4 / Fe(CO)5 comparison against
         enable_bend_dof=False.
2.12.0 : BUGFIX -- solve() now also builds q3_symmetric_groups (a within-
         single-ligand-instance restriction of symmetric_groups) and
         threads it through to JointIKSolver.solve() (ik_solver.py 2.8.0),
         used only for the Q3 approach-angle tie/mode-commit, not the M-D
         bond-length tie. Root cause and full rationale: see the inline
         comment at symmetric_groups' construction site above and
         ik_solver.py's 2.8.0 changelog. In short: q3_mode_commit (2.5.0)
         was validated only against EDTA (a single hexadentate ligand,
         donors mechanically coupled through one backbone); applied
         unmodified to N separate instances of a repeated monodentate
         ligand (e.g. [Fe(H2O)6]3+'s 6 independent waters), it hard-commits
         all of them to whichever Q3 GMM mode is nearest one arbitrary
         initial pose, with no requirement that mode be the dominant one --
         confirmed this session to reliably lock onto a ~4%-weight minority
         mode, producing a wildly asymmetric, chemically wrong ligand
         orientation (M-O-H angles ~68/172 deg instead of ~126/126) despite
         a higher raw solve score. Restricting q3_symmetric_groups to
         within-one-ligand-instance groups leaves EDTA-style cases (already
         single-instance) byte-identical while letting independent repeated
         monodentate ligands each score softly against the full, correctly-
         weighted Q3 mixture instead of being force-locked together.
"""'''
assert old_changelog_end in e, "engine.py: changelog tail anchor not found"
e = e.replace(old_changelog_end, new_changelog_end, 1)

with open(epath, "w") as f:
    f.write(e)
ast.parse(e)
print("engine.py patched OK")

# ============================== ik_solver.py ==============================
ipath = f"{ROOT}/src/jacobik/ik/ik_solver.py"
with open(ipath) as f:
    i = f.read()

# 1. signature
old_sig = '''        symmetric_groups: Optional[List[List[int]]] = None,
        q3_mode_commit: bool = True,
    ):'''
new_sig = '''        symmetric_groups: Optional[List[List[int]]] = None,
        q3_symmetric_groups: Optional[List[List[int]]] = None,
        q3_mode_commit: bool = True,
    ):'''
assert old_sig in i, "ik_solver.py: solve() signature not found verbatim"
i = i.replace(old_sig, new_sig, 1)

# 2. docstring addition
old_doc = '''        q3_mode_commit  : bool, default True (2.5.0). When True and a
                         symmetric group has >=2 Q3-eligible members, commit
                         that group to a single GMM component (chosen from
                         the initial pose) for the whole solve, instead of
                         letting each member independently draw from the
                         full (possibly multi-modal) Q3 mixture. See class
                         2.5.0 changelog. False restores pre-2.5.0 behaviour
                         byte-identically -- useful for A/B comparison.'''
new_doc = '''        q3_symmetric_groups : list of donor-index groups, same index
                         convention as symmetric_groups. Optional (2.8.0);
                         when given, used IN PLACE OF symmetric_groups for
                         the Q3 approach-angle tie and q3_mode_commit only
                         -- the M-D bond-length tie (sym_weight) always uses
                         the unrestricted symmetric_groups. When None
                         (default), falls back to symmetric_groups, i.e.
                         pre-2.8.0 behaviour. Callers should pass a
                         within-single-ligand-instance-only subset here;
                         see engine.py 2.12.0 changelog for why the Q3 tie
                         needs a narrower grouping than the bond-length tie.
        q3_mode_commit  : bool, default True (2.5.0). When True and a
                         q3_symmetric_groups (or, if not given,
                         symmetric_groups) group has >=2 Q3-eligible
                         members, commit that group to a single GMM
                         component (chosen from the initial pose) for the
                         whole solve, instead of letting each member
                         independently draw from the full (possibly
                         multi-modal) Q3 mixture. See class 2.5.0 and 2.8.0
                         changelogs. False restores pre-2.5.0 behaviour
                         byte-identically -- useful for A/B comparison.'''
assert old_doc in i, "ik_solver.py: docstring block not found verbatim"
i = i.replace(old_doc, new_doc, 1)

# 3. compute _q3_groups once, right before q3_site_to_idx map construction
old_anchor = '''        q3_site_to_idx: Dict[int, int] = {}
        if symmetric_groups:'''
new_anchor = '''        # 2.8.0: use the caller's narrower q3_symmetric_groups for the Q3
        # tie/mode-commit when given, else fall back to symmetric_groups
        # (pre-2.8.0 behaviour) -- see engine.py 2.12.0 changelog.
        _q3_groups = (
            q3_symmetric_groups if q3_symmetric_groups is not None else symmetric_groups
        )

        q3_site_to_idx: Dict[int, int] = {}
        if symmetric_groups:'''
assert old_anchor in i, "ik_solver.py: q3_site_to_idx anchor not found"
i = i.replace(old_anchor, new_anchor, 1)

# 4. mode-commit gating + loop
old_gate = '''        q3_mode_override: Optional[torch.Tensor] = None
        if q3_mode_commit and symmetric_groups and q3_site_to_idx:'''
new_gate = '''        q3_mode_override: Optional[torch.Tensor] = None
        if q3_mode_commit and _q3_groups and q3_site_to_idx:'''
assert old_gate in i, "ik_solver.py: mode-commit gate not found"
i = i.replace(old_gate, new_gate, 1)

old_loop1 = '''                    any_group_qualifies = False
                    for group in symmetric_groups:
                        q3_idxs = [
                            q3_site_to_idx[g] for g in group if g in q3_site_to_idx
                        ]
                        if len(q3_idxs) < 2:
                            continue
                        any_group_qualifies = True'''
new_loop1 = '''                    any_group_qualifies = False
                    for group in _q3_groups:
                        q3_idxs = [
                            q3_site_to_idx[g] for g in group if g in q3_site_to_idx
                        ]
                        if len(q3_idxs) < 2:
                            continue
                        any_group_qualifies = True'''
assert old_loop1 in i, "ik_solver.py: mode-commit loop not found"
i = i.replace(old_loop1, new_loop1, 1)

# 5. sym_angle_penalty tie loop
old_loop2 = '''                    if q3_site_to_idx:
                        q3_angles_now = None  # lazy: skip the extra fwd pass if no group qualifies
                        n_angle_groups = 0
                        for group in symmetric_groups:
                            q3_idxs = [q3_site_to_idx[g] for g in group if g in q3_site_to_idx]
                            if len(q3_idxs) < 2:
                                continue'''
new_loop2 = '''                    if q3_site_to_idx:
                        q3_angles_now = None  # lazy: skip the extra fwd pass if no group qualifies
                        n_angle_groups = 0
                        for group in _q3_groups:
                            q3_idxs = [q3_site_to_idx[g] for g in group if g in q3_site_to_idx]
                            if len(q3_idxs) < 2:
                                continue'''
assert old_loop2 in i, "ik_solver.py: sym_angle_penalty loop not found"
i = i.replace(old_loop2, new_loop2, 1)

# 6. changelog entry
old_tail = '''        sym_use_own_alm (default True) selects this mechanism.
        sym_weight_tracks_mu (2.6.0) is retained for backward
        compatibility and is only consulted when sym_use_own_alm=False;
        with both False, behaviour is byte-identical to pre-2.6.0 fixed
        weights. Not yet empirically validated against a live solve at
        the time of writing (implemented from the section-14 trace
        evidence above, since Claude cannot run torch in the environment
        this was developed in) -- see the test-suite notebook for the
        intended EDTA / PtCl4 / Fe(CO)5 A/B/C comparison against
        sym_use_own_alm=False (both the tracks_mu=True and tracks_mu=False
        sub-variants).
"""'''
new_tail = '''        sym_use_own_alm (default True) selects this mechanism.
        sym_weight_tracks_mu (2.6.0) is retained for backward
        compatibility and is only consulted when sym_use_own_alm=False;
        with both False, behaviour is byte-identical to pre-2.6.0 fixed
        weights. Not yet empirically validated against a live solve at
        the time of writing (implemented from the section-14 trace
        evidence above, since Claude cannot run torch in the environment
        this was developed in) -- see the test-suite notebook for the
        intended EDTA / PtCl4 / Fe(CO)5 A/B/C comparison against
        sym_use_own_alm=False (both the tracks_mu=True and tracks_mu=False
        sub-variants).
2.8.0 : solve() gains q3_symmetric_groups, an optional, narrower stand-in
        for symmetric_groups used ONLY by the Q3 approach-angle tie
        (sym_angle_penalty) and q3_mode_commit; symmetric_groups itself is
        untouched and still used as-is for the M-D bond-length tie
        (sym_penalty). None (default) falls back to symmetric_groups,
        preserving pre-2.8.0 behaviour byte-for-byte for any caller that
        doesn't pass it.

        Root cause addressed (found empirically this session, live solve --
        first live validation q3_mode_commit has had since its 2.5.0
        introduction): q3_mode_commit's "commit the whole group to whichever
        GMM component best explains its initial pose" logic is sound for a
        single ligand's own internally-symmetric donor arms (EDTA's 4
        carboxylate O, mechanically coupled through one backbone -- a
        genuine joint decision), which is the only configuration it was
        designed and reasoned through for. Called on a symmetric_groups
        entry spanning MULTIPLE separate ligand instances of a repeated
        monodentate ligand instead (e.g. all 6 waters of [Fe(H2O)6]3+,
        which land in one group purely because they share a SMILES and
        canonical rank), the same mechanism hard-locks chemically
        independent donors together with no chemical justification -- nothing
        requires 6 separate, freely-rotating water molecules to share one
        Q3 approach-angle mode, and the real fitted Q3 distribution for
        O_water is itself genuinely multi-modal (6 components, pooled from
        many different real crystal environments). Live-tested on
        [Fe(H2O)6]3+ (octahedral): the initial-pose argmax reliably selected
        a ~4%-weight minority mode identically for all 6 waters, every
        trial, regardless of seed -- producing an M-O-H geometry of ~68 deg
        / ~172 deg (one H nearly eclipsing the metal, the other pointing
        almost straight out) instead of the chemically expected symmetric
        ~126/126 deg, and doing so at a HIGHER solve score than the correct
        geometry (q3_mode_commit=False on the same case: clean ~126/126 deg,
        lower score) -- i.e. this is not an optimizer-quality problem, the
        committed objective itself was rewarding the wrong answer.

        This is a strong candidate for the "CO/TBP donor-orientation
        straightness fix, then a later change regressed it again" pattern
        reported this session: q3_mode_commit was layered on top of the
        very Q2/Q3 GMM-cos refit that made the original CO/TBP fix work
        (2.4.0 changelog), specifically to patch EDTA's within-ligand
        carboxylate multi-modality, and was never subsequently validated
        against a live solve of any simple repeated-monodentate case (its
        own 2.5.0 changelog says as much). See engine.py's 2.12.0 changelog
        for the q3_symmetric_groups construction (within-single-ligand-
        instance filter of symmetric_groups) that fixes this while leaving
        EDTA -- already single-instance -- byte-identical.
"""'''
assert old_tail in i, "ik_solver.py: changelog tail anchor not found"
i = i.replace(old_tail, new_tail, 1)

with open(ipath, "w") as f:
    f.write(i)
ast.parse(i)
print("ik_solver.py patched OK")
