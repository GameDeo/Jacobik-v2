# P2R Step 1: class-aware A4b coverage metric (A4c), 2026-11-12

Directive `.claude/directive_p2r_metric_2026-11-12.md`. Branch `jgen/p2r-metric` (from `infra/gate-fixes` 92d9f53), code
`research/bench/tools/jgen/a4c/`, tests `tests/test_a4c_classes.py`, `tests/test_a4c_relax_select.py` (15 tests).
Bench code only; production fingerprint `d5e2f4113497` unchanged. Data (gitignored): `research/data/p2r/metric/`.
CPU only, workstation, <= 3 processes, one thread per numerical library. No GPU, Perlmutter compute or Colab.


## 0. Owner rulings 2026-11-12 (they supersede the earlier numbers in this report)
1. **Enantiomers are distinct classes.** A mirror-only match is reported as `enantiomer` and is never a hit (code and
   tests changed; controls rerun, `controls/run_v4/`, `controls/mirror_v2.json`: still 16/16 and 119/119).
2. **Coverage = the reference class is anywhere in the relaxed ensemble** (C-MISS is the gate number). The C-TOP1 /
   C-RANK split is diagnostic only: crystals are not necessarily the lowest configuration.
3. **Top-1 is judged by energy, in any class:** does top-1 relax to a basin at or below the reference? Reported as
   top-1 offset <= +5 and <= 0 kJ/mol.

The metric changes measurement only. No solver code changed (`src/` untouched, fingerprint `d5e2f4113497`); the numbers
re-measure the stored outputs of old production `273f2ccbf892`.

| cal (115 parents, 345 units) | A4b (old) | A4c, depth 2 (registered) | A4c, depth 1 |
|---|---|---|---|
| **coverage miss** | 0.365 [0.284, 0.446] | **0.319 [0.241, 0.400]** | 0.252 [0.180, 0.328] |
| top-1 relaxed offset <= +5 kJ/mol (any class) | | 0.409 [0.330, 0.487] | |
| top-1 relaxed offset <= 0 | | 0.246 [0.177, 0.319] | |

- **cal:** `strict_any` 0.748, `strict_all` 0.609.
- **tail** (9 parents): miss 0.815 -> 0.778; top-1 <= +5 is 0.111.
- **Enantiomers:** 14 cal units would be hits only through an enantiomer match. They are now misses: BOHWUT s12/13,
  BOTYUH s11-13, JUPDON s11, NUWFAM s11, PANBOY s11, QEZVEY s13, USAPOU s11/13, ZOJDEH s11-13.
- **Hit -> miss** (cal parents): BOHWUT, BOTYUH, HEDJOP, JUCBIV, JUPDON, LISVOC, PANBOY, QEZVEY, WEYJIT, ZOJDEH.
- **Old `equivalent` hits of another class:** 73 rows, 39 of them the enantiomer.
- **No old full-RMSD hit is an enantiomer** (a proper rotation cannot superimpose one).
- **Where the enantiomer misses come from:**
  - 12 of the 14 enantiomer-only units have an achiral coordination sphere; only ligand centres are mirrored (carbon
    centres: BOTYUH, QEZVEY, ZOJDEH; binding-created donor centres: BOHWUT, JUPDON, USAPOU). NUWFAM and PANBOY mirror
    both the sphere and the ligand centres;
  - coordination-sphere Delta/Lambda are enumerated (`embed_aware_dedup`, proper rotations, default since 2026-10-17);
  - current production's `stereo.enumerate_stereo_variants` collapses ligand mirror pairs in an achiral environment by
    design, so Step 2 will count these as misses by construction.
- **Fix, agreed for Step 4:** emit the mirror image of every chiral output class instead of collapsing pairs. Reflection
  is exact: same MACE energy, and the same score if the statistics are mirror-symmetric (check Q5 torsion signs). Step 2
  reports the miss rate both as solved and with reflected classes added (stored data: 0.319 and 0.284).
- Sections 3 and 5 below were written before these rulings; their bucket numbers count enantiomers as hits.

## 1. Controls (real cal parents)
Order deviation from the directive: the controls ran after a first retrospective pass, not before it.

| Control | Parents | Pass | Notes |
|---|---:|---:|---|
| Mirror self-consistency (no xtb): mirror image of the relaxed reference is the same class; a proper-rotation-superimposable mirror matches directly | 119 (all cal with a relaxed reference) | **119** | 79 configurationally achiral (direct match), 40 chiral (mirror match only) |
| Positive: up to 3 peripheral rotors rotated (outside the core by the graph rule), GFN2-relaxed (crude, 200 cycles) -> same class, core RMSD <= 0.3 A | 8 (BOCRAN, BOTYUH, CEHDUS, CIFZEZ, CIXYUD, DUHHIX, EQODOF, FASPOH; 3 automorphism-capped) | **8** | relaxed core RMSD 0.00-0.21 A. **7/8 relaxed constructs have full heavy RMSD > 0.3 A (0.31-3.45 A)**, so a full-RMSD hit misses them and the A4c core path finds them (BOTYUH's rotor is symmetric: full 0.03 A). 1-2 rotors applied per parent (the clean-construct rule rejected the rest) |
| Negative: one stereogenic non-donor C-H centre inverted, GFN2-relaxed -> never a direct match | 8 (BOTYUH, DAMVAQ, FUTSOD, JUPPOZ, LISVOC, PANBOY, QUGXUN, USAPOU) | **8** | 7/8 relaxed with graph and contacts kept, so the centre descriptor alone decided (FUTSOD, LISVOC, USAPOU, PANBOY, QUGXUN by `chi`); BOTYUH, JUPPOZ match as enantiomers, legitimately (rest of the complex verified achiral); DAMVAQ also changed contacts on relaxation (weak case). **The relaxed FUTSOD inversion product (a diastereomer) is 0.293 A from the reference in full heavy RMSD, an A4b full-RMSD hit; A4c rejects it at the centre.** JUPPOZ's enantiomer is 0.21 A |
| Synthetic unit tests | - | 13 + 2 | cis/trans (OC-6, SP-4), fac/mer, Delta/Lambda (mirror match), R/S with two centres, constructed inversion, E/Z, binding-created N, bound thioether (not a centre) vs bound P (centre), peripheral rotor (core RMSD 0), energy path needs the same class, topology change, contact noise vs real dissociation, relax-set rule and its cap |

First control run (16 relaxations): the rotor constructs were not clean. 5 of 8 already clashed under the 0.4 A bond rule
or touched the metal before relaxation, and DUHHIX and EDOBUV then relaxed into new bonds or contacts. Fixed in the construct, not in
the metric: a rotation is kept only if the start is clash-free (r_i + r_j + 0.6 A) and still the reference class. Rerun:
16/16 pass. Final run with full RMSDs recorded: `research/data/p2r/metric/controls/run_v3/` (earlier runs `run/`, `run_v2/`).

## 2. The metric as built
Unit = parent x seed; rows = the relaxed rows of that unit; the target is the GFN2-relaxed reference (A4b
`relax/references.json`). Hit = **same class** AND (**core RMSD <= 0.3 A** OR **offset <= +5 kJ/mol**), offset = the A4b
quantity (MACE energy of the GFN2-relaxed row minus MACE of the relaxed reference). Buckets C-TOP1 / C-RANK / C-MISS.

Same class = some automorphism p of the reference heavy-atom graph (VF2, `runkit.geom.automorphisms`, cap 4000). For the
17 capped parents, geometry-guided automorphisms are added from two searches:
- VF2 restricted to geometrically plausible pairs (same colour, no contact change, core and near-metal atoms within
  1.5 A after a contact-aware alignment);
- Kabsch + Hungarian within WL colour classes, kept only if graph-preserving.

The match is accepted
directly or under the global mirror, with all of:
1. **covalent graph** unchanged (incl. H; 0.4 A bond rule);
2. **contacts**: no atom crosses the A4b contact cutoff (r_M + r_d + 0.6 A) **with a change of > 0.3 A** in its metal
   distance (pure cutoff crossings flipped on 0.01 A noise: SESWIZ, YISQUM, MEJYIN);
3. **site arrangement**: labelled unit site vectors (eta groups -> centroid) Kabsch-aligned onto the reference (proper;
   improper for the mirror), nearest-site Hungarian assignment = identity;
4. **centres**: chirality sign (dead band 0.1-0.3, coplanar = unknown) at every 4-connected atom and 3-connected P/As/S/Se
   (free N excluded: fast inversion; bound thioether/selenoether excluded: inverts at 298 K, same rule as production
   `stereo.perceive_binding_created_donors`). Covers binding-created centres, P centres, eta-alkene faces, bridgeheads;
5. **E/Z**: sign of cos(dihedral) across C=C < 1.42 A and C=N/N=N < 1.32 A bonds between trigonal atoms outside rings of <= 7.

Core = M, donors, atoms <= 2 bonds from a donor, and shortest donor-donor paths (chelate rings). Core RMSD = min over the
class-matching automorphisms only. Thresholds 0.3 A, +5 kJ/mol and depth 2 were fixed in the directive and not changed.

**Changed during development (stated plainly):** three descriptor rules were revised after reading individual
retrospective rows that were obvious false misses (a 0.42 A row "of another class", full-RMSD hits "of another class"):
- cis/trans angle bands and triple products were replaced by the arrangement alignment (AWONOR);
- the contact rule got the 0.3 A shift requirement;
- bound thioethers are no longer centres (NAGHUC);
- after review, the restricted-VF2 guided search was added for capped groups. NUWFAM binds the symmetry-equivalent
  arm (atom 27 instead of 38); without a valid automorphism for that swap it read as a contact change. The union of
  both searches is kept: VF2 alone lost the guided automorphisms of QEZVEY, XAQCAT and NOYTUS.

The bucket tables were not used to choose between versions. Still, the stored A4b data was development data for these
definitions, so the retrospective numbers below are not an independent validation. The Step-2 baseline is the first
clean application.

## 3. Retrospective on stored A4b data (old production `273f2ccbf892`; validates the metric, not a production number)
Source `research/data/jgen/campaign/A4b/` (375 units, 124 parents). Locked-set check: no code in
`locked_exclusions.json` reserve. NEPNEF: no stored pools (recorded failure). Atom order: the recomputed full heavy RMSD of
all 1174 rows of uncapped parents reproduces the stored `rmsd_refrel_A` to 8e-14 A, so the stored `X` is in reference order.
Parent-clustered bootstrap, 10k resamples. Units with an A4b bucket: cal 345 (115 parents), tail 27 (9).

| cal | A4b (old) | A4c (new, depth 2) | A4c depth 1 (sensitivity) |
|---|---|---|---|
| TOP1 | 0.441 [0.359, 0.522] | 0.470 [0.388, 0.551] | 0.513 [0.429, 0.594] |
| RANK | 0.194 [0.139, 0.255] | 0.246 [0.183, 0.313] | 0.270 [0.206, 0.339] |
| **MISS** | **0.365 [0.284, 0.446]** | **0.284 [0.212, 0.359]** | 0.217 [0.151, 0.287] |

tail (9 parents, wide CIs): R-MISS 0.815 -> C-MISS 0.704. cal seed consistency (new): `strict_any` 0.800, `strict_all` 0.626.

If enantiomers counted as different classes (owner decision, section 5): cal C-MISS 0.319 [0.241, 0.400]. 14 units hit
only through mirror matches: BOHWUT s12/13, BOTYUH s11-13, JUPDON s11, NUWFAM s11, PANBOY s11, QEZVEY s13, USAPOU s11/13,
ZOJDEH s11-13.

Transitions (cal units): 38 R-MISS -> hit (23 to TOP1, 15 to RANK; plus 3 in tail), **all through core RMSD** (the floppy-rotor rescue). Codes: BOCRAN, EDOBUV,
EQODOF, HOHZAG, HOYBED, MEBHEK, MIDQEY, NUWFAM, OMAGAK, QAPFAS, SILROY, SIVBUV, TIKZOF, XIKGEE, XOJHIN, YABYEJ, YOVNEC,
ZAFPOP, ZAZREZ. 10 hit -> miss units (BOHWUT, HEDJOP, JUCBIV, LISVOC, PANBOY, WEYJIT): their old hit was an A4b `equivalent` row
(offset <= 5, any class) of **another class**. 34 such rows in all:
- 20 differ at a centre: a bound N-H in JUCBIV, a bound sulfoxide S in LISVOC, and the eta-alkene coordination face in
  WEYJIT, whose other-face rows are 15-48 kJ/mol *below* the reference;
- 7 differ in the site arrangement;
- 6 differ in contacts;
- 1 differs in E/Z.

No A4b full-RMSD hit (<= 0.3 A) is another class. One full-RMSD hit is not an A4c hit: MEJYIN s13 row 25, same class,
core RMSD > 0.3 A, offset > 5.

Row facts:
- 1300 relaxed rows: 298 hit by core RMSD, 262 by energy only;
- not the same class: contacts 84, centres 63, arrangement 27, E/Z 15, topology 1;
- 95 rows match only under the mirror;
- capped-parent contact rejections were re-checked without automorphisms (in-contact atom counts per WL colour): all
  are real except the two NUWFAM rows fixed above (HEDJOP's are real);
- core RMSD exceeds full RMSD by > 0.05 A in 133 rows. Core RMSD only uses class-consistent automorphisms, while A4b's
  full RMSD minimises over all of them, including relabellings that put a bound C=C on an unbound one (AWONOR: full 0.77
  A against core 1.25 A).

**Selection bias (as expected, so it is sized here):**
- the unrelaxed screen walks every pooled row A4b never relaxed (12024 rows; A4b relaxed only the 3 nearest rows within
  1.5 A full RMSD plus top-1 and floor);
- it finds 429 rows that are the reference class with unrelaxed core RMSD <= 0.5 A, in 99 units (57 with such a row
  beyond 1.5 A full);
- 11 of those units are cal C-MISS, so the retrospective under-counts miss -> hit by up to 11 cal units (about 0.03 of
  the cal C-MISS share).

The Step-2 rule below relaxes them. Depth 1 vs 2 changes 38 units; depth 2 stays registered.

Named cases (seed 11/12/13, A4b -> A4c):

| Case | A4b → A4c |
|---|---|
| YABYEJ | MISS/MISS/TOP1 → RANK/MISS/RANK (never-relaxed same-class rows in all seeds) |
| HOHZAG | all MISS → MISS/RANK/MISS (TOP1 at depth 1) |
| SILROY | all MISS → all TOP1 |
| USAPOU | all MISS → RANK/MISS/TOP1 (s11, s13 by enantiomer matches) |
| NUWFAM | all MISS → RANK/MISS/MISS (s11 by enantiomer match) |
| DOHTOM | all MISS at depth 2 (never-relaxed same-class rows in s11, s12); TOP1 at depth 1 |
| FEKKAJ | MISS at depth 2; RANK at depth 1 |
| FUTSOD, ZAQNUE, XACMOE, UYOVUA, LIPDEU | MISS everywhere |
| QEZVEY | MISS/MISS/TOP1 unchanged (s13 by enantiomer match) |
| LOPJEF, TEDPOI | TOP1 everywhere |

SOKVET and CBTYCO are not in the stored A4b set.

## 4. Relax-set rule for Step 2 (`a4c/relax_select.py`; A4b's `relax_set.select` unchanged)
At most 5 rows per unit, in this priority:
1. top-1;
2. floor;
3. up to 3 same-class rows nearest by **core** RMSD (<= 1.5 A). Only same-class rows can make the unit a hit, so they go
   first (changed after review: putting other classes first would bring the bias back);
4. the best-scored row of every other class present (greedy class representatives built on the reference graph), in the
   remaining slots.

Anything the cap drops is logged.

## 5. Not done here / open
- Class completeness per seed and within-class garbage (directive diagnostics) need the generator's class tuple on every
  row and a fresh solve: deferred to the Step-2 baseline, which can record them.
- **Resolved (owner 2026-11-12, section 0):** enantiomers are distinct classes. The text below is the original question.
  - cal C-MISS is 0.284 when it counts and 0.319 when it does not;
  - 14 units hit only through enantiomer matches;
  - your class list names Delta/Lambda separately, so the plain reading is that enantiomers are different classes,
    which would argue for counting them as misses. For a racemic crystal reference, though, the enantiomer is the same
    compound.
- The metric is sensitive to core depth (cal C-MISS 0.284 at depth 2, 0.217 at depth 1). Depth 2 is kept as registered.
  Step 3's miss classification should list the depth-1-only hits (DOHTOM, FEKKAJ, HOHZAG, ...) as "near-core" cases.
- Capped automorphism groups (17 parents) depend on guided automorphisms. Positive controls pass on 3 capped parents.
  False misses on capped parents are still possible; they are flagged per row (`guided`).
- WEYJIT: rows of the other alkene face are 15-48 kJ/mol below the reference (MACE). That is a score/energy question for
  Step 3, not a coverage miss of the reference class (those units are C-MISS for the reference class).
