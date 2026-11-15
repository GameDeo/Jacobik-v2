# DS findings log — mechanisms understood, not changed (2026-11-11)

## F1. LIPDEU: the all-donor QO fallback is a poor model for P donors, and the P family sits one row under the threshold
Group `QO:(2, 4)` (2 substituents, 4-membered ring), D2 raw table after charge/locked filters: 7,229 rows from 3,679 parents (the fitted D2 group had 5,742 rows after the
H-bond cut and LnQM sibling rule; the descriptive table here uses the filtered rows before those steps). Composition: N_amidinate 2,806, O_alkoxide 1,496, N_donor 962,
N_amidate 393 ... P_donor 46 (34 parents) and P_phosphine 3. `orientation.N_MIN = 50`, so the P family (49 rows) falls through every specific ladder level into the
ring-only fallback.
The P_donor features (g0, g1, g2) have means (-0.350, +0.046, -0.159) against the fallback mean (-0.710, -0.133, -0.510), a shift of 2.4, 1.1 and 3.3 fallback standard
deviations. The fallback is dominated by amidinate-like N rows (39%); any change of the typing that moves rows into N_amidinate changes the fallback and therefore
LIPDEU's score, which matches the observed QO margin shift of 0.538.
Implications (Step 7 candidates, not changes): (a) a QO partial-pooling estimate, shrinking the 49 P rows towards the fallback as Q1 does for sparse keys, instead of
all-or-nothing group membership; (b) a chemistry-conditioned intermediate level (pnictogen-donor family) with its own threshold; (c) lowering N_MIN would pick up the P family
(49 rows) but leaves the fit data-starved. The robustness consequence: any sparse-donor target depends on the composition of the all-donor fallback.

## F2. LUQVOK Q1 stress pilot (see DS_STRESS_PILOT.md)
The 22 erroneous observations are the whole mechanism for that key: removing their 11 parents reproduces the control table exactly and improves the selected
energy by about 130 kJ/mol on two seeds; random removal of up to 4 of the 71 supporting parents does not change the fit or the selection.

## F3. tmQM xTB geometry versus CSD (owner inspection, 22 disagreement structures)
tmQM refinement sometimes changes the binding mode away from CSD (DOKCEL: kappa1 tosylate becomes kappa2 and eta5 Cp slips to eta3; SOWPIE: N contacts replaced by F contacts;
KEXVUI: a Cp-derived ring and an N approach into binding position; KIFXUS: a second N binds; YAJWEN: octahedral drifts to square-planar plus one weak contact). The ledger
category `supported_contact_not_declared` (and weak declared donors) is the reproducible export for these parents; they are data-quality evidence for the NN and stereo
work, not fit exclusions.

## F4. Face detection
The face detector seeds carbons from the nearest-seed distance including sigma donors (NUHPEL: As sets the scale and pulls in its phenyl carbons). Narrowing the scale to
pi-capable seeds fixes NUHPEL but removes sigma-donor sites downstream (census 31/34 -> 25/34), so the detector needs a restructure, not a one-line change. The evidence rule
under `donor_assignment` (no BO and no contact within ratio 1.21) removes the NUHPEL face without collateral effect on 2,500 sampled parents.

## F5. The charge filter removes most Zn/Ni/Cd parents on a SMILES charge-encoding artifact, and empties several Q1 keys (full corpus, `charge_os_census.json`)
Of 1,626 parents flagged implausible, Zn alone is 498 (Ni 199, Cd 151, Cu 97, Hg 86, Pd 70, Ag 67, Ti 60). The Zn records carry `metal_formal_charge` of +4 (293), +6 (142), +8 (27),
+10 (17), +14 (14) against a plausible range of -2..+2; for a d10 Zn(II) that is the charge written on the metal in the source SMILES (it accumulates with dative bonds),
not a chemical oxidation state. The check compares that written charge, so these parents are removed whole although their geometry is ordinary.
Effect on Q1 (125,044 parents, 98,269 keys, locked exclusions applied first): the filter reduces the independent support of 3,228 keys, never takes a key from at least 50
independent parents to below 50 (0 keys), takes 53 keys from at least 10 to below 10, and empties six Zn keys completely (22-38 parents each, e.g.
`('Zn', 1, 4, 'N_amine', 'square_pyr', 'N_amine')` 38 -> 0, `('Zn', 1, 4, 'Hal_Cl', 'tetrahedral', 'none')` 30 -> 0); V oxo (25 -> 9), Ni carboxylate/water (26 -> 10, 26 -> 13) and
Ti/Zr C_eta5 (34 -> 18, 21 -> 12) lose 40-64%. Keys with at least 20 parents and at least 15% lost: 17 (`research/data/ds_proto/charge_filter_q1_keys.csv`).
Consequence: Zn geometry cannot be learned for those keys and falls to broader ladder levels. This is the strongest candidate for the separately attributed charge arm
(DS_STEP1_POLICY): rebuild with the filter applied only to quantities that use the oxidation-state key, or with `metal_formal_charge` corrected for dative-bond accounting, and attribute the
change on its own. It is not in the four-arm build; the policy keeps the charge filter fixed across D0/D-data/D-sample/D-both.

## F6 (2026-11-11): the bond-order floor drops some real short contacts; accepted, no exception
Owner CSD review of dropped donors: SUYSIO and MAJSOF (Au-Se 3.35-3.40 A), LAWDUL N5 and HEKHEK N5 are stretched or broken in the tmQM geometry, so
dropping them is right for the target geometry. BUYBUT's Cu-S (2.61 A, ratio 1.1, tmQM BO 0.00) is a real bond the bond-order file misses.
A distance exception for non-carbon donors (ratio <= 1.21 with BO < 0.05) would restore 234 donors in 167 of ~108k parents (0.15%);
the 794 carbon cases at the same distance are mostly slipped Cp carbons that the face detector handles. Decision (owner): keep the rule
"a donor needs bond-order evidence (BO >= 0.05) within ratio 1.6" with no element- or distance-specific exception. The 167 parents stay a known,
bounded defect. The tried exception is commit e3f027d, reverted.
