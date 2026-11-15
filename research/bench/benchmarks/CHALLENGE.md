# Specialized challenge panel

**29 targets, 39 ligand definitions, 12 research papers.** The original 40 cases remain a baseline. This is the primary panel for specific construction and sampling failures.

Fourteen inputs are bespoke designs, eight are chemical inputs taken from research papers, and seven use established motifs. A paper/motif input is not automatically an experimental reference or proof that every proposed isomer is stable.

The offline archive includes a graph-mapped 55-heavy-atom experimental Zr-HOPO structure (D03) and twelve mapped published computed geometries spanning three labeled arrangement/fold families for each of B03 and B05. The two optimization methods are recorded separately. They are not twelve independent thermally populated states.

The 25 `direct` cases have source-checked Architector input exports. S03 additionally exports with `--no-final-relax`. H04-H06 retain full chemical targets but need richer haptic adapters. None of these labels means that Linux Architector execution has already passed.

The chemistry checks cover both RDKit and Open Babel, exact donor identities, formal charges, electron parity, pi-group/site bookkeeping and preservation of D02's four S carbon stereocenters.

## What makes these useful

| Family | Cases | Failure being tested |
| --- | --- | --- |
| Pi binding | H01-H06 | Complete ring faces, fused-ring selection, ring slippage, ansa tethers, open-chain allyl and two pi groups in one diene |
| Sulfur / binding mode | S01-S05 | Crowded/asymmetric N2S, sulfur stereochemistry, measured SNS isomer exchange, one molecule using two donor sets, polar N2S2 macrocycles |
| Bite constraints | B01-B05 | Four-membered chelate, flexible wide bite, crowded heteroleptic Cu, and Xantphos versus spacer-extended trans-spanning XL-Xantphos |
| High denticity | D01-D05 | Octadentate tetraamides and HOPO, nonadentate N6O3 cage, fixed chirality and ten-contact overpacking |
| Polar / secondary sphere | P01-P03 | Directional urea-hydroxo hydrogen bonds, a matched NH-blocked analogue, remote sulfonates and solvent-sensitive collapse |
| Multiple arrangements | I01-I03 | Asymmetric tridentates, cyclometalated C,N fac/mer arrangements and coupled cis/trans/sulfur configurations |
| Dinitrogen | N01-N02 | End-on N2 orientation, cis/trans placements, flexible phosphines and a sterically protected triamidoamine pocket |

## Full selection

For pi targets, `contacts` counts metal-bound atoms and `slots` is the separate Architector template convention. One aromatic pi ring uses three facial slots; it is not modeled as a collection of independent sigma donors.

| ID | Target | Contacts / slots | Specific challenge | Current input adapter |
| --- | --- | --- | --- | --- |
| H01 | [Fe(eta5-Cp*)2] | 10 / 6 | Haptic positive control: preserve both five-carbon faces, vary ring rotation without treating ten contacts as CN=10. [ARCH](https://doi.org/10.1038/s41467-023-38169-2) | direct |
| H02 | [Ru(eta6-p-cymene)(6-Me-NS-Et)Cl]+ | 9 / 6 | Piano stool with an asymmetric crowded N,S chelate: arene face, Ru configuration, and thioether configuration are coupled. [ARCH](https://doi.org/10.1038/s41467-023-38169-2) [SNS23](https://doi.org/10.1002/ejic.202300313) | direct |
| H03 | [Rh(eta5-indenyl)(CO)2] | 7 / 5 | Select the five-membered face of a fused aromatic system; diagnose eta5-to-eta3 slippage separately from successful preservation. [ARCH](https://doi.org/10.1038/s41467-023-38169-2) | direct |
| H04 | rac/meso-[Zr(ethylene-bis-indenyl)Cl2] | 12 / pending | One covalently tethered dianion spans two eta5 faces; recover rac/meso families while preserving the bridge.  | requires adapter |
| H05 | [Pd(eta3-cinnamyl)(Xantphos)]+ | 5 / pending | Open-chain eta3 allyl, substituted-end orientation, and a hinged wide-bite chelate; distinguish eta1 collapse. [ALLYL](https://doi.org/10.1016/S1631-0748(02)01406-6) | requires adapter |
| H06 | [Rh(eta2,eta2-COD)(P(o-tolyl)3)Cl] | 6 / pending | Two alkene pi groups in one flexible ring must both bind; ring conformation and phosphine shielding are coupled.  | requires adapter |
| S01 | [Cu(6,6'-Me2-N2S)Cl2] | 5 / 5 | Two crowded pyridine donors and a soft thioether compete to close the same coordination pocket. [RUSULF](https://doi.org/10.1039/DT9910002335) | direct |
| S02 | [Cu(asym-Me/OMe-N2S)Cl2] | 5 / 5 | Constitutionally different arms make bound sulfur stereogenic; avoid promoting the methoxy oxygen into the specified donor set. [RUSULF](https://doi.org/10.1039/DT9910002335) | direct |
| S03 | [Ru{HN(CH2CH2SEt)2}Cl(H)(PPh3)] | 6 / 6 | Published amine-based SNS complex with three crystallographically characterized sulfur diastereomers and measured exchange. [SNS23](https://doi.org/10.1002/ejic.202300313) | assembly only |
| S04 | [Ru(thienyl-bpy-kappa2N,N)(thienyl-bpy-kappa3N,N,S)Cl]+ | 6 / 6 | The identical chemical ligand occurs twice with different donor sets; one thiophene sulfur binds and the other remains unbound. [RUSULF](https://doi.org/10.1039/DT9910002335) | direct |
| S05 | [Cu(sulfonated-asym-N2S2-macrocycle)Cl] | 5 / 5 | An asymmetric N2S2 ring carries a sulfonate-bearing benzyl arm; prevent spectator oxygens from replacing the intended donors.  | direct |
| B01 | [Pd(dppm)Cl2] | 4 / 4 | Four-membered Pd-P-C-P chelate is an intentionally narrow-bite contrast to the spanning phosphines. [ALLYL](https://doi.org/10.1016/S1631-0748(02)01406-6) | direct |
| B02 | [Pd(DPEphos)Cl2] | 4 / 4 | Large P-P span with an ether hinge and competing ether oxygen; fixed P,P mode must survive folding. [ALLYL](https://doi.org/10.1016/S1631-0748(02)01406-6) | direct |
| B03 | [Pd(Xantphos)Cl2] | 4 / 4 | Probe cis versus trans spanning placements and xanthene folding; compare with the spacer-extended B05 ligand. [XLXANT](https://doi.org/10.1021/acs.organomet.9b00141) | direct |
| B04 | [Cu(Xantphos)(6,6'-Me2-bpy)]+ | 4 / 4 | Crowded N,N chelate and wide P,P chelate compete for a tetrahedral Cu(I) sphere; measure unintended flattening and phenyl packing. [CUPP](https://doi.org/10.3390/molecules24213934) | direct |
| B05 | [Pd(XL-Xantphos)Cl2] | 4 / 4 | Phenylene-extended Xantphos spans opposite metal sites; use B03/B05 as a matched test of local versus global bite constraints. [XLXANT](https://doi.org/10.1021/acs.organomet.9b00141) | direct |
| D01 | [Eu(DTMA)(H2O)]3+ | 9 / 9 | N4O4 tetraamide macrocycle has coupled ring inversion and pendant-arm rotation; retain the inner-sphere water. [DOTA09](https://doi.org/10.1021/ic9017008) | direct |
| D02 | [Eu(DOTA-tetra-sec-butylamide)(H2O)]3+ | 9 / 9 | Four specified sec-butyl carbon stereocenters bias the coupled macrocycle/arm arrangements; preserve every ligand stereocenter. [DOTA09](https://doi.org/10.1021/ic9017008) | direct |
| D03 | [Zr(3,4,3-LI(1,2-HOPO))] | 8 / 8 | A long four-arm O8 chelator wraps the metal; select ring O donors over linker amide O atoms and sample distinct backbone folds. [HOPO](https://doi.org/10.1021/acs.bioconjchem.5b00572) | direct |
| D04 | [La(TPEN)(kappa2-NO3)2]+ | 10 / 10 | Ten atomic contacts combine a crowded six-N cage with two acute-bite nitrates; a designed overpacking boundary case.  | direct |
| D05 | [Lu(tpatcn)] | 9 / 9 | Nine N6O3 donors on a TACN-based tripod must all close; pyridyl orientation and carboxylate choices are coupled. [TPATCN](https://doi.org/10.1002/hlca.200900150) | direct |
| P01 | [FeIII(H3buea)(OH)]- | 5 / 5 | Three urea NH groups should organize around the hydroxo pocket; carbonyl O atoms remain outside the primary N4 donor sphere. [BUEA](https://doi.org/10.1021/ja061618x) | direct |
| P02 | [FeIII(N-methyl-H3buea-analogue)(OH)]- | 5 / 5 | A custom NH-blocked partner to P01 changes the directional hydrogen-bond network while retaining the same N4 primary donor topology. [BUEA](https://doi.org/10.1021/ja061618x) | direct |
| P03 | [Cu(trisulfonato-TPA)(H2O)]- | 5 / 5 | Three remote sulfonates make many polar oxygen atoms available without changing the intended N4 tripod; test collapse and donor swapping.  | direct |
| I01 | [Fe(asym-N2O-Schiff-base)2]+ | 6 / 6 | Two directional tridentates create head-tail, fac/mer-like, and chiral arrangement alternatives; peripheral methoxy groups add conformational choices.  | direct |
| I02 | [Ir(difluoro-tert-butyl-ppy)3] | 6 / 6 | Three unsymmetrical C,N chelates require carbon sigma donation and fac/mer family sampling with fluorinated, bulky outer groups.  | direct |
| I03 | [Pd(6-Me-NS-Et)2]2+ | 4 / 4 | Square-planar cis/trans N/S arrangements combine with two independent sulfur configurations and constrained chelate folds. [SNS23](https://doi.org/10.1002/ejic.202300313) | direct |
| N01 | [Mo(N2)2(depe)2] | 6 / 6 | Two end-on N2 ligands share a flexible four-phosphorus environment; preserve N-N connectivity and sample N2 placement alternatives. [MO69](https://doi.org/10.1039/C29690001392) | direct |
| N02 | [MoIII(Dipp-N3N)(N2)] | 5 / 5 | Custom reduced-size aryl-shielded triamidoamine pocket challenges approach to N2 and collective aryl rotation; low-spin input is an explicit assumption for this derivative. [MOEPR](https://doi.org/10.1021/ja1004619) | direct |

## Evidence for multiple states

S03 is the experimental exchange anchor: three isolated sulfur diastereomers, interconversion by NMR, and activation Gibbs energies of 65-70 kJ/mol. The barriers are not differences in isomer free energy. [Rummel et al.](https://doi.org/10.1002/ejic.202300313)

D01/D02 address SAP/TSAP forms and coupled arm/ring motions of DOTA tetraamides. The author-specified all-S sec-butyl derivative adds steric and stereochemical constraints. Do not transfer an equilibrium ratio without matching the exact derivative and experimental conditions. [Mani et al.](https://doi.org/10.1021/ic9017008)

B03/B05 have published cis/trans/folded computed coordinate sets. The source reports opposite computed cis/trans preferences for Xantphos and XL-Xantphos; the archived geometries themselves have P-Pd-P angles of approximately 101-161 degrees. This is a reference for geometry/isomer search, not a measured thermal population. [Whiteker et al.](https://doi.org/10.1021/acs.organomet.9b00141)

The custom cases' alternative arrangements are explicit hypotheses. Their stability and thermal accessibility must be established by a shared optimization and free-energy protocol before claiming population coverage.

## Running and reviewing

```sh
pixi run --environment audit validate
pixi run --environment audit python scripts/run_architector.py --suite challenge --all --dry-run
# Linux runtime; example with hydride-compatible policy:
pixi run python scripts/run_architector.py --suite challenge --case S03 --no-final-relax --n-conformers 25
```

The runner defaults to the challenge suite. Use `--suite baseline` for the original catalog. Every job records its suite and exports the complete target, even where no faithful input adapter is available.

Definitions, groups, stereochemistry and evidence live in `data/challenge/`. See [CHALLENGE_PROTOCOL.md](CHALLENGE_PROTOCOL.md) for scoring, solvent choices, budgets and denominators. No tmQM inputs were used; molecular-identity overlap has not been audited.

## Research sources

| ID | Paper | How it is used |
| --- | --- | --- |
| ARCH | [Architector for high-throughput cross-periodic table 3D complex building (2023)](https://doi.org/10.1038/s41467-023-38169-2) | API and capability context; aromatic sandwich groups use three facial template slots. Not evidence for a particular designed molecule. |
| SNS23 | [Slow Inversion of Coordinated Thioether Groups in SNS-Type Ruthenium Pincer Complexes (2023)](https://doi.org/10.1002/ejic.202300313) | Exact S03 connectivity and three sulfur diastereomers; experimentally measured exchange, with activation Gibbs energies 65-70 kJ/mol. |
| RUSULF | [Different bonding modes in octahedral complexes of 6-(2-thienyl)-2,2'-bipyridine (HL) with transition metal ions (1991)](https://doi.org/10.1039/DT9910002335) | Exact mixed N2/N2S coordination motif for S04. Related N2S targets S01/S02 are designed derivatives, not structures established by this paper. |
| ALLYL | [Synthesis and characterisation of bite angle-dependent (eta1-allyl)Rh and (eta3-allyl)Rh complexes bearing diphosphine ligands (2002)](https://doi.org/10.1016/S1631-0748(02)01406-6) | Experimental coupling of bite angle and allyl binding mode; catalytic relevance. Supplies mechanism context, not coordinates or a population reference for the authored Pd allyl target. |
| DOTA09 | [The Effect of the Regiochemistry of Butyl Amide Substituents on the Solution-State Structures of Lanthanide(III) DOTA-tetraamide Complexes (2009)](https://doi.org/10.1021/ic9017008) | Solution SAP/TSAP isomers and butyl-amide substitution effects; the cited DTMA 3.2:1 ratio is contextual until temperature and medium are recovered. The four fixed S centers of D02 are explicit authored stereochemistry. |
| HOPO | [p-SCN-Bn-HOPO: A Superior Bifunctional Chelator for 89Zr ImmunoPET (2015)](https://doi.org/10.1021/acs.bioconjchem.5b00572) | The paper reports a Zr complex of unfunctionalized 3,4,3-LI(1,2-HOPO) with all eight O donors bound. D03 uses that unfunctionalized chelator, not the antibody-conjugation derivative in the title. |
| TPATCN | [Water Stability and Luminescence of Lanthanide Complexes of Tripodal Ligands Derived from 1,4,7-Triazacyclononane: Pyridinecarboxamide versus Pyridinecarboxylate Donors (2009)](https://doi.org/10.1002/hlca.200900150) | Establishes the TACN/picolinate ligand family. This study examines Eu/Tb; D05's Lu substitution is a motif-level selection, not an exact reference structure from this paper. |
| BUEA | [X-ray Absorption Spectroscopy and Density Functional Theory Studies of [(H3buea)FeIII-X]n- (X = S2-, O2-, OH-) (2006)](https://doi.org/10.1021/ja061618x) | Primary/secondary sphere and hydrogen-bonded Fe-hydroxo chemistry for P01; P02 is a deliberately designed NH-blocked analogue. |
| MO69 | [A trans-dinitrogen complex of molybdenum (1969)](https://doi.org/10.1039/C29690001392) | Establishes a diphosphine-supported trans-Mo dinitrogen motif using dppe. N01 uses depe; this source alone does not establish its exact geometry or cis population. |
| MOEPR | [Experimental and Theoretical EPR Study of Jahn-Teller-Active [HIPTN3N]MoL Complexes (L = N2, CO, NH3) (2010)](https://doi.org/10.1021/ja1004619) | Low-spin d3 Mo in a protected triamidoamine pocket. N02 is a smaller Dipp-substituted designed model; measured spin/vibronic properties of HIPTN3N are not transferred as ground truth. |
| CUPP | [Hinged and Wide: A New P^P Ligand for Emissive [Cu(P^P)(N^N)][PF6] Complexes (2019)](https://doi.org/10.3390/molecules24213934) | Mechanism context for hinged/wide P,P chelates, interligand packing and Cu(I) luminescence. Not an experimental structure for every substituent combination in B04. |
| XLXANT | [XL-Xantphos: Design and Synthesis of a Mechanistic Probe of Xantphos O-Coordination in Catalytic Reactions (2019)](https://doi.org/10.1021/acs.organomet.9b00141) | Matched Xantphos/XL-Xantphos Pd(II) systems have opposite computed cis/trans preferences. A computed preference is neither a barrier nor a measured thermal population. |
