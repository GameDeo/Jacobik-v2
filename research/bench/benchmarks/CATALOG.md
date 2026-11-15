# Baseline complexes

The primary workload is now the specialized [challenge panel](CHALLENGE.md).
These original cases remain available with `--suite baseline`.

All formulas describe the isolated coordination unit; counterions and lattice
solvents are omitted. S = control, A = literature anchor, D = designed target.
`uhf` is the assumed number of unpaired electrons, not the spin multiplicity.
An unspecified shape means that only the coordination number is prescribed.

| ID | Complex | CN | Charge | uhf | Expected shape | Selection rationale |
| --- | --- | ---: | ---: | ---: | --- | --- |
| S01 | [ZnCl4]2- | 4 | -2 | 0 | tetrahedral | small anionic tetrahedral control |
| S02 | [NiCl4]2- | 4 | -2 | 2 | tetrahedral | high-spin tetrahedral d8; compare S03 |
| S03 | [PdCl4]2- | 4 | -2 | 0 | square planar | square-planar d8; compare S02 |
| S04 | [Ag(NH3)2]+ | 2 | 1 | 0 | linear | low coordination number |
| S05 | [Fe(H2O)6]2+ | 6 | 2 | 4 | octahedral | high-spin aqua control |
| S06 | [Cu(H2O)6]2+ | 6 | 2 | 1 | octahedral | Jahn-Teller distortion; isolated gas-phase ion may dissociate |
| A01 | [Co(en)3]3+ | 6 | 3 | 0 | octahedral | three flexible chelates and Delta/Lambda alternatives ([paper](https://doi.org/10.1107/S2056989021009336)) |
| A02 | [Zn(terpy)Cl2] | 5 | 0 | 0 | square pyramidal | rigid tridentate ligand and distorted five-coordinate core ([paper](https://doi.org/10.1107/S1600536814023605)) |
| A03 | [Co(NCS-N)4]2- | 4 | -2 | 3 | tetrahedral | explicit N-bound linkage; sulfur remains uncoordinated ([paper](https://doi.org/10.1107/S2056989023000981)) |
| A04 | [Cu(acac)2(ambim)] | 5 | 0 | 1 | square pyramidal | axial donor plus two delocalized O,O chelates ([paper](https://doi.org/10.1107/S2056989024011538)) |
| A05 | [Fe(oxalate)3]3- | 6 | -3 | 5 | octahedral | three tight dianionic chelates and metal chirality ([paper](https://doi.org/10.1107/S2056989018008022)) |
| A06 | [Ni(Hdmg)2] | 4 | 0 | 0 | square planar | intramolecular hydrogen bonds and proton-placement sensitivity ([paper](https://doi.org/10.1107/S1600536803024656)) |
| A07 | [Ni(NCS-N)2(3-methylpyridine)4] | 6 | 0 | 2 | octahedral | mixed monodentate donors; use ethanol-solvate crystal as anchor ([paper](https://doi.org/10.1107/S2056989022008891)) |
| A08 | [Zn(4'-Cl-terpy)2]2+ | 6 | 2 | 0 | octahedral | two rigid tridentates and peripheral chlorine atoms ([paper](https://doi.org/10.1107/S2414314622010963)) |
| D01 | [Fe(bpy)3]2+ | 6 | 2 | 0 | octahedral | low-spin polypyridyl target; contrast high-spin S05 |
| D02 | [Ru(bpy)3]2+ | 6 | 2 | 0 | octahedral | 4d analogue of D01; rigid chelate stereochemistry |
| D03 | [Cr(oxalate)3]3- | 6 | -3 | 3 | octahedral | metal substitution matched to A05 |
| D04 | [Cu(acac)2] | 4 | 0 | 1 | square planar | remove axial donor from A04 |
| D05 | [Al(acac)3] | 6 | 0 | 0 | octahedral | p-block metal with delocalized O,O chelates |
| D06 | [Zn(phen)Cl2] | 4 | 0 | 0 | tetrahedral | fused rigid chelate |
| D07 | [Zn(en)Cl2] | 4 | 0 | 0 | tetrahedral | flexible chelate paired with D06 |
| D08 | [Ni(en)3]2+ | 6 | 2 | 2 | octahedral | metal and spin change relative to A01 |
| D09 | [Co(dien)2]3+ | 6 | 3 | 0 | octahedral | two flexible tridentates; mer/fac possibilities |
| D10 | [Fe(terpy)2]2+ | 6 | 2 | 0 | octahedral | two rigid tridentates; compare D09 |
| D11 | [Zn(tren)Cl]+ | 5 | 1 | 0 | trigonal bipyramidal | branched tetradentate wrap |
| D12 | [Cu(cyclam)]2+ | 4 | 2 | 1 | square planar | 14-membered macrocycle; ring conformer challenge |
| D13 | [Ni(cyclen)(H2O)2]2+ | 6 | 2 | 2 | octahedral | 12-membered macrocycle with two additional donors |
| D14 | [Cu(salen)] | 4 | 0 | 1 | square planar | mixed N2O2 tetradentate Schiff base |
| D15 | [Fe(EDTA)]- | 6 | -1 | 5 | octahedral | flexible hexadentate N2O4 ligand; fully deprotonated |
| D16 | [Pd(dppe)Cl2] | 4 | 0 | 0 | square planar | bulky phosphine chelate with phenyl torsions |
| D17 | [Ni(dmdtc)2] | 4 | 0 | 0 | square planar | small-bite S,S chelates and delocalization |
| D18 | [Zn(dipic)(H2O)2] | 5 | 0 | 0 | unspecified | mixed O,N,O rigid tridentate; competing five-coordinate shapes |
| D19 | [Mg(18-crown-6)]2+ | 6 | 2 | 0 | unspecified | small s-block ion in a large six-donor macrocycle; deliberate mismatch stress |
| D20 | [Ca(18-crown-6)(H2O)2]2+ | 8 | 2 | 0 | unspecified | larger s-block ion; high-CN macrocycle |
| D21 | [Y(acac)3(H2O)2] | 8 | 0 | 0 | unspecified | high-CN multichelate target without f-electron spin ambiguity |
| D22 | [La(oxalate)3(H2O)3]3- | 9 | -3 | 0 | unspecified | nine-coordinate f-block boundary case |
| D23 | [Lu(acac)3(phen)] | 8 | 0 | 0 | unspecified | bulky mixed chelates on a closed-shell lanthanide |
| D24 | [Cu(PMe3)3]+ | 3 | 1 | 0 | trigonal planar | three-coordinate soft-donor target |
| D25 | [Zn(diglyme)Cl2] | 5 | 0 | 0 | unspecified | flexible neutral O3 chelate |
| D26 | [ZrF7]3- | 7 | -3 | 0 | unspecified | seven-coordinate d0 inorganic boundary case |

The expected shapes of designed cases are targets to test, not experimental
assignments. Geometry labels and reference coordinates are withheld from the
primary Architector inputs. The archived references cover A01-A08 only.
See [PROTOCOL.md](PROTOCOL.md) for scoring and source-independence limits.
