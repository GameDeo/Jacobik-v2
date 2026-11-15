Common-ligand whitelist — rule and frozen list

  PRINCIPLE
  Exempt a ligand from scaffold grouping only if it is a generic, widely used ancillary ligand
  whose presence does not define the complex's identity, i.e. sharing it across splits is
  "shared chemistry", not a shared compound family. Every ligand NOT on the frozen list stays in
  scaffold grouping. Exempt ligands remain fully modeled in all NCI families.

  MATCHING
  - Exact graph match on the stripped ligand component (heavy-atom graph with elements; exact
    isomorphism or canonical SMILES of the fragment), not WL hash and not formula.
  - PARENT ligand only. Substituted variants (4,4'-dtbbpy, Me-bpy, PPh2(o-tol), substituted Cp
    other than Cp*) are NOT exempt; they stay grouped.
  - Ligands with < 5 heavy atoms (halides, CO, CN, NO, H2O, NH3, MeCN, OAc, DMSO…) are already
    stripped by the ≥5-heavy-atom rule and need no list entry.
  - Store each entry as {name, canonical graph key, heavy formula, sibling-family count}.

  EXEMPT (frozen list)
  - Monodentate phosphines/phosphites: PPh3, PEt3, PMe2Ph, PMePh2, PCy3, PiPr3, PtBu3, P(OMe)3,
    P(OPh)3
  - Cp-type: Cp, Cp*, indenyl
  - Neutral π-ligands: COD, NBD, benzene, p-cymene, mesitylene, hexamethylbenzene
  - Simple N-donors: pyridine, 4-picoline, DMAP, 2,2'-bipyridine (parent), 1,10-phenanthroline
    (parent), TMEDA
  - Simple O-donors: THF, Et2O, DME, acac (parent), hfac, triflate
  - Bis(diphenylphosphino)alkanes: dppm, dppe, dppp
  - Common NHCs (parent only): IMes, IPr, SIMes, SIPr                  [judgment call]
  - Silyl/alkyl ancillaries ≥ 5 heavy atoms: CH2SiMe3, N(SiMe3)2       [judgment call]

  NOT EXEMPT (always grouped)
  - Macrocycles: porphyrins, phthalocyanines, corroles, cyclam-type, crown ethers, cryptands,
    calixarenes
  - Tris(pyrazolyl)borates (Tp, Tp*), all pincers (PNP, PCP, NCN, NNN…), terpyridine
  - Salens/salophens and other tetradentate Schiff bases; tri-/tetraphosphines (triphos,
    tetraphos)
  - Any ligand with ≥ 3 donor atoms, except η5-Cp-type
  - Anything not explicitly listed above

  PROCEDURE
  1. Map each list entry to its graph key using the corpus. Report its sibling-family count and
     any list entry not found.
  2. Re-run grouping. Report the largest connected component (families and complexes) and the
     group-size distribution before and after. If a non-exempt ligand creates a component over
     ~5% of the eligible corpus, report it and propose a named addition. Do not raise a
     frequency threshold.
  3. Report exempt-ligand overlap fit vs cal/val/reserve and vs benchmarks (counts and
     fractions of representatives).
  4. Freeze the list (graph keys + sha256) in spec.json BEFORE re-sampling the splits.
     Additions after splits are drawn create a new frame version.

  The judgment calls, so you can decide:
  - NHCs (IMes/IPr). They're as ubiquitous as PPh₃ in modern organometallics. IPr (C27N2) is already in the top-25 by frequency. Grouping them would merge a large share of carbene chemistry into a few big groups. I'd exempt them. If you
    want "novel ancillary ligand class" to be part of what the validation split establishes, keep them grouped instead.
  - dppe/dppm/dppp. Chelating, but generic. Exempting them is consistent with exempting bpy.
  - p-cymene and the arenes. Exempting them is consistent with Cp; most of these are Ru/Os η⁶ complexes.
  - Terpy. I put it in the grouped list because it's tridentate and defines the complex's identity. That one is arguable too.

  Whatever you choose, the principle that matters is "not listed → grouped." That way the list fails safe toward stricter splits, and any future argument becomes a named addition rather than a threshold change.
