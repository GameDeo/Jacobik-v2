"""Render the authored challenge selection as a reviewable Markdown catalog."""
from catalog import ROOT, load_catalog, read_json


def main():
    cases, ligands = load_catalog("challenge")
    sources = read_json(ROOT / "data/challenge/sources.json")["sources"]
    lookup = {source["id"]: source for source in sources}
    lines = ["# Specialized challenge panel", "",
             f"**{len(cases)} targets, {len(ligands)} ligand definitions, 12 research papers.** "
             "The original 40 cases remain a baseline. This is the primary panel for specific construction and sampling failures.", "",
             "Fourteen inputs are bespoke designs, eight are chemical inputs taken from research papers, and seven use "
             "established motifs. A paper/motif input is not automatically an experimental reference or proof that every proposed isomer is stable.", "",
             "The offline archive includes a graph-mapped 55-heavy-atom experimental Zr-HOPO structure (D03) and twelve "
             "mapped published computed geometries spanning three labeled arrangement/fold families for each of B03 and B05. "
             "The two optimization methods are recorded separately. They are not twelve independent thermally populated states.", "",
             "The 25 `direct` cases have source-checked Architector input exports. S03 additionally exports with "
             "`--no-final-relax`. H04-H06 retain full chemical targets but need richer haptic adapters. None of these labels "
             "means that Linux Architector execution has already passed.", "",
             "The chemistry checks cover both RDKit and Open Babel, exact donor identities, formal charges, electron parity, "
             "pi-group/site bookkeeping and preservation of D02's four S carbon stereocenters.", "",
             "## What makes these useful", "",
             "| Family | Cases | Failure being tested |",
             "| --- | --- | --- |",
             "| Pi binding | H01-H06 | Complete ring faces, fused-ring selection, ring slippage, ansa tethers, open-chain allyl and two pi groups in one diene |",
             "| Sulfur / binding mode | S01-S05 | Crowded/asymmetric N2S, sulfur stereochemistry, measured SNS isomer exchange, one molecule using two donor sets, polar N2S2 macrocycles |",
             "| Bite constraints | B01-B05 | Four-membered chelate, flexible wide bite, crowded heteroleptic Cu, and Xantphos versus spacer-extended trans-spanning XL-Xantphos |",
             "| High denticity | D01-D05 | Octadentate tetraamides and HOPO, nonadentate N6O3 cage, fixed chirality and ten-contact overpacking |",
             "| Polar / secondary sphere | P01-P03 | Directional urea-hydroxo hydrogen bonds, a matched NH-blocked analogue, remote sulfonates and solvent-sensitive collapse |",
             "| Multiple arrangements | I01-I03 | Asymmetric tridentates, cyclometalated C,N fac/mer arrangements and coupled cis/trans/sulfur configurations |",
             "| Dinitrogen | N01-N02 | End-on N2 orientation, cis/trans placements, flexible phosphines and a sterically protected triamidoamine pocket |", "",
             "## Full selection", "",
             "For pi targets, `contacts` counts metal-bound atoms and `slots` is the separate Architector template convention. "
             "One aromatic pi ring uses three facial slots; it is not modeled as a collection of independent sigma donors.", "",
             "| ID | Target | Contacts / slots | Specific challenge | Current input adapter |",
             "| --- | --- | --- | --- | --- |"]
    for case in cases:
        slots = case["architector_core_cn"] if case["architector_core_cn"] is not None else "pending"
        citations = " ".join(f"[{key}](https://doi.org/{lookup[key]['doi']})" for key in case["source_ids"])
        lines.append(f"| {case['id']} | {case['name']} | {case['metal_contact_count']} / {slots} | "
                     f"{case['challenge']} {citations} | {case['architector_support']['status'].replace('_',' ')} |")
    lines += ["", "## Evidence for multiple states", "",
              "S03 is the experimental exchange anchor: three isolated sulfur diastereomers, interconversion by NMR, "
              "and activation Gibbs energies of 65-70 kJ/mol. The barriers are not differences in isomer free energy. "
              "[Rummel et al.](https://doi.org/10.1002/ejic.202300313)", "",
              "D01/D02 address SAP/TSAP forms and coupled arm/ring motions of DOTA tetraamides. The author-specified "
              "all-S sec-butyl derivative adds steric and stereochemical constraints. Do not transfer an equilibrium "
              "ratio without matching the exact derivative and experimental conditions. "
              "[Mani et al.](https://doi.org/10.1021/ic9017008)", "",
              "B03/B05 have published cis/trans/folded computed coordinate sets. The source reports opposite computed "
              "cis/trans preferences for Xantphos and XL-Xantphos; the archived geometries themselves have P-Pd-P angles "
              "of approximately 101-161 degrees. This is a reference for geometry/isomer search, not a measured "
              "thermal population. [Whiteker et al.](https://doi.org/10.1021/acs.organomet.9b00141)", "",
              "The custom cases' alternative arrangements are explicit hypotheses. Their stability and thermal accessibility "
              "must be established by a shared optimization and free-energy protocol before claiming population coverage.", "",
              "## Running and reviewing", "", "```sh",
              "pixi run --environment audit validate",
              "pixi run --environment audit python scripts/run_architector.py --suite challenge --all --dry-run",
              "# Linux runtime; example with hydride-compatible policy:",
              "pixi run python scripts/run_architector.py --suite challenge --case S03 --no-final-relax --n-conformers 25",
              "```", "",
              "The runner defaults to the challenge suite. Use `--suite baseline` for the original catalog. "
              "Every job records its suite and exports the complete target, even where no faithful input adapter is available.", "",
              "Definitions, groups, stereochemistry and evidence live in `data/challenge/`. "
              "See [CHALLENGE_PROTOCOL.md](CHALLENGE_PROTOCOL.md) for scoring, solvent choices, budgets and denominators. "
              "No tmQM inputs were used; molecular-identity overlap has not been audited.", "",
              "## Research sources", "", "| ID | Paper | How it is used |", "| --- | --- | --- |"]
    for source in sources:
        lines.append(f"| {source['id']} | [{source['title']} ({source['year']})](https://doi.org/{source['doi']}) | {source['evidence']} |")
    (ROOT / "CHALLENGE.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Wrote CHALLENGE.md")


if __name__ == "__main__":
    main()
