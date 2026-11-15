"""Author the v2 challenge catalog from atom-mapped ligand definitions.

This is a deterministic curation script, not a random ligand generator.
Maps identify intended binding atoms; serialized SMILES retain the original
atom order, with maps removed without canonicalization.
"""
from __future__ import annotations

from collections import Counter
import re

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from catalog import ROOT, read_json, write_json


LIGANDS = {}


def parse_smiles(smiles):
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    return Chem.MolFromSmiles(smiles, parameters)


def ligand(key, name, mapped_smiles, donor_maps, family, *, eta_groups=None, notes=""):
    molecule = parse_smiles(mapped_smiles)
    if molecule is None:
        raise ValueError(f"Invalid authored SMILES: {key}")
    maps = {atom.GetAtomMapNum(): atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomMapNum()}
    if len(maps) != sum(bool(atom.GetAtomMapNum()) for atom in molecule.GetAtoms()):
        raise ValueError(f"Duplicate atom maps: {key}")
    smiles = re.sub(r":\d+(?=\])", "", mapped_smiles)
    plain = parse_smiles(smiles)
    if plain is None or [a.GetAtomicNum() for a in plain.GetAtoms()] != [a.GetAtomicNum() for a in molecule.GetAtoms()]:
        raise ValueError(f"Removing atom maps changed atom order: {key}")
    indices = [maps[number] for number in donor_maps]
    eta_groups = eta_groups or []
    pi_atoms = {number for group in eta_groups for number in group}
    groups = [{"kind": "kappa", "atom_indices": [maps[number]]} for number in donor_maps if number not in pi_atoms]
    groups.extend({"kind": "eta", "atom_indices": [maps[number] for number in group], "hapticity": len(group)} for group in eta_groups)
    Chem.AssignStereochemistry(plain, cleanIt=True, force=True)
    LIGANDS[key] = {
        "id": key, "name": name, "smiles": smiles, "authored_mapped_smiles": mapped_smiles,
        "donor_indices": indices, "donor_elements": [plain.GetAtomWithIdx(i).GetSymbol() for i in indices],
        "formal_charge": Chem.GetFormalCharge(plain), "family": family, "binding_groups": groups,
        "formula": rdMolDescriptors.CalcMolFormula(plain), "heavy_atom_count": plain.GetNumHeavyAtoms(),
        "radical_electrons": sum(atom.GetNumRadicalElectrons() for atom in plain.GetAtoms()),
        "fixed_stereocenters": [{"atom_index": atom.GetIdx(), "cip": atom.GetProp("_CIPCode")}
                                for atom in plain.GetAtoms() if atom.HasProp("_CIPCode")],
        "notes": notes,
    }


def define_ligands():
    ligand("cl", "chloride", "[Cl-:1]", [1], "monodentate")
    ligand("water", "water", "[OH2:1]", [1], "monodentate")
    ligand("hydroxide", "hydroxide", "[OH-:1]", [1], "monodentate")
    ligand("hydride", "hydride", "[H-:1]", [1], "hydride", notes="Explicit hydrogen is itself a coordinating atom; Architector 0.0.10 forces relax=False.")
    ligand("co", "carbon monoxide", "[C-:1]#[O+]", [1], "carbonyl")
    ligand("n2", "end-on dinitrogen", "[N:1]#N", [1], "dinitrogen")
    ligand("pph3", "triphenylphosphine", "[P:1](c1ccccc1)(c2ccccc2)c3ccccc3", [1], "bulky_phosphine")
    ligand("potol3", "tris(2-methylphenyl)phosphine", "[P:1](c1ccccc1C)(c2ccccc2C)c3ccccc3C", [1], "bulky_phosphine")
    ligand("cpstar", "pentamethylcyclopentadienyl", "C[c-:1]1[c:2](C)[c:3](C)[c:4](C)[c:5]1C", [1,2,3,4,5], "pi_ring", eta_groups=[[1,2,3,4,5]])
    ligand("pcym", "p-cymene", "C[c:1]1[cH:2][cH:3][c:4](C(C)C)[cH:5][cH:6]1", [1,2,3,4,5,6], "pi_ring", eta_groups=[[1,2,3,4,5,6]])
    ligand("indenyl", "indenyl", "[cH-:1]1[cH:2][cH:3][c:4]2cccc[c:5]21", [1,2,3,4,5], "pi_ring", eta_groups=[[1,2,3,4,5]])
    ligand("ebi", "ethylene-bridged bis(indenyl) dianion", "C(C[c-:1]1[cH:2][cH:3][c:4]2cccc[c:5]21)[c-:6]3[cH:7][cH:8][c:9]4cccc[c:10]43", list(range(1,11)), "tethered_pi_rings", eta_groups=[list(range(1,6)),list(range(6,11))], notes="One covalent ligand; splitting into two indenyl ligands would destroy the ansa tether.")
    ligand("cinnamyl", "cinnamyl anion, eta3 target", "[CH2-:1][CH:2]=[CH:3]c1ccccc1", [1,2,3], "open_chain_pi", eta_groups=[[1,2,3]])
    ligand("cod", "cycloocta-1,5-diene", "[CH:1]1=[CH:2]CC[CH:3]=[CH:4]CC1", [1,2,3,4], "multi_pi_diene", eta_groups=[[1,2],[3,4]])
    ligand("ns6me", "2-(ethylthiomethyl)-6-methylpyridine", "CC[S:1]Cc1cccc(C)[n:2]1", [1,2], "hemilabile_NS")
    ligand("n2s6me", "bis(6-methyl-2-pyridylmethyl) sulfide", "Cc1cccc(C[S:1]Cc2cccc(C)[n:2]2)[n:3]1", [1,2,3], "hemilabile_N2S")
    ligand("n2s_asym", "asymmetric methyl/methoxy N2S thioether", "COc1cccc(C[S:1]Cc2cccc(C)[n:2]2)[n:3]1", [1,2,3], "hemilabile_N2S", notes="The two arms are constitutionally different; coordinated sulfur can be stereogenic.")
    ligand("sns_eth", "bis(2-ethylthioethyl)amine", "CC[S:1]CC[NH:2]CC[S:3]CC", [1,2,3], "sulfur_stereochemistry")
    thienyl = "[n:1]1ccccc1-c2cccc(-c3ccc[s:3]3)[n:2]2"
    ligand("thbpy_N2", "6-(2-thienyl)-bipyridine, N2 binding", thienyl, [1,2], "alternative_binding")
    ligand("thbpy_N2S", "6-(2-thienyl)-bipyridine, N2S binding", thienyl, [1,2,3], "alternative_binding")
    ligand("polar_n2s2", "asymmetric N-benzyl N2S2 macrocycle with pendant sulfonate", "Cc1ccc(C[N:1]2CC[S:2]CC[N:3](Cc3ccc(S(=O)(=O)[O-])cc3)CC[S:4]CC2)cc1", [1,2,3,4], "polar_macrocycle")
    ligand("dppm", "bis(diphenylphosphino)methane", "[P:1](c1ccccc1)(c2ccccc2)C[P:2](c3ccccc3)c4ccccc4", [1,2], "small_bite")
    ligand("dpephos", "DPEphos", "[P:1](c1ccccc1)(c2ccccc2)c3ccccc3Oc4ccccc4[P:2](c5ccccc5)c6ccccc6", [1,2], "wide_bite_flexible")
    xant = "O6c1c(cccc1[P:1](c2ccccc2)c3ccccc3)C(c7cccc([P:2](c4ccccc4)c5ccccc5)c67)(C)C"
    ligand("xantphos", "Xantphos", xant, [1,2], "wide_bite_hinged", notes="The bridging ether is deliberately not a coordinating atom in this fixed-mode task.")
    xl = "O6c1c(cccc1-c8ccc([P:1](c2ccccc2)c3ccccc3)cc8)C(c7cccc(-c9ccc([P:2](c4ccccc4)c5ccccc5)cc9)c67)(C)C"
    ligand("xl_xantphos", "XL-Xantphos", xl, [1,2], "trans_spanning_diphosphine", notes="Two para-phenylene spacers lengthen the P-P span; the bridge oxygen cannot occupy the metal pocket as in ordinary Xantphos.")
    ligand("dm_bpy", "6,6'-dimethyl-2,2'-bipyridine", "Cc1cccc(-c2cccc(C)[n:1]2)[n:2]1", [1,2], "crowded_NN")
    for key, substituent, label in [("dtma", "NC", "DOTA tetra-N-methylamide"),
                                      ("dotasec", "N[C@@H](C)CC", "DOTA tetra-N-sec-butylamide, four specified carbon stereocenters")]:
        smiles = (f"[N:1]1(CC(=[O:5]){substituent})CC[N:2](CC(=[O:6]){substituent})"
                  f"CC[N:3](CC(=[O:7]){substituent})CC[N:4](CC(=[O:8]){substituent})CC1")
        ligand(key, label, smiles, list(range(1,9)), "N4O4_macrocyclic_tetraamide")
    hopo = ("[O-:1]n1c(=[O:2])cccc1C(=O)NCCCN(C(=O)c2cccc(=[O:3])n2[O-:4])"
            "CCCCN(C(=O)c3cccc(=[O:5])n3[O-:6])CCCNC(=O)c4cccc(=[O:7])n4[O-:8]")
    ligand("hopo343", "3,4,3-LI(1,2-HOPO) tetraanion", hopo, list(range(1,9)), "octadentate_O8", notes="Ring carbonyl and N-oxide oxygens bind; four linker amide oxygens do not.")
    ligand("tpen", "TPEN", "[N:1](Cc1cccc[n:2]1)(Cc2cccc[n:3]2)CC[N:4](Cc3cccc[n:5]3)Cc4cccc[n:6]4", list(range(1,7)), "six_donor_cage")
    ligand("nitrate_bi", "bidentate nitrate", "O=[N+]([O-:1])[O-:2]", [1,2], "acute_bite")
    ligand("tpatcn", "TACN tris(picolinate), tpatcn3-", "[N:1]1(Cc2cccc(C(=O)[O-:7])[n:4]2)CC[N:2](Cc3cccc(C(=O)[O-:8])[n:5]3)CC[N:3](Cc4cccc(C(=O)[O-:9])[n:6]4)CC1", list(range(1,10)), "nonadentate_N6O3")
    for key, terminal, name in [("buea", "NC(C)(C)C", "H3buea3-, urea hydrogen-bond cavity"),
                                 ("buea_nme", "N(C)C(C)(C)C", "N-methylated H3buea analogue, no terminal urea NH donors")]:
        smiles = (f"[N:1](CC[N-:2]C(=O){terminal})(CC[N-:3]C(=O){terminal})"
                  f"CC[N-:4]C(=O){terminal}")
        ligand(key, name, smiles, [1,2,3,4], "secondary_sphere_tripod")
    ligand("sulfo_tpa", "tris(4-sulfonato-2-pyridylmethyl)amine", "[N:1](Cc1cc(S(=O)(=O)[O-])cc[n:2]1)(Cc2cc(S(=O)(=O)[O-])cc[n:3]2)Cc3cc(S(=O)(=O)[O-])cc[n:4]3", [1,2,3,4], "polyanionic_N4_tripod")
    ligand("asym_NNO", "methoxy-substituted pyridylmethyl salicylaldiminate", "[n:1]1ccccc1C[N:2]=Cc2cc(OC)ccc2[O-:3]", [1,2,3], "unsymmetrical_N2O")
    ligand("df_tbu_ppy", "difluoro-tert-butyl phenylpyridinate, C,N mode", "CC(C)(C)c1ccc(-c2cc(F)cc(F)[c-:1]2)[n:2]c1", [1,2], "cyclometalated_CN", notes="The aryl C anion is the intended sigma donor; it must not be interpreted as an eta ligand.")
    ligand("depe", "1,2-bis(diethylphosphino)ethane", "CC[P:1](CC)CC[P:2](CC)CC", [1,2], "flexible_alkyl_phosphine")
    aryl = "c1c(C(C)C)cccc1C(C)C"
    ligand("dipp_n3n", "tris(2-(2,6-diisopropylphenylamido)ethyl)amine trianion", f"[N:1](CC[N-:2]{aryl})(CC[N-:3]{aryl})CC[N-:4]{aryl}", [1,2,3,4], "sterically_shielded_triamidoamine", notes="A deliberately smaller aryl-shielded model of a Schrock-type pocket; it is not HIPTN3N.")


def make_case(key, name, metal, oxidation, spin, composition, modes, challenge, *, source_ids=(), origin="designed", ensemble=None, support="direct", support_reason="", solvent="none"):
    entries = [{"ligand_id": item.split(":")[0], "count": int(item.split(":")[1])} for item in composition.split()]
    expanded = [LIGANDS[entry["ligand_id"]] for entry in entries for _ in range(entry["count"])]
    has_eta = any(group["kind"] == "eta" for lig in expanded for group in lig["binding_groups"])
    contact_count = sum(len(lig["donor_indices"]) for lig in expanded)
    binding_sites = sum(len(lig["binding_groups"]) for lig in expanded)
    # Architector reserves three facial template slots for one aromatic pi ring.
    core_cn = sum(3 if group["kind"] == "eta" else 1 for lig in expanded for group in lig["binding_groups"])
    charge = oxidation + sum(lig["formal_charge"] for lig in expanded)
    return {"id": key, "name": name, "metal": metal, "oxidation_state": oxidation,
            "charge": charge, "unpaired_electrons": spin,
            "coordination_number": None if has_eta else contact_count,
            "metal_contact_count": contact_count, "binding_site_count": binding_sites,
            "architector_core_cn": core_cn if support in {"direct", "assembly_only"} else None,
            "ligands": entries, "origin": origin, "source_ids": list(source_ids),
            "source_relationship": ("source chemical input; consult individual evidence notes" if origin == "literature_input"
                                    else "motif or mechanism analogy; not an experimental reference for this exact target"),
            "failure_modes": modes.split(), "challenge": challenge,
            "ensemble_target": ensemble or {"evidence": "design_hypothesis", "targets": [], "populations": None},
            "architector_support": {"status": support, "reason": support_reason,
                                    "runtime_verified": False},
            "environment": {"xtb_solvent": solvent, "temperature_k": 298.15},
            "reference_status": "no_coordinate_reference_archived",
            "stereochemistry": "preserve explicit ligand stereocenters; metal/sulfur configurations not fixed",
            "spin_status": "specified scalar-spin benchmark input; not a spin-state prediction task"}


def define_cases():
    C = make_case
    return [
        C("H01", "[Fe(eta5-Cp*)2]", "Fe", 2, 0, "cpstar:2", "hapticity ring_face torsional_sampling", "Haptic positive control: preserve both five-carbon faces, vary ring rotation without treating ten contacts as CN=10.", source_ids=["ARCH"], origin="literature_motif"),
        C("H02", "[Ru(eta6-p-cymene)(6-Me-NS-Et)Cl]+", "Ru", 2, 0, "pcym:1 ns6me:1 cl:1", "hapticity sulfur_stereochemistry steric_gating", "Piano stool with an asymmetric crowded N,S chelate: arene face, Ru configuration, and thioether configuration are coupled.", source_ids=["ARCH","SNS23"], ensemble={"evidence":"design_hypothesis","targets":["metal/sulfur diastereomers", "arene rotations"],"populations":None}),
        C("H03", "[Rh(eta5-indenyl)(CO)2]", "Rh", 1, 0, "indenyl:1 co:2", "hapticity fused_ring ring_slippage", "Select the five-membered face of a fused aromatic system; diagnose eta5-to-eta3 slippage separately from successful preservation.", source_ids=["ARCH"], origin="literature_motif"),
        C("H04", "rac/meso-[Zr(ethylene-bis-indenyl)Cl2]", "Zr", 4, 0, "ebi:1 cl:2", "tethered_haptics global_closure stereoisomer_coverage", "One covalently tethered dianion spans two eta5 faces; recover rac/meso families while preserving the bridge.", support="requires_adapter", support_reason="Two haptic groups belong to one covalent ligand; one flat sandwich coordList does not represent this tethered target.", ensemble={"evidence":"design_hypothesis","targets":["rac family", "meso family"],"populations":None}),
        C("H05", "[Pd(eta3-cinnamyl)(Xantphos)]+", "Pd", 2, 0, "cinnamyl:1 xantphos:1", "open_chain_haptics allyl_orientation wide_bite", "Open-chain eta3 allyl, substituted-end orientation, and a hinged wide-bite chelate; distinguish eta1 collapse.", source_ids=["ALLYL"], support="requires_adapter", support_reason="Nonaromatic eta3 connectivity and site accounting require an explicit validated haptic adapter; do not send three independent sigma donors."),
        C("H06", "[Rh(eta2,eta2-COD)(P(o-tolyl)3)Cl]", "Rh", 1, 0, "cod:1 potol3:1 cl:1", "multiple_haptic_groups ring_puckering steric_gating", "Two alkene pi groups in one flexible ring must both bind; ring conformation and phosphine shielding are coupled.", support="requires_adapter", support_reason="Two eta2 groups in one diene require separate group placement within the same covalent ligand."),
        C("S01", "[Cu(6,6'-Me2-N2S)Cl2]", "Cu", 2, 1, "n2s6me:1 cl:2", "hemilability steric_gating five_coordinate_shape", "Two crowded pyridine donors and a soft thioether compete to close the same coordination pocket.", source_ids=["RUSULF"]),
        C("S02", "[Cu(asym-Me/OMe-N2S)Cl2]", "Cu", 2, 1, "n2s_asym:1 cl:2", "sulfur_stereochemistry competing_heteroatoms hemilability", "Constitutionally different arms make bound sulfur stereogenic; avoid promoting the methoxy oxygen into the specified donor set.", source_ids=["RUSULF"]),
        C("S03", "[Ru{HN(CH2CH2SEt)2}Cl(H)(PPh3)]", "Ru", 2, 0, "sns_eth:1 cl:1 hydride:1 pph3:1", "sulfur_stereochemistry exchangeable_isomers hydride_policy", "Published amine-based SNS complex with three crystallographically characterized sulfur diastereomers and measured exchange.", source_ids=["SNS23"], origin="literature_input", ensemble={"evidence":"experimental_exchange","source_id":"SNS23","targets":["2a", "2b", "2c"],"activation_gibbs_kj_mol_range":[65,70],"temperature_k":None,"populations":None,"notes":"Exchange barriers are kinetic quantities, not energy gaps or equal-population claims."}, support="assembly_only", support_reason="Architector 0.0.10 forces relax=False for explicit [H-]; use --no-final-relax and a separate common optimization stage."),
        C("S04", "[Ru(thienyl-bpy-kappa2N,N)(thienyl-bpy-kappa3N,N,S)Cl]+", "Ru", 2, 0, "thbpy_N2:1 thbpy_N2S:1 cl:1", "binding_mode_discrimination competing_heteroatoms mixed_denticity", "The identical chemical ligand occurs twice with different donor sets; one thiophene sulfur binds and the other remains unbound.", source_ids=["RUSULF"], origin="literature_input"),
        C("S05", "[Cu(sulfonated-asym-N2S2-macrocycle)Cl]", "Cu", 2, 1, "polar_n2s2:1 cl:1", "macrocycle_closure sulfur_stereochemistry charged_sidechain", "An asymmetric N2S2 ring carries a sulfonate-bearing benzyl arm; prevent spectator oxygens from replacing the intended donors.", solvent="water"),
        C("B01", "[Pd(dppm)Cl2]", "Pd", 2, 0, "dppm:1 cl:2", "acute_bite small_chelate_ring", "Four-membered Pd-P-C-P chelate is an intentionally narrow-bite contrast to the spanning phosphines.", source_ids=["ALLYL"], origin="literature_motif"),
        C("B02", "[Pd(DPEphos)Cl2]", "Pd", 2, 0, "dpephos:1 cl:2", "wide_bite backbone_folding competing_heteroatoms", "Large P-P span with an ether hinge and competing ether oxygen; fixed P,P mode must survive folding.", source_ids=["ALLYL"], origin="literature_motif"),
        C("B03", "[Pd(Xantphos)Cl2]", "Pd", 2, 0, "xantphos:1 cl:2", "wide_bite cis_trans_coverage backbone_folding", "Probe cis versus trans spanning placements and xanthene folding; compare with the spacer-extended B05 ligand.", source_ids=["XLXANT"], origin="literature_input", ensemble={"evidence":"computed_relative_stability","source_id":"XLXANT","targets":["cis-P,P arrangement", "trans-P,P arrangement"],"population_reference":False,"notes":"The paper calculates a cis preference for Xantphos; this is not evidence of equal or thermally accessible populations."}),
        C("B04", "[Cu(Xantphos)(6,6'-Me2-bpy)]+", "Cu", 1, 0, "xantphos:1 dm_bpy:1", "steric_gating backbone_folding interligand_clash", "Crowded N,N chelate and wide P,P chelate compete for a tetrahedral Cu(I) sphere; measure unintended flattening and phenyl packing.", source_ids=["CUPP"], origin="literature_motif"),
        C("B05", "[Pd(XL-Xantphos)Cl2]", "Pd", 2, 0, "xl_xantphos:1 cl:2", "trans_spanning global_closure cis_trans_coverage", "Phenylene-extended Xantphos spans opposite metal sites; use B03/B05 as a matched test of local versus global bite constraints.", source_ids=["XLXANT"], origin="literature_input", ensemble={"evidence":"computed_relative_stability","source_id":"XLXANT","targets":["trans-P,P arrangement", "cis-P,P arrangement"],"population_reference":False,"notes":"The reported computed trans preference is not a measured solution population or an activation barrier."}),
        C("D01", "[Eu(DTMA)(H2O)]3+", "Eu", 3, 6, "dtma:1 water:1", "octadentate_closure SAP_TSAP arm_ring_coupling", "N4O4 tetraamide macrocycle has coupled ring inversion and pendant-arm rotation; retain the inner-sphere water.", source_ids=["DOTA09"], origin="literature_input", solvent="water", ensemble={"evidence":"solution_isomers_reported","source_id":"DOTA09","targets":["SAP", "TSAP"],"reported_SAP_to_TSAP_ratio":3.2,"ratio_scope":"Background DTMA value cited in this paper; conditions must be recovered before quantitative use.","temperature_k":None,"populations":None}),
        C("D02", "[Eu(DOTA-tetra-sec-butylamide)(H2O)]3+", "Eu", 3, 6, "dotasec:1 water:1", "octadentate_closure fixed_ligand_chirality SAP_TSAP steric_gating", "Four specified sec-butyl carbon stereocenters bias the coupled macrocycle/arm arrangements; preserve every ligand stereocenter.", source_ids=["DOTA09"], origin="literature_input", solvent="water", ensemble={"evidence":"solution_isomer_study","source_id":"DOTA09","targets":["SAP-like family", "TSAP-like family"],"populations":None,"notes":"No numeric population or accessibility threshold is assigned to this authored stereoisomer without matching the experimental stereochemical definition."}),
        C("D03", "[Zr(3,4,3-LI(1,2-HOPO))]", "Zr", 4, 0, "hopo343:1", "octadentate_closure global_wrapping competing_heteroatoms hydrogen_bonding", "A long four-arm O8 chelator wraps the metal; select ring O donors over linker amide O atoms and sample distinct backbone folds.", source_ids=["HOPO"], origin="literature_input", solvent="water"),
        C("D04", "[La(TPEN)(kappa2-NO3)2]+", "La", 3, 0, "tpen:1 nitrate_bi:2", "high_contact_count acute_bite coupled_chelate_closure", "Ten atomic contacts combine a crowded six-N cage with two acute-bite nitrates; a designed overpacking boundary case."),
        C("D05", "[Lu(tpatcn)]", "Lu", 3, 0, "tpatcn:1", "nonadentate_closure rigid_arm_cage competing_heteroatoms", "Nine N6O3 donors on a TACN-based tripod must all close; pyridyl orientation and carboxylate choices are coupled.", source_ids=["TPATCN"], origin="literature_motif", solvent="water"),
        C("P01", "[FeIII(H3buea)(OH)]-", "Fe", 3, 5, "buea:1 hydroxide:1", "secondary_sphere hydrogen_bonding competing_heteroatoms", "Three urea NH groups should organize around the hydroxo pocket; carbonyl O atoms remain outside the primary N4 donor sphere.", source_ids=["BUEA"], origin="literature_input", solvent="water"),
        C("P02", "[FeIII(N-methyl-H3buea-analogue)(OH)]-", "Fe", 3, 5, "buea_nme:1 hydroxide:1", "secondary_sphere conformer_ranking matched_chemical_perturbation", "A custom NH-blocked partner to P01 changes the directional hydrogen-bond network while retaining the same N4 primary donor topology.", source_ids=["BUEA"], solvent="water"),
        C("P03", "[Cu(trisulfonato-TPA)(H2O)]-", "Cu", 2, 1, "sulfo_tpa:1 water:1", "charged_sidechain solvent_sensitivity competing_heteroatoms", "Three remote sulfonates make many polar oxygen atoms available without changing the intended N4 tripod; test collapse and donor swapping.", solvent="water"),
        C("I01", "[Fe(asym-N2O-Schiff-base)2]+", "Fe", 3, 5, "asym_NNO:2", "asymmetric_tridentates isomer_coverage orientation_coupling", "Two directional tridentates create head-tail, fac/mer-like, and chiral arrangement alternatives; peripheral methoxy groups add conformational choices.", ensemble={"evidence":"design_hypothesis","targets":["distinct N/O arrangement families", "metal helicities"],"populations":None}),
        C("I02", "[Ir(difluoro-tert-butyl-ppy)3]", "Ir", 3, 0, "df_tbu_ppy:3", "cyclometalation fac_mer_coverage steric_gating", "Three unsymmetrical C,N chelates require carbon sigma donation and fac/mer family sampling with fluorinated, bulky outer groups.", ensemble={"evidence":"design_hypothesis","targets":["fac donor arrangement", "mer donor arrangement", "metal helicities"],"populations":None}),
        C("I03", "[Pd(6-Me-NS-Et)2]2+", "Pd", 2, 0, "ns6me:2", "cis_trans_coverage sulfur_stereochemistry coupled_isomerism", "Square-planar cis/trans N/S arrangements combine with two independent sulfur configurations and constrained chelate folds.", source_ids=["SNS23"], ensemble={"evidence":"design_hypothesis","targets":["cis-S,S", "trans-S,S", "sulfur diastereomers"],"populations":None}),
        C("N01", "[Mo(N2)2(depe)2]", "Mo", 0, 0, "n2:2 depe:2", "dinitrogen_binding cis_trans_coverage phosphine_torsions", "Two end-on N2 ligands share a flexible four-phosphorus environment; preserve N-N connectivity and sample N2 placement alternatives.", source_ids=["MO69"], origin="literature_motif", ensemble={"evidence":"candidate_families_not_population_reference","targets":["cis-N2", "trans-N2"],"populations":None}),
        C("N02", "[MoIII(Dipp-N3N)(N2)]", "Mo", 3, 1, "dipp_n3n:1 n2:1", "dinitrogen_binding steric_cavity low_spin_input global_wrapping", "Custom reduced-size aryl-shielded triamidoamine pocket challenges approach to N2 and collective aryl rotation; low-spin input is an explicit assumption for this derivative.", source_ids=["MOEPR"]),
    ]


def main():
    define_ligands()
    cases = define_cases()
    destination = ROOT / "data/challenge"
    destination.mkdir(exist_ok=True)
    if (destination / "references.json").exists():
        prepared = read_json(destination / "references.json")["references"]
        for case in cases:
            kinds = {ref["kind"] for ref in prepared if ref["case_id"] == case["id"]}
            if "experimental_crystal_heavy_atoms" in kinds:
                case["reference_status"] = "experimental_heavy_atom_reference_prepared"
            elif "published_computed_geometry" in kinds:
                case["reference_status"] = "published_computed_geometry_set_prepared"
    write_json(destination / "ligands.json", {"schema_version": 2,
        "index_convention": "Zero-based atoms in the exact SMILES before implicit hydrogens are expanded. An explicit [H-] is a real indexed atom.",
        "ligands": list(LIGANDS.values())})
    write_json(destination / "cases.json", {"schema_version": 2, "dataset_id": "coordination-failure-modes-v2",
        "selection": "Mechanism-led challenge panel authored before running either builder; no tmQM inputs.", "cases": cases})
    print(f"Authored {len(cases)} challenge cases and {len(LIGANDS)} ligands")
    print("Support:", dict(Counter(case["architector_support"]["status"] for case in cases)))
    for ligand_data in LIGANDS.values():
        print(ligand_data["id"], ligand_data["formula"], "donors", ligand_data["donor_elements"],
              "radicals", ligand_data["radical_electrons"], "stereo", ligand_data["fixed_stereocenters"])


if __name__ == "__main__":
    main()
