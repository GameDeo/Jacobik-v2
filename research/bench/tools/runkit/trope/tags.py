"""Trope tag rules (versioned: RULES_VERSION). Input: a features dict from features.extract. Output: a sorted set of tag strings.

Rules
- chelate_ringN        a non-haptic ligand with >= 2 donors forms a metallacycle of N atoms (metal + donors + shortest ligand path between
                       two donors); N = 4, 5, 6, 7 and "8plus" for 8 or more. One tag per ring size present.
- haptic_etaN          a ligand the data model marks haptic, with order N (2..6) and at least one carbon donor (excludes N,N "eta-2"
                       amidinate-like donor pairs).
- macrocycle           a ligand with a macrocycle in its decomposition.
- bulky_phosphine      a P donor with three P-C bonds whose carbons each carry >= 2 further heavy neighbours (aryl, cyclohexyl, tBu,
                       iPr-type; PMe3 and PEt3 do not count).
- anionic_{halide,O,N,C,S}   donor subclass in the anionic sets below (Hal_*; O_phenoxide, O_alkoxide, O_carboxylate, O_diketonate,
                       O_dioxolene, O_hydroxide, O_ketoiminate; N_amido, N_amide, N_amidate, N_amidinate, N_diketiminate, N_ketoiminate,
                       N_imido; C_alkyl, C_vinyl, C_acetylide, C_acyl; S_thiolate).
- carbene_nhc          a C donor with >= 2 N neighbours and no H (N-heterocyclic carbene type).
- carbene_other        donor subclass C_carbene or C_alkylidene that is not carbene_nhc.
- cnN_GEOM             coordination number 4, 5 or 6 with the data model's coordination geometry label.
- lanthanide           metal La..Lu.
"""
from __future__ import annotations

RULES_VERSION = "1"
ANION = {
    "halide": lambda s: s.startswith("Hal_"),
    "O": lambda s: s in {"O_phenoxide", "O_alkoxide", "O_carboxylate", "O_diketonate", "O_dioxolene", "O_hydroxide", "O_ketoiminate"},
    "N": lambda s: s in {"N_amido", "N_amide", "N_amidate", "N_amidinate", "N_diketiminate", "N_ketoiminate", "N_imido"},
    "C": lambda s: s in {"C_alkyl", "C_vinyl", "C_acetylide", "C_acyl"},
    "S": lambda s: s == "S_thiolate",
}
LANTHANIDES = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"}


def tags_of(f: dict) -> set:
    t = set()
    for L in f["ligands"]:
        for r in L["rings"]:
            t.add(f"chelate_ring{r}" if 4 <= r <= 7 else ("chelate_ring8plus" if r >= 8 else f"chelate_ring{r}"))
        if L["haptic_order"] >= 2 and "C" in L["haptic_donors"] and L["haptic_order"] <= 6:
            t.add(f"haptic_eta{L['haptic_order']}")
        if L["macrocycle"]:
            t.add("macrocycle")
        if any(p["n_c"] == 3 and min(p["branching"]) >= 2 for p in L["phosphine"]):
            t.add("bulky_phosphine")
        nhc = [c for c in L["carbene"] if c["n_N"] >= 2 and c["n_H"] == 0]
        if nhc:
            t.add("carbene_nhc")
        elif {"C_carbene", "C_alkylidene"} & set(L["subclasses"]):
            t.add("carbene_other")
    for s in f["donor_subclasses"]:
        for k, fn in ANION.items():
            if fn(s):
                t.add(f"anionic_{k}")
    if f["cn"] in (4, 5, 6) and f.get("coord_geom"):
        t.add(f"cn{f['cn']}_{f['coord_geom']}")
    if f["metal"] in LANTHANIDES:
        t.add("lanthanide")
    t = {x for x in t if not x.startswith("chelate_ring") or x in {"chelate_ring4", "chelate_ring5", "chelate_ring6", "chelate_ring7", "chelate_ring8plus"}}
    return t
