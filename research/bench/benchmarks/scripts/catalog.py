"""Offline catalog loading and Architector input construction. No JacobIK imports."""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def catalog_directory(suite="baseline"):
    if suite not in {"baseline", "challenge"}:
        raise ValueError(f"Unknown suite: {suite}")
    return ROOT / "data" / "challenge" if suite == "challenge" else ROOT / "data"


def load_catalog(suite="baseline"):
    directory = catalog_directory(suite)
    cases = read_json(directory / "cases.json")["cases"]
    ligands = read_json(directory / "ligands.json")["ligands"]
    return cases, {ligand["id"]: ligand for ligand in ligands}


class UnsupportedInput(ValueError):
    def __init__(self, status, reason):
        self.status = status
        super().__init__(reason)


def make_input(case, ligands, seed=17, n_conformers=10, final_relax=True, temp_prefix=None):
    """Only CN is disclosed; expected geometry and crystal coordinates are withheld."""
    support = case.get("architector_support", {"status": "direct"})
    if support["status"] == "requires_adapter":
        raise UnsupportedInput("unsupported_representation", support["reason"])
    if support["status"] == "assembly_only" and final_relax:
        raise UnsupportedInput("requires_no_final_relax", support["reason"])
    expanded = []
    for entry in case["ligands"]:
        ligand = ligands[entry["ligand_id"]]
        for _ in range(entry["count"]):
            item = {"smiles": ligand["smiles"], "coordList": list(ligand["donor_indices"])}
            pi_groups = [group for group in ligand.get("binding_groups", []) if group["kind"] == "eta"]
            if pi_groups:
                # The challenge validator permits only a single complete aromatic
                # ring for this direct adapter. No splitting of covalent ligands.
                if len(pi_groups) != 1 or set(pi_groups[0]["atom_indices"]) != set(item["coordList"]):
                    raise UnsupportedInput("unsupported_representation", "Multiple or mixed pi groups require an adapter")
                item["ligType"] = "sandwich"
            elif len(item["coordList"]) == 1:
                item["ligType"] = "mono"
            # Let Architector assign multidentate ligand types using its own algorithm.
            expanded.append(item)
    parameters = {
        "metal_ox": case["oxidation_state"],
        "metal_spin": case["unpaired_electrons"],
        "full_charge": case["charge"],
        "full_spin": case["unpaired_electrons"],
        "test_alternate_metal_spin": False,
        "seed": seed,
        "n_conformers": n_conformers,
        "n_symmetries": max(10, n_conformers),
        "relax": final_relax,
        "save_init_geos": True,
        "return_timings": True,
        "assemble_method": "GFN2-xTB",
        "full_method": "GFN2-xTB",
        "crest_sampling": False,
        "xtb_solvent": case.get("environment", {}).get("xtb_solvent", "none"),
        "force_generation": False,
    }
    if temp_prefix is not None:
        parameters["temp_prefix"] = str(temp_prefix)
    return copy.deepcopy({
        "core": {"metal": case["metal"], "coreCN": case.get("architector_core_cn", case["coordination_number"])},
        "ligands": expanded,
        "parameters": parameters,
    })


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
