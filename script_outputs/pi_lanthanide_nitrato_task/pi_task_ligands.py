"""
Ligand definitions + complex-building helpers for the PI's lanthanide/Fe
aqua & nitrato request (DOI 10.25584/3019314 comparison).
"""
from jacobik.engine import LigandInput, ComplexBuilder

WATER_SMILES = "O"
NITRATE_SMILES = "[O-][N+](=O)[O-]"  # idx0=O-, idx1=N+, idx2==O, idx3=O-

def water() -> LigandInput:
    return LigandInput(WATER_SMILES, donors=[[0]])

def nitrate_bidentate() -> LigandInput:
    # chelating (kappa2-O,O') binding, matching the DOE reference structures
    # (confirmed via M-O distance analysis: 2 O's at normal bond length,
    # 3rd O ~4+ A away / non-coordinating in every reference file checked)
    return LigandInput(NITRATE_SMILES, donors=[[0], [3]])

def nitrate_monodentate() -> LigandInput:
    return LigandInput(NITRATE_SMILES, donors=[[0]])


def build_aqua(metal: str, y: int, charge: int, spin: int) -> ComplexBuilder:
    geom = {6: "octahedral", 7: "pent_bipyr", 8: "square_anti", 9: "tricap_pris"}[y]
    return ComplexBuilder(
        metal=metal, geometry=geom, charge=charge, spin=spin,
        ligands=[water() for _ in range(y)],
    )


def build_nitrato_bidentate(metal: str, y_water: int, charge: int, spin: int) -> ComplexBuilder:
    """y_water coordinated waters + 1 bidentate nitrate -> CN = y_water + 2."""
    cn = y_water + 2
    geom = {6: "octahedral", 7: "pent_bipyr", 8: "square_anti", 9: "tricap_pris"}[cn]
    return ComplexBuilder(
        metal=metal, geometry=geom, charge=charge, spin=spin,
        ligands=[water() for _ in range(y_water)] + [nitrate_bidentate()],
    )
