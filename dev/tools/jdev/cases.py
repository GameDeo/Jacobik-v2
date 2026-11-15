"""Case registry: every benchmark complex jdev can solve, with its reference structure.

Names:
  aqua_<X>            DOE aqua system from dev/tests/regression/fixtures/doe_aqua/<X>.xyz  (e.g. aqua_Er3_8H2O)
  <CSDCODE>           tmQM complex in the regression fixture split (dev/tests/regression/fixtures/tmqm)
  tmqm:<dir>:<split>:<CODE>   any tmQM-format split on disk (dir containing xyz/, bo/, tmQM_y.csv)
  legacy_co_edta | legacy_la_edta_cn9 | legacy_co_en3   hand-built chelates (no reference structure)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .env import FIX, use_src

EDTA = "[O-]C(=O)CN(CC(=O)[O-])CCN(CC(=O)[O-])CC(=O)[O-]"
SPIN = {"La": 1, "Ce": 2, "Pr": 3, "Nd": 4, "Pm": 5, "Sm": 6, "Eu": 7, "Gd": 8, "Tb": 7, "Dy": 6, "Ho": 5,
        "Er": 4, "Tm": 3, "Yb": 2, "Lu": 1, "Fe2": 5, "Fe3": 6}
GEOM = {8: "square_anti", 9: "tricap_pris", 6: "octahedral"}


@dataclass
class Case:
    name: str
    make_builder: Callable[[], object]
    charge: int
    mult: int
    ref_symbols: Optional[list] = None      # reference structure (None for hand-built cases)
    ref_X: Optional[np.ndarray] = None
    ref_metal_row: Optional[int] = None
    tmqm_mol: object = None                  # parsed TMComplex (tmQM cases only)
    ligand_groups: list = field(default_factory=list)
    ligand_inputs: list = field(default_factory=list)
    n_donors_crystal: Optional[int] = None


def _read_xyz(f: Path):
    L = [x.split() for x in f.read_text().splitlines()[2:] if x.strip()]
    return [x[0] for x in L], np.array([[float(v) for v in x[1:4]] for x in L])


def _tmqm(split_dir: Path, split: str, code: str) -> Case:
    use_src()
    from jacobik import ComplexBuilder
    from fit.parser import TMQMParser
    from jacobik.builder.ligands import _build_ligand_groups, ligand_inputs_from_mol

    m = next((x for x in TMQMParser(split_dir).stream_split(split) if x.csd_code == code), None)
    if m is None:
        raise KeyError(f"{code} not found in {split_dir} split {split}")
    groups = _build_ligand_groups(m, m.smiles)
    lig_in = ligand_inputs_from_mol(m, groups)
    if lig_in is None:
        raise LookupError(f"{code}: a ligand has no SMILES or no mapped donors (tmQM SMILES missing or unmappable)")
    txt = (Path(split_dir) / "xyz" / f"tmQM_{split}.xyz").read_text()
    mnd = re.search(rf"CSD_code = {code} .*?MND = (\d+)", txt)
    rows = {a.idx: i for i, a in enumerate(m.atoms)}
    return Case(
        name=code,
        make_builder=lambda: ComplexBuilder(metal=m.metal_symbol, geometry=m.coord_geom, charge=m.charge,
                                            spin=m.spin_state, ligands=lig_in),
        charge=m.charge, mult=m.spin_state,
        ref_symbols=[a.symbol for a in m.atoms], ref_X=np.array([a.position for a in m.atoms]),
        ref_metal_row=rows[m.metal_idx], tmqm_mol=m, ligand_groups=groups, ligand_inputs=lig_in,
        n_donors_crystal=int(mnd.group(1)) if mnd else None,
    )


def donor_mismatches(case: Case) -> list:
    """Input donors whose SMILES atom differs from the crystal atom it is mapped to (element or H count), and donors
    that are not the ligand atom nearest the metal among their bonded neighbours. Empty = the solver is asked to bind
    the same atoms the crystal binds."""
    from rdkit import Chem

    m = case.tmqm_mol
    A = {a.idx: a for a in m.atoms}
    M = np.asarray(A[m.metal_idx].position, float)
    nh, nbr = {}, {}
    for b in m.bonds:
        if m.metal_idx in (b.i, b.j) or b.order < 0.5:
            continue
        for u, w in ((b.i, b.j), (b.j, b.i)):
            nbr.setdefault(u, []).append(w)
            if A[w].symbol == "H":
                nh[u] = nh.get(u, 0) + 1
    d = lambda i: float(np.linalg.norm(np.asarray(A[i].position, float) - M))  # noqa: E731
    out = []
    for li, lg in zip(case.ligand_inputs, case.ligand_groups):
        mol = Chem.MolFromSmiles(li.smiles)
        for g in li.donors:
            for r in g:
                x = lg["rdkit_to_xyz"].get(r)
                if x is None:
                    out.append((li.smiles, r, "unmapped"))
                    continue
                a = mol.GetAtomWithIdx(r)
                if a.GetSymbol() != A[x].symbol or a.GetTotalNumHs() != nh.get(x, 0):
                    out.append((li.smiles, r, f"SMILES {a.GetSymbol()}H{a.GetTotalNumHs()} vs crystal {A[x].symbol}H{nh.get(x, 0)}"))
                if len(g) == 1 and any(A[y].symbol != "H" and d(y) < d(x) - 0.05 for y in nbr.get(x, [])):
                    out.append((li.smiles, r, "a bonded neighbour is nearer the metal"))
    return out


def get_case(name: str) -> Case:
    use_src()
    from jacobik import ComplexBuilder, LigandInput

    if name.startswith("aqua_"):
        x = name[5:]
        metal, q, cn = re.match(r"([A-Z][a-z]?)(\d)_(\d)H2O", x).groups()
        q, cn = int(q), int(cn)
        mult = SPIN[metal if metal != "Fe" else f"Fe{q}"]
        sy, X = _read_xyz(FIX / "doe_aqua" / f"{x}.xyz")
        return Case(name, lambda: ComplexBuilder(metal, GEOM[cn], charge=q, spin=mult,
                                                 ligands=[LigandInput("O", donors=[[0]]) for _ in range(cn)]),
                    q, mult, sy, X, sy.index(metal), n_donors_crystal=cn)
    if name == "legacy_co_edta":
        return Case(name, lambda: ComplexBuilder("Co", "octahedral", charge=-1, spin=1,
                    ligands=[LigandInput(EDTA, donors=[[0], [4], [7], [11], [15], [19]])]), -1, 1)
    if name == "legacy_la_edta_cn9":
        return Case(name, lambda: ComplexBuilder("La", "tricap_pris", charge=-1, spin=1,
                    ligands=[LigandInput(EDTA, donors=[[4], [11], [0], [8], [15], [19]])]
                    + [LigandInput("O", donors=[[0]]) for _ in range(3)]), -1, 1)
    if name == "legacy_co_en3":
        return Case(name, lambda: ComplexBuilder("Co", "octahedral", charge=3, spin=1,
                    ligands=[LigandInput("NCCN", donors=[[0], [3]]) for _ in range(3)]), 3, 1)
    if name.startswith("tmqm:"):
        # Split from the right so that an absolute Windows path in `d` (e.g. C:\...) is
        # preserved intact — a naive split(":", 3) eats the drive-letter colon.
        rest = name[len("tmqm:"):]          # strip leading "tmqm:"
        d, split, code = rest.rsplit(":", 2)
        return _tmqm(Path(d), split, code)
    return _tmqm(FIX / "tmqm", "P", name)


def panel_case_names() -> list:
    txt = (FIX / "tmqm" / "xyz" / "tmQM_P.xyz").read_text()
    codes = re.findall(r"CSD_code = (\w+)", txt)
    aqua = sorted(p.stem for p in (FIX / "doe_aqua").glob("*.xyz"))
    return [f"aqua_{a}" for a in aqua] + codes + ["legacy_co_edta", "legacy_la_edta_cn9", "legacy_co_en3"]
