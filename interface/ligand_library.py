"""Curated debug complexes and polyatomic ligand library for Jacobik.

This single module replaces the old ``interface/examples/`` directory with:

1. **Solvable debug complex constants** that unpack directly into ``jacobik.generate``::

       import jacobik
       from interface.ligand_library import DEBUG_COMPLEX_CBTYCO, DEBUG_COMPLEX_OJIMIF

       res = jacobik.generate(*DEBUG_COMPLEX_CBTYCO)

2. **Common polyatomic coordination ligands (excluding trivial single-atom donors)** with
   verified SMILES, formal charges, multiple binding modes (hapticities / kappa modes /
   ambidentate donor atoms), and intuitive aliases usable from both Python and the CLI::

       # Python (direct helpers or string specs inside jacobik.generate)
       from interface.ligand_library import ligand, ligands
       res = jacobik.generate("Fe", "octahedral", 2, 1, ligands("bpy*3"))
       res = jacobik.generate("Pt", "square_plan", 0, 1, ["dmso@kappa1_O", "en", "PPh3"])

       # CLI
       # python interface/jacobik_cli.py --debug-complex CBTYCO
       # python interface/jacobik_cli.py --metal Fe --geometry octahedral --charge 2 --spin 1 --ligand "bpy*3"
       # python interface/jacobik_cli.py ligands --query thiocyanate
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from jacobik.builder.result import LigandInput


class DebugComplex(NamedTuple):
    """Predefined complex tuple matching ``jacobik.generate(metal, geometry, charge, spin, ligands)``."""

    metal: str
    geometry: str
    charge: int
    spin: int
    ligands: Tuple[LigandInput, ...]

    def as_dict(self) -> Dict[str, Any]:
        """Return keyword arguments suitable for ``jacobik.generate(**dc.as_dict())``."""
        return {
            "metal": self.metal,
            "geometry": self.geometry,
            "charge": self.charge,
            "spin": self.spin,
            "ligands": list(self.ligands),
        }

    def builder(self, **kwargs):
        """Construct a ``jacobik.ComplexBuilder`` for this complex."""
        import jacobik

        return jacobik.ComplexBuilder(*self, **kwargs)

    def generate(self, **kwargs):
        """Run ``jacobik.generate(*self, **kwargs)``."""
        import jacobik

        return jacobik.generate(*self, **kwargs)

    def solve(self, **kwargs):
        """Alias for :meth:`generate`."""
        return self.generate(**kwargs)


# Reference crystal structures formerly stored under interface/examples/
DEBUG_COMPLEX_OJIMIF = DebugComplex(
    metal="Pt",
    geometry="square_plan",
    charge=0,
    spin=1,
    ligands=(
        LigandInput("[Cl-]", donors=[[0]]),
        LigandInput("CCSC#N", donors=[[4]]),
        LigandInput("[Cl-]", donors=[[0]]),
        LigandInput("CCSC#N", donors=[[4]]),
    ),
)

DEBUG_COMPLEX_CBTYCO = DebugComplex(
    metal="Co",
    geometry="see_saw",
    charge=0,
    spin=1,
    ligands=(
        LigandInput("[Cl-]", donors=[[0]]),
        LigandInput("[Cl-]", donors=[[0]]),
        LigandInput("COC(N=c1[nH]c2ccccc2s1)C1=CC=C=C(C)[N-]1", donors=[[3], [19]]),
    ),
)

DEBUG_COMPLEXES: Dict[str, DebugComplex] = {
    "OJIMIF": DEBUG_COMPLEX_OJIMIF,
    "CBTYCO": DEBUG_COMPLEX_CBTYCO,
}


def debug_complex(name: str) -> DebugComplex:
    """Look up a predefined debug complex by name (case-insensitive)."""
    key = str(name).strip().upper()
    if key in DEBUG_COMPLEXES:
        return DEBUG_COMPLEXES[key]
    raise KeyError(f"Unknown debug complex {name!r}. Available: {sorted(DEBUG_COMPLEXES)}")


@dataclass(frozen=True)
class BindingMode:
    """Single coordination mode for a library ligand."""

    name: str
    donors: Tuple[Tuple[int, ...], ...]
    summary: str
    aliases: Tuple[str, ...] = ()

    @property
    def denticity(self) -> int:
        return len(self.donors)


@dataclass(frozen=True)
class LigandDef:
    """Curated polyatomic ligand definition with one or more binding modes."""

    name: str
    smiles: str
    charge: int
    category: str
    default_mode: str
    modes: Mapping[str, BindingMode]
    aliases: Tuple[str, ...] = ()
    notes: str = ""

    def resolve_mode(self, mode: Optional[str] = None) -> BindingMode:
        """Resolve a mode name or mode alias (defaults to ``self.default_mode``)."""
        if mode is None or not str(mode).strip():
            return self.modes[self.default_mode]
        raw = str(mode).strip()
        if raw in self.modes:
            return self.modes[raw]
        norm = _normalize_mode_token(raw)
        candidates: Dict[str, BindingMode] = {}
        for key, bm in self.modes.items():
            candidates[key.lower()] = bm
            candidates[_normalize_mode_token(key)] = bm
            for alias in bm.aliases:
                candidates[alias.lower()] = bm
                candidates[_normalize_mode_token(alias)] = bm
        if raw.lower() in candidates:
            return candidates[raw.lower()]
        if norm in candidates:
            return candidates[norm]
        raise ValueError(
            f"Ligand {self.name!r} has no binding mode {mode!r}. "
            f"Available modes: {list(self.modes.keys())}"
        )

    def to_input(
        self,
        mode: Optional[str] = None,
        *,
        sites: Optional[Sequence[int]] = None,
    ) -> LigandInput:
        """Build a ``jacobik.LigandInput`` in the requested binding mode."""
        bm = self.resolve_mode(mode)
        return LigandInput(
            smiles=self.smiles,
            donors=[list(group) for group in bm.donors],
            sites=list(sites) if sites is not None else None,
        )


def _normalize_mode_token(text: str) -> str:
    s = text.strip().replace("\u03ba", "kappa").replace("\u03b7", "eta").lower()
    return re.sub(r"[-_\s]+", "", s)


def _bm(
    name: str,
    donors: Iterable[Sequence[int]],
    summary: str,
    *aliases: str,
) -> Tuple[str, BindingMode]:
    return name, BindingMode(
        name=name,
        donors=tuple(tuple(int(i) for i in grp) for grp in donors),
        summary=summary,
        aliases=tuple(aliases),
    )


def _lig(
    name: str,
    smiles: str,
    charge: int,
    category: str,
    modes: Iterable[Tuple[str, BindingMode]],
    *,
    default_mode: Optional[str] = None,
    aliases: Tuple[str, ...] = (),
    notes: str = "",
) -> LigandDef:
    mode_map = dict(modes)
    first_mode = next(iter(mode_map))
    chosen_default = default_mode or first_mode
    if chosen_default not in mode_map:
        raise ValueError(f"{name}: default_mode {chosen_default!r} not in {list(mode_map)}")
    return LigandDef(
        name=name,
        smiles=smiles,
        charge=charge,
        category=category,
        default_mode=chosen_default,
        modes=mode_map,
        aliases=aliases,
        notes=notes,
    )


_RAW_LIGAND_LIST: Tuple[LigandDef, ...] = (
    # ------------------------------------------------------------------
    # Pi-acids and small ambidentate polyatomics (9)
    # ------------------------------------------------------------------
    _lig(
        "CO", "[C-]#[O+]", 0, "pi_acid",
        [
            _bm("kappa1_C", [[0]], "C-bound carbonyl", "C", "kappa1", "carbonyl"),
            _bm("kappa1_O", [[1]], "O-bound isocarbonyl", "O", "isocarbonyl"),
        ],
        aliases=("carbonyl",),
        notes="Carbon monoxide.",
    ),
    _lig(
        "CN", "[C-]#N", -1, "pi_acid",
        [
            _bm("kappa1_C", [[0]], "C-bound cyanide", "C", "kappa1", "cyano"),
            _bm("kappa1_N", [[1]], "N-bound isocyanide", "N", "isocyano"),
        ],
        aliases=("cyanide", "cyano", "CN-"),
        notes="Cyanide anion.",
    ),
    _lig(
        "NO", "[N]=O", 0, "pi_acid",
        [
            _bm("kappa1_N", [[0]], "N-bound nitrosyl", "N", "kappa1"),
            _bm("kappa1_O", [[1]], "O-bound isonitrosyl", "O"),
        ],
        aliases=("nitrosyl",),
        notes="Nitric oxide / nitrosyl.",
    ),
    _lig(
        "SCN", "[S-]C#N", -1, "ambidentate",
        [
            _bm("kappa1_S", [[0]], "S-bound thiocyanate", "S"),
            _bm("kappa1_N", [[2]], "N-bound isothiocyanate", "N", "NCS"),
        ],
        aliases=("thiocyanate", "SCN-"),
        notes="Ambidentate thiocyanate anion.",
    ),
    _lig(
        "OCN", "[O-]C#N", -1, "ambidentate",
        [
            _bm("kappa1_N", [[2]], "N-bound isocyanate", "N", "NCO"),
            _bm("kappa1_O", [[0]], "O-bound cyanate", "O"),
        ],
        aliases=("cyanate", "OCN-"),
        notes="Ambidentate cyanate anion.",
    ),
    _lig(
        "N3", "[N-]=[N+]=[N-]", -1, "ambidentate",
        [_bm("kappa1_N", [[0]], "terminal N-bound azide", "N", "kappa1")],
        aliases=("azide", "azido", "N3-"),
        notes="Azide anion.",
    ),
    _lig(
        "NO2", "O=N[O-]", -1, "ambidentate",
        [
            _bm("kappa1_N", [[1]], "N-bound nitro", "N", "nitro"),
            _bm("kappa1_O", [[2]], "O-bound nitrito", "O", "nitrito"),
            _bm("kappa2_OO", [[0], [2]], "O,O'-chelating nitrito", "OO", "kappa2", "bidentate"),
        ],
        aliases=("nitrite", "NO2-"),
        notes="Ambidentate nitrite anion (nitro / nitrito / bidentate).",
    ),
    _lig(
        "NO3", "O=[N+]([O-])[O-]", -1, "oxyanion",
        [
            _bm("kappa2_OO", [[2], [3]], "bidentate O,O'-nitrate", "OO", "kappa2", "bidentate"),
            _bm("kappa1_O", [[2]], "monodentate O-nitrate", "O", "kappa1", "monodentate"),
        ],
        aliases=("nitrate", "NO3-"),
        notes="Nitrate anion.",
    ),
    _lig(
        "CO3", "O=C([O-])[O-]", -2, "oxyanion",
        [
            _bm("kappa2_OO", [[2], [3]], "bidentate O,O'-carbonate", "OO", "kappa2", "bidentate"),
            _bm("kappa1_O", [[2]], "monodentate O-carbonate", "O", "kappa1", "monodentate"),
        ],
        aliases=("carbonate", "CO3^2-"),
        notes="Carbonate dianion.",
    ),

    # ------------------------------------------------------------------
    # Polyatomic heterocycles, solvents & oxyanions (9)
    # ------------------------------------------------------------------
    _lig(
        "py", "c1ccncc1", 0, "heterocycle",
        [_bm("kappa1_N", [[3]], "N-bound pyridine", "N", "kappa1")],
        aliases=("pyridine", "pyr"),
        notes="Pyridine.",
    ),
    _lig(
        "MeCN", "CC#N", 0, "solvent",
        [_bm("kappa1_N", [[2]], "nitrile N-bound acetonitrile", "N", "kappa1")],
        aliases=("acetonitrile", "CH3CN", "NCMe", "ACN"),
        notes="Acetonitrile.",
    ),
    _lig(
        "Him", "c1c[nH]cn1", 0, "heterocycle",
        [_bm("kappa1_N", [[4]], "iminic N3-bound imidazole", "N", "kappa1")],
        aliases=("imidazole", "ImH"),
        notes="Neutral 1H-imidazole coordinating via the imine nitrogen.",
    ),
    _lig(
        "MeIm", "Cn1ccnc1", 0, "heterocycle",
        [_bm("kappa1_N", [[4]], "iminic N3-bound N-methylimidazole", "N", "kappa1")],
        aliases=("N-methylimidazole", "1-MeIm", "NMI"),
        notes="1-Methylimidazole.",
    ),
    _lig(
        "thf", "C1CCOC1", 0, "solvent",
        [_bm("kappa1_O", [[3]], "ether O-bound tetrahydrofuran", "O", "kappa1")],
        aliases=("tetrahydrofuran",),
        notes="Tetrahydrofuran.",
    ),
    _lig(
        "dmso", "CS(C)=O", 0, "solvent",
        [
            _bm("kappa1_S", [[1]], "S-bound dimethyl sulfoxide", "S"),
            _bm("kappa1_O", [[3]], "O-bound dimethyl sulfoxide", "O"),
        ],
        aliases=("dimethylsulfoxide",),
        notes="Ambidentate dimethyl sulfoxide (soft metals favor S, hard metals O).",
    ),
    _lig(
        "dmf", "CN(C)C=O", 0, "solvent",
        [
            _bm("kappa1_O", [[4]], "carbonyl O-bound DMF", "O", "kappa1"),
            _bm("kappa1_N", [[1]], "amide N-bound DMF", "N"),
        ],
        aliases=("dimethylformamide",),
        notes="N,N-Dimethylformamide.",
    ),
    _lig(
        "OTf", "O=S(=O)([O-])C(F)(F)F", -1, "oxyanion",
        [
            _bm("kappa1_O", [[3]], "monodentate O-triflate", "O", "kappa1", "monodentate"),
            _bm("kappa2_OO", [[2], [3]], "bidentate O,O'-triflate", "OO", "kappa2", "bidentate"),
        ],
        aliases=("triflate", "TfO", "CF3SO3", "OTf-"),
        notes="Trifluoromethanesulfonate (triflate).",
    ),
    _lig(
        "SO4", "O=S(=O)([O-])[O-]", -2, "oxyanion",
        [
            _bm("kappa2_OO", [[3], [4]], "bidentate O,O'-sulfate", "OO", "kappa2", "bidentate"),
            _bm("kappa1_O", [[3]], "monodentate O-sulfate", "O", "kappa1", "monodentate"),
        ],
        aliases=("sulfate", "SO4^2-"),
        notes="Sulfate dianion.",
    ),

    # ------------------------------------------------------------------
    # Phosphines, phosphites, phosphine oxides & diphosphines (11)
    # ------------------------------------------------------------------
    _lig(
        "PMe3", "CP(C)C", 0, "phosphine",
        [_bm("kappa1_P", [[1]], "P-bound trimethylphosphine", "P", "kappa1")],
        aliases=("trimethylphosphine",),
        notes="Trimethylphosphine.",
    ),
    _lig(
        "PPh3", "P(c1ccccc1)(c1ccccc1)c1ccccc1", 0, "phosphine",
        [_bm("kappa1_P", [[0]], "P-bound triphenylphosphine", "P", "kappa1")],
        aliases=("triphenylphosphine", "Ph3P"),
        notes="Triphenylphosphine.",
    ),
    _lig(
        "PCy3", "P(C1CCCCC1)(C1CCCCC1)C1CCCCC1", 0, "phosphine",
        [_bm("kappa1_P", [[0]], "P-bound tricyclohexylphosphine", "P", "kappa1")],
        aliases=("tricyclohexylphosphine", "Cy3P"),
        notes="Tricyclohexylphosphine.",
    ),
    _lig(
        "P(OMe)3", "COP(OC)OC", 0, "phosphine",
        [_bm("kappa1_P", [[2]], "P-bound trimethylphosphite", "P", "kappa1")],
        aliases=("trimethylphosphite", "POMe3"),
        notes="Trimethyl phosphite.",
    ),
    _lig(
        "OPPh3", "O=P(c1ccccc1)(c1ccccc1)c1ccccc1", 0, "phosphine",
        [_bm("kappa1_O", [[0]], "O-bound triphenylphosphine oxide", "O", "kappa1")],
        aliases=("TPPO", "Ph3PO", "triphenylphosphine oxide"),
        notes="Triphenylphosphine oxide.",
    ),
    _lig(
        "dppm", "c1ccc(P(CP(c2ccccc2)c2ccccc2)c2ccccc2)cc1", 0, "phosphine",
        [
            _bm("kappa2_PP", [[4], [6]], "P,P'-chelating dppm", "PP", "kappa2", "bidentate"),
            _bm("kappa1_P", [[4]], "monodentate P-bound dppm", "P", "kappa1", "monodentate"),
        ],
        aliases=("bis(diphenylphosphino)methane",),
        notes="1,1-Bis(diphenylphosphino)methane.",
    ),
    _lig(
        "dppe", "c1ccc(P(CCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1", 0, "phosphine",
        [
            _bm("kappa2_PP", [[4], [7]], "P,P'-chelating dppe", "PP", "kappa2", "bidentate"),
            _bm("kappa1_P", [[4]], "monodentate P-bound dppe", "P", "kappa1", "monodentate"),
        ],
        aliases=("diphos", "1,2-bis(diphenylphosphino)ethane"),
        notes="1,2-Bis(diphenylphosphino)ethane.",
    ),
    _lig(
        "dppp", "c1ccc(P(CCCP(c2ccccc2)c2ccccc2)c2ccccc2)cc1", 0, "phosphine",
        [
            _bm("kappa2_PP", [[4], [8]], "P,P'-chelating dppp", "PP", "kappa2", "bidentate"),
            _bm("kappa1_P", [[4]], "monodentate P-bound dppp", "P", "kappa1", "monodentate"),
        ],
        aliases=("1,3-bis(diphenylphosphino)propane",),
        notes="1,3-Bis(diphenylphosphino)propane.",
    ),
    _lig(
        "dmpe", "CP(C)CCP(C)C", 0, "phosphine",
        [
            _bm("kappa2_PP", [[1], [5]], "P,P'-chelating dmpe", "PP", "kappa2", "bidentate"),
            _bm("kappa1_P", [[1]], "monodentate P-bound dmpe", "P", "kappa1", "monodentate"),
        ],
        aliases=("1,2-bis(dimethylphosphino)ethane",),
        notes="1,2-Bis(dimethylphosphino)ethane.",
    ),
    _lig(
        "BINAP", "c1ccc(P(c2ccccc2)c2ccc3ccccc3c2-c2c(P(c3ccccc3)c3ccccc3)ccc3ccccc23)cc1", 0, "phosphine",
        [
            _bm("kappa2_PP", [[4], [23]], "P,P'-chelating BINAP", "PP", "kappa2", "bidentate"),
            _bm("kappa1_P", [[4]], "monodentate P-bound BINAP", "P", "kappa1", "monodentate"),
        ],
        notes="2,2'-Bis(diphenylphosphino)-1,1'-binaphthyl.",
    ),
    _lig(
        "Xantphos", "CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c1cccc2P(c1ccccc1)c1ccccc1", 0, "phosphine",
        [
            _bm("kappa2_PP", [[8], [29]], "P,P'-chelating Xantphos", "PP", "kappa2", "bidentate"),
            _bm("kappa3_POP", [[8], [22], [29]], "P,O,P-pincer Xantphos", "POP", "kappa3", "pincer"),
        ],
        notes="4,5-Bis(diphenylphosphino)-9,9-dimethylxanthene.",
    ),

    # ------------------------------------------------------------------
    # N-Heterocyclic carbenes (3)
    # ------------------------------------------------------------------
    _lig(
        "IMe", "Cn1cc[n+](C)[c-]1", 0, "nhc",
        [_bm("kappa1_C", [[6]], "carbene C2-bound IMe", "C", "kappa1")],
        notes="1,3-Dimethylimidazol-2-ylidene.",
    ),
    _lig(
        "IMes", "Cc1cc(C)c(-n2cc[n+](-c3c(C)cc(C)cc3C)[c-]2)c(C)c1", 0, "nhc",
        [_bm("kappa1_C", [[19]], "carbene C2-bound IMes", "C", "kappa1")],
        notes="1,3-Bis(2,4,6-trimethylphenyl)imidazol-2-ylidene.",
    ),
    _lig(
        "IPr", "CC(C)c1cccc(C(C)C)c1-n1cc[n+](-c2c(C(C)C)cccc2C(C)C)[c-]1", 0, "nhc",
        [_bm("kappa1_C", [[28]], "carbene C2-bound IPr", "C", "kappa1")],
        notes="1,3-Bis(2,6-diisopropylphenyl)imidazol-2-ylidene.",
    ),

    # ------------------------------------------------------------------
    # Bidentate chelators: N,N / O,O / N,O / S,S (20)
    # ------------------------------------------------------------------
    _lig(
        "en", "NCCN", 0, "bidentate",
        [
            _bm("kappa2_NN", [[0], [3]], "N,N'-chelating ethylenediamine", "NN", "kappa2", "bidentate"),
            _bm("kappa1_N", [[0]], "monodentate N-bound ethylenediamine", "N", "kappa1", "monodentate"),
        ],
        aliases=("ethylenediamine", "1,2-diaminoethane"),
        notes="Ethylenediamine.",
    ),
    _lig(
        "tmeda", "CN(C)CCN(C)C", 0, "bidentate",
        [
            _bm("kappa2_NN", [[1], [5]], "N,N'-chelating TMEDA", "NN", "kappa2", "bidentate"),
            _bm("kappa1_N", [[1]], "monodentate N-bound TMEDA", "N", "kappa1", "monodentate"),
        ],
        aliases=("tetramethylethylenediamine",),
        notes="N,N,N',N'-Tetramethylethylenediamine.",
    ),
    _lig(
        "bpy", "c1ccc(-c2ccccn2)nc1", 0, "bidentate",
        [_bm("kappa2_NN", [[9], [10]], "N,N'-chelating 2,2'-bipyridine", "NN", "kappa2")],
        aliases=("bipy", "bipyridine", "2,2'-bipyridine", "2,2'-bpy"),
        notes="2,2'-Bipyridine.",
    ),
    _lig(
        "phen", "c1cnc2c(c1)ccc1cccnc12", 0, "bidentate",
        [_bm("kappa2_NN", [[2], [12]], "N,N'-chelating 1,10-phenanthroline", "NN", "kappa2")],
        aliases=("phenanthroline", "1,10-phenanthroline", "o-phen"),
        notes="1,10-Phenanthroline.",
    ),
    _lig(
        "bpym", "c1cnc(-c2ncccn2)nc1", 0, "bidentate",
        [_bm("kappa2_NN", [[5], [10]], "N,N'-chelating 2,2'-bipyrimidine", "NN", "kappa2")],
        aliases=("2,2'-bipyrimidine", "bipyrimidine"),
        notes="2,2'-Bipyrimidine (one chelating N,N' pocket).",
    ),
    _lig(
        "DMG", "CC(=NO)C(=N[O-])C", -1, "bidentate",
        [
            _bm("kappa2_NN", [[2], [5]], "N,N'-chelating dimethylglyoximate", "NN", "kappa2"),
            _bm("kappa2_NO", [[2], [6]], "N,O-chelating dimethylglyoximate", "NO"),
        ],
        aliases=("dimethylglyoximate", "dmgH-"),
        notes="Monoanionic dimethylglyoximate.",
    ),
    _lig(
        "acac", "CC(=O)[CH-]C(C)=O", -1, "bidentate",
        [
            _bm("kappa2_OO", [[2], [6]], "O,O'-chelating acetylacetonate", "OO", "kappa2", "bidentate"),
            _bm("kappa1_C", [[3]], "gamma-C-bound acetylacetonate", "C", "C3", "kappa1_C"),
        ],
        aliases=("acetylacetonate", "acac-"),
        notes="Acetylacetonate anion.",
    ),
    _lig(
        "hfac", "O=C([CH-]C(=O)C(F)(F)F)C(F)(F)F", -1, "bidentate",
        [
            _bm("kappa2_OO", [[0], [4]], "O,O'-chelating hexafluoroacetylacetonate", "OO", "kappa2"),
            _bm("kappa1_C", [[2]], "gamma-C-bound hexafluoroacetylacetonate", "C", "C3"),
        ],
        aliases=("hexafluoroacetylacetonate", "hfacac"),
        notes="Hexafluoroacetylacetonate anion.",
    ),
    _lig(
        "OAc", "CC(=O)[O-]", -1, "bidentate",
        [
            _bm("kappa1_O", [[3]], "monodentate O-acetate", "O", "kappa1", "monodentate"),
            _bm("kappa2_OO", [[2], [3]], "chelating O,O'-acetate", "OO", "kappa2", "bidentate"),
        ],
        aliases=("acetate", "AcO", "CH3COO", "OAc-"),
        notes="Acetate anion.",
    ),
    _lig(
        "HCO2", "O=C[O-]", -1, "bidentate",
        [
            _bm("kappa1_O", [[2]], "monodentate O-formate", "O", "kappa1", "monodentate"),
            _bm("kappa2_OO", [[0], [2]], "chelating O,O'-formate", "OO", "kappa2", "bidentate"),
        ],
        aliases=("formate", "OCHO", "HCOO"),
        notes="Formate anion.",
    ),
    _lig(
        "OBz", "O=C([O-])c1ccccc1", -1, "bidentate",
        [
            _bm("kappa1_O", [[2]], "monodentate O-benzoate", "O", "kappa1", "monodentate"),
            _bm("kappa2_OO", [[0], [2]], "chelating O,O'-benzoate", "OO", "kappa2", "bidentate"),
        ],
        aliases=("benzoate", "BzO", "PhCOO"),
        notes="Benzoate anion.",
    ),
    _lig(
        "ox", "O=C([O-])C(=O)[O-]", -2, "bidentate",
        [
            _bm("kappa2_OO", [[2], [5]], "O,O'-chelating oxalate", "OO", "kappa2", "bidentate"),
            _bm("kappa1_O", [[2]], "monodentate O-oxalate", "O", "kappa1", "monodentate"),
        ],
        aliases=("oxalate", "C2O4", "ox^2-"),
        notes="Oxalate dianion.",
    ),
    _lig(
        "cat", "[O-]c1ccccc1[O-]", -2, "bidentate",
        [
            _bm("kappa2_OO", [[0], [7]], "O,O'-chelating catecholate", "OO", "kappa2", "bidentate"),
            _bm("kappa1_O", [[0]], "monodentate O-catecholate", "O", "kappa1", "monodentate"),
        ],
        aliases=("catecholate", "catechol"),
        notes="1,2-Benzenediolate (catecholate) dianion.",
    ),
    _lig(
        "dtbc", "CC(C)(C)c1cc([O-])c([O-])c(C(C)(C)C)c1", -2, "bidentate",
        [_bm("kappa2_OO", [[7], [9]], "O,O'-chelating 3,5-di-tert-butylcatecholate", "OO", "kappa2")],
        aliases=("3,5-dtbc", "DBCat"),
        notes="3,5-Di-tert-butylcatecholate dianion.",
    ),
    _lig(
        "gly", "NCC(=O)[O-]", -1, "bidentate",
        [
            _bm("kappa2_NO", [[0], [4]], "N,O-chelating glycinate", "NO", "kappa2", "bidentate"),
            _bm("kappa1_N", [[0]], "N-bound glycinate", "N"),
            _bm("kappa1_O", [[4]], "O-bound glycinate", "O"),
        ],
        aliases=("glycinate", "gly-"),
        notes="Glycinate anion.",
    ),
    _lig(
        "pic", "O=C([O-])c1ccccn1", -1, "bidentate",
        [
            _bm("kappa2_NO", [[8], [2]], "N,O-chelating picolinate", "NO", "kappa2", "bidentate"),
            _bm("kappa1_O", [[2]], "monodentate O-picolinate", "O", "kappa1"),
        ],
        aliases=("picolinate",),
        notes="2-Pyridinecarboxylate (picolinate) anion.",
    ),
    _lig(
        "q", "[O-]c1cccc2cccnc12", -1, "bidentate",
        [
            _bm("kappa2_NO", [[9], [0]], "N,O-chelating 8-hydroxyquinolinate", "NO", "kappa2", "bidentate"),
            _bm("kappa1_O", [[0]], "monodentate O-oxinate", "O", "kappa1"),
        ],
        aliases=("oxinate", "8-hydroxyquinolinate", "8-quinolinate", "hq"),
        notes="8-Hydroxyquinolinate (oxinate) anion.",
    ),
    _lig(
        "Et2dtc", "CCN(CC)C(=S)[S-]", -1, "bidentate",
        [
            _bm("kappa2_SS", [[6], [7]], "S,S'-chelating diethyldithiocarbamate", "SS", "kappa2", "bidentate"),
            _bm("kappa1_S", [[7]], "monodentate S-dithiocarbamate", "S", "kappa1"),
        ],
        aliases=("dtc", "diethyldithiocarbamate", "dedtc"),
        notes="N,N-Diethyldithiocarbamate anion.",
    ),
    _lig(
        "mnt", "N#C/C([S-])=C(/[S-])C#N", -2, "bidentate",
        [_bm("kappa2_SS", [[3], [5]], "S,S'-chelating cis-maleonitriledithiolate", "SS", "kappa2")],
        aliases=("maleonitriledithiolate",),
        notes="cis-1,2-Dicyanoethylene-1,2-dithiolate dianion.",
    ),
    _lig(
        "bdt", "[S-]c1ccccc1[S-]", -2, "bidentate",
        [
            _bm("kappa2_SS", [[0], [7]], "S,S'-chelating benzene-1,2-dithiolate", "SS", "kappa2", "bidentate"),
            _bm("kappa1_S", [[0]], "monodentate S-benzenedithiolate", "S", "kappa1"),
        ],
        aliases=("benzenedithiolate", "1,2-benzenedithiolate"),
        notes="Benzene-1,2-dithiolate dianion.",
    ),

    # ------------------------------------------------------------------
    # Polydentates, pincers, tripodals & macrocycles (18)
    # ------------------------------------------------------------------
    _lig(
        "terpy", "c1ccc(-c2cccc(-c3ccccn3)n2)nc1", 0, "polydentate",
        [
            _bm("kappa3_NNN", [[14], [15], [16]], "N,N',N''-tridentate terpyridine", "NNN", "kappa3", "tridentate"),
            _bm("kappa2_NN", [[14], [15]], "N,N'-bidentate terpyridine", "NN", "kappa2", "bidentate"),
        ],
        aliases=("tpy", "terpyridine", "2,2';6',2''-terpyridine"),
        notes="2,2':6',2''-Terpyridine.",
    ),
    _lig(
        "dien", "NCCNCCN", 0, "polydentate",
        [
            _bm("kappa3_NNN", [[0], [3], [6]], "N,N',N''-tridentate diethylenetriamine", "NNN", "kappa3", "tridentate"),
            _bm("kappa2_NN", [[0], [3]], "N,N'-bidentate diethylenetriamine", "NN", "kappa2", "bidentate"),
        ],
        aliases=("diethylenetriamine", "deta"),
        notes="Diethylenetriamine.",
    ),
    _lig(
        "tacn", "C1CNCCNCCN1", 0, "macrocycle",
        [
            _bm("kappa3_NNN", [[2], [5], [8]], "facial N3-capping 1,4,7-triazacyclononane", "NNN", "kappa3", "tridentate"),
            _bm("kappa2_NN", [[2], [5]], "bidentate N,N'-tacn", "NN", "kappa2", "bidentate"),
        ],
        aliases=("1,4,7-triazacyclononane", "[9]aneN3"),
        notes="1,4,7-Triazacyclononane.",
    ),
    _lig(
        "Tp", "[BH-](n1cccn1)(n1cccn1)n1cccn1", -1, "polydentate",
        [
            _bm("kappa3_NNN", [[5], [10], [15]], "facial N3-scorpionate Tp", "NNN", "kappa3", "tridentate"),
            _bm("kappa2_NN", [[5], [10]], "bidentate N,N'-Tp", "NN", "kappa2", "bidentate"),
        ],
        aliases=("tris(pyrazolyl)borate", "scorpionate"),
        notes="Hydrotris(pyrazol-1-yl)borate anion.",
    ),
    _lig(
        "Tp*", "Cc1cc(C)n([BH-](n2c(C)cc(C)n2)n2c(C)cc(C)n2)n1", -1, "polydentate",
        [
            _bm("kappa3_NNN", [[13], [20], [21]], "facial N3-scorpionate Tp*", "NNN", "kappa3", "tridentate"),
            _bm("kappa2_NN", [[13], [20]], "bidentate N,N'-Tp*", "NN", "kappa2", "bidentate"),
        ],
        aliases=("TpMe2", "TpStar"),
        notes="Hydrotris(3,5-dimethylpyrazol-1-yl)borate anion.",
    ),
    _lig(
        "PyBox", "O1CCN=C1c1cccc(C2=NCCO2)n1", 0, "polydentate",
        [
            _bm("kappa3_NNN", [[3], [15], [11]], "N,N',N''-pincer PyBox", "NNN", "kappa3", "tridentate"),
            _bm("kappa2_NN", [[3], [15]], "N,N'-bidentate PyBox", "NN", "kappa2", "bidentate"),
        ],
        aliases=("2,6-bis(oxazolinyl)pyridine",),
        notes="2,6-Bis(4,5-dihydrooxazol-2-yl)pyridine.",
    ),
    _lig(
        "triphos", "CC(CP(c1ccccc1)c1ccccc1)(CP(c1ccccc1)c1ccccc1)CP(c1ccccc1)c1ccccc1", 0, "polydentate",
        [
            _bm("kappa3_PPP", [[3], [17], [31]], "facial P3-tripodal triphos", "PPP", "kappa3", "tridentate"),
            _bm("kappa2_PP", [[3], [17]], "P,P'-bidentate triphos", "PP", "kappa2", "bidentate"),
        ],
        aliases=("tripod-triphos",),
        notes="1,1,1-Tris(diphenylphosphinomethyl)ethane.",
    ),
    _lig(
        "trien", "NCCNCCNCCN", 0, "polydentate",
        [
            _bm("kappa4_NNNN", [[0], [3], [6], [9]], "tetradentate N4 triethylenetetramine", "NNNN", "kappa4"),
            _bm("kappa3_NNN", [[0], [3], [6]], "tridentate N3 trien", "NNN", "kappa3"),
        ],
        aliases=("triethylenetetramine", "teta"),
        notes="Triethylenetetramine.",
    ),
    _lig(
        "tren", "NCCN(CCN)CCN", 0, "polydentate",
        [
            _bm("kappa4_NNNN", [[0], [3], [6], [9]], "tripodal tetradentate N4 tren", "NNNN", "kappa4"),
            _bm("kappa3_NNN", [[0], [3], [6]], "tridentate N3 tren", "NNN", "kappa3"),
        ],
        aliases=("tris(2-aminoethyl)amine",),
        notes="Tris(2-aminoethyl)amine.",
    ),
    _lig(
        "cyclen", "C1CNCCNCCNCCN1", 0, "macrocycle",
        [_bm("kappa4_NNNN", [[2], [5], [8], [11]], "tetradentate N4 macrocycle cyclen", "NNNN", "kappa4")],
        aliases=("1,4,7,10-tetraazacyclododecane", "[12]aneN4"),
        notes="1,4,7,10-Tetraazacyclododecane.",
    ),
    _lig(
        "cyclam", "C1CNCCCNCCNCCCN1", 0, "macrocycle",
        [_bm("kappa4_NNNN", [[2], [6], [9], [13]], "tetradentate N4 macrocycle cyclam", "NNNN", "kappa4")],
        aliases=("1,4,8,11-tetraazacyclotetradecane", "[14]aneN4"),
        notes="1,4,8,11-Tetraazacyclotetradecane.",
    ),
    _lig(
        "salen", "[O-]c1ccccc1C=NCCN=Cc1ccccc1[O-]", -2, "polydentate",
        [_bm("kappa4_ONNO", [[0], [8], [11], [19]], "O,N,N',O'-tetradentate salen", "ONNO", "kappa4")],
        aliases=("salen^2-",),
        notes="N,N'-Ethylenebis(salicylideneiminate) dianion.",
    ),
    _lig(
        "porphine", "c1cc2cc3ccc([n-]3)cc3ccc(n3)cc3ccc([n-]3)cc1n2", -2, "macrocycle",
        [_bm("kappa4_NNNN", [[8], [14], [20], [23]], "tetradentate N4 porphyrinate", "NNNN", "kappa4")],
        aliases=("porphyrin", "porphyrinate"),
        notes="Unsubstituted porphyrinate core dianion.",
    ),
    _lig(
        "TPP", "c1ccc(-c2c3nc(c(-c4ccccc4)c4ccc([n-]4)c(-c4ccccc4)c4nc(c(-c5ccccc5)c5ccc2[n-]5)C=C4)C=C3)cc1", -2, "macrocycle",
        [_bm("kappa4_NNNN", [[6], [19], [28], [41]], "tetradentate N4 tetraphenylporphyrinate", "NNNN", "kappa4")],
        aliases=("tetraphenylporphyrin", "tetraphenylporphyrinate", "H2TPP^2-"),
        notes="meso-Tetraphenylporphyrinate dianion.",
    ),
    _lig(
        "NTA", "O=C([O-])CN(CC(=O)[O-])CC(=O)[O-]", -3, "polydentate",
        [
            _bm("kappa4_NOOO", [[4], [2], [8], [12]], "tetradentate N,O3 nitrilotriacetate", "NOOO", "kappa4"),
            _bm("kappa3_NOO", [[4], [2], [8]], "tridentate N,O2 nitrilotriacetate", "NOO", "kappa3"),
        ],
        aliases=("nitrilotriacetate",),
        notes="Nitrilotriacetate trianion.",
    ),
    _lig(
        "EDTA", "[O-]C(=O)CN(CC(=O)[O-])CCN(CC(=O)[O-])CC(=O)[O-]", -4, "polydentate",
        [
            _bm("kappa6_N2O4", [[0], [4], [8], [11], [15], [19]], "hexadentate N2O4 EDTA", "N2O4", "kappa6"),
            _bm("kappa5_N2O3", [[0], [4], [8], [11], [15]], "pentadentate N2O3 EDTA", "N2O3", "kappa5"),
            _bm("kappa4_N2O2", [[0], [4], [11], [15]], "tetradentate N2O2 EDTA", "N2O2", "kappa4"),
        ],
        aliases=("edta^4-", "ethylenediaminetetraacetate"),
        notes="Ethylenediaminetetraacetate tetraanion.",
    ),
    _lig(
        "DOTA", "O=C([O-])CN1CCN(CC(=O)[O-])CCN(CC(=O)[O-])CCN(CC(=O)[O-])CC1", -4, "macrocycle",
        [
            _bm("kappa8_N4O4", [[4], [7], [14], [21], [2], [11], [18], [25]], "octadentate N4O4 DOTA", "N4O4", "kappa8"),
            _bm("kappa4_NNNN", [[4], [7], [14], [21]], "tetradentate cyclen-ring N4 DOTA", "NNNN", "kappa4"),
        ],
        aliases=("dota^4-",),
        notes="1,4,7,10-Tetraazacyclododecane-1,4,7,10-tetraacetate tetraanion.",
    ),
    _lig(
        "18-crown-6", "C1COCCOCCOCCOCCOCCO1", 0, "macrocycle",
        [_bm("kappa6_O6", [[2], [5], [8], [11], [14], [17]], "hexadentate O6 crown ether", "O6", "kappa6")],
        aliases=("18c6", "18crown6"),
        notes="18-Crown-6 ether macrocycle.",
    ),

    # ------------------------------------------------------------------
    # Organometallic pi / haptic ligands (10)
    # ------------------------------------------------------------------
    _lig(
        "ethylene", "C=C", 0, "haptic",
        [_bm("eta2", [[0, 1]], "eta2 alkene ethylene")],
        aliases=("ethene", "C2H4"),
        notes="Ethylene (pi-bound alkene).",
    ),
    _lig(
        "allyl", "[CH2-]C=C", -1, "haptic",
        [
            _bm("eta3", [[0, 1, 2]], "eta3 pi-allyl"),
            _bm("kappa1_C", [[0]], "kappa1 sigma-allyl", "C", "kappa1", "sigma"),
        ],
        aliases=("C3H5-",),
        notes="Allyl anion (eta3 pi-allyl / kappa1 sigma-allyl).",
    ),
    _lig(
        "cod", "C1=CCCC=CCC1", 0, "haptic",
        [
            _bm("eta2_eta2", [[0, 1], [4, 5]], "chelating bis(eta2) 1,5-cyclooctadiene", "eta4", "bidentate"),
            _bm("eta2", [[0, 1]], "monodentate eta2 1,5-cyclooctadiene", "monodentate"),
        ],
        aliases=("1,5-cod", "cyclooctadiene", "1,5-cyclooctadiene"),
        notes="1,5-Cyclooctadiene (counts as 2 coordination sites in eta2_eta2 mode).",
    ),
    _lig(
        "nbd", "C1=CC2C=CC1C2", 0, "haptic",
        [
            _bm("eta2_eta2", [[0, 1], [3, 4]], "chelating bis(eta2) norbornadiene", "eta4", "bidentate"),
            _bm("eta2", [[0, 1]], "monodentate eta2 norbornadiene", "monodentate"),
        ],
        aliases=("norbornadiene", "bicyclo[2.2.1]hepta-2,5-diene"),
        notes="2,5-Norbornadiene.",
    ),
    _lig(
        "Cp", "[cH-]1cccc1", -1, "haptic",
        [
            _bm("eta5", [[0, 1, 2, 3, 4]], "eta5 cyclopentadienyl"),
            _bm("eta3", [[0, 1, 2]], "ring-slipped eta3 cyclopentadienyl"),
            _bm("kappa1_C", [[0]], "sigma-bound kappa1 cyclopentadienyl", "C", "kappa1", "sigma"),
        ],
        aliases=("cyclopentadienyl", "C5H5-"),
        notes="Cyclopentadienyl anion (eta5 / eta3 / kappa1-C).",
    ),
    _lig(
        "MeCp", "C[c-]1cccc1", -1, "haptic",
        [
            _bm("eta5", [[1, 2, 3, 4, 5]], "eta5 methylcyclopentadienyl"),
            _bm("kappa1_C", [[1]], "kappa1-C methylcyclopentadienyl", "C", "kappa1"),
        ],
        aliases=("Cp'", "methylcyclopentadienyl"),
        notes="Methylcyclopentadienyl anion.",
    ),
    _lig(
        "Cp*", "Cc1c(C)c(C)[c-](C)c1C", -1, "haptic",
        [
            _bm("eta5", [[1, 2, 4, 6, 8]], "eta5 pentamethylcyclopentadienyl"),
            _bm("kappa1_C", [[6]], "kappa1-C pentamethylcyclopentadienyl", "C", "kappa1"),
        ],
        aliases=("CpStar", "pentamethylcyclopentadienyl", "C5Me5-"),
        notes="Pentamethylcyclopentadienyl anion.",
    ),
    _lig(
        "Ind", "[cH-]1cc2ccccc2c1", -1, "haptic",
        [
            _bm("eta5", [[0, 1, 2, 7, 8]], "eta5 indenyl"),
            _bm("eta3", [[8, 0, 1]], "ring-slipped eta3 indenyl"),
            _bm("kappa1_C", [[0]], "kappa1-C indenyl", "C", "kappa1"),
        ],
        aliases=("indenyl", "C9H7-"),
        notes="Indenyl anion (facile eta5 <-> eta3 indenyl effect).",
    ),
    _lig(
        "benzene", "c1ccccc1", 0, "haptic",
        [
            _bm("eta6", [[0, 1, 2, 3, 4, 5]], "eta6 benzene"),
            _bm("eta2", [[0, 1]], "eta2 benzene"),
        ],
        aliases=("C6H6", "bz"),
        notes="Benzene arene ligand.",
    ),
    _lig(
        "cymene", "Cc1ccc(C(C)C)cc1", 0, "haptic",
        [
            _bm("eta6", [[1, 2, 3, 4, 8, 9]], "eta6 p-cymene"),
            _bm("eta2", [[2, 3]], "eta2 p-cymene"),
        ],
        aliases=("p-cymene", "para-cymene"),
        notes="para-Cymene (1-methyl-4-isopropylbenzene).",
    ),
)

# Canonical map preserving insertion order
LIGANDS: Dict[str, LigandDef] = {entry.name: entry for entry in _RAW_LIGAND_LIST}

# Mode-specific global aliases mapping intuitive chemical names -> "ligand@mode"
_MODE_SPECIFIC_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("isocarbonyl", "CO@kappa1_O"),
    ("isocyanide", "CN@kappa1_N"),
    ("thiocyanate-S", "SCN@kappa1_S"),
    ("S-thiocyanate", "SCN@kappa1_S"),
    ("isothiocyanate", "SCN@kappa1_N"),
    ("NCS", "SCN@kappa1_N"),
    ("N-thiocyanate", "SCN@kappa1_N"),
    ("cyanate-O", "OCN@kappa1_O"),
    ("O-cyanate", "OCN@kappa1_O"),
    ("isocyanate", "OCN@kappa1_N"),
    ("NCO", "OCN@kappa1_N"),
    ("N-cyanate", "OCN@kappa1_N"),
    ("nitro", "NO2@kappa1_N"),
    ("nitrito", "NO2@kappa1_O"),
    ("nitrito-O", "NO2@kappa1_O"),
    ("chelating-nitrite", "NO2@kappa2_OO"),
    ("dmso-S", "dmso@kappa1_S"),
    ("S-dmso", "dmso@kappa1_S"),
    ("dmso-O", "dmso@kappa1_O"),
    ("O-dmso", "dmso@kappa1_O"),
    ("dmf-O", "dmf@kappa1_O"),
    ("dmf-N", "dmf@kappa1_N"),
    ("kappa1-OAc", "OAc@kappa1_O"),
    ("kappa2-OAc", "OAc@kappa2_OO"),
    ("chelating-acetate", "OAc@kappa2_OO"),
    ("kappa1-acac", "acac@kappa1_C"),
    ("C-acac", "acac@kappa1_C"),
    ("eta2-cod", "cod@eta2"),
    ("eta3-allyl", "allyl@eta3"),
    ("pi-allyl", "allyl@eta3"),
    ("sigma-allyl", "allyl@kappa1_C"),
    ("kappa1-allyl", "allyl@kappa1_C"),
    ("eta5-Cp", "Cp@eta5"),
    ("eta3-Cp", "Cp@eta3"),
    ("sigma-Cp", "Cp@kappa1_C"),
    ("kappa1-Cp", "Cp@kappa1_C"),
    ("eta5-Cp*", "Cp*@eta5"),
    ("eta5-Ind", "Ind@eta5"),
    ("eta3-Ind", "Ind@eta3"),
    ("eta6-benzene", "benzene@eta6"),
    ("eta6-cymene", "cymene@eta6"),
    ("eta6-p-cymene", "cymene@eta6"),
)


def _normalize_alias_key(text: str) -> str:
    return text.strip().replace("\u03ba", "kappa").replace("\u03b7", "eta").lower()


def _build_aliases() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build exact lowercase lookup and normalized alias -> target mappings without collisions."""
    exact_ci: Dict[str, str] = {}
    aliases: Dict[str, str] = {}

    def _register(alias: str, target: str, *, allow_exact: bool = True) -> None:
        key = _normalize_alias_key(alias)
        if allow_exact:
            if key in exact_ci and exact_ci[key] != target:
                raise ValueError(f"Duplicate ligand key {alias!r}: {exact_ci[key]!r} vs {target!r}")
            exact_ci[key] = target
        if alias != target:
            if key in aliases and aliases[key] != target:
                raise ValueError(f"Duplicate ligand alias {alias!r}: {aliases[key]!r} vs {target!r}")
            aliases[key] = target

    for name, ldef in LIGANDS.items():
        _register(name, name)
        for alias in ldef.aliases:
            _register(alias, name)
    for alias, target in _MODE_SPECIFIC_ALIASES:
        _register(alias, target)
    return exact_ci, aliases


_EXACT_CI, ALIASES = _build_aliases()

_REPEAT_RE = re.compile(r"^(.+?)\*(\d+)$")


def _split_count_suffix(spec: str) -> Tuple[str, int]:
    """Parse optional ``*N`` repeat suffix (for example ``bpy*3`` or ``Cp**2``)."""
    m = _REPEAT_RE.fullmatch(spec.strip())
    if m is None:
        return spec.strip(), 1
    count = int(m.group(2))
    if count < 1:
        raise ValueError(f"Ligand repeat count must be >= 1 in {spec!r}")
    return m.group(1).strip(), count


def _split_mode_suffix(spec: str) -> Tuple[str, Optional[str]]:
    """Split ``name@mode`` or ``name:mode`` (when RHS is not a JSON donor list)."""
    text = spec.strip()
    if "@" in text:
        base, mode = text.rsplit("@", 1)
        return base.strip(), mode.strip()
    if ":" in text:
        base, rhs = text.rsplit(":", 1)
        if not rhs.lstrip().startswith("["):
            return base.strip(), rhs.strip()
    return text, None


def resolve_ligand_def(name: str) -> Tuple[LigandDef, Optional[str]]:
    """Resolve a canonical ligand name or alias to ``(LigandDef, implied_mode)``."""
    raw = str(name).strip()
    if raw in LIGANDS:
        return LIGANDS[raw], None
    key = _normalize_alias_key(raw)
    target = _EXACT_CI.get(key) or ALIASES.get(key)
    if target is None:
        known = sorted(list(LIGANDS.keys()) + list(ALIASES.keys()))
        suggestions = difflib.get_close_matches(raw, known, n=5, cutoff=0.55)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise KeyError(f"Unknown ligand {name!r} in ligand_library.{hint}")
    if "@" in target:
        canon, implied_mode = target.split("@", 1)
        return LIGANDS[canon], implied_mode
    return LIGANDS[target], None


def ligand(
    spec: str,
    mode: Optional[str] = None,
    *,
    sites: Optional[Sequence[int]] = None,
) -> LigandInput:
    """Resolve a ligand name, alias, or ``name@mode`` string into a ``LigandInput``.

    Examples::

        ligand("bpy")
        ligand("dmso", "kappa1_O")
        ligand("dmso@kappa1_O")
        ligand("isothiocyanate")
    """
    base, inline_mode = _split_mode_suffix(spec)
    ldef, alias_mode = resolve_ligand_def(base)
    if mode is not None and inline_mode is not None and _normalize_mode_token(mode) != _normalize_mode_token(inline_mode):
        raise ValueError(f"Conflicting modes {inline_mode!r} (in spec {spec!r}) and {mode!r}")
    chosen_mode = mode if mode is not None else (inline_mode if inline_mode is not None else alias_mode)
    return ldef.to_input(chosen_mode, sites=sites)


def ligands(*specs: str) -> List[LigandInput]:
    """Expand one or more ligand specs (supporting ``*N`` counts, e.g. ``ligands('bpy*3')``)."""
    out: List[LigandInput] = []
    for raw in specs:
        base, count = _split_count_suffix(raw)
        item = ligand(base)
        for _ in range(count):
            out.append(LigandInput(item.smiles, donors=[list(g) for g in item.donors], sites=item.sites))
    return out


def expand_ligand_spec(spec: str) -> List[LigandInput]:
    """Expand a single library ligand string (with optional ``@mode`` and ``*N``) into ``List[LigandInput]``."""
    return ligands(spec)


def ligand_modes(name: str) -> List[str]:
    """Return available binding-mode names for a ligand or alias."""
    ldef, _ = resolve_ligand_def(name)
    return list(ldef.modes.keys())


def list_ligands(query: Optional[str] = None, *, category: Optional[str] = None) -> List[LigandDef]:
    """Filter curated ligands by substring query (names, aliases, notes) and/or category."""
    q = query.strip().lower() if query else None
    cat = category.strip().lower() if category else None
    results: List[LigandDef] = []
    for ldef in LIGANDS.values():
        if cat and ldef.category.lower() != cat:
            continue
        if q:
            haystack = " ".join(
                [ldef.name, ldef.category, ldef.notes, *ldef.aliases, *ldef.modes.keys()]
                + [bm.summary for bm in ldef.modes.values()]
                + [a for bm in ldef.modes.values() for a in bm.aliases]
            ).lower()
            if q not in haystack:
                continue
        results.append(ldef)
    return results


__all__ = [
    "ALIASES",
    "BindingMode",
    "DEBUG_COMPLEXES",
    "DEBUG_COMPLEX_CBTYCO",
    "DEBUG_COMPLEX_OJIMIF",
    "DebugComplex",
    "LIGANDS",
    "LigandDef",
    "debug_complex",
    "expand_ligand_spec",
    "ligand",
    "ligand_modes",
    "ligands",
    "list_ligands",
]
