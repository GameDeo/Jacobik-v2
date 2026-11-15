"""NUWFAM with the crystal's Zr...F contact as a donor: wraps the bench, adding the ortho F of the first C6F5 ring
(SMILES atom 8 of the borate ligand; crystal Zr-F 2.25 A) as a kappa donor and setting the geometry to tetrahedral
(Cp, Cp, eta2, F). Diagnostic only; no source change. Usage: same arguments as `python -m tools.jgen bench run`."""
import sys

import jacobik.core.env as P

_orig = P.build_builder_and_inputs


def _with_f(mol):
    b, lig, err = _orig(mol)
    if b is None or getattr(mol, 'csd_code', None) != 'NUWFAM':
        return b, lig, err
    L = b.ligands[0]
    assert L.smiles.startswith('[CH2-]C(C)=CC[B-]') and [list(g) for g in L.donors] == [[0, 1]], (L.smiles, L.donors)
    L.donors = [[0, 1], [8]]
    b.geometry = 'tetrahedral'
    return b, lig, err


P.build_builder_and_inputs = _with_f

# The crystal parse has no Zr-F donor, so the reference-to-generated atom mapping (which matches donor flags) must
# ignore the F donor: map on a copy whose ligands list it as an ordinary atom.
import copy  # noqa: E402

import jacobik.builder.atom_map as AM  # noqa: E402

_orig_map = AM.build_atom_mapping


def _map(ref_mol, gen_mol=None, **kw):
    g = copy.deepcopy(gen_mol)
    sym = {a.idx: a.symbol for a in g.atoms}
    for lig in g.ligands:
        lig.donor_atoms = [d for d in lig.donor_atoms if sym.get(d) != 'F']
    return _orig_map(ref_mol, gen_mol=g, **kw)


AM.build_atom_mapping = _map

if __name__ == '__main__':
    from tools.jgen.bench import run
    sys.exit(run.main(sys.argv[1:]))
