"""Geometry invariants needed to interpret the bond-only diagnostic experiment."""
import itertools
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from tools.jdev.budget.nci_internal_experiment import bridge_lengths


def molecule(smiles):
    m=Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(m,randomSeed=42)>=0
    return m,m.GetConformer().GetPositions()


def test_bridge_experiment_changes_lengths_without_changing_angles_or_ring():
    m,x=molecule('c1ccccc1NNc2ccccc2')
    target=x*1.04
    y=bridge_lengths(m,x,target)
    for b in m.GetBonds():
        a,c=b.GetBeginAtomIdx(),b.GetEndAtomIdx()
        heavy=m.GetAtomWithIdx(a).GetAtomicNum()>1 and m.GetAtomWithIdx(c).GetAtomicNum()>1
        expected=np.linalg.norm((target if heavy and not b.IsInRing() else x)[a]-(target if heavy and not b.IsInRing() else x)[c])
        assert np.isclose(np.linalg.norm(y[a]-y[c]),expected,atol=1e-10)
    for atom in m.GetAtoms():
        b=atom.GetIdx()
        for a,c in itertools.combinations([n.GetIdx() for n in atom.GetNeighbors()],2):
            cosine=lambda p: np.dot(p[a]-p[b],p[c]-p[b])/np.linalg.norm(p[a]-p[b])/np.linalg.norm(p[c]-p[b])
            assert np.isclose(cosine(x),cosine(y),atol=1e-10)
    for ring in m.GetRingInfo().AtomRings():
        for a,b in itertools.combinations(ring,2):
            assert np.isclose(np.linalg.norm(x[a]-x[b]),np.linalg.norm(y[a]-y[b]),atol=1e-10)


def test_bridge_experiment_preserves_torsions_and_does_not_mutate_inputs():
    m,x=molecule('CCNCC');original=x.copy();target=x*1.05
    y=bridge_lengths(m,x,target)
    from rdkit.Chem.rdMolTransforms import GetDihedralRad
    changed=Chem.Conformer(m.GetConformer())
    for i,p in enumerate(y):changed.SetAtomPosition(i,p)
    for ids in ((0,1,2,3),(1,2,3,4)):
        delta=GetDihedralRad(changed,*ids)-GetDihedralRad(m.GetConformer(),*ids)
        assert abs(np.arctan2(np.sin(delta),np.cos(delta)))<1e-10
    np.testing.assert_array_equal(x,original)
