"""A cached MMFF fallback can never become a valid later benchmark parent."""
import logging
from types import SimpleNamespace
import numpy as np
import pytest
from rdkit import Chem
from tools.jgen.embedding_validity import EmbeddingValidity


def test_actual_and_cached_fallback_rejected_without_retry_or_geometry_change():
    calls=[];cache={}
    def refine(mol,positions,donors=None,*,experiment_flags=True):
        calls.append(True)
        if 'value' not in cache:
            logging.getLogger('jacobik.builder.engine').warning('GFN2 embed refinement failed for test; keeping the MMFF embed')
            cache['value']=positions.copy()
        return cache['value']
    engine=SimpleNamespace(_gfn2_refine=refine,EMBED_BOND_CHECK=False,EMBED_LI_CAP=True)
    mol=Chem.AddHs(Chem.MolFromSmiles('O'));X=np.zeros((mol.GetNumAtoms(),3))
    guard=EmbeddingValidity(engine).install()
    try:
        for parent in ('first','second'):
            guard.parent=parent
            assert np.array_equal(engine._gfn2_refine(mol,X,[[0]]),X)
            with pytest.raises(RuntimeError,match='all generated structures invalid'):guard.check(parent)
        assert len(calls)==2
        assert guard.invalid['second'][0]['cached_failure']
    finally:guard.close()
    assert engine._gfn2_refine is refine


def test_valid_geometry_is_retained_and_not_marked_invalid():
    def refine(mol,positions,donors=None,*,experiment_flags=True):return positions+1
    engine=SimpleNamespace(_gfn2_refine=refine,EMBED_BOND_CHECK=False,EMBED_LI_CAP=True)
    mol=Chem.AddHs(Chem.MolFromSmiles('O'));X=np.zeros((mol.GetNumAtoms(),3))
    guard=EmbeddingValidity(engine).install();guard.parent='ok'
    try:
        assert np.array_equal(engine._gfn2_refine(mol,X),X+1)
        guard.check('ok');assert not guard.invalid
    finally:guard.close()
