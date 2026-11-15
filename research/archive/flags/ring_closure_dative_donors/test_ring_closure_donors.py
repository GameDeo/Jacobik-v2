"""ring_closure_dative_donors (roadmap_generation_v1.md S7): legacy False; True keeps both SMILES donors when their
proximity remap lands on one xyz atom. Legacy output is the stored behaviour; a parent with no ring-closure dative is
unchanged by construction."""
import re

import pytest

from tests.jgen._data import need_data

RC = re.compile(r"(?:->|<-)\d")


def _donor_sites(code, shard, dataset, flag):
    from jacobik.stats import mapping as M
    from tools.jgen.inputs import load_mol

    old = M.RING_CLOSURE_DATIVE_DONORS
    M.RING_CLOSURE_DATIVE_DONORS = flag
    try:
        mol = load_mol(code, shard, dataset)
        assert mol is not None, code
        return sorted(tuple(s.donor_indices) for s in mol.coord_sites), len(mol.donor_labels)
    finally:
        M.RING_CLOSURE_DATIVE_DONORS = old


def test_legacy_default_is_false():
    import jacobik.profile as P
    from jacobik.stats import mapping as M

    assert P.FIELDS["ring_closure_dative_donors"][2] is False
    assert "ring_closure_dative_donors" not in P.PROFILES["u1"]
    assert M.RING_CLOSURE_DATIVE_DONORS is False


@need_data
def test_legacy_nepnef_has_four_donor_labels():
    from tools.jgen.inputs import frame_shards

    shard, ds = frame_shards()["NEPNEF"]
    sites, n_labels = _donor_sites("NEPNEF", shard, ds, False)
    assert n_labels == 4
    assert len(sites) == 4


@need_data
def test_on_nepnef_has_five_donor_labels():
    from tools.jgen.inputs import frame_shards

    shard, ds = frame_shards()["NEPNEF"]
    sites, n_labels = _donor_sites("NEPNEF", shard, ds, True)
    assert n_labels == 5
    assert len(sites) == 5


@need_data
def test_on_tinnio_gains_the_second_nitrate_oxygen():
    from tools.jgen.inputs import frame_shards

    shard, ds = frame_shards()["TINNIO"]
    _, off = _donor_sites("TINNIO", shard, ds, False)
    _, on = _donor_sites("TINNIO", shard, ds, True)
    assert (off, on) == (3, 4)


@need_data
def test_parent_without_ring_closure_dative_is_unchanged():
    from tools.jgen.inputs import frame_shards, tmqm_smiles, CLEAN

    smiles = tmqm_smiles(CLEAN / "tmQM")
    frame = frame_shards()
    code = next(c for c in sorted(frame) if c in smiles and smiles[c] and not RC.search(smiles[c]))
    shard, ds = frame[code]
    assert _donor_sites(code, shard, ds, False) == _donor_sites(code, shard, ds, True), code
