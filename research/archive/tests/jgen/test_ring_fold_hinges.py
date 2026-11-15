"""ring_fold_hinges (roadmap_generation_v1.md 2.6, steps 1-3): legacy True; False removes only the ("hinge", 0) specs."""
import pytest

from tests.jgen._data import need_data

FOLD_PARENTS = ["HOXJUZ", "ZELTAP", "POJHIH"]
NO_FOLD_PARENTS = ["XOJHIN", "BEBQEE"]  # COD ring joints and a kappa-3 triazacyclohexane: 0 fold hinges
CEHPIO = ("CEHPIO", "X1s007")


def _specs(code, shard, dataset, flag):
    """bend specs of the solver shell of one parent with ring_fold_hinges set to `flag` (the module attribute)."""
    from jacobik.ik import angle_bends as AB
    from jacobik.topology.decompose import decompose_ligand
    from tools.jgen.inputs import load_mol
    from refit.nci_v2_pilot import build_builder_and_inputs

    mol = load_mol(code, shard, dataset)
    assert mol is not None, code
    builder, _, err = build_builder_and_inputs(mol)
    assert builder is not None, (code, err)
    shell, _, _ = builder._build_shell(fast_embed=True, debug=False)
    old = AB.RING_FOLD_HINGES
    AB.RING_FOLD_HINGES = flag
    try:
        out = []
        for lig in shell.ligands:
            decompose_ligand(shell, lig)
            out.append([sp for sp in AB.bend_specs(shell, lig)])
        return out
    finally:
        AB.RING_FOLD_HINGES = old


def _folds(specs_per_lig):
    return sum(1 for specs in specs_per_lig for sp in specs if len(sp) > 4 and sp[4] == ("hinge", 0))


def test_legacy_default_is_true():
    import jacobik.profile as P
    from jacobik.ik import angle_bends as AB

    assert P.FIELDS["ring_fold_hinges"][2] is True
    assert "ring_fold_hinges" not in P.PROFILES["u1"]
    assert AB.RING_FOLD_HINGES is True


@need_data
@pytest.mark.parametrize("code", FOLD_PARENTS)
def test_fold_parents_lose_their_fold_hinges(code):
    from tools.jgen.inputs import frame_shards

    shard, ds = frame_shards()[code]
    on = _specs(code, shard, ds, True)
    off = _specs(code, shard, ds, False)
    assert _folds(on) > 0, f"{code}: no fold hinge under legacy (the test parent is not a fold parent)"
    assert _folds(off) == 0


@need_data
@pytest.mark.parametrize("code", NO_FOLD_PARENTS)
def test_non_fold_parents_are_identical(code):
    from tools.jgen.inputs import frame_shards

    shard, ds = frame_shards()[code]
    assert _specs(code, shard, ds, True) == _specs(code, shard, ds, False)


@need_data
def test_cehpio_is_identical():
    code, shard = CEHPIO
    assert _specs(code, shard, "tmQM", True) == _specs(code, shard, "tmQM", False)
