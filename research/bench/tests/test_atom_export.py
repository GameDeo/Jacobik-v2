"""Regression: FK coordinates must never inherit frame-order element labels."""
from types import SimpleNamespace
from tools.jgen.atom_export import symbols_for_conformer


def test_export_uses_conformer_order_even_when_frame_order_differs():
    frame = SimpleNamespace(atoms=[SimpleNamespace(symbol=s) for s in ['Fe','H','C']])
    conf = SimpleNamespace(complex=SimpleNamespace(atoms=[SimpleNamespace(symbol=s) for s in ['Fe','C','H']]))
    exported = symbols_for_conformer(conf)
    assert exported == [a.symbol for a in conf.complex.atoms]
    assert exported != [a.symbol for a in frame.atoms]
