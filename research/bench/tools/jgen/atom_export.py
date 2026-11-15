"""Element labels for coordinates exported in builder/FK atom order."""
def symbols_for_conformer(conf):
    symbols = [a.symbol for a in conf.complex.atoms]
    assert symbols == [a.symbol for a in conf.complex.atoms]
    return symbols
