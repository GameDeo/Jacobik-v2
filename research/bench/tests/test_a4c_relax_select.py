"""Step-2 relax-set rule (A4c): same-class rows by core RMSD, one representative per other class, logged cap."""
import numpy as np

from tools.jgen.a4c.classes import Reference
from tools.jgen.a4c.relax_select import select
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from test_a4c_classes import mono, rot  # noqa: E402


def test_select_prefers_same_class_by_core_and_one_row_per_other_class():
    trans = mono({"+z": "Cl", "-z": "Cl", "+x": "N", "-x": "N", "+y": "N", "-y": "N"})
    cis = mono({"+z": "Cl", "+x": "Cl", "-z": "N", "-x": "N", "+y": "N", "-y": "N"})
    fac_like = mono({"+z": "Cl", "+x": "Cl", "-z": "N", "-x": "N", "+y": "N", "-y": "N"})
    ref = Reference(*trans)
    near = trans[1].copy(); near[1:] *= 1.05
    lost = trans[1].copy(); lost[1:] *= 1.8                     # donors pushed out of contact: another class
    pool = np.array([cis[1], rot(cis[1], 1), near, lost, rot(trans[1], 2), fac_like[1]])
    scores = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])          # row 0 = top-1 (a cis row), row 5 = floor
    out = select(ref, pool, scores, top=0, floor=5)
    assert out["rows"][:2] == [0, 5]
    assert out["same_class"] == [4, 2]                                     # nearest by core RMSD first
    assert out["classes"] == [[0, 1, 5], [3]]                              # cis class (3 rows) and the lost-contact row
    assert out["rows"] == [0, 5, 4, 2, 3] and out["dropped"] == []          # same-class rows before other classes


def test_select_logs_the_cap():
    trans = mono({"+z": "Cl", "-z": "Cl", "+x": "N", "-x": "N", "+y": "N", "-y": "N"})
    ref = Reference(*trans)
    pool = np.array([rot(trans[1], k) for k in range(6)])
    out = select(ref, pool, np.arange(6.0), top=0, floor=1, cap=2)
    assert out["rows"] == [0, 1] and out["dropped"]                        # the cap bites and is logged
