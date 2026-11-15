import numpy as np
import pytest

from tools.runkit import cache, mace


class Mock:
    """energies(structs) -> eV: sum of first coordinate + atom count; counts calls and the largest call."""
    def __init__(self, oom_above=None):
        self.calls, self.max_call, self.oom_above = 0, 0, oom_above

    def energies(self, structs):
        self.calls += 1
        self.max_call = max(self.max_call, len(structs))
        if self.oom_above and len(structs) > self.oom_above:
            raise RuntimeError("CUDA out of memory. Tried to allocate")
        return np.array([float(np.asarray(x)[:, 0].sum()) + len(s) + 0.1 * q for (s, x, q, m) in structs])


def S(i, q=0):
    return (["O", "H", "H"], np.array([[i, 0, 0], [1, 0, 0], [0, 1, 0.0]]), q, 1)


def test_dedup_does_not_change_values():
    structs = [S(1), S(2), S(1), S(3), S(2), S(1), S(1, q=1)]
    ev = mace.Evaluator("cpu", backend=Mock())
    got = ev.energies_ev(structs)
    ref = Mock().energies(structs)
    assert np.array_equal(got, ref)                                  # bit-identical to the undeduplicated call
    assert ev.stats["evaluated"] == 4 and ev.stats["dedup_saved"] == 3
    assert ev._backend.calls == 1 and ev._backend.max_call == 4      # ONE call with unique structures only
    assert np.array_equal(ev.energies_kj(structs), ref * mace.EV_TO_KJ)


def test_oom_retry_halves_the_batch():
    ev = mace.Evaluator("cpu", backend=Mock(oom_above=3))
    structs = [S(i) for i in range(8)]
    got = ev.energies_ev(structs)
    assert np.array_equal(got, Mock().energies(structs))
    assert ev.stats["oom_retries"] >= 1 and ev._backend.max_call <= 8
    with pytest.raises(RuntimeError):                                # a non-OOM error is not swallowed
        class Bad(Mock):
            def energies(self, s): raise RuntimeError("shape mismatch")
        mace.Evaluator("cpu", backend=Bad()).energies_ev([S(1)])


def test_cache_hook(disk_tmp):
    c = cache.Store(disk_tmp / "c")
    ev = mace.Evaluator("cpu", backend=Mock(), cache=c)
    a = ev.energies_ev([S(1), S(2)])
    ev2 = mace.Evaluator("cpu", backend=Mock(), cache=c)
    b = ev2.energies_ev([S(2), S(1)])
    assert np.array_equal(b, a[::-1]) and ev2._backend.calls == 0 and ev2.stats["cache_hits"] == 2


def test_cpu_warning_once():
    ev = mace.Evaluator("cpu", backend=Mock())
    with pytest.warns(UserWarning, match="s/structure"):
        ev.energies_ev([S(1)])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ev.energies_ev([S(2)])
