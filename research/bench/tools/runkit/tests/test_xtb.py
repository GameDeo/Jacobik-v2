import logging
import os
import shutil
import stat

import numpy as np
import pytest

from tools.runkit import cache, xtb

FAKE = r'''#!/bin/bash
if [ "$1" = "--version" ]; then echo "xtb version 6.7.1 (fake)"; exit 0; fi
echo "$@" >> "$FAKE_LOG"
case "$FAKE_MODE" in
  static) cp in.xyz xtbopt.xyz; echo "TOTAL ENERGY  -5.07 Eh"; echo "GEOMETRY OPTIMIZATION CONVERGED AFTER 3 ITERATIONS";;
  moved) awk 'NR>2{$2=$2+0.1} {print}' in.xyz > xtbopt.xyz; echo "TOTAL ENERGY  -5.07 Eh"; echo "GEOMETRY OPTIMIZATION CONVERGED AFTER 4 ITERATIONS";;
  noconv) awk 'NR>2{$2=$2+0.1} {print}' in.xyz > xtbopt.xyz; echo "TOTAL ENERGY  -5.07 Eh"; echo "FAILED TO CONVERGE GEOMETRY OPTIMIZATION";;
  hang) sleep 20;;
  nothing) echo "boom"; exit 3;;
esac
'''
SYM = ["O", "H", "H"]
X = np.array([[0.0, 0, 0], [1.1, 0, 0], [-0.3, 0.95, 0.2]])


@pytest.fixture
def fake(disk_tmp, monkeypatch):
    p = disk_tmp / "xtb"
    p.write_text(FAKE)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_LOG", str(disk_tmp / "calls.log"))
    def mode(m):
        monkeypatch.setenv("FAKE_MODE", m)
    return str(p), mode, disk_tmp


def test_require_rejects_binary_that_does_not_move_geometry(fake):
    path, mode, d = fake
    mode("static")
    with pytest.raises(SystemExit, match="self-check failed"):
        xtb.require(path)
    mode("nothing")
    with pytest.raises(SystemExit):
        xtb.require(path)


def test_require_unresolvable():
    with pytest.raises(SystemExit):
        xtb.resolve("/nonexistent/xtb")


def test_fallback_watch_catches_log_line():
    with xtb.FallbackWatch() as fw:
        logging.getLogger("jacobik.builder.engine").warning("GFN2 embed refinement failed for X: boom")
        logging.getLogger("other").warning("fine")
    assert fw.hit == 1
    with pytest.raises(RuntimeError, match="MMFF"):
        fw.check()
    with pytest.raises(RuntimeError):
        with xtb.FallbackWatch(raise_on_exit=True):
            logging.getLogger().warning("GFN2 embed refinement failed")
    logging.getLogger().warning("GFN2 embed refinement failed")   # handler removed on exit: no effect
    assert fw.hit == 1


def test_opt_never_raises_and_statuses(fake):
    path, mode, d = fake
    mode("moved")
    r = xtb.opt(SYM, X, 0, 1, d / "w", binary=path)
    assert r["status"] == "ok" and r["converged"] and r["cycles"] == 4 and r["E_h"] == -5.07
    assert np.allclose(r["X"][:, 0], X[:, 0] + 0.1) and r["xtb_version"] == "6.7.1"
    mode("noconv")
    assert xtb.opt(SYM, X, 0, 1, d / "w", binary=path)["status"] == "not_converged"
    mode("nothing")
    assert xtb.opt(SYM, X, 0, 1, d / "w", binary=path)["status"].startswith("no_output")
    r = xtb.opt(SYM, X, 0, 1, d / "w", binary="/no/such/xtb")
    assert r["status"].startswith("SystemExit") and r["X"] is None
    mode("hang")
    r = xtb.opt(SYM, X, 0, 1, d / "w", timeout=1, binary=path)
    assert r["status"].startswith("timeout") and r["wall_s"] < 10


def test_cycle_cap_charge_spin_in_command(fake):
    path, mode, d = fake
    mode("moved")
    xtb.opt(SYM, X, -1, 3, d / "w", cycles=7, binary=path)
    line = (d / "calls.log").read_text().strip().splitlines()[-1]
    assert "--cycles 7" in line and "--chrg -1" in line and "--uhf 2" in line and "--opt crude" in line and "--gfn 2" in line


def test_cache_hit_skips_the_call(fake):
    path, mode, d = fake
    mode("moved")
    c = cache.Store(d / "cache")
    r1 = xtb.opt(SYM, X, 0, 1, d / "w", cache=c, binary=path)
    n = len((d / "calls.log").read_text().splitlines())
    r2 = xtb.opt(SYM, X, 0, 1, d / "w", cache=c, binary=path)
    assert r2["cached"] and len((d / "calls.log").read_text().splitlines()) == n
    assert r2["E_h"] == r1["E_h"] and np.array_equal(r2["X"], r1["X"])
    xtb.opt(SYM, X, 0, 1, d / "w", cache=c, cycles=9, binary=path)      # other options: miss
    assert len((d / "calls.log").read_text().splitlines()) == n + 1


def test_opt_many_order_and_failure_ledger(fake):
    path, mode, d = fake
    mode("nothing")
    res = xtb.opt_many([(SYM, X, 0, 1)] * 3, 2, d / "w", failures=d / "failures.jsonl", codes=["A", "B", "C"], binary=path)
    assert [r["status"][:9] for r in res] == ["no_output"] * 3
    assert len((d / "failures.jsonl").read_text().splitlines()) == 3
    assert xtb.timing(res)["n_failed"] == 3


def test_grad_parsing():
    txt = """$grad
  cycle =      1    SCF energy =    -5.07000000000   |dE/dxyz| =  0.012
  0.0 0.0 0.0 o
  2.0 0.0 0.0 h
 -0.5 1.8 0.4 h
  1.0D-02  -2.0D-03  0.0
 -4.0D-03   1.0D-03  5.0D-04
 -6.0D-03   1.0D-03 -5.0D-04
$end
"""
    e, g = xtb.parse_gradient(txt, 3)
    assert e == -5.07 and g.shape == (3, 3) and g[0, 0] == 0.01 and abs(g.sum()) < 1e-12
    with pytest.raises(ValueError):
        xtb.parse_gradient(txt.replace("SCF energy", "x"), 3)


@pytest.mark.skipif(shutil.which("xtb") is None, reason="xtb not on PATH")
def test_real_xtb_water(disk_tmp):
    info = xtb.require()
    assert info["version"].startswith("6.")
    r = xtb.opt(SYM, X, 0, 1, disk_tmp / "w")
    assert r["status"] == "ok" and -5.2 < r["E_h"] < -4.9
    g = xtb.grad(SYM, X, 0, 1, disk_tmp / "w")
    assert g["status"] == "ok" and g["grad_h_bohr"].shape == (3, 3)
    assert abs(g["grad_h_bohr"].sum(0)).max() < 1e-6      # net force on a molecule is zero
