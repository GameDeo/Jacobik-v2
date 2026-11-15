import os
import time
from pathlib import Path

import pytest

from tools.runkit import colab
from tools.runkit.resources import Refused

FAKE = Path(__file__).with_name("fake_cj.py")


@pytest.fixture
def st(disk_tmp, monkeypatch):
    s = disk_tmp / "state"
    (s / "files").mkdir(parents=True)
    monkeypatch.setenv("FAKE_CJ_STATE", str(s))
    return s


def mk(disk_tmp, **kw):
    return colab.Session("rkt", cj_path=FAKE, repo=disk_tmp, log_dir=disk_tmp / "log", poll_s=0.1, **kw)


def downs(st):
    p = st / "down.log"
    return p.read_text().split() if p.exists() else []


def test_ready_detection_and_down_on_exit(disk_tmp, st):
    with mk(disk_tmp) as vm:
        assert vm.ready_line.startswith("READY:")
        assert downs(st) == []
    assert downs(st) == ["rkt"]
    vm.down()
    assert downs(st) == ["rkt"]                       # idempotent


def test_exit_downs_after_exception(disk_tmp, st):
    with pytest.raises(ValueError):
        with mk(disk_tmp):
            raise ValueError("job crashed")
    assert downs(st) == ["rkt"]


def test_failed_up_downs_the_vm(disk_tmp, st):
    (st / "up_mode").write_text("fail")
    with pytest.raises(RuntimeError, match="READY"):
        mk(disk_tmp).up()
    assert downs(st) == ["rkt"]


def test_never_downs_a_vm_it_did_not_start(disk_tmp, st):
    vm = mk(disk_tmp)
    vm.down()                                          # never started
    assert downs(st) == []


def test_finish_count_ok_and_mismatch(disk_tmp, st):
    for i in range(3):
        (st / "files" / f"f{i}.npz").write_bytes(b"x")
    vm = mk(disk_tmp).up()
    r = vm.finish("/content/out/pool", str(disk_tmp / "got"), expected_files=3)
    assert r["ok"] and r["count"] == 3 and downs(st) == ["rkt"] and not vm.alerts
    vm2 = mk(disk_tmp).up()
    r = vm2.finish("/content/out/pool", str(disk_tmp / "got2"), expected_files=5)
    assert not r["ok"] and r["count"] == 3 and vm2.alerts and "5" in vm2.alerts[0]
    assert downs(st) == ["rkt", "rkt"]                 # short count: alert, then still down
    assert (disk_tmp / "log" / "ALERT.txt").exists()


def test_status_and_watchdog_fires(disk_tmp, st):
    (st / "jobs").write_text("solve running\n")
    (st / "gpu").write_text("97 %, 4000 MiB")
    (st / "files" / "a.npz").write_bytes(b"x")
    vm = mk(disk_tmp).up()
    s = vm.status()
    assert s["running_jobs"] == ["solve"] and s["gpu_util"] == 97
    clock = [1000.0]
    wd = colab.Watchdog(vm, idle_min=20, remote="/content/out/pool", local_dir=str(disk_tmp / "w"), expected=1,
                        clock=lambda: clock[0])
    assert not wd.tick()                               # busy
    (st / "jobs").write_text("solve done\n")
    (st / "gpu").write_text("0 %, 10 MiB")
    assert not wd.tick()                               # idle starts
    clock[0] += 19 * 60
    assert not wd.tick()
    clock[0] += 2 * 60
    assert wd.tick() and wd.fired                      # 21 idle minutes: fetch, count, down
    assert downs(st) == ["rkt"] and (disk_tmp / "w" / "pool" / "a.npz").exists()


def test_watchdog_busy_resets_idle(disk_tmp, st):
    vm = mk(disk_tmp).up()
    clock = [0.0]
    wd = colab.Watchdog(vm, idle_min=10, clock=lambda: clock[0])
    wd.tick()
    clock[0] += 9 * 60
    (st / "jobs").write_text("x running\n")
    wd.tick()
    assert wd.idle_since is None
    vm.down()


def test_stop_rule_refuses():
    with pytest.raises(Refused):
        colab.stop_rule(10.0, cap_units=8)
    assert colab.stop_rule(1.0, cap_units=8) != 0
    fake = colab.load_cj(FAKE)
    os.environ["FAKE_CJ_STATE"] = "/nonexistent"
    with pytest.raises(Refused):
        colab.stop_rule(1000.0, cj=fake)                # remaining cap = 50 - (100-90) = 40
    assert colab.stop_rule(30.0, cj=fake) == 40.0


def test_downed_only_after_vm_really_stopped(disk_tmp, st):
    """Regression (live run 2026-10-28): a second caller must not see downed=True while the first is still stopping the VM."""
    import threading
    vm = mk(disk_tmp).up()
    seen = []
    orig = vm._cli

    def slow(*a, **k):
        if a and a[0] == "down":
            time.sleep(1.0)
        return orig(*a, **k)
    vm._cli = slow
    t = threading.Thread(target=vm.down)
    t.start()
    time.sleep(0.3)
    seen.append(vm.downed)           # mid-down: must be False
    vm.down()                        # blocks until the first finishes
    seen.append(vm.downed)
    t.join()
    assert seen == [False, True] and downs(st) == ["rkt"]
