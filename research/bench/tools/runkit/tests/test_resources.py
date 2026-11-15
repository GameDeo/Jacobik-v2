import os
import subprocess
import sys
import time
import warnings

import pytest

from tools.runkit import resources as R


def test_env_contents():
    e = R.env(threads=1, cuda=False)
    assert e["OMP_NUM_THREADS"] == e["MKL_NUM_THREADS"] == e["OPENBLAS_NUM_THREADS"] == "1"
    assert e["CUDA_VISIBLE_DEVICES"] == ""
    assert e["PATH"].split(":")[0] == str(R.PIXI_BIN)
    assert e["JACOBIK_XTB"].endswith("/xtb")
    assert "src" in e["PYTHONPATH"]


def test_workroot_refuses_ram(disk_tmp):
    with pytest.raises(R.Refused):
        R.workroot("/tmp/runkit_x")
    with pytest.raises(R.Refused):
        R.workroot("/dev/shm/x")
    assert R.workroot("/tmp/runkit_ok", allow_ram=True).startswith("/tmp")
    assert R.workroot(disk_tmp).endswith("_xtbtmp")


def _budget(disk_tmp, agent="T", cpus=1, enforce_cpu=False):
    import json
    bf = disk_tmp / "budget.json"
    bf.write_text(json.dumps(dict(cpu={agent: cpus}, gpu_owner="Gemini", enforce=dict(cpu=enforce_cpu, gpu=True, kill=True))))
    return R.Budget(agent, budget_file=bf, run_dir=disk_tmp)


def test_quota_warns_then_refuses(disk_tmp):
    b = _budget(disk_tmp, cpus=1)
    p1 = R.spawn(["sleep", "30"], b, "a")
    try:
        with pytest.warns(UserWarning, match="quota"):
            p2 = R.spawn(["sleep", "30"], b, "b")
        b.cfg["enforce"]["cpu"] = True
        with pytest.raises(R.Refused):
            R.spawn(["sleep", "30"], b, "c")
        assert b.in_use() == 2
    finally:
        for p in (p1, p2):
            p.kill()


def test_kill_tree_refuses_unregistered_and_kills_tree(disk_tmp):
    b = _budget(disk_tmp, cpus=2)
    foreign = subprocess.Popen(["sleep", "30"])
    try:
        with pytest.raises(R.Refused):
            R.kill_tree(foreign.pid, b)
        assert foreign.poll() is None
        p = R.spawn(["bash", "-c", "sleep 60 & sleep 60; wait"], b, "tree")
        time.sleep(0.5)
        import psutil
        kids = [c.pid for c in psutil.Process(p.pid).children(recursive=True)]
        assert kids
        killed = p.kill(grace=2)
        assert p.pid in killed and not R._alive(p.pid)
        time.sleep(0.3)
        assert not any(R._alive(k) for k in kids)
        assert b.running() == []
    finally:
        foreign.kill()
        foreign.wait()


def test_gpu_lock_exclusive_and_owner(disk_tmp):
    import json
    bf = disk_tmp / "b.json"
    bf.write_text(json.dumps(dict(gpu_owner="Gemini", enforce=dict(gpu=True))))
    with pytest.raises(R.Refused):
        R.gpu_lock("A4", run_dir=disk_tmp, budget_file=bf).__enter__()
    code = ("import sys; from tools.runkit import resources as R\n"
            f"l = R.gpu_lock('Gemini', run_dir={str(disk_tmp)!r}, budget_file={str(bf)!r}, check_smi=False)\n"
            "try:\n    l.__enter__(); print('GOT')\nexcept R.Refused: print('REFUSED')\n")
    with R.gpu_lock("Gemini", run_dir=disk_tmp, budget_file=bf, check_smi=False):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=R.env())
        assert "REFUSED" in r.stdout, r.stdout + r.stderr
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=R.env())
    assert "GOT" in r.stdout, r.stdout + r.stderr


def test_spawn_gpu_needs_owner_and_lock(disk_tmp):
    b = _budget(disk_tmp, agent="T")
    with pytest.raises(R.Refused):
        R.spawn(["true"], b, "g", gpu=True)


def test_manifest_has_fingerprint_fields():
    m = R.manifest()
    for k in ("host", "cores", "git_commit", "xtb_path", "fingerprint"):
        assert k in m
