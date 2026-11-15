"""Dev runner, control cache, MPS verifier, device resolution, bench specs, and notify tests."""

import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.run import control_cache as CC
from tools.run import lane
from tools.run import notify
from tools.run import orchestrator as O
from tools.run import resources as R
from tools.run.orchestrator import Job, Scheduler, estimate, lane_env, pack_outputs

REPO = Path(__file__).resolve().parents[2]


def _cc_plan(tmp_path, db, **kw):
    return dict(
        repo=str(REPO),
        device="cpu",
        backend="bench",
        seeds=[21, 22],
        env={"JACOBIK_DB": str(db)},
        arms=[dict(arm="C", flags={}, seed_budget="three_w16"), dict(arm="K", flags={"start_kabsch": True})],
        cpu_identity=True,
        deterministic=True,
        pack_arms=False,
        out=str(tmp_path / "out"),
        **kw,
    )


def _run_plan(tmp_path, split=False, costs=(30.0, None, 300.0)):
    return dict(
        out=str(tmp_path / "run"),
        arms=[dict(arm="C"), dict(arm="K", out=str(tmp_path / "k_elsewhere"))],
        seeds=[11, 12, 13],
        split_seeds=split,
        parents=[dict(code=c, cost=v) for c, v in zip(("AAA", "BBB", "CCC"), costs)],
    )


def _v2_plan(tmp_path, name, claim=None, n=6):
    p = dict(
        out=str(tmp_path / name),
        arms=[dict(arm="C")],
        seeds=[11],
        parents=[dict(code=f"P{i}", cost=100.0 - i) for i in range(n)],
    )
    if claim:
        p["claim_dir"] = str(claim)
    return p


class _Fake:
    def __init__(self, job, state="starting", code=None):
        self.job, self.state, self.code, self.ready = job, state, code, None
        self.send = lambda m: True

    @property
    def live(self):
        return self.state not in ("stopped", "dead")


class _Isomer(lane.IsomerBackend):
    def __init__(self, tmp, status):
        self.stage, self.lane, self.device = {}, 0, "cpu"
        self.lane_dir, self.out = tmp / "lane", tmp / "out"
        self.arms, self.seeds = [dict(arm="A")], [11]
        self.arm_dir = {"A": self.out / "A"}
        self._status = status

        class _E:
            _gfn2_refine = None

        self.E = _E()

        class _M:
            @staticmethod
            def main(argv, _self=self):
                d = _self.lane_dir / "A" / "records"
                d.mkdir(parents=True, exist_ok=True)
                (d / "UYOVUA_11.json").write_text(
                    json.dumps(dict(code="UYOVUA", seed=11, status=_self._status, reason="GFN2 embedding fallback"))
                )

        self.mod = _M()


def test_control_cache_keys_store_restore_and_canaries(tmp_path, monkeypatch):
    from tools.run.__main__ import _control_cache_jobs

    db = tmp_path / "db.json"
    db.write_text("{}")
    meta = tmp_path / "db.meta.json"
    meta.write_text('{"q5_model": "x"}')

    monkeypatch.setattr(CC, "_git", lambda repo, *a: "" if a[0] == "status" else "tree-" + a[-1])
    p = _cc_plan(tmp_path, db)
    k1, inputs0 = CC.control_key(p, p["arms"][0])
    assert k1 and k1 == CC.control_key(p, p["arms"][0])[0]
    assert "target" not in inputs0
    assert CC.control_key(p, p["arms"][1])[0] != k1
    assert CC.control_key(dict(p, seeds=[21]), p["arms"][0])[0] != k1
    meta.write_text('{"q5_model": "y"}')
    assert CC.control_key(p, p["arms"][0])[0] != k1

    ov = tmp_path / "overlay.json"
    ov.write_text('{"v": 1}')
    k_ov1, inputs1 = CC.control_key(dict(p, target_overlay=str(ov)), p["arms"][0])
    assert k_ov1 and k_ov1 != k1 and inputs1["target"]["overlay_sha256"]
    ov.write_text('{"v": 2}')
    k_ov2 = CC.control_key(dict(p, target_overlay=str(ov)), p["arms"][0])[0]
    assert k_ov2 != k_ov1
    assert CC.control_key(dict(p, target_overlay=str(ov), gate_eligibility_sha256="a" * 64), p["arms"][0])[0] != k_ov2
    bad_key, bad_info = CC.control_key(dict(p, target_overlay=str(tmp_path / "missing.json")), p["arms"][0])
    assert bad_key is None and "overlay" in bad_info["reason"]

    monkeypatch.setattr(CC, "_git", lambda repo, *a: " M src/x.py" if a[0] == "status" else "t")
    dirty_k, dirty_info = CC.control_key(p, p["arms"][0])
    assert dirty_k is None and "uncommitted" in dirty_info["reason"]

    monkeypatch.setattr(CC, "_git", lambda repo, *a: "" if a[0] == "status" else "t")
    monkeypatch.setenv("JACOBIK_CONTROL_CACHE", str(tmp_path / "cache"))
    cc = CC.ControlCache(p, "C")
    src = tmp_path / "run1" / "C"
    (src / "records").mkdir(parents=True)
    (src / "xyz" / "AAA").mkdir(parents=True)
    (src / "xyz" / "AAA" / "AAA_pool.json").write_text(json.dumps({"xyz": str(src / "xyz/AAA/a.xyz")}))
    for code, status in (("AAA", "ok"), ("BBB", "failed")):
        for s in (21, 22):
            (src / "records" / f"{code}_{s}.json").write_text(
                json.dumps(dict(code=code, seed=s, status=status, pool_manifest=str(src / f"xyz/{code}/{code}_pool.json")))
            )
    assert cc.store(["AAA", "BBB"], src) == 1
    assert cc.cached_codes(["AAA", "BBB"]) == ["AAA"]
    dst = tmp_path / "run2" / "C"
    assert cc.restore(["AAA"], dst) == 2
    assert json.loads((dst / "records" / "AAA_21.json").read_text())["pool_manifest"].startswith(str(dst))
    assert str(dst) in (dst / "xyz" / "AAA" / "AAA_pool.json").read_text()

    codes = [f"C{i:03d}" for i in range(40)]
    assert CC.ControlCache.canary(codes, 5) == CC.ControlCache.canary(list(reversed(codes)), 5)
    assert len(CC.ControlCache.canary(codes, 5)) == 5

    for idx, (new, ref, exp) in enumerate([("failed", "failed", 0), ("ok", "failed", 0), ("failed", "ok", 0), ("ok", "ok", 1)]):
        dirs = [tmp_path / f"can_{idx}_new", tmp_path / f"can_{idx}_ref"]
        for d, st in zip(dirs, (new, ref)):
            (d / "records").mkdir(parents=True)
            (d / "records" / "AAA_21.json").write_text(json.dumps({"status": st}))
        assert CC.compare_canary(*dirs, ["AAA"], [21])["identical"] == exp

    class FakeCache:
        name, key, usable, inputs, dir = "C", "key", True, {}, tmp_path

        def __init__(self, plan, name):
            pass

        def cached_codes(self, codes):
            return []

        def canary(self, codes, n):
            return list(codes)

    monkeypatch.setattr(CC, "ControlCache", FakeCache)
    out = tmp_path / "res_out"
    rec = out / "_canary/C/records/AAA_21.json"
    rec.parent.mkdir(parents=True)
    rec.write_text('{"status":"ok"}')
    args = SimpleNamespace(control_repo="", control_cache="C", control_cache_refresh=False, canary=1, no_resume=False)
    _, _, jobs = _control_cache_jobs(args, dict(out=str(out), arms=[dict(arm="C")], repo=str(tmp_path), seeds=[21], parents=[dict(code="AAA", cost=1)], name="test"))
    canary = next(j for j in jobs if j.name.endswith(":canary"))
    assert canary.todo == ["AAA"] and canary.arm_dir["C"] != rec.parent.parent


def test_control_cache_packed_canary(tmp_path, monkeypatch):
    import numpy as np
    from tools.run import compare

    db = tmp_path / "db.json"
    db.write_text("{}")
    (tmp_path / "db.meta.json").write_text('{"q5_model": "x"}')
    root = tmp_path / "run"
    root.mkdir()
    arm = root / "C"
    orig_manifest = str(arm / "xyz/AAA/pool.json")
    for number, status, xyz in ((2, "failed", "1"), (10, "ok", "2")):
        files = {
            "C/records/AAA_21.json": json.dumps(dict(status=status, scores=[1.0], row_keys=["a"], pool_manifest=orig_manifest)),
            "C/records/AAA_22.json": json.dumps(dict(status=status, scores=[1.0], row_keys=["a"], pool_manifest=orig_manifest)),
            "C/xyz/AAA/pool.json": json.dumps({"rows": [{"file": "top.xyz"}]}),
            "C/xyz/AAA/top.xyz": xyz,
            "C/xyz/OTHER/unrelated.xyz": "must not materialize",
        }
        with tarfile.open(root / f"run_pack{number}.tar", "w") as tar:
            for name, content in files.items():
                data = content.encode()
                m = tarfile.TarInfo(name)
                m.size = len(data)
                tar.addfile(m, io.BytesIO(data))

    new = root / "canary"
    (new / "records").mkdir(parents=True)
    (new / "top.xyz").write_text("2")
    for seed in (21, 22):
        (new / "records" / f"AAA_{seed}.json").write_text(json.dumps(dict(status="ok", scores=[1.0], row_keys=["a"], pool_manifest=str(new / "pool.json"))))
    (new / "pool.json").write_text("{}")

    def xyz(record):
        manifest = Path(record["pool_manifest"])
        assert manifest.is_file()
        return np.array([[float((manifest.parent / "top.xyz").read_text())]])

    monkeypatch.setattr(compare, "_top1_xyz", xyz)
    assert CC.compare_canary_packed(new, arm, ["AAA"], [21, 22], root)["identical"] == 2

    monkeypatch.setattr(CC, "_git", lambda repo, *a: "" if a[0] == "status" else "tree")
    monkeypatch.setenv("JACOBIK_CONTROL_CACHE", str(tmp_path / "cache"))
    cc = CC.ControlCache(_cc_plan(tmp_path, db), "C")
    assert CC.store_packed(cc, ["AAA"], arm, root) == 1 and cc.cached_codes(["AAA"]) == ["AAA"]
    assert json.loads((cc.dir / "records/AAA_21.json").read_text())["pool_manifest"] == str(cc.dir / "xyz/AAA/pool.json")
    assert (cc.dir / "xyz/AAA/top.xyz").read_text() == "2" and not list(root.glob(".control-stage-*"))

    (arm / "xyz/AAA").mkdir(parents=True)
    (arm / "xyz/AAA/pool.json").write_text(json.dumps({"rows": [{"file": "missing.xyz"}]}))
    monkeypatch.setattr(compare, "_top1_xyz", lambda record: None)
    bad = CC.compare_canary_packed(new, arm, ["AAA"], [21, 22], root)
    assert bad["identical"] == 0 and "frame unavailable" in bad["rows"][0]["why"]


def test_tools_run_jobs_budget_pack_retry_and_mps(tmp_path, monkeypatch):
    from tools.run import mps

    j = Job(_run_plan(tmp_path))
    assert j.todo[0] == "CCC"
    assert estimate([j], lanes=8) == {**estimate([j], lanes=8), "wall_s": 300, "unknown_cost": 1}
    assert j.budget_for("AAA", 180, 900) == 180 and j.budget_for("CCC", 180, 900) == 900 and j.budget_for("BBB", 180, 900) == 900
    j.write_status("CCC", "timeout", "budget")
    assert json.loads((tmp_path / "k_elsewhere" / "records" / "CCC_13.json").read_text())["status"] == "timeout"

    p_split = _run_plan(tmp_path / "sp", split=True)
    js = Job(p_split)
    assert len(js.todo) == 9 and js.todo[0].startswith("CCC@") and js.item("AAA@12") == ("AAA", [12])
    for arm_dir in (tmp_path / "sp" / "run" / "C", tmp_path / "sp" / "k_elsewhere"):
        (arm_dir / "records").mkdir(parents=True, exist_ok=True)
        (arm_dir / "records" / "AAA_12.json").write_text("{}")
    js2 = Job(p_split)
    assert "AAA@12" not in js2.todo and "AAA@11" in js2.todo and js2.skipped == 1

    p_pack = _run_plan(tmp_path / "pk")
    p_pack["arms"] = [dict(arm="C")]
    out = tmp_path / "pk" / "run"
    for s in (11, 12, 13):
        (out / "C" / "records").mkdir(parents=True, exist_ok=True)
        (out / "C" / "records" / f"AAA_{s}.json").write_text("{}")
    (out / "run_summary.json").write_text("{}")
    info = pack_outputs(out)
    assert info["files"] >= 3 and not (out / "C").exists() and (out / "run_summary.json").exists()
    with tarfile.open(info["tar"]) as t:
        assert "C/records/AAA_11.json" in t.getnames()
    assert "AAA" not in Job(p_pack).todo

    jr = Job(p_pack)
    jr.write_status("AAA", "timeout", "deadline", 12.5)
    assert "AAA" not in Job(p_pack).todo and Job(p_pack).skipped_timeouts == ["AAA"]
    pack_outputs(out)
    assert "AAA" not in Job(p_pack).todo and "AAA" in Job(dict(p_pack, retry_timeouts=True)).todo
    jr.write_status("BBB", "timeout", "deadline", 12.5)
    assert "BBB" in Job(dict(p_pack, retry_timeouts=True)).todo and (out / "C/records/BBB_11.json").exists()
    Job(dict(p_pack, retry_timeouts=True)).write_status("BBB", "timeout", "second", 42.0)
    updated = json.loads((out / "C/records/BBB_11.json").read_text())
    assert updated["reason"] == "second" and updated["elapsed_s"] == 42.0

    root_sub = tmp_path / "subrun"
    root_sub.mkdir()
    for number, status in ((2, "timeout"), (10, "ok")):
        with tarfile.open(root_sub / f"run_pack{number}.tar", "w") as tar:
            for seed in (11, 12, 13):
                data = json.dumps({"status": status, "run_timing": {"parent_wall_s": 400}}).encode()
                ti = tarfile.TarInfo(f"C/records/AAA_{seed}.json")
                ti.size = len(data)
                tar.addfile(ti, io.BytesIO(data))
    plan_sub = _run_plan(tmp_path / "subunused")
    plan_sub.update(out=str(root_sub / "_jobs/control"), pack_root=str(root_sub), arms=[dict(arm="C", out=str(root_sub / "C"))], retry_timeouts=True)
    j_sub = Job(plan_sub)
    assert "AAA" not in j_sub.todo and j_sub.cost["AAA"] == 400 and "AAA" not in j_sub.skipped_timeouts

    assert R.cpu_budget() >= 1 and R.mem_available() > 0 and R.tree_rss(os.getpid()) > 0
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("TOOLS_RUN_MPS_CLIENT", "1")
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/private-g3")
    env = lane_env(dict(out=str(tmp_path), device="cuda"))
    assert "CUDA_VISIBLE_DEVICES" not in env and env["CUDA_MPS_PIPE_DIRECTORY"] == "/tmp/private-g3"
    assert lane_env(dict(out=str(tmp_path), device="cpu"))["CUDA_VISIBLE_DEVICES"] == ""

    calls = []
    monkeypatch.setattr(mps.shutil, "which", lambda _: "/bin/mps")
    monkeypatch.setattr(mps.tempfile, "mkdtemp", lambda **_: str(tmp_path / f"m{len(calls)}"))
    monkeypatch.setattr(mps.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw["env"].copy())) or SimpleNamespace(returncode=0, stderr=""))
    monkeypatch.setattr(mps, "_own_mps_pids", lambda pipe: [])
    daemon = mps.MPS().start()
    assert calls[0][1]["CUDA_VISIBLE_DEVICES"] == "3" and calls[0][1]["CUDA_MPS_PIPE_DIRECTORY"] == daemon.pipe
    assert daemon.stop()["stopped"] and calls[1][1]["CUDA_MPS_PIPE_DIRECTORY"] == daemon.pipe


def test_scheduler_claims_init_failures_lane_and_mps_verifier(tmp_path, monkeypatch):
    def _lane(job, sent):
        return SimpleNamespace(job=job, retiring=False, state="idle", code=None, t_task=None, budget=None, timed_out=False, send=lambda m: sent.append(m["item"]) or True, live=True)

    claim = tmp_path / "claims"
    a, b = Job(_v2_plan(tmp_path, "a", claim)), Job(_v2_plan(tmp_path, "b", claim))
    sa, sb = Scheduler(tmp_path / "sa", device="cpu"), Scheduler(tmp_path / "sb", device="cpu")
    got_a, got_b = [], []
    la, lb = _lane(a, got_a), _lane(b, got_b)
    for i in range(3):
        sa.dispatch(la); la.state = "idle"
        if i % 2 == 0:
            sb.dispatch(lb); lb.state = "idle"
    assert sorted(got_a + got_b) == sorted(set(got_a + got_b)) and len(got_a) > len(got_b)
    a.prune(); b.prune()
    assert all(c in a.mine or not (claim / c).exists() for c in a.todo)

    c2 = tmp_path / "claims2"
    a2, b2 = Job(_v2_plan(tmp_path, "a2", c2, n=3)), Job(_v2_plan(tmp_path, "b2", c2, n=3))
    for _ in range(3):
        Scheduler(tmp_path / "s2", device="cpu").dispatch(_lane(a2, []))
    b2.prune()
    assert b2.todo == []

    c3 = tmp_path / "claims3"
    j3 = Job(_v2_plan(tmp_path, "a3", c3, n=2))
    Scheduler(tmp_path / "s3", device="cpu").dispatch(SimpleNamespace(job=j3, retiring=False, send=lambda m: False))
    assert "P0" in j3.todo and not (c3 / "P0").exists() and "P0" not in j3.mine

    j_init = Job(_v2_plan(tmp_path, "init1"))
    s_init = Scheduler(tmp_path / "si", device="cpu")
    s_init.jobs = [j_init]
    s_init.lanes[0] = _Fake(j_init, "busy", "P0")
    for i in range(1, 9):
        s_init.lanes[i] = _Fake(j_init)
        s_init.handle(i, {"ready": False, "error": "CUDA error: out of memory"})
    assert len(j_init.todo) == 6 and not j_init.not_run and j_init.init_block_until > time.time() and s_init.job_for_new_lane() is None

    j_die = Job(_v2_plan(tmp_path, "die"))
    s_die = Scheduler(tmp_path / "sd", device="cpu")
    s_die.jobs = [j_die]
    for i in range(O.INIT_FAIL_LIMIT):
        s_die.lanes[i] = _Fake(j_die)
        s_die.handle(i, {"ready": False, "error": "oom"})
    assert j_die.todo == [] and len(j_die.not_run) == 6 and not list((tmp_path / "die").rglob("records/*.json"))

    j_ok = Job(_v2_plan(tmp_path, "ok"))
    s_ok = Scheduler(tmp_path / "sok", device="cpu")
    j_ok.init_failures = 4
    s_ok.lanes[0] = _Fake(j_ok)
    s_ok.handle(0, {"ready": True, "startup_s": 1.0, "rss": 10**9})
    assert j_ok.init_failures == 0

    j_adm = Job(_run_plan(tmp_path / "adm", costs=(10, 10, 300)))
    s_adm = Scheduler(tmp_path / "sa2", device="cpu", cpu_cap=8, mem_headroom_gb=0)
    lane0 = SimpleNamespace(live=True, done=0, state="busy", job=j_adm, code="CCC", pid=1)
    s_adm.lanes[0] = lane0
    s_adm.jobs = [j_adm]
    monkeypatch.setattr(R, "cpu_budget", lambda *_: 8)
    monkeypatch.setattr(R, "foreign_cpu_load", lambda *_: 0)
    monkeypatch.setattr(R, "mem_available", lambda: 16 * R.GB)
    assert s_adm.can_add()[0]
    lane0.code = "AAA"
    assert not s_adm.can_add()[0]
    j_adm.prior_timeouts.add("AAA")
    assert s_adm.can_add()[0]

    j_to = Job(_run_plan(tmp_path / "to"))
    j_to.write_status("CCC", "timeout", "previous")
    j_to = Job(dict(_run_plan(tmp_path / "to"), retry_timeouts=True))
    Scheduler(tmp_path / "sto", device="cpu").dispatch(SimpleNamespace(job=j_to, retiring=False, send=lambda _: False))
    assert json.loads((tmp_path / "to" / "run/C/records/CCC_11.json").read_text())["status"] == "timeout" and "CCC" in j_to.todo

    r_fail = dict(error=None, records=2, failed_records=[dict(arm="C", seed=11, status="failed", reason="GFN2 embedding fallback for UYOVUA")])
    assert "C/s11 failed" in lane._mark_failed_records(r_fail)["error"]
    assert lane._mark_failed_records(dict(error=None, records=3, failed_records=[]))["error"] is None
    assert lane._mark_failed_records(dict(error="RuntimeError: boom", failed_records=[dict(arm="C", seed=11, status="failed", reason="x")]))["error"] == "RuntimeError: boom"

    res = _Isomer(tmp_path / "iso1", "failed").run_parent("UYOVUA")
    assert res["error"] is None and len(res["failed_records"]) == 1 and lane._mark_failed_records(res)["error"]
    assert _Isomer(tmp_path / "iso2", "ok").run_parent("UYOVUA")["failed_records"] == []

    mps_dir = tmp_path / "mps_verify"
    mps_dir.mkdir()
    for gpu in range(4):
        gout = mps_dir / f"g{gpu}"
        gout.mkdir()
        (gout / "mps.json").write_text(json.dumps({"stopped": True, "left": []}))
        (gout / "run_events.jsonl").write_text("\n".join(json.dumps({"kind": "parent_done", "ok": True}) for _ in range(2)))
        (gout / "telemetry.jsonl").write_text(json.dumps({"gpu_power_w": 90}))
        with tarfile.open(gout / "run_pack0.tar", "w") as tar:
            for idx in range(2):
                data = json.dumps({"status": "failed" if gpu == 3 and idx == 1 else "ok"}).encode()
                ti = tarfile.TarInfo(f"C/records/P{idx}_11.json")
                ti.size = len(data)
                tar.addfile(ti, io.BytesIO(data))
    verifier = REPO / "dev/tools/check/verify_mps_node.py"
    v_res = subprocess.run([sys.executable, str(verifier), str(mps_dir)], capture_output=True, text=True)
    assert v_res.returncode != 0 and "g3: expected two ok control records" in v_res.stderr

    defs = {"A": {"stereo_candidates": False}, "AS": {"stereo_candidates": True}}
    lane._apply_isomer_overrides(defs, [{"arm": "A", "flags": {"example": True}}, {"arm": "AS", "flags": {}}], lambda _: False)
    assert defs["A"] == {"stereo_candidates": False, "example": True}
    assert defs["AS"] == {"stereo_candidates": True, "example": False}


def test_devices_bench_specs_case_runner_and_notify(monkeypatch, tmp_path):
    import torch
    from jacobik.core.energy import resolve_device
    from tests.regression.run_panel import CaseRunner
    from tools.jdev import bench

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))

    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") == ("cuda" if torch.cuda.is_available() else "cpu")
    with pytest.raises(ValueError):
        resolve_device("gpu0")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="cannot use CUDA"):
            resolve_device("cuda")

    kabsch_src = (REPO / "src" / "jacobik" / "ik" / "kabsch.py").read_text()
    assert "torch.randn(B, 3, dtype=P.dtype).to(device)" in kabsch_src and "torch.rand(B, dtype=P.dtype).to(device)" in kabsch_src

    cfg = {"cur": {"angle_bends": True, "_env": {"JACOBIK_PROFILE_OVERRIDES": '{"donor_bends": true}'}}}
    cpu = bench.matrix_specs(cfg, [11, 12], tmp_path / "db.json", tmp_path, "n48")
    gpu = bench.matrix_specs(cfg, [11], tmp_path / "db.json", tmp_path, "n48", solver_device="cuda")
    run, env = cpu[0]
    assert len(cpu) == 2 and env == (("JACOBIK_PROFILE_OVERRIDES", '{"donor_bends": true}'),)
    assert run["ctx"]["solve_kw"] == {"angle_bends": True, "seed": 11} and run["ctx"]["tmqm_split"] == "N"
    assert len(run["cases"]) == 48 and run["out"].endswith("n48__cur__s11__mace-energy-only-float32.json")
    assert gpu[0][0]["out"].endswith("__solver-cuda-replay.json")
    assert tuple(Path(gpu[0][0]["out"]).stem.split("__")[:3]) == ("n48", "cur", "s11")

    r = CaseRunner("loky", 4, "cpu", None, 60)
    assert r.driver_mace is False and r.ev is None and r.jobs == 4
    assert CaseRunner("inline", 1, "cpu", None, 60).driver_mace is False

    nroot = tmp_path / "notify"
    for rel, parents, done in (("g0", ["A", "B"], ["A"]), ("stage1/g0", ["C", "D", "E"], ["C", "D"])):
        d = nroot / rel
        d.mkdir(parents=True)
        (d / "run_plan.json").write_text(json.dumps({"parents": parents}))
        (d / "run_events.jsonl").write_text("".join(json.dumps({"kind": "parent_done", "code": c, "ok": True}) + "\n" for c in done))
    s = notify.stats(nroot)
    assert "`gate` 1/2" in s and "`stage1` 2/3" in s

    monkeypatch.delenv("JK_DISCORD_WEBHOOK", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))
    assert notify.webhook() is None and notify.send("x") is False and notify.main(["msg", "hello"]) == 0
