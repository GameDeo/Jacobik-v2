import os
import time
import zipfile
import zlib

import numpy as np
import pytest

from tools.runkit import pipeline as pl


def touch(d, name, age=100, data=b"x"):
    p = d / name
    p.write_bytes(data)
    t = time.time() - age
    os.utime(p, (t, t))
    return p


def test_watcher_needs_three_seeds_and_settle(disk_tmp):
    for s in (11, 12, 13):
        touch(disk_tmp, f"AAAAAA_s{s}.npz")
    touch(disk_tmp, "BBBBBB_s11.npz")
    touch(disk_tmp, "BBBBBB_s12.npz")
    for s in (11, 12, 13):
        touch(disk_tmp, f"CCCCCC_s{s}.npz", age=0)           # newest file still being written
    got = list(pl.Watcher(disk_tmp, settle=8, watch=False))
    assert [k for k, _ in got] == ["AAAAAA"] and len(got[0][1]) == 3
    # once the producer is done, incomplete groups are released too
    (disk_tmp / "DONE").write_text("")
    got = list(pl.Watcher(disk_tmp, settle=8, watch=False, done_file=disk_tmp / "DONE"))
    assert sorted(k for k, _ in got) == ["AAAAAA", "BBBBBB"]


def test_shards_are_disjoint_and_cover():
    codes = [f"{a}{b}CDEF" for a in "ABCDEFGH" for b in "ABCDEFGH"]
    sets = [{c for c in codes if pl.crc_shard(c, f"{k}/3")} for k in range(3)]
    assert sum(len(s) for s in sets) == len(codes) and not (sets[0] & sets[1]) and not (sets[1] & sets[2]) and not (sets[0] & sets[2])
    assert all(zlib.crc32(c.encode()) % 3 == 1 for c in sets[1])


def test_skip_file_reread_done_file_and_resume(disk_tmp):
    for c in ("AAAAAA", "BBBBBB", "CCCCCC"):
        for s in (11, 12, 13):
            touch(disk_tmp, f"{c}_s{s}.npz")
    skip, done, state = disk_tmp / "skip.txt", disk_tmp / "done", disk_tmp / "state.txt"
    skip.write_text("BBBBBB\n")
    sleeps = []

    def sleeper(s):                        # first poll: producer finishes
        sleeps.append(s)
        done.write_text("")
    w = pl.Watcher(disk_tmp, settle=0, skip_file=skip, done_file=None, state_file=state, poll=1, sleep=sleeper)
    w.done_file = str(done)
    seen = []
    for k, files in w:
        seen.append(k)
        if k == "AAAAAA":
            w.mark_done(k)                 # CCCCCC "crashes" before being marked
    assert seen == ["AAAAAA", "CCCCCC"] and sleeps == [1]
    # resume after a crash: AAAAAA is in the state file, CCCCCC is redone; BBBBBB un-skipped is picked up
    skip.write_text("")
    again = [k for k, _ in pl.Watcher(disk_tmp, settle=0, state_file=state, watch=False)]
    assert again == ["BBBBBB", "CCCCCC"]


def test_split_by_cost_balance():
    rng = np.random.default_rng(0)
    items = list(range(40))
    cost = {i: float(rng.integers(20, 120)) ** 2 for i in items}
    ms = [pl.Machine("local", 4, 1.0), pl.Machine("vm", 12, 0.5)]
    sp = pl.split_by_cost(items, cost.get, ms)
    assert sorted(sp["local"] + sp["vm"]) == items
    t = pl.split_makespan(sp, cost.get, ms)
    ideal = sum(cost.values()) / sum(m.capacity for m in ms)
    assert max(t.values()) < 1.15 * ideal
    assert sp["local"] == sorted(sp["local"])


def test_write_skip_file(disk_tmp):
    pl.write_skip_file({"local": ["A", "B"], "vm": ["C"]}, "local", disk_tmp / "s.txt")
    assert (disk_tmp / "s.txt").read_text().split() == ["C"]


def npz(path, ok=True):
    np.savez_compressed(path, a=np.arange(1000))
    if not ok:
        b = path.read_bytes()
        path.write_bytes(b[: len(b) // 2])


class FakeVM:
    def __init__(self, src):
        self.src = src

    def get(self, remote, local):
        import shutil
        shutil.copytree(self.src, os.path.join(local, os.path.basename(remote)), dirs_exist_ok=True)


def test_remote_feed_rejects_truncated_zip(disk_tmp):
    src = disk_tmp / "vm" / "pool"
    src.mkdir(parents=True)
    npz(src / "AAAAAA_s11.npz")
    npz(src / "AAAAAA_s12.npz", ok=False)
    npz(src / "ZZZZZZ_s11.npz")                                  # not our share
    f = pl.RemoteFeed(FakeVM(src), "/content/out/pool", disk_tmp / "local", ["AAAAAA"], stage=disk_tmp / "stage")
    assert f.sync_once() == 1 and f.rejected == 1
    assert sorted(p.name for p in (disk_tmp / "local").iterdir()) == ["AAAAAA_s11.npz"]
    npz(src / "AAAAAA_s12.npz")                                  # now complete: picked up on the next sync
    assert f.sync_once() == 1 and f.copied == 2


def test_run_groups_continues_after_failure(disk_tmp):
    def fn(k, p):
        if k == "B":
            raise ValueError("boom")
        return p * 2
    r = pl.run_groups(fn, [("A", 1), ("B", 2), ("C", 3)], workers=1, ledger=disk_tmp / "l.jsonl")
    assert r["A"] == 2 and r["C"] == 6 and "boom" in r["B"]["failed"]
    assert len((disk_tmp / "l.jsonl").read_text().splitlines()) == 1
    r = pl.run_groups(fn, [("A", 1), ("B", 2), ("C", 3)], workers=3)
    assert r["C"] == 6 and "failed" in r["B"]


def test_prefetcher_order_overlap_and_failure():
    def fn(x):
        time.sleep(0.05)
        if x == 2:
            raise KeyError("bad")
        return x * 10
    t0 = time.time()
    out = []
    for it, v in pl.Prefetcher(fn, [0, 1, 2, 3], depth=2):
        time.sleep(0.05)                                          # consumer work overlaps with the next prefetch
        out.append((it, v))
    assert [i for i, _ in out] == [0, 1, 2, 3] and isinstance(out[2][1], pl.Failed) and out[3][1] == 30
    assert time.time() - t0 < 0.4 * 1.0
