"""Control cache: reuse a recorded control arm instead of rerunning it in every comparison.

A control arm's records are a pure function of what it runs: the code (production src, bench runner, isomer script),
the constraints DB and its sidecars, the device type, the arm's flags and seed budget, the seeds and the run settings.
`control_key` hashes exactly those. A run with `--control-cache ARM`:

1. computes the key for ARM; parents whose records for every seed are in the cache under that key are "cached";
2. copies the cached records (and their pool files) into OUT/ARM, and removes ARM from the main job, so lanes run only
   the other arms for those parents;
3. reruns ARM itself on a few cached parents (the canary, default 5, chosen by a fixed hash so it is stable) in the
   same run, on the same machine, and compares them to the cache: top-1 row identity per parent and seed
   (`tools.run.compare`); any mismatch rejects the cache (the run reports it and exits non-zero; rerun with
   `--control-cache-refresh` to rebuild it);
4. parents not in the cache run the full arm list as usual, and their control records are added to the cache.

A dirty checkout (uncommitted changes under any hashed path) gets no key: the cache is not used and not filled.
Records with a status other than "ok" are never cached.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import tarfile
from contextlib import contextmanager
from pathlib import Path

#: paths whose committed content decides a control arm's records (git tree hashes; dirty -> no key)
CODE_PATHS = ("src", "dev/tools/jgen", "dev/tools/jdev", "research/bench/tools/jgen/bench", "research/bench/tools/jgen/c2")
CANARY_N = 5


def compare_canary(new_dir: Path, ref_dir: Path, codes, seeds) -> dict:
    """Compare successful records only; two failures can never validate a cache."""
    from .compare import compare
    result = compare(new_dir, ref_dir, codes, seeds, "")
    for row in result["rows"]:
        name = f"{row['code']}_{row['seed']}.json"
        for label, directory in (("rerun", new_dir), ("cached", ref_dir)):
            try:
                status = json.loads((directory / "records" / name).read_text()).get("status")
            except (OSError, ValueError):
                status = None
            if status != "ok":
                row["identical"] = False
                row["why"] = "; ".join(filter(None, (row["why"], f"{label} status {status!r}")))
    result["identical"] = sum(row["identical"] for row in result["rows"])
    return result


@contextmanager
def materialize_control_parent(pack_root: Path, arm_dir: Path, code: str, seeds):
    """Stage one parent's loose/packed control files; newer packs and loose files win.

    Temporary files are under the output filesystem, not RAM-backed /tmp. Paths embedded in
    records and manifests are rewritten to the staging directory for geometry comparison and
    cache copying. Only this parent's records and xyz directory are materialized at once.
    """
    pack_root, arm_dir = Path(pack_root), Path(arm_dir)
    pack_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".control-stage-", dir=pack_root) as tmp:
        stage = Path(tmp) / "arm"
        stage.mkdir()
        records = {f"records/{code}_{s}.json" for s in seeds}
        xyz_prefix = f"xyz/{code}/"
        try:
            arm_prefix = arm_dir.relative_to(pack_root).as_posix().rstrip("/") + "/"
        except ValueError:
            arm_prefix = None
        def wanted(name):
            return name in records or name.startswith(xyz_prefix)
        def archive_number(path):
            suffix = path.stem.removeprefix("run_pack")
            return int(suffix) if suffix.isdigit() else -1
        if arm_prefix is not None:
            for archive in sorted(pack_root.glob("run_pack*.tar"), key=archive_number):
                with tarfile.open(archive) as tar:
                    for member in tar:
                        if not member.isfile() or not member.name.startswith(arm_prefix):
                            continue
                        name = member.name[len(arm_prefix):]
                        path = Path(name)
                        if not wanted(name) or path.is_absolute() or ".." in path.parts:
                            continue
                        dest = stage / path
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with tar.extractfile(member) as src, dest.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
        for name in records:
            src = arm_dir / name
            if src.is_file():
                dest = stage / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
        xyz = arm_dir / "xyz" / code
        if xyz.is_dir():
            shutil.copytree(xyz, stage / "xyz" / code, dirs_exist_ok=True)
        for file in stage.rglob("*.json"):
            file.write_text(file.read_text().replace(str(arm_dir), str(stage)))
        yield stage


def compare_canary_packed(new_dir: Path, ref_dir: Path, codes, seeds, pack_root: Path) -> dict:
    """Compare fresh canaries with controls, materializing at most one control parent."""
    results = []
    for code in codes:
        with materialize_control_parent(pack_root, ref_dir, code, seeds) as staged:
            result = compare_canary(new_dir, staged, [code], seeds)
            for row in result["rows"]:
                for label, directory in (("control", staged), ("canary", new_dir)):
                    problem = _pool_problem(directory / "records" / f"{code}_{row['seed']}.json", directory)
                    if problem:
                        row["identical"] = False
                        row["why"] = "; ".join(filter(None, (row["why"], f"{label}: {problem}")))
            result["identical"] = sum(row["identical"] for row in result["rows"])
            results.append(result)
    return dict(n=sum(r["n"] for r in results), identical=sum(r["identical"] for r in results),
                pool_rows_equal=sum(r["pool_rows_equal"] for r in results),
                rows=[row for r in results for row in r["rows"]])


def store_packed(cc, codes, src_arm_dir: Path, pack_root: Path) -> int:
    """Store previously packed or loose controls with bounded temporary inode use."""
    n = 0
    for code in codes:
        with materialize_control_parent(pack_root, src_arm_dir, code, cc.seeds) as staged:
            if not any(_pool_problem(staged / "records" / f"{code}_{seed}.json", staged) for seed in cc.seeds):
                n += cc.store([code], staged)
    return n


def _pool_problem(record_path: Path, staged: Path) -> str | None:
    """Do not silently accept a packed reference whose declared geometry is missing."""
    try:
        record = json.loads(record_path.read_text())
        manifest_name = record.get("pool_manifest")
        if not manifest_name:
            return None
        manifest = Path(manifest_name)
        if not manifest.is_relative_to(staged) or not manifest.is_file():
            return "packed pool manifest unavailable"
        for row in json.loads(manifest.read_text()).get("rows", []):
            path = manifest.parent / row["file"]
            if not path.is_relative_to(staged) or not path.is_file():
                return "packed pool frame unavailable"
    except (OSError, ValueError, KeyError, TypeError):
        return "packed control record or pool unreadable"
    return None


def _git(repo: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def _sha_file(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gpu_name(device: str) -> str:
    if device != "cuda":
        return "cpu"
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", vis],
                             capture_output=True, text=True, check=True).stdout.strip()
        return out.splitlines()[0]
    except Exception:
        return "cuda-unknown"


def _target_identity(plan: dict):
    """Identity of the declared benchmark target, or None when the plan declares none (key unchanged for old runs)."""
    overlay = plan.get("target_overlay")
    if not overlay:
        return None
    repo = Path(plan["repo"])
    review = repo / "src/fit/data/source_review.json"
    if not review.is_file():
        review = repo / "src/jacobik/fit/data/source_review.json"
    return dict(overlay_sha256=_sha_file(Path(overlay)), source_review_sha256=_sha_file(review),
                eligibility_gate=plan.get("gate_eligibility_sha256"))


def control_key(plan: dict, arm: dict) -> tuple[str | None, dict]:
    """(key, inputs) for ARM of PLAN; key is None when the checkout is dirty or a hashed input is missing."""
    repo = Path(plan["repo"])
    try:
        dirty = _git(repo, "status", "--porcelain", "--", *CODE_PATHS)
        trees = {p: _git(repo, "rev-parse", f"HEAD:{p}") for p in CODE_PATHS}
    except subprocess.CalledProcessError as exc:
        return None, dict(reason=f"git: {exc}")
    if dirty:
        return None, dict(reason="uncommitted changes under " + ", ".join(CODE_PATHS))
    db_s = plan.get("env", {}).get("JACOBIK_DB") or os.environ.get("JACOBIK_DB") or ""
    db = Path(db_s) if db_s else None
    if db is None or not db.is_file():
        # the DB the control checkout's bench would load (tools.jgen.inputs.DB_PATH: data root, JACOBIK_DATA_ROOT)
        import sys
        env = dict(os.environ, PYTHONPATH=f"{repo}/src:{repo}/dev:{repo}:{repo}/research/bench")
        try:
            db = Path(subprocess.run([sys.executable, "-c", "from tools.jgen.inputs import DB_PATH; print(DB_PATH)"],
                                     cwd=str(repo), env=env, capture_output=True, text=True, check=True,
                                     timeout=120).stdout.strip().splitlines()[-1])
        except Exception as exc:
            return None, dict(reason=f"cannot resolve the control checkout's DB: {exc!r}"[:300])
    stem = db.with_suffix("")
    # a self-describing container (embedded _meta/_tables) is its own identity: the human .meta.json beside it is not read
    # by any code and carries nothing the key needs. Legacy DBs still hash their sidecars.
    try:
        from jacobik.core.container import read_container_head
        embedded = read_container_head(db)[0] is not None
    except Exception:
        embedded = False
    sidecars = {} if embedded else {s: _sha_file(Path(str(stem) + s)) for s in (".meta.json", ".bends.json")}
    inputs = dict(
        schema="control_cache.1", trees=trees, db_sha256=_sha_file(db), db_sidecars=sidecars,
        # the seed list belongs in the key: the bench solves a parent's seeds in one shared batch
        seeds=[int(x) for x in plan["seeds"]],
        device=plan["device"], gpu=_gpu_name(plan["device"]), backend=plan["backend"],
        isomer_script=_sha_file(Path(plan["isomer_script"])) if plan.get("isomer_script") else None,
        arm=dict(flags=arm.get("flags", {}), seed_budget=arm.get("seed_budget"),
                 extra={k: v for k, v in arm.items() if k not in ("arm", "flags", "seed_budget", "out", "xyz_dir")}),
        # the benchmark target definition: corrected-target overlay file and the source-review/eligibility policy that
        # decides which parents are comparable. None for runs that do not declare one (existing keys are unchanged).
        cpu_identity=plan.get("cpu_identity"), deterministic=plan.get("deterministic"),
        pack_arms=plan.get("pack_arms"), split_seeds=plan.get("split_seeds"),
        env={k: v for k, v in sorted(plan.get("env", {}).items()) if k != "JACOBIK_DB"},
    )
    if inputs["db_sha256"] is None:
        return None, dict(reason=f"DB not found: {db}")
    target = _target_identity(plan)
    if target is not None:  # added only when declared, so every existing control key stays valid
        if target["overlay_sha256"] is None:
            return None, dict(reason=f"target overlay not found: {plan.get('target_overlay')}")
        inputs["target"] = target
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return key, inputs


def cache_root(plan: dict) -> Path:
    from .orchestrator import main_checkout
    env = os.environ.get("JACOBIK_CONTROL_CACHE")
    if env:
        return Path(env)
    if os.environ.get("NERSC_HOST") and os.environ.get("SCRATCH"):
        return Path(os.environ["SCRATCH"]) / "jacobik/control_cache"
    return main_checkout(Path(plan["repo"])) / "research/data/runs/control_cache"


class ControlCache:
    def __init__(self, plan: dict, arm_name: str):
        arms = {a["arm"]: a for a in plan["arms"]}
        if arm_name not in arms:
            raise SystemExit(f"--control-cache {arm_name}: no such arm in {list(arms)}")
        self.arm = arms[arm_name]
        self.name = arm_name
        self.key, self.inputs = control_key(plan, self.arm)
        self.dir = cache_root(plan) / self.key[:20] if self.key else None
        self.seeds = [int(s) for s in plan["seeds"]]

    @property
    def usable(self) -> bool:
        return self.key is not None

    def _rec(self, code, s) -> Path:
        return self.dir / "records" / f"{code}_{s}.json"

    def cached_codes(self, codes) -> list[str]:
        if not self.usable or not self.dir.exists():
            return []
        out = []
        for c in codes:
            ok = True
            for s in self.seeds:
                p = self._rec(c, s)
                if not p.exists():
                    ok = False
                    break
            if ok:
                out.append(c)
        return out

    @staticmethod
    def canary(codes, n=CANARY_N) -> list[str]:
        return sorted(codes, key=lambda c: hashlib.sha256(c.encode()).hexdigest())[:n]

    def _copy_parent(self, code, src_arm_dir: Path, dst_arm_dir: Path, seeds) -> int:
        """records/<code>_<s>.json and xyz/<code>/ from SRC to DST; absolute SRC paths inside the JSON files are
        rewritten to DST so pool manifests resolve. Returns the number of records copied."""
        n = 0
        src_x, dst_x = src_arm_dir / "xyz" / code, dst_arm_dir / "xyz" / code
        if src_x.is_dir():
            if dst_x.exists():
                shutil.rmtree(dst_x)
            shutil.copytree(src_x, dst_x)
            for j in dst_x.glob("*.json"):
                j.write_text(j.read_text().replace(str(src_arm_dir), str(dst_arm_dir)))
        (dst_arm_dir / "records").mkdir(parents=True, exist_ok=True)
        for s in seeds:
            sp = src_arm_dir / "records" / f"{code}_{s}.json"
            if sp.exists():
                (dst_arm_dir / "records" / sp.name).write_text(sp.read_text().replace(str(src_arm_dir), str(dst_arm_dir)))
                n += 1
        return n

    def restore(self, codes, dst_arm_dir: Path) -> int:
        return sum(self._copy_parent(c, self.dir, dst_arm_dir, self.seeds) for c in codes)

    def store(self, codes, src_arm_dir: Path) -> int:
        """Add finished control parents to the cache (only parents whose records are all status ok)."""
        if not self.usable:
            return 0
        self.dir.mkdir(parents=True, exist_ok=True)
        kf = self.dir / "key.json"
        if not kf.exists():
            kf.write_text(json.dumps(dict(key=self.key, inputs=self.inputs), indent=1))
        n = 0
        for c in codes:
            recs = [src_arm_dir / "records" / f"{c}_{s}.json" for s in self.seeds]
            try:
                if not all(json.loads(p.read_text()).get("status") == "ok" for p in recs):
                    continue
            except Exception:
                continue
            n += self._copy_parent(c, src_arm_dir, self.dir, self.seeds) and 1
        return n
