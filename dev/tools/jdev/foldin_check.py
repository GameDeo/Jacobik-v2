"""foldin_check.py -- consistency lint for a production fold-in (read-only; no solves, no writes).

    python -m tools.jdev.foldin_check [--previous <old_fp>] [--agents <path to AGENTS.md>] [--shared-root <main checkout>]

Checks that every place pinning the production identity agrees after a fold-in:
  1. profile.PRODUCTION_FINGERPRINT == profile.snapshot()["fingerprint"] at defaults (no flag overrides in the env).
  2. research/gpuwork/colab/cj.py FINGERPRINT == production.
  3. src/jacobik/ik/solver_replay.py SOURCE_MD5 == md5(ik_solver.py, CRLF-normalised), i.e. replay was regenerated.
  4. tests/: fingerprint literals asserted with == that are not the production one (reported; old fingerprints that a
     test reaches on purpose through flags are listed, not failed, unless --strict).
  5. Docs mention the production fingerprint: README.md, research/archive/flags/FLAGS.md, research/gpuwork/colab/README.md,
     and AGENTS.md (its "Production fingerprint" line).
  6. The DB the profile uses has its <stem>.meta.json sidecar.
  7. With --previous: the archived golden directory research/archive/data/2026-11_golden_history/golden_<previous>/
     exists under the shared root, and research/data/golden/ is newer than it.
Exit status 0 only if every check passes. Prints one line per check.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path

FP_RE = re.compile(r"\b[0-9a-f]{12}\b")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--previous", default=None, help="fingerprint being replaced (checks the golden archive)")
    ap.add_argument("--agents", default=None, help="path to AGENTS.md (default: <shared-root>/AGENTS.md)")
    ap.add_argument("--shared-root", default="/mnt/e/Dev/Projects/jacobik_2026-10-02_full",
                    help="main checkout holding the shared gitignored data")
    ap.add_argument("--strict", action="store_true", help="fail on any non-production fingerprint asserted in tests")
    a = ap.parse_args(argv)

    root = _repo_root()
    shared = Path(a.shared_root)
    sys.path.insert(0, str(root / "src"))
    for k in list(os.environ):
        if k.startswith("JACOBIK_PROFILE") or k == "JACOBIK_FLAGS":
            print(f"[warn] {k} is set in the environment; unset it for a default-profile check")
    import jacobik.core.profile as profile

    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))

    prod = profile.PRODUCTION_FINGERPRINT
    snap = profile.snapshot()["fingerprint"]
    check("profile default == PRODUCTION_FINGERPRINT", snap == prod, f"snapshot {snap}, constant {prod}")

    cj = (root / "research/gpuwork/colab/cj.py").read_text()
    m = re.search(r'^FINGERPRINT\s*=\s*"([0-9a-f]+)"', cj, re.M)
    check("cj.py FINGERPRINT", bool(m) and m.group(1) == prod, m.group(1) if m else "not found")

    src = (root / "src/jacobik/ik/ik_solver.py").read_bytes().replace(b"\r\n", b"\n")
    md5 = hashlib.md5(src).hexdigest()
    rep = (root / "src/jacobik/ik/solver_replay.py").read_text()
    m = re.search(r'^SOURCE_MD5\s*=\s*"([0-9a-f]+)"', rep, re.M)
    check("solver_replay.py regenerated from ik_solver.py", bool(m) and m.group(1) == md5,
          f"replay {m.group(1) if m else None}, source {md5}")

    stale = []
    for t in sorted((root / "dev" / "tests").rglob("*.py")):
        for i, line in enumerate(t.read_text(errors="replace").splitlines(), 1):
            if "fingerprint" in line and "==" in line:
                for fp in FP_RE.findall(line):
                    if fp != prod:
                        stale.append(f"{t.relative_to(root)}:{i} {fp}")
    if stale:
        print("[info] tests asserting a non-production fingerprint (fine only if the test sets flags on purpose):")
        for s in stale:
            print("       " + s)
    check("tests: no stale fingerprint asserts", not stale or not a.strict, f"{len(stale)} listed")

    agents = Path(a.agents) if a.agents else shared / "AGENTS.md"
    docs = [root / "README.md", root / "research/archive/flags/FLAGS.md", root / "research/gpuwork/colab/README.md"]
    for d in docs:
        ok = d.exists() and prod in d.read_text(errors="replace")
        check(f"doc mentions production: {d.relative_to(root)}", ok)
    if agents.exists():
        line = next((l for l in agents.read_text().splitlines() if "Production fingerprint" in l), "")
        check("AGENTS.md production line", prod in line, line[:100])
    else:
        check("AGENTS.md found", False, str(agents))

    try:
        from jacobik.stats import stats as st
        db = getattr(st, "DEFAULT_DB", None) or getattr(profile, "DB_PATH", None)
    except Exception as e:  # pragma: no cover
        db = None
        print(f"[warn] could not resolve the default DB path: {e}")
    if db:
        dbp = Path(db)
        if not dbp.is_absolute():
            dbp = shared / dbp
        sidecar = dbp.with_suffix("").with_suffix(".meta.json") if dbp.suffix == ".json" else dbp.with_name(dbp.stem + ".meta.json")
        check("DB meta sidecar", sidecar.exists(), str(sidecar))
    else:
        print("[info] default DB path not exposed by the code; check the .meta.json sidecar by hand")

    if a.previous:
        arch = shared / f"research/archive/data/2026-11_golden_history/golden_{a.previous}"
        cur = shared / "research/data/golden"
        def _h(p):
            return hashlib.sha256(p.read_bytes()).hexdigest()
        names = sorted(p.name for p in arch.glob("*.npz")) if arch.is_dir() else []
        same = [n for n in names if (cur / n).exists() and _h(cur / n) == _h(arch / n)]
        # content-based: the archive holds the old references, and at least one current file differs from it
        ok = bool(names) and cur.is_dir() and len(same) < len(names)
        check(f"golden archived as golden_{a.previous} and regenerated", ok,
              f"{len(names)} archived, {len(names) - len(same)} differ from current")

    n_fail = results.count(False)
    print(f"{len(results) - n_fail}/{len(results)} checks passed")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
