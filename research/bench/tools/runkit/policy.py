"""Fail-closed data policy: runners call this before they touch any parent.

Reads `research/bench/benchmarks/locked_exclusions.json` (reserve + motion-validation codes). Never reads `.claude/locked/`.
The file is untracked in worktrees, so it is looked up in the main checkout first (RUNKIT_EXCLUSIONS overrides).

  assert_allowed(codes)       strict: refuses every excluded code (val and reserve). Use for new suites and training data.
  assert_panel("cal"|"tail")  tail and cal only. The tail panel is the named exception: AGENTS.md makes it a development
                              panel although 5 of its 10 parents are val parents (LOCKED_SETS.md); reserve is never allowed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from . import resources

REL = "research/bench/benchmarks/locked_exclusions.json"
#: panels.TAIL of tools.jgen.bench.panels (a test keeps the two in step)
TAIL_PANEL = ("DAMVAQ", "HOXJUZ", "IFUROQ", "JEGPUJ", "NEPNEF", "SOBHAS", "USAPOU", "XEYTUT", "YUCZUS", "ZELTAP")
REQUIRED = ("reserve_all_codes", "motion_validation_codes")


class PolicyError(ValueError):
    pass


def parent(code) -> str:
    c = str(code).upper()
    if not re.fullmatch(r"[A-Z]{6}(?:\d{2})?", c):
        raise PolicyError(f"expected a CSD code (six letters, optional two digits), got {code!r}")
    return c[:6]


def default_path() -> Path:
    if os.environ.get("RUNKIT_EXCLUSIONS"):
        return Path(os.environ["RUNKIT_EXCLUSIONS"])
    for base in (resources.MAIN, Path(__file__).resolve().parents[4]):
        if (base / REL).exists():
            return base / REL
    return resources.MAIN / REL


class DataPolicy:
    def __init__(self, path=None):
        self.path = Path(path) if path else default_path()
        try:
            raw = self.path.read_bytes()
            content = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise PolicyError(f"cannot read the exclusion list {self.path}: {exc}") from exc
        if not all(isinstance(content.get(k), list) and content[k] for k in REQUIRED):
            raise PolicyError("exclusion list lacks a mandatory non-empty partition: " + ", ".join(REQUIRED))
        self.reserve = {parent(c) for c in content["reserve_all_codes"]}
        self.val = {parent(c) for c in content["motion_validation_codes"]}
        self.excluded = self.reserve | self.val
        self.sha256 = hashlib.sha256(raw).hexdigest()

    def _record(self, codes, what):
        return dict(path=str(self.path.resolve()), sha256=self.sha256, parents=sorted({parent(c) for c in codes}),
                    excluded_parent_count=len(self.excluded), asserted=what)

    def assert_allowed(self, codes) -> dict:
        codes = list(codes)
        blocked = sorted({parent(c) for c in codes} & self.excluded)
        if blocked:
            raise PolicyError("excluded parents (val or reserve): " + ", ".join(blocked))
        return self._record(codes, "allowed")

    def assert_panel(self, panel: str, codes, cal_codes=None) -> dict:
        """panel 'tail': codes must be inside the tail panel (its val parents are allowed, reserve never).
        panel 'cal': codes must not be excluded; if `cal_codes` (the calibration split) is given they must lie inside it."""
        codes = list(codes)
        ps = {parent(c) for c in codes}
        if panel not in ("tail", "cal"):
            raise PolicyError(f"panel {panel!r} is not allowed here; develop on tail and cal only")
        if ps & self.reserve:
            raise PolicyError("reserve parents are never allowed: " + ", ".join(sorted(ps & self.reserve)))
        if panel == "tail":
            if ps - set(TAIL_PANEL):
                raise PolicyError("not in the tail panel: " + ", ".join(sorted(ps - set(TAIL_PANEL))))
        else:
            blocked = sorted(ps & self.val)
            if blocked:
                raise PolicyError("val parents are not allowed on cal: " + ", ".join(blocked))
            if cal_codes is not None and ps - {parent(c) for c in cal_codes}:
                raise PolicyError("not in the calibration split: " + ", ".join(sorted(ps - {parent(c) for c in cal_codes})))
        return self._record(codes, f"panel:{panel}")


_DEFAULT = None


def get() -> DataPolicy:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = DataPolicy()
    return _DEFAULT


def assert_allowed(codes) -> dict:
    return get().assert_allowed(codes)


def assert_panel(panel: str, codes, cal_codes=None) -> dict:
    return get().assert_panel(panel, codes, cal_codes)


def smoke_limit(codes, lo=1, hi=5):
    n = len({parent(c) for c in codes})
    if not lo <= n <= hi:
        raise PolicyError(f"a smoke run takes {lo}..{hi} parents, got {n}")
