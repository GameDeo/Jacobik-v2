"""Code-state hash for NCI v2 report and pool provenance.

`git diff HEAD` alone misses untracked files, and all NCI v2 code was untracked, so the old
`code_dirty_diff_sha256` did not change when that code changed. This hashes the tracked diff
plus the path and contents of every untracked, non-ignored file under the code roots.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

CODE_ROOTS = ("src", "refit", "research/archive/code/refit", "research/archive/code/interactions", "research/bench/tools", "tests", "tools")


def code_state_sha256(repo_root: Path) -> str:
    """sha256 over `git diff HEAD -- <roots>` plus untracked files under the roots ("clean" if neither)."""
    try:
        diff = subprocess.check_output(["git", "diff", "HEAD", "--", *CODE_ROOTS], cwd=repo_root)
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", *CODE_ROOTS], cwd=repo_root
        ).split(b"\0")
    except Exception:
        return "unknown"
    untracked = sorted(p for p in untracked if p)
    if not diff.strip() and not untracked:
        return "clean"
    h = hashlib.sha256(diff)
    for rel in untracked:
        h.update(b"\0untracked\0" + rel + b"\0")
        h.update((Path(repo_root) / rel.decode()).read_bytes())
    return h.hexdigest()
