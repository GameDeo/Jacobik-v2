#!/usr/bin/env python3
"""
CRITICAL FIX: stats.py's _is_empty_dist() has a hardcoded type whitelist
that never learned about "gmm_cos" (added this session for the Q2/Q3
refit). Every gmm_cos entry -- all 17607 of them, well-populated ones
included (n in the thousands) -- was being treated as EMPTY by the
resolver, so _resolve() skipped past every candidate at every fallback
level for every Q2/Q3 lookup and raised "No distribution found ... after
full fallback", which ik/scorer.py's try/except catches and silently
falls back to its own crude, generic, donor-subclass-blind default
(Q2: cos=0.0/var=0.5; Q3: cos=-1.0/var=0.1) FOR EVERY SINGLE DONOR/PAIR IN
EVERY CASE, discarding the entire fitted refit. Reproduced and confirmed
against the live constraints.json with real code (verify_stats_bug.py)
before this patch; same script re-run after confirms the fix.
"""
import sys

PATH = "src/jacobik/stats/stats.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

steps_done = []


def do_replace(label, old, new, count=1):
    global src
    n = src.count(old)
    if n == 0:
        print(f"FATAL: pattern for step [{label}] not found", file=sys.stderr)
        sys.exit(1)
    if n != count:
        print(f"FATAL: pattern for step [{label}] found {n} times, expected {count}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, count)
    steps_done.append(label)


OLD = '''def _is_empty_dist(fd: "FittedDist") -> bool:
    """
    True if a FittedDist carries no usable parameters and must not be returned
    by the resolver, regardless of its observation count ``n``.
    """
    t = fd.type
    if t in ("gmm", "vmm"):
        return (not fd.means) or (not fd.weights)
    if t == "halfgauss":
        return fd.mu is None or fd.sigma is None
    return True'''
NEW = '''def _is_empty_dist(fd: "FittedDist") -> bool:
    """
    True if a FittedDist carries no usable parameters and must not be returned
    by the resolver, regardless of its observation count ``n``.

    BUGFIX (this session): "gmm_cos" (Q2/Q3 GMM-on-cos(angle), added
    alongside "gmm"/"vmm" -- see stats/fitter.py's fit_gmm_cosangle()
    docstring) was missing from this whitelist. Every gmm_cos entry fell
    through to the `return True` default below, so _resolve() treated
    EVERY Q2/Q3 distribution as empty regardless of how well-populated it
    was (n in the thousands for some keys) -- the resolver skipped past
    every candidate at every fallback level for every single Q2/Q3
    lookup, raised "No distribution found ... after full fallback", and
    ik/scorer.py's try/except silently swallowed that into its own crude,
    donor-subclass-blind generic default for every donor/pair in every
    case. gmm_cos entries have the same means/vars/weights shape as gmm
    entries (just interpreted in cos-space), so the check is identical.
    """
    t = fd.type
    if t in ("gmm", "vmm", "gmm_cos"):
        return (not fd.means) or (not fd.weights)
    if t == "halfgauss":
        return fd.mu is None or fd.sigma is None
    return True'''
do_replace("S1-is-empty-dist-whitelist", OLD, NEW)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("Steps done so far:", steps_done)
