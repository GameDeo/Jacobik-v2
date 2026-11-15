"""T-P0 and T-G1 / T-G3: profile fingerprint invariance and gating of new flags (roadmap_generation_v1.md 1.2)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONTROL_OVERRIDES = {"nci": False, "deterministic_embed": True}


def _snapshot(env_extra=None, unset_profile=False, overrides=None):
    """Run snapshot() in a fresh interpreter (the profile is read at import)."""
    env = {k: v for k, v in os.environ.items() if k not in ("JACOBIK_PROFILE", "JACOBIK_PROFILE_OVERRIDES")}
    if unset_profile is False:
        env["JACOBIK_PROFILE"] = "u1"
    if overrides is not None:
        env["JACOBIK_PROFILE_OVERRIDES"] = json.dumps(overrides)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO)])
    code = "import json, jacobik.profile as P; print(json.dumps(P.snapshot()))"
    out = subprocess.check_output([sys.executable, "-c", code], env=env, cwd=str(REPO), text=True)
    return json.loads(out.strip().splitlines()[-1])


def test_fingerprint_legacy_fields_are_the_pre_p0_set():
    """The frozen list is exactly the 36 fields that existed before P0 (the set the fingerprint always hashed)."""
    import jacobik.profile as P

    assert len(P.LEGACY_HASHED_FIELDS) == 36
    # fields added after P0 are not legacy; each is listed here when it is added
    assert set(P.FIELDS) - set(P.LEGACY_HASHED_FIELDS) == {"ring_fold_hinges"}


def test_control_arm_fingerprint_is_unchanged():
    """T-P0: the control arm (u1 with nci off and deterministic embed) keeps the fingerprint of every stored pool."""
    snap = _snapshot(overrides=CONTROL_OVERRIDES)
    assert snap["fingerprint"] == "485e943af036"


def test_bare_u1_fingerprint_is_unchanged_by_p0():
    """The bare u1 profile (no overrides) keeps the value it had before P0 (18dec82067b1, measured 2026-10-25)."""
    snap = _snapshot()
    assert snap["fingerprint"] == "18dec82067b1"


def test_fingerprint_changes_with_one_non_legacy_value():
    import jacobik.profile as P

    live = {k: v[2] for k, v in P.FIELDS.items()}
    base = P.fingerprint(live)
    changed = dict(live)
    changed["out_window"] = 201.0
    assert P.fingerprint(changed) != base


def test_new_field_at_legacy_value_keeps_fingerprint(monkeypatch):
    """A field added after P0 is hashed only when it differs from its legacy value (the T-P0 rule)."""
    import jacobik.profile as P

    live = {k: v[2] for k, v in P.FIELDS.items()}
    base = P.fingerprint(live)
    fields = dict(P.FIELDS)
    fields["probe_flag"] = ("jacobik.profile", "PROBE", True, "test only")
    monkeypatch.setattr(P, "FIELDS", fields)
    assert P.fingerprint({**live, "probe_flag": True}) == base
    assert P.fingerprint({**live, "probe_flag": False}) != base


@pytest.mark.parametrize("bad", [{"not_a_field": 1}, {"ring_fold_hinge": False}])
def test_misspelled_override_raises(bad):
    """T-G3: an unknown field in JACOBIK_PROFILE_OVERRIDES is an error, not a silent no-op."""
    env = dict(os.environ, JACOBIK_PROFILE_OVERRIDES=json.dumps(bad), PYTHONPATH=str(REPO / "src"))
    env.pop("JACOBIK_PROFILE", None)
    r = subprocess.run([sys.executable, "-c", "import jacobik.profile"], env=env, cwd=str(REPO),
                       capture_output=True, text=True)
    assert r.returncode != 0 and "unknown fields" in r.stderr


def test_setting_ring_fold_hinges_changes_the_fingerprint():
    """T-P0 second half: a new flag at a non-legacy value changes the identity; at its legacy value it does not."""
    on = _snapshot(overrides=CONTROL_OVERRIDES)["fingerprint"]
    off = _snapshot(overrides=dict(CONTROL_OVERRIDES, ring_fold_hinges=False))["fingerprint"]
    legacy = _snapshot(overrides=dict(CONTROL_OVERRIDES, ring_fold_hinges=True))["fingerprint"]
    assert on == "485e943af036"
    assert off != on
    assert legacy == on
