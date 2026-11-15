"""test_profile: production fingerprint, flag registration, and override contracts."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

from jacobik.core import profile as P

REPO = Path(__file__).resolve().parents[2]
PRODUCTION_FINGERPRINT = "2c87e2940af2"


def _subproc(code, env_extra=None, overrides=None, check=True):
    env = {k: v for k, v in os.environ.items() if k not in ("JACOBIK_PROFILE", "JACOBIK_PROFILE_OVERRIDES")}
    if overrides is not None:
        env["JACOBIK_PROFILE_OVERRIDES"] = json.dumps(overrides)
    if env_extra:
        env.update(env_extra)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO)])
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=str(REPO), capture_output=True, text=True, check=check)


def test_production_baseline_values_and_hypothesis_tables():
    """Production fingerprint, profile.get() frozen baseline, and hypothesis layer imports."""
    snap = P.snapshot()
    assert snap["profile"] in ("production", "u1")
    assert snap["fingerprint"] == PRODUCTION_FINGERPRINT
    assert P.fingerprint() == P.PRODUCTION_FINGERPRINT == PRODUCTION_FINGERPRINT
    assert snap["differs_from_profile"] == {}

    assert P.get("nci") is False
    assert P.get("deterministic_embed") is True
    assert P.get("ring_fold_hinges") is False
    assert P.get("bend_release") == [0.1, 100]
    assert P.get("q6_model") == "repulsive"
    assert P.get("oxidation_model") == "training"
    assert P.get("solve_floor") == "thermal"
    assert P.get("rank_agg") == "sum"
    assert P.get("score_agg") == "sum"

    from jacobik.stats import hypothesis_tables as HT, reference_ll  # noqa: F401
    assert callable(HT.fit_geometry_prior) and callable(HT.fit_partition_prior)
    assert "from jacobik.stats import hypothesis_tables as HT" in (REPO / "src" / "fit" / "extractor.py").read_text()


def test_flag_registration_and_module_binding():
    """Every FIELDS entry resolves to its module attribute with matching default and valid metadata."""
    for name, (module, attr, default, doc) in sorted(P.FIELDS.items()):
        assert module.startswith(("jacobik.", "fit.")), name
        assert doc and isinstance(doc, str), name
        value = getattr(importlib.import_module(module), attr)
        assert value == default, f"{module}.{attr} = {value!r} but registered default is {default!r}"


def test_profile_overrides_and_validation():
    """Defaults, restatements, registered overrides, finish_phase C1 identity, and strict errors."""
    code = (
        "import json, jacobik.core.profile as P; "
        "d = P.snapshot(); "
        "P.FIELDS['test_experiment_flag'] = ('jacobik.builder.engine', 'TEST_EXP', False, 'test flag'); "
        "fp_reg = P.fingerprint(); "
        "diff_reg = P.snapshot()['differs_from_profile']; "
        "fp_ovr = P.fingerprint({**P.snapshot()['fields'], 'test_experiment_flag': True}); "
        "print(json.dumps({'d': d, 'fp_reg': fp_reg, 'diff_reg': diff_reg, 'fp_ovr': fp_ovr}))"
    )
    res = json.loads(_subproc(code).stdout.strip().splitlines()[-1])
    default_snap = res["d"]
    assert default_snap["fingerprint"] == PRODUCTION_FINGERPRINT
    assert default_snap["differs_from_profile"] == {}
    assert res["fp_reg"] == PRODUCTION_FINGERPRINT and res["diff_reg"] == {}
    assert res["fp_ovr"] not in (PRODUCTION_FINGERPRINT, res["fp_reg"])
    assert default_snap["fields"]["finish_phase"] == {
        "search": "irprop+", "optimizer": "irprop+", "switch": 100,
        "release": ["symmetry"], "sigma_source": "thermal",
        "sigma": {"translation": 0.0585, "rotation": 0.08609, "torsion": 0.2402, "ring": 0.2402, "bend": 0.08609},
    }

    # Explicit finish_phase=None changes fingerprint; restating production defaults preserves it
    fp_none = json.loads(_subproc(
        "import json, jacobik.core.profile as P; print(json.dumps(P.snapshot()['fingerprint']))",
        overrides={"finish_phase": None},
    ).stdout.strip().splitlines()[-1])
    assert fp_none == "f3f88072302c"

    fp_restate = json.loads(_subproc(
        "import json, jacobik.core.profile as P; print(json.dumps(P.snapshot()['fingerprint']))",
        overrides={
            "nci": False, "deterministic_embed": True, "nci_types": ["hb"],
            "finish_phase": default_snap["fields"]["finish_phase"],
        },
    ).stdout.strip().splitlines()[-1])
    assert fp_restate == PRODUCTION_FINGERPRINT

    # Misspelled overrides and hardcoded production settings must fail at import
    for bad in ({"not_a_field": 1}, {"unknown_flag": True}, {"ring_fold_hinges": True}):
        r = _subproc("import jacobik.core.profile", overrides=bad, check=False)
        assert r.returncode != 0, bad
        if "ring_fold_hinges" not in bad:
            assert "unknown fields" in r.stderr
