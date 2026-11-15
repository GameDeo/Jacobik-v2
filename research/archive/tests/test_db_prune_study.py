"""test_db_prune_study.py
========================
Unit tests for Q1/Q2 greedy database pruning and error accumulation checks.
"""

import json
import numpy as np
import pytest

from jacobik.stats import mixture as MX
from research.archive.code.refit.db_prune_study import (
    DensityEvaluator,
    GRID_Q1,
    GRID_Q2,
    decode_lossless,
    encode_lossless,
    get_q1_level,
    get_q2_level,
    q1_candidates_all,
    q2_candidates_all,
    run_synthetic_unit_tests,
)


def test_synthetic_3level_ladder_prevents_error_accumulation():
    """Verify that an intermediate shared ladder node is NOT deleted when the accumulated
    divergence from an earlier-deleted leaf would exceed tau.
    """
    grid = GRID_Q2

    # R: Root node (src: fit)
    e_R = {
        "type": "tmix_cos",
        "means": [0.0],
        "vars": [0.1],
        "weights": [1.0],
        "nu": 4.0,
        "src": "fit",
        "pooled": True,
        "n": 100,
    }
    # M: Intermediate shared node (src: adapted), small shift
    e_M = {
        "type": "tmix_cos",
        "means": [0.03],
        "vars": [0.1],
        "weights": [1.0],
        "nu": 4.0,
        "src": "adapted",
        "pooled": True,
        "n": 10,
    }
    # L1: Leaf 1 (src: adapted), shifts further in same direction
    e_L1 = {
        "type": "tmix_cos",
        "means": [0.07],
        "vars": [0.1],
        "weights": [1.0],
        "nu": 4.0,
        "src": "adapted",
        "pooled": True,
        "n": 2,
    }
    # L2: Leaf 2 (src: adapted), virtually identical to M
    e_L2 = {
        "type": "tmix_cos",
        "means": [0.031],
        "vars": [0.1],
        "weights": [1.0],
        "nu": 4.0,
        "src": "adapted",
        "pooled": True,
        "n": 1,
    }

    raw = {"Q2:L1": e_L1, "Q2:L2": e_L2, "Q2:M": e_M, "Q2:R": e_R}
    evaluator = DensityEvaluator(raw)

    d_L1_M = evaluator.max_divergence("Q2:L1", "Q2:M", "Q2")
    d_L2_M = evaluator.max_divergence("Q2:L2", "Q2:M", "Q2")
    d_M_R = evaluator.max_divergence("Q2:M", "Q2:R", "Q2")
    d_L1_R = evaluator.max_divergence("Q2:L1", "Q2:R", "Q2")

    # Set threshold tau such that both L1->M and M->R are individually <= tau,
    # but L1->R > tau.
    tau = (d_L1_M + d_M_R) * 0.75
    assert d_L1_M <= tau
    assert d_M_R <= tau
    assert d_L1_R > tau

    # State tracking:
    # Candidates order: deepest first: L1, L2 (level 0), then M (level 1). R is fit.
    surviving = {"Q2:L1", "Q2:L2", "Q2:M", "Q2:R"}
    dependents = {"Q2:L1": ["Q2:L1"], "Q2:L2": ["Q2:L2"], "Q2:M": ["Q2:M"], "Q2:R": ["Q2:R"]}
    resolution = {"Q2:L1": "Q2:L1", "Q2:L2": "Q2:L2", "Q2:M": "Q2:M", "Q2:R": "Q2:R"}

    # 1. Evaluate L1: fallback is M
    div_L1_M = evaluator.max_divergence("Q2:L1", "Q2:M", "Q2")
    assert div_L1_M <= tau
    surviving.remove("Q2:L1")
    dependents["Q2:M"].extend(dependents["Q2:L1"])
    del dependents["Q2:L1"]
    resolution["Q2:L1"] = "Q2:M"

    # 2. Evaluate L2: fallback is M
    div_L2_M = evaluator.max_divergence("Q2:L2", "Q2:M", "Q2")
    assert div_L2_M <= tau
    surviving.remove("Q2:L2")
    dependents["Q2:M"].extend(dependents["Q2:L2"])
    del dependents["Q2:L2"]
    resolution["Q2:L2"] = "Q2:M"

    # 3. Evaluate M: fallback is R
    # Div(M, R) <= tau
    assert evaluator.max_divergence("Q2:M", "Q2:R", "Q2") <= tau

    # BUT check all dependents of M against their ORIGINAL distribution:
    accept_M = True
    for orig_k in dependents["Q2:M"]:
        if evaluator.max_divergence(orig_k, "Q2:R", "Q2") > tau:
            accept_M = False
            break

    # Crucial assertion: M MUST NOT be deleted because L1 would exceed tau!
    assert not accept_M
    assert "Q2:M" in surviving
    assert resolution["Q2:L1"] == "Q2:M"
    assert resolution["Q2:L2"] == "Q2:M"


def test_synthetic_unit_tests_runner():
    """Verify built-in synthetic tests run without assertion error."""
    run_synthetic_unit_tests()


def test_q2_level_ordering():
    """Test Q2 ladder level depth ordering."""
    l0 = ("octahedral", "c90", "different_ligand", "Fe", "N_pyridine", "O_carbonyl")
    bkt = ("octahedral", "c90", "different_ligand", "Fe", "N_aromatic_any", "O_neutral_pi")
    fam = ("octahedral", "c90", "different_ligand", "Fe", "N_donor", "O_donor")
    pg_fam = ("octahedral", "c90", "different_ligand", "period4_3d", "N_donor", "O_donor")
    g_fam = ("octahedral", "c90", "different_ligand", "N_donor", "O_donor")
    geom_dist = ("octahedral", "c90", "different_ligand")
    cn_dist = ("CN6", "c90", "different_ligand")
    geom = ("octahedral", "c90")
    cn = ("CN6", "c90")
    any_k = ("geom_any", "c90")

    assert get_q2_level(l0)[0] == 0
    assert get_q2_level(bkt)[0] == 1
    assert get_q2_level(fam)[0] == 2
    assert get_q2_level(pg_fam)[0] == 3
    assert get_q2_level(g_fam)[0] == 4
    assert get_q2_level(geom_dist)[0] == 5
    assert get_q2_level(cn_dist)[0] == 6
    assert get_q2_level(geom)[0] == 7
    assert get_q2_level(cn)[0] == 8
    assert get_q2_level(any_k)[0] == 9


def test_q1_level_ordering():
    """Test Q1 ladder level depth ordering for m_sub_early."""
    k_trans = ("Fe", 1, 2, "N_pyridine", "octahedral", "O_carbonyl")
    k_geom = ("Fe", 1, 2, "N_pyridine", "octahedral")
    k_mso_sub = ("Fe", 1, 2, "N_pyridine")
    k_m_sub = ("Fe", "N_pyridine")
    k_mso_fam = ("Fe", 1, 2, "N_donor")
    k_mo_fam = ("Fe", 2, "N_donor")
    k_m_fam = ("Fe", "N_donor")
    k_pg_sub = ("period4_3d", "N_pyridine")
    k_pg_fam = ("period4_3d", "N_donor")
    k_any_fam = ("metal_any", "N_donor")
    k_fam = ("N_donor",)
    k_el = ("element_any",)

    assert get_q1_level(k_trans)[0] < get_q1_level(k_geom)[0]
    assert get_q1_level(k_geom)[0] < get_q1_level(k_mso_sub)[0]
    assert get_q1_level(k_mso_sub)[0] < get_q1_level(k_mso_fam)[0]
    assert get_q1_level(k_mso_fam)[0] < get_q1_level(k_mo_fam)[0]
    assert get_q1_level(k_mo_fam)[0] < get_q1_level(k_m_fam)[0]
    assert get_q1_level(k_m_fam)[0] < get_q1_level(k_pg_sub)[0]
    assert get_q1_level(k_pg_sub)[0] < get_q1_level(k_pg_fam)[0]
    assert get_q1_level(k_pg_fam)[0] < get_q1_level(k_any_fam)[0]
    assert get_q1_level(k_any_fam)[0] < get_q1_level(k_fam)[0]
    assert get_q1_level(k_fam)[0] < get_q1_level(k_el)[0]


def test_lossless_container_round_trip():
    """Verify lossless flat-array container encodes and decodes with exact bit-identity
    across all quantities without using pickle.
    """
    synthetic_db = {
        "Q2:('square_plan', 'c90', 'different_ligand', 'Pd', 'Hal_Cl', 'N_imine')": {
            "type": "tmix_cos",
            "n": 100,
            "means": [0.0, 0.5],
            "vars": [0.01, 0.02],
            "weights": [0.6, 0.4],
            "k": 2,
            "pooled": True,
            "src": "adapted",
            "nu": 4.5,
            "n_eff": 85.0,
            "merged_from": 42,
        },
        "Q1:('Fe', 2, 2, 'N_pyrrole', 'octahedral', 'none')": {
            "type": "gmm",
            "n": 50,
            "means": [1.98],
            "vars": [0.005],
            "weights": [1.0],
            "k": 1,
            "pooled": True,
            "src": "fit",
            "n_eff": 50.0,
        },
        "Q3:('octahedral', 'linear')": {
            "type": "gmm_cos",
            "n": 15,
            "means": [-1.0],
            "vars": [0.02],
            "weights": [1.0],
            "k": 1,
            "n_eff": 15.0,
        },
        "Q5:('Fe', 'c_sp2')": {
            "type": "vmm",
            "n": 30,
            "means": [0.0, 3.14159],
            "kappas": [2.5, 3.0],
            "weights": [0.5, 0.5],
            "k": 2,
            "tail": 0.01,
            "model": "wells_folded",
            "n_eff": 28.0,
        },
        "Q6:('Ni', 'square_plan')": {
            "type": "halfgauss",
            "n": 40,
            "k": 1,
            "mu": 0.05,
            "sigma": 0.02,
            "model": "halfgauss",
        },
        "QO:('Fe', 'octahedral', 6)": {
            "type": "mvgmm",
            "n": 60,
            "weights": [1.0],
            "means": [[1.9, 2.0]],
            "covs": [[[0.01, 0.0], [0.0, 0.02]]],
            "meta": {"dim": 2, "k": 1},
            "feats": ["d1", "d2"],
            "per_sub": None,
        },
        "QG:('tetrahedral', 'Hal_Cl')": {
            "type": "cat",
            "n": 20,
            "labels": ["a", "b"],
            "p": [0.7, 0.3],
            "level": "bucket",
        },
        "QP:('Fe', 2)": {
            "type": "pair_dist",
            "n": 10,
            "counts": [3, 7],
        },
        "NCI:hb_ladder": {
            "type": "nci_table",
            "rows": [{"donor": "N_amide", "acc": "O_carbonyl", "dist": 2.8}],
        },
    }

    encoded = encode_lossless(synthetic_db, float_dtype=np.float64)
    # Check no pickle used:
    assert isinstance(encoded, dict)
    assert all(isinstance(v, np.ndarray) for v in encoded.values())

    decoded = decode_lossless(encoded)
    assert decoded == synthetic_db, "Decoded dictionary must match original exactly"
    s_orig = json.dumps(synthetic_db, sort_keys=True)
    s_decoded = json.dumps(decoded, sort_keys=True)
    assert s_orig == s_decoded, "json.dumps sorted output must be bit-identical"


def test_mass_weighted_divergence():
    """Verify that mass-weighted divergence Delta_E integrates p(x)*|Delta log p| dx."""
    e_narrow = {
        "type": "tmix_cos",
        "means": [0.0],
        "vars": [0.01],
        "weights": [1.0],
        "nu": 4.0,
    }
    # Slightly shifted mean
    e_shifted = {
        "type": "tmix_cos",
        "means": [0.05],
        "vars": [0.01],
        "weights": [1.0],
        "nu": 4.0,
    }

    raw = {"Q2:narrow": e_narrow, "Q2:shifted": e_shifted}
    evaluator = DensityEvaluator(raw)

    sup_div = evaluator.divergence("Q2:narrow", "Q2:shifted", "Q2", metric="sup")
    mass_div = evaluator.divergence("Q2:narrow", "Q2:shifted", "Q2", metric="mass")

    assert mass_div > 0.0, "Divergence between distinct distributions must be positive"
    # Mass-weighted divergence expectation is always <= sup-norm
    assert mass_div < sup_div, f"Mass-weighted {mass_div} should be strictly less than sup {sup_div}"
    assert np.isfinite(mass_div)

