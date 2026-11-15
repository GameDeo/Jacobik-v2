"""tests/test_nci_v2_frame.py
===========================
Verification test suite for Jacobik NCI v2 spec.json and frame.json:
- Spec schema integrity, three frozen arms, thermodynamics, geometry constants, radial windows
- Shard audit SHA256 integrity and exact corpus counts (34 shards, 125,810 complexes)
- Zero benchmark overlap and zero scaffold leakage against dev32/n48/heldout28/ring10
- Orthogonality and stratification of the 240-group main allocation + 48 reserve groups
- 12-parent development pilot stratification, opportunity coverage, and repeatability designations
- Full TMQMParser / TMComplex assembly validity of pilot parents and sampled allocation
"""

import json
from pathlib import Path
import sys
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SPEC_PATH = REPO_ROOT / "data/refit/nci_v2/pilot_v1/spec.json"
FRAME_PATH = REPO_ROOT / "data/refit/nci_v2/pilot_v1/frame.json"
SCALED_SPEC_PATH = REPO_ROOT / "data/refit/nci_v2/scaled_v1/spec.json"
SCALED_FRAME_PATH = REPO_ROOT / "data/refit/nci_v2/scaled_v1/frame.json"
SCALED_REPORT_PATH = REPO_ROOT / "data/refit/nci_v2/scaled_v1/reports/scaled_scaling_and_support.json"
CLEAN_DIR = REPO_ROOT / "data/refit/clean"


@pytest.fixture(scope="module")
def spec_data():
    assert SPEC_PATH.exists(), f"Spec file not found at {SPEC_PATH}"
    with open(SPEC_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def frame_data():
    assert FRAME_PATH.exists(), f"Frame file not found at {FRAME_PATH}"
    with open(FRAME_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def scaled_spec_data():
    assert SCALED_SPEC_PATH.exists(), f"Scaled spec file not found at {SCALED_SPEC_PATH}"
    with open(SCALED_SPEC_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def scaled_frame_data():
    assert SCALED_FRAME_PATH.exists(), f"Scaled frame file not found at {SCALED_FRAME_PATH}"
    with open(SCALED_FRAME_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def scaled_report_data():
    assert SCALED_REPORT_PATH.exists(), f"Scaled report file not found at {SCALED_REPORT_PATH}"
    with open(SCALED_REPORT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_spec_schema_and_arms(spec_data):
    """Verify spec.json schema, arms, and thermodynamic constants."""
    assert spec_data["$schema"] == "nci_v2.0_pilot"
    assert spec_data["version"] == "2.0.0-pilot.3"
    assert "arms" in spec_data

    # Three frozen arms
    arms = spec_data["arms"]
    assert "control" in arms
    assert "incumbent" in arms
    assert "candidate" in arms

    assert arms["control"]["nci_enabled"] is False
    assert arms["incumbent"]["nci_enabled"] is True
    assert arms["candidate"]["nci_enabled"] is True

    # Seeds and thermodynamics
    assert spec_data["seeds"] == [11, 12, 13]
    thermo = spec_data["thermodynamics"]
    assert thermo["temperature_k"] == 298.15
    assert abs(thermo["gas_constant_r_kj_per_mol_k"] - 0.00831446261815324) < 1e-12

    # Geometry regularizations
    geom = spec_data["regularized_geometry"]
    assert geom["delta_r_angstrom"] == 1e-4
    assert geom["delta_n_angstrom2"] == 1e-6
    assert geom["positivity_eps_nats"] == 0.2


def test_spec_radial_windows_and_admission_rules(spec_data):
    """Verify radial windows, feature dims, and admission rules."""
    rw = spec_data["radial_windows"]
    assert rw["bond_atom"]["h_endpoint"]["window"] == [1.10, 1.35, 6.0, 8.0]
    assert rw["bond_atom"]["non_h_endpoint"]["window"] == [1.80, 2.20, 6.0, 8.0]
    assert rw["bond_ring"]["window"] == [1.60, 2.00, 6.0, 8.0]
    assert rw["ring_ring"]["window"] == [2.40, 2.90, 8.0, 10.0]

    # Admission rules
    ar = spec_data["admission_rules"]
    assert ar["min_ligand_graph_separation_edges"] == 5

    ba = ar["bond_atom"]
    assert "polar_h_acceptor" in ba
    assert "activated_ch_acceptor" in ba
    assert "sp2_ch_acceptor" in ba
    assert "sigma_hole_acceptor" in ba
    assert "polar_multiple_bond" in ba

    # Declarative parameters
    assert ba["polar_h_acceptor"]["allow_anionic_carbon_acceptor"] is False
    assert ba["polar_multiple_bond"]["min_bond_order"] == 1.4

    # Whitelist definition
    assert "common_ligands_exempt_from_scaffold_grouping" in spec_data
    wl = spec_data["common_ligands_exempt_from_scaffold_grouping"]["frozen_whitelist"]
    assert "monodentate_phosphines_phosphites" in wl
    assert "PPh3" in wl["monodentate_phosphines_phosphites"]
    assert "Cp" in wl["cp_type"]
    assert "COD" in wl["neutral_pi_ligands"]
    assert "pyridine" in wl["simple_n_donors"]
    assert "THF" in wl["simple_o_donors"]
    assert "dppe" in wl["bis_diphenylphosphino_alkanes"]
    assert "IMes" in wl["common_nhcs_parent_only"]
    assert "CH2SiMe3" in wl["silyl_alkyl_ancillaries_ge_5_heavy_atoms"]

    not_exempt = spec_data["common_ligands_exempt_from_scaffold_grouping"]["not_exempt_always_grouped"]
    assert "macrocycles" in not_exempt
    assert "polydentate_and_pincers" in not_exempt

    # Context descriptors
    l1_desc = spec_data["context_hierarchy"]["level_1"]["descriptors"]
    expected_desc = [
        "family",
        "admission_class",
        "endpoint_elements",
        "endpoint_charge_classes",
        "bond_order_class",
        "ring_composition",
        "metal_binding_roles",
        "intraligand_flag",
        "has_c_neighbors_flag",
    ]
    assert l1_desc == expected_desc

    br = ar["bond_ring"]
    assert "dh_ring" in br
    assert "ch_ring" in br
    assert "lp_or_halogen_ring" in br

    rr = ar["ring_ring"]
    assert "sp2_5_6_ring_pair" in rr

    # Spline and support grid
    sg = spec_data["joint_support_grid"]
    assert sg["radial_intervals"] == 6
    assert sg["angular_intervals"] == 3
    assert sg["fallback_angular_intervals"] == 2
    assert sg["tau_grp_factor"] == 1e-3
    assert sg["support_regimes"]["matched"]["min_ref_groups"] == 10
    assert sg["support_regimes"]["matched"]["min_ref_ess"] == 10.0
    assert sg["support_regimes"]["ref_only"]["min_root_pool_groups"] == 20
    assert sg["min_ref_only_cal_groups"] == 5

    # Pilot thresholds
    pt = spec_data["pilot_thresholds"]
    assert pt["occurrence_scaling"]["max_median_opf_per_class"] == 100
    assert pt["occurrence_scaling"]["max_p95_opf_per_class"] == 300
    assert pt["control_repeatability"]["repeatability_mean_tol_kj_mol"] == 1.0
    assert pt["control_repeatability"]["repeatability_seed_tol_kj_mol"] == 5.0


def test_frame_shard_audit_integrity(frame_data):
    """Verify shard audit counts, SHA256 hashes, and canonical benchmark exclusions."""
    audit = frame_data["shard_audit"]
    assert audit["total_shards"] == 34
    assert audit["total_complexes"] == 125810

    # Verify every shard file exists and SHA256 is 64 hex characters
    for sh in audit["shards"]:
        assert len(sh["xyz_sha256"]) == 64
        assert len(sh["bo_sha256"]) == 64
        assert sh["n_complexes"] > 0
        xp = REPO_ROOT / sh["xyz_file"]
        bp = REPO_ROOT / sh["bo_file"]
        assert xp.exists(), f"Missing {xp}"
        assert bp.exists(), f"Missing {bp}"

    # Verify canonical benchmark exclusions
    bench = audit["benchmark_exclusions"]
    assert bench["canonical_benchmark_count"] == 188
    assert len(bench["codes"]) == 188
    assert bench["by_source"]["N48"] == 144
    assert bench["by_source"]["RING10"] == 12
    assert bench["by_source"]["DEV32"] == 32


def test_zero_benchmark_leakage(frame_data):
    """Verify zero overlap with canonical benchmarks across all groups, splits, and pilot."""
    benchmarks = set(frame_data["shard_audit"]["benchmark_exclusions"]["codes"])
    assert len(benchmarks) == 188

    # Check pilot parents
    pilot_codes = {p["code"] for p in frame_data["pilot_cohort"]["parents"]}
    assert len(pilot_codes & benchmarks) == 0, f"Pilot contains benchmarks: {pilot_codes & benchmarks}"

    # Check all main allocation groups
    splits = frame_data["main_allocation"]["splits"]
    for s_name in ("surface_fit", "calibration", "motion_validation", "reserve"):
        for g in splits[s_name]:
            members = set(g["member_codes"])
            assert len(members & benchmarks) == 0, f"Split {s_name} group {g['group_id']} contains benchmarks: {members & benchmarks}"
            assert g["representative_code"] not in benchmarks

    # Grouping verification status
    leakage = frame_data["grouping_summary"]["leakage_verification"]
    assert leakage["benchmark_overlap_count"] == 0
    assert leakage["shared_scaffold_overlap_count"] == 0
    assert leakage["cross_split_group_overlap"] == 0


def test_split_disjointness_and_allocation_quotas(frame_data):
    """Verify main allocation quotas (160/40/40 + 48 reserve) and pairwise orthogonality."""
    pilot_codes = {p["code"] for p in frame_data["pilot_cohort"]["parents"]}
    pilot_groups = {p["group_id"] for p in frame_data["pilot_cohort"]["parents"]}
    assert len(pilot_codes) == 12
    assert len(pilot_groups) == 12

    splits = frame_data["main_allocation"]["splits"]
    assert len(splits["surface_fit"]) == 160
    assert len(splits["calibration"]) == 40
    assert len(splits["motion_validation"]) == 40
    assert len(splits["reserve"]) == 48

    split_gids = {}
    split_all_codes = {}
    for s_name, groups in splits.items():
        gids = {g["group_id"] for g in groups}
        codes = {c for g in groups for c in g["member_codes"]}
        split_gids[s_name] = gids
        split_all_codes[s_name] = codes

        # Check inclusion probabilities
        for g in groups:
            assert 0.0 < g["inclusion_probability"] <= 1.0

    # Pairwise disjointness of groups
    all_named_splits = ["surface_fit", "calibration", "motion_validation", "reserve"]
    for i, s1 in enumerate(all_named_splits):
        # Disjoint from pilot
        assert len(split_gids[s1] & pilot_groups) == 0, f"Split {s1} overlaps with pilot groups"
        assert len(split_all_codes[s1] & pilot_codes) == 0, f"Split {s1} overlaps with pilot codes"

        for s2 in all_named_splits[i + 1:]:
            assert len(split_gids[s1] & split_gids[s2]) == 0, f"Split {s1} overlaps with {s2} by group ID"
            assert len(split_all_codes[s1] & split_all_codes[s2]) == 0, f"Split {s1} overlaps with {s2} by complex codes"


def test_pilot_cohort_stratification_and_opportunities(frame_data):
    """Verify pilot cohort stratification, opportunity coverage, and repeatability designations."""
    pilot = frame_data["pilot_cohort"]
    assert pilot["total_parents"] == 12
    parents = pilot["parents"]

    # Stratification checks
    rows = {p["metal_row"] for p in parents}
    assert rows == {"3d", "4d", "5d", "4f"}, f"Pilot rows missing: {rows}"

    q_classes = {p["charge_class"] for p in parents}
    assert q_classes == {"neg", "zero", "pos"}, f"Pilot charge classes missing: {q_classes}"

    size_classes = {p["size_class"] for p in parents}
    assert size_classes == {"small", "medium", "large"}, f"Pilot size classes missing: {size_classes}"

    # Opportunity coverage
    has_zero_opp = any(
        (
            p["opportunities"]["total"] == 0
            or (
                p["opportunities"].get("polar_h_acceptor", 0) == 0
                and p["opportunities"].get("activated_ch_acceptor", 0) == 0
                and p["opportunities"].get("sigma_hole_acceptor", 0) == 0
                and p["opportunities"].get("polar_multiple_bond", 0) == 0
                and p["opportunities"].get("dh_ring", 0) == 0
                and p["opportunities"].get("ch_ring", 0) == 0
                and p["opportunities"].get("lp_or_halogen_ring", 0) == 0
                and p["opportunities"].get("sp2_5_6_ring_pair", 0) == 0
            )
        )
        for p in parents
    )
    has_polar_h = any(p["opportunities"]["polar_h_acceptor"] > 0 for p in parents)
    has_non_hb = any((p["opportunities"]["activated_ch_acceptor"] > 0 or p["opportunities"]["sigma_hole_acceptor"] > 0) for p in parents)
    has_bond_ring = any((p["opportunities"]["ch_ring"] > 0 or p["opportunities"]["dh_ring"] > 0 or p["opportunities"]["lp_or_halogen_ring"] > 0) for p in parents)
    has_ring_ring = any(p["opportunities"]["sp2_5_6_ring_pair"] > 0 for p in parents)

    assert has_zero_opp, "Pilot lacks at least 1 zero-opportunity complex"
    assert has_polar_h, "Pilot lacks polar_h_acceptor opportunities"
    assert has_non_hb, "Pilot lacks non-HB bond-atom opportunities"
    assert has_bond_ring, "Pilot lacks bond_ring opportunities"
    assert has_ring_ring, "Pilot lacks ring_ring opportunities"

    # Repeatability designation
    rep_parents = [p for p in parents if p["is_repeatability"]]
    assert len(rep_parents) == 6
    assert set(pilot["repeatability_parents"]) == {p["code"] for p in rep_parents}

    rep_rows = {p["metal_row"] for p in rep_parents}
    assert len(rep_rows) >= 3, "Repeatability parents must span diverse metal rows"
    rep_sizes = {p["size_class"] for p in rep_parents}
    assert len(rep_sizes) >= 2, "Repeatability parents must span diverse sizes"


def test_parser_verification_status(frame_data):
    """Verify that all sampled complexes were audited with TMQMParser with 0 failures."""
    verification = frame_data["audit_verification"]
    assert verification["failed_count"] == 0
    assert verification["verified_count"] >= 300
    assert len(verification["errors"]) == 0


def test_live_tmqm_parser_on_pilot_parents(frame_data):
    """Live test parsing the 12 pilot parents with TMQMParser to ensure TMComplex validity."""
    from jacobik.stats.parser import TMQMParser, _iter_xyz_blocks, _iter_bo_blocks, _assemble_raw, _attach_qm_properties
    from jacobik.stats.mapping import assemble, METALS
    from jacobik.core.types import TMComplex

    clean_dir = REPO_ROOT / "data/refit/clean"
    parents = frame_data["pilot_cohort"]["parents"]
    parsers = {
        "tmQM": TMQMParser(clean_dir / "tmQM"),
        "lnQM": TMQMParser(clean_dir / "lnQM"),
    }

    for p in parents:
        code = p["code"]
        shard = p["shard"]
        dataset = p["dataset"]
        data_kind = "lnQM" if dataset == "LNf" else "tmQM"
        parent_dir = clean_dir / data_kind
        parser = parsers[data_kind]

        xp = parent_dir / "xyz" / f"tmQM_{shard}.xyz"
        bp = parent_dir / "bo" / f"tmQM_{shard}.BO"

        found = False
        for (comment, atoms), (bo_code, bonds) in zip(_iter_xyz_blocks(xp), _iter_bo_blocks(bp)):
            if bo_code == code:
                csv_row = parser.csv_index.get(code)
                raw = _assemble_raw(comment, atoms, code, bonds, csv_row)
                mol = assemble(raw)
                if csv_row:
                    _attach_qm_properties(mol, csv_row)

                assert isinstance(mol, TMComplex)
                assert mol.csd_code == code
                assert mol.metal_symbol == p["metal"]
                assert mol.metal_symbol in METALS
                assert len(mol.atoms) == p["n_atoms"]
                assert mol.charge == p["charge"]
                assert mol.spin_state == p["spin_multiplicity"]
                found = True
                break

        assert found, f"Could not find and parse pilot parent {code} in shard {shard}"


def test_direct_1hop_benchmark_quarantine(frame_data):
    """Verify Direct 1-Hop Neighbor Quarantine summary and giant-component recovery."""
    grouping = frame_data["grouping_summary"]
    assert grouping["algorithm"] == "direct_1hop_neighbor_quarantine_and_stratified_pairwise_disjoint"
    assert "quarantine" in grouping
    q = grouping["quarantine"]

    assert q["rule"] == "direct_1hop_neighbor_quarantine"
    assert q["benchmark_direct_count"] == 188
    assert q["benchmark_source_sibling_count"] == 0
    assert q["benchmark_shared_informative_scaffold_count"] == 7032
    assert q["total_quarantined_complexes"] == 7220
    assert q["eligible_complexes_count"] == 118590
    assert q["recovered_complexes_from_giant_component"] == 21859
    assert grouping["frozen_common_ligands_count"] == 41


def test_within_split_clustering_and_weighting(frame_data):
    """Verify within-split Union-Find clustering and w_g = 1 / N_g weights."""
    splits = frame_data["main_allocation"]["splits"]
    sampling = frame_data["grouping_summary"]["sampling"]
    split_stats = sampling["split_clustering_and_weighting"]

    # Verify quotas
    assert len(splits["surface_fit"]) == 160
    assert len(splits["calibration"]) == 40
    assert len(splits["motion_validation"]) == 40
    assert len(splits["reserve"]) == 48

    # Verify clustering stats
    assert split_stats["surface_fit"]["independent_groups_count"] == 154
    assert split_stats["surface_fit"]["largest_group_size"] == 3
    assert split_stats["calibration"]["independent_groups_count"] == 40
    assert split_stats["calibration"]["largest_group_size"] == 1
    assert split_stats["motion_validation"]["independent_groups_count"] == 40
    assert split_stats["motion_validation"]["largest_group_size"] == 1
    assert split_stats["reserve"]["independent_groups_count"] == 48
    assert split_stats["reserve"]["largest_group_size"] == 1

    # Verify w_g = 1 / N_g weighting in each entry
    for s_name, group_list in splits.items():
        for g in group_list:
            n_g = g["within_split_group_size"]
            w_g = g["group_weight"]
            assert abs(w_g - 1.0 / n_g) < 1e-9
            assert 0 < w_g <= 1.0


def test_unmappable_input_policy_record(frame_data):
    """Verify predeclared unmappable input policy in frame schema."""
    assert "unmappable_input_policy" in frame_data
    policy = frame_data["unmappable_input_policy"]

    assert policy["policy_name"] == "predeclared_unmappable_input_policy"
    assert "recorded as unmappable outcomes" in policy["failure_recording"]
    assert "surface_fit" in policy["reserve_replacement_scope"]
    assert "predeclared reserve order" in policy["replacement_order"]
    assert "Strict zero-replacement" in policy["evaluation_cohort_policy"]


def test_gate_1_minimums_record(frame_data):
    """Verify Gate 1 minimums (>=10 informative groups in cal and val per activated family)."""
    assert "gate_1_verification" in frame_data
    g1 = frame_data["gate_1_verification"]

    assert g1["min_required_per_activated_family"] == 10
    assert g1["overall_status"] == "passed_gate_1_minimums_satisfied"

    cal = g1["calibration"]
    assert cal["bond_atom"] >= 10
    assert cal["bond_ring"] >= 10
    assert cal["ring_ring"] >= 10
    assert cal["status"] == "passed_gate_1_minimums"

    val = g1["motion_validation"]
    assert val["bond_atom"] >= 10
    assert val["bond_ring"] >= 10
    assert val["ring_ring"] >= 10
    assert val["status"] == "passed_gate_1_minimums"


def test_rdkit_second_hash_check_record(frame_data):
    """Verify the RDKit SMILES/InChIKey second-hash check (same fragments as the WL keys)."""
    leakage = frame_data["grouping_summary"]["leakage_verification"]
    assert "rdkit_second_hash_check" in leakage
    audit = leakage["rdkit_second_hash_check"]
    assert audit["scope"] == "second_hash_check_same_fragments_and_exempt_set_as_wl_keys"

    assert audit["method"] == "rdkit_canonical_smiles_and_inchikey_pairwise_overlap"
    assert audit["total_shared_canonical_smiles"] == 0
    assert audit["total_shared_inchikeys"] == 0
    assert audit["pairwise_overlap_percentage"] == 0.0
    assert audit["pairwise_comparisons_count"] == 15
    assert audit["status"] == "passed_zero_second_hash_overlap"



def test_mappability_screen_and_pilot_overlap_record(frame_data):
    """Every allocated representative passed the mappability screen; excluded strata and the declared
    SUDMEM pilot overlap are recorded."""
    ms = frame_data["grouping_summary"]["sampling"]["mappability_screen"]
    strata = frame_data["main_allocation"]["strata"]
    for k in ms["excluded_strata"]:
        assert strata[k]["allocated_total"] == 0 and strata[k]["prescreen_mappable"] == 0
    leak = frame_data["grouping_summary"]["leakage_verification"]
    assert leak["accepted_pilot_benchmark_scaffold_overlaps"] == ["SUDMEM"]
    assert leak["common_ligand_overlap"]["whitelist_keys_absent_from_corpus"] == []


# =============================================================================
# Scaled Frame & Spec Verification Tests (Directive S1)
# =============================================================================


def test_scaled_spec_schema_and_version(scaled_spec_data):
    """Verify scaled spec version 2.0.0-scaled.1 and schema integrity."""
    assert scaled_spec_data["$schema"] == "nci_v2.0_pilot"
    assert scaled_spec_data["version"] == "2.0.0-scaled.1"
    assert "revision_notes" in scaled_spec_data
    assert any("2.0.0-scaled.1" in note for note in scaled_spec_data["revision_notes"])


def test_scaled_frame_quotas_and_disjointness(scaled_frame_data):
    """Verify scaled frame allocation quotas (1200 fit, 120 cal, 120 val, 240 reserve) and disjointness."""
    splits = scaled_frame_data["main_allocation"]["splits"]
    assert len(splits["surface_fit"]) == 1200
    assert len(splits["calibration"]) == 120
    assert len(splits["motion_validation"]) == 120
    assert len(splits["reserve"]) == 240

    pilot = scaled_frame_data["pilot_cohort"]
    pilot_codes = {p["code"] for p in pilot["parents"]}
    pilot_groups = {p["group_id"] for p in pilot["parents"]}
    assert len(pilot_codes) == 12

    split_gids = {}
    split_all_codes = {}
    for s_name, groups in splits.items():
        gids = {g["group_id"] for g in groups}
        codes = {c for g in groups for c in g["member_codes"]}
        split_gids[s_name] = gids
        split_all_codes[s_name] = codes

        for g in groups:
            assert 0.0 < g["inclusion_probability"] <= 1.0
            assert g["within_split_group_size"] >= 1
            assert abs(g["group_weight"] - 1.0 / g["within_split_group_size"]) < 1e-9

    all_named_splits = ["surface_fit", "calibration", "motion_validation", "reserve"]
    for i, s1 in enumerate(all_named_splits):
        assert len(split_gids[s1] & pilot_groups) == 0, f"Split {s1} overlaps with pilot groups"
        assert len(split_all_codes[s1] & pilot_codes) == 0, f"Split {s1} overlaps with pilot codes"
        for s2 in all_named_splits[i + 1:]:
            assert len(split_gids[s1] & split_gids[s2]) == 0, f"Split {s1} overlaps with {s2} by group ID"
            assert len(split_all_codes[s1] & split_all_codes[s2]) == 0, f"Split {s1} overlaps with {s2} by complex codes"


def test_scaled_frame_zero_leakage_and_benchmark_quarantine(scaled_frame_data):
    """Verify zero benchmark overlap and zero cross-split leakage on scaled frame."""
    benchmarks = set(scaled_frame_data["shard_audit"]["benchmark_exclusions"]["codes"])
    assert len(benchmarks) == 188

    leakage = scaled_frame_data["grouping_summary"]["leakage_verification"]
    assert leakage["status"] == "passed_zero_leakage_verified"
    assert leakage["benchmark_overlap_count"] == 0
    assert leakage["sibling_leakage"]["benchmark_sibling_overlap_count"] == 0
    assert leakage["sibling_leakage"]["cross_split_sibling_overlap_count"] == 0
    assert leakage["informative_scaffold_leakage"]["benchmark_informative_scaffold_overlap_count"] == 0
    assert leakage["informative_scaffold_leakage"]["cross_split_informative_scaffold_overlap_count"] == 0

    rdkit_audit = leakage["rdkit_second_hash_check"]
    assert rdkit_audit["status"] == "passed_zero_second_hash_overlap"
    assert rdkit_audit["total_shared_canonical_smiles"] == 0
    assert rdkit_audit["total_shared_inchikeys"] == 0
    assert rdkit_audit["pairwise_comparisons_count"] == 15


def test_scaled_frame_gate_1_minimums(scaled_frame_data):
    """Verify Gate 1 minimums (>=10 informative groups in cal and val) on scaled frame."""
    g1 = scaled_frame_data["gate_1_verification"]
    assert g1["overall_status"] == "passed_gate_1_minimums_satisfied"
    assert g1["min_required_per_activated_family"] == 10

    cal = g1["calibration"]
    assert cal["status"] == "passed_gate_1_minimums"
    assert cal["bond_atom"] >= 10
    assert cal["bond_ring"] >= 10
    assert cal["ring_ring"] >= 10

    val = g1["motion_validation"]
    assert val["status"] == "passed_gate_1_minimums"
    assert val["bond_atom"] >= 10
    assert val["bond_ring"] >= 10
    assert val["ring_ring"] >= 10


def test_scaled_occurrence_scaling_and_ch_ring_resolution(scaled_report_data):
    """Verify occurrence scaling on the 1200 scaled fit sample and resolution of ch_ring."""
    evaluated = scaled_report_data["evaluated_dataset"]
    assert evaluated["total_complexes_evaluated"] == 1200
    assert evaluated["surface_fit_complexes_count"] == 1200
    assert evaluated["validation_leakage_prevented"] is True

    scaling = scaled_report_data["step_1_admission_scaling"]
    assert scaling["overall_compliance"] == "PASS"

    classes = scaling["per_admission_class_statistics"]
    for c_name, c_info in classes.items():
        dist = c_info["distribution"]
        tc = c_info["threshold_compliance"]
        assert tc["status"] == "PASS", f"Class {c_name} failed: median={dist['median']}, p95={dist['p95']}"
        assert dist["median"] <= 100.0, f"Class {c_name} median {dist['median']} > 100"
        assert dist["p95"] <= 300.0, f"Class {c_name} p95 {dist['p95']} > 300"

    # Specific check on ch_ring resolution
    ch_ring = classes["ch_ring"]
    assert ch_ring["distribution"]["p95"] <= 300.0, f"ch_ring p95 {ch_ring['distribution']['p95']} exceeds 300"
    assert ch_ring["distribution"]["median"] <= 100.0, f"ch_ring median {ch_ring['distribution']['median']} exceeds 100"

    # Joint support resolution
    supp = scaled_report_data["joint_support_feasibility"]
    assert supp["resolution_decision"]["recommended_angular_intervals"] == 3
    assert supp["resolution_decision"]["coarsening_required_by_step1_rule"] is False


def test_scaled_frame_audit_verification(scaled_frame_data):
    """Verify TMQMParser assembly audit on scaled frame (0 failures)."""
    verification = scaled_frame_data["audit_verification"]
    assert verification["failed_count"] == 0
    assert verification["verified_count"] >= 1700
    assert len(verification["errors"]) == 0


def test_hydrogen_mapping_on_symmetric_terminal_groups():
    """Directive R2: verify hydrogen mapping succeeds on symmetric terminal groups

    (aqua O-H, ammine N-H, methyl C-H) across sample reference structures, and that
    IWOLEN is diagnosed as non-isomorphic without forcing a mapping.
    """
    import numpy as np
    from refit.nci_v2_pilot import load_mol
    from jacobik.builder.atom_map import build_atom_mapping

    test_cases = [
        ("CAQLEP", "X1s002", "X1"),
        ("AXOSOY", "X1s005", "X1"),
        ("BEZKOH", "X3s008", "X3"),
        ("CEXPOM", "X3s007", "X3"),
        ("DUQPAJ", "X1s003", "X1"),
        ("FEYWAH", "X3s001", "X3"),
        ("NALJOZ", "X1s007", "X1"),
    ]

    pools_dir = REPO_ROOT / "data/refit/nci_v2/scaled_v1/pools"
    for code, shard, ds in test_cases:
        mol = load_mol(code, shard, ds)
        assert mol is not None, f"Failed to load {code}"
        pool_npz = pools_dir / f"{code}_11_control.npz"
        if pool_npz.exists():
            data = np.load(pool_npz, allow_pickle=True)
            X_gen = data["X"][0]
            ref_X = data["ref_X"]
            sy = data["symbols"]
            ref_sy = data["ref_symbols"]

            res = build_atom_mapping(mol, X_ref=ref_X, X_gen=X_gen)
            assert len(res.gen_to_ref_idx) == len(mol.atoms), (
                f"{code}: mapped {len(res.gen_to_ref_idx)} of {len(mol.atoms)} atoms"
            )
            # Convention: X[:, gen_to_ref_idx] puts coords in ref order, so symbols[gen_to_ref_idx] == ref_symbols
            assert list(sy[res.gen_to_ref_idx]) == list(ref_sy), (
                f"{code}: symbol mismatch under gen_to_ref_idx"
            )

    # IWOLEN: must raise isomorphic error (diagnosed, not forced)
    iwolen_mol = load_mol("IWOLEN", "X3s008", "X3")
    assert iwolen_mol is not None
    iwolen_npz = pools_dir / "IWOLEN_11_control.npz"
    if iwolen_npz.exists():
        data = np.load(iwolen_npz, allow_pickle=True)
        with pytest.raises(RuntimeError, match="Could not find any isomorphic mapping"):
            build_atom_mapping(iwolen_mol, X_ref=data["ref_X"], X_gen=data["X"][0])

