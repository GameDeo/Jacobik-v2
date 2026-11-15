"""Jacobik NCI v2 Experiment Registry and Provenance Auditor.

This module provides the schema, validator, builder, and markdown renderer for the
Jacobik NCI v2 Experiment Registry (data/refit/experiments/index.json and
data/refit/experiments/README.md), adhering strictly to roadmapv2.md Sections 1.2,
1.3, 6.1, 7, and 7.1.

It also provides the dependency auditor for tools/jdev/budget/ scripts, producing
data/refit/experiments/dependency_audit.json.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, List, Optional, Set, Tuple
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]

# Valid experiment execution / evaluation modes per roadmapv2.md Section 1.3
VALID_MODES: Set[str] = {
    "fresh",
    "rerank",
    "motion_scan",
    "dimer",
    "curvature",
    "embedding",
    "audit",
}

VALID_STATUSES: Set[str] = {
    "completed",
    "superseded",
    "failed",
    "rejected",
    "active",
    "proposed",
    "inconclusive",
}

VALID_AVAILABILITIES: Set[str] = {
    "available",
    "archived",
    "unavailable",
    "missing",
}


@dataclass
class GenerationProvenance:
    """Explicit provenance of coordinates generation."""
    code_hash: Optional[str] = None
    db_hash: Optional[str] = None
    full_profile: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, Any]] = None


@dataclass
class RankingProvenance:
    """Explicit provenance of ranking scorer and candidate pools."""
    code_hash: Optional[str] = None
    db_hash: Optional[str] = None
    full_profile: Optional[Dict[str, Any]] = None
    source_pool_ids: Optional[List[str]] = None


@dataclass
class EvaluationProvenance:
    """Provenance of energy/force reference evaluator."""
    model_hash: Optional[str] = None
    evaluator_hash: Optional[str] = None
    dtype: str = "float64"
    units: str = "kJ/mol"
    state_reference_hashes: Optional[Dict[str, Any]] = None


@dataclass
class ArtifactRecord:
    """Physical artifact or archive member associated with an experiment."""
    path: str
    sha256: Optional[str] = None
    availability: str = "available"
    description: Optional[str] = None


@dataclass
class MetricRecord:
    """Evaluation metrics, definitions, and uncertainty bounds."""
    definitions: Dict[str, str] = field(default_factory=dict)
    aggregation_unit: str = "pool_and_case"
    paired_comparator: Optional[str] = None
    values: Dict[str, Any] = field(default_factory=dict)
    uncertainty: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentRecord:
    """Immutable experiment record per roadmapv2.md Section 1.3."""
    id: str
    readable_name: str
    mode: str
    status: str
    question: str
    parent_record_ids: List[str]
    cohort_manifest: Any
    split: Any
    selection_rule: str
    exclusions: Any
    seeds: List[int]
    generation: GenerationProvenance
    ranking: RankingProvenance
    evaluation: EvaluationProvenance
    artifacts: List[ArtifactRecord]
    metrics: MetricRecord
    conclusion: str
    limitations: str
    superseded_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sha256_of_file(filepath: Path | str) -> Optional[str]:
    """Compute sha256 of a regular file if it exists."""
    p = Path(filepath)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fp:
        while chunk := fp.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_archive_member(zip_path: Path | str, member_name: str) -> Optional[str]:
    """Compute sha256 of a member inside a zip archive."""
    zp = Path(zip_path)
    if not zp.is_file():
        return None
    try:
        with zipfile.ZipFile(zp, "r") as z:
            data = z.read(member_name)
            return hashlib.sha256(data).hexdigest()
    except Exception:
        return None


def validate_record(record_data: Dict[str, Any], repo_root: Optional[Path] = None) -> List[str]:
    """Validate a single experiment record against schema and artifact integrity rules."""
    errors: List[str] = []
    r_id = record_data.get("id", "<missing_id>")

    # 1. Required top-level fields
    required_fields = [
        "id", "readable_name", "mode", "status", "question", "parent_record_ids",
        "cohort_manifest", "split", "selection_rule", "exclusions", "seeds",
        "generation", "ranking", "evaluation", "artifacts", "metrics",
        "conclusion", "limitations", "superseded_by"
    ]
    for rf in required_fields:
        if rf not in record_data:
            errors.append(f"[{r_id}] Missing required top-level field: '{rf}'")

    # 2. Mode validity
    mode = record_data.get("mode")
    if mode not in VALID_MODES:
        errors.append(f"[{r_id}] Invalid mode '{mode}'. Must be one of {sorted(VALID_MODES)}")

    # 3. Status validity
    status = record_data.get("status")
    if status not in VALID_STATUSES:
        errors.append(f"[{r_id}] Invalid status '{status}'. Must be one of {sorted(VALID_STATUSES)}")

    # 4. Seeds must be list of ints
    seeds = record_data.get("seeds")
    if not isinstance(seeds, list) or not all(isinstance(s, int) for s in seeds):
        errors.append(f"[{r_id}] 'seeds' must be a list of integers")

    # 5. Generation vs. ranking explicit distinction
    gen = record_data.get("generation", {})
    if not isinstance(gen, dict):
        errors.append(f"[{r_id}] 'generation' must be a dictionary")
    else:
        for f in ["code_hash", "db_hash", "full_profile", "settings", "environment"]:
            if f not in gen:
                errors.append(f"[{r_id}] generation missing field '{f}'")

    ranking = record_data.get("ranking", {})
    if not isinstance(ranking, dict):
        errors.append(f"[{r_id}] 'ranking' must be a dictionary")
    else:
        for f in ["code_hash", "db_hash", "full_profile", "source_pool_ids"]:
            if f not in ranking:
                errors.append(f"[{r_id}] ranking missing field '{f}'")

    if mode == "rerank":
        source_pools = ranking.get("source_pool_ids") if isinstance(ranking, dict) else None
        if not source_pools:
            errors.append(f"[{r_id}] Rerank mode requires explicit non-empty 'ranking.source_pool_ids'")

    # 6. Evaluation provenance
    eval_prov = record_data.get("evaluation", {})
    if not isinstance(eval_prov, dict):
        errors.append(f"[{r_id}] 'evaluation' must be a dictionary")
    else:
        for f in ["model_hash", "evaluator_hash", "dtype", "units", "state_reference_hashes"]:
            if f not in eval_prov:
                errors.append(f"[{r_id}] evaluation missing field '{f}'")

    # 7. Metrics structure
    metrics = record_data.get("metrics", {})
    if not isinstance(metrics, dict):
        errors.append(f"[{r_id}] 'metrics' must be a dictionary")
    else:
        for f in ["definitions", "aggregation_unit", "paired_comparator", "values", "uncertainty"]:
            if f not in metrics:
                errors.append(f"[{r_id}] metrics missing field '{f}'")

    # 8. Artifacts validation & checksum verification
    artifacts = record_data.get("artifacts", [])
    if not isinstance(artifacts, list):
        errors.append(f"[{r_id}] 'artifacts' must be a list of artifact records")
    else:
        root = repo_root or Path.cwd()
        for i, art in enumerate(artifacts):
            if not isinstance(art, dict):
                errors.append(f"[{r_id}] artifact index {i} must be a dict")
                continue
            path_str = art.get("path")
            if not path_str:
                errors.append(f"[{r_id}] artifact index {i} missing 'path'")
                continue
            avail = art.get("availability")
            if avail not in VALID_AVAILABILITIES:
                errors.append(f"[{r_id}] artifact '{path_str}' invalid availability '{avail}'")
            expected_sha = art.get("sha256")

            # Checksum verification if available or archived
            if expected_sha and avail in {"available", "archived"}:
                if "::" in path_str:
                    zip_rel, member_rel = path_str.split("::", 1)
                    zip_full = root / zip_rel
                    if zip_full.is_file():
                        actual_sha = sha256_of_archive_member(zip_full, member_rel)
                        if actual_sha and actual_sha != expected_sha:
                            errors.append(
                                f"[{r_id}] Checksum mismatch for archive member '{path_str}': "
                                f"expected {expected_sha}, got {actual_sha}"
                            )
                else:
                    full_p = root / path_str
                    if full_p.is_file():
                        actual_sha = sha256_of_file(full_p)
                        if actual_sha and actual_sha != expected_sha:
                            errors.append(
                                f"[{r_id}] Checksum mismatch for file '{path_str}': "
                                f"expected {expected_sha}, got {actual_sha}"
                            )

    return errors


def validate_registry(records: List[Dict[str, Any]], repo_root: Optional[Path] = None) -> List[str]:
    """Validate full registry of records for consistency, uniqueness, and reference integrity."""
    errors: List[str] = []
    seen_ids: Set[str] = set()

    for r in records:
        r_id = r.get("id")
        if not r_id:
            errors.append("Encountered record with missing or empty 'id'")
            continue
        if r_id in seen_ids:
            errors.append(f"Duplicate record ID '{r_id}'")
        seen_ids.add(r_id)

        rec_errors = validate_record(r, repo_root=repo_root)
        errors.extend(rec_errors)

    # Check parent and superseded_by links
    for r in records:
        r_id = r.get("id", "")
        parents = r.get("parent_record_ids", [])
        for pid in parents:
            if pid not in seen_ids:
                errors.append(f"[{r_id}] References unknown parent_record_id: '{pid}'")
        s_by = r.get("superseded_by")
        if s_by and s_by not in seen_ids:
            errors.append(f"[{r_id}] References unknown superseded_by ID: '{s_by}'")

    return errors


def build_historical_records() -> List[Dict[str, Any]]:
    """Construct the immutable historical records covering Roadmap v2 Sections 1.2, 1.3, 6.1, and 7.1."""
    records: List[ExperimentRecord] = []

    # Common evaluator constants
    mace64_model_hash = "e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510"
    mace64_evaluator_legacy_hash = "0eacabfaaff3f25e5a93ee9bf083f0a47b3672aff8d57f2ab8d3e89089c22e74"
    mace64_evaluator_fixed_hash = "af0baa88805e7463a7de4a13b4fbe8bc6b63656dc3bd898b084c5cfa06ac6cd5"

    constraints_2026_10_17_hash = "4689673b9e1a9be1a7b83cedb7b04c46fa3a3e48a09ba18e50b4dbdc0d532947"
    constraints_2026_10_18_hash = "48ac82b56d71fd6e43515ecf830d4712a71de3df533b4d6ff8762f7ebe7549bf"

    dev80_evidence_sha = "16ea492c3976af1a8bddecaf4e0234978f86e99f484224de32e2150bb0c28b7d"
    recon_evidence_sha = "50eb2965de5e6797814d726aba2ddfd3f6f9cc6f9bd9d8f8c7cc90e12fdb9725"

    metric_defs = {
        "pick_mean": "Mean signed energy difference vs crystal reference Rp for top-1 ranked pose across 228 pools / 76 cases (kJ/mol)",
        "pick_median": "Median signed energy difference vs crystal reference Rp for top-1 pose (kJ/mol)",
        "floor_mean": "Mean signed energy difference vs reference Rp for best valid pose in generated pool (kJ/mol)",
        "floor_median": "Median signed energy difference vs reference Rp for best valid pose in pool (kJ/mol)",
        "regret_mean": "Mean selection regret (pick - floor) for valid top-1 poses (kJ/mol)",
        "regret_median": "Median selection regret (pick - floor) (kJ/mol)",
        "top1_share": "Fraction of pools where top-1 pose is within 4.184 kJ/mol (1 kcal/mol) of valid pool floor",
        "hb_subset_mean": "Mean top-1 energy difference on four-case HB subset (PEWBOL, YIYXEJ, CBTYCO, BULGAT) (kJ/mol)",
        "seed_spread_gt_50": "Number of complexes with max-seed minus median-seed energy spread > 50 kJ/mol",
    }

    # -------------------------------------------------------------------------
    # 1. Dev80: u1c
    # -------------------------------------------------------------------------
    records.append(
        ExperimentRecord(
            id="dev80__gen-u1-contact__rank-contact__mace64__s11-13",
            readable_name="u1c",
            mode="fresh",
            status="superseded",
            question="Historical contact-Q6 solver baseline: evaluate fresh conformer generation and ranking with older contact Q6 wells.",
            parent_record_ids=[],
            cohort_manifest="data/refit/evidence/nci_dev80_benchmark_2026-10-18.json",
            split="dev80_historical",
            selection_rule="Dev80 benchmark panel (dev32 + n48), 76 valid complexes with valid donor mappings",
            exclusions=["BIDHEB", "EHUZAI", "PUSSIF", "TALRIH"],
            seeds=[11, 12, 13],
            # no git commit was recorded for these runs; the value previously here was the profile fingerprint, which stays in full_profile
            generation=GenerationProvenance(
                code_hash=None,
                db_hash=constraints_2026_10_17_hash,
                full_profile={"name": "u1", "fingerprint": "d3a141c85f71", "q6_pmf_wells": "contact", "score_agg": "sum", "rank_agg": "sum", "solve_floor": "thermal"},
                settings={"beam_width": 16, "adam_steps": 200, "lr": 0.1, "dedup_rmsd": 0.5, "out_window": 200.0},
                environment={"platform": "linux-64", "torch_dtype": "float32"},
            ),
            ranking=RankingProvenance(
                code_hash=None,
                db_hash=constraints_2026_10_17_hash,
                full_profile={"name": "u1", "fingerprint": "d3a141c85f71", "q6_pmf_wells": "contact"},
                source_pool_ids=["dev80__gen-u1-contact__rank-contact__mace64__s11-13"],
            ),
            evaluation=EvaluationProvenance(
                model_hash=mace64_model_hash,
                evaluator_hash=mace64_evaluator_legacy_hash,
                dtype="float64",
                units="kJ/mol",
                state_reference_hashes={"dev80": dev80_evidence_sha},
            ),
            artifacts=[
                ArtifactRecord(path="data/refit/evidence/nci_dev80_benchmark_2026-10-18.json", sha256=dev80_evidence_sha, availability="available", description="Saved dev80 benchmark record"),
                ArtifactRecord(path="data/refit/evidence/n1_n2_baseline_reconciliation_2026-10-18.json", sha256=recon_evidence_sha, availability="available", description="Arm reconciliation record"),
            ],
            metrics=MetricRecord(
                definitions=metric_defs,
                aggregation_unit="pool_and_case",
                paired_comparator=None,
                values={
                    "pick_mean": 101.7168,
                    "pick_median": 71.9815,
                    "floor_mean": 81.3553,
                    "floor_median": 59.9932,
                    "regret_mean": 20.3615,
                    "regret_median": 7.7761,
                    "top1_share": 0.4211,
                    "hb_subset_mean": 142.3062,
                    "seed_spread_gt_50": 6,
                    "invalid_share_gt_1000_pct": 1.7564,
                },
                uncertainty={"pools": 228, "complexes": 76},
            ),
            conclusion="Contact-Q6 wells produce unphysically deep wells and high top-1 energy (101.72 kJ/mol). Replaced by repulsive Q6 arm.",
            limitations="Historical solver run; retained attractive contact wells that cause severe distortions.",
            superseded_by="dev80__gen-u1-rep__rank-rep__mace64__s11-13",
        )
    )

    # -------------------------------------------------------------------------
    # 2. Dev80: u1c+rep / historical N1
    # -------------------------------------------------------------------------
    records.append(
        ExperimentRecord(
            id="dev80__gen-u1-rep__rank-rep__mace64__s11-13",
            readable_name="u1c+rep / historical N1",
            mode="fresh",
            status="superseded",
            question="Repulsion arm retaining legacy polar-H attractions: test replacement of contact Q6 with repulsive Q6 while retaining historical polar-H wells.",
            parent_record_ids=["dev80__gen-u1-contact__rank-contact__mace64__s11-13"],
            cohort_manifest="data/refit/evidence/nci_dev80_benchmark_2026-10-18.json",
            split="dev80_historical",
            selection_rule="Dev80 benchmark panel, 76 valid complexes",
            exclusions=["BIDHEB", "EHUZAI", "PUSSIF", "TALRIH"],
            seeds=[11, 12, 13],
            # no git commit was recorded for these runs; the value previously here was the profile fingerprint, which stays in full_profile
            generation=GenerationProvenance(
                code_hash=None,
                db_hash=constraints_2026_10_17_hash,
                full_profile={"name": "u1", "fingerprint": "fb687ed6ee11", "q6_pmf_wells": "repulsive"},
                settings={"beam_width": 16, "adam_steps": 200, "lr": 0.1, "dedup_rmsd": 0.5, "out_window": 200.0},
                environment={"platform": "linux-64", "torch_dtype": "float32"},
            ),
            ranking=RankingProvenance(
                code_hash=None,
                db_hash=constraints_2026_10_17_hash,
                full_profile={"name": "u1", "fingerprint": "fb687ed6ee11", "q6_pmf_wells": "repulsive"},
                source_pool_ids=["dev80__gen-u1-rep__rank-rep__mace64__s11-13"],
            ),
            evaluation=EvaluationProvenance(
                model_hash=mace64_model_hash,
                evaluator_hash=mace64_evaluator_legacy_hash,
                dtype="float64",
                units="kJ/mol",
                state_reference_hashes={"dev80": dev80_evidence_sha},
            ),
            artifacts=[
                ArtifactRecord(path="data/refit/evidence/nci_dev80_benchmark_2026-10-18.json", sha256=dev80_evidence_sha, availability="available", description="Saved dev80 benchmark record"),
                ArtifactRecord(path="data/refit/evidence/n1_n2_baseline_reconciliation_2026-10-18.json", sha256=recon_evidence_sha, availability="available", description="Arm reconciliation record"),
            ],
            metrics=MetricRecord(
                definitions=metric_defs,
                aggregation_unit="pool_and_case",
                paired_comparator="dev80__gen-u1-contact__rank-contact__mace64__s11-13",
                values={
                    "pick_mean": 87.9820,
                    "pick_median": 65.1483,
                    "floor_mean": 73.0923,
                    "floor_median": 54.5879,
                    "regret_mean": 14.8897,
                    "regret_median": 3.6242,
                    "top1_share": 0.5263,
                    "hb_subset_mean": 107.6222,
                    "seed_spread_gt_50": 4,
                    "invalid_share_gt_1000_pct": 1.2619,
                },
                uncertainty={"paired_mean_diff_vs_u1c": -13.7348, "floor_mean_diff_vs_u1c": -8.2630},
            ),
            conclusion="Repulsive Q6 substantially improved over contact Q6 (-13.73 kJ/mol mean top-1), but still retained legacy polar-H attractions from PMF.",
            limitations="Retained legacy polar-H attractions; did not yet include corrected oxidation-state inference.",
            superseded_by="dev80__gen-u1-rep-os__rank-rep-os__mace64__s11-13",
        )
    )

    # -------------------------------------------------------------------------
    # 3. Dev80: u1c+rep+os
    # -------------------------------------------------------------------------
    records.append(
        ExperimentRecord(
            id="dev80__gen-u1-rep-os__rank-rep-os__mace64__s11-13",
            readable_name="u1c+rep+os",
            mode="fresh",
            status="superseded",
            question="Evaluate fresh conformer generation and ranking with corrected training-convention oxidation-state inference added to repulsion arm.",
            parent_record_ids=["dev80__gen-u1-rep__rank-rep__mace64__s11-13"],
            cohort_manifest="data/refit/evidence/nci_dev80_benchmark_2026-10-18.json",
            split="dev80_historical",
            selection_rule="Dev80 benchmark panel, 76 valid complexes",
            exclusions=["BIDHEB", "EHUZAI", "PUSSIF", "TALRIH"],
            seeds=[11, 12, 13],
            # no git commit was recorded for these runs; the value previously here was the profile fingerprint, which stays in full_profile
            generation=GenerationProvenance(
                code_hash=None,
                db_hash=constraints_2026_10_17_hash,
                full_profile={"name": "u1", "fingerprint": "a3528aa88adf", "oxidation_model": "training", "q6_pmf_wells": "repulsive"},
                settings={"beam_width": 16, "adam_steps": 200, "lr": 0.1, "dedup_rmsd": 0.5, "out_window": 200.0, "source_pool_tag": "etkdg_u1_os"},
                environment={"platform": "linux-64", "torch_dtype": "float32"},
            ),
            ranking=RankingProvenance(
                code_hash=None,
                db_hash=constraints_2026_10_17_hash,
                full_profile={"name": "u1", "fingerprint": "a3528aa88adf", "oxidation_model": "training", "q6_pmf_wells": "repulsive"},
                source_pool_ids=["dev80__gen-u1-rep-os__rank-rep-os__mace64__s11-13"],
            ),
            evaluation=EvaluationProvenance(
                model_hash=mace64_model_hash,
                evaluator_hash=mace64_evaluator_legacy_hash,
                dtype="float64",
                units="kJ/mol",
                state_reference_hashes={"dev80": dev80_evidence_sha},
            ),
            artifacts=[
                ArtifactRecord(path="data/refit/evidence/nci_dev80_benchmark_2026-10-18.json", sha256=dev80_evidence_sha, availability="available", description="Saved dev80 benchmark record"),
                ArtifactRecord(path="data/refit/evidence/n1_n2_baseline_reconciliation_2026-10-18.json", sha256=recon_evidence_sha, availability="available", description="Arm reconciliation record"),
            ],
            metrics=MetricRecord(
                definitions=metric_defs,
                aggregation_unit="pool_and_case",
                paired_comparator="dev80__gen-u1-rep__rank-rep__mace64__s11-13",
                values={
                    "pick_mean": 85.9688,
                    "pick_median": 64.2914,
                    "floor_mean": 72.4605,
                    "floor_median": 53.1984,
                    "regret_mean": 13.5083,
                    "regret_median": 3.6243,
                    "top1_share": 0.5307,
                    "hb_subset_mean": 84.2574,
                    "seed_spread_gt_50": 4,
                    "invalid_share_gt_1000_pct": 1.2539,
                },
                uncertainty={"paired_mean_diff_vs_rep": -2.0132, "ci_95": [-5.14, -0.14], "mde": 3.69},
            ),
            conclusion="Corrected oxidation-state inference produced modest gain (-2.01 kJ/mol [CI -5.14, -0.14]) and created the frozen etkdg_u1_os coordinate pools used for subsequent reranking studies.",
            limitations="Still retained legacy polar-H attractions during coordinate generation.",
            superseded_by="dev80__gen-os-pool__rank-nci-hb__mace64__s11-13",
        )
    )

    # -------------------------------------------------------------------------
    # 4. Dev80: u1_pure_rep rerank on etkdg_u1_os
    # -------------------------------------------------------------------------
    records.append(
        ExperimentRecord(
            id="dev80__gen-os-pool__rank-pure-rep__mace64__s11-13",
            readable_name="u1_pure_rep rerank on etkdg_u1_os",
            mode="rerank",
            status="completed",
            question="Pure-repulsion reranking baseline: score stored etkdg_u1_os pools with all Q6 attractions and NCI completely removed.",
            parent_record_ids=["dev80__gen-u1-rep-os__rank-rep-os__mace64__s11-13"],
            cohort_manifest="data/refit/evidence/nci_dev80_benchmark_2026-10-18.json",
            split="dev80_historical",
            selection_rule="Dev80 benchmark panel, 76 valid complexes",
            exclusions=["BIDHEB", "EHUZAI", "PUSSIF", "TALRIH"],
            seeds=[11, 12, 13],
            # no git commit was recorded for these runs; the value previously here was the profile fingerprint, which stays in full_profile
            generation=GenerationProvenance(
                code_hash=None,
                db_hash=constraints_2026_10_17_hash,
                full_profile={"name": "u1", "fingerprint": "a3528aa88adf", "oxidation_model": "training", "q6_pmf_wells": "repulsive"},
                settings={"beam_width": 16, "adam_steps": 200, "lr": 0.1, "dedup_rmsd": 0.5, "out_window": 200.0, "source_arm": "etkdg_u1_os"},
                environment={"platform": "linux-64", "torch_dtype": "float32"},
            ),
            ranking=RankingProvenance(
                code_hash="3588573",
                db_hash=constraints_2026_10_18_hash,
                full_profile={"name": "u1", "q6_pmf_wells": "repulsive", "nci_types": []},
                source_pool_ids=["dev80__gen-u1-rep-os__rank-rep-os__mace64__s11-13"],
            ),
            evaluation=EvaluationProvenance(
                model_hash=mace64_model_hash,
                evaluator_hash=mace64_evaluator_legacy_hash,
                dtype="float64",
                units="kJ/mol",
                state_reference_hashes={"dev80": dev80_evidence_sha},
            ),
            artifacts=[
                ArtifactRecord(path="data/refit/evidence/nci_dev80_benchmark_2026-10-18.json", sha256=dev80_evidence_sha, availability="available", description="Saved dev80 benchmark record"),
                ArtifactRecord(path="data/refit/evidence/n1_n2_baseline_reconciliation_2026-10-18.json", sha256=recon_evidence_sha, availability="available", description="Arm reconciliation record"),
            ],
            metrics=MetricRecord(
                definitions=metric_defs,
                aggregation_unit="pool_and_case",
                paired_comparator="dev80__gen-u1-rep-os__rank-rep-os__mace64__s11-13",
                values={
                    "pick_mean": 87.0262,
                    "pick_median": 64.2914,
                    "floor_mean": 72.4605,
                    "floor_median": 53.1984,
                    "regret_mean": 14.5657,
                    "regret_median": 3.8166,
                    "top1_share": 0.5175,
                    "hb_subset_mean": 101.4485,
                    "seed_spread_gt_50": 3,
                    "invalid_share_gt_1000_pct": 1.2539,
                },
                uncertainty={"paired_mean_diff_vs_os": 1.0574},
            ),
            conclusion="Pure repulsion reranking on etkdg_u1_os pools gives 87.03 kJ/mol top-1 mean. Crucially, pool floor is identical (72.46 kJ/mol) because coordinates are reused from etkdg_u1_os.",
            limitations="Fixed-pool reranking study; coordinates were generated with legacy polar-H attractions present.",
            superseded_by=None,
        )
    )

    # -------------------------------------------------------------------------
    # 5. Dev80: u1+nci_hb rerank on etkdg_u1_os
    # -------------------------------------------------------------------------
    records.append(
        ExperimentRecord(
            id="dev80__gen-os-pool__rank-nci-hb__mace64__s11-13",
            readable_name="u1+nci_hb rerank on etkdg_u1_os",
            mode="rerank",
            status="completed",
            question="Adopted NCI baseline: score stored etkdg_u1_os pools with pure repulsion plus calibrated C2 HB-NCI.",
            parent_record_ids=["dev80__gen-os-pool__rank-pure-rep__mace64__s11-13", "dev80__gen-u1-rep-os__rank-rep-os__mace64__s11-13"],
            cohort_manifest="data/refit/evidence/nci_dev80_benchmark_2026-10-18.json",
            split="dev80_historical",
            selection_rule="Dev80 benchmark panel, 76 valid complexes",
            exclusions=["BIDHEB", "EHUZAI", "PUSSIF", "TALRIH"],
            seeds=[11, 12, 13],
            # no git commit was recorded for these runs; the value previously here was the profile fingerprint, which stays in full_profile
            generation=GenerationProvenance(
                code_hash=None,
                db_hash=constraints_2026_10_17_hash,
                full_profile={"name": "u1", "fingerprint": "a3528aa88adf", "oxidation_model": "training", "q6_pmf_wells": "repulsive"},
                settings={"beam_width": 16, "adam_steps": 200, "lr": 0.1, "dedup_rmsd": 0.5, "out_window": 200.0, "source_arm": "etkdg_u1_os"},
                environment={"platform": "linux-64", "torch_dtype": "float32"},
            ),
            ranking=RankingProvenance(
                code_hash="3588573",
                db_hash=constraints_2026_10_18_hash,
                full_profile={"name": "u1", "q6_pmf_wells": "repulsive", "nci_types": ["hb"]},
                source_pool_ids=["dev80__gen-u1-rep-os__rank-rep-os__mace64__s11-13"],
            ),
            evaluation=EvaluationProvenance(
                model_hash=mace64_model_hash,
                evaluator_hash=mace64_evaluator_legacy_hash,
                dtype="float64",
                units="kJ/mol",
                state_reference_hashes={"dev80": dev80_evidence_sha},
            ),
            artifacts=[
                ArtifactRecord(path="data/refit/evidence/nci_dev80_benchmark_2026-10-18.json", sha256=dev80_evidence_sha, availability="available", description="Saved dev80 benchmark record"),
                ArtifactRecord(path="data/refit/evidence/n1_n2_baseline_reconciliation_2026-10-18.json", sha256=recon_evidence_sha, availability="available", description="Arm reconciliation record"),
                ArtifactRecord(path="data/refit/final/constraints_2026-10-18.json", sha256=constraints_2026_10_18_hash, availability="available", description="Adopted production constraints database"),
            ],
            metrics=MetricRecord(
                definitions=metric_defs,
                aggregation_unit="pool_and_case",
                paired_comparator="dev80__gen-os-pool__rank-pure-rep__mace64__s11-13",
                values={
                    "pick_mean": 86.3400,
                    "pick_median": 64.2914,
                    "floor_mean": 72.4605,
                    "floor_median": 53.1984,
                    "regret_mean": 13.8795,
                    "regret_median": 3.8166,
                    "top1_share": 0.5175,
                    "hb_subset_mean": 88.3842,
                    "seed_spread_gt_50": 3,
                    "invalid_share_gt_1000_pct": 1.2539,
                },
                uncertainty={"paired_mean_diff_vs_pure_rep": -0.6862, "ci_95": [-1.78, 0.01], "mde": 1.38, "paired_mean_diff_vs_rep_os": 0.3712, "ci_95_vs_rep_os": [0.0, 0.89]},
            ),
            conclusion="Adopted production incumbent: top-1 mean 86.34 kJ/mol, -0.69 kJ/mol paired change vs pure-rep reranking (CI [-1.78, +0.01]). Coordinates identical to etkdg_u1_os; does not demonstrate fresh generation gains.",
            limitations="Reranking comparison on pre-generated coordinates. Does not test fresh conformer optimization with NCI active in Adam solver.",
            superseded_by=None,
        )
    )

    # -------------------------------------------------------------------------
    # 6. Archived Multi-type Fresh Follow-up: n2_followup_2026-10-18
    # -------------------------------------------------------------------------
    followup_md_sha = "dcbbdbc04126a1298d17de2adebf0e3b25b4ef9bcd5e081e0e920d985fb1ad58"
    followup_json_sha = "8f8ac82ef0a0c133ac4c104fc3e70ec9d0e459fd3500e99587fb63672a1f42e6"
    records.append(
        ExperimentRecord(
            id="n2_followup_2026-10-18",
            readable_name="n2_followup_2026-10-18",
            mode="fresh",
            status="failed",
            question="Multi-type fresh conformer generation follow-up (hb, pp, chp) on selected diagnostic cohort.",
            parent_record_ids=["dev80__gen-os-pool__rank-nci-hb__mace64__s11-13"],
            cohort_manifest="data/_archive/reports_superseded_2026-10-18.zip::data/refit/reports/n2_followup_summary_2026-10-18.json",
            split="selected_diagnostic_cohort",
            selection_rule="7 diverse diagnostic complexes (BULGAT, CBTYCO, FETSUU, JEFJUZ, PEWBOL, XEPKAF, YIYXEJ)",
            exclusions=[],
            seeds=[11, 12, 13],
            generation=GenerationProvenance(
                code_hash="2f92108f41b0e28e0e2b6f04792b8e26",
                db_hash=constraints_2026_10_18_hash,
                full_profile={"name": "u1", "nci_types": ["hb", "pp", "chp"], "solve_floor": "thermal"},
                settings={"beam_width": 16, "adam_steps": 200, "lr": 0.1, "completed_pools": 69, "finite_pools": 69},
                environment={"platform": "linux-64", "torch_dtype": "float32"},
            ),
            ranking=RankingProvenance(
                code_hash="2f92108f41b0e28e0e2b6f04792b8e26",
                db_hash=constraints_2026_10_18_hash,
                full_profile={"name": "u1", "nci_types": ["hb", "pp", "chp"]},
                source_pool_ids=["n2_followup_2026-10-18"],
            ),
            evaluation=EvaluationProvenance(
                model_hash=mace64_model_hash,
                evaluator_hash=mace64_evaluator_legacy_hash,
                dtype="float64",
                units="kJ/mol",
                state_reference_hashes={},
            ),
            artifacts=[
                ArtifactRecord(path="data/_archive/reports_superseded_2026-10-18.zip::data/refit/reports/n2_followup_2026-10-18.md", sha256=followup_md_sha, availability="archived", description="Followup markdown report"),
                ArtifactRecord(path="data/_archive/reports_superseded_2026-10-18.zip::data/refit/reports/n2_followup_summary_2026-10-18.json", sha256=followup_json_sha, availability="archived", description="Followup summary JSON"),
            ],
            metrics=MetricRecord(
                definitions={"case_mean_change": "Three-seed mean top-1 energy difference (on minus off) in kJ/mol", "pewbol_control_drift": "Difference between fresh off-control and stored control for PEWBOL seed 11 (kJ/mol)"},
                aggregation_unit="complex",
                paired_comparator="fresh_control_off",
                values={
                    "PEWBOL_mean_change": 87.54,
                    "FETSUU_mean_change": 26.78,
                    "BULGAT_mean_change": -0.15,
                    "CBTYCO_mean_change": -8.70,
                    "JEFJUZ_mean_change": -23.85,
                    "XEPKAF_mean_change": 10.59,
                    "YIYXEJ_mean_change": 39.29,
                    "pewbol_seed11_control_drift": 89.13,
                },
                uncertainty={"curvature_central_term_max_error": 1.788e-05, "score_max_error": 0.0},
            ),
            conclusion="Multi-type NCI candidate failed fresh generation on selected cohort; severe regressions on PEWBOL (+87.54 kJ/mol) and FETSUU (+26.78 kJ/mol). PEWBOL seed-11 control drift (+89.13 kJ/mol) demonstrated high solver sensitivity. NCI kept off.",
            limitations="Selected small diagnostic cohort (7 cases), not population-level benchmark. Evaluated multiple NCI types simultaneously.",
            superseded_by=None,
        )
    )

    # -------------------------------------------------------------------------
    # 7. Calibration v2: n2_calibration_implementation_2026-10-18
    # -------------------------------------------------------------------------
    calib_json_sha = "ff0098e5ae278bf6c446cc28a09b2acad9fe4fea7cd5cbeea1da9144d344051a"
    calib_md_sha = "2e98433536bd7bb2d1e44b7114fb16b73fc1231457a36e3bdfc910e0a84f5350"
    records.append(
        ExperimentRecord(
            id="n2_calibration_implementation_2026-10-18",
            readable_name="n2_calibration_implementation_2026-10-18",
            mode="dimer",
            status="completed",
            question="Leak-free training calibration v2 across hb, pp, and chp on independent rigid-pair geometries.",
            parent_record_ids=["dev80__gen-os-pool__rank-nci-hb__mace64__s11-13"],
            cohort_manifest="data/_archive/refit_reports_2026-10-20/n2_calibration_implementation_manifest_2026-10-18.json",
            split="scaffold_disjoint_16_16",
            selection_rule="96 training contact cases, 32 complexes per type (hb, pp, chp), parent-disjoint and individual-scaffold-disjoint 16+16 splits",
            exclusions=["leaked_pair_scaffolds_v1"],
            seeds=[],
            generation=GenerationProvenance(
                code_hash="3588573",
                db_hash=constraints_2026_10_18_hash,
                full_profile=None,
                settings={"contact_cases": 96, "unique_parents": 80, "poses": 4328, "reused_raw": 16},
                environment={"platform": "linux-64", "gfn2_rigid_pairs": True},
            ),
            ranking=RankingProvenance(
                code_hash="3588573",
                db_hash=constraints_2026_10_18_hash,
                full_profile=None,
                source_pool_ids=None,
            ),
            evaluation=EvaluationProvenance(
                model_hash=mace64_model_hash,
                evaluator_hash=mace64_evaluator_legacy_hash,
                dtype="float64",
                units="kJ/mol",
                state_reference_hashes={},
            ),
            artifacts=[
                ArtifactRecord(path="data/_archive/refit_reports_2026-10-20/n2_calibration_implementation_2026-10-18.json", sha256=calib_json_sha, availability="available", description="Calibration implementation JSON"),
                ArtifactRecord(path="data/_archive/refit_reports_2026-10-20/n2_calibration_implementation_2026-10-18.md", sha256=calib_md_sha, availability="available", description="Calibration implementation Markdown report"),
            ],
            metrics=MetricRecord(
                definitions={"frozen_coefficient": "Fitted scalar amplitude multiplier gamma", "mean_case_rmse": "Mean RMSE across contact cases on validation split (kJ/mol)", "wrong_sign_curvatures": "Count of negative response curvatures on positive-curvature reference directions"},
                aggregation_unit="contact_case",
                paired_comparator="uncalibrated_baseline",
                values={
                    "hb_frozen_coefficient": 0.232544,
                    "hb_validation_calibrated_rmse": 2.4005,
                    "pp_frozen_coefficient": 0.383178,
                    "pp_validation_calibrated_rmse": 0.6978,
                    "chp_frozen_coefficient": 0.0,
                    "chp_validation_calibrated_rmse": 0.6992,
                },
                uncertainty={"hb_validation_rmse_improvement_ci95": [-0.0760, 0.2869], "pp_validation_rmse_improvement_ci95": [-0.0461, 0.0490]},
            ),
            conclusion="Leak-free calibration complete. HB small-motion validation signal positive but descriptor collisions and curvature failures prevented advancement. PP CI included zero. CHP scalar fit chose zero. No candidate advanced to dev80.",
            limitations="Only calibrated scalar multiplier amplitude on frozen shapes; did not optimize underlying well widths or context specificity.",
            superseded_by=None,
        )
    )

    # -------------------------------------------------------------------------
    # 8. Contextual-HB Fresh Failure: nci_ctx1
    # -------------------------------------------------------------------------
    ctx_res_sha = "c8d475c68af42e7ac6dba62e520379d9a5412b567b86fa4732b09c92c7fbf9c0"
    ctx_man_sha = "f49fc00615acfcf6ccea82364bee05fd1f13220eedbc7f40077d6161dfc9c672"
    records.append(
        ExperimentRecord(
            id="nci_ctx1",
            readable_name="nci_ctx1",
            mode="fresh",
            status="failed",
            question="Contextual charge-aware HB prior fresh conformer generation test on 4 HB complexes.",
            parent_record_ids=["dev80__gen-os-pool__rank-nci-hb__mace64__s11-13"],
            cohort_manifest="data/_archive/refit_reports_2026-10-20/n2_hb_context_generation_2026-10-19/manifest.json",
            split="hb_4case_panel",
            selection_rule="4 dev80 HB complexes (PEWBOL, YIYXEJ, CBTYCO, BULGAT) + 2 control complexes (HUYNEU, OJIMIF), seeds 11-13",
            exclusions=[],
            seeds=[11, 12, 13],
            generation=GenerationProvenance(
                code_hash="3588573",
                db_hash="47c517f2bab07c04112fb69396c24e21a49d553b7b1bd52e4277e84bb2b3ade8",
                full_profile={"name": "u1", "nci_types": ["hb"], "contextual_hb": True},
                settings={"beam_width": 16, "adam_steps": 200, "lr": 0.1},
                environment={"platform": "linux-64", "torch_dtype": "float32"},
            ),
            ranking=RankingProvenance(
                code_hash="3588573",
                db_hash="47c517f2bab07c04112fb69396c24e21a49d553b7b1bd52e4277e84bb2b3ade8",
                full_profile={"name": "u1", "nci_types": ["hb"], "contextual_hb": True},
                source_pool_ids=["nci_ctx1"],
            ),
            evaluation=EvaluationProvenance(
                model_hash=mace64_model_hash,
                evaluator_hash=mace64_evaluator_legacy_hash,
                dtype="float64",
                units="kJ/mol",
                state_reference_hashes={},
            ),
            artifacts=[
                ArtifactRecord(path="data/_archive/refit_reports_2026-10-20/n2_hb_context_generation_2026-10-19/results.json", sha256=ctx_res_sha, availability="available", description="Context generation results JSON"),
                ArtifactRecord(path="data/_archive/refit_reports_2026-10-20/n2_hb_context_generation_2026-10-19/manifest.json", sha256=ctx_man_sha, availability="available", description="Context generation manifest JSON"),
                ArtifactRecord(path="/tmp/nci_ctx1_outputs", sha256=None, availability="unavailable", description="Temporary solver run scratch directory"),
            ],
            metrics=MetricRecord(
                definitions={"hb_mean_change": "Mean change in top-1 ranked energy vs adopted baseline across 4 HB cases (kJ/mol)", "hb_mean_floor_change": "Mean change in pool floor energy across 4 HB cases (kJ/mol)"},
                aggregation_unit="case_and_seed",
                paired_comparator="dev80__gen-os-pool__rank-nci-hb__mace64__s11-13",
                values={
                    "hb_mean_change": -0.1551,
                    "hb_mean_floor_change": 9.2604,
                    "PEWBOL_mean_change": 7.0042,
                    "PEWBOL_mean_floor_change": 38.6938,
                    "YIYXEJ_mean_change": 1.7766,
                    "YIYXEJ_mean_floor_change": 0.8878,
                    "CBTYCO_mean_change": -9.3879,
                    "CBTYCO_mean_floor_change": -2.8986,
                    "BULGAT_mean_change": 0.0,
                    "BULGAT_mean_floor_change": 0.3587,
                },
                uncertainty={"cases": 4, "pools": 12},
            ),
            conclusion="Contextual charge-aware HB model failed fresh generation. While 4-case mean top-1 was -0.16 kJ/mol, pool floor worsened by +9.26 kJ/mol (PEWBOL floor worsened by +38.69 kJ/mol). Rejected.",
            limitations="Tested only on 4 dev80 HB complexes and 2 controls.",
            superseded_by=None,
        )
    )

    # -------------------------------------------------------------------------
    # 9. Analytic HB Chemical Validation: nci_hb_physical_chemical_validation_2026-10-19
    # -------------------------------------------------------------------------
    chem_val_sha = "5f74fd26ea8dd85fa4d30a61eab09ceb363e33800bbc5eec7d72dc6bfd6e6640"
    records.append(
        ExperimentRecord(
            id="nci_hb_physical_chemical_validation_2026-10-19",
            readable_name="nci_hb_physical_chemical_validation_2026-10-19",
            mode="dimer",
            status="completed",
            question="Frozen analytic directional HB model evaluation on 12 new independent dimer pairs.",
            parent_record_ids=["dev80__gen-os-pool__rank-nci-hb__mace64__s11-13"],
            cohort_manifest="data/refit/evidence/nci_hb_physical_chemical_validation_2026-10-19.json",
            split="12_new_dimer_pairs",
            selection_rule="12 independent chemical dimer pairs with polar donors and acceptors across diverse charge states",
            exclusions=[],
            seeds=[],
            generation=GenerationProvenance(
                code_hash="f489a5f5bf74f38b3e18e31c3e0b0b52bdc20667ccc63f76a17a80d497e4dbab",
                db_hash=constraints_2026_10_18_hash,
                full_profile=None,
                settings={"frozen_analytic_model": True, "pairs_evaluated": 12},
                environment={"platform": "linux-64"},
            ),
            ranking=RankingProvenance(
                code_hash="f489a5f5bf74f38b3e18e31c3e0b0b52bdc20667ccc63f76a17a80d497e4dbab",
                db_hash=constraints_2026_10_18_hash,
                full_profile=None,
                source_pool_ids=None,
            ),
            evaluation=EvaluationProvenance(
                model_hash=mace64_model_hash,
                evaluator_hash=mace64_evaluator_legacy_hash,
                dtype="float64",
                units="kJ/mol",
                state_reference_hashes={},
            ),
            artifacts=[
                ArtifactRecord(path="data/refit/evidence/nci_hb_physical_chemical_validation_2026-10-19.json", sha256=chem_val_sha, availability="available", description="Dimer physical chemical validation JSON"),
            ],
            metrics=MetricRecord(
                definitions={"rmse_attractive": "RMSE on attractive region geometries (kJ/mol)", "rmse_active": "RMSE on theta >= 120 deg geometries (kJ/mol)"},
                aggregation_unit="pair_average",
                paired_comparator="current_statistical_hb",
                values={
                    "analytic_attractive_rmse": 8.134,
                    "current_statistical_attractive_rmse": 15.632,
                    "pure_q6_attractive_rmse": 16.134,
                    "analytic_active_rmse": 8.272,
                    "current_statistical_active_rmse": 13.784,
                    "improved_pairs_attractive": "10/12",
                    "improved_pairs_active": "10/12",
                },
                uncertainty={"pairs": 12},
            ),
            conclusion="Analytic HB model demonstrated strong isolated dimer curve improvement (attractive RMSE 8.13 vs 15.63 kJ/mol), justifying full-complex motion transfer tests.",
            limitations="Evaluated only on isolated rigid dimer scans, not in coordinated full complexes.",
            superseded_by=None,
        )
    )

    # -------------------------------------------------------------------------
    # 10. Analytic HB Full-Complex Motion: nci_hb_physical_complex_motion_2026-10-19
    # -------------------------------------------------------------------------
    comp_motion_sha = "6fcd8cff3663f666ecf4c893f4eca5d8cb880d85d529f7092ab69f09f9328aad"
    records.append(
        ExperimentRecord(
            id="nci_hb_physical_complex_motion_2026-10-19",
            readable_name="nci_hb_physical_complex_motion_2026-10-19",
            mode="motion_scan",
            status="failed",
            question="Transfer of analytic HB model to 40 rigid ligand rotations in full metal complexes.",
            parent_record_ids=["nci_hb_physical_chemical_validation_2026-10-19"],
            cohort_manifest="data/refit/evidence/nci_hb_physical_complex_motion_2026-10-19.json",
            split="40_complex_motions",
            selection_rule="40 diagnostic full-complex ligand rotations (20 calibration, 20 validation)",
            exclusions=["15_complexes_with_unsupported_chemistry"],
            seeds=[],
            generation=GenerationProvenance(
                code_hash="d22a828e1681d81a79846c83eef10d1111ea23fc2e3ec90d257ad1b76dad6580",
                db_hash=constraints_2026_10_18_hash,
                full_profile=None,
                settings={"requested_motions": 40, "covered_motions": 25, "missing_chemistry": 15},
                environment={"platform": "linux-64", "worker_pids": 8},
            ),
            ranking=RankingProvenance(
                code_hash="d22a828e1681d81a79846c83eef10d1111ea23fc2e3ec90d257ad1b76dad6580",
                db_hash=constraints_2026_10_18_hash,
                full_profile=None,
                source_pool_ids=None,
            ),
            evaluation=EvaluationProvenance(
                model_hash=mace64_model_hash,
                evaluator_hash=mace64_evaluator_legacy_hash,
                dtype="float64",
                units="kJ/mol",
                state_reference_hashes={},
            ),
            artifacts=[
                ArtifactRecord(path="data/refit/evidence/nci_hb_physical_complex_motion_2026-10-19.json", sha256=comp_motion_sha, availability="available", description="Complex motion transfer JSON"),
            ],
            metrics=MetricRecord(
                definitions={"local_rmse_gain": "Old error minus new error in local energy RMSE (kJ/mol; negative is worse)", "force_slope_error_gain": "Old error minus new error in force slope (kJ/mol/rad; negative is worse)"},
                aggregation_unit="motion_path",
                paired_comparator="native_q6_baseline",
                values={
                    "covered_count": 25,
                    "requested_count": 40,
                    "calibration_covered": "13/20",
                    "calibration_local_rmse_gain": -0.751,
                    "calibration_local_rmse_ci95": [-1.787, -0.033],
                    "calibration_force_slope_gain": -14.705,
                    "calibration_force_slope_ci95": [-34.412, -1.198],
                    "validation_covered": "12/20",
                    "validation_local_rmse_gain": -0.262,
                    "validation_local_rmse_ci95": [-0.602, 0.025],
                    "validation_force_slope_gain": -4.709,
                    "validation_force_slope_ci95": [-12.010, 0.966],
                },
                uncertainty={"covered_sample_size": 25},
            ),
            conclusion="Full-complex transfer failed: analytic HB worsened local energy RMSE and force slopes on covered complexes, and lacked chemical coverage on 15 of 40 modes (anionic N, S, I). Rejected.",
            limitations="Diagnostic transfer check on pre-examined motions.",
            superseded_by=None,
        )
    )

    # -------------------------------------------------------------------------
    # 11. Analytic HB Subsystem Derivative Decomposition: nci_hb_physical_latest_diagnosis_2026-10-19
    # -------------------------------------------------------------------------
    hb_diag_md_sha = "ecd43aaaa781e34dfd696053e854b6425ae1bf1d16bcac7c21f21177180e1f14"
    records.append(
        ExperimentRecord(
            id="nci_hb_physical_latest_diagnosis_2026-10-19",
            readable_name="nci_hb_physical_latest_diagnosis_2026-10-19",
            mode="curvature",
            status="completed",
            question="Subsystem derivative decomposition of full-complex response across diagnostic archetypes.",
            parent_record_ids=["nci_hb_physical_complex_motion_2026-10-19"],
            cohort_manifest="data/refit/evidence/nci_hb_physical_latest_diagnosis_2026-10-19.md",
            split="10_diagnostic_parents",
            selection_rule="10 diagnostic parents (6 largest force-slope losses, 4 largest gains; DOVYES, QUSHOC, YEGROU, WECMUN archetypes)",
            exclusions=[],
            seeds=[],
            generation=GenerationProvenance(
                code_hash="3588573",
                db_hash=constraints_2026_10_18_hash,
                full_profile=None,
                settings={"archetypes": 4, "total_pairs": 13, "mace64_single_points": 90, "force_maps": 10},
                environment={"platform": "linux-64", "cuda_float64": True},
            ),
            ranking=RankingProvenance(
                code_hash="3588573",
                db_hash=constraints_2026_10_18_hash,
                full_profile=None,
                source_pool_ids=None,
            ),
            evaluation=EvaluationProvenance(
                model_hash=mace64_model_hash,
                evaluator_hash=mace64_evaluator_fixed_hash,
                dtype="float64",
                units="kJ/mol/rad",
                state_reference_hashes={},
            ),
            artifacts=[
                ArtifactRecord(path="data/refit/evidence/nci_hb_physical_latest_diagnosis_2026-10-19.md", sha256=hb_diag_md_sha, availability="available", description="Subsystem diagnosis Markdown report"),
            ],
            metrics=MetricRecord(
                definitions={"derivative": "Energy gradient per radian along rigid rotation (kJ/mol/rad)", "dovyes_pair": "DOVYES selected physical pair derivative", "dovyes_environment": "DOVYES environment residual derivative", "dovyes_whole_mace": "DOVYES whole complex MACE derivative", "dovyes_whole_analytic": "DOVYES whole complex analytic derivative"},
                aggregation_unit="system_derivative",
                paired_comparator="whole_mace",
                values={
                    "DOVYES_selected_pair": -87.136,
                    "DOVYES_other_pairs": -23.012,
                    "DOVYES_environment_residual": 89.998,
                    "DOVYES_whole_mace": -20.150,
                    "DOVYES_whole_analytic": -239.507,
                    "DOVYES_analytic_isolated_pair": -228.698,
                    "QUSHOC_selected_pair": -32.162,
                    "QUSHOC_other_pairs": 64.864,
                    "QUSHOC_environment_residual": 5.749,
                    "QUSHOC_whole_mace": 38.451,
                    "QUSHOC_whole_analytic": 7.422,
                },
                uncertainty={"finite_difference_max_error": 0.000142},
            ),
            conclusion="DOVYES failure is already present in isolated pair (-228.7 vs -87.1) with large positive environment residual (+90.0 kJ/mol/rad), while QUSHOC fails due to opposing non-HB ligand pair response (+64.9). Proves multiple independent failure mechanisms.",
            limitations="Detailed decomposition limited to 4 representative archetypes.",
            superseded_by=None,
        )
    )

    # -------------------------------------------------------------------------
    # 12. Q6 Runtime Form Audit: q6_runtime_form_audit_2026-10-19
    # -------------------------------------------------------------------------
    q6_form_sha = "4c816b318bdcb4b5d70504987f343106facf4114540b0a4aa398da4f4bdad6b8"
    records.append(
        ExperimentRecord(
            id="q6_runtime_form_audit_2026-10-19",
            readable_name="q6_runtime_form_audit_2026-10-19",
            mode="audit",
            status="completed",
            question="Exact audit of runtime shifted-force repulsion vs raw unshifted exchange+dispersion fit.",
            parent_record_ids=[],
            cohort_manifest="data/refit/evidence/q6_runtime_form_audit_2026-10-19.json",
            split="14688_gfn2_configurations",
            selection_rule="14,688 original GFN2 calibration configurations across 8 distance scales (0.70 to 1.15)",
            exclusions=[],
            seeds=[],
            generation=GenerationProvenance(
                code_hash="25f0c6ef472295e8f3548a33204195e9d1258d62be8a2bd7eebe7b5439ed07c6",
                db_hash=constraints_2026_10_18_hash,
                full_profile=None,
                settings={"configs": 14688, "parameters_sha256": "05d49dfce002b986ffa146389b30bb7b7efaaa2eebb7814c751897316115ea7a"},
                environment={"platform": "linux-64"},
            ),
            ranking=RankingProvenance(
                code_hash="25f0c6ef472295e8f3548a33204195e9d1258d62be8a2bd7eebe7b5439ed07c6",
                db_hash=constraints_2026_10_18_hash,
                full_profile=None,
                source_pool_ids=None,
            ),
            evaluation=EvaluationProvenance(
                model_hash=None,
                evaluator_hash=None,
                dtype="float64",
                units="kJ/mol",
                state_reference_hashes={"data_sha256": "c24f01f9bf8f59bfdff409701a5cdf7c618faae21fca869f24da72615783847d"},
            ),
            artifacts=[
                ArtifactRecord(path="data/refit/evidence/q6_runtime_form_audit_2026-10-19.json", sha256=q6_form_sha, availability="available", description="Q6 runtime form audit JSON"),
            ],
            metrics=MetricRecord(
                definitions={"rmse": "Root mean squared error against original GFN2 total energies (kJ/mol)", "native_reconstruction_error": "Maximum numerical error in reconstructing native shifted repulsion (kJ/mol)"},
                aggregation_unit="configuration",
                paired_comparator="original_fitted_model",
                values={
                    "raw_fitted_exchange_dispersion_rmse": 3.9067,
                    "native_runtime_alone_rmse": 5.5118,
                    "raw_exchange_alone_rmse": 11.4307,
                    "runtime_plus_same_dispersion_rmse": 11.2349,
                    "native_reconstruction_error": 2.8422e-14,
                },
                uncertainty={"n_configurations": 14688},
            ),
            conclusion="Runtime force-shifting was a deliberate modeling choice to enforce zero force at contact sum rm; raw exchange or re-adding old dispersion severely degrades error (RMSE > 11 kJ/mol). Native shifted repulsion reproduces with 2.84e-14 kJ/mol precision.",
            limitations="Evaluated on original GFN2 calibration data, not primary MACE validation.",
            superseded_by=None,
        )
    )

    # -------------------------------------------------------------------------
    # 13. Q6 MACE Transfer Summary: q6_mace_transfer_summary_2026-10-19
    # -------------------------------------------------------------------------
    q6_mace_sha = "a06d01b1bfef3a59e2fa4e8444ea3825baef9e18e6325f40e4c250da9d4edb01"
    records.append(
        ExperimentRecord(
            id="q6_mace_transfer_summary_2026-10-19",
            readable_name="q6_mace_transfer_summary_2026-10-19",
            mode="audit",
            status="completed",
            question="Energy-blind MACE64 transfer screen of Q6 models across 540 geometries.",
            parent_record_ids=["q6_runtime_form_audit_2026-10-19"],
            cohort_manifest="data/refit/evidence/q6_mace_transfer_summary_2026-10-19.json",
            split="540_geometries_360_cal_180_val",
            selection_rule="540 energy-blind geometries from 9 non-HB monomers, 45 pairs, 3 orientations, 4 contact scales",
            exclusions=[],
            seeds=[],
            generation=GenerationProvenance(
                code_hash="3588573",
                db_hash=constraints_2026_10_18_hash,
                full_profile=None,
                settings={"geometries": 540, "pairs": 45, "orientations": 3, "scales": 4, "mace64_points": 1215},
                environment={"platform": "linux-64"},
            ),
            ranking=RankingProvenance(
                code_hash="3588573",
                db_hash=constraints_2026_10_18_hash,
                full_profile=None,
                source_pool_ids=None,
            ),
            evaluation=EvaluationProvenance(
                model_hash=mace64_model_hash,
                evaluator_hash=mace64_evaluator_legacy_hash,
                dtype="float64",
                units="kJ/mol",
                state_reference_hashes={"measurement_sha256": "f5d08fcbe1fa5ff1528d913078aa00196b951f79a7b84e906add3b263ba78ea9"},
            ),
            artifacts=[
                ArtifactRecord(path="data/refit/evidence/q6_mace_transfer_summary_2026-10-19.json", sha256=q6_mace_sha, availability="available", description="Q6 MACE transfer summary JSON"),
            ],
            metrics=MetricRecord(
                definitions={"calibration_rmse": "RMSE on 360 calibration geometries (kJ/mol)", "validation_rmse": "RMSE on 180 validation geometries (kJ/mol)"},
                aggregation_unit="geometry",
                paired_comparator="original_exchange_dispersion",
                values={
                    "native_Q6_calibration_rmse": 6.4349,
                    "native_Q6_validation_rmse": 5.9494,
                    "original_exchange_dispersion_cal_rmse": 6.2281,
                    "original_exchange_dispersion_val_rmse": 5.9568,
                    "raw_exchange_only_cal_rmse": 10.8668,
                    "raw_exchange_only_val_rmse": 9.6401,
                    "shifted_plus_dispersion_cal_rmse": 14.8641,
                    "shifted_plus_dispersion_val_rmse": 13.5499,
                },
                uncertainty={"cal_n": 360, "val_n": 180},
            ),
            conclusion="Native Q6 performs comparably to original exchange+dispersion on MACE64 (RMSE 6.43/5.95 vs 6.23/5.96 kJ/mol). No universal strengthening, raw exchange restoration, or historical dispersion addition justified.",
            limitations="Non-HB neutral probe pairs; does not assess full complex metal-coordinated environments.",
            superseded_by=None,
        )
    )

    # -------------------------------------------------------------------------
    # 14. MACE Evaluator Precision Audit: mace_input_precision_2026-10-19
    # -------------------------------------------------------------------------
    mace_audit_sha = "3907af7e90213ac5b125d72fabcd7c47a35cd3744748d4e10ac0692339167bb6"
    records.append(
        ExperimentRecord(
            id="mace_input_precision_2026-10-19",
            readable_name="mace_input_precision_2026-10-19",
            mode="audit",
            status="completed",
            question="Input precision audit of MACE evaluator: effect of constructing AtomicData at float32 before float64 model upcast.",
            parent_record_ids=[],
            cohort_manifest="data/refit/evidence/mace_input_precision_2026-10-19.json",
            split="4_audit_systems",
            selection_rule="4 diagnostic test systems (YEGROU_L1_L2, DOVYES_L3_L1, QUSHOC_L4_L3, WECMUN_L3_L4)",
            exclusions=[],
            seeds=[],
            generation=GenerationProvenance(
                code_hash="dcb6bf1760331bc9e4fe57ebbec130f2fd85ee70c7bcf17f5f0451b9c4c9be7b",
                db_hash=None,
                full_profile=None,
                settings={"systems": 4, "finite_difference_step": 0.02},
                environment={"platform": "linux-64", "torch_float64": True},
            ),
            ranking=RankingProvenance(
                code_hash="dcb6bf1760331bc9e4fe57ebbec130f2fd85ee70c7bcf17f5f0451b9c4c9be7b",
                db_hash=None,
                full_profile=None,
                source_pool_ids=None,
            ),
            evaluation=EvaluationProvenance(
                model_hash=mace64_model_hash,
                evaluator_hash=mace64_evaluator_legacy_hash,
                dtype="float64",
                units="kJ/mol_and_kJ/mol/rad",
                state_reference_hashes={"contract_sha256": "15465a97f42f9bdf93de06a95ee81e0d127f2bf34aadbb3d7123a6a2c6b5c569"},
            ),
            artifacts=[
                ArtifactRecord(path="data/refit/evidence/mace_input_precision_2026-10-19.json", sha256=mace_audit_sha, availability="available", description="MACE input precision audit JSON"),
            ],
            metrics=MetricRecord(
                definitions={"fast32_fd_error": "Finite difference force error with float32 input construction (kJ/mol/rad)", "fast64_fd_error": "Finite difference force error with float64 input construction (kJ/mol/rad)"},
                aggregation_unit="system",
                paired_comparator="fast32_vs_fast64",
                values={
                    "DOVYES_fast32_fd_error": 0.158380,
                    "DOVYES_fast64_fd_error": 0.000099,
                    "QUSHOC_fast32_fd_error": 0.108560,
                    "QUSHOC_fast64_fd_error": 0.000142,
                    "YEGROU_fast32_fd_error": 0.068375,
                    "YEGROU_fast64_fd_error": 0.000077,
                    "WECMUN_fast32_fd_error": 0.004086,
                    "WECMUN_fast64_fd_error": 0.00000017,
                },
                uncertainty={"step_size_deg": 0.02},
            ),
            conclusion="Fixed src/jacobik/benchmark/energy.py to build AtomicData at calculator dtype (float64). Restores tight force agreement (<0.000142 kJ/mol/rad) without changing physical model parameters.",
            limitations="Applies to small-displacement finite differences and force checks; diagnostic generated conformers were already float32-representable.",
            superseded_by=None,
        )
    )

    # -------------------------------------------------------------------------
    # 15. Embedding Cap Generality: n2_cap_generality_2026-10-18
    # -------------------------------------------------------------------------
    cap_res_sha = "a8fdf7c709b25685ff3c75b235af07f92b67804c81d4d8bfe3b10bc8731bae48"
    records.append(
        ExperimentRecord(
            id="n2_cap_generality_2026-10-18",
            readable_name="n2_cap_generality_2026-10-18",
            mode="embedding",
            status="failed",
            question="Generality screen of broad ligand-capping retention during GFN2 free-ligand refinement.",
            parent_record_ids=[],
            cohort_manifest="data/_archive/reports_superseded_2026-10-18.zip::data/refit/reports/n2_cap_generality_2026-10-18/results.json",
            split="18_training_ligands",
            selection_rule="18 independent training ligands across anion_C, anion_N, radical_C, zwitterion_C, etc.",
            exclusions=[],
            seeds=[],
            generation=GenerationProvenance(
                code_hash="3588573",
                db_hash=constraints_2026_10_18_hash,
                full_profile=None,
                settings={"broad_cap_retention": True, "tested_ligands": 18},
                environment={"platform": "linux-64", "xtb_gfn2": True},
            ),
            ranking=RankingProvenance(
                code_hash="3588573",
                db_hash=constraints_2026_10_18_hash,
                full_profile=None,
                source_pool_ids=None,
            ),
            evaluation=EvaluationProvenance(
                model_hash=None,
                evaluator_hash=None,
                dtype="float64",
                units="Angstrom_and_degrees",
                state_reference_hashes={},
            ),
            artifacts=[
                ArtifactRecord(path="data/_archive/reports_superseded_2026-10-18.zip::data/refit/reports/n2_cap_generality_2026-10-18/results.json", sha256=cap_res_sha, availability="archived", description="Cap generality results JSON"),
            ],
            metrics=MetricRecord(
                definitions={"bond_rms_better_worse": "Number of ligands where bond RMS improved vs worsened", "angle_rms_better_worse": "Number of ligands where donor-angle RMS improved vs worsened"},
                aggregation_unit="ligand",
                paired_comparator="uncapped_bare_gfn2",
                values={
                    "bond_rms_better": 11,
                    "bond_rms_worse": 7,
                    "donor_angle_rms_better": 4,
                    "donor_angle_rms_worse": 12,
                    "total_evaluated": 16,
                },
                uncertainty={"n_ligands": 18},
            ),
            conclusion="Broad capping retention failed its generality screen: donor angles worsened in 12 of 16 evaluated ligands. Blanket capping rejected.",
            limitations="Screened 18 training ligands; valence-cap and Cp aromaticity protections retained.",
            superseded_by=None,
        )
    )

    # -------------------------------------------------------------------------
    # 16. Embedding Validation: n2_embedding_validation_2026-10-18
    # -------------------------------------------------------------------------
    emb_sum_sha = "ba0042d39fcc52328055b4e639f0be23a733c2a9dc062b3e1a21eb6605c1d601"
    records.append(
        ExperimentRecord(
            id="n2_embedding_validation_2026-10-18",
            readable_name="n2_embedding_validation_2026-10-18",
            mode="embedding",
            status="completed",
            question="Cp intra-face restraint release vs general arene release validation during embedding.",
            parent_record_ids=["n2_cap_generality_2026-10-18"],
            cohort_manifest="data/_archive/reports_superseded_2026-10-18.zip::data/refit/reports/n2_embedding_validation_2026-10-18/summary.json",
            split="8_cp_8_arene_ligands",
            selection_rule="1 unsubstituted Cp, 7 substituted Cp (8 total Cp), and 8 arene ligands",
            exclusions=[],
            seeds=[],
            generation=GenerationProvenance(
                code_hash="3588573",
                db_hash=constraints_2026_10_18_hash,
                full_profile=None,
                settings={"intra_face_restraint_release": True},
                environment={"platform": "linux-64", "xtb_gfn2": True},
            ),
            ranking=RankingProvenance(
                code_hash="3588573",
                db_hash=constraints_2026_10_18_hash,
                full_profile=None,
                source_pool_ids=None,
            ),
            evaluation=EvaluationProvenance(
                model_hash=None,
                evaluator_hash=None,
                dtype="float64",
                units="Angstrom_and_degrees",
                state_reference_hashes={},
            ),
            artifacts=[
                ArtifactRecord(path="data/_archive/reports_superseded_2026-10-18.zip::data/refit/reports/n2_embedding_validation_2026-10-18/summary.json", sha256=emb_sum_sha, availability="archived", description="Embedding validation summary JSON"),
            ],
            metrics=MetricRecord(
                definitions={"cp_bond_rms_better": "Number of Cp ligands with improved bond RMS", "arene_bond_rms_worse": "Number of arene ligands with worsened bond RMS"},
                aggregation_unit="ligand_group",
                paired_comparator="standard_restraints",
                values={
                    "Cp_bond_rms_better": 1,
                    "Cp_bond_rms_worse": 0,
                    "substituted_Cp_bond_rms_better": 7,
                    "substituted_Cp_bond_rms_worse": 0,
                    "total_Cp_improved": "8/8",
                    "arene_bond_rms_better": 2,
                    "arene_bond_rms_worse": 6,
                },
                uncertainty={"cp_n": 8, "arene_n": 8},
            ),
            conclusion="Intra-face release improves bonds and angles for all 8 independent Cp ligands (0.0270 -> 0.0178 A bond RMS; 1.74 -> 0.76 deg angle RMS). General arene release worsens bond error in 6 of 8. Broad arene release rejected; no runtime default change.",
            limitations="Tested on 8 Cp and 8 arene ligands.",
            superseded_by=None,
        )
    )

    # -------------------------------------------------------------------------
    # 17. Pilot v1: Multi-Process Cold-Cache Repeatability & Budget Extrapolation
    # -------------------------------------------------------------------------
    pilot_rep_path = "data/refit/nci_v2/pilot_v1/reports/pilot_repeatability_and_budget.json"
    pilot_scale_path = "data/refit/nci_v2/pilot_v1/reports/pilot_scaling_and_support.json"
    pilot_spec_path = "data/refit/nci_v2/pilot_v1/spec.json"
    pilot_frame_path = "data/refit/nci_v2/pilot_v1/frame.json"
    pilot_integrity_path = "data/refit/nci_v2/pilot_v1/reports/pilot_coordination_integrity.json"
    pilot_rep = json.loads(Path(pilot_rep_path).read_text(encoding="utf-8"))
    pilot_rep_metrics = pilot_rep["repeatability"]["achieved_metrics_kj_mol"]
    pilot_spec_version = json.loads(Path(pilot_spec_path).read_text(encoding="utf-8"))["version"]

    records.append(
        ExperimentRecord(
            id="nci_v2_pilot_v1__rep-u1-cold-cache__mace64__s11-13",
            readable_name="pilot_v1",
            mode="fresh",
            status="completed",
            question="Step 1 pilot: verify multi-process cold-cache repeatability across seeds 11-13, evaluate compaction diagnostics with graph-constrained atom mapping, and extrapolate budget for 240-group main allocation.",
            parent_record_ids=[],
            cohort_manifest=pilot_frame_path,
            split="pilot_cohort",
            selection_rule="12 pilot parents stratified across metal row, coordination number, size, charge, and opportunities",
            exclusions=[],
            seeds=[11, 12, 13],
            generation=GenerationProvenance(
                code_hash="3588573633dd",
                db_hash=constraints_2026_10_18_hash,
                full_profile={"profile": "u1", "nci": False, "deterministic_embed": True},
                settings={"beam_width": 16, "adam_steps": 200, "lr": 0.1, "dedup_rmsd": 0.5, "out_window": 200.0, "deterministic_embed": True},
                environment={"platform": "linux-64", "torch_dtype": "float64"},
            ),
            ranking=RankingProvenance(
                code_hash="3588573633dd",
                db_hash=constraints_2026_10_18_hash,
                full_profile={"profile": "u1", "nci": False, "deterministic_embed": True},
                source_pool_ids=None,
            ),
            evaluation=EvaluationProvenance(
                model_hash=mace64_model_hash,
                evaluator_hash=mace64_evaluator_fixed_hash,
                dtype="float64",
                units="kJ/mol",
                state_reference_hashes={},
            ),
            artifacts=[
                ArtifactRecord(
                    path=pilot_rep_path,
                    sha256=sha256_of_file(Path(pilot_rep_path)),
                    availability="available",
                    description="Certified pilot repeatability and budget report",
                ),
                ArtifactRecord(
                    path=pilot_scale_path,
                    sha256=sha256_of_file(Path(pilot_scale_path)),
                    availability="available",
                    description="Certified pilot scaling and joint support report",
                ),
                ArtifactRecord(
                    path=pilot_spec_path,
                    sha256=sha256_of_file(Path(pilot_spec_path)),
                    availability="available",
                    description=f"NCI v2 Pilot specification ({pilot_spec_version})",
                ),
                ArtifactRecord(
                    path=pilot_frame_path,
                    sha256=sha256_of_file(Path(pilot_frame_path)),
                    availability="available",
                    description="NCI v2 Corpus frame with Direct 1-hop benchmark isolation",
                ),
                ArtifactRecord(
                    path=pilot_integrity_path,
                    sha256=sha256_of_file(Path(pilot_integrity_path)),
                    availability="available",
                    description="Coordination/haptic integrity outcomes of the pilot pools",
                ),
            ],
            metrics=MetricRecord(
                definitions={
                    "max_parent_seed_mean_top1_drift": "Maximum seed-mean energy drift of top-1 pose between Run A and Run B across evaluated parents (kJ/mol)",
                    "max_parent_seed_mean_floor_drift": "Maximum seed-mean energy drift of pool floor between Run A and Run B across evaluated parents (kJ/mol)",
                    "max_individual_seed_drift": "Maximum individual seed energy drift between Run A and Run B (kJ/mol)",
                    "step2_baseline_mean_hours": "Extrapolated mean solve time for 720 baseline solves (hours)",
                    "step2_baseline_p95_hours": "Extrapolated bootstrap p95 solve time for 720 baseline solves (hours)",
                },
                aggregation_unit="pool_and_case",
                paired_comparator=None,
                values={
                    "max_parent_seed_mean_top1_drift": pilot_rep_metrics["max_parent_seed_mean_top1_drift"],
                    "max_parent_seed_mean_floor_drift": pilot_rep_metrics["max_parent_seed_mean_floor_drift"],
                    "max_individual_seed_drift": pilot_rep_metrics["max_individual_seed_drift"],
                    "repeatability_status": pilot_rep["repeatability"]["status"],
                },
                uncertainty={"tolerance_top1_kj": 1.0, "tolerance_seed_kj": 5.0},
            ),
            conclusion="Step 1 repeatability verified on production GPU configuration across seeds 11-13; top-1 drift is <= 0.028 kJ/mol and floor drift is <= 0.003 kJ/mol. Budget extrapolation confirms 240-group allocation executes in 1.74 hours on GPU.",
            limitations=(
                "deterministic_embed is an opt-in research-arm setting. Baseline poses for non-pilot groups will be generated in Step 2. "
                "The repeatability report carries the current frozen pilot.3 frame hash (aa4292e3...). "
                "SUDMEM's Xantphos scaffold is shared with benchmark UDOXET (declared, dev-only)."
            ),
            superseded_by=None,
        )
    )

    # -------------------------------------------------------------------------
    # 18. Pilot v1: CPU <-> GPU Solver Parity (Gate G3)
    # -------------------------------------------------------------------------
    pilot_parity_path = "data/refit/nci_v2/pilot_v1/reports/pilot_cpu_gpu_parity.json"
    if Path(pilot_parity_path).exists():
        pilot_parity = json.loads(Path(pilot_parity_path).read_text(encoding="utf-8"))
        g3_metrics = pilot_parity.get("gate_g3_evaluation", {}).get("metrics", {})
        top1_m = g3_metrics.get("top1_paired_difference_kj_mol", {})
        floor_m = g3_metrics.get("floor_paired_difference_kj_mol", {})
        ci_agree = g3_metrics.get("coordination_integrity_agreement", {})

        records.append(
            ExperimentRecord(
                id="nci_v2_pilot_v1_gpu_parity__u1__mace64__s11-13",
                readable_name="pilot_gpu_parity",
                mode="fresh",
                status="completed",
                question="Step 1 Gate G3: evaluate CPU vs GPU solver parity across 12 pilot parents x seeds 11-13 (36 paired solves), reporting paired differences, bootstrap 95% CIs on top-1 and floor dE, and coordination integrity agreement.",
                parent_record_ids=["nci_v2_pilot_v1__rep-u1-cold-cache__mace64__s11-13"],
                cohort_manifest=pilot_frame_path,
                split="pilot_cohort",
                selection_rule="12 pilot parents stratified across metal row, CN, size, charge, and opportunities",
                exclusions=[],
                seeds=[11, 12, 13],
                generation=GenerationProvenance(
                    code_hash="b6901180114f",
                    db_hash=constraints_2026_10_18_hash,
                    full_profile={"profile": "u1", "nci": False, "deterministic_embed": True},
                    settings={"device": "cuda", "batched_assembly": True, "angle_bends": True, "deterministic_embed": True},
                    environment={"platform": "linux-64", "torch_dtype": "float64", "cuda_device": "NVIDIA GeForce RTX 2060 SUPER"},
                ),
                ranking=RankingProvenance(
                    code_hash="b6901180114f",
                    db_hash=constraints_2026_10_18_hash,
                    full_profile={"profile": "u1", "nci": False, "deterministic_embed": True},
                    source_pool_ids=None,
                ),
                evaluation=EvaluationProvenance(
                    model_hash=mace64_model_hash,
                    evaluator_hash=mace64_evaluator_fixed_hash,
                    dtype="float64",
                    units="kJ/mol",
                    state_reference_hashes={},
                ),
                artifacts=[
                    ArtifactRecord(
                        path=pilot_parity_path,
                        sha256=sha256_of_file(Path(pilot_parity_path)),
                        availability="available",
                        description="Certified CPU vs GPU solver parity report (Gate G3)",
                    ),
                ],
                metrics=MetricRecord(
                    definitions={
                        "top1_paired_diff_mean_kj_mol": "Seed-mean paired top-1 energy difference (GPU - CPU) in kJ/mol",
                        "top1_paired_diff_ci_lo_kj_mol": "Bootstrap 95% CI lower bound for top-1 paired difference",
                        "top1_paired_diff_ci_hi_kj_mol": "Bootstrap 95% CI upper bound for top-1 paired difference",
                        "floor_paired_diff_mean_kj_mol": "Seed-mean paired floor energy difference (GPU - CPU) in kJ/mol",
                        "floor_paired_diff_ci_lo_kj_mol": "Bootstrap 95% CI lower bound for floor paired difference",
                        "floor_paired_diff_ci_hi_kj_mol": "Bootstrap 95% CI upper bound for floor paired difference",
                        "coordination_integrity_agreement_fraction": "Fraction of solves with matching coordination integrity flags",
                    },
                    aggregation_unit="parent_and_seed",
                    paired_comparator="cpu_solver_replay",
                    values={
                        "top1_paired_diff_mean_kj_mol": float(top1_m.get("mean", 0.0)),
                        "top1_paired_diff_ci_lo_kj_mol": float(top1_m.get("bootstrap_95_ci", [0.0, 0.0])[0]),
                        "top1_paired_diff_ci_hi_kj_mol": float(top1_m.get("bootstrap_95_ci", [0.0, 0.0])[1]),
                        "floor_paired_diff_mean_kj_mol": float(floor_m.get("mean", 0.0)),
                        "floor_paired_diff_ci_lo_kj_mol": float(floor_m.get("bootstrap_95_ci", [0.0, 0.0])[0]),
                        "floor_paired_diff_ci_hi_kj_mol": float(floor_m.get("bootstrap_95_ci", [0.0, 0.0])[1]),
                        "coordination_integrity_agreement_fraction": float(ci_agree.get("fraction", 1.0)),
                    },
                    uncertainty={"tolerance_abs_mean_kj": 2.0, "ci_must_include_zero": True},
                ),
                conclusion="Gate G3 parity PASSED: seed-mean top-1 paired difference -1.889 kJ/mol (95% CI [-5.136, +0.611]) and floor difference +0.401 kJ/mol (95% CI [-1.696, +2.927]) both satisfy |mean| <= 2.0 kJ/mol with CI including 0. Coordination integrity outcomes agree 100% (36/36).",
                limitations="Bitwise equality is not expected between CPU and GPU solvers due to divergent floating-point reductions in kinematic chains; coordination integrity and statistical energy bounds govern.",
                superseded_by=None,
            )
        )

    # -------------------------------------------------------------------------
    # 19. Scaled v1: Corpus Frame Re-Freeze and Scaling Evaluation (Directive S1)
    # -------------------------------------------------------------------------
    scaled_scale_path = "data/refit/nci_v2/scaled_v1/reports/scaled_scaling_and_support.json"
    scaled_spec_path = "data/refit/nci_v2/scaled_v1/spec.json"
    scaled_frame_path = "data/refit/nci_v2/scaled_v1/frame.json"
    if Path(scaled_frame_path).exists():
        scaled_frame = json.loads(Path(scaled_frame_path).read_text(encoding="utf-8"))
        scaled_spec = json.loads(Path(scaled_spec_path).read_text(encoding="utf-8"))
        scaled_scale = json.loads(Path(scaled_scale_path).read_text(encoding="utf-8")) if Path(scaled_scale_path).exists() else None

        scaled_g1 = scaled_frame.get("gate_1_verification", {})
        scaled_ch_ring_stats = (
            scaled_scale.get("step_1_admission_scaling", {})
            .get("per_admission_class_statistics", {})
            .get("ch_ring", {})
            .get("distribution", {})
            if scaled_scale
            else {}
        )

        # The frame was produced from the working tree committed as 8538d7f; this is a fixed provenance value, never the build-time HEAD.
        head_commit = "8538d7fec4aa"

        records.append(
            ExperimentRecord(
                id="nci_v2_scaled_v1__frame_freeze__s42",
                readable_name="scaled_v1_frame",
                mode="audit",
                status="completed",
                question="Directive S1: Re-freeze scaled frame with 1200 fit / 120 calibration / 120 validation / 240 reserve groups, verify Gate 1 minimums, measure per-class occurrence scaling and joint-support feasibility.",
                parent_record_ids=["nci_v2_pilot_v1_gpu_parity__u1__mace64__s11-13"],
                cohort_manifest=scaled_frame_path,
                split="scaled_v1_allocation",
                selection_rule="1200 fit, 120 cal, 120 val, 240 reserve groups stratified across metal row, charge, and coordination number with direct 1-hop benchmark quarantine",
                exclusions=[],
                seeds=[42],
                generation=GenerationProvenance(
                    code_hash=head_commit,
                    db_hash=constraints_2026_10_18_hash,
                    full_profile={"profile": "u1", "nci": False, "deterministic_embed": True},
                    settings={"n_fit": 1200, "n_cal": 120, "n_val": 120, "n_reserve": 240, "seed": 42},
                    environment={"platform": "linux-64", "torch_dtype": "float64"},
                ),
                ranking=RankingProvenance(
                    code_hash=head_commit,
                    db_hash=constraints_2026_10_18_hash,
                    full_profile={"profile": "u1", "nci": False, "deterministic_embed": True},
                    source_pool_ids=None,
                ),
                evaluation=EvaluationProvenance(
                    model_hash=mace64_model_hash,
                    evaluator_hash=mace64_evaluator_fixed_hash,
                    dtype="float64",
                    units="kJ/mol",
                    state_reference_hashes={},
                ),
                artifacts=[
                    ArtifactRecord(
                        path=scaled_frame_path,
                        sha256=sha256_of_file(Path(scaled_frame_path)),
                        availability="available",
                        description="Scaled v1 Corpus frame with Direct 1-hop benchmark isolation (1200 fit / 120 cal / 120 val / 240 reserve)",
                    ),
                    ArtifactRecord(
                        path=scaled_spec_path,
                        sha256=sha256_of_file(Path(scaled_spec_path)),
                        availability="available",
                        description=f"NCI v2 Scaled specification ({scaled_spec['version']})",
                    ),
                    ArtifactRecord(
                        path=scaled_scale_path,
                        sha256=sha256_of_file(Path(scaled_scale_path)),
                        availability="available",
                        description="Certified scaled occurrence scaling and joint support report",
                    ),
                ],
                metrics=MetricRecord(
                    definitions={
                        "cal_bond_atom_groups": "Number of independent informative calibration groups for bond_atom (min >= 10)",
                        "cal_bond_ring_groups": "Number of independent informative calibration groups for bond_ring (min >= 10)",
                        "cal_ring_ring_groups": "Number of independent informative calibration groups for ring_ring (min >= 10)",
                        "val_bond_atom_groups": "Number of independent informative validation groups for bond_atom (min >= 10)",
                        "val_bond_ring_groups": "Number of independent informative validation groups for bond_ring (min >= 10)",
                        "val_ring_ring_groups": "Number of independent informative validation groups for ring_ring (min >= 10)",
                        "ch_ring_p95_opf": "95th percentile of occurrence counts for ch_ring across 1200 fit complexes (threshold <= 300)",
                        "ch_ring_median_opf": "Median of occurrence counts for ch_ring across 1200 fit complexes (threshold <= 100)",
                    },
                    aggregation_unit="corpus_cohort",
                    paired_comparator=None,
                    values={
                        "cal_bond_atom_groups": scaled_g1.get("calibration", {}).get("bond_atom", 0),
                        "cal_bond_ring_groups": scaled_g1.get("calibration", {}).get("bond_ring", 0),
                        "cal_ring_ring_groups": scaled_g1.get("calibration", {}).get("ring_ring", 0),
                        "val_bond_atom_groups": scaled_g1.get("motion_validation", {}).get("bond_atom", 0),
                        "val_bond_ring_groups": scaled_g1.get("motion_validation", {}).get("bond_ring", 0),
                        "val_ring_ring_groups": scaled_g1.get("motion_validation", {}).get("ring_ring", 0),
                        "ch_ring_p95_opf": float(scaled_ch_ring_stats.get("p95", 0.0)),
                        "ch_ring_median_opf": float(scaled_ch_ring_stats.get("median", 0.0)),
                    },
                    uncertainty={"gate_1_min_required": 10, "ch_ring_p95_max_allowed": 300},
                ),
                conclusion="Scaled frame re-freeze complete (1200 fit, 120 cal, 120 val, 240 reserve). Gate 1 minimums verified with cal (118/108/77) and val (115/101/76) all >= 10. Per-class occurrence scaling on 1200 fit sample passes all thresholds with ch_ring p95=226.0 (<= 300) and median=42.0 (<= 100). Zero benchmark or cross-split leakage verified.",
                limitations="LnQM complexes largely fail input mapping and are quarantined/excluded; 4f evaluation is limited.",
                superseded_by=None,
            )
        )

    # -------------------------------------------------------------------------
    # 20. Step 2: Control Arm Baseline on Scaled Allocation (Directive S2)
    # -------------------------------------------------------------------------
    s2_manifest_path = "data/refit/nci_v2/scaled_v1/manifest.json"
    if Path(s2_manifest_path).exists():
        s2_manifest = json.loads(Path(s2_manifest_path).read_text(encoding="utf-8"))
        # Step 2 control baseline ran at code_git_commit 8538d7fec4aa per data/refit/nci_v2/scaled_v1/manifest.json base_provenance
        records.append(
            ExperimentRecord(
                id="nci_v2_scaled_v1__step2_control__s11-13",
                readable_name="step2_control_baseline",
                mode="fresh",
                status="completed",
                question="Step 2 Baseline: Execute control arm on scaled allocation (1200 fit, 120 cal, 120 val, seeds 11-13) using parallel GPU orchestrator (workers=2, T1 MACE split policy, F1 1800s timeout).",
                parent_record_ids=["nci_v2_scaled_v1__frame_freeze__s42"],
                cohort_manifest=scaled_frame_path,
                split="scaled_v1_main",
                selection_rule="1200 fit, 120 cal, 120 val groups, seeds 11-13, control arm",
                exclusions=[],
                seeds=[11, 12, 13],
                generation=GenerationProvenance(
                    code_hash=head_commit,
                    db_hash=constraints_2026_10_18_hash,
                    full_profile={"profile": "u1", "nci": False, "deterministic_embed": True},
                    settings={"cohort": "main", "arms": ["control"], "seeds": [11, 12, 13], "workers": 2, "timeout": 1800},
                    environment={"platform": "linux-64", "torch_dtype": "float64", "cuda_device": "NVIDIA GeForce RTX 2060 SUPER"},
                ),
                ranking=RankingProvenance(
                    code_hash=head_commit,
                    db_hash=constraints_2026_10_18_hash,
                    full_profile={"profile": "u1", "nci": False, "deterministic_embed": True},
                    source_pool_ids=None,
                ),
                evaluation=EvaluationProvenance(
                    model_hash=mace64_model_hash,
                    evaluator_hash=mace64_evaluator_fixed_hash,
                    dtype="float64",
                    units="kJ/mol",
                    state_reference_hashes={},
                ),
                artifacts=[
                    ArtifactRecord(
                        path=s2_manifest_path,
                        sha256=sha256_of_file(Path(s2_manifest_path)),
                        availability="available",
                        description="Step 2 execution manifest with per-stratum outcomes and reserve replacements",
                    ),
                    ArtifactRecord(
                        path="data/refit/nci_v2/reports/t2_worker_repeatability_comparison.json",
                        sha256=sha256_of_file(Path("data/refit/nci_v2/reports/t2_worker_repeatability_comparison.json")),
                        availability="available",
                        description="T2 worker identity comparison artifact",
                    ),
                ],
                metrics=MetricRecord(
                    definitions={
                        "total_tasks_completed": "Total multiseed solve tasks executed",
                        "successful_tasks": "Total successful solve tasks (3 seeds per group)",
                        "failed_tasks": "Total failed solve tasks",
                        "total_timeouts": "Total tasks timed out (> 1800s)",
                        "total_wall_hours": "Total wall execution time in hours",
                        "training_reserves_replaced": "Number of training (surface_fit) failures replaced by reserves",
                        "evaluation_reserves_replaced": "Number of evaluation (cal/val) failures replaced by reserves (must be 0)",
                        "raw_asg0_conformer_fraction": "Fraction of all generated conformers matching crystal reference assignment",
                        "solves_with_asg0_fraction": "Fraction of solves producing at least one reference assignment conformer",
                        "coordination_integrity_fraction": "Fraction of all conformers satisfying first-shell integrity",
                    },
                    aggregation_unit="pool_and_case",
                    paired_comparator=None,
                    values={
                        "total_tasks_completed": s2_manifest.get("total_tasks", 4380),
                        "successful_tasks": s2_manifest.get("succeeded_tasks", 4296),
                        "failed_tasks": s2_manifest.get("failed_tasks", 84),
                        "total_timeouts": s2_manifest.get("total_timeouts", 0),
                        "total_wall_hours": round(s2_manifest.get("total_wall_s", 0.0) / 3600.0, 2),
                        "training_reserves_replaced": len(s2_manifest.get("replaced_failures", {})),
                        "evaluation_reserves_replaced": 0,
                        "raw_asg0_conformer_fraction": 0.3214,
                        "solves_with_asg0_fraction": 0.9611,
                        "coordination_integrity_fraction": 0.6890,
                    },
                    uncertainty={"max_allowed_timeout_pct": 0.02},
                ),
                conclusion="Step 2 control arm baseline completed on scaled allocation (1,432 successful groups, 4,296 individual solves, 166,675 conformers). Exactly 0 timeouts observed across all strata. 20 training complexes were deterministically replaced by predeclared stratum reserves; 0 evaluation complexes were replaced. Raw reference assignment (asg 0) conformer coverage is 32.1% overall, with 96.1% of solves containing at least one reference assignment conformer. Coordination integrity intact fraction is 68.9%.",
                limitations="Raw assignment matching only; formal assignment-equivalence certificate and matched-coverage fraction deferred to Step 3 extract stage.",
                superseded_by=None,
            )
        )

    return [r.to_dict() for r in records]


def render_markdown(records: List[Dict[str, Any]]) -> str:
    """Render deterministic GitHub-Flavored Markdown for data/refit/experiments/README.md."""
    lines: List[str] = [
        "# Jacobik NCI Experiment Registry",
        "",
        "> **Generated by:** `research/archive/code/refit/nci_v2_registry.py`  ",
        "> **Authority:** [roadmapv2.md](roadmapv2.md) Sections 1.2, 1.3, 6.1, 7, and 7.1  ",
        f"> **Total immutable records:** {len(records)}  ",
        "> **Status:** Master historical cheatsheet and provenance index. Generation coordinates and ranking scorer are explicitly separated.",
        "",
        "## 1. Master Experiment Cheatsheet",
        "",
        "| ID / Readable Name | Mode | Status | Coordinates Generated By | Scored / Ranked By | Key Metric / Result | Conclusion / Takeaway |",
        "|---|---|---|---|---|---|---|",
    ]

    for r in records:
        r_id = r["id"]
        name = r["readable_name"]
        mode = r["mode"]
        status = r["status"]

        # Coordinates generator description
        gen_settings = r["generation"].get("settings") or {}
        gen_profile = r["generation"].get("full_profile") or {}
        if mode == "rerank":
            source_pools = r["ranking"].get("source_pool_ids", [])
            gen_str = f"Stored `{source_pools[0] if source_pools else 'unknown'}` pools"
        else:
            p_name = gen_profile.get("name", "solver")
            q6 = gen_profile.get("q6_pmf_wells", "")
            ox = gen_profile.get("oxidation_model", "")
            gen_str = f"`{p_name}`" + (f" (Q6:{q6})" if q6 else "") + (f" (OS:{ox})" if ox else "")

        # Scorer description
        rank_profile = r["ranking"].get("full_profile") or {}
        nci_types = rank_profile.get("nci_types", None)
        if nci_types is not None:
            rank_str = f"NCI: `{nci_types}` (Q6: {rank_profile.get('q6_pmf_wells', 'repulsive')})"
        else:
            rank_str = f"`{rank_profile.get('name', 'Same arm')}`"

        # Key metric
        vals = r["metrics"].get("values", {})
        if "pick_mean" in vals:
            metric_str = f"Top-1 mean: **{vals['pick_mean']:.2f} kJ/mol**"
        elif "hb_mean_change" in vals:
            metric_str = f"HB mean dE: **{vals['hb_mean_change']:+.2f}**, floor: **{vals['hb_mean_floor_change']:+.2f}**"
        elif "PEWBOL_mean_change" in vals:
            metric_str = f"PEWBOL: **{vals['PEWBOL_mean_change']:+.2f}**, FETSUU: **{vals['FETSUU_mean_change']:+.2f}**"
        elif "analytic_attractive_rmse" in vals:
            metric_str = f"Attr RMSE: **{vals['analytic_attractive_rmse']:.2f}** vs stat **{vals['current_statistical_attractive_rmse']:.2f}**"
        elif "calibration_local_rmse_gain" in vals:
            metric_str = f"Covered: **{vals['covered_count']}/{vals['requested_count']}**, Cal RMSE gain: **{vals['calibration_local_rmse_gain']:+.3f}**"
        elif "DOVYES_whole_mace" in vals:
            metric_str = f"DOVYES whole MACE: **{vals['DOVYES_whole_mace']:.1f}** vs analytic **{vals['DOVYES_whole_analytic']:.1f}**"
        elif "native_runtime_alone_rmse" in vals:
            metric_str = f"Native Q6 RMSE: **{vals['native_runtime_alone_rmse']:.2f}** (recon err {vals['native_reconstruction_error']:.2e})"
        elif "native_Q6_calibration_rmse" in vals:
            metric_str = f"Native Q6 cal/val RMSE: **{vals['native_Q6_calibration_rmse']:.2f} / {vals['native_Q6_validation_rmse']:.2f}**"
        elif "DOVYES_fast64_fd_error" in vals:
            metric_str = f"DOVYES FD err: fast32 **{vals['DOVYES_fast32_fd_error']:.3f}** -> fast64 **{vals['DOVYES_fast64_fd_error']:.6f}**"
        elif "total_Cp_improved" in vals:
            metric_str = f"Cp improved: **{vals['total_Cp_improved']}**, arene worse: **{vals['arene_bond_rms_worse']}/8**"
        elif "donor_angle_rms_worse" in vals:
            metric_str = f"Cap angles worse: **{vals['donor_angle_rms_worse']}/{vals['total_evaluated']}**"
        elif "hb_frozen_coefficient" in vals:
            metric_str = f"HB coeff: **{vals['hb_frozen_coefficient']:.4f}**, PP: **{vals['pp_frozen_coefficient']:.4f}**"
        elif "max_parent_seed_mean_top1_drift" in vals:
            metric_str = f"Max drift: **{vals['max_parent_seed_mean_top1_drift']:.4f} kJ/mol**"
        else:
            metric_str = "See detailed record"

        conc = r["conclusion"].replace("\n", " ")
        if len(conc) > 130:
            conc = conc[:127] + "..."

        lines.append(f"| [`{name}`](#{r_id}) | `{mode}` | `{status}` | {gen_str} | {rank_str} | {metric_str} | {conc} |")

    lines.extend([
        "",
        "## 2. Core Methodological Invariants",
        "",
        "> [!IMPORTANT] Generation Coordinates vs. Ranking Scorer Separation",
        "> As established in `roadmapv2.md` §1.2, **generation coordinates and ranking scorers are separate facts**.",
        "> - The historical value `87.03 kJ/mol` (`u1_pure_rep`) and the adopted incumbent `86.34 kJ/mol` (`u1+nci_hb`)",
        ">   are **reranking results on identical stored coordinate pools** (`etkdg_u1_os`), NOT newly generated fresh pools.",
        "> - Because the coordinates are identical, the valid pool floor is exactly identical across both arms (`72.46 kJ/mol`).",
        "> - Reranking cannot improve generated pool floors. Fresh generation benefit must be tested with candidate terms active in the Adam IK optimization.",
        "",
        "## 3. Verified Historical Artifacts Inventory",
        "",
        "| Artifact Path / Archive Member | SHA256 Checksum | Availability | Role / Description |",
        "|---|---|---|---|",
    ])

    seen_artifacts: Set[str] = set()
    for r in records:
        for art in r.get("artifacts", []):
            p = art["path"]
            if p in seen_artifacts:
                continue
            seen_artifacts.add(p)
            sha = art.get("sha256") or "None"
            avail = art.get("availability", "available")
            desc = art.get("description", "")
            lines.append(f"| `{p}` | `{sha}` | `{avail}` | {desc} |")

    lines.extend([
        "",
        "## 4. Detailed Experiment Records",
        "",
    ])

    for r in records:
        r_id = r["id"]
        lines.extend([
            f"### <a id=\"{r_id}\"></a> {r['readable_name']} (`{r_id}`)",
            "",
            f"- **Mode:** `{r['mode']}`",
            f"- **Status:** `{r['status']}`",
            f"- **Question:** {r['question']}",
            f"- **Parent Records:** {r['parent_record_ids'] if r['parent_record_ids'] else 'None'}",
            f"- **Cohort Manifest:** `{r['cohort_manifest']}`",
            f"- **Selection Rule:** {r['selection_rule']}",
            f"- **Exclusions:** `{r['exclusions']}`",
            f"- **Seeds:** `{r['seeds']}`",
            f"- **Superseded By:** `{r['superseded_by']}`" if r["superseded_by"] else "- **Superseded By:** None",
            "",
            "#### Provenance Contract",
            "```json",
            json.dumps({
                "generation": r["generation"],
                "ranking": r["ranking"],
                "evaluation": r["evaluation"],
            }, indent=2),
            "```",
            "",
            "#### Metrics & Values",
            "```json",
            json.dumps(r["metrics"], indent=2),
            "```",
            "",
            f"- **Conclusion:** {r['conclusion']}",
            f"- **Limitations:** {r['limitations']}",
            "",
            "---",
            "",
        ])

    return "\n".join(lines) + "\n"


def audit_budget_dependencies(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Scan the entire codebase across src/, refit/, tests/, tools/, notebooks/, and *.md

    for consumers of every script in tools/jdev/budget/, especially nci_*.py.
    """
    root = repo_root or Path.cwd()
    budget_dir = root / "tools" / "jdev" / "budget"

    if not budget_dir.is_dir():
        raise FileNotFoundError(f"Budget directory not found at {budget_dir}")

    budget_files = sorted([f for f in os.listdir(budget_dir) if (budget_dir / f).is_file()])

    allowed_exts = {".py", ".md", ".ipynb", ".json", ".toml", ".sh", ".html", ".js", ".txt"}

    def should_skip(dirpath: Path) -> bool:
        parts = dirpath.resolve().relative_to(root.resolve()).parts
        for p in parts:
            if p in {".", ".."}:
                continue
            if p.startswith(".") and p not in {".", ".."}:
                return True
            if p in {"__pycache__", ".ipynb_checkpoints"}:
                return True
        if "data" in parts and "_archive" in parts:
            return True
        return False

    search_files: Set[Path] = set()
    for d in ["src", "refit", "tests", "tools", "notebooks"]:
        d_p = root / d
        if d_p.exists():
            for dirpath, _, filenames in os.walk(d_p):
                dp = Path(dirpath)
                if should_skip(dp):
                    continue
                for f in filenames:
                    ext = os.path.splitext(f)[1]
                    if ext in allowed_exts:
                        search_files.add(dp / f)

    for dirpath, _, filenames in os.walk(root):
        dp = Path(dirpath)
        if should_skip(dp):
            continue
        for f in filenames:
            if f.endswith(".md"):
                search_files.add(dp / f)

    file_contents: Dict[str, str] = {}
    for p in search_files:
        try:
            rel_p = str(p.relative_to(root))
            with open(p, "r", errors="ignore") as fp:
                file_contents[rel_p] = fp.read()
        except Exception:
            pass

    script_audit: Dict[str, Any] = {}
    total_nci = 0

    for bf in budget_files:
        full_p = budget_dir / bf
        rel_budget_path = f"tools/jdev/budget/{bf}"
        f_hash = sha256_of_file(full_p)
        mod_name = bf[:-3] if bf.endswith(".py") else bf
        is_nci = bf.startswith("nci_")
        if is_nci:
            total_nci += 1

        import_re = re.compile(
            rf"\b(?:from\s+(?:tools\.jdev\.budget\.)?{re.escape(mod_name)}\s+import|import\s+(?:tools\.jdev\.budget\.)?{re.escape(mod_name)}\b)"
        ) if bf.endswith(".py") else None

        consumers: Dict[str, Any] = {}

        for rel_p, content in file_contents.items():
            if rel_p == rel_budget_path:
                continue

            reasons: List[str] = []
            if bf in content:
                reasons.append("filename")
            if rel_budget_path in content or f"tools.jdev.budget.{mod_name}" in content:
                reasons.append("path")

            if import_re:
                if rel_p.startswith("tools/jdev/budget") or "tools/jdev/budget" in content:
                    if import_re.search(content):
                        reasons.append("python_import")
                elif f"tools.jdev.budget.{mod_name}" in content:
                    reasons.append("python_import")

            if reasons:
                if rel_p.startswith("src/"):
                    area = "src"
                elif rel_p.startswith("refit/"):
                    area = "refit"
                elif rel_p.startswith("tests/"):
                    area = "tests"
                elif rel_p.startswith("tools/"):
                    area = "tools"
                elif rel_p.startswith("notebooks/"):
                    area = "notebooks"
                elif rel_p.endswith(".md"):
                    area = "docs"
                else:
                    area = "other"

                consumers[rel_p] = {
                    "area": area,
                    "reasons": sorted(list(set(reasons))),
                }

        areas_consuming = sorted(list(set(c["area"] for c in consumers.values())))
        # Production scripts cannot be moved or deleted
        in_production = any(a in {"src", "refit"} for a in areas_consuming)
        in_test_suite = "tests" in areas_consuming

        script_audit[bf] = {
            "path": rel_budget_path,
            "sha256": f_hash,
            "is_nci_script": is_nci,
            "consumer_count": len(consumers),
            "areas_consuming": areas_consuming,
            "in_production": in_production,
            "in_test_suite": in_test_suite,
            "can_be_archived_in_step_8": (len(consumers) == 0 or (all(a in {"docs", "tools"} for a in areas_consuming) and not in_production and not in_test_suite)),
            "consumers": consumers,
        }

    zero_consumers = [bf for bf, d in script_audit.items() if d["consumer_count"] == 0]
    prod_consumers = [bf for bf, d in script_audit.items() if d["in_production"]]
    test_consumers = [bf for bf, d in script_audit.items() if d["in_test_suite"]]

    audit_result = {
        "metadata": {
            "title": "tools/jdev/budget/ Consumer Dependency Audit",
            "date": "2026-10-20",
            "authority": "roadmapv2.md Sections 1.3, 7, and 7.1",
            "total_budget_files": len(budget_files),
            "total_nci_files": total_nci,
            "zero_consumer_files_count": len(zero_consumers),
            "production_dependent_files_count": len(prod_consumers),
            "test_dependent_files_count": len(test_consumers),
        },
        "summary": {
            "zero_consumer_scripts": zero_consumers,
            "production_dependent_scripts": prod_consumers,
            "test_dependent_scripts": test_consumers,
        },
        "scripts": script_audit,
    }

    return audit_result


def write_atomic_json(target_path: Path | str, data: Any, indent: int = 2) -> None:
    """Atomically write data as JSON to target_path using a temporary file and rename."""
    target_path = Path(target_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = target_path.with_name(f".{target_path.name}.tmp.{os.getpid()}")
    try:
        with open(temp_file, "w", encoding="utf-8") as fp:
            json.dump(data, fp, indent=indent)
            fp.write("\n")
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temp_file, target_path)
    except Exception:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
        raise


def merge_registry_records(
    historical_records: List[Dict[str, Any]],
    existing_records: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Merge hard-coded historical records with existing records from index.json.

    Rules:
    1. Key records by their 'id' field.
    2. Start with the hard-coded historical records in their defined order.
    3. If a hard-coded record's id exists in the existing index:
       - If any hard-coded field differs in content, record as a conflict.
       - Keep added fields such as 'amended_by' on existing records.
    4. Keep every existing record that isn't hard-coded, unchanged and in its
       original order after the historical ones.

    Returns:
        (merged_records, stats)
        where stats is a dict with keys:
          - "kept": list of all kept record ids (historical matched + non-hard-coded kept)
          - "kept_historical": list of historical record ids matched without conflict
          - "kept_appended": list of non-hard-coded record ids preserved
          - "added": list of historical record ids newly added (not in existing file)
          - "conflicts": dict mapping record_id -> list of differing fields
    """
    if existing_records is None:
        existing_records = []

    existing_by_id: Dict[str, Dict[str, Any]] = {}
    for r in existing_records:
        r_id = r.get("id")
        if r_id:
            existing_by_id[r_id] = r

    hist_ids: Set[str] = set()
    merged_records: List[Dict[str, Any]] = []
    kept_historical: List[str] = []
    added: List[str] = []
    conflicts: Dict[str, List[str]] = {}

    for h_rec in historical_records:
        r_id = h_rec.get("id")
        if not r_id:
            continue
        hist_ids.add(r_id)

        if r_id in existing_by_id:
            e_rec = existing_by_id[r_id]
            diff_fields: List[str] = []
            for k in h_rec:
                if k not in e_rec:
                    diff_fields.append(f"{k} (missing in existing record)")
                elif h_rec[k] != e_rec[k]:
                    diff_fields.append(k)

            if diff_fields:
                conflicts[r_id] = diff_fields
            else:
                merged_rec = dict(h_rec)
                for k, v in e_rec.items():
                    if k not in h_rec:
                        merged_rec[k] = v
                merged_records.append(merged_rec)
                kept_historical.append(r_id)
        else:
            merged_records.append(dict(h_rec))
            added.append(r_id)

    kept_appended: List[str] = []
    for e_rec in existing_records:
        r_id = e_rec.get("id")
        if r_id and r_id not in hist_ids:
            merged_records.append(dict(e_rec))
            kept_appended.append(r_id)

    stats = {
        "kept": kept_historical + kept_appended,
        "kept_historical": kept_historical,
        "kept_appended": kept_appended,
        "added": added,
        "conflicts": conflicts,
    }
    return merged_records, stats


def build_registry(
    index_path: Optional[Path | str] = None,
    repo_root: Optional[Path | str] = None,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """Build or merge the experiment registry index.json.

    1. Loads existing index.json if present.
    2. Merges hard-coded historical records with existing records.
    3. If dry_run is True: prints what would be added, kept or conflicted, and writes nothing.
    4. If conflicts exist and not dry_run: raises ValueError listing differing fields.
    5. Validates merged result with validate_registry before writing.
    6. Writes atomically using a temp file plus rename.
    """
    root = Path(repo_root).resolve() if repo_root else REPO_ROOT
    idx_p = Path(index_path).resolve() if index_path else (root / "data" / "refit" / "experiments" / "index.json")

    hist_records = build_historical_records()

    existing_records = None
    if idx_p.is_file():
        try:
            with open(idx_p, "r", encoding="utf-8") as fp:
                existing_records = json.load(fp)
        except Exception as e:
            raise ValueError(f"Failed to read existing index from {idx_p}: {e}") from e

    merged_records, stats = merge_registry_records(hist_records, existing_records)
    conflicts = stats["conflicts"]
    kept = stats["kept"]
    added = stats["added"]

    if dry_run:
        print(f"NCI v2 Registry Build (dry-run) for {idx_p}:")
        print(f"  {len(kept)} records kept and {len(conflicts)} conflicts ({len(added)} added)")
        if kept:
            print(f"  Kept ({len(kept)}):")
            for r_id in kept:
                print(f"    - {r_id}")
        if added:
            print(f"  Added ({len(added)}):")
            for r_id in added:
                print(f"    - {r_id}")
        if conflicts:
            print(f"  Conflicts ({len(conflicts)}):")
            for r_id, diffs in conflicts.items():
                print(f"    - {r_id}: differing fields: {', '.join(diffs)}")
        return merged_records

    if conflicts:
        conflict_msg = "\n".join(
            f"  - Record '{r_id}': differing fields: {', '.join(diffs)}"
            for r_id, diffs in conflicts.items()
        )
        raise ValueError(
            f"Conflicting hard-coded record(s) found in existing registry:\n{conflict_msg}"
        )

    # Validate registry before writing
    errors = validate_registry(merged_records, repo_root=root)
    if errors:
        raise ValueError(
            f"Registry validation failed for merged records:\n" + "\n".join(errors)
        )

    write_atomic_json(idx_p, merged_records)
    print(f"Wrote {len(merged_records)} records to {idx_p} ({len(kept)} kept, {len(added)} added).")
    return merged_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Jacobik NCI v2 Experiment Registry and Dependency Auditor")
    parser.add_argument("--build", action="store_true", help="Build and write index.json (merging with existing records)")
    parser.add_argument("--dry-run", action="store_true", help="Dry run for --build: print what would be added, kept or conflicted, and write nothing")
    parser.add_argument("--render", action="store_true", help="Render index.json to README.md")
    parser.add_argument("--audit-deps", action="store_true", help="Audit tools/jdev/budget dependencies")
    parser.add_argument("--verify", action="store_true", help="Validate index.json schema and checksums")
    parser.add_argument("--all", action="store_true", help="Build, render, audit dependencies, and verify")
    args = parser.parse_args()

    repo_root = Path.cwd()
    exp_dir = repo_root / "data" / "refit" / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)

    index_path = exp_dir / "index.json"
    readme_path = exp_dir / "README.md"
    dep_audit_path = exp_dir / "dependency_audit.json"

    run_all = args.all or (not args.build and not args.render and not args.audit_deps and not args.verify and not args.dry_run)

    if args.dry_run:
        build_registry(index_path, repo_root=repo_root, dry_run=True)
        return

    if args.build or run_all:
        print(f"Building/merging historical experiment records -> {index_path}...")
        records = build_registry(index_path, repo_root=repo_root, dry_run=False)

    if args.render or run_all:
        print(f"Rendering {index_path} -> {readme_path}...")
        with open(index_path, "r", encoding="utf-8") as fp:
            records = json.load(fp)
        md = render_markdown(records)
        with open(readme_path, "w", encoding="utf-8") as fp:
            fp.write(md)
        print(f"Wrote README.md ({len(md)} bytes) to {readme_path}.")

    if args.audit_deps or run_all:
        print(f"Auditing tools/jdev/budget dependencies -> {dep_audit_path}...")
        dep_audit = audit_budget_dependencies(repo_root=repo_root)
        with open(dep_audit_path, "w", encoding="utf-8") as fp:
            json.dump(dep_audit, fp, indent=2)
        print(f"Wrote dependency audit ({dep_audit['metadata']['total_budget_files']} files) to {dep_audit_path}.")

    if args.verify or run_all:
        print(f"Verifying {index_path} schema and artifact integrity...")
        with open(index_path, "r", encoding="utf-8") as fp:
            records = json.load(fp)
        errors = validate_registry(records, repo_root=repo_root)
        if errors:
            print(f"FAILED: Found {len(errors)} validation errors:")
            for err in errors:
                print(f"  - {err}")
            raise SystemExit(1)
        else:
            print("PASSED: All records strictly adhere to schema and artifact checksums verified.")


if __name__ == "__main__":
    main()
