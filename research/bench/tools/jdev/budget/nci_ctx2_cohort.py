"""Select a fresh, parent/scaffold-disjoint cohort for the matched-pair NCI plan.

Selection uses only native structure metadata, strict contact chemistry,
scaffold connectivity, size limits, and a stable hash. It never reads solver
scores or MACE energies. Previously examined NCI-CTX1 scaffold components are
excluded in full so the new validation parents remain genuinely fresh.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SCAN = Path("/tmp/nci_ctx1_native_partial")
DEFAULT_OUT = Path("/tmp/nci_ctx2_fresh/cohort")
SPLIT_SIZES = {"surface_fit": 160, "scale_calibration": 40, "validation": 40}
OLD_DIAGNOSTICS = {"PEWBOL", "YIYXEJ", "FETSUU"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _key2(row: dict) -> set[tuple[str, ...]]:
    return {tuple(contact["keys"][2]) for contact in row["contacts"] if len(contact["keys"]) > 2}


def _tie(value: str) -> int:
    return int.from_bytes(hashlib.sha256(("nci-ctx2-select-v1:" + value).encode()).digest()[:8], "big")


def _eligible(row: dict, max_atoms: int, max_partner_heavy: int) -> bool:
    if row["n_atoms"] > max_atoms:
        return False
    return max((p["heavy_atoms"] for c in row["contacts"] for p in c["partners"]), default=0) <= max_partner_heavy


def _load_rows(scan: Path) -> tuple[list[dict], dict]:
    summary = json.loads((scan / "summary.json").read_text())
    if summary.get("status") != "complete":
        raise ValueError("scan summary is not complete")
    rows = []
    for shard in summary["identity"]["shards"]:
        rows.extend(json.loads((scan / f"{shard}.json").read_text())["positive_parents"])
    return rows, summary


def _components(rows: list[dict]) -> list[tuple[str, set[str], list[dict]]]:
    """Connect parents through every non-tiny ligand scaffold in strict contacts."""
    graph = nx.Graph()
    by_code = {row["code"]: row for row in rows}
    for row in rows:
        parent = "parent:" + row["code"]
        for contact in row["contacts"]:
            for partner in contact["partners"]:
                if int(partner["heavy_atoms"]) > 2:
                    graph.add_edge(parent, "scaffold:" + partner["scaffold"])
        if parent not in graph:
            graph.add_node(parent)
    out = []
    for nodes in nx.connected_components(graph):
        codes = {node.removeprefix("parent:") for node in nodes if node.startswith("parent:")}
        cid = hashlib.sha256("\n".join(sorted(nodes)).encode()).hexdigest()[:20]
        out.append((cid, codes, [by_code[c] for c in sorted(codes)]))
    return sorted(out, key=lambda item: item[0])


def _select_groups(groups: list[tuple[str, list[dict]]], n: int) -> list[dict]:
    """Greedily cover common chemistry and dataset strata without reusing a group."""
    group_support: Counter = Counter()
    dataset_support: Counter = Counter()
    for _gid, options in groups:
        group_support.update(set().union(*(_key2(row) for row in options)))
        dataset_support.update({row["dataset"] for row in options})
    common = [key for key, count in group_support.items() if count >= max(12, n // 10)]
    common.sort(key=lambda key: (-group_support[key], repr(key)))
    common = common[:32]
    max_support = max((group_support[key] for key in common), default=1)
    key_target = {
        key: min(max(4, n // 4), max(3, round((n // 4) * (group_support[key] / max_support) ** 0.5)))
        for key in common
    }
    total_group_dataset_memberships = sum(dataset_support.values())
    ds_target = {dataset: n * count / total_group_dataset_memberships
                 for dataset, count in dataset_support.items()}
    chosen: list[dict] = []
    remaining = {gid: options for gid, options in groups}
    have_key: Counter = Counter()
    have_dataset: Counter = Counter()
    while len(chosen) < n:
        best = None
        best_score = float("-inf")
        for gid, options in remaining.items():
            for row in options:
                keys = _key2(row)
                key_gain = sum(max(0, key_target[key] - have_key[key]) / key_target[key]
                               for key in keys if key in key_target)
                dataset = row["dataset"]
                ds_gain = max(0.0, ds_target[dataset] - have_dataset[dataset]) / max(ds_target[dataset], 1.0)
                score = key_gain + 0.2 * ds_gain
                score += (_tie(gid + ":" + row["code"]) / 2**64) * 1e-9
                if score > best_score:
                    best, best_score = (gid, row), score
        if best is None:
            raise RuntimeError(f"only selected {len(chosen)} of {n} requested groups")
        gid, row = best
        payload = dict(row)
        payload["scaffold_component"] = gid
        chosen.append(payload)
        have_key.update(key for key in _key2(row) if key in key_target)
        have_dataset[row["dataset"]] += 1
        remaining.pop(gid)
    return chosen


def _assign_splits(selected: list[dict]) -> dict[str, list[dict]]:
    """Allocate whole scaffold components to 160/40/40 splits with stratification."""
    ratios = {key: size / len(selected) for key, size in SPLIT_SIZES.items()}
    key_groups: Counter = Counter()
    dataset_counts = Counter(row["dataset"] for row in selected)
    key_counts: Counter = Counter()
    for row in selected:
        keys = _key2(row)
        key_counts.update(keys)
        key_groups.update(keys)
    key_targets = {split: {key: ratios[split] * count for key, count in key_counts.items()}
                   for split in SPLIT_SIZES}
    dataset_targets = {split: {d: ratios[split] * n for d, n in dataset_counts.items()}
                       for split in SPLIT_SIZES}
    # Rare chemistry groups are assigned first; hash order resolves exact ties.
    ordered = sorted(selected, key=lambda row: (
        min((key_groups[key] for key in _key2(row)), default=10**9),
        -len(_key2(row)),
        _tie(row["scaffold_component"]),
    ))
    assigned = {split: [] for split in SPLIT_SIZES}
    assigned_keys = {split: Counter() for split in SPLIT_SIZES}
    assigned_datasets = {split: Counter() for split in SPLIT_SIZES}
    for row in ordered:
        keys = _key2(row)
        options = [split for split, target in SPLIT_SIZES.items() if len(assigned[split]) < target]
        def score(split: str) -> float:
            ds = row["dataset"]
            size_deficit = (SPLIT_SIZES[split] - len(assigned[split])) / SPLIT_SIZES[split]
            dtarget = dataset_targets[split][ds]
            dataset_deficit = max(0.0, dtarget - assigned_datasets[split][ds]) / max(dtarget, 1.0)
            key_deficits = [max(0.0, key_targets[split][key] - assigned_keys[split][key]) /
                            max(key_targets[split][key], 1.0) for key in keys]
            key_deficit = sum(key_deficits) / max(len(key_deficits), 1)
            return (size_deficit + 0.30 * dataset_deficit + 0.50 * key_deficit
                    + _tie(split + row["scaffold_component"]) * 1e-30)
        split = max(options, key=score)
        assigned[split].append(row)
        assigned_datasets[split][row["dataset"]] += 1
        assigned_keys[split].update(keys)
    if {k: len(v) for k, v in assigned.items()} != SPLIT_SIZES:
        raise RuntimeError("split allocation did not meet exact targets")
    return assigned


def select(scan: Path, old_selection: Path, out: Path, *, max_atoms: int,
           max_partner_heavy: int, n: int) -> dict:
    rows, scan_summary = _load_rows(scan)
    old = json.loads(old_selection.read_text()) if old_selection.is_file() else {"selected_rows": {}}
    old_codes = {row["code"] for items in old.get("selected_rows", {}).values() for row in items}
    old_codes |= OLD_DIAGNOSTICS
    identity_exclusions = scan_summary.get("identity", {}).get("benchmark_codes_excluded", [])
    if isinstance(identity_exclusions, list):
        old_codes |= set(identity_exclusions)
    eligible = [row for row in rows if _eligible(row, max_atoms, max_partner_heavy)]
    eligible_by_code = {row["code"]: row for row in eligible}
    components = _components(rows)
    fresh_groups = []
    excluded_group_count = 0
    for cid, codes, all_rows in components:
        if codes & old_codes:
            excluded_group_count += 1
            continue
        options = [eligible_by_code[code] for code in sorted(codes) if code in eligible_by_code]
        if options:
            fresh_groups.append((cid, options))
    if len(fresh_groups) < n:
        raise ValueError(f"only {len(fresh_groups)} untouched eligible scaffold groups; need {n}")
    chosen = _select_groups(fresh_groups, n)
    splits = _assign_splits(chosen)
    # Audits: exact parent/scaffold separation from previous runs and across all three new splits.
    split_by_component = {row["scaffold_component"]: split for split, items in splits.items() for row in items}
    if len(split_by_component) != len(chosen):
        raise AssertionError("a scaffold component was selected more than once")
    if any(row["code"] in old_codes for row in chosen):
        raise AssertionError("a previously examined parent was selected")
    all_scaffolds = {}
    for split, items in splits.items():
        for row in items:
            for contact in row["contacts"]:
                for partner in contact["partners"]:
                    if partner["heavy_atoms"] <= 2:
                        continue
                    scaffold = partner["scaffold"]
                    prior = all_scaffolds.setdefault(scaffold, split)
                    if prior != split:
                        raise AssertionError(f"scaffold leakage across splits: {scaffold}")
    payload = {
        "schema": "nci_ctx2_fresh_cohort_v1",
        "scope": "fresh selected TMQM parents; previous NCI-CTX1 parent/scaffold components excluded",
        "scan": str(scan), "scan_identity": scan_summary["identity"],
        "scan_summary_sha256": sha256(scan / "summary.json"),
        "old_selection_sha256": sha256(old_selection) if old_selection.is_file() else None,
        "selection_rule": {
            "whole_group_unit": "connected parent/non-tiny-participant Murcko-scaffold component",
            "one_parent_per_group": True,
            "prior_components_excluded": excluded_group_count,
            "previously_examined_codes_excluded": sorted(old_codes),
            "max_complex_atoms": max_atoms, "max_participant_heavy_atoms": max_partner_heavy,
            "sizes": SPLIT_SIZES,
            "selection_inputs": "strict native metadata, local chemistry strata, dataset, size, and stable hash only; no energies or solver scores",
            "selection_objective": "greedy coverage of common level-2 chemistry keys and X1/X2/X3 dataset strata",
        },
        "eligible_parent_count": len(eligible),
        "fresh_scaffold_group_count": len(fresh_groups),
        "fresh_eligible_parent_count": sum(len(options) for _, options in fresh_groups),
        "selected_rows": splits,
        "split_summary": {},
    }
    for split, items in splits.items():
        ds = Counter(row["dataset"] for row in items)
        keys = Counter(key for row in items for key in _key2(row))
        payload["split_summary"][split] = {
            "parents": len(items), "datasets": dict(ds),
            "level2_keys": {"n": len(keys), "counts": {"|".join(k): v for k, v in keys.most_common()}},
            "native_strict_contacts": sum(len(row["contacts"]) for row in items),
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", type=Path, default=DEFAULT_SCAN)
    p.add_argument("--old-selection", type=Path, default=DEFAULT_SCAN / "cohort/selection.json")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT / "selection.json")
    p.add_argument("--max-atoms", type=int, default=120)
    p.add_argument("--max-partner-heavy", type=int, default=48)
    p.add_argument("--n", type=int, default=sum(SPLIT_SIZES.values()))
    a = p.parse_args()
    if a.n != sum(SPLIT_SIZES.values()):
        raise ValueError("this preregistered cohort requires 160/40/40 exactly")
    result = select(a.scan.resolve(), a.old_selection.resolve(), a.out.resolve(),
                    max_atoms=a.max_atoms, max_partner_heavy=a.max_partner_heavy, n=a.n)
    print(json.dumps({"selected": {k: v["parents"] for k, v in result["split_summary"].items()},
                      "fresh_groups": result["fresh_scaffold_group_count"],
                      "summary": result["split_summary"], "selection": str(a.out.resolve())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
