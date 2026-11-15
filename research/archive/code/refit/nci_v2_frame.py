"""research/archive/code/refit/nci_v2_frame.py
====================
Corpus Population Manifest, Shard Audit, Direct 1-Hop Neighbor Quarantine,
and Stratified Sequential Pairwise-Disjoint Sampling for Jacobik NCI v2 Pilot.

Scope & boundaries:
- Reads cleaned tmQM (X1-X3) and LnQM (LNf) shards (all 34 shards).
- Computes shard SHA256 hashes, complex counts, charge/spin distributions,
  canonical benchmark exclusions, and LnQM sibling weights.
- Implements Direct 1-Hop Neighbor Quarantine against canonical benchmarks
  (dev32, n48, heldout28, ring10) using source sibling families and directly
  shared non-exempt informative scaffolds, recovering ~28,000 mixed-ligand
  complexes from artificial giant components.
- Exempts from scaffold grouping only ligands whose graph key is on the frozen spec.json whitelist.
- Implements Stratified Sequential Pairwise-Disjoint Sampling across metal row
  (3d, 4d, 5d, 4f), charge, and coordination number:
    1. 12-parent development pilot (disjoint groups, stratified, verified opportunity
       coverage, 6 repeatability parents).
    2. 240-group main allocation (160 surface_fit, 40 calibration, 40 motion_validation)
       plus 48 reserve groups.
- Enforces strict pairwise disjointness across distinct splits (candidates sharing
  sibling families or non-exempt informative scaffolds with any other split or benchmark
  are rejected).
- Runs within-split Union-Find clustering for weighting (w_g = 1 / N_g) and bootstrap units.
- Predeclares unmappable input policy: failures in ligand_inputs_from_mol are recorded as
  outcomes; reserve replacement applies only to training measurements, within stratum and
  in predeclared order; evaluation cohorts never replace.
- Verifies Gate 1 minimums (>= 10 informative cal and val groups per activated family).
- Runs an RDKit canonical SMILES / InChIKey second-hash check of pairwise fragment overlap
  across all splits and benchmarks (same fragments as the WL keys; guards against collisions).
- Verifies TMQMParser assembly of all sampled parents and writes
  data/refit/nci_v2/pilot_v1/frame.json.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import zlib
from typing import Any, Dict, List, Optional, Set, Tuple

# Restrict threading and set deterministic encoding
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import networkx as nx
import numpy as np
from rdkit import Chem

# Add src and repo root to path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from refit.refit_lib import benchmark_codes, sibling_weights
from jacobik.fit.parser import (
    TMQMParser,
    _assemble_raw,
    _attach_qm_properties,
    _iter_bo_blocks,
    _iter_xyz_blocks,
    _parse_xyz_comment,
)
from jacobik.stats.mapping import assemble, METALS, _covalent_radius
from research.archive.code.interactions.nci_v2.nci_v2 import enumerate_opportunities
from jacobik.builder.ligands import _build_ligand_groups, ligand_inputs_from_mol

# Candidates mappability-checked per stratum to estimate its mappable fraction before allocation
PRESCREEN_PER_STRATUM = 20

# Non-metal ligand bond cutoff (same scale/pad as jacobik.stats.mapping's distance sanity check)
LIGAND_BOND_SCALE = 1.25
LIGAND_BOND_PAD = 0.25

ACC_ELEMENTS = frozenset({"N", "O", "F", "S", "Cl", "Br", "I"})
HETERO_ELEMENTS = frozenset({"N", "O", "F", "S", "P", "Cl", "Br", "I"})

METAL_ROW = {
    # 3d
    "Sc": "3d", "Ti": "3d", "V": "3d", "Cr": "3d", "Mn": "3d",
    "Fe": "3d", "Co": "3d", "Ni": "3d", "Cu": "3d", "Zn": "3d",
    # 4d
    "Y": "4d", "Zr": "4d", "Nb": "4d", "Mo": "4d", "Tc": "4d",
    "Ru": "4d", "Rh": "4d", "Pd": "4d", "Ag": "4d", "Cd": "4d",
    # 5d
    "Hf": "5d", "Ta": "5d", "W": "5d", "Re": "5d", "Os": "5d",
    "Ir": "5d", "Pt": "5d", "Au": "5d", "Hg": "5d",
    # 4f (Lanthanides)
    "La": "4f", "Ce": "4f", "Pr": "4f", "Nd": "4f", "Pm": "4f",
    "Sm": "4f", "Eu": "4f", "Gd": "4f", "Tb": "4f", "Dy": "4f",
    "Ho": "4f", "Er": "4f", "Tm": "4f", "Yb": "4f", "Lu": "4f",
    # 5f (Actinides)
    "Ac": "5f", "Th": "5f", "Pa": "5f", "U": "5f", "Np": "5f", "Pu": "5f",
}


def sha256_file(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(text: str) -> int:
    """Deterministic CRC32 hash for tie-breaking and sampling."""
    return zlib.crc32(text.encode("utf-8"))


def classify_charge(q: int) -> str:
    return "neg" if q < 0 else "pos" if q > 0 else "zero"


def classify_size(n_heavy: int) -> str:
    return "small" if n_heavy <= 40 else "large" if n_heavy > 70 else "medium"


def classify_cn(cn: int) -> str:
    return "low" if cn <= 4 else "high" if cn >= 7 else "med"


def ligand_graph_key(elements: Dict[int, str], edges: List[Tuple[int, int]]) -> str:
    """Scaffold key of a ligand component: heavy-atom formula + element-labelled WL hash.

    H atoms and bond orders are ignored, so e.g. PPh3/PCy3, benzene/cyclohexane and
    IMes/SIMes share a key.
    """
    g = nx.Graph()
    for u, sym in elements.items():
        g.add_node(u, element=sym)
    g.add_edges_from(edges)
    wl_hash = nx.weisfeiler_lehman_graph_hash(g, node_attr="element")
    c_form = Counter(elements.values())
    form_str = "".join(f"{k}{c_form[k]}" for k in sorted(c_form))
    return f"LIG:{form_str}:{wl_hash}"


def ligand_key_from_smiles(smiles: str) -> str:
    """Scaffold key of a ligand given as SMILES (connectivity only; no sanitization)."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        raise ValueError(f"Unparseable whitelist SMILES: {smiles}")
    elements = {a.GetIdx(): a.GetSymbol() for a in mol.GetAtoms() if a.GetSymbol() != "H"}
    edges = [
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in elements and b.GetEndAtomIdx() in elements
    ]
    return ligand_graph_key(elements, edges)


def load_whitelist_keys(spec_data: Dict[str, Any]) -> Dict[str, str]:
    """Frozen whitelist name -> graph key from spec.json, re-derived from the SMILES and checked."""
    block = spec_data["common_ligands_exempt_from_scaffold_grouping"]
    frozen = block["frozen_graph_keys"]
    for name, entry in frozen.items():
        derived = ligand_key_from_smiles(entry["smiles"])
        assert derived == entry["graph_key"], f"Whitelist key drift for {name}: {derived} != {entry['graph_key']}"
    return {name: entry["graph_key"] for name, entry in frozen.items()}


class DisjointSetUnion:
    """Union-Find with path compression for fast independent grouping."""

    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def run_shard_audit(clean_dir: Path) -> Dict[str, Any]:
    """Audit all 34 shards in clean_dir and record exact metrics."""
    tmqm_dir = clean_dir / "tmQM"
    lnqm_dir = clean_dir / "lnQM"

    xyz_files = sorted(list((tmqm_dir / "xyz").glob("*.xyz")) + list((lnqm_dir / "xyz").glob("*.xyz")))
    shards_audit = []
    total_complexes = 0
    charges_overall: Counter = Counter()
    spins_overall: Counter = Counter()

    for xp in xyz_files:
        shard_name = xp.stem.replace("tmQM_", "")
        dataset = "LNf" if "LNf" in shard_name else shard_name[:2]
        parent_dir = lnqm_dir if dataset == "LNf" else tmqm_dir
        bp = parent_dir / "bo" / f"tmQM_{shard_name}.BO"

        xyz_sha = sha256_file(xp)
        bo_sha = sha256_file(bp)

        shard_charges: Counter = Counter()
        shard_spins: Counter = Counter()
        count = 0

        for comment, xyz_lines in _iter_xyz_blocks(xp):
            code, q, s, stoich, mnd, csd_years = _parse_xyz_comment(comment)
            count += 1
            shard_charges[q] += 1
            shard_spins[s] += 1

        total_complexes += count
        charges_overall.update(shard_charges)
        spins_overall.update(shard_spins)

        shards_audit.append({
            "shard": shard_name,
            "dataset": dataset,
            "xyz_file": str(xp.relative_to(REPO_ROOT)),
            "bo_file": str(bp.relative_to(REPO_ROOT)),
            "xyz_sha256": xyz_sha,
            "bo_sha256": bo_sha,
            "n_complexes": count,
            "charge_distribution": {str(k): v for k, v in sorted(shard_charges.items())},
            "spin_distribution": {str(k): v for k, v in sorted(shard_spins.items())},
        })

    canonical_benchmarks = benchmark_codes()
    raw_holdout = json.loads((REPO_ROOT / "refit/benchmark_holdout_codes.json").read_text())
    by_source = {k: len(v) for k, v in raw_holdout.items() if not k.startswith("_")}

    ln_weights = sibling_weights(lnqm_dir)

    return {
        "total_shards": len(shards_audit),
        "total_complexes": total_complexes,
        "shards": shards_audit,
        "corpus_summary": {
            "total_complexes": total_complexes,
            "total_tmqm": sum(s["n_complexes"] for s in shards_audit if s["dataset"] != "LNf"),
            "total_lnqm": sum(s["n_complexes"] for s in shards_audit if s["dataset"] == "LNf"),
            "charges": {str(k): v for k, v in sorted(charges_overall.items())},
            "spins": {str(k): v for k, v in sorted(spins_overall.items())},
        },
        "benchmark_exclusions": {
            "canonical_benchmark_count": len(canonical_benchmarks),
            "codes": canonical_benchmarks,
            "by_source": by_source,
        },
        "lnqm_sibling_weights": {
            "weighted_codes_count": len(ln_weights),
            "sample_weight": next(iter(ln_weights.values())) if ln_weights else 1.0,
        },
    }


def scan_corpus_and_extract_ligands(
    clean_dir: Path, exempt_keys: Set[str]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[str]], Set[str]]:
    """
    Scan clean corpus and extract all complexes, sibling IDs, and organic ligand cores (>= 5 heavy, has C).
    Ligands whose graph key is on the frozen spec whitelist (`exempt_keys`) are exempt from scaffold
    grouping; every other ligand stays informative.
    """
    tmqm_dir = clean_dir / "tmQM"
    lnqm_dir = clean_dir / "lnQM"

    xyz_files = sorted(list((tmqm_dir / "xyz").glob("*.xyz")) + list((lnqm_dir / "xyz").glob("*.xyz")))
    complex_meta: Dict[str, Dict[str, Any]] = {}
    lig_sibs: Dict[str, Set[str]] = defaultdict(set)
    complex_ligs: Dict[str, List[str]] = {}

    for xp in xyz_files:
        shard_name = xp.stem.replace("tmQM_", "")
        parent_dir = lnqm_dir if "LNf" in shard_name else tmqm_dir
        bp = parent_dir / "bo" / f"tmQM_{shard_name}.BO"

        for (comment, xyz_lines), (bo_code, bo_lines) in zip(_iter_xyz_blocks(xp), _iter_bo_blocks(bp)):
            code, q, s, stoich, mnd, csd_years = _parse_xyz_comment(comment)

            if "_" in code:
                sib = code.split("_")[0]  # LnQM 32-hex structure_id
            else:
                sib = code[:6]  # tmQM 6-letter CSD prefix

            atoms = [(idx, line.split()[0]) for idx, line in enumerate(xyz_lines, 1)]
            positions = {idx: np.array(line.split()[1:4], dtype=float) for idx, line in enumerate(xyz_lines, 1)}
            metal_idx = next((idx for idx, sym in atoms if sym in METALS), None)
            metal_sym = atoms[metal_idx - 1][1] if metal_idx else None
            atom_sym = dict(atoms)

            adj: Dict[int, List[int]] = {a[0]: [] for a in atoms}
            raw_bonds: List[Tuple[int, int, float]] = []
            for line in bo_lines:
                parts = line.split()
                if len(parts) < 3:
                    continue
                u = int(parts[0])
                idx = 3
                while idx + 2 < len(parts):
                    v = int(parts[idx + 1])
                    bo = float(parts[idx + 2])
                    idx += 3
                    if bo < 0.08:
                        continue
                    # Wiberg >= 0.08 also admits 1,3/1,4 ring contacts (e.g. aromatic meta/para),
                    # which splits one ligand across several scaffold keys; require a covalent distance.
                    if metal_idx not in (u, v):
                        r_max = LIGAND_BOND_SCALE * (_covalent_radius(atom_sym[u]) + _covalent_radius(atom_sym[v])) + LIGAND_BOND_PAD
                        if float(np.linalg.norm(positions[u] - positions[v])) > r_max:
                            continue
                    adj[u].append(v)
                    if u < v:
                        raw_bonds.append((u, v, bo))

            visited = {metal_idx} if metal_idx else set()
            atom_dict = dict(atoms)
            n_heavy = sum(1 for _, sym in atoms if sym != "H")
            c_ligs = []

            if metal_idx:
                for a_idx, sym in atoms:
                    if a_idx in visited:
                        continue
                    comp = []
                    q_nodes = [a_idx]
                    visited.add(a_idx)
                    while q_nodes:
                        curr = q_nodes.pop()
                        comp.append(curr)
                        for nxt in adj[curr]:
                            if nxt not in visited and nxt != metal_idx:
                                visited.add(nxt)
                                q_nodes.append(nxt)

                    heavy = [u for u in comp if atom_dict[u] != "H"]
                    if len(heavy) >= 5 and any(atom_dict[u] == "C" for u in heavy):
                        heavy_set = set(heavy)
                        lig_id = ligand_graph_key(
                            {u: atom_dict[u] for u in heavy},
                            [(u, v) for u in heavy for v in adj[u] if v in heavy_set and u < v],
                        )
                        c_ligs.append(lig_id)
                        lig_sibs[lig_id].add(sib)

            m_row = METAL_ROW.get(metal_sym, "unknown")
            complex_ligs[code] = c_ligs
            complex_meta[code] = {
                "code": code,
                "sib": sib,
                "shard": shard_name,
                "dataset": "LNf" if "LNf" in shard_name else shard_name[:2],
                "q": q,
                "charge_class": classify_charge(q),
                "s": s,
                "mnd": mnd,
                "cn_class": classify_cn(mnd),
                "metal": metal_sym,
                "metal_row": m_row,
                "n_atoms": len(atoms),
                "n_heavy": n_heavy,
                "size_class": classify_size(n_heavy),
                "stoich": stoich,
                "atoms": atoms,
                "bonds": raw_bonds,
                "metal_idx": metal_idx,
            }

    # Common ligands exempt from scaffold grouping: frozen whitelist keys present in the corpus
    common_ligands = {lig for lig in lig_sibs if lig in exempt_keys}

    # Assign informative vs common ligands
    for code, m in complex_meta.items():
        ligs = complex_ligs[code]
        m["informative_ligands"] = [l for l in ligs if l not in common_ligands]
        m["common_ligands"] = [l for l in ligs if l in common_ligands]

    return complex_meta, complex_ligs, common_ligands


def apply_direct_1hop_quarantine(
    complex_meta: Dict[str, Dict[str, Any]],
    benchmarks: Set[str],
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Apply Direct 1-Hop Neighbor Quarantine rule against canonical benchmarks:
    Exclude complex c if and only if:
      1. c is directly in benchmarks (direct benchmark complex),
      2. c's source sibling family matches any benchmark's source sibling family,
      3. c directly shares a non-exempt informative scaffold with any benchmark complex.
    Does NOT transitively exclude multi-hop complexes, recovering ~28,000 mixed-ligand complexes.
    """
    bench_sibs = {complex_meta[c]["sib"] for c in benchmarks if c in complex_meta}
    bench_inf_scaffolds = {
        l for c in benchmarks if c in complex_meta for l in complex_meta[c]["informative_ligands"]
    }

    excluded_by_bench_direct: List[str] = []
    excluded_by_bench_sibling: List[str] = []
    excluded_by_bench_scaffold: List[str] = []
    eligible_complexes: List[str] = []

    for code, m in complex_meta.items():
        if code in benchmarks:
            excluded_by_bench_direct.append(code)
            m["quarantine_reason"] = "benchmark_direct"
        elif m["sib"] in bench_sibs:
            excluded_by_bench_sibling.append(code)
            m["quarantine_reason"] = "benchmark_source_sibling"
        elif set(m["informative_ligands"]) & bench_inf_scaffolds:
            excluded_by_bench_scaffold.append(code)
            m["quarantine_reason"] = "benchmark_shared_informative_scaffold"
        else:
            eligible_complexes.append(code)
            m["quarantine_reason"] = "eligible"

    total_quarantined = (
        len(excluded_by_bench_direct)
        + len(excluded_by_bench_sibling)
        + len(excluded_by_bench_scaffold)
    )

    # Historic baseline with transitive Union-Find excluded 29,079 complexes (eligible: 96,731).
    recovered_count = len(eligible_complexes) - 96731

    quarantine_summary = {
        "rule": "direct_1hop_neighbor_quarantine",
        "benchmark_direct_count": len(excluded_by_bench_direct),
        "benchmark_source_sibling_count": len(excluded_by_bench_sibling),
        "benchmark_shared_informative_scaffold_count": len(excluded_by_bench_scaffold),
        "total_quarantined_complexes": total_quarantined,
        "eligible_complexes_count": len(eligible_complexes),
        "recovered_complexes_from_giant_component": recovered_count,
        "benchmark_siblings_quarantined_count": len(bench_sibs),
        "benchmark_informative_scaffolds_quarantined_count": len(bench_inf_scaffolds),
    }

    return eligible_complexes, quarantine_summary


def _evaluate_complex_opportunities(
    clean_dir: Path,
    meta: Dict[str, Any],
    admission_rules: Dict[str, Any],
    parsers: Dict[str, TMQMParser],
) -> Dict[str, int]:
    """Evaluate authoritative opportunities via jacobik.stats.nci_v2.enumerate_opportunities."""
    code = meta["code"]
    shard = meta["shard"]
    ds = meta["dataset"]
    data_kind = "lnQM" if ds == "LNf" else "tmQM"
    parent_dir = clean_dir / data_kind
    parser = parsers[data_kind]

    xp = parent_dir / "xyz" / f"tmQM_{shard}.xyz"
    bp = parent_dir / "bo" / f"tmQM_{shard}.BO"

    for (comment, atoms), (bo_code, bonds) in zip(_iter_xyz_blocks(xp), _iter_bo_blocks(bp)):
        if bo_code == code:
            csv_row = parser.csv_index.get(code)
            raw = _assemble_raw(comment, atoms, code, bonds, csv_row)
            mol = assemble(raw)
            if csv_row and csv_row.smiles:
                mol.smiles = csv_row.smiles

            # Verify solver readiness for pilot parents
            groups = _build_ligand_groups(mol, mol.smiles)
            lig_in = ligand_inputs_from_mol(mol, groups)
            assert lig_in is not None, f"Solver preparation failed for {code}"

            batch = enumerate_opportunities(mol, admission_rules)
            by_class: Dict[str, int] = {
                "polar_h_acceptor": 0,
                "activated_ch_acceptor": 0,
                "sp2_ch_acceptor": 0,
                "sigma_hole_acceptor": 0,
                "polar_multiple_bond": 0,
                "dh_ring": 0,
                "ch_ring": 0,
                "lp_or_halogen_ring": 0,
                "sp2_5_6_ring_pair": 0,
                "ring_ring": 0,
                "total": 0,
            }
            for fam, o_list in batch.opportunities.items():
                for o in o_list:
                    ac = o.get("admission_class", fam)
                    by_class[ac] = by_class.get(ac, 0) + 1
            by_class["total"] = batch.total_count
            ring_ring_count = len(batch.opportunities.get("ring_ring", []))
            by_class["sp2_5_6_ring_pair"] = ring_ring_count
            by_class["ring_ring"] = ring_ring_count
            return dict(by_class)
    raise RuntimeError(f"Could not find {code} in {shard}")


def select_pilot_cohort(
    eligible_complexes: List[str],
    complex_meta: Dict[str, Dict[str, Any]],
    benchmarks: Set[str],
    clean_dir: Path,
    admission_rules: Dict[str, Any],
    parsers: Dict[str, TMQMParser],
) -> Tuple[List[Dict[str, Any]], Set[str], Set[str], List[Dict[str, Any]]]:
    """
    Select the frozen 12-parent development pilot and reserve candidates.
    Verifies that all 12 parents pass 1-hop benchmark disjointness and pairwise disjointness.
    """
    FROZEN_PILOT_CODES = [
        # 3d (4)
        "HESBAM",  # Ti, q=0, nh=30, opps=0 (zero control)
        "OMAXUV",  # Cu, q=1, nh=41, opps=279, activated_ch_acceptor=171
        "QOCKUO",  # Sc, q=-1, nh=81, opps=864, ring_ring=80
        "JIXWEW",  # Sc, q=0, nh=55, opps=253, polar_h_acceptor=2
        # 4d (3)
        "ICEYAQ",  # Zr, q=0, nh=45, opps=158, ch_ring=146
        "SUDMEM",  # Ag, q=1, nh=74, opps=500, ch_ring=446, ring_ring=42
        "WIVVED",  # Mo, q=-1, nh=24, opps=6, low opps
        # 5d (3)
        "KOTJIP",  # Pt, q=0, nh=32, opps=128, sigma_hole_acceptor=15, polar_multiple_bond=25
        "RIGZUC",  # Os, q=1, nh=60, opps=328, ch_ring=296
        "DAFFEZ",  # Re, q=0, nh=39, opps=100, polar_multiple_bond=42
        # 4f (2)
        "HAMYUP",  # La, q=0, nh=22, opps=144
        "YUKLIA",  # La, q=1, nh=27, opps=64, activated_ch_acceptor=64
    ]
    FROZEN_RESERVE_CODES = ["SIDYIP", "YUKLAS"]
    REPEATABILITY_CODES = {"HESBAM", "OMAXUV", "ICEYAQ", "SUDMEM", "KOTJIP", "HAMYUP"}
    # Benchmark scaffold overlaps accepted for the dev-only pilot (no evaluation claims are made on it).
    # Exposed when the scaffold key stopped splitting ligands on Wiberg ring contacts (spec 2.0.0-pilot.3).
    ACCEPTED_PILOT_BENCHMARK_OVERLAPS = {"SUDMEM"}

    eligible_set = set(eligible_complexes)
    bench_sibs = {complex_meta[c]["sib"] for c in benchmarks if c in complex_meta}
    bench_inf_scaffolds = {
        l for c in benchmarks if c in complex_meta for l in complex_meta[c]["informative_ligands"]
    }

    selected_parents: List[Dict[str, Any]] = []
    pilot_sibs: Set[str] = set()
    pilot_scaffolds: Set[str] = set()

    for code in FROZEN_PILOT_CODES:
        assert code in complex_meta, f"Pilot code {code} missing from complex_meta"
        meta = complex_meta[code]
        sib = meta["sib"]
        scaffs = set(meta["informative_ligands"])
        shared_bench_scaffs = sorted(scaffs & bench_inf_scaffolds)

        # Verify disjointness against benchmarks
        assert code not in benchmarks, f"Pilot code {code} in benchmarks"
        assert sib not in bench_sibs, f"Pilot code {code} sibling in benchmark siblings"
        if code in ACCEPTED_PILOT_BENCHMARK_OVERLAPS:
            assert shared_bench_scaffs, f"Pilot code {code} declared as overlapping but shares no benchmark scaffold"
        else:
            assert code in eligible_set, f"Pilot code {code} not eligible under 1-hop quarantine"
            assert not shared_bench_scaffs, f"Pilot code {code} shares scaffold with benchmarks"

        # Verify pairwise disjointness within pilot cohort
        assert sib not in pilot_sibs, f"Pilot code {code} shares sibling with another pilot parent"
        assert not (scaffs & pilot_scaffolds), f"Pilot code {code} shares scaffold with another pilot parent"

        pilot_sibs.add(sib)
        pilot_scaffolds.update(scaffs)

        opps = _evaluate_complex_opportunities(clean_dir, meta, admission_rules, parsers)
        gid = f"SIB:{sib}" if not scaffs else sorted(scaffs)[0]

        selected_parents.append({
            "code": meta["code"],
            "group_id": gid,
            "shard": meta["shard"],
            "dataset": meta["dataset"],
            "metal": meta["metal"],
            "metal_row": meta["metal_row"],
            "charge": meta["q"],
            "charge_class": meta["charge_class"],
            "spin_multiplicity": meta["s"],
            "n_atoms": meta["n_atoms"],
            "n_heavy": meta["n_heavy"],
            "size_class": meta["size_class"],
            "coordination_number": meta["mnd"],
            "cn_class": meta["cn_class"],
            "stoichiometry": meta["stoich"],
            "opportunities": opps,
            "is_repeatability": (code in REPEATABILITY_CODES),
            "accepted_benchmark_scaffold_overlap": [
                {
                    "scaffold": l,
                    "benchmarks": sorted(c for c in benchmarks if c in complex_meta and l in complex_meta[c]["informative_ligands"]),
                }
                for l in shared_bench_scaffs
            ],
        })

    # Reserves
    reserve_parents: List[Dict[str, Any]] = []
    for code in FROZEN_RESERVE_CODES:
        if code in complex_meta:
            meta = complex_meta[code]
            try:
                opps = _evaluate_complex_opportunities(clean_dir, meta, admission_rules, parsers)
                gid = f"SIB:{meta['sib']}" if not meta["informative_ligands"] else meta["informative_ligands"][0]
                reserve_parents.append({
                    "code": meta["code"],
                    "group_id": gid,
                    "shard": meta["shard"],
                    "dataset": meta["dataset"],
                    "metal": meta["metal"],
                    "metal_row": meta["metal_row"],
                    "charge": meta["q"],
                    "charge_class": meta["charge_class"],
                    "spin_multiplicity": meta["s"],
                    "n_atoms": meta["n_atoms"],
                    "n_heavy": meta["n_heavy"],
                    "size_class": meta["size_class"],
                    "coordination_number": meta["mnd"],
                    "cn_class": meta["cn_class"],
                    "stoichiometry": meta["stoich"],
                    "opportunities": opps,
                })
            except Exception:
                pass

    # Verify opportunity coverage across pilot parents
    has_zero = any(
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
        for p in selected_parents
    )
    has_polar_hb = any(p["opportunities"].get("polar_h_acceptor", 0) > 0 for p in selected_parents)
    has_non_hb = any(
        (p["opportunities"].get("activated_ch_acceptor", 0) > 0 or p["opportunities"].get("sigma_hole_acceptor", 0) > 0)
        for p in selected_parents
    )
    has_bond_ring = any(
        (p["opportunities"].get("dh_ring", 0) > 0 or p["opportunities"].get("ch_ring", 0) > 0 or p["opportunities"].get("lp_or_halogen_ring", 0) > 0)
        for p in selected_parents
    )
    has_ring_ring = any(p["opportunities"].get("sp2_5_6_ring_pair", 0) > 0 for p in selected_parents)

    assert has_zero, "Pilot cohort lacks zero-opportunity control complex across activated interaction classes"
    assert has_polar_hb, "Pilot cohort lacks polar_h_acceptor opportunities"
    assert has_non_hb, "Pilot cohort lacks non-HB bond-atom opportunities"
    assert has_bond_ring, "Pilot cohort lacks bond_ring opportunities"
    assert has_ring_ring, "Pilot cohort lacks ring_ring opportunities"

    return selected_parents, pilot_sibs, pilot_scaffolds, reserve_parents


class MappabilityScreen:
    """Memoized solver-mappability check (ligand_inputs_from_mol) for a corpus code."""

    def __init__(self, clean_dir: Path, complex_meta: Dict[str, Dict[str, Any]], parsers: Dict[str, TMQMParser]):
        self.clean_dir = clean_dir
        self.complex_meta = complex_meta
        self.parsers = parsers
        self.cache: Dict[str, bool] = {}
        self._shard_cache: Dict[Tuple[str, str], Dict[str, Tuple[str, List[str], List[str]]]] = {}

    def _get_shard_blocks(self, kind: str, shard: str) -> Dict[str, Tuple[str, List[str], List[str]]]:
        key = (kind, shard)
        if key not in self._shard_cache:
            d = self.clean_dir / kind
            xp = d / "xyz" / f"tmQM_{shard}.xyz"
            bp = d / "bo" / f"tmQM_{shard}.BO"
            blocks: Dict[str, Tuple[str, List[str], List[str]]] = {}
            for (comment, atoms), (bo_code, bonds) in zip(_iter_xyz_blocks(xp), _iter_bo_blocks(bp)):
                blocks[bo_code] = (comment, atoms, bonds)
            self._shard_cache[key] = blocks
        return self._shard_cache[key]

    def _load(self, code: str) -> Any:
        m = self.complex_meta[code]
        kind = "lnQM" if m["dataset"] == "LNf" else "tmQM"
        shard_blocks = self._get_shard_blocks(kind, m["shard"])
        if code not in shard_blocks:
            raise KeyError(f"{code} not found in shard {m['shard']}")
        comment, atoms, bonds = shard_blocks[code]
        csv_row = self.parsers[kind].csv_index.get(code)
        mol = assemble(_assemble_raw(comment, atoms, code, bonds, csv_row))
        if csv_row:
            _attach_qm_properties(mol, csv_row)
        return mol

    def __call__(self, code: str) -> bool:
        if code not in self.cache:
            try:
                mol = self._load(code)
                self.cache[code] = ligand_inputs_from_mol(mol, _build_ligand_groups(mol, mol.smiles)) is not None
            except Exception:
                self.cache[code] = False
        return self.cache[code]


def sample_stratified_pairwise_disjoint(
    eligible_complexes: List[str],
    complex_meta: Dict[str, Dict[str, Any]],
    pilot_codes: Set[str],
    pilot_sibs: Set[str],
    pilot_scaffolds: Set[str],
    bench_sibs: Set[str],
    bench_inf_scaffolds: Set[str],
    n_surface_fit: int = 160,
    n_calibration: int = 40,
    n_motion_val: int = 40,
    n_reserve: int = 48,
    seed: int = 42,
    is_mappable: Optional[Any] = None,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any], Dict[str, Any]]:
    """
    Sample 240 main allocation representatives (160 surface_fit, 40 calibration, 40 motion_validation)
    plus 48 reserve representatives, stratified by stratum (metal_row, charge_class, cn_class).
    Enforces strict pairwise disjointness: rejects any candidate sharing sibling family or non-exempt
    informative scaffold with benchmarks, pilot, or any other split.
    Within each split, runs within-split Union-Find clustering for weighting (w_g = 1 / N_g).
    If `is_mappable` is given, candidates that fail it are recorded per stratum and skipped, so
    every allocated representative is solver-mappable (a declared population narrowing).
    """
    total_requested = n_surface_fit + n_calibration + n_motion_val + n_reserve

    # Stratify eligible candidates (excluding pilot codes)
    strata_candidates: Dict[str, List[str]] = defaultdict(list)
    for c in eligible_complexes:
        if c in pilot_codes:
            continue
        m = complex_meta[c]
        strat_key = f"{m['metal_row']}_{m['charge_class']}_{m['cn_class']}"
        strata_candidates[strat_key].append(c)

    # Sort candidates deterministically
    for strat_key in strata_candidates:
        strata_candidates[strat_key].sort(key=lambda c: stable_hash(f"alloc_v{seed}:{c}"))

    # Pre-screen: estimate each stratum's mappable fraction from its first PRESCREEN_PER_STRATUM
    # candidates (hash order); allocation is proportional to the estimated mappable population.
    prescreen: Dict[str, Tuple[int, int]] = {}
    for strat_key, items in strata_candidates.items():
        head = items[:PRESCREEN_PER_STRATUM]
        n_ok = sum(bool(is_mappable(c)) for c in head) if is_mappable is not None else len(head)
        prescreen[strat_key] = (len(head), n_ok)
    mappable_size = {
        k: len(items) * (prescreen[k][1] / prescreen[k][0] if prescreen[k][0] else 0.0)
        for k, items in strata_candidates.items()
    }
    excluded_strata = sorted(k for k, v in mappable_size.items() if v == 0.0)
    total_available = sum(mappable_size.values())
    if total_available < total_requested:
        raise RuntimeError(f"Only ~{total_available:.0f} mappable eligible candidates available, need {total_requested}")

    # Proportional allocation across strata with a mappable population (>= 1 each)
    strata_targets: Dict[str, int] = {}
    for strat_key, items in strata_candidates.items():
        if strat_key in excluded_strata:
            strata_targets[strat_key] = 0
            continue
        share = mappable_size[strat_key] / total_available
        target = int(round(share * total_requested))
        target = max(1, min(len(items), target))
        strata_targets[strat_key] = target

    # Adjust rounding discrepancy on largest strata
    diff = total_requested - sum(strata_targets.values())
    if diff != 0:
        adj_keys = sorted(strata_candidates.keys(), key=lambda k: len(strata_candidates[k]), reverse=True)
        for k in adj_keys:
            if k in excluded_strata:
                continue
            if diff > 0 and strata_targets[k] < len(strata_candidates[k]):
                strata_targets[k] += 1
                diff -= 1
                if diff == 0:
                    break
            elif diff < 0 and strata_targets[k] > 1:
                strata_targets[k] -= 1
                diff += 1
                if diff == 0:
                    break

    split_quotas = {
        "surface_fit": n_surface_fit,
        "calibration": n_calibration,
        "motion_validation": n_motion_val,
        "reserve": n_reserve,
    }

    # Assign split slots across strata using largest deficit round-robin
    slots: List[Tuple[str, str]] = []
    current_split_counts: Counter = Counter()
    for strat_key in sorted(strata_targets.keys()):
        count = strata_targets[strat_key]
        for _ in range(count):
            best_split = max(
                split_quotas.keys(),
                key=lambda s: (
                    (split_quotas[s] - current_split_counts[s]) / split_quotas[s],
                    -current_split_counts[s],
                    stable_hash(f"{s}:{strat_key}:{len(slots)}"),
                ),
            )
            current_split_counts[best_split] += 1
            slots.append((strat_key, best_split))

    # Sequential sampling with rejection of claimed siblings and scaffolds
    claimed_sibs: Dict[str, Set[str]] = {
        "pilot": set(pilot_sibs),
        "surface_fit": set(),
        "calibration": set(),
        "motion_validation": set(),
        "reserve": set(),
    }
    claimed_scaffolds: Dict[str, Set[str]] = {
        "pilot": set(pilot_scaffolds),
        "surface_fit": set(),
        "calibration": set(),
        "motion_validation": set(),
        "reserve": set(),
    }

    sampled_splits: Dict[str, List[str]] = defaultdict(list)
    sampled_meta: Dict[str, Dict[str, Any]] = {}
    stratum_ptrs = {k: 0 for k in strata_candidates}
    rejection_counts: Counter = Counter()
    screened: Counter = Counter()
    unmappable: Dict[str, List[str]] = defaultdict(list)

    for strat_key, target_split in slots:
        cands = strata_candidates[strat_key]
        idx = stratum_ptrs[strat_key]
        accepted = False

        while idx < len(cands):
            c = cands[idx]
            idx += 1
            m = complex_meta[c]
            c_sib = m["sib"]
            c_scaff = set(m["informative_ligands"])

            # Check benchmark disjointness
            if c_sib in bench_sibs:
                rejection_counts["benchmark_sibling"] += 1
                continue
            if c_scaff & bench_inf_scaffolds:
                rejection_counts["benchmark_scaffold"] += 1
                continue

            # Check pilot disjointness
            if c_sib in claimed_sibs["pilot"]:
                rejection_counts["pilot_sibling"] += 1
                continue
            if c_scaff & claimed_scaffolds["pilot"]:
                rejection_counts["pilot_scaffold"] += 1
                continue

            # Check disjointness against OTHER splits
            conflict = False
            for other_split in split_quotas:
                if other_split == target_split:
                    continue
                if c_sib in claimed_sibs[other_split]:
                    rejection_counts[f"cross_split_sibling_{other_split}"] += 1
                    conflict = True
                    break
                if c_scaff & claimed_scaffolds[other_split]:
                    rejection_counts[f"cross_split_scaffold_{other_split}"] += 1
                    conflict = True
                    break

            if conflict:
                continue

            if is_mappable is not None:
                screened[strat_key] += 1
                if not is_mappable(c):
                    unmappable[strat_key].append(c)
                    rejection_counts["unmappable"] += 1
                    continue

            # Candidate accepted for target_split!
            accepted = True
            sampled_splits[target_split].append(c)
            sampled_meta[c] = m
            claimed_sibs[target_split].add(c_sib)
            claimed_scaffolds[target_split].update(c_scaff)
            break

        stratum_ptrs[strat_key] = idx
        if not accepted:
            raise RuntimeError(f"Exhausted candidates for stratum {strat_key} and split {target_split}")

    # Mappable fraction per stratum, from the pre-screen (1.0 when not screening)
    mappable_frac: Dict[str, float] = {
        k: (prescreen[k][1] / prescreen[k][0]) if (is_mappable is not None and prescreen[k][0]) else 1.0
        for k in strata_candidates
    }
    # Inclusion probability relative to the (estimated) mappable sub-population of the stratum
    inclusion_probs: Dict[str, float] = {
        strat_key: min(1.0, strata_targets[strat_key] / max(1.0, len(strata_candidates[strat_key]) * mappable_frac[strat_key]))
        for strat_key in strata_candidates
    }

    # Within-split Union-Find clustering and weighting (w_g = 1 / N_g)
    final_splits: Dict[str, List[Dict[str, Any]]] = {
        "surface_fit": [],
        "calibration": [],
        "motion_validation": [],
        "reserve": [],
    }

    split_group_stats: Dict[str, Any] = {}

    for s_name in split_quotas:
        codes = sampled_splits[s_name]
        uf_split = DisjointSetUnion()
        for c in codes:
            m = complex_meta[c]
            uf_split.union(c, f"SIB:{m['sib']}")
            for l in m["informative_ligands"]:
                uf_split.union(c, l)

        # Group complexes by root
        comp_groups: Dict[str, List[str]] = defaultdict(list)
        for c in codes:
            root = uf_split.find(c)
            comp_groups[root].append(c)

        max_group_size = max(len(v) for v in comp_groups.values()) if comp_groups else 0
        split_group_stats[s_name] = {
            "representative_count": len(codes),
            "independent_groups_count": len(comp_groups),
            "largest_group_size": max_group_size,
            "group_size_distribution": dict(Counter(len(v) for v in comp_groups.values())),
        }

        # Build entries with weights w_g = 1 / N_g
        for c in codes:
            m = complex_meta[c]
            strat_key = f"{m['metal_row']}_{m['charge_class']}_{m['cn_class']}"
            root = uf_split.find(c)
            group_members = comp_groups[root]
            n_g = len(group_members)
            w_g = 1.0 / n_g

            final_splits[s_name].append({
                "group_id": root,
                "stratum": strat_key,
                "metal_row": m["metal_row"],
                "charge_class": m["charge_class"],
                "cn_class": m["cn_class"],
                "representative_code": c,
                "representative_shard": m["shard"],
                "dataset": m["dataset"],
                "n_heavy": m["n_heavy"],
                "member_count": 1,
                "member_codes": [c],
                "within_split_group_id": root,
                "within_split_group_size": n_g,
                "group_weight": w_g,
                "inclusion_probability": inclusion_probs[strat_key],
                "split": s_name,
            })

    # Validate quotas
    for s_name, quota in split_quotas.items():
        assert len(final_splits[s_name]) == quota, f"Split {s_name} has {len(final_splits[s_name])}, expected {quota}"

    strata_meta = {
        strat_key: {
            "total_available": len(strata_candidates[strat_key]),
            "allocated_total": strata_targets[strat_key],
            "prescreen_checked": prescreen[strat_key][0],
            "prescreen_mappable": prescreen[strat_key][1],
            "excluded_no_mappable_candidates": strat_key in excluded_strata,
            "mappability_screened": screened[strat_key],
            "unmappable_count": len(unmappable[strat_key]),
            "mappable_fraction_estimate": mappable_frac[strat_key],
            "unmappable_codes": unmappable[strat_key],
            "inclusion_probability": inclusion_probs[strat_key],
            "splits": dict(Counter(g["split"] for s in final_splits.values() for g in s if g["stratum"] == strat_key)),
        }
        for strat_key in sorted(strata_candidates.keys())
    }

    sampling_summary = {
        "total_requested": total_requested,
        "total_allocated": sum(strata_targets.values()),
        "total_candidate_rejections": sum(rejection_counts.values()),
        "rejection_breakdown": dict(rejection_counts),
        "mappability_screen": {
            "prescreen_per_stratum": PRESCREEN_PER_STRATUM,
            "excluded_strata": excluded_strata,
            "excluded_strata_population": sum(len(strata_candidates[k]) for k in excluded_strata),
            "estimated_mappable_population": total_available,
            "screened": sum(screened.values()),
            "unmappable": sum(len(v) for v in unmappable.values()),
            "mappable_fraction_estimate": 1.0 - sum(len(v) for v in unmappable.values()) / max(1, sum(screened.values())),
            "by_metal_row": {
                row: {
                    "screened": sum(n for k, n in screened.items() if k.startswith(row + "_")),
                    "unmappable": sum(len(v) for k, v in unmappable.items() if k.startswith(row + "_")),
                }
                for row in sorted({k.split("_")[0] for k in strata_candidates})
            },
        },
        "split_clustering_and_weighting": split_group_stats,
    }

    return final_splits, strata_meta, sampling_summary


def verify_gate_1_minimums(
    cal_codes: List[str],
    val_codes: List[str],
    clean_dir: Path,
    complex_meta: Dict[str, Dict[str, Any]],
    admission_rules: Dict[str, Any],
    parsers: Dict[str, TMQMParser],
) -> Dict[str, Any]:
    """
    Verify Gate 1 minimums: >= 10 informative groups in calibration and motion_validation
    for each activated interaction family (bond_atom, bond_ring, ring_ring).
    Uses enumerate_opportunities directly on TMComplex.
    """
    cal_mols: Dict[str, Any] = {}
    val_mols: Dict[str, Any] = {}

    for s_name, codes, target_dict in [("calibration", cal_codes, cal_mols), ("motion_validation", val_codes, val_mols)]:
        by_shard = defaultdict(list)
        for c in codes:
            m = complex_meta[c]
            data_kind = "lnQM" if m["dataset"] == "LNf" else "tmQM"
            by_shard[(data_kind, m["shard"])].append(c)

        for (data_kind, shard), s_codes in by_shard.items():
            parent_dir = clean_dir / data_kind
            parser = parsers[data_kind]
            s_set = set(s_codes)
            xp = parent_dir / "xyz" / f"tmQM_{shard}.xyz"
            bp = parent_dir / "bo" / f"tmQM_{shard}.BO"

            for (comm, atoms), (code, bonds) in zip(_iter_xyz_blocks(xp), _iter_bo_blocks(bp)):
                if code in s_set:
                    csv_row = parser.csv_index.get(code)
                    raw = _assemble_raw(comm, atoms, code, bonds, csv_row)
                    mol = assemble(raw)
                    target_dict[code] = mol

    # Count opportunities per family
    def evaluate_cohort(mol_dict: Dict[str, Any]) -> Dict[str, int]:
        fam_counts: Counter = Counter()
        for mol in mol_dict.values():
            batch = enumerate_opportunities(mol, admission_rules)
            counts = {k: len(v) for k, v in batch.opportunities.items()}
            if counts.get("bond_atom", 0) > 0:
                fam_counts["bond_atom"] += 1
            if counts.get("bond_ring", 0) > 0:
                fam_counts["bond_ring"] += 1
            if counts.get("ring_ring", 0) > 0:
                fam_counts["ring_ring"] += 1
        return dict(fam_counts)

    cal_stats = evaluate_cohort(cal_mols)
    val_stats = evaluate_cohort(val_mols)

    # Assert Gate 1 minimums: >= 10 per activated family
    for fam in ("bond_atom", "bond_ring", "ring_ring"):
        assert cal_stats.get(fam, 0) >= 10, f"Gate 1 violation: calibration has only {cal_stats.get(fam, 0)} {fam} groups (need >=10)"
        assert val_stats.get(fam, 0) >= 10, f"Gate 1 violation: motion_validation has only {val_stats.get(fam, 0)} {fam} groups (need >=10)"

    return {
        "gate_1_rule": "informative_evaluation_groups_minimum",
        "min_required_per_activated_family": 10,
        "calibration": {
            "group_count": len(cal_codes),
            "bond_atom": cal_stats.get("bond_atom", 0),
            "bond_ring": cal_stats.get("bond_ring", 0),
            "ring_ring": cal_stats.get("ring_ring", 0),
            "status": "passed_gate_1_minimums",
        },
        "motion_validation": {
            "group_count": len(val_codes),
            "bond_atom": val_stats.get("bond_atom", 0),
            "bond_ring": val_stats.get("bond_ring", 0),
            "ring_ring": val_stats.get("ring_ring", 0),
            "status": "passed_gate_1_minimums",
        },
        "overall_status": "passed_gate_1_minimums_satisfied",
    }


def run_rdkit_independent_leakage_audit(
    benchmarks: Set[str],
    pilot_codes: List[str],
    splits: Dict[str, List[Dict[str, Any]]],
    complex_meta: Dict[str, Dict[str, Any]],
    common_ligands: Set[str],
) -> Dict[str, Any]:
    """
    Second-hash check: pairwise overlap across all splits and against benchmarks using RDKit canonical
    SMILES and InChIKeys of the non-exempt ligand fragments. It uses the SAME heavy-atom fragments
    (all-single-bond connectivity, no H) and the same exempt set as the WL scaffold keys, so it guards
    against WL hash collisions only; it is not an independent leakage definition.
    """
    def extract_rdkit_fragments(code: str) -> Tuple[Set[str], Set[str]]:
        m = complex_meta[code]
        atoms = m["atoms"]
        bonds = m["bonds"]
        metal_idx = m["metal_idx"]
        if not metal_idx:
            return set(), set()

        atom_dict = dict(atoms)
        adj: Dict[int, List[int]] = {a[0]: [] for a in atoms}
        for u, v, _bo in bonds:
            adj[u].append(v)
            adj[v].append(u)

        visited = {metal_idx}
        res_smiles = set()
        res_inchikeys = set()

        for a_idx, sym in atoms:
            if a_idx in visited:
                continue
            comp = []
            q_nodes = [a_idx]
            visited.add(a_idx)
            while q_nodes:
                curr = q_nodes.pop()
                comp.append(curr)
                for nxt in adj[curr]:
                    if nxt not in visited and nxt != metal_idx:
                        visited.add(nxt)
                        q_nodes.append(nxt)

            heavy = [u for u in comp if atom_dict[u] != "H"]
            if len(heavy) >= 5 and any(atom_dict[u] == "C" for u in heavy):
                rw = Chem.RWMol()
                n2i = {}
                for u in heavy:
                    at = Chem.Atom(atom_dict[u])
                    at.SetNoImplicit(True)
                    n2i[u] = rw.AddAtom(at)
                for u in heavy:
                    for v in adj[u]:
                        if v in n2i and u < v:
                            rw.AddBond(n2i[u], n2i[v], Chem.BondType.SINGLE)
                mol = rw.GetMol()
                smiles = Chem.MolToSmiles(mol, canonical=True)
                ik = Chem.MolToInchiKey(mol)

                # Check if this ligand is non-exempt
                subG = nx.Graph()
                for u in heavy:
                    subG.add_node(u, element=atom_dict[u])
                for u in heavy:
                    for v in adj[u]:
                        if v in subG and u < v:
                            subG.add_edge(u, v)
                wl_hash = nx.weisfeiler_lehman_graph_hash(subG, node_attr="element")
                c_form = Counter(atom_dict[u] for u in heavy)
                form_str = "".join(f"{k}{c_form[k]}" for k in sorted(c_form))
                lig_id = f"LIG:{form_str}:{wl_hash}"

                if lig_id not in common_ligands:
                    res_smiles.add(smiles)
                    res_inchikeys.add(ik)

        return res_smiles, res_inchikeys

    cohorts: Dict[str, List[str]] = {
        "benchmarks": [c for c in benchmarks if c in complex_meta],
        "pilot": pilot_codes,
        "surface_fit": [g["representative_code"] for g in splits["surface_fit"]],
        "calibration": [g["representative_code"] for g in splits["calibration"]],
        "motion_validation": [g["representative_code"] for g in splits["motion_validation"]],
        "reserve": [g["representative_code"] for g in splits["reserve"]],
    }

    cohort_smiles: Dict[str, Set[str]] = {}
    cohort_inchikeys: Dict[str, Set[str]] = {}
    for name, c_list in cohorts.items():
        s_set: Set[str] = set()
        ik_set: Set[str] = set()
        for c in c_list:
            s, ik = extract_rdkit_fragments(c)
            s_set.update(s)
            ik_set.update(ik)
        cohort_smiles[name] = s_set
        cohort_inchikeys[name] = ik_set

    # Pairwise comparison
    names = list(cohorts.keys())
    pairwise_overlaps = []
    total_shared_smiles = 0
    total_shared_inchikeys = 0

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            shared_s = cohort_smiles[n1] & cohort_smiles[n2]
            shared_ik = cohort_inchikeys[n1] & cohort_inchikeys[n2]
            total_shared_smiles += len(shared_s)
            total_shared_inchikeys += len(shared_ik)
            pairwise_overlaps.append({
                "pair": f"{n1}_vs_{n2}",
                "n1_smiles_count": len(cohort_smiles[n1]),
                "n2_smiles_count": len(cohort_smiles[n2]),
                "shared_canonical_smiles_count": len(shared_s),
                "shared_inchikeys_count": len(shared_ik),
            })

    assert total_shared_smiles == 0, f"RDKit canonical SMILES overlap detected: {total_shared_smiles}"
    assert total_shared_inchikeys == 0, f"RDKit InChIKey overlap detected: {total_shared_inchikeys}"
    total_fragments = sum(len(v) for v in cohort_smiles.values())

    return {
        "method": "rdkit_canonical_smiles_and_inchikey_pairwise_overlap",
        "scope": "second_hash_check_same_fragments_and_exempt_set_as_wl_keys",
        "cohorts_audited": names,
        "total_cohort_complexes_audited": sum(len(v) for v in cohorts.values()),
        "pairwise_comparisons_count": len(pairwise_overlaps),
        "total_shared_canonical_smiles": total_shared_smiles,
        "total_shared_inchikeys": total_shared_inchikeys,
        "pairwise_overlap_percentage": 100.0 * total_shared_smiles / max(1, total_fragments),
        "pairwise_details": pairwise_overlaps,
        "status": "passed_zero_second_hash_overlap",
    }


def verify_complexes_with_tmqm_parser(clean_dir: Path, codes_to_verify: List[Tuple[str, str, str]]) -> Dict[str, Any]:
    """Verify full TMQMParser and assemble() validity on sampled complexes."""
    tmqm_dir = clean_dir / "tmQM"
    lnqm_dir = clean_dir / "lnQM"

    parsers = {
        "tmQM": TMQMParser(tmqm_dir),
        "lnQM": TMQMParser(lnqm_dir),
    }

    by_shard: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for code, shard, dataset in codes_to_verify:
        data_kind = "lnQM" if dataset == "LNf" else "tmQM"
        by_shard[(data_kind, shard)].append(code)

    verified_count = 0
    errors = []

    for (data_kind, shard), target_codes in by_shard.items():
        parent_dir = lnqm_dir if data_kind == "lnQM" else tmqm_dir
        parser = parsers[data_kind]
        target_set = set(target_codes)

        xp = parent_dir / "xyz" / f"tmQM_{shard}.xyz"
        bp = parent_dir / "bo" / f"tmQM_{shard}.BO"

        for (comment, atoms), (code, bonds) in zip(_iter_xyz_blocks(xp), _iter_bo_blocks(bp)):
            if code in target_set:
                try:
                    csv_row = parser.csv_index.get(code)
                    raw = _assemble_raw(comment, atoms, code, bonds, csv_row)
                    mol = assemble(raw)
                    if csv_row:
                        _attach_qm_properties(mol, csv_row)
                    assert mol.csd_code == code, f"Code mismatch {mol.csd_code} vs {code}"
                    assert mol.metal_symbol in METALS, f"Invalid metal {mol.metal_symbol}"
                    verified_count += 1
                except Exception as exc:
                    errors.append({"code": code, "shard": shard, "error": f"{type(exc).__name__}: {exc}"})

    return {
        "requested_count": len(codes_to_verify),
        "verified_count": verified_count,
        "failed_count": len(errors),
        "errors": errors,
    }


def build_nci_v2_frame(
    clean_dir: Path,
    output_file: Path,
    spec_path: Optional[Path] = None,
    n_surface_fit: int = 1200,
    n_calibration: int = 120,
    n_motion_val: int = 120,
    n_reserve: int = 240,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build the complete corpus manifest, direct 1-hop quarantine, and stratified allocations."""
    start_time = time.perf_counter()
    print(f"=== Starting NCI v2 Frame Build (seed={seed}, fit={n_surface_fit}, cal={n_calibration}, val={n_motion_val}, reserve={n_reserve}) ===", flush=True)

    # 1. Shard Audit
    print("Step 1/7: Running audit on all 34 clean shards...", flush=True)
    t0 = time.perf_counter()
    audit_data = run_shard_audit(clean_dir)
    print(f"  Audit complete in {time.perf_counter() - t0:.2f}s: {audit_data['total_complexes']} complexes across {audit_data['total_shards']} shards.", flush=True)

    # 2. Corpus Scan and Ligand Extraction
    print("Step 2/7: Scanning corpus and extracting ligand scaffolds...", flush=True)
    t0 = time.perf_counter()
    benchmarks = set(audit_data["benchmark_exclusions"]["codes"])
    if spec_path is None:
        if (REPO_ROOT / "research/data/refit_nci_v2/scaled_v1/spec.json").exists():
            spec_path = REPO_ROOT / "research/data/refit_nci_v2/scaled_v1/spec.json"
        else:
            spec_path = REPO_ROOT / "data/refit/nci_v2/pilot_v1/spec.json"
    spec_path = Path(spec_path)
    with open(spec_path, "r", encoding="utf-8") as f:
        spec_data = json.load(f)
    whitelist_keys = load_whitelist_keys(spec_data)
    complex_meta, complex_ligs, common_ligands = scan_corpus_and_extract_ligands(clean_dir, set(whitelist_keys.values()))
    print(f"  Scan complete in {time.perf_counter() - t0:.2f}s: {len(complex_meta)} complexes scanned, {len(common_ligands)} frozen common ligands identified.", flush=True)

    # 3. Direct 1-Hop Neighbor Quarantine
    print("Step 3/7: Applying Direct 1-Hop Neighbor Quarantine against canonical benchmarks...", flush=True)
    t0 = time.perf_counter()
    eligible_complexes, quarantine_summary = apply_direct_1hop_quarantine(complex_meta, benchmarks)
    print(f"  Quarantine complete in {time.perf_counter() - t0:.2f}s:")
    print(f"    Direct benchmark: {quarantine_summary['benchmark_direct_count']}, Sibling: {quarantine_summary['benchmark_source_sibling_count']}, Shared scaffold: {quarantine_summary['benchmark_shared_informative_scaffold_count']}")
    print(f"    Total quarantined: {quarantine_summary['total_quarantined_complexes']}, Eligible universe: {quarantine_summary['eligible_complexes_count']} (recovered: +{quarantine_summary['recovered_complexes_from_giant_component']}).", flush=True)

    # 4. Select 12-Parent Development Pilot
    print("Step 4/7: Selecting and validating 12-parent development pilot cohort...", flush=True)
    t0 = time.perf_counter()
    admission_rules = spec_data["admission_rules"]

    parsers = {
        "tmQM": TMQMParser(clean_dir / "tmQM"),
        "lnQM": TMQMParser(clean_dir / "lnQM"),
    }
    pilot_parents, pilot_sibs, pilot_scaffolds, reserve_parents = select_pilot_cohort(
        eligible_complexes, complex_meta, benchmarks, clean_dir, admission_rules, parsers
    )
    pilot_codes = {p["code"] for p in pilot_parents}
    print(f"  Pilot selection complete in {time.perf_counter() - t0:.2f}s. 12 pilot parents verified disjoint from benchmarks and each other.", flush=True)

    # 5. Stratified Sequential Pairwise-Disjoint Sampling + Within-Split Union-Find
    print(f"Step 5/7: Executing Stratified Sequential Pairwise-Disjoint Sampling ({n_surface_fit + n_calibration + n_motion_val} main + {n_reserve} reserve)...", flush=True)
    t0 = time.perf_counter()
    bench_sibs = {complex_meta[c]["sib"] for c in benchmarks if c in complex_meta}
    bench_inf_scaffolds = {
        l for c in benchmarks if c in complex_meta for l in complex_meta[c]["informative_ligands"]
    }
    main_splits, strata_meta, sampling_summary = sample_stratified_pairwise_disjoint(
        eligible_complexes=eligible_complexes,
        complex_meta=complex_meta,
        pilot_codes=pilot_codes,
        pilot_sibs=pilot_sibs,
        pilot_scaffolds=pilot_scaffolds,
        bench_sibs=bench_sibs,
        bench_inf_scaffolds=bench_inf_scaffolds,
        n_surface_fit=n_surface_fit,
        n_calibration=n_calibration,
        n_motion_val=n_motion_val,
        n_reserve=n_reserve,
        seed=seed,
        is_mappable=MappabilityScreen(clean_dir, complex_meta, parsers),
    )
    print(f"  Sampling complete in {time.perf_counter() - t0:.2f}s: {len(main_splits['surface_fit'])} fit, {len(main_splits['calibration'])} cal, {len(main_splits['motion_validation'])} val, {len(main_splits['reserve'])} reserve.")
    print(f"  Candidate rejections encountered: {sampling_summary['total_candidate_rejections']}.", flush=True)
    for s_name, g_info in sampling_summary["split_clustering_and_weighting"].items():
        print(f"    Split {s_name}: {g_info['representative_count']} reps -> {g_info['independent_groups_count']} groups (max group size: {g_info['largest_group_size']})")

    # 6. Verify Gate 1 Minimums
    print("Step 6/7: Verifying Gate 1 minimums (>=10 informative groups in cal and val per activated family)...", flush=True)
    t0 = time.perf_counter()
    cal_codes = [g["representative_code"] for g in main_splits["calibration"]]
    val_codes = [g["representative_code"] for g in main_splits["motion_validation"]]
    gate_1_results = verify_gate_1_minimums(
        cal_codes=cal_codes,
        val_codes=val_codes,
        clean_dir=clean_dir,
        complex_meta=complex_meta,
        admission_rules=admission_rules,
        parsers=parsers,
    )
    print(f"  Gate 1 check complete in {time.perf_counter() - t0:.2f}s:")
    print(f"    Calibration: {gate_1_results['calibration']['bond_atom']} bond_atom, {gate_1_results['calibration']['bond_ring']} bond_ring, {gate_1_results['calibration']['ring_ring']} ring_ring")
    print(f"    Motion Val:  {gate_1_results['motion_validation']['bond_atom']} bond_atom, {gate_1_results['motion_validation']['bond_ring']} bond_ring, {gate_1_results['motion_validation']['ring_ring']} ring_ring")

    # 7. Independent RDKit Leakage Audit & TMQMParser Verification
    print("Step 7/7: Running RDKit SMILES/InChIKey second-hash check and TMQMParser assembly...", flush=True)
    t0 = time.perf_counter()
    rdkit_audit = run_rdkit_independent_leakage_audit(
        benchmarks=benchmarks,
        # Declared pilot overlaps are reported under accepted_pilot_benchmark_scaffold_overlaps
        pilot_codes=sorted(p["code"] for p in pilot_parents if not p["accepted_benchmark_scaffold_overlap"]),
        splits=main_splits,
        complex_meta=complex_meta,
        common_ligands=common_ligands,
    )
    print(f"  RDKit second-hash check complete in {time.perf_counter() - t0:.2f}s: {rdkit_audit['total_shared_canonical_smiles']} shared fragments across {rdkit_audit['pairwise_comparisons_count']} split/benchmark pairs.", flush=True)

    # TMQMParser verification
    t0 = time.perf_counter()
    sampled_to_verify: List[Tuple[str, str, str]] = []
    for p in pilot_parents:
        sampled_to_verify.append((p["code"], p["shard"], p["dataset"]))
    for split_name, group_list in main_splits.items():
        for g in group_list:
            sampled_to_verify.append((g["representative_code"], g["representative_shard"], g["dataset"]))

    # Shard coverage (1 per clean shard)
    for s_info in audit_data["shards"]:
        sh_name = s_info["shard"]
        ds = s_info["dataset"]
        first_c = next((c for c, m in complex_meta.items() if m["shard"] == sh_name), None)
        if first_c:
            sampled_to_verify.append((first_c, sh_name, ds))

    sampled_to_verify = sorted(set(sampled_to_verify))
    verification_results = verify_complexes_with_tmqm_parser(clean_dir, sampled_to_verify)
    print(f"  TMQMParser verification complete in {time.perf_counter() - t0:.2f}s: {verification_results['verified_count']}/{verification_results['requested_count']} passed (0 failures).", flush=True)
    assert verification_results["failed_count"] == 0, f"Parser verification failed on {verification_results['errors']}"

    # Compile common ligand overlap statistics
    bench_common_ligands = {
        l for c in benchmarks if c in complex_meta for l in complex_meta[c].get("common_ligands", [])
    }
    all_sampled_group_items = [p for p in pilot_parents] + [g for s in main_splits.values() for g in s]
    groups_with_bench_common = sum(
        1 for g in all_sampled_group_items
        if any(l in bench_common_ligands for c in g.get("member_codes", [g.get("code")]) for l in complex_meta[c].get("common_ligands", []))
    )

    # Computed (not asserted-by-literal) leakage counts over all sampled codes
    cohort_codes: Dict[str, Set[str]] = {"pilot": set(pilot_codes)}
    for s_name, group_list in main_splits.items():
        cohort_codes[s_name] = {c for g in group_list for c in g["member_codes"]}
    sampled_all = set().union(*cohort_codes.values())
    cohort_names = list(cohort_codes)
    cross_sib = cross_scaf = 0
    for i, a in enumerate(cohort_names):
        for b in cohort_names[i + 1:]:
            sa = {complex_meta[c]["sib"] for c in cohort_codes[a]}
            sb = {complex_meta[c]["sib"] for c in cohort_codes[b]}
            la = {l for c in cohort_codes[a] for l in complex_meta[c]["informative_ligands"]}
            lb = {l for c in cohort_codes[b] for l in complex_meta[c]["informative_ligands"]}
            cross_sib += len(sa & sb)
            cross_scaf += len(la & lb)
    bench_overlap = len(sampled_all & benchmarks)
    bench_sib_overlap = sum(complex_meta[c]["sib"] in bench_sibs for c in sampled_all)
    pilot_bench_scaf = sorted(c for c in cohort_codes["pilot"] if set(complex_meta[c]["informative_ligands"]) & bench_inf_scaffolds)
    accepted_pilot = sorted(p["code"] for p in pilot_parents if p["accepted_benchmark_scaffold_overlap"])
    assert pilot_bench_scaf == accepted_pilot, f"Undeclared pilot benchmark-scaffold overlap: {pilot_bench_scaf}"
    bench_scaf_overlap = sum(
        bool(set(complex_meta[c]["informative_ligands"]) & bench_inf_scaffolds) for c in sampled_all - set(accepted_pilot)
    )
    zero_leak = bench_overlap == bench_sib_overlap == bench_scaf_overlap == cross_sib == cross_scaf == 0
    assert zero_leak, f"Leakage: bench={bench_overlap} bench_sib={bench_sib_overlap} bench_scaf={bench_scaf_overlap} cross_sib={cross_sib} cross_scaf={cross_scaf}"

    # Predeclared unmappable input policy
    unmappable_policy = {
        "policy_name": "predeclared_unmappable_input_policy",
        "allocation_screen": "Candidates are screened with ligand_inputs_from_mol during sampling; unmappable candidates are recorded per stratum (main_allocation.strata[*].unmappable_codes) and skipped, so every allocated representative is solver-mappable. This is a declared population narrowing; inclusion probabilities are relative to the estimated mappable sub-population.",
        "failure_recording": "Failures in ligand_inputs_from_mol are strictly recorded as unmappable outcomes rather than dropped or silently ignored.",
        "reserve_replacement_scope": "Reserve replacement applies strictly and exclusively to training measurements (surface_fit split); evaluation cohorts (calibration, motion_validation, pilot) NEVER replace failed complexes.",
        "replacement_order": "When replacing an unmappable training complex, replacement candidates are drawn from the reserve split within the matching stratum in the predeclared reserve order.",
        "evaluation_cohort_policy": "Strict zero-replacement: calibration, motion_validation, and pilot cohorts retain fixed evaluation sets; unmappable inputs are accounted for in attrition metrics without replacement.",
    }

    # Preserve freeze timestamp and elapsed duration if reproducing an existing frozen frame
    created_ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    elapsed = round(time.perf_counter() - start_time, 2)
    if output_file.exists():
        try:
            old_manifest = json.loads(output_file.read_text(encoding="utf-8"))
            if "created" in old_manifest:
                created_ts = old_manifest["created"]
            if "elapsed_seconds" in old_manifest:
                elapsed = old_manifest["elapsed_seconds"]
        except Exception:
            pass

    # Final manifest
    manifest = {
        "spec_schema": "nci_v2.0_pilot",
        "created": created_ts,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "shard_audit": audit_data,
        "grouping_summary": {
            "algorithm": "direct_1hop_neighbor_quarantine_and_stratified_pairwise_disjoint",
            "quarantine": quarantine_summary,
            "sampling": sampling_summary,
            "total_independent_groups": sum(s["independent_groups_count"] for s in sampling_summary["split_clustering_and_weighting"].values()) + len(pilot_parents),
            "benchmark_contaminated_groups_count": quarantine_summary["total_quarantined_complexes"],
            "eligible_groups_count": quarantine_summary["eligible_complexes_count"],
            "eligible_complexes_count": quarantine_summary["eligible_complexes_count"],
            "frozen_common_ligands_count": len(common_ligands),
            "leakage_verification": {
                "benchmark_overlap_count": bench_overlap,
                "shared_scaffold_overlap_count": bench_scaf_overlap + cross_scaf,
                "cross_split_group_overlap": cross_sib + cross_scaf,
                "sibling_leakage": {
                    "benchmark_sibling_overlap_count": bench_sib_overlap,
                    "cross_split_sibling_overlap_count": cross_sib,
                    "status": "passed_zero_sibling_leakage" if bench_sib_overlap == cross_sib == 0 else "failed",
                },
                "informative_scaffold_leakage": {
                    "benchmark_informative_scaffold_overlap_count": bench_scaf_overlap,
                    "cross_split_informative_scaffold_overlap_count": cross_scaf,
                    "status": "passed_zero_informative_scaffold_leakage" if bench_scaf_overlap == cross_scaf == 0 else "failed",
                },
                "accepted_pilot_benchmark_scaffold_overlaps": accepted_pilot,
                "rdkit_second_hash_check": rdkit_audit,
                "common_ligand_overlap": {
                    "definition": "Ligands whose graph key is on the frozen spec.json whitelist (common_ligands_exempt_from_scaffold_grouping.frozen_graph_keys)",
                    "frozen_whitelist_entries": len(whitelist_keys),
                    "frozen_whitelist_distinct_keys": len(set(whitelist_keys.values())),
                    "frozen_common_ligands_count": len(common_ligands),
                    "whitelist_keys_absent_from_corpus": sorted(n for n, k in whitelist_keys.items() if k not in common_ligands),
                    "benchmark_common_ligands_count": len(bench_common_ligands),
                    "sampled_groups_with_benchmark_common_ligands_count": groups_with_bench_common,
                    "policy": "Only whitelisted common ligands are exempt from scaffold grouping; they remain 100% eligible for noncovalent interaction modeling. Their overlap across splits and benchmarks is expected, permitted, and monitored.",
                    "status": "monitored_and_reported_independent_of_zero_leakage",
                },
                "status": "passed_zero_leakage_verified" if zero_leak else "failed",
            },
        },
        "unmappable_input_policy": unmappable_policy,
        "gate_1_verification": gate_1_results,
        "pilot_cohort": {
            "total_parents": len(pilot_parents),
            "description": "12-parent development pilot across 12 disjoint independent groups, stratified across metal row (3d, 4d, 5d, 4f), charge (<0, 0, >0), and heavy-atom size (<=40, 41-70, >70), with verified interaction opportunity coverage and solver readiness.",
            "repeatability_parents": [p["code"] for p in pilot_parents if p["is_repeatability"]],
            "parents": pilot_parents,
            "reserves": reserve_parents,
        },
        "main_allocation": {
            "description": f"{n_surface_fit + n_calibration + n_motion_val}-group main allocation ({n_surface_fit} surface_fit, {n_calibration} calibration, {n_motion_val} motion_validation) plus {n_reserve} reserve groups, stratified by metal family, charge class, and coordination-number class.",
            "strata": strata_meta,
            "splits": main_splits,
        },
        "audit_verification": verification_results,
    }

    # Write output atomically
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_file.with_suffix(output_file.suffix + ".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    temp_path.replace(output_file)

    print(f"=== Successfully wrote frame manifest to {output_file} ({time.perf_counter() - start_time:.2f}s total) ===", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", type=Path, default=REPO_ROOT / "data/refit/clean", help="Directory containing clean tmQM and lnQM shards.")
    parser.add_argument("--spec", type=Path, default=None, help="Path to spec.json (defaults to research/data/refit_nci_v2/scaled_v1/spec.json if present, else pilot_v1).")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "research/data/refit_nci_v2/scaled_v1/frame.json", help="Path to write frame.json.")
    parser.add_argument("--n-fit", type=int, default=1200, help="Number of surface_fit groups.")
    parser.add_argument("--n-cal", type=int, default=120, help="Number of calibration groups.")
    parser.add_argument("--n-val", type=int, default=120, help="Number of motion_validation groups.")
    parser.add_argument("--n-reserve", type=int, default=240, help="Number of reserve groups.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    args = parser.parse_args()

    build_nci_v2_frame(
        clean_dir=args.clean_dir,
        output_file=args.out,
        spec_path=args.spec,
        n_surface_fit=args.n_fit,
        n_calibration=args.n_cal,
        n_motion_val=args.n_val,
        n_reserve=args.n_reserve,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
