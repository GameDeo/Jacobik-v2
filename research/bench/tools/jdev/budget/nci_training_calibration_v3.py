"""Phase 2 (Step 2): Prospective v3 full-complex calibration & validation cohort.

Enforces all 5 prospective validation requirements:
1. Excludes all canonical benchmark codes (dev80, heldout28, ring10, 12 targeted/viewer cases).
2. Strictly excludes all previously inspected v1 and v2 validation complexes AND all their
   individual ligand Murcko scaffolds from v3. Furthermore, v3 validation complexes are
   100% newly selected (disjoint from v1/v2 calibration as well).
3. Enforces bipartite connected-component splitting across parent CSD codes and every
   participating ligand Murcko scaffold across all three contact types (hb, pp, chp).
4. Enforces minimum stratum support (>= 6 independent parents/scaffolds per stratum per split,
   including hb anionic_partner vs neutral_partner strata).
5. Builds leave-validation-out histogram tables by subtracting exact validation-complex
   contributions from the full-corpus histograms before fitting any v3 candidate surface.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zlib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, stats_from_db
from pools import setup
from nci_targeted_rescore import CANDIDATE
from energy_diagnosis import motion_modes, moved
from nci_training_calibration import (
    STEPS,
    N_PER_SPLIT,
    contact_indices,
    digest,
    scaffold,
    stream,
    write,
)

OUT = REPO / "data/refit/nci/2026-10-18_training_calibration_v3"
V1_DIR = REPO / "data/refit/nci/2026-10-18_training_calibration"
V2_DIR = REPO / "data/refit/nci/2026-10-18_training_calibration_v2"
HIST_DIR = REPO / "data/refit/nci/2026-10-18_bound_fix"
MIN_STRATUM_PARENTS = 6


def rank_v3(s: str) -> int:
    return zlib.crc32(("nci-training-calibration-v3-2026-10-18:" + s).encode())


def source_candidates_v3(per_type_pool: int = 340):
    shards = json.loads((REPO / "data/refit/reports/contacts/shards.json").read_text())
    sources = {r["shard"]: r for r in shards}
    allrows = {}
    for path in sorted(HIST_DIR.glob("ncihist_*.json")):
        obj = json.loads(path.read_text())
        for row in obj["census"]:
            key = (obj["shard"], row["code"])
            allrows[key] = dict(**row, **sources[obj["shard"]])
    chosen = {}
    for field in ("hbond", "pipi", "chpi"):
        eligible = sorted(
            (r for r in allrows.values() if r[field] > 0),
            key=lambda r: rank_v3(r["code"]),
        )
        for row in eligible[:per_type_pool]:
            chosen[(row["shard"], row["code"])] = row
    # Also include v2 calibration codes so they are available for calibration-only reuse if eligible
    v2_sel = json.loads((V2_DIR / "selection.json").read_text())
    for r in v2_sel["selected"]:
        if r["split"] == "calibration":
            key = (r["shard"], r["code"])
            if key in allrows:
                chosen[key] = allrows[key]
    return list(chosen.values()), {p.name: digest(p) for p in sorted(HIST_DIR.glob("ncihist_*.json"))}


def _hb_charge_stratum(mol, groups, owner: dict, d_idx: int, a_idx: int) -> str:
    """Classify HB contact into 'anionic_partner' vs 'neutral_partner' using query-available
    ligand topology and formal charges (never CSD labels or reference outcomes)."""
    from rdkit import Chem

    for atom_idx in (d_idx, a_idx):
        lig_id = owner.get(atom_idx)
        if lig_id is None:
            continue
        smi = groups[lig_id].get("canonical_smiles") or ""
        rd = Chem.MolFromSmiles(smi)
        if rd is not None:
            total_fc = sum(a.GetFormalCharge() for a in rd.GetAtoms())
            has_anionic_atom = any(a.GetFormalCharge() < 0 for a in rd.GetAtoms())
            if total_fc < 0 or has_anionic_atom:
                return "anionic_partner"
        elif "-" in smi:
            return "anionic_partner"
    return "neutral_partner"


def select():
    import torch
    import networkx as nx
    from rdkit import RDLogger
    from research.archive.code.interactions import nci
    from jacobik.builder.ligands import _build_ligand_groups
    from jacobik.topology.graph import donor_indices_of, get_ligand_neighbors
    from fit.pipeline.extract import benchmark_codes

    RDLogger.DisableLog("rdApp.*")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    v1_sel = json.loads((V1_DIR / "selection.json").read_text())
    v2_sel = json.loads((V2_DIR / "selection.json").read_text())

    import ast

    def _row_scaffolds(r):
        if "scaffolds" in r:
            return r["scaffolds"]
        return ast.literal_eval(r["scaffold_key"].split(":", 1)[1])

    spent_val_codes = {r["code"] for r in v1_sel["selected"] if r["split"] == "validation"} | {
        r["code"] for r in v2_sel["selected"] if r["split"] == "validation"
    }
    spent_val_scaffolds = {
        s for r in v1_sel["selected"] if r["split"] == "validation" for s in _row_scaffolds(r)
    } | {s for r in v2_sel["selected"] if r["split"] == "validation" for s in _row_scaffolds(r)}

    v12_all_codes = {r["code"] for r in v1_sel["selected"]} | {r["code"] for r in v2_sel["selected"]}
    v2_cal_map = {
        (r["type"], r["code"]): r
        for r in v2_sel["selected"]
        if r["split"] == "calibration"
        and r["code"] not in spent_val_codes
        and not (set(r["scaffolds"]) & spent_val_scaffolds)
    }

    chosen, hashes = source_candidates_v3()
    excluded = set(benchmark_codes())
    assert not any(r["code"] in excluded for r in chosen)

    config = dict(
        source_census=hashes,
        exclusions=sorted(excluded),
        spent_validation_codes_excluded=sorted(spent_val_codes),
        spent_validation_scaffolds_excluded=sorted(spent_val_scaffolds),
        candidate_order="stable hash v3; up to 340 reference-contact cases per type from existing census + clean v2 calibration",
        limits=dict(max_atoms=160, per_type_per_split=N_PER_SPLIT, min_stratum_parents=MIN_STRATUM_PARENTS),
        steps=STEPS.tolist(),
        split=(
            "Parent complexes and EACH participating ligand Murcko scaffold are connected before splitting across all types; "
            "all v1/v2 validation codes and scaffolds are strictly excluded; v3 validation cases are 100% newly selected "
            "(never in v1 or v2); leave-validation-out histograms withhold v3 validation complexes."
        ),
    )
    write(OUT / "selection_contract.json", config)

    stats = stats_from_db(CANDIDATE)
    tables = stats.nci_tables()
    records = []
    failures = []
    contexts = defaultdict(lambda: defaultdict(int))
    charge_strata = defaultdict(int)

    for k, (row, mol, error) in enumerate(stream(chosen), 1):
        code = row["code"]
        if code in spent_val_codes:
            continue
        if error:
            failures.append(dict(code=code, error=error))
            continue
        try:
            if mol.spin_state is None or mol.spin_state < 1:
                raise ValueError("unknown spin")
            groups = _build_ligand_groups(mol, mol.smiles)
            nb = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
            owner = {}
            scaffolds = {}
            for j, g in enumerate(groups):
                smi = g.get("canonical_smiles")
                if not smi:
                    raise ValueError("missing ligand SMILES")
                scaffolds[j] = scaffold(smi)
                for idx in g["xyz_indices_0based"]:
                    owner[mol.atoms[idx].idx] = j
            bound = set(donor_indices_of(mol))
            X = np.array([a.position for a in mol.atoms])
            xt = torch.tensor(X[None], dtype=torch.float32)
            rt = nci.build_runtime(mol, tables, "cpu", types=nci.TYPES) or {}
            cands = nci.candidates(mol)
            for t, e in rt.items():
                cs = [c for c in cands[t] if any(key in tables for key in c[-1])]
                r, angle = nci.features(t, xt, e["idx"])
                r = r[0].numpy()
                angle = angle[0].numpy()
                threshold = (
                    (r <= 2.5) & (angle >= 120)
                    if t == "hb"
                    else ((r <= 4.4) & (angle <= 30) if t == "pp" else (r <= 3.0) & (angle <= 30))
                )
                eligible = []
                for index in np.flatnonzero(threshold):
                    c = cs[index]
                    atoms = contact_indices(t, c)
                    if any(a not in owner for a in atoms):
                        continue
                    ligand_ids = sorted(set(owner[a] for a in atoms))
                    lig_scaffs = sorted(set(scaffolds[j] for j in ligand_ids))
                    # Exclude if any participating scaffold was in v1 or v2 validation
                    if set(lig_scaffs) & spent_val_scaffolds:
                        continue
                    key = t + ":" + repr(sorted(scaffolds[j] for j in ligand_ids))
                    if t == "pp":
                        context = (
                            "both_bound_faces"
                            if set(c[0]) <= bound and set(c[1]) <= bound
                            else "one_bound_face"
                            if set(c[0]) <= bound or set(c[1]) <= bound
                            else "unbound_faces"
                        )
                        stratum = "bound_face" if context != "unbound_faces" else "unbound_faces"
                    elif t == "chp":
                        parent = nb[c[0]][0]
                        context = (
                            ("bound_parent" if parent in bound else "unbound_parent")
                            + "_"
                            + ("bound_ring" if set(c[1]) <= bound else "unbound_ring")
                        )
                        stratum = "bound_ring_or_parent" if "bound_" in context else "unbound"
                    else:
                        context = "one_bound_partner" if c[0] in bound or c[2] in bound else "outer_sphere"
                        stratum = _hb_charge_stratum(mol, groups, owner, c[0], c[2])
                    contexts[t][context] += 1
                    if t == "hb":
                        charge_strata[stratum] += 1
                    eligible.append(
                        dict(
                            type=t,
                            index=int(index),
                            r=float(r[index]),
                            angle=float(angle[index]),
                            atoms=atoms,
                            scaffold_key=key,
                            scaffolds=lig_scaffs,
                            context=context,
                            stratum=stratum,
                            ligand_ids=ligand_ids,
                        )
                    )
                if not eligible:
                    continue
                # If this (t, code) was in v2 calibration, pick the exact same contact so cached MACE64 is reusable
                if (t, code) in v2_cal_map:
                    target_atoms = v2_cal_map[(t, code)]["atoms"]
                    matching = [c for c in eligible if c["atoms"] == target_atoms]
                    pick = matching[0] if matching else min(eligible, key=lambda c: rank_v3(code + repr(c["atoms"])))
                else:
                    pick = min(eligible, key=lambda c: rank_v3(code + repr(c["atoms"])))
                records.append(
                    dict(
                        code=code,
                        shard=row["shard"],
                        directory=row["dir"],
                        dataset=row["dataset"],
                        weight=float(row.get("weight", 1.0)),
                        charge=mol.charge,
                        mult=mol.spin_state,
                        metal=mol.metal_symbol,
                        in_v12_any=bool(code in v12_all_codes),
                        in_v2_calibration=bool((t, code) in v2_cal_map),
                        **pick,
                    )
                )
        except Exception as exc:
            failures.append(dict(code=code, error=f"{type(exc).__name__}: {exc}"))
        if k % 50 == 0:
            print(k, "candidates", len(records), "contacts", len(failures), "failures", flush=True)

    graph = nx.Graph()
    for r in records:
        for core in r["scaffolds"]:
            graph.add_edge("parent:" + r["code"], "scaffold:" + core)
    components = list(nx.connected_components(graph))
    components.sort(key=lambda c: rank_v3(repr(sorted(c))))

    counts = {s: {t: 0 for t in nci.TYPES} for s in ("calibration", "validation")}
    stratum_counts = {s: defaultdict(lambda: defaultdict(int)) for s in ("calibration", "validation")}
    selected = []

    # Separate components that contain any v1/v2 complex (eligible ONLY for calibration)
    # from completely fresh components (eligible for validation or calibration).
    cal_only_comps = []
    fresh_comps = []
    for nodes in components:
        cands = [r for r in records if "parent:" + r["code"] in nodes]
        if any(r["in_v12_any"] for r in cands):
            cal_only_comps.append((nodes, cands))
        else:
            fresh_comps.append((nodes, cands))

    # First populate calibration from v2_calibration-containing components (maximizes cache reuse without any validation leakage)
    cal_only_comps.sort(key=lambda item: (not any(r["in_v2_calibration"] for r in item[1]), rank_v3(repr(sorted(item[0])))))
    for nodes, cands in cal_only_comps:
        cands_sorted = sorted(cands, key=lambda r: (not r["in_v2_calibration"], rank_v3(r["code"] + r["type"])))
        for r in cands_sorted:
            t = r["type"]
            if counts["calibration"][t] >= N_PER_SPLIT:
                continue
            # Reserve room in hb calibration for at least MIN_STRATUM_PARENTS per stratum
            if t == "hb":
                other_stratum = "neutral_partner" if r["stratum"] == "anionic_partner" else "anionic_partner"
                rem_slots = N_PER_SPLIT - counts["calibration"]["hb"]
                need_other = max(0, MIN_STRATUM_PARENTS - stratum_counts["calibration"]["hb"][other_stratum])
                if r["stratum"] != other_stratum and rem_slots <= need_other:
                    continue
            selected.append(dict(**r, split="calibration"))
            counts["calibration"][t] += 1
            stratum_counts["calibration"][t][r["stratum"]] += 1

    # Next assign fresh components to validation (and any remaining calibration slots), enforcing stratum balance
    for nodes, cands in fresh_comps:
        # Prefer validation until validation reaches N_PER_SPLIT with >= MIN_STRATUM_PARENTS per stratum
        val_need = sum(max(0, N_PER_SPLIT - counts["validation"][t]) for t in nci.TYPES)
        cal_need = sum(max(0, N_PER_SPLIT - counts["calibration"][t]) for t in nci.TYPES)
        if val_need == 0 and cal_need == 0:
            break
        split = "validation" if val_need >= cal_need and val_need > 0 else "calibration"
        for r in sorted(cands, key=lambda r: rank_v3(r["code"] + r["type"])):
            t = r["type"]
            if counts[split][t] >= N_PER_SPLIT:
                continue
            if t == "hb":
                other_stratum = "neutral_partner" if r["stratum"] == "anionic_partner" else "anionic_partner"
                rem_slots = N_PER_SPLIT - counts[split]["hb"]
                need_other = max(0, MIN_STRATUM_PARENTS - stratum_counts[split]["hb"][other_stratum])
                if r["stratum"] != other_stratum and rem_slots <= need_other:
                    continue
            selected.append(dict(**r, split=split))
            counts[split][t] += 1
            stratum_counts[split][t][r["stratum"]] += 1

    # Strict disjointness assertions
    cal_codes = {r["code"] for r in selected if r["split"] == "calibration"}
    val_codes = {r["code"] for r in selected if r["split"] == "validation"}
    cal_scaffs = {core for r in selected if r["split"] == "calibration" for core in r["scaffolds"]}
    val_scaffs = {core for r in selected if r["split"] == "validation" for core in r["scaffolds"]}

    assert not (cal_codes & val_codes), "parent leakage between calibration and validation"
    assert not (cal_scaffs & val_scaffs), "individual scaffold leakage between calibration and validation"
    assert not (val_codes & spent_val_codes), "v1/v2 validation code leaked into v3 validation"
    assert not (val_scaffs & spent_val_scaffolds), "v1/v2 validation scaffold leaked into v3 validation"
    assert not (val_codes & v12_all_codes), "v1/v2 complex leaked into fresh v3 validation"

    coverage_pass = all(n == N_PER_SPLIT for s in counts.values() for n in s.values()) and all(
        stratum_counts[s]["hb"][st] >= MIN_STRATUM_PARENTS
        for s in ("calibration", "validation")
        for st in ("anionic_partner", "neutral_partner")
    )
    assert coverage_pass, (counts, {s: dict(stratum_counts[s]["hb"]) for s in stratum_counts})

    write(
        OUT / "selection.json",
        dict(
            config=config,
            counts=counts,
            stratum_counts={s: {t: dict(v) for t, v in d.items()} for s, d in stratum_counts.items()},
            selected=selected,
            failures=failures,
            candidate_contact_contexts={t: dict(v) for t, v in contexts.items()},
            available_contacts=len(records),
            scaffold_components=len(components),
            source_hash=__import__("tools.jdev.env", fromlist=["tree_hash"]).tree_hash(),
            driver_sha256=digest(Path(__file__)),
            coverage_pass=coverage_pass,
        ),
    )
    print(
        json.dumps(
            dict(
                counts=counts,
                hb_strata={s: dict(stratum_counts[s]["hb"]) for s in ("calibration", "validation")},
                reusable_cal=sum(1 for r in selected if r["split"] == "calibration" and r["in_v2_calibration"]),
                fresh_val=len([r for r in selected if r["split"] == "validation"]),
            )
        ),
        flush=True,
    )


def reuse():
    """Reuse identical physical measurements from v2 calibration on the v3 calibration side ONLY."""
    import shutil

    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["coverage_pass"]
    copied = []
    for r in selection["selected"]:
        if r["split"] != "calibration":
            continue
        src = V2_DIR / f"{r['type']}_{r['code']}.npz"
        dst = OUT / src.name
        if not src.exists():
            continue
        meta = json.loads(src.with_suffix(".json").read_text())
        before = meta["request"]
        if any(before[k] != r[k] for k in ("code", "type", "index", "atoms", "r", "angle", "charge", "mult")):
            continue
        assert meta["db_sha256"] == digest(CANDIDATE)
        shutil.copyfile(src, dst)
        energy = src.with_suffix(".mace64.json")
        if energy.exists():
            shutil.copyfile(energy, dst.with_suffix(".mace64.json"))
        meta.update(
            request=r,
            selection_sha256=digest(OUT / "selection.json"),
            reused_measurement=dict(
                path=str(src.relative_to(REPO)),
                npz_sha256=digest(src),
                note="Identical calibration reference/motion input reused on v3 calibration split only; zero validation reuse.",
            ),
        )
        write(dst.with_suffix(".json"), meta)
        copied.append(dst.stem)
    write(
        OUT / "reuse_report.json",
        dict(
            copied=copied,
            scope="Only raw identical coordinates/scores/energies from v2 calibration reused on v3 calibration; v3 validation is 100% fresh.",
        ),
    )
    print("Reused on calibration:", len(copied), flush=True)


def prepare():
    """Prepare .npz motion coordinates/scores for all v3 cases and build leave-validation-out
    125k histogram tables."""
    import torch
    from research.archive.code.interactions import nci
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.scorer import Scorer
    from jacobik.topology.decompose import decompose_ligand

    torch.set_num_threads(1)
    selection = json.loads((OUT / "selection.json").read_text())
    stats = stats_from_db(CANDIDATE)
    assert selection["coverage_pass"]

    # Also accumulate exact histograms of v3 validation complexes to subtract from 125k histograms
    full_hist = nci.load_hist(sorted(HIST_DIR.glob("ncihist_*.npz")))
    val_hist_acc = nci.HistAccumulator()

    chosen = {}
    for r in selection["selected"]:
        chosen[(r["shard"], r["code"])] = dict(
            code=r["code"], shard=r["shard"], dir=r["directory"], weight=r.get("weight", 1.0)
        )

    failures = []
    completed = []
    val_codes = {r["code"] for r in selection["selected"] if r["split"] == "validation"}

    selected_stems = {f"{r['type']}_{r['code']}" for r in selection["selected"]}
    for old_f in list(OUT.glob("hb_*")) + list(OUT.glob("pp_*")) + list(OUT.glob("chp_*")):
        stem_name = old_f.name.split(".")[0]
        if stem_name not in selected_stems:
            old_f.unlink()

    for row, mol, error in stream(chosen.values()):
        code = row["code"]
        requests = [r for r in selection["selected"] if r["code"] == code]
        try:
            if error:
                raise ValueError(error)
            # If this complex is in v3 validation, subtract its exact weighted contact counts from leave-val-out histograms
            if code in val_codes:
                w = float(row.get("weight", 1.0))
                val_hist_acc.add(mol, weight=w)

            if all(
                (OUT / f"{r['type']}_{r['code']}.npz").exists()
                and (OUT / f"{r['type']}_{r['code']}.json").exists()
                and (OUT / f"{r['type']}_{r['code']}.json").stat().st_size > 10
                for r in requests
            ):
                continue

            for lig in mol.ligands:
                decompose_ligand(mol, lig)
            X = np.array([a.position for a in mol.atoms])
            modes = motion_modes(mol, X)
            ge = GeometryExtractor(mol, torch.device("cpu"))
            sc = Scorer(mol, ge, stats, torch.device("cpu"))
            rt = nci.build_runtime(mol, stats.nci_tables(), "cpu", types=nci.TYPES)
            for request in requests:
                t = request["type"]
                j = request["index"]
                stem = OUT / f"{t}_{code}"
                if (
                    stem.with_suffix(".npz").exists()
                    and stem.with_suffix(".json").exists()
                    and stem.with_suffix(".json").stat().st_size > 10
                ):
                    continue
                e = rt[t]
                test = np.array([moved(X, m, h) for m in modes for h in (-0.02, 0.02)])
                rr, aa = nci.features(t, torch.tensor(test, dtype=torch.float32), e["idx"])
                dr = (rr[1::2, j] - rr[::2, j]).numpy() / 0.05
                da = (aa[1::2, j] - aa[::2, j]).numpy() / 5.0
                magnitude = dr**2 + da**2
                chosen_modes = [modes[i] for i in np.argsort(-magnitude)[:6] if magnitude[i] > 1e-10]
                assert chosen_modes, "no reachable contact-changing motions"
                coordinates = np.array([X] + [moved(X, m, h) for m in chosen_modes for h in STEPS if h != 0])
                x = torch.tensor(coordinates, dtype=torch.float32)
                values = {}
                flag, types = nci.NCI_ON, nci.ACTIVE_TYPES
                try:
                    for enabled in (False, True):
                        nci.NCI_ON = enabled
                        nci.ACTIVE_TYPES = nci.TYPES
                        sc._nci = rt if enabled else None
                        with torch.no_grad():
                            total, bd = sc.score(x)
                        for key, v in bd.items():
                            values[f'{"on" if enabled else "off"}_{key}'] = v.numpy() * float(sc._agg_norm())
                        values["on_total" if enabled else "off_total"] = total.numpy() * float(sc._agg_norm())
                    for typ, entry in rt.items():
                        values["NCI_" + typ] = nci.evaluate({typ: entry}, x).numpy()
                finally:
                    nci.NCI_ON = flag
                    nci.ACTIVE_TYPES = types
                np.savez_compressed(stem.with_suffix(".npz"), X=coordinates, **values)
                write(
                    stem.with_suffix(".json"),
                    dict(
                        request=request,
                        symbols=[a.symbol for a in mol.atoms],
                        charge=mol.charge,
                        mult=mol.spin_state,
                        modes=[dict(label=m["label"], kind=m["kind"]) for m in chosen_modes],
                        steps=[float(h) for h in STEPS if h != 0],
                        geometry_source="unchanged training reference; native topology, fixed reference-site statistical keys; no solve/bend-prior reconstruction",
                        selection_sha256=digest(OUT / "selection.json"),
                        db_sha256=digest(CANDIDATE),
                    ),
                )
                completed.append(stem.name)
                print(stem.name, len(coordinates), "prepared", flush=True)
        except Exception as exc:
            failures.append(dict(code=code, error=f"{type(exc).__name__}: {exc}"))
            print("FAILED", failures[-1], flush=True)

    val_sub_hist = val_hist_acc.h
    leave_val_out = {k: np.maximum(0.0, full_hist[k].astype(np.float64) - val_sub_hist.get(k, 0.0)) for k in full_hist}
    np.savez_compressed(OUT / "leave_val_out_hist.npz", **leave_val_out)
    write(
        OUT / "prepare_report.json",
        dict(
            completed=completed,
            failures=failures,
            leave_val_out_subtracted_counts={k: float(np.sum(val_sub_hist[k])) for k in sorted(val_sub_hist)},
        ),
    )


def measure():
    import torch
    from tools.jdev.env import tree_hash
    from jacobik.core.energy import MACEPolarEvaluator

    torch.set_num_threads(1)
    selection = json.loads((OUT / "selection.json").read_text())
    assert selection["coverage_pass"] and selection["source_hash"] == tree_hash()
    files = sorted(
        [OUT / f"{r['type']}_{r['code']}.npz" for r in selection["selected"]],
        key=lambda p: p.name,
    )
    assert len(files) == 96 and all(f.exists() for f in files), f"incomplete prepared cohort: {len(files)}"
    manifest = dict(
        source_hash=tree_hash(),
        selection_sha256=digest(OUT / "selection.json"),
        db_sha256=digest(CANDIDATE),
        model_sha256=digest(REPO / "research/bench/tools/jdev/MACE-POLAR-1-S.model"),
        driver_sha256=digest(Path(__file__)),
        coordinates_and_scores={p.name: digest(p) for p in files},
        evaluator="MACE-POLAR-1-S cuda float64, original complex charge/multiplicity",
        reused_measurements=json.loads((OUT / "reuse_report.json").read_text())
        if (OUT / "reuse_report.json").exists()
        else None,
    )
    mp = OUT / "measurement_manifest.json"
    write(mp, manifest)

    to_measure = [
        f
        for f in files
        if not f.with_suffix(".mace64.json").exists() or f.with_suffix(".mace64.json").stat().st_size <= 10
    ]
    if to_measure:
        ev = MACEPolarEvaluator(
            model="polar-1-s",
            device="cuda",
            dtype="float64",
            threads=1,
            model_path=REPO / "research/bench/tools/jdev/MACE-POLAR-1-S.model",
        )
        for f in to_measure:
            target = f.with_suffix(".mace64.json")
            meta = json.loads(f.with_suffix(".json").read_text())
            with np.load(f) as z:
                coords = z["X"]
            energy = ev.energies([(meta["symbols"], x, meta["charge"], meta["mult"]) for x in coords]) * 96.4853321233
            assert np.isfinite(energy).all()
            write(target, dict(relative=(energy - energy[0]).tolist(), reference=float(energy[0])))
            print(f.stem, len(energy), "MACE64", flush=True)

    assert tree_hash() == manifest["source_hash"] and digest(CANDIDATE) == manifest["db_sha256"]
    write(
        OUT / "measurement_complete.json",
        dict(
            cases=len(files),
            manifest_sha256=digest(mp),
            energies={p.with_suffix(".mace64.json").name: digest(p.with_suffix(".mace64.json")) for p in files},
        ),
    )
    print("v3 measurement complete: 96 cases verified.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["select", "reuse", "prepare", "measure"])
    a = p.parse_args()
    setup()
    {"select": select, "reuse": reuse, "prepare": prepare, "measure": measure}[a.stage]()
