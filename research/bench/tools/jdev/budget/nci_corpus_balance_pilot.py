"""Phase 1 (Step 3): Representative multi-shard corpus sampling-balance pilot
and single-factor ablations B1 (parent-balanced) and B2 (scaffold-balanced).

Streams a stratified hash sample across all 34 cleaned tmQM and LnQM shards
(preserving canonical benchmark exclusions and LnQM sibling weights),
computes raw counts, effective independent parent/scaffold sample sizes (ESS),
bootstrap peak/table uncertainty, and evaluates B1 and B2 as single-factor
ablations on the physical probes and the v2 calibration split.

Writes versioned artifacts under data/refit/reports/n2_corpus_balance_pilot_2026-10-18/.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import zlib

import numpy as np
import torch

_SRC = str(Path(__file__).resolve().parents[5] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
sys.path.insert(0, str(Path(__file__).parent))

from common import REPO, stats_from_db
from pools import setup
from nci_targeted_rescore import CANDIDATE
from nci_training_calibration import contact_indices, scaffold

OUT = REPO / "data/refit/reports/n2_corpus_balance_pilot_2026-10-18"
PROBE_DIR = REPO / "data/refit/nci/2026-10-18_energy_calibration_v2"
TRAIN_V2_DIR = REPO / "data/refit/nci/2026-10-18_training_calibration_v2"
KT = 2.479
PER_SHARD_LIMIT = 150  # 34 shards * 150 = ~5,100 representative structures across tmQM X1-X3 + LnQM


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_representative_sample():
    """Extract contact observations with parent, Murcko scaffold, Ln sibling weight,
    and bound-context metadata across all 34 shards."""
    from rdkit import RDLogger
    from research.archive.code.interactions import nci
    from fit.parser import TMQMParser, _iter_xyz_blocks, _iter_bo_blocks, _assemble_raw, _attach_qm_properties
    from jacobik.topology.mapping import assemble
    from jacobik.builder.ligands import _build_ligand_groups
    from jacobik.topology.graph import donor_indices_of, get_ligand_neighbors
    from fit.pipeline.extract import benchmark_codes, sibling_weights

    RDLogger.DisableLog("rdApp.*")
    torch.set_num_threads(1)

    shards = json.loads((REPO / "data/refit/reports/contacts/shards.json").read_text())
    excluded = set(benchmark_codes())
    # Also record v2 validation codes/scaffolds so we can verify zero leakage when needed
    v2_sel = json.loads((TRAIN_V2_DIR / "selection.json").read_text())
    v2_val_codes = {r["code"] for r in v2_sel["selected"] if r["split"] == "validation"}
    v2_val_scaffolds = {s for r in v2_sel["selected"] if r["split"] == "validation" for s in r["scaffolds"]}

    parsers = {}
    weights_by_dir = {}
    observations = []
    shard_stats = []
    failures = []
    t0 = time.perf_counter()

    for sh in shards:
        d_str = sh["dir"]
        shard_name = sh["shard"]
        dataset = sh["dataset"]
        root = Path(d_str)
        if d_str not in parsers:
            parsers[d_str] = TMQMParser(root)
            weights_by_dir[d_str] = sibling_weights(Path(root) / "tmQM_y.csv")
        parser = parsers[d_str]
        sib_w = weights_by_dir[d_str]

        xyz_path = root / "xyz" / f"tmQM_{shard_name}.xyz"
        bo_path = root / "bo" / f"tmQM_{shard_name}.BO"
        codes_all = re.findall(r"CSD_code = ([^ |\r\n]+)", xyz_path.read_text())
        eligible_codes = [c for c in codes_all if c not in excluded and c not in v2_val_codes]
        chosen_codes = set(
            sorted(eligible_codes, key=lambda c: zlib.crc32(f"nci-balance-pilot-2026-10-18:{c}".encode()))[:PER_SHARD_LIMIT]
        )

        n_used = 0
        n_fail = 0
        for (comment, atoms), (code, bonds) in zip(_iter_xyz_blocks(xyz_path), _iter_bo_blocks(bo_path)):
            if code not in chosen_codes:
                continue
            try:
                csv_row = parser.csv_index.get(code)
                mol = assemble(_assemble_raw(comment, atoms, code, bonds, csv_row))
                if csv_row is not None:
                    _attach_qm_properties(mol, csv_row)
                w_sib = float(sib_w.get(code, 1.0))

                # Build ligand Murcko scaffolds when SMILES is available (tmQM); for LnQM fallback to topological ring/graph hash
                owner = {}
                if getattr(mol, "smiles", None):
                    groups = _build_ligand_groups(mol, mol.smiles)
                    for g in groups:
                        core = scaffold(g["canonical_smiles"])
                        for idx0 in g["xyz_indices_0based"]:
                            owner[mol.atoms[idx0].idx] = core
                else:
                    # LnQM has no CSV SMILES: group by connected ligand component formula + ring sizes
                    nb_lig = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)
                    sym_map = {a.idx: a.symbol for a in mol.atoms}
                    seen_at = set()
                    for a in mol.atoms:
                        if a.idx == mol.metal_idx or a.idx in seen_at:
                            continue
                        comp = []
                        q = [a.idx]
                        seen_at.add(a.idx)
                        while q:
                            u = q.pop()
                            comp.append(u)
                            for v in nb_lig.get(u, ()):
                                if v not in seen_at:
                                    seen_at.add(v)
                                    q.append(v)
                        formula = "".join(f"{el}{cnt}" for el, cnt in sorted(Counter(sym_map[u] for u in comp).items()))
                        deg_sig = "".join(str(len(nb_lig.get(u, ()))) for u in sorted(comp, key=lambda u: (sym_map[u], len(nb_lig.get(u, ())))))
                        core = f"ln_graph:{formula}:{hashlib.md5(deg_sig.encode()).hexdigest()[:8]}"
                        for u in comp:
                            owner[u] = core

                cands = nci.candidates(mol)
                pos = {a.idx: i for i, a in enumerate(mol.atoms)}
                idx = nci.index_tensors(cands, pos, "cpu")
                if not idx:
                    n_used += 1
                    continue
                x = torch.tensor(np.array([a.position for a in mol.atoms], dtype=np.float64)[None])
                bound = set(donor_indices_of(mol))
                nb = get_ligand_neighbors(mol.atoms, mol.bonds, mol.metal_idx)

                for typ, entry in idx.items():
                    r, a = nci.features(typ, x, entry)
                    r = r[0].numpy()
                    a = a[0].numpy()
                    lo, dr, nr, al, da, na = nci._grid(typ)
                    in_range = np.flatnonzero((r >= lo) & (r < lo + dr * nr))
                    if len(in_range) == 0:
                        continue
                    for j in in_range:
                        c = cands[typ][j]
                        c_atoms = contact_indices(typ, c)
                        if any(at not in owner for at in c_atoms):
                            continue
                        scaf_list = sorted(set(owner[at] for at in c_atoms))
                        if any(s in v2_val_scaffolds for s in scaf_list):
                            continue
                        family = repr(scaf_list)
                        bi = int(np.clip((r[j] - lo) / dr, 0, nr - 1))
                        bj = int(np.clip((a[j] - al) / da, 0, na - 1))
                        if typ == "pp":
                            n_bound = int(set(c[0]) <= bound) + int(set(c[1]) <= bound)
                        elif typ == "chp":
                            n_bound = int(set(c[1]) <= bound)
                        else:
                            n_bound = int(c[0] in bound) + int(c[2] in bound)
                        for key_tup in c[-1]:
                            observations.append(
                                dict(
                                    code=code,
                                    dataset=dataset,
                                    w_sib=w_sib,
                                    type=typ,
                                    key="|".join(key_tup),
                                    family=family,
                                    i=bi,
                                    j=bj,
                                    n_bound=n_bound,
                                )
                            )
                n_used += 1
            except Exception as exc:
                n_fail += 1
                if len(failures) < 50:
                    failures.append(dict(code=code, shard=shard_name, error=f"{type(exc).__name__}: {exc}"))
        shard_stats.append(dict(shard=shard_name, dataset=dataset, used=n_used, failed=n_fail))
        print(f"Shard {shard_name} ({dataset}): {n_used} complexes, {len(observations)} obs ({time.perf_counter()-t0:.1f}s)", flush=True)

    return observations, shard_stats, failures


def analyze_balance(observations: list, stats) -> dict:
    """Compare raw, parent-balanced (B1), and scaffold-balanced (B2) tables across keys,
    including bootstrap peak/table uncertainty and ESS."""
    from research.archive.code.interactions import nci

    full_tables = stats.nci_tables()
    by_key = defaultdict(list)
    for o in observations:
        by_key[o["key"]].append(o)

    key_summaries = {}
    fitted_by_mode = {"raw": {}, "B1_parent_balanced": {}, "B2_scaffold_balanced": {}, "pp_exclude_cobound_faces": {}}

    for key_str, obs in sorted(by_key.items()):
        key_tup = tuple(key_str.split("|"))
        typ = key_tup[0]
        lo, dr, nr, al, da, na = nci._grid(typ)
        w_sib = np.array([o["w_sib"] for o in obs], dtype=np.float64)
        raw_weighted_sum = float(w_sib.sum())
        parents = [o["code"] for o in obs]
        uniq_parents = sorted(set(parents))
        n_parents = len(uniq_parents)
        families = [o["family"] for o in obs]
        uniq_families = sorted(set(families))
        n_families = len(uniq_families)

        # Count contacts per parent and per (parent, family)
        per_parent_cnt = Counter(parents)
        parent_sib_w = {o["code"]: o["w_sib"] for o in obs}
        per_pf_cnt = Counter((o["code"], o["family"]) for o in obs)
        parents_in_fam = defaultdict(set)
        for o in obs:
            parents_in_fam[o["family"]].add(o["code"])
        fam_sib_mass = {f: sum(parent_sib_w[p] for p in ps) for f, ps in parents_in_fam.items()}

        # Mode weights (preserving LnQM sibling weights)
        w_raw = w_sib.copy()
        # B1: each parent complex gets weight w_sib(parent), split equally across its contacts in this key
        w_b1 = np.array([o["w_sib"] / per_parent_cnt[o["code"]] for o in obs], dtype=np.float64)
        # B2: each scaffold family gets equal weight (1 / sqrt or 1 / n_parents_in_fam damped so singletons don't explode,
        # and exact equal-family mass as in n2_density_balance_audit.py)
        w_b2 = np.array(
            [o["w_sib"] / per_pf_cnt[(o["code"], o["family"])] / max(fam_sib_mass[o["family"]], 1e-12) for o in obs],
            dtype=np.float64,
        )

        modes = {
            "raw": w_raw,
            "B1_parent_balanced": w_b1,
            "B2_scaffold_balanced": w_b2,
        }
        if typ == "pp":
            modes["pp_exclude_cobound_faces"] = w_raw * np.array([o["n_bound"] < 2 for o in obs], dtype=np.float64)
            modes["pp_unbound_only"] = w_raw * np.array([o["n_bound"] == 0 for o in obs], dtype=np.float64)

        mode_results = {}
        raw_tab = None
        ii = np.array([o["i"] for o in obs], dtype=np.int64)
        jj = np.array([o["j"] for o in obs], dtype=np.int64)

        for mname, w in modes.items():
            if w.sum() <= 0:
                continue
            # Match total weighted mass to raw_weighted_sum ONLY to hold pseudocount shrinkage fixed
            # while reporting true ESS_parent and ESS_family separately.
            w_scaled = w * (raw_weighted_sum / w.sum())
            h = np.zeros((nr, na), dtype=np.float64)
            np.add.at(h, (ii, jj), w_scaled)
            tab_dict = nci.fit_table(typ, h)
            tab = np.asarray(tab_dict["s"], dtype=np.float64)
            if mname == "raw":
                raw_tab = tab

            # Effective sample sizes (parents and families)
            pm = defaultdict(float)
            fm = defaultdict(float)
            for o, val in zip(obs, w):
                pm[o["code"]] += float(val)
                fm[o["family"]] += float(val)
            ess_parent = float(sum(pm.values()) ** 2 / max(sum(v * v for v in pm.values()), 1e-12))
            ess_family = float(sum(fm.values()) ** 2 / max(sum(v * v for v in fm.values()), 1e-12))

            peak = np.unravel_index(int(np.argmax(tab)), tab.shape)
            full_ref = np.asarray(full_tables[key_tup]["s"], dtype=np.float64) if key_tup in full_tables else None
            mode_results[mname] = dict(
                ess_parents=ess_parent,
                ess_scaffold_families=ess_family,
                peak_r_A=float(lo + dr * (peak[0] + 0.5)),
                peak_angle_deg=float(al + da * (peak[1] + 0.5)),
                peak_score_nats=float(tab[peak]),
                rms_vs_sample_raw=float(np.sqrt(np.mean((tab - raw_tab) ** 2))),
                max_abs_vs_sample_raw=float(np.max(np.abs(tab - raw_tab))),
                rms_vs_full_125k_table=float(np.sqrt(np.mean((tab - full_ref) ** 2))) if full_ref is not None else None,
            )

            # Store table if key is in the 13 candidate tables (or fallback)
            if mname in fitted_by_mode and key_tup in full_tables:
                # Also compute a shrinkage-respecting version where count mass is NOT artificially inflated
                # beyond effective independent evidence when comparing on probes/calibration
                fitted_by_mode[mname][key_tup] = tab_dict

        # Parent-cluster bootstrap (100 replicates) on pooled keys to quantify sampling uncertainty of B1/B2 shifts
        boot_summary = None
        if key_tup in (("hb", "*", "*"), ("hb", "O", "O"), ("hb", "N", "O"), ("pp", "*"), ("chp", "*")) and n_parents >= 20:
            rng = np.random.default_rng(20261018)
            parent_to_idx = defaultdict(list)
            for idx_o, o in enumerate(obs):
                parent_to_idx[o["code"]].append(idx_o)
            p_list = list(parent_to_idx.keys())
            b1_rms_draws = []
            b2_rms_draws = []
            raw_peak_r_draws = []
            b1_peak_r_draws = []
            b2_peak_r_draws = []
            for _ in range(100):
                sampled_p = rng.choice(p_list, size=len(p_list), replace=True)
                sel_indices = [idx_o for p in sampled_p for idx_o in parent_to_idx[p]]
                if not sel_indices:
                    continue
                sel_idx = np.array(sel_indices, dtype=np.int64)
                tabs_b = {}
                for mname, w_base in [("raw", w_raw), ("B1", w_b1), ("B2", w_b2)]:
                    wb = w_base[sel_idx]
                    wb = wb * (raw_weighted_sum / max(wb.sum(), 1e-12))
                    hb = np.zeros((nr, na), dtype=np.float64)
                    np.add.at(hb, (ii[sel_idx], jj[sel_idx]), wb)
                    tb = np.asarray(nci.fit_table(typ, hb)["s"], dtype=np.float64)
                    tabs_b[mname] = tb
                b1_rms_draws.append(float(np.sqrt(np.mean((tabs_b["B1"] - tabs_b["raw"]) ** 2))))
                b2_rms_draws.append(float(np.sqrt(np.mean((tabs_b["B2"] - tabs_b["raw"]) ** 2))))
                raw_peak_r_draws.append(float(lo + dr * (np.unravel_index(int(np.argmax(tabs_b["raw"])), (nr, na))[0] + 0.5)))
                b1_peak_r_draws.append(float(lo + dr * (np.unravel_index(int(np.argmax(tabs_b["B1"])), (nr, na))[0] + 0.5)))
                b2_peak_r_draws.append(float(lo + dr * (np.unravel_index(int(np.argmax(tabs_b["B2"])), (nr, na))[0] + 0.5)))
            boot_summary = dict(
                b1_vs_raw_rms_ci95=[float(x) for x in np.quantile(b1_rms_draws, [0.025, 0.975])],
                b2_vs_raw_rms_ci95=[float(x) for x in np.quantile(b2_rms_draws, [0.025, 0.975])],
                raw_peak_r_ci95=[float(x) for x in np.quantile(raw_peak_r_draws, [0.025, 0.975])],
                b1_peak_r_ci95=[float(x) for x in np.quantile(b1_peak_r_draws, [0.025, 0.975])],
                b2_peak_r_ci95=[float(x) for x in np.quantile(b2_peak_r_draws, [0.025, 0.975])],
            )

        key_summaries[key_str] = dict(
            raw_contacts=len(obs),
            sibling_weighted_contacts=raw_weighted_sum,
            independent_parents=n_parents,
            scaffold_families=n_families,
            in_candidate_13_tables=bool(key_tup in full_tables),
            bound_context_counts=dict(Counter(o["n_bound"] for o in obs)),
            largest_parent_share=float(max(per_parent_cnt.values()) / len(obs)),
            largest_family_share=float(max(Counter(families).values()) / len(obs)),
            modes=mode_results,
            bootstrap_uncertainty=boot_summary,
        )

    # Fill any of the 13 tables that had <1000 counts in the 5,100-complex sample with their fallback in that mode
    for mname in ("raw", "B1_parent_balanced", "B2_scaffold_balanced"):
        for key_tup in full_tables:
            if key_tup not in fitted_by_mode[mname]:
                fitted_by_mode[mname][key_tup] = full_tables[key_tup]
    for key_tup in full_tables:
        if key_tup not in fitted_by_mode["pp_exclude_cobound_faces"]:
            fitted_by_mode["pp_exclude_cobound_faces"][key_tup] = fitted_by_mode["raw"][key_tup]

    return key_summaries, fitted_by_mode


def evaluate_balance_ablations_on_v2_cal(fitted_by_mode: dict) -> dict:
    """Evaluate sample-raw, B1_parent_balanced, B2_scaffold_balanced, and pp_exclude_cobound_faces
    on the 16-per-type calibration split of 2026-10-18_training_calibration_v2."""
    from research.archive.code.interactions import nci
    from jacobik.topology.decompose import decompose_ligand
    from nci_training_calibration import stream
    from nci_training_analysis import metrics as base_metrics, paired_ci

    selection = json.loads((TRAIN_V2_DIR / "selection.json").read_text())
    cal_items = [r for r in selection["selected"] if r["split"] == "calibration"]
    chosen = {(r["shard"], r["code"]): dict(code=r["code"], shard=r["shard"], dir=r["directory"]) for r in cal_items}
    mols = {}
    for row, mol, err in stream(chosen.values()):
        assert err is None and mol is not None
        for lig in mol.ligands:
            decompose_ligand(mol, lig)
        mols[row["code"]] = mol

    summary = {}
    for t in nci.TYPES:
        ds = [r for r in cal_items if r["type"] == t]
        modes_to_test = ["sample_raw", "B1_parent_balanced", "B2_scaffold_balanced"]
        if t == "pp":
            modes_to_test.append("pp_exclude_cobound_faces")

        records = []
        for r in ds:
            code = r["code"]
            mol = mols[code]
            npz_path = TRAIN_V2_DIR / f"{t}_{code}.npz"
            meta = json.loads(npz_path.with_suffix(".json").read_text())
            energy = np.array(json.loads(npz_path.with_suffix(".mace64.json").read_text())["relative"], dtype=float)
            with np.load(npz_path) as z:
                X = z["X"]
                old = -KT * z["off_total"].astype(float)
                base = old - KT * (z["on_Q6"].astype(float) - z["off_Q6"].astype(float)) if t == "hb" else old.copy()
                full_125k = -KT * z["NCI_" + t].astype(float)
            shift = lambda arr: arr - arr[0]
            xt = torch.tensor(X, dtype=torch.float32)
            rec = dict(request=r, meta=meta, energy=energy, old=shift(old), base=shift(base), full_125k=shift(full_125k))
            for mname in modes_to_test:
                tab_key = "raw" if mname == "sample_raw" else mname
                rt = nci.build_runtime(mol, fitted_by_mode[tab_key], "cpu", types=(t,))
                s_val = nci.evaluate(rt, xt).numpy() if rt else np.zeros(len(X))
                rec[mname] = shift(-KT * s_val)
            records.append(rec)

        off_mr = [base_metrics(d, d["old"]) for d in records]
        summary[t] = {
            "off": dict(
                mean_case_rmse=float(np.mean([m["rmse"] for m in off_mr])),
                mean_sampled_minimum_regret=float(np.mean([m["sampled_minimum_regret"] for m in off_mr])),
                wrong_sign_curvatures=int(sum(m["wrong_sign_curvatures"] for m in off_mr)),
            )
        }
        for mname in ["full_125k"] + modes_to_test:
            num = sum(float(d[mname][1:] @ (d["energy"][1:] - d["base"][1:])) / (len(d["energy"]) - 1) for d in records)
            den = sum(float(d[mname][1:] @ d[mname][1:]) / (len(d["energy"]) - 1) for d in records)
            coeff = max(0.0, num / den) if den > 1e-12 else 0.0
            summary[t][mname] = dict(fitted_coefficient=coeff)
            for smode, w_s in [("unit", 1.0), ("calibrated", coeff)]:
                mr = [base_metrics(d, d["base"] + w_s * d[mname]) for d in records]
                ratios = [v for m in mr for v in m["curvature_energy_over_surrogate"]]
                delta = [off_mr[i]["rmse"] - mr[i]["rmse"] for i in range(len(records))]
                summary[t][mname][smode] = dict(
                    mean_case_rmse=float(np.mean([m["rmse"] for m in mr])),
                    rmse_improvement_ci95=paired_ci(delta),
                    mean_sampled_minimum_regret=float(np.mean([m["sampled_minimum_regret"] for m in mr])),
                    wrong_sign_curvatures=int(sum(m["wrong_sign_curvatures"] for m in mr)),
                    median_curvature_ratio=float(np.median(ratios)) if ratios else None,
                )
    return summary


def run():
    from tools.jdev.env import tree_hash

    OUT.mkdir(parents=True, exist_ok=True)
    stats = stats_from_db(CANDIDATE)
    observations, shard_stats, failures = extract_representative_sample()
    key_summaries, fitted_by_mode = analyze_balance(observations, stats)
    cal_eval = evaluate_balance_ablations_on_v2_cal(fitted_by_mode)

    report = dict(
        source_hash=tree_hash(),
        candidate_db_sha256=digest(CANDIDATE),
        per_shard_limit=PER_SHARD_LIMIT,
        total_complexes_used=sum(s["used"] for s in shard_stats),
        total_contact_observations=len(observations),
        shard_stats=shard_stats,
        failures=failures,
        key_summaries=key_summaries,
        v2_calibration_evaluation=cal_eval,
    )
    write_json(OUT / "corpus_balance_report.json", report)
    print("Pooled key summaries:")
    for k in ("hb|*|*", "hb|O|O", "hb|N|O", "pp|*", "chp|*"):
        if k in key_summaries:
            print(k, json.dumps(key_summaries[k]["modes"], indent=2), flush=True)
    print("v2 calibration evaluation:")
    print(json.dumps(cal_eval, indent=2), flush=True)


if __name__ == "__main__":
    setup()
    run()
