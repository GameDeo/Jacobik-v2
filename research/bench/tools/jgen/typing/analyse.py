"""Summarize the T1 re-extractions and the read-only far-field mixture census."""
from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from pathlib import Path as _Path

from tools.jgen.inputs import DATA as _DATA

from . import reextract as _rx
from .reextract import MODES, SPLITS, target_codes

# the T1 analysis runs against the main checkout's data (the src/fit extractor no longer hard-codes these paths)
MAIN = _DATA.parent
ARCH = MAIN / "research/archive/data/refit/observations"
OUT = MAIN / "research/data/jgen/campaign/T1"
_rx.configure(_DATA, OUT, MAIN / "data/refit/frame_v1/frame.json",
              MAIN / "research/bench/benchmarks/locked_exclusions.json")


def _paths(mode, kind):
    return sorted((OUT / mode / "shards").glob(f"*_{kind}.parquet"))


def _counter_from_files(mode, kind, key_cols):
    counts = Counter()
    for path in _paths(mode, kind):
        df = pd.read_parquet(path)
        if not df.empty:
            counts.update(map(tuple, df[key_cols].itertuples(index=False, name=None)))
    return counts


def _row_counts(mode, quantity):
    if quantity == "Q1":
        counts = Counter()
        for p in _paths(mode, "Q1"):
            df = pd.read_parquet(p, columns=["key"])
            counts.update(df.key.astype(str))
        return counts
    if quantity == "Q2":
        counts = Counter()
        for p in _paths(mode, "Q2"):
            df = pd.read_parquet(p, columns=["key"])
            counts.update(df.key.astype(str))
        return counts
    cols = ["k", "ring", "sub", "metal", "geom", "anion"]
    return _counter_from_files(mode, "S_orient", cols)


def table_deltas():
    report = []
    changes = {q: {m: _row_counts(m, q) for m in MODES} for q in ("Q1", "Q2", "S_orient")}
    for q, modes in changes.items():
        base = modes["off"]
        for mode in ("gate", "both"):
            current = modes[mode]
            for key in set(base) | set(current):
                a, b = base.get(key, 0), current.get(key, 0)
                delta = b - a
                rel = abs(delta) / a if a else (float("inf") if b else 0.0)
                if (a == 0) != (b == 0) or abs(delta) >= 20 or rel > .10:
                    report.append(dict(quantity=q, setting=mode, key=repr(key), rows_off=a, rows_after=b,
                                       delta=delta, relative_change=(rel if np.isfinite(rel) else None),
                                       abs_delta=abs(delta), appeared=(a == 0 and b > 0),
                                       disappeared=(a > 0 and b == 0)))
    df = pd.DataFrame(report).sort_values(["quantity", "setting", "abs_delta", "key"],
                                         ascending=[True, True, False, True])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "key_count_changes.csv", index=False)
    return df, changes


def face_census():
    data = {}
    for mode in MODES:
        files = _paths(mode, "faces")
        frames = [pd.read_parquet(p) for p in files if p.stat().st_size]
        data[mode] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    off = data["off"]
    summary = {"n_faces": {}, "ratio_quantiles_by_eta": {}, "ratio_quantiles_by_pair": {},
               "removed_eta2_by_pair": {}, "removed_with_declared_kappa": {}}
    for mode, df in data.items():
        summary["n_faces"][mode] = {str(int(k - 10)): int(n) for k, n in df.k.value_counts().sort_index().items()}
    def quantiles(values):
        x = np.asarray(values, float)
        return {f"p{p}": float(np.percentile(x, p)) for p in (1, 5, 25, 50, 75, 95, 99)} if len(x) else {}
    for eta, g in off.groupby("eta"):
        summary["ratio_quantiles_by_eta"][str(int(eta))] = quantiles(g.max_ratio)
    for pair, g in off.groupby("element_pair"):
        summary["ratio_quantiles_by_pair"][str(pair)] = quantiles(g.max_ratio)
    eta2 = off[off.eta == 2]
    removed = eta2[eta2.removed_by_gate]
    summary["removed_eta2_by_pair"] = {str(k): int(v) for k, v in removed.element_pair.value_counts().sort_index().items()}
    summary["removed_with_declared_kappa"] = {
        "faces": int((removed.n_declared_donors > 0).sum()),
        "donor_atoms": int(removed.n_declared_donors.sum()),
        "by_pair": {str(k): int(v) for k, v in removed[removed.n_declared_donors > 0].element_pair.value_counts().sort_index().items()},
    }
    summary["ratios_eta3_plus_unchanged"] = quantiles(off.loc[off.eta >= 3, "max_ratio"])
    (OUT / "face_census.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, data


def parent_changes():
    parents = {}
    for mode in ("off", "both"):
        rows = []
        for path in sorted((OUT / mode / "shards").glob("*_parents.jsonl")):
            if path.stat().st_size:
                rows.extend(pd.read_json(path, lines=True).to_dict("records"))
        parents[mode] = {r["csd_code"]: r for r in rows}
    locked = json.loads((MAIN / "research/bench/benchmarks/locked_exclusions.json").read_text())
    excluded = set(locked["reserve_all_codes"]) | set(locked["motion_validation_codes"])
    artifacts = {"MIDQEY", "NUHPEL", "UMUBAF", "YUCZUS", "NUWFAM", "CAFWIQ"}
    report = []
    for code in sorted((set(parents["off"]) | set(parents["both"])) & target_codes()):
        old, new = parents["off"].get(code), parents["both"].get(code)
        if old is None or new is None:
            report.append(dict(csd_code=code, changed=True, failure="missing in one setting",
                               known_artifact=code in artifacts, excluded=code in excluded))
            continue
        fields = ("donor_set", "coordination_number", "site_labels", "geometry_label")
        changed = [k for k in fields if old[k] != new[k]]
        if changed:
            report.append(dict(csd_code=code, changed_fields=changed, known_artifact=code in artifacts,
                excluded=code in excluded, off={k:old[k] for k in fields}, both={k:new[k] for k in fields}))
    (OUT / "parent_changes.json").write_text(json.dumps(report, indent=2) + "\n")
    return report, parents


def nuwfam_summary(changes):
    rows = {}
    frames = {}
    for mode in MODES:
        all_rows = []
        for p in _paths(mode, "Q1"):
            df = pd.read_parquet(p, columns=["csd_code", "key", "value"])
            if len(df): all_rows.append(df)
        df = pd.concat(all_rows, ignore_index=True)
        frames[mode] = df
        parsed = df.key.map(lambda x: ast.literal_eval(x) if isinstance(x, str) else tuple(x))
        family = parsed.map(lambda x: tuple(x[:5]) == ("Zr", 1, 4, "C_haptic", "trigonal"))
        center = parsed.map(lambda x: tuple(x[:4]) == ("Zr", 1, 4, "C_haptic"))
        rows[mode] = dict(trigonal_family=int(family.sum()), broad_family=int(center.sum()))
    baseline = frames["off"]
    base_keys = baseline.key.map(lambda x: tuple(ast.literal_eval(x)))
    narrow = baseline[base_keys.map(lambda x: tuple(x[:4]) == ("Zr", 1, 4, "C_haptic")) &
                      baseline.value.between(2.66, 2.92)]
    site_ids = Counter((r.csd_code, round(float(r.value), 6)) for r in narrow.itertuples())
    rows["off"]["rows_2.66_2.92"] = int(sum(site_ids.values()))
    for mode, df in frames.items():
        ids = Counter((r.csd_code, round(float(r.value), 6)) for r in df.itertuples())
        rows[mode]["rows_2.66_2.92_surviving"] = int(sum(min(n, ids[k]) for k, n in site_ids.items()))
    return rows


def farfield_census():
    db_path = MAIN / "data/db/constraints_2026-11-01.json"
    db = json.loads(db_path.read_text())
    components = []
    for q in ("Q1", "Q2"):
        for dbkey, entry in db.items():
            if not dbkey.startswith(q + ":") or "weights" not in entry:
                continue
            weights = np.asarray(entry["weights"], float)
            means = np.asarray(entry.get("means", []), float).reshape(-1)
            vars_ = np.asarray(entry.get("vars", []), float).reshape(-1)
            if len(weights) < 2 or len(means) != len(weights) or len(vars_) != len(weights):
                continue
            sd = np.sqrt(np.maximum(vars_, 0.0))
            main = int(np.argmax(weights))
            for i, w in enumerate(weights):
                if i == main or w >= .15 or sd[i] >= .5 * sd[main] or abs(means[i]-means[main]) <= 2 * sd[main]:
                    continue
                components.append(dict(quantity=q, db_key=dbkey, key=ast.literal_eval(dbkey[len(q)+1:]),
                    n=int(entry.get("n", 0)), source=entry.get("src"), inherited=entry.get("src") == "adapted",
                    component=i, weight=float(w), mean=float(means[i]), sd=float(sd[i]),
                    main_mean=float(means[main]), main_sd=float(sd[main]),
                    distance_main_sd=float(abs(means[i]-means[main])/sd[main]) if sd[main] else None,
                    backoff_parent="not stored" if entry.get("src") == "adapted" else None))
    (OUT / "farfield_components.json").write_text(json.dumps(components, indent=2) + "\n")
    return components


def check_off_reproduction():
    """Require every archived haptic QO row to be matched by flags-off re-extraction."""
    from .reextract import SPLITS

    results = {}
    for split in SPLITS:
        pieces = []
        for p in _paths("off", "QO"):
            shard = p.name.split("_", 1)[0]
            if shard.startswith(split + "s"):
                df = pd.read_parquet(p)
                pieces.append(df[df.k >= 10])
        new = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        old = pd.read_parquet(ARCH / f"observations_QO_{split}.parquet")
        old = old[old.k >= 10]
        def identity(r):
            return (str(r.csd_code), int(r.k), np.asarray(r.raw_face, float).tobytes())
        new_keys = [identity(r) for r in new.itertuples()]
        old_keys = [identity(r) for r in old.itertuples()]
        new_key_set = set(new_keys)
        old_key_set = set(old_keys)
        if len(new_key_set) != len(new_keys):
            raise AssertionError(f"duplicate flags-off haptic keys in {split}")
        missing = sum(k not in new_key_set for k in old_keys)
        result = dict(new_haptic_rows=len(new_keys), archived_haptic_rows=len(old_keys),
                      matched=len(old_keys)-missing, unmatched=missing,
                      extra_new_rows=len(new_key_set-old_key_set))
        results[split] = result
        print(split, json.dumps(result), flush=True)
        if missing:
            raise SystemExit(f"flags-off archive reproduction failed for {split}: {missing} archived haptic rows unmatched")
    (OUT / "flags_off_archive_check.json").write_text(json.dumps(results, indent=2)+"\n")
    return results


def join_farfield_sources(components, face_data, parent_data, include_diversity=False):
    """Join component windows to the archived rows feeding each fitted key."""
    from fit import q1, q2

    targets = defaultdict(list)
    for c in components:
        targets[(c["quantity"], tuple(c["key"]))].append(c)
        c.update(source_rows=0, removed_confirmed=0, removed_possible=0)
        if include_diversity:
            c["_source_parent_counts"] = Counter()
            c["_source_metal_counts"] = Counter()
    bad_faces = face_data["off"]
    bad_faces = bad_faces[(bad_faces.eta == 2) & bad_faces.removed_by_gate]
    removed_by_code = defaultdict(set)
    face_rows = {}
    for p in _paths("off", "QO"):
        df = pd.read_parquet(p, columns=["csd_code", "k", "sub", "raw_face"])
        df = df[df.k == 12]
        for r in df.itertuples():
            raw_face = np.asarray(r.raw_face, float)
            identity = (r.csd_code, int(r.k), raw_face.tobytes())
            centroid_distance = float(np.linalg.norm(raw_face.reshape(-1, 3).mean(axis=0)))
            face_rows[identity] = (str(r.sub), round(centroid_distance, 3))
    for r in bad_faces.itertuples():
        # QO stores raw_face but no scalar value; recover the exact site row
        # by its raw face vector, then use the face centroid distance carried
        # by Q1 for the source-row join.
        identity = (r.csd_code, int(r.k), np.asarray(r.raw_face, float).tobytes())
        match = face_rows.get(identity)
        if match is not None:
            removed_by_code[r.csd_code].add(match)
    face_sub = defaultdict(set)
    for code, entries in removed_by_code.items():
        face_sub[code].update(sub for sub, _ in entries)
    # Count all Q1 coordination sites by subclass for Q2 source-row linkage.
    sites_by_code = defaultdict(Counter)
    for split in SPLITS:
        df = pd.read_parquet(ARCH / f"observations_Q1_{split}.parquet", columns=["csd_code", "key"])
        for r in df.itertuples(index=False):
            try:
                raw_key = tuple(ast.literal_eval(r.key))
                if len(raw_key) >= 4:
                    sites_by_code[r.csd_code][str(raw_key[3])] += 1
            except Exception:
                continue

    for q, ds in (("Q1", q1), ("Q2", q2)):
        for split in SPLITS:
            p = ARCH / f"observations_{q}_{split}.parquet"
            df = pd.read_parquet(p, columns=["csd_code", "key", "value"])
            for r in df.itertuples(index=False):
                try:
                    raw_key = tuple(ast.literal_eval(r.key))
                    ancestors = {tuple(x[1]) for x in ds.ancestor_rows(raw_key)} | {raw_key}
                except Exception:
                    continue
                source_value = (float(np.cos(np.deg2rad(float(r.value))) if q == "Q2" else r.value))
                for target in ancestors:
                    for c in targets.get((q, target), ()):
                        if not (c["mean"] - 2 * c["sd"] <= source_value <= c["mean"] + 2 * c["sd"]):
                            continue
                        c["source_rows"] += 1
                        if include_diversity:
                            c["_source_parent_counts"][str(r.csd_code)] += 1
                            metal_index = 0 if q == "Q1" else 3
                            if len(raw_key) > metal_index:
                                c["_source_metal_counts"][str(raw_key[metal_index])] += 1
                        if q == "Q1" and len(raw_key) >= 4:
                            if (str(raw_key[3]), round(float(r.value), 3)) in removed_by_code.get(r.csd_code, set()):
                                c["removed_confirmed"] += 1
                        elif q == "Q2" and len(raw_key) >= 6:
                            affected = face_sub.get(r.csd_code, set()) & {str(raw_key[4]), str(raw_key[5])}
                            if any(sites_by_code[r.csd_code][sub] == 1 for sub in affected):
                                c["removed_confirmed"] += 1
                            elif affected:
                                c["removed_possible"] += 1
    for c in components:
        c["removed_share_confirmed"] = (c["removed_confirmed"] / c["source_rows"]
                                         if c["source_rows"] else None)
        c["removed_share_possible_upper"] = ((c["removed_confirmed"] + c["removed_possible"]) / c["source_rows"]
                                               if c["source_rows"] else None)
        c["backoff_parent"] = "not stored" if c["inherited"] else None
        if include_diversity:
            parent_counts = c.pop("_source_parent_counts")
            metal_counts = c.pop("_source_metal_counts")
            top_parent = sorted(parent_counts.items(), key=lambda x: (-x[1], x[0]))
            c["distinct_source_parents"] = len(parent_counts)
            c["largest_source_parent"] = top_parent[0][0] if top_parent else None
            c["largest_source_parent_share"] = (top_parent[0][1] / c["source_rows"]
                                                 if top_parent and c["source_rows"] else None)
            c["source_metal_counts"] = dict(sorted(metal_counts.items()))
    (OUT / "farfield_source_rows.json").write_text(json.dumps(components, indent=2) + "\n")
    return components


def t1b_analysis():
    """Chelate-exclusion deltas and a fitted-component-only far-field census."""
    # Face census delta vs the already-computed T1 both-flags setting.
    face_frames = {}
    for mode in ("both", "t1b"):
        parts = [pd.read_parquet(p, columns=["eta", "element_pair", "metal"])
                 for p in _paths(mode, "faces")]
        face_frames[mode] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        face_frames[mode] = face_frames[mode][face_frames[mode].eta == 2]

    def group_counts(df, cols):
        return df.groupby(cols, dropna=False).size().rename("faces").reset_index()

    face_count_tables = {}
    for name, cols in (("pair", ["element_pair"]), ("metal", ["metal"]),
                       ("pair_metal", ["element_pair", "metal"])):
        left = group_counts(face_frames["both"], cols).rename(columns={"faces": "both_faces"})
        right = group_counts(face_frames["t1b"], cols).rename(columns={"faces": "t1b_faces"})
        merged = left.merge(right, on=cols, how="outer").fillna(0)
        merged["both_faces"] = merged["both_faces"].astype(int)
        merged["t1b_faces"] = merged["t1b_faces"].astype(int)
        merged["delta_t1b_minus_both"] = merged["t1b_faces"] - merged["both_faces"]
        merged["excluded_faces"] = -merged["delta_t1b_minus_both"].clip(upper=0)
        face_count_tables[name] = merged.sort_values("excluded_faces", ascending=False)
        face_count_tables[name].to_csv(OUT / f"t1b_face_delta_{name}.csv", index=False)

    # Count-only key deltas: list exactly the keys with |delta| >= 20.
    key_rows = []
    key_counts = {}
    for quantity in ("Q1", "Q2", "S_orient"):
        both = _row_counts("both", quantity)
        t1b = _row_counts("t1b", quantity)
        key_counts[quantity] = {"both": both, "t1b": t1b}
        for key in set(both) | set(t1b):
            old, new = both.get(key, 0), t1b.get(key, 0)
            delta = new - old
            if abs(delta) >= 20:
                key_rows.append(dict(quantity=quantity, key=repr(key), rows_both=old,
                                     rows_t1b=new, delta=delta, abs_delta=abs(delta)))
    key_delta = pd.DataFrame(key_rows).sort_values(
        ["quantity", "abs_delta", "key"], ascending=[True, False, True])
    key_delta.to_csv(OUT / "t1b_key_count_changes.csv", index=False)

    zinc_carboxylate_key = ("Zn", 1, 2, "O_carboxylate", "octahedral", "O_carboxylate")
    key_counts["Q1"]["gate"] = _row_counts("gate", "Q1")
    zinc_counts = {mode: key_counts["Q1"][mode].get(repr(zinc_carboxylate_key), 0)
                   for mode in ("gate", "both", "t1b")}
    zinc_check = dict(key=repr(zinc_carboxylate_key), counts=zinc_counts,
                      within_810_plusminus5=abs(zinc_counts["t1b"] - 810) <= 5,
                      within_gate_plusminus5=abs(zinc_counts["t1b"] - zinc_counts["gate"]) <= 5)

    # Parent snapshots use the same unlocked cal/tail target set as T1.
    parent_modes = {}
    for mode in ("both", "t1b"):
        rows = []
        for path in sorted((OUT / mode / "shards").glob("*_parents.jsonl")):
            if path.stat().st_size:
                rows.extend(pd.read_json(path, lines=True).to_dict("records"))
        parent_modes[mode] = {r["csd_code"]: r for r in rows}
    parent_changes = []
    fields = ("donor_set", "coordination_number", "site_labels", "geometry_label")
    for code in sorted((set(parent_modes["both"]) | set(parent_modes["t1b"])) & target_codes()):
        old, new = parent_modes["both"].get(code), parent_modes["t1b"].get(code)
        if old is None or new is None:
            parent_changes.append(dict(csd_code=code, missing_in="both" if old is None else "t1b"))
            continue
        changed = [field for field in fields if old[field] != new[field]]
        if changed:
            parent_changes.append(dict(csd_code=code, changed_fields=changed,
                both={field: old[field] for field in fields},
                t1b={field: new[field] for field in fields}))
    (OUT / "t1b_parent_changes.json").write_text(json.dumps(parent_changes, indent=2) + "\n")

    # Census only the 791 components fitted directly in the DB (not adapted/back-off copies).
    components = [c for c in farfield_census() if not c["inherited"]]
    fitted_expected = {"Q1": 315, "Q2": 476}
    fitted_counts = Counter(c["quantity"] for c in components)
    faces, face_data = face_census()
    components = join_farfield_sources(components, face_data, {}, include_diversity=True)
    fitted_cols = ["quantity", "db_key", "n", "source", "component", "weight", "mean", "sd",
                   "main_mean", "main_sd", "distance_main_sd", "source_rows", "removed_confirmed",
                   "removed_possible", "removed_share_confirmed", "removed_share_possible_upper",
                   "distinct_source_parents", "largest_source_parent", "largest_source_parent_share",
                   "source_metal_counts"]
    fitted_df = pd.DataFrame(components, columns=fitted_cols)
    fitted_df["source_metal_counts"] = fitted_df.source_metal_counts.map(
        lambda x: json.dumps(x, sort_keys=True) if isinstance(x, dict) else "{}")
    fitted_df.to_csv(OUT / "t1b_fitted_components.csv", index=False)
    diversity = dict(n_components=len(components), components_by_quantity=dict(fitted_counts),
        expected_components_by_quantity=fitted_expected,
        components_with_source_rows=int(fitted_df.source_rows.gt(0).sum()),
        zero_source_row_components=int(fitted_df.source_rows.eq(0).sum()),
        sum_component_source_rows=int(fitted_df.source_rows.sum()),
        total_distinct_parent_counts=int(fitted_df.distinct_source_parents.sum()),
        total_confirmed_removed_rows=int(fitted_df.removed_confirmed.sum()),
        total_possible_removed_rows=int(fitted_df.removed_possible.sum()))
    summary = dict(face_delta_by_pair=face_count_tables["pair"].to_dict("records"),
        face_delta_by_metal=face_count_tables["metal"].to_dict("records"),
        face_delta_by_pair_metal=face_count_tables["pair_metal"].to_dict("records"),
        eta2_faces_both=int(len(face_frames["both"])), eta2_faces_t1b=int(len(face_frames["t1b"])),
        eta2_faces_excluded=int(len(face_frames["both"]) - len(face_frames["t1b"])),
        key_changes_ge20=len(key_delta), key_changes_by_quantity=key_delta.quantity.value_counts().to_dict(),
        zinc_carboxylate=zinc_check, parent_changes=parent_changes,
        fitted_components=diversity, fitted_component_census_csv="t1b_fitted_components.csv")
    (OUT / "t1b_analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in
                      ("face_delta_by_pair", "face_delta_by_metal", "face_delta_by_pair_metal")}, indent=2))
    return summary


def t1c_analysis():
    """Geometric-bound chelate deltas versus the stored T1b extraction."""
    face_frames = {}
    for mode in ("t1b", "t1c"):
        parts = [pd.read_parquet(p, columns=["eta", "element_pair", "metal"])
                 for p in _paths(mode, "faces")]
        face_frames[mode] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        face_frames[mode] = face_frames[mode][face_frames[mode].eta == 2]

    def group_counts(df, cols):
        return df.groupby(cols, dropna=False).size().rename("faces").reset_index()

    face_tables = {}
    for name, cols in (("pair", ["element_pair"]), ("metal", ["metal"]),
                       ("pair_metal", ["element_pair", "metal"])):
        left = group_counts(face_frames["t1b"], cols).rename(columns={"faces": "t1b_faces"})
        right = group_counts(face_frames["t1c"], cols).rename(columns={"faces": "t1c_faces"})
        merged = left.merge(right, on=cols, how="outer").fillna(0)
        merged["t1b_faces"] = merged["t1b_faces"].astype(int)
        merged["t1c_faces"] = merged["t1c_faces"].astype(int)
        merged["delta_t1c_minus_t1b"] = merged["t1c_faces"] - merged["t1b_faces"]
        merged["excluded_faces"] = -merged["delta_t1c_minus_t1b"].clip(upper=0)
        face_tables[name] = merged.sort_values("excluded_faces", ascending=False)
        face_tables[name].to_csv(OUT / f"t1c_face_delta_{name}.csv", index=False)

    key_rows = []
    for quantity in ("Q1", "Q2", "S_orient"):
        t1b = _row_counts("t1b", quantity)
        t1c = _row_counts("t1c", quantity)
        for key in set(t1b) | set(t1c):
            old, new = t1b.get(key, 0), t1c.get(key, 0)
            delta = new - old
            if abs(delta) >= 20:
                key_rows.append(dict(quantity=quantity, key=repr(key), rows_t1b=old,
                                     rows_t1c=new, delta=delta, abs_delta=abs(delta)))
    key_delta = pd.DataFrame(key_rows, columns=["quantity", "key", "rows_t1b", "rows_t1c", "delta", "abs_delta"])
    if len(key_delta):
        key_delta = key_delta.sort_values(["quantity", "abs_delta", "key"],
                                          ascending=[True, False, True])
    key_delta.to_csv(OUT / "t1c_key_count_changes.csv", index=False)

    zinc_key = ("Zn", 1, 2, "O_carboxylate", "octahedral", "O_carboxylate")
    zinc_counts = {mode: _row_counts(mode, "Q1").get(repr(zinc_key), 0)
                   for mode in ("gate", "both", "t1b", "t1c")}
    zinc = dict(key=repr(zinc_key), counts=zinc_counts,
                within_810_plusminus5=abs(zinc_counts["t1c"] - 810) <= 5,
                within_gate_plusminus5=abs(zinc_counts["t1c"] - zinc_counts["gate"]) <= 5)

    parent_modes = {}
    for mode in ("t1b", "t1c"):
        rows = []
        for path in sorted((OUT / mode / "shards").glob("*_parents.jsonl")):
            if path.stat().st_size:
                rows.extend(pd.read_json(path, lines=True).to_dict("records"))
        parent_modes[mode] = {r["csd_code"]: r for r in rows}
    fields = ("donor_set", "coordination_number", "site_labels", "geometry_label")
    parent_changes = []
    for code in sorted((set(parent_modes["t1b"]) | set(parent_modes["t1c"])) & target_codes()):
        old, new = parent_modes["t1b"].get(code), parent_modes["t1c"].get(code)
        if old is None or new is None:
            parent_changes.append(dict(csd_code=code, missing_in="t1b" if old is None else "t1c"))
            continue
        changed = [field for field in fields if old[field] != new[field]]
        if changed:
            parent_changes.append(dict(csd_code=code, changed_fields=changed,
                t1b={field: old[field] for field in fields},
                t1c={field: new[field] for field in fields}))
    (OUT / "t1c_parent_changes.json").write_text(json.dumps(parent_changes, indent=2) + "\n")

    summary = dict(
        eta2_faces_t1b=int(len(face_frames["t1b"])), eta2_faces_t1c=int(len(face_frames["t1c"])),
        eta2_faces_excluded=int(len(face_frames["t1b"]) - len(face_frames["t1c"])),
        face_delta_by_pair=face_tables["pair"].to_dict("records"),
        face_delta_by_metal=face_tables["metal"].to_dict("records"),
        face_delta_by_pair_metal=face_tables["pair_metal"].to_dict("records"),
        key_changes_ge20=len(key_delta),
        key_changes_by_quantity=key_delta.quantity.value_counts().to_dict() if len(key_delta) else {},
        zinc_carboxylate=zinc, parent_changes=parent_changes)
    (OUT / "t1c_analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in
                      ("face_delta_by_pair", "face_delta_by_metal", "face_delta_by_pair_metal")}, indent=2))
    return summary


def t1d_analysis():
    """Compare face-only chelate rejection with separate geometric donor promotion."""
    import json
    from collections import Counter

    modes = ("t1d_base", "t1d_promote")

    def frames(mode, kind, columns=None):
        parts = [pd.read_parquet(p, columns=columns) for p in _paths(mode, kind) if p.stat().st_size]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns or [])

    # Face deltas by pair, metal and pair-metal; compare also to the completed T1c setting.
    face = {mode: frames(mode, "faces", ["eta", "element_pair", "metal"])
            for mode in (*modes, "t1c")}
    face = {mode: df[df.eta == 2] for mode, df in face.items()}
    face_tables = {}
    for name, cols in (("pair", ["element_pair"]), ("metal", ["metal"]),
                       ("pair_metal", ["element_pair", "metal"])):
        counts = {mode: df.groupby(cols, dropna=False).size().rename(mode)
                  for mode, df in face.items()}
        table = pd.concat(counts.values(), axis=1).fillna(0).astype(int).reset_index()
        table["promotion_delta"] = table["t1d_promote"] - table["t1d_base"]
        table["base_vs_t1c"] = table["t1d_base"] - table["t1c"]
        table.to_csv(OUT / f"t1d_face_delta_{name}.csv", index=False)
        face_tables[name] = table

    # Fit table deltas and key counts.
    key_rows = []
    key_counts = {}
    for quantity in ("Q1", "Q2", "S_orient"):
        base, promoted = (_row_counts(mode, quantity) for mode in modes)
        key_counts[quantity] = {"base": base, "promote": promoted}
        for key in set(base) | set(promoted):
            a, b = base.get(key, 0), promoted.get(key, 0)
            if abs(b - a) >= 20:
                key_rows.append(dict(quantity=quantity, key=str(key), rows_no_promotion=a,
                                     rows_with_promotion=b, delta=b-a, abs_delta=abs(b-a)))
    key_delta = pd.DataFrame(key_rows, columns=["quantity", "key", "rows_no_promotion",
                                               "rows_with_promotion", "delta", "abs_delta"])
    if len(key_delta):
        key_delta = key_delta.sort_values(["quantity", "abs_delta", "key"],
                                          ascending=[True, False, True])
    key_delta.to_csv(OUT / "t1d_key_count_changes.csv", index=False)

    zinc_key = ("Zn", 1, 2, "O_carboxylate", "octahedral", "O_carboxylate")
    zinc = {mode: _row_counts(mode, "Q1").get(repr(zinc_key), 0)
            for mode in (*modes, "t1c")}

    # Added Q1 distance rows per key, via a parent + rounded-value multiset join.
    q1 = {mode: frames(mode, "Q1", ["csd_code", "key", "value"]) for mode in modes}
    def row_counter(df):
        return Counter((str(k), str(c), round(float(v), 8))
                       for k, c, v in df[["key", "csd_code", "value"]].itertuples(index=False, name=None))
    q1_rows = {mode: row_counter(df) for mode, df in q1.items()}
    base_values = defaultdict(list)
    promoted_values = defaultdict(list)
    for mode, target in (("base", base_values), ("promote", promoted_values)):
        d = q1["t1d_base" if mode == "base" else "t1d_promote"]
        for key, value in d[["key", "value"]].itertuples(index=False, name=None):
            target[str(key)].append(float(value))
    added = defaultdict(list)
    remaining = Counter(q1_rows["t1d_base"])
    for key, code, value in q1_rows["t1d_promote"].elements():
        if remaining[(key, code, value)]:
            remaining[(key, code, value)] -= 1
        else:
            added[key].append(value)
    # Select the Zn key plus the ten largest positive Q1 row gains.
    zinc_key_str = repr(zinc_key)
    top_q1 = sorted(((k, len(v)) for k, v in added.items() if v and k != zinc_key_str),
                    key=lambda x: (-x[1], x[0]))[:10]
    selected = [zinc_key_str, *(k for k, _ in top_q1)]
    quantile_rows = []
    for key in selected:
        base_arr = np.asarray(base_values.get(key, []), float)
        base_p95 = float(np.percentile(base_arr, 95)) if len(base_arr) else np.nan
        base_max = float(np.max(base_arr)) if len(base_arr) else np.nan
        for group, values in (("added_by_promotion", added.get(key, [])),
                              ("present_without_promotion", base_values.get(key, []))):
            arr = np.asarray(values, float)
            q = np.percentile(arr, [0, 5, 25, 50, 75, 95, 100]) if len(arr) else [np.nan] * 7
            quantile_rows.append(dict(key=key, group=group, n=len(arr), p0=q[0], p5=q[1], p25=q[2],
                p50=q[3], p75=q[4], p95=q[5], p100=q[6],
                share_above_no_promotion_p95=(float(np.mean(arr > base_p95)) if len(arr) and np.isfinite(base_p95) else np.nan),
                share_above_no_promotion_max=(float(np.mean(arr > base_max)) if len(arr) and np.isfinite(base_max) else np.nan)))
    pd.DataFrame(quantile_rows).to_csv(OUT / "t1d_q1_added_row_distances.csv", index=False)

    # Promoted atoms: per-metal and per-face-pair ratio and M-X distance.
    promotions = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "t1d_promote" / "shards").glob("*_promotions.parquet"))],
                           ignore_index=True)
    promotion_tables = {}
    for name, group_cols in (("metal", ["metal"]), ("pair", ["pair"]), ("metal_pair", ["metal", "pair"])):
        out = []
        for group_key, group in promotions.groupby(group_cols, dropna=False):
            if not isinstance(group_key, tuple): group_key = (group_key,)
            record = dict(zip(group_cols, group_key)); record["n"] = len(group)
            for val_name, col in (("ratio", "ratio"), ("distance_A", "distance")):
                qs = np.percentile(group[col].to_numpy(float), [0, 5, 25, 50, 75, 95, 100])
                for p, v in zip((0, 5, 25, 50, 75, 95, 100), qs): record[f"{val_name}_p{p}"] = float(v)
            out.append(record)
        promotion_tables[name] = pd.DataFrame(out).sort_values("n", ascending=False)
        promotion_tables[name].to_csv(OUT / f"t1d_promotion_{name}.csv", index=False)

    # Compare all source parents. The promotion atom table provides the parent metal.
    def parent_map(mode):
        rows = []
        for p in sorted((OUT / mode / "shards").glob("*_parents.jsonl")):
            if p.stat().st_size:
                rows.extend(pd.read_json(p, lines=True).to_dict("records"))
        return {r["csd_code"]: r for r in rows}
    parents = {mode: parent_map(mode) for mode in modes}
    metal_by_code = promotions.drop_duplicates("csd_code").set_index("csd_code").metal.to_dict()
    fields = ("donor_set", "coordination_number", "site_labels", "geometry_label")
    all_changes, panel_changes = [], []
    eligible = target_codes()
    for code in sorted(set(parents["t1d_base"]) | set(parents["t1d_promote"])):
        old, new = parents["t1d_base"].get(code), parents["t1d_promote"].get(code)
        if old is None or new is None:
            continue
        changed = [f for f in fields if old[f] != new[f]]
        if not changed: continue
        item = dict(csd_code=code, metal=metal_by_code.get(code), changed_fields=changed,
                    no_promotion={f: old[f] for f in fields}, promotion={f: new[f] for f in fields})
        all_changes.append(item)
        if code in eligible: panel_changes.append(item)
    pd.DataFrame(all_changes).to_json(OUT / "t1d_dbwide_parent_changes.json", orient="records", indent=2)
    pd.DataFrame(panel_changes).to_json(OUT / "t1d_cal_tail_parent_changes.json", orient="records", indent=2)
    parent_stats = defaultdict(Counter)
    for change in all_changes:
        metal = change["metal"] or "unknown"
        parent_stats[metal]["any"] += 1
        for field in change["changed_fields"]: parent_stats[metal][field] += 1
    stats_rows = [dict(metal=m, **dict(c)) for m, c in sorted(parent_stats.items())]
    pd.DataFrame(stats_rows).to_csv(OUT / "t1d_dbwide_parent_counts_by_metal.csv", index=False)

    # Re-extraction provenance totals from the saved shard outputs.
    extraction_totals = {}
    for mode in modes:
        files = [json.loads(p.read_text()) for p in [(OUT / mode / "run_summary.json")]]
        summary = files[0]
        shard_rows = [pd.read_parquet(p).shape[0] for p in _paths(mode, "Q1")]
        extraction_totals[mode] = dict(shards=len(shard_rows), q1_rows=sum(shard_rows),
            q2_rows=sum(pd.read_parquet(p).shape[0] for p in _paths(mode, "Q2")),
            qo_rows=sum(pd.read_parquet(p).shape[0] for p in _paths(mode, "QO")),
            s_orient_rows=sum(pd.read_parquet(p).shape[0] for p in _paths(mode, "S_orient")),
            parent_rows=sum(len(pd.read_json(p, lines=True)) for p in (OUT / mode / "shards").glob("*_parents.jsonl")),
            errors=sum(int(s["errors"]) for s in summary["shards"]))
    summary = dict(extraction=extraction_totals, promotion_atoms=int(len(promotions)),
        promoted_ratio_distance_by_metal=promotion_tables["metal"].to_dict("records"),
        key_changes_ge20=len(key_delta), key_changes_by_quantity=key_delta.quantity.value_counts().to_dict()
            if len(key_delta) else {}, zinc_carboxylate_key=repr(zinc_key), zinc_carboxylate_counts=zinc,
        q1_distance_keys=selected, q1_top_added_keys=top_q1,
        changed_parents_dbwide=len(all_changes), changed_parents_by_metal=stats_rows,
        changed_cal_tail_parents=panel_changes,
        changes_missing_metal=sum(x["metal"] is None for x in all_changes),
        face_delta_by_pair=face_tables["pair"].to_dict("records"),
        face_delta_by_metal=face_tables["metal"].to_dict("records"))
    (OUT / "t1d_analysis_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("promoted_ratio_distance_by_metal",
        "changed_parents_by_metal", "changed_cal_tail_parents", "face_delta_by_pair", "face_delta_by_metal")},
        indent=2, default=str))
    return summary


def main():
    if sys.argv[1:] == ["check-off"]:
        check_off_reproduction()
        return
    if sys.argv[1:] == ["t1b"]:
        t1b_analysis()
        return
    if sys.argv[1:] == ["t1c"]:
        t1c_analysis()
        return
    if sys.argv[1:] == ["t1d"]:
        t1d_analysis()
        return
    key_delta, key_counts = table_deltas()
    faces, face_data = face_census()
    parents, parent_data = parent_changes()
    components = farfield_census()
    components = join_farfield_sources(components, face_data, parent_data)
    component_cols = ["quantity", "db_key", "n", "source", "inherited", "component", "weight", "mean", "sd",
                      "main_mean", "main_sd", "distance_main_sd", "backoff_parent", "source_rows",
                      "removed_confirmed", "removed_possible", "removed_share_confirmed",
                      "removed_share_possible_upper"]
    pd.DataFrame(components, columns=component_cols).to_csv(OUT / "farfield_source_rows.csv", index=False)
    result = dict(key_changes=len(key_delta), key_counts={q:{m:len(c) for m,c in v.items()} for q,v in key_counts.items()},
        faces=faces, parents_changed=len(parents), parent_changes=parents,
        nuwfam=nuwfam_summary(key_counts), farfield_components=components)
    (OUT / "analysis_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("parent_changes", "farfield_components")}, indent=2))


if __name__ == "__main__":
    main()
