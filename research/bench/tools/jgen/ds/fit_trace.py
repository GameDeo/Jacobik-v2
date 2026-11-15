"""Trace LUQVOK's Q1 fit to the build inputs and inspect contributing parents."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

O_KEY = ("Pd", 1, 2, "O_carboxylate", "square_plan", "O_carboxylate")
C_KEY = ("Pd", 1, 2, "C_pi", "square_plan", "C_pi")
SPLITS = ("X1", "X2", "X3", "LNf")


def _fit(rows, key):
    from jacobik.stats import mixture
    from fit import pooling

    # This is the independent branch of PooledFitter, with its per-structure
    # evidence weights and CRC-keyed initialization.
    counts = Counter(code for code, _ in rows)
    x = [v for _, v in rows]
    w = [1 / counts[code] for code, _ in rows]
    family, kind, _, floor = pooling.SPEC["Q1"]
    _, f, err = pooling.fit_one_checked((repr(key), x, w, family, floor))
    if err or f is None:
        raise ValueError(err or "independent fit failed")
    return mixture.to_entry(f, kind, len(rows), len(counts), pooled=True, src="fit")


def _l0_rows(directory, key):
    import pandas as pd

    rows = []
    for split in SPLITS:
        df = pd.read_parquet(directory / f"observations_Q1_{split}.parquet",
                             columns=["csd_code", "value"], filters=[("key", "=", repr(key))])
        rows.extend(df.itertuples(index=False, name=None))
    return rows


def direct_fit(main, evidence):
    import numpy as np
    from fit.sanity import excluded_charge_codes, locked_codes
    from fit import pooling
    from tools.jgen.ds.cross_score import _sha

    old = main / "research/archive/data/refit/observations"
    new = evidence / "fit_inputs/d2_obs"
    d2 = main / "research/data/d2/db_d2/constraints_2026-11-09.json"
    meta = json.loads(d2.with_suffix(".meta.json").read_text())
    hashes = {p.name: _sha(p) for p in new.glob("observations_Q1_*.parquet")}
    if len(hashes) != 4 or any(meta["inputs"]["observation_files"].get(k) != v for k, v in hashes.items()):
        raise ValueError("Q1 observation files do not match build provenance")
    lock_path = main / "research/bench/benchmarks/locked_exclusions.json"
    if _sha(lock_path) != meta["inputs"]["locked_exclusions_sha256"]:
        raise ValueError("fit exclusion manifest changed")
    census_path = evidence / "fit_inputs/charge_os_census.json"
    census = json.loads(census_path.read_text())
    census_sha = hashlib.sha256(json.dumps(census, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    if census_sha != meta["charge_os_census_sha256"]:
        raise ValueError("charge census differs from build provenance")
    locked = locked_codes(lock_path)
    bad = excluded_charge_codes(census)
    recorded = {
        "C": json.loads((main / "data/db/constraints_2026-11-01.json").read_text())[f"Q1:{O_KEY!r}"],
        "D": json.loads(d2.read_text())[f"Q1:{O_KEY!r}"],
    }
    c, d_raw = _l0_rows(old, O_KEY), _l0_rows(new, O_KEY)
    d = [(code, v) for code, v in d_raw if code not in locked and code not in bad]
    fit = {"C": _fit(c, O_KEY), "D": _fit(d, O_KEY)}
    fields = ("means", "vars", "weights")
    for name in fit:
        if (fit[name]["n"] != recorded[name]["n"]
                or fit[name]["n_eff"] != recorded[name]["n_eff"]
                or any(not np.allclose(fit[name][k], recorded[name][k], atol=1e-10, rtol=1e-7) for k in fields)):
            raise ValueError(f"independent fit did not reproduce {name}")
    c_codes, d_codes = {k for k, _ in c}, {k for k, _ in d}
    added = [(k, v) for k, v in d if k not in c_codes]
    shared = [(k, v) for k, v in d if k in c_codes]
    controls = {
        "C_plus_added_D_rows": _fit(c + added, O_KEY),
        "D_minus_added_rows": _fit(shared, O_KEY),
    }
    by_code = {code: [v for k, v in d if k == code] for code in sorted(d_codes)}
    rng = np.random.default_rng(11)
    boot = []
    for _ in range(64):
        picks = rng.choice(list(by_code), len(by_code), replace=True)
        # Distinct occurrence IDs retain the sampled cluster multiplicities.
        b = [(str(i), value) for i, k in enumerate(picks) for value in by_code[k]]
        trial = _fit(b, O_KEY)
        boot.append(dict(K=len(trial["means"]),
                         long_weight=sum(w for m, w in zip(trial["means"], trial["weights"]) if m > 2.4)))
    return dict(key=O_KEY, d2_observation_hashes=hashes, charge_census_sha256=census_sha,
                exclusions_sha256=_sha(lock_path), direct_fit=fit, recorded=recorded, controls=controls,
                C_rows=c, D_rows=d, added_rows=added, dropped_parent_count=len(c_codes - d_codes),
                d_rows_removed_charge=sum(k in bad for k, _ in d_raw),
                d_rows_removed_locked=sum(k in locked for k, _ in d_raw),
                independent_min=pooling.N_INDEP, bootstrap=boot)


def replay_samples(main, evidence):
    """Replay the production Q1 stream, retaining only watched ancestor buckets."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from fit.sanity import excluded_charge_codes, locked_codes
    from fit.expander import _AncestorReservoir, _expand_into_reservoir
    from jacobik.stats.ladders import q1_candidates

    locked = locked_codes(main / "research/bench/benchmarks/locked_exclusions.json")
    bad = excluded_charge_codes(json.loads((evidence / "fit_inputs/charge_os_census.json").read_text()))
    watched = {repr(key) for requested in (C_KEY, O_KEY) for key in q1_candidates(requested)}

    class Capture(_AncestorReservoir):
        def add_l0(self, row):
            pass

        def add_ancestor(self, key, sub, row):
            if key in watched:
                super().add_ancestor(key, sub, row)
            else:
                # Advance the exact production stream without storing rows for
                # unrelated buckets. Those draws still affect watched buckets.
                bucket = (key, sub)
                n = self._seen[bucket] = self._seen.get(bucket, 0) + 1
                if n > self.cap:
                    self._rng.randint(1, n)

    def replay(excluded):
        capture = Capture()
        skip = pa.array(sorted(excluded), type=pa.string())
        for split in SPLITS:
            p = evidence / "fit_inputs/d2_obs" / f"observations_Q1_{split}.parquet"
            for batch in pq.ParquetFile(p).iter_batches(batch_size=50_000):
                filtered = batch.filter(pc.invert(pc.is_in(batch["csd_code"], value_set=skip)))
                _expand_into_reservoir("Q1", filtered, capture)
        return capture

    before, after = replay(locked), replay(locked | bad)
    out = []
    for key, sub in sorted(before._reservoir.keys() | after._reservoir.keys()):
        bucket = (key, sub)
        a, b = before._reservoir.get(bucket, []), after._reservoir.get(bucket, [])
        raw_a, raw_b = before._seen.get(bucket, 0), after._seen.get(bucket, 0)
        survived = [row for row in a if row[0] not in bad]
        out.append(dict(key=key, bucket_subclass=sub, before_raw=raw_a, after_raw=raw_b,
                        before_sample=len(a), after_sample=len(b),
                        direct_sample_removals=len(a) - len(survived),
                        surviving_rows_resampled_out=sum((Counter(survived) - Counter(b)).values()),
                        exact_same_eligible_rows=raw_a == raw_b,
                        mean_before_A=sum(row[2] for row in a) / len(a) if a else None,
                        mean_after_A=sum(row[2] for row in b) / len(b) if b else None))
    return dict(comparison="D2 typed observations; locked fixed; charge filter off versus on",
                note="Unchanged raw bucket => stable coupled sample would replace zero rows.",
                buckets=out)


def orientation_fit(main, evidence):
    """Reconstruct LIPDEU's pooled two-substituent/four-ring orientation fit."""
    import numpy as np
    import pandas as pd
    from fit.sanity import excluded_charge_codes, locked_codes
    from jacobik.stats import orientation as O
    from fit import qo as QO
    from jacobik.stats.mixture import structure_representative
    from tools.jgen.ds.cross_score import _sha

    key = (2, 4)
    d2 = main / "research/data/d2/db_d2/constraints_2026-11-09.json"
    meta = json.loads(d2.with_suffix(".meta.json").read_text())
    dirs = {"C": main / "research/archive/data/refit/observations",
            "D": evidence / "fit_inputs/d2_obs"}
    data = {}
    hashes = {}
    cols = O.feature_cols(key[0])
    for name, directory in dirs.items():
        frames = []
        for split in SPLITS:
            path = directory / f"observations_QO_{split}.parquet"
            if name == "D":
                hashes[path.name] = _sha(path)
                if hashes[path.name] != meta["inputs"]["observation_files"][path.name]:
                    raise ValueError("QO observation files differ from build provenance")
            frames.append(pd.read_parquet(path, columns=["csd_code", "k", "ring", "hb_min", "sub", "metal", "geom", *cols]))
        data[name] = pd.concat(frames, ignore_index=True)
    locked = locked_codes(main / "research/bench/benchmarks/locked_exclusions.json")
    bad = excluded_charge_codes(json.loads((evidence / "fit_inputs/charge_os_census.json").read_text()))

    def refit(df, excluded):
        # Fit_rows duplicates haptic rows but contributes no k=2 rows or new
        # representative codes. Match training_groups' HB and LnQM policies.
        df = df[~(df.hb_min < O.HB_CUT) & ~df.csd_code.isin(excluded)]
        keep = structure_representative(df.csd_code.to_numpy())
        rows = df.loc[keep & (df.k == key[0]) & (df.ring == key[1])]
        G = np.ascontiguousarray(rows[cols].to_numpy(float))
        fit = QO._fit_one(G, QO.group_seed(key), O.K_MAX, O.COV_FLOOR, O.W_MIN)
        return dict(fit=fit, rows=rows[["csd_code", "sub", "metal", "geom", *cols]].to_dict("records"))

    fits = {"C": refit(data["C"], set()), "C_with_D2_filters": refit(data["C"], locked | bad),
            "D_locked_only": refit(data["D"], locked), "D": refit(data["D"], locked | bad)}
    for name, path in (("C", main / "data/db/constraints_2026-11-01.json"), ("D", d2)):
        actual = json.loads(path.read_text())[f"QO:{key!r}"]
        fit = fits[name]["fit"]
        matches = actual["n"] == fit["n"]
        for field in ("means", "covs", "weights"):
            a, b = np.asarray(actual[field]), np.asarray(fit[field])
            matches &= a.shape == b.shape and np.allclose(a, b, atol=1e-8, rtol=1e-6)
        fits[name]["matches_recorded"] = bool(matches)
    return dict(key=key, d2_observation_hashes=hashes, comparisons=fits)


def run(parent):
    """Input-only inspection of a training parent in a fresh typing profile."""
    from tools.jgen.bench.run import _setup_env

    flags = ({"hapto_bond_gate": True, "hapto_label_canon": True, "hapto_chelate_exclusion": True}
             if parent["typing"] == "D" else {})
    _setup_env(flags)
    import numpy as np
    from jacobik.stats.keys import q1_site_keys
    from fit.q1 import q1_bond_lengths
    from tools.jgen.inputs import load_mol

    mol = load_mol(parent["code"], parent["shard"], clean_root=Path(parent["ref_dir"]))
    if mol is None:
        raise ValueError("training parent not found")
    atoms = {a.idx: a for a in mol.atoms}
    metal = np.asarray(atoms[mol.metal_idx].position)
    orders = {b.j if b.i == mol.metal_idx else b.i: b.order for b in mol.bonds if mol.metal_idx in (b.i, b.j)}
    contacts = []
    for idx, atom in atoms.items():
        if idx == mol.metal_idx or atom.symbol == "H":
            continue
        distance = float(np.linalg.norm(np.asarray(atom.position) - metal))
        if distance <= 3.2:
            contacts.append(dict(index=idx, element=atom.symbol, distance_A=distance, bond_order=orders.get(idx)))
    sites = [dict(key=key, atoms=list(s.donor_indices), is_haptic=s.is_haptic)
             for key, s in zip(q1_site_keys(mol), mol.coord_sites, strict=True)]
    oxygens = {}
    for b in mol.bonds:
        for c, o in ((b.i, b.j), (b.j, b.i)):
            if atoms[c].symbol == "C" and atoms[o].symbol == "O":
                oxygens.setdefault(c, []).append(o)
    carboxylates = [dict(carbon=i, oxygens=oo) for i, oo in oxygens.items() if len(oo) == 2]
    return dict(code=parent["code"], typing=parent["typing"], smiles=mol.smiles,
                geometry=mol.coord_geom, sites=sites, observations=q1_bond_lengths(mol),
                contacts=sorted(contacts, key=lambda c: c["distance_A"]), carboxylates=carboxylates)
