"""Re-score saved D3 pools under two DBs without running the generator."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
import logging
from pathlib import Path

CASES = "LUQVOK SESWIZ LIPDEU YABYEJ LOPJEF ZAQNUE HOXJUZ USAPOU UMUBAF".split()
PHASES = {"C": ("61080071", "s1_CS"), "D": ("61080071", "s1_DDS"),
          "DQ6": ("61100009", "s1_DQ6")}
FINGERPRINTS = {"C": "6b47c3fb45a9", "D": "302a15e37622"}


def manifest(root):
    """Freeze just the named cases and exact record/export paths."""
    from tools.jgen.inputs import frame_shards

    frame = frame_shards(root / "data/refit/frame_v1")
    base = root / "research/data/jgen/campaign/D3"
    dbs = {"C": root / "data/db/constraints_2026-11-01.json",
           "D": root / "research/data/d2/db_d2/constraints_2026-11-09.json"}
    dbs = {k: {"path": str(v), "sha256": _sha(v), "meta_sha256": _sha(v.with_suffix(".meta.json"))}
           for k, v in dbs.items()}
    rows = []
    for code in CASES:
        paths = {}
        for arm, (job, phase) in PHASES.items():
            seed_paths = []
            for seed in (11, 12, 13):
                [record] = (base / f"job_{job}" / phase).glob(f"g*/{arm}/records/{code}_{seed}.json")
                pool = record.parent.parent / "xyz" / code / f"{code}_{arm}_s{seed}_pool.json"
                seed_paths.append({"record": str(record), "pool": str(pool)})
            paths[arm] = seed_paths
        for runtime in PHASES:
            rows.append(dict(id=f"{code}_{runtime}", code=code, runtime=runtime, paths=paths, dbs=dbs,
                             shard=frame[code][0], dataset=frame[code][1], ref_dir=str(root / "data/ref")))
    return rows


def _sha(path):
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _read_pool(pair, dbs):
    import numpy as np
    from tools.jgen.bench.pool_export import _read_frames

    p = Path(pair["pool"])
    rec = json.loads(Path(pair["record"]).read_text())
    if rec["status"] != "ok":
        raise ValueError(f"{rec['code']}/{rec['arm']}/{rec['seed']}: {rec['reason']}")
    if rec["constraints_db_sha256"] != dbs["C" if rec["arm"] == "C" else "D"]["sha256"]:
        raise ValueError("record DB hash differs from frozen DB")
    m = json.loads(p.read_text())
    if (m["code"], m["arm"], m["seed"], m["n_rows"]) != tuple(rec[k] for k in ("code", "arm", "seed", "n_rows")):
        raise ValueError("pool identity/count mismatch")
    frames = [frame for f in dict.fromkeys(r["file"] for r in m["rows"]) for frame in _read_frames(p.parent / f)]
    if len(frames) != len(m["rows"]):
        raise ValueError("XYZ row count mismatch")
    for (head, symbols, X), row, key in zip(frames, m["rows"], rec["row_keys"], strict=True):
        if head != {k: v for k, v in row.items() if k != "file"} or head["row_key"] != key:
            raise ValueError("XYZ, manifest and record row metadata mismatch")
        if [symbols[j] for j in m["gen_to_ref_idx"]] != m["ref_symbols"] or not np.isfinite(X).all():
            raise ValueError("invalid coordinates or atom mapping")
    return rec, m, frames


class _RejectFallback(logging.Handler):
    def emit(self, record):
        if "GFN2 embed refinement failed" in record.getMessage():
            raise RuntimeError(record.getMessage())


def _terms(scorer, X, groups):
    import torch
    from tools.jdev.budget.pools import rank_terms

    score, parts = rank_terms(scorer, X, groups)
    summed = sum(parts.values())
    torch.testing.assert_close(score, summed, atol=1e-6, rtol=1e-6)
    return score, parts


def _build_scorers(ctx, assignments, lookups):
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.scorer import Scorer

    ge = GeometryExtractor(ctx.mol, ctx.device)

    def build(lookup):
        return Scorer(ctx.mol, ge, lookup, device=ctx.device, quantity_weights=ctx.quantity_weights,
                      assignments=assignments, donor_groups=[g for _, _, g in ctx.all_donor_groups],
                      ideal=ctx.ideal, orientation_model=ctx.orientation_model,
                      angle_bend_penalty=ctx.angle_bend_penalty)

    return build, {name: build(lookup) for name, lookup in lookups.items()}


def _source_scorers(parent, mol, lookups):
    from jacobik.builder import engine
    from jacobik import ComplexBuilder
    from jacobik.topology.decompose import decompose_ligand
    from jacobik.builder.ligands import _build_ligand_groups, ligand_inputs_from_mol
    from tools.jdev import multiseed

    ligands = ligand_inputs_from_mol(mol, _build_ligand_groups(mol, mol.smiles))
    if ligands is None:
        raise ValueError("unmappable ligand inputs")
    builder = ComplexBuilder(metal=mol.metal_symbol, geometry=mol.coord_geom, charge=mol.charge,
                             spin=mol.spin_state, ligands=ligands)
    variants = builder._ez_combinations()
    builder._ez_inner = True
    native_db = "C" if parent["runtime"] == "C" else "D"
    log = logging.getLogger(engine.__name__)
    reject = _RejectFallback()
    log.addHandler(reject)
    try:
        for i, fixes in enumerate(variants):
            for ligand, fix in zip(ligands, fixes, strict=True):
                ligand.ez_fix = fix
            ctx, assignments, _, _ = multiseed.prepare(
                builder, 11, stats=lookups[native_db], device="cpu", angle_bends=True,
                stereo_enum=False,
            )
            for ligand in ctx.mol.ligands:
                decompose_ligand(ctx.mol, ligand)
                if ctx.macro_pool <= 0:
                    ligand.macrocycles = []
            key = i if len(variants) > 1 else None
            yield key, ctx, assignments, fixes
    finally:
        log.removeHandler(reject)


def _lookup_type():
    from jacobik.stats.stats import StatsLookup

    class Traced(StatsLookup):
        def __init__(self, db):
            super().__init__(db)
            self.trace = {}
            self.override = {}

        def dist(self, quantity, key):
            fd = super().dist(quantity, key)
            self.trace[quantity, key] = (self._last_resolved_key, asdict(fd))
            return self.override.get((quantity, key), fd)

        def orientation_dist(self, *key):
            resolved, fd = super().orientation_dist(*key)
            self.trace["QO", key] = (resolved, asdict(fd) if fd else None)
            return self.override.get(("QO", key), (resolved, fd))

    return Traced


def _signature(ctx, assignments):
    # These are builder indices. A different atom/site/assignment graph cannot
    # use the origin arm's row group as a same-target counterfactual.
    return dict(symbols=[a.symbol for a in ctx.mol.atoms],
                sites=[list(s.donor_indices) for s in ctx.mol.coord_sites],
                labels={str(i): l.subclass for i, l in ctx.mol.donor_labels.items()},
                groups=[list(g) for _, _, g in ctx.all_donor_groups],
                assignments=[a.tolist() for a in assignments], ideal=ctx.ideal.tolist())


def _pose_rows(frames, variant, signature):
    import numpy as np
    import torch

    ii = [i for i, (r, s, _) in enumerate(frames)
          if r["row_key"][0] == variant and s == signature["symbols"]
          and 0 <= r["row_key"][1] < len(signature["assignments"])]
    if not ii:
        return ii, None, None
    X = torch.as_tensor(np.stack([frames[i][2] for i in ii]), dtype=torch.float32)
    groups = torch.as_tensor([frames[i][0]["row_key"][1] for i in ii], dtype=torch.long)
    return ii, X, groups


def _swap_fields(left, right, mode):
    """Switch C/D fit fields without deleting any mixture component."""
    if mode == "table":
        return right
    if not left.means or not right.means or len(left.means) != len(right.means):
        return None
    order = sorted(range(len(left.means)), key=lambda i: left.means[i])
    match = dict(zip(order, sorted(range(len(right.means)), key=lambda i: right.means[i])))
    if mode == "weights":
        return replace(left, weights=[right.weights[match[i]] for i in range(len(order))])
    return replace(left, means=[right.means[match[i]] for i in range(len(order))],
                   vars=[right.vars[match[i]] for i in range(len(order))])


def _counterfactuals(build, lookups, X, groups, base, quantities):
    """Marginal effects of substitutions into D, with C as donor."""
    from jacobik.stats.stats import FittedDist

    d, c = lookups["D"], lookups["C"]
    results = []
    for q, key in quantities:
        v, u = d.trace.get((q, key)), c.trace.get((q, key))
        if not v or not u or v[1] is None or u[1] is None or v == u:
            continue
        cfit = FittedDist(**u[1])
        d.override[q, key] = (u[0], cfit) if q == "QO" else cfit
        try:
            pred, _ = _terms(build(d), X, groups)
            results.append(dict(quantity=q, key=key, resolved_C=u[0], resolved_D=v[0],
                                delta=(pred - base).tolist()))
        finally:
            d.override.clear()
    return results


def _luqvok(build, lookups, X, groups, ctx):
    import torch
    from jacobik.stats.stats import FittedDist

    d, c = lookups["D"], lookups["C"]
    outputs = {}
    for label in ("O_carboxylate", "C_pi", "both"):
        for mode in ("weights", "shape", "table"):
            changed, unmatched = [], []
            for (q, key), (_, dd) in list(d.trace.items()):
                if q != "Q1" or key[0] != "Pd" or key[3] not in (
                        ("O_carboxylate", "C_pi") if label == "both" else (label,)):
                    continue
                cd = c.trace.get((q, key), (None, None))[1]
                if not dd or not cd:
                    continue
                fit = _swap_fields(FittedDist(**dd), FittedDist(**cd), mode)
                if fit is None:
                    unmatched.append(key)
                else:
                    d.override[q, key] = fit
                    changed.append(key)
            try:
                score, parts = _terms(build(d), X, groups)
                outputs[f"{label}_{mode}"] = dict(keys=changed, unmatched=unmatched, score=score.tolist(),
                                                  Q1=parts["Q1"].tolist())
            finally:
                d.override.clear()

    # Translate each carboxylate-bearing ligand rigidly outward from Pd. This
    # changes all of that ligand's contacts together and preserves its internals.
    start = X[0]
    coords = start.repeat(17, 1, 1)
    offsets = torch.linspace(0, 0.8, 17)
    moved = []
    for lig in ctx.mol.ligands:
        atoms = [i for i, a in enumerate(ctx.mol.atoms) if a.idx in lig.atom_indices]
        oxygen = [i for i in atoms if (ctx.mol.donor_labels.get(ctx.mol.atoms[i].idx)
                  and ctx.mol.donor_labels[ctx.mol.atoms[i].idx].subclass == "O_carboxylate")]
        if not oxygen:
            continue
        metal = next(i for i, a in enumerate(ctx.mol.atoms) if a.idx == ctx.mol.metal_idx)
        direction = start[oxygen].mean(0) - start[metal]
        direction /= torch.linalg.vector_norm(direction)
        coords[:, atoms] += offsets[:, None, None] * direction
        moved.append(oxygen)
    if moved:
        outputs["rigid_scan"] = dict(offset_A=offsets.tolist(), donor_indices=moved)
        for name, lookup in lookups.items():
            scores, terms = _terms(build(lookup), coords, groups[0].repeat(len(coords)))
            outputs["rigid_scan"][name] = dict(score=scores.tolist(), terms={q: t.tolist() for q, t in terms.items()})
    return outputs


def run(parent):
    import numpy as np
    import torch
    from tools.jgen.bench.run import _setup_env

    native = parent["runtime"]
    flags = json.loads(Path(parent["paths"][native][0]["record"]).read_text())["flags"]
    _setup_env(flags)  # Must precede the first jacobik import.
    from jacobik.core import profile
    from jacobik.stats.stats import load_db
    from tools.jgen.inputs import load_mol

    torch.set_num_threads(1)
    dbs = parent["dbs"]
    for db in dbs.values():
        path = Path(db["path"])
        if _sha(path) != db["sha256"] or _sha(path.with_suffix(".meta.json")) != db["meta_sha256"]:
            raise ValueError("database or sidecar changed after manifest freeze")
    Traced = _lookup_type()
    lookups = {name: Traced(load_db(db["path"])) for name, db in dbs.items()}
    mol = load_mol(parent["code"], parent["shard"], parent["dataset"], clean_root=Path(parent["ref_dir"]))
    if mol is None:
        raise ValueError("source geometry missing")
    ref_X = np.asarray([a.position for a in mol.atoms])
    pools = [_read_pool(pair, dbs) for paths in parent["paths"].values() for pair in paths]
    if any(not np.array_equal(ref_X, m["ref_X"]) for _, m, _ in pools):
        raise ValueError("export reference differs from source geometry")
    output = {"code": parent["code"], "runtime": native, "flags": flags, "variants": {}}
    with torch.no_grad():
        for variant, ctx, assignments, fixes in _source_scorers(parent, mol, lookups):
            for lookup in lookups.values():
                lookup.trace.clear()
            signature = _signature(ctx, assignments)
            build, scorers = _build_scorers(ctx, assignments, lookups)
            result = {"signature": signature, "ez_fixes": fixes, "pools": {}}
            tops = {}
            for rec, m, frames in pools:
                ii, X, groups = _pose_rows(frames, variant, signature)
                if not ii:
                    continue
                arm, seed = rec["arm"], rec["seed"]
                stat, parts = {}, {}
                for name, sc in scorers.items():
                    val, terms = _terms(sc, X, groups)
                    stat[name] = val.tolist()
                    parts[name] = {q: x.tolist() for q, x in terms.items()}
                native_db = "C" if arm == "C" else "D"
                checked = {}
                for tag in ("top1", "floor"):
                    hits = [j for j, i in enumerate(ii) if m["rows"][i][f"is_{tag}"]]
                    prior = rec.get(f"{tag}_bend_prior")
                    if hits and prior is not None and arm == native:
                        j = hits[0]
                        checked[tag] = m["rows"][ii[j]]["score"] - stat[native_db][j] - prior
                if seed == 11:
                    for j, i in enumerate(ii):
                        if m["rows"][i]["is_top1"]:
                            tops[arm] = X[j], groups[j]
                rows = [m["rows"][i] for i in ii]
                sc = scorers["C"]
                result["pools"][f"{arm}_{seed}"] = dict(
                    rows=rows, stat=stat, terms=parts, native_score_residual=checked,
                    q1_dist_A=sc._ge.q1_distances(X).tolist(),
                    recorded_fingerprint=rec["fingerprint"],
                )
                if arm == "C" and seed == 11:
                    ref = torch.as_tensor(ref_X[np.argsort(m["gen_to_ref_idx"])], dtype=torch.float32)[None]
                    group = next((groups[j] for j, i in enumerate(ii) if m["rows"][i]["is_top1"]), groups[0])
                    result["reference_score_same_assignment"] = {name: _terms(sc, ref, group[None])[0].item()
                                                                  for name, sc in scorers.items()}
            result["lookup_trace"] = {name: [dict(quantity=q, requested=k, resolved=r, fit=f)
                                             for (q, k), (r, f) in lookup.trace.items()]
                                      for name, lookup in lookups.items()}
            if "C" in tops and "D" in tops:
                X = torch.stack([tops[k][0] for k in ("C", "D")])
                g = torch.stack([tops[k][1] for k in ("C", "D")])
                # A one-key substitution is a marginal effect; report any
                # interaction/remainder rather than force additivity.
                base, bd = _terms(scorers["D"], X, g)
                _, bc = _terms(scorers["C"], X, g)
                qs = [q for q in bd if abs(float((bd[q] - bc[q]).diff().item())) > 0.001]
                if "Q3" in qs or "S_orient" in qs or "SO" in qs:
                    qs.append("QO")
                keys = [key for key in lookups["D"].trace if key[0] in qs]
                result["key_swaps_top1_s11"] = _counterfactuals(build, lookups, X, g, base, keys)
                if parent["code"] == "LUQVOK":
                    result["luqvok_top1_s11"] = _luqvok(build, lookups, X, g, ctx)
            output["variants"][str(variant)] = result
    output["effective_fingerprint"] = profile.snapshot()["fingerprint"]
    if native in FINGERPRINTS and output["effective_fingerprint"] != FINGERPRINTS[native]:
        raise ValueError(f"effective fingerprint mismatch: {output['effective_fingerprint']}")
    return output
