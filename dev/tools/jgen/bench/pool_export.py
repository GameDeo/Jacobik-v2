"""Full returned-pool XYZ export and symmetry-aware reference coverage.

Coordinates and labels are always in builder order. The manifest stores the
explicit reference-order permutation; scores are returned ConformerResult.score.
"""
from __future__ import annotations

import argparse
import json
import lzma
from pathlib import Path

import numpy as np


def xyz_text(symbols, positions, header):
    if np.asarray(positions).shape != (len(symbols), 3):
        raise ValueError("XYZ atom/coordinate count mismatch")
    return (f"{len(symbols)}\n{json.dumps(header, allow_nan=False)}\n" +
            "".join(f"{s:<2} {x:12.6f} {y:12.6f} {z:12.6f}\n"
                    for s, (x, y, z) in zip(symbols, positions)))


def _number(value):
    return float(value) if np.isfinite(value) else None


def export_pool(root, code, arm, seed, confs, energies_kj, ref_symbols, ref_X,
                gen_to_ref, top, floor, compression="none"):
    """Write every returned row, compatibility top/floor copies, and a manifest.

    xz stores a multi-frame XYZ per parent/arm/seed (not one compressor per row).
    Null energy/score means nonfinite, never a substituted value.
    """
    from tools.jgen.atom_export import symbols_for_conformer
    if len(confs) != len(energies_kj):
        raise ValueError("pool energy count mismatch")
    directory = Path(root) / code
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"{code}_{arm}_s{seed}"
    manifest = dict(format="jgen_pool_xyz_v1", code=code, arm=arm, seed=seed,
                    atom_order="builder", score_scale="returned_conformer_score",
                    gen_to_ref_idx=np.asarray(gen_to_ref, int).tolist(),
                    ref_symbols=list(ref_symbols), ref_X=np.asarray(ref_X).tolist(),
                    n_rows=len(confs), rows=[], compression=compression)
    bundle = directory / f"{stem}_pool.xyz.xz"
    stream = lzma.open(bundle, "wt") if compression == "xz" else None
    try:
        for i, (conf, energy) in enumerate(zip(confs, energies_kj)):
            symbols = symbols_for_conformer(conf)
            if [symbols[j] for j in gen_to_ref] != list(ref_symbols):
                raise ValueError("builder/reference symbol mapping mismatch")
            positions = np.asarray([a.position for a in conf.complex.atoms], float)
            header = dict(code=code, arm=arm, seed=seed, row_index=i,
                          row_key=[conf.embed_idx, int(conf.assignment_idx), int(conf.trial_idx)],
                          score=_number(conf.score), score_scale="returned_conformer_score",
                          mace_energy_kj=_number(energy), is_top1=i == top, is_floor=i == floor)
            if hasattr(conf, "descriptors"):          # the row's class tuple + descriptors (output by class)
                header.update(conf.descriptors())
            content = xyz_text(symbols, positions, header)
            name = bundle.name if stream else f"{stem}_row{i:05d}.xyz"
            if stream:
                stream.write(content)
            else:
                (directory / name).write_text(content)
            manifest["rows"].append(dict(header, file=name))
            for label, selected in (("top1", top), ("floor", floor)):
                if i == selected:
                    (directory / f"{stem}_{label}.xyz").write_text(content)
    finally:
        if stream:
            stream.close()
    by_cls = {}
    for r in manifest["rows"]:
        by_cls.setdefault(r.get("class_label"), []).append(r["row_index"])
    manifest["classes"] = [dict(class_label=k, rows=v) for k, v in by_cls.items()]   # per-class pools (row indices)
    path = directory / f"{stem}_pool.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return path


def _read_frames(path):
    opener = lzma.open if path.suffix == ".xz" else open
    with opener(path, "rt") as stream:
        while line := stream.readline():
            n = int(line)
            header = json.loads(stream.readline())
            atoms = [stream.readline().split() for _ in range(n)]
            yield header, [a[0] for a in atoms], np.asarray([a[1:4] for a in atoms], float)


def reference_context(symbols, ref_X):
    from tools.runkit.geom import automorphisms, find_metal
    heavy = np.asarray([i for i, s in enumerate(symbols) if s != "H"], int)
    perms, capped = automorphisms(np.asarray(ref_X), symbols, find_metal(symbols))
    return heavy, perms, capped


def coverage_rows(positions, ref_symbols, ref_X, top, floor, context=None):
    """1 A hit for any row, using heavy atoms incl. metal, Kabsch + symmetry.

    Nonfinite geometries cannot hit. A capped automorphism search is reported.
    """
    from tools.runkit.geom import sym_rmsd_to_ref
    heavy, perms, capped = context or reference_context(ref_symbols, ref_X)
    X = np.asarray(positions, float)
    rmsd = np.full(len(X), np.inf)
    finite = np.isfinite(X).all(axis=(1, 2))
    for chunk in np.array_split(np.flatnonzero(finite), max(1, (int(finite.sum()) + 63) // 64)):
        if len(chunk):
            rmsd[chunk] = sym_rmsd_to_ref(X[chunk][:, heavy], np.asarray(ref_X)[heavy], perms)
    def hit(i):
        return i is not None and bool(rmsd[i] <= 1.0)
    return dict(threshold_A=1.0, metric="symmetry_aware_heavy_atom_Kabsch_including_metal",
                n_rows=len(X), n_finite_geometry=int(finite.sum()),
                rmsd_A=[_number(v) for v in rmsd], min_rmsd_A=_number(np.min(rmsd)) if len(X) else None,
                top1_within_1A=hit(top), floor_within_1A=hit(floor),
                top1_or_floor_within_1A=hit(top) or hit(floor),
                full_pool_within_1A=bool(np.any(rmsd <= 1.0)),
                automorphisms_capped=bool(capped), num_automorphisms=len(perms))


def coverage_export(manifest_path, context=None):
    """Recompute from the serialized coordinates, verifying all row metadata."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    rows = manifest["rows"]
    positions, headers = [], []
    files = dict.fromkeys(row["file"] for row in rows)
    for name in files:
        for header, symbols, X in _read_frames(path.parent / name):
            i = len(headers)
            expected = {k: v for k, v in rows[i].items() if k != "file"}
            if header != expected:
                raise ValueError("XYZ/manifest row metadata mismatch")
            mapping = manifest["gen_to_ref_idx"]
            if [symbols[j] for j in mapping] != manifest["ref_symbols"]:
                raise ValueError("XYZ/reference atom order mismatch")
            positions.append(X[mapping]); headers.append(header)
    if len(headers) != manifest["n_rows"]:
        raise ValueError("exported pool row count mismatch")
    top = next((i for i, h in enumerate(headers) if h["is_top1"]), None)
    floor = next((i for i, h in enumerate(headers) if h["is_floor"]), None)
    return coverage_rows(positions, manifest["ref_symbols"], manifest["ref_X"], top, floor, context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    results = []
    for path in args.manifest:
        try:
            results.append(dict(manifest=str(path), status="ok", coverage=coverage_export(path)))
        except Exception as exc:
            results.append(dict(manifest=str(path), status="failed", reason=f"{type(exc).__name__}: {exc}"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
