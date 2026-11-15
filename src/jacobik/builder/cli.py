"""Command-line interface for Jacobik (``jacobik`` console script)."""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Optional

from jacobik.builder.engine import ComplexBuilder, LigandInput, generate, load_ligand_library
from jacobik.builder.ligands import _build_ligand_groups, ligand_inputs_from_mol
from jacobik.core import profile as _profile
from jacobik.core.env import (
    PRODUCTION_DB_NAME,
    JacobikEnvironmentError,
    check_environment,
    default_db_path,
    resolve_device,
)
from jacobik.stats.stats import StatsLookup, load_db

log = logging.getLogger("jacobik.cli")


_REPEAT_SPEC_RE = re.compile(r"^(.+?)\*(\d+)$")


def _parse_ligand_spec(spec: str) -> List[LigandInput]:
    """Parse JSON, ``SMILES:[[0],[1]]``, or a curated library alias/mode (optionally with ``*N``)."""
    text = spec.strip()
    if text.startswith("{"):
        data = json.loads(text)
        return [LigandInput(data["smiles"], donors=data["donors"], sites=data.get("sites"))]
    m = _REPEAT_SPEC_RE.fullmatch(text)
    if m is not None:
        base, count = m.group(1).strip(), int(m.group(2))
        if count < 1:
            raise ValueError(f"Ligand repeat count must be >= 1 in {spec!r}")
    else:
        base, count = text, 1
    if ":" in base:
        lhs, rhs = base.rsplit(":", 1)
        if rhs.lstrip().startswith("["):
            donors = json.loads(rhs)
            return [LigandInput(lhs, donors=donors) for _ in range(count)]
    return load_ligand_library().expand_ligand_spec(text)


def _solve_dataset(args: argparse.Namespace) -> int:
    from fit.parser import TMQMParser

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db) if args.db else default_db_path()
    if db_path is None or not db_path.is_file():
        raise JacobikEnvironmentError(
            f"Constraints DB not found (pass --db, set JACOBIK_DB, or ensure {PRODUCTION_DB_NAME} is in jacobik/data or data/db)"
        )
    require_xtb = (args.embed or ("etkdg" if args.fast_embed else "gfn2")) == "gfn2"
    check_environment(require_xtb=require_xtb, require_db=True, constraints_db=db_path, device=args.device)

    stats = StatsLookup(load_db(db_path))
    parser = TMQMParser(args.dataset)
    codes_filter = set(args.codes) if args.codes else None
    dev = resolve_device(args.device)

    n_ok = 0
    n_fail = 0
    for mol in parser.stream_split(args.split):
        if codes_filter is not None and mol.csd_code not in codes_filter:
            continue
        t0 = time.perf_counter()
        try:
            groups = _build_ligand_groups(mol, mol.smiles)
            lig_inputs = ligand_inputs_from_mol(mol, groups)
            builder = ComplexBuilder(
                metal=mol.metal_symbol,
                geometry=mol.coord_geom,
                charge=mol.charge,
                spin=mol.spin_state,
                ligands=lig_inputs,
            )
            res = builder.solve(
                stats=stats,
                seed=args.seed,
                device=dev,
                n_conformers=args.n_conformers,
                beam_width=args.beam_width,
                n_ik_steps=args.n_ik_steps,
                n_coord_samples=args.n_coord_samples,
                embed=args.embed,
                fast_embed=args.fast_embed,
                use_symmetry_enumeration=not args.no_symmetry_enumeration,
                out_per_class=args.out_per_class,
                out_per_class_k=args.out_per_class_k,
                out_per_class_window=args.out_per_class_window,
                out_window=args.out_window,
                dedup_rmsd=args.dedup_rmsd,
                dedup_max_dev=args.dedup_max_dev,
            )
            elapsed = time.perf_counter() - t0
            if not res.conformers:
                n_fail += 1
                log.error("%s produced no conformers (%.2fs): %s", mol.csd_code, elapsed, res.failure)
                continue
            target_dir = out_dir / mol.csd_code
            target_dir.mkdir(parents=True, exist_ok=True)
            res.write_xyz(target_dir / f"{mol.csd_code}_ensemble.xyz", by_class=bool(args.by_class))
            n_ok += 1
            log.info("Finished %s in %.2fs (best score %.4f, %d conformers)", mol.csd_code, elapsed, res.best.score, len(res.conformers))
        except Exception as exc:
            n_fail += 1
            log.error("Failed %s: %s", getattr(mol, "csd_code", "?"), exc)

    log.info("Batch complete: %d succeeded, %d failed under %s", n_ok, n_fail, out_dir)
    return 0 if n_ok > 0 or n_fail == 0 else 1


def _solve_single_complex(
    args: argparse.Namespace,
    *,
    metal: str,
    geometry: str,
    charge: int,
    spin: int,
    ligands: List[LigandInput],
    oxidation_state: Optional[int],
    xyz_stem: str,
) -> int:
    res = generate(
        metal=metal,
        geometry=geometry,
        charge=charge,
        spin=spin,
        ligands=ligands,
        oxidation_state=oxidation_state,
        constraints_db=args.db,
        device=args.device,
        seed=args.seed,
        n_conformers=args.n_conformers,
        beam_width=args.beam_width,
        n_ik_steps=args.n_ik_steps,
        n_coord_samples=args.n_coord_samples,
        embed=args.embed,
        fast_embed=args.fast_embed,
        use_symmetry_enumeration=not args.no_symmetry_enumeration,
        out_per_class=args.out_per_class,
        out_per_class_k=args.out_per_class_k,
        out_per_class_window=args.out_per_class_window,
        out_window=args.out_window,
        dedup_rmsd=args.dedup_rmsd,
        dedup_max_dev=args.dedup_max_dev,
    )
    res.summary()
    out_target = args.out or (Path(args.out_dir) / f"{xyz_stem}.xyz" if args.out_dir else None)
    if out_target is not None:
        out_path = Path(out_target)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        res.write_xyz(out_path, by_class=bool(args.by_class))
    return 0 if res.conformers else 1


def _solve_smiles(args: argparse.Namespace) -> int:
    ligands: List[LigandInput] = []
    for spec in args.ligand:
        ligands.extend(_parse_ligand_spec(spec))
    return _solve_single_complex(
        args,
        metal=args.metal,
        geometry=args.geometry,
        charge=args.charge,
        spin=args.spin,
        ligands=ligands,
        oxidation_state=args.oxidation_state,
        xyz_stem=f"{args.metal}_{args.geometry}",
    )


def _solve_debug_complex(args: argparse.Namespace) -> int:
    lib = load_ligand_library()
    dc = lib.debug_complex(args.debug_complex)
    return _solve_single_complex(
        args,
        metal=dc.metal,
        geometry=dc.geometry,
        charge=dc.charge,
        spin=dc.spin,
        ligands=list(dc.ligands),
        oxidation_state=args.oxidation_state,
        xyz_stem=args.debug_complex.strip().upper(),
    )


def _cmd_ligands(args: argparse.Namespace) -> int:
    lib = load_ligand_library()
    if args.name:
        try:
            ldef, implied_mode = lib.resolve_ligand_def(args.name)
        except KeyError:
            try:
                dc = lib.debug_complex(args.name)
            except KeyError:
                raise
            if args.json:
                print(json.dumps({
                    "type": "debug_complex",
                    "name": args.name.strip().upper(),
                    "metal": dc.metal,
                    "geometry": dc.geometry,
                    "charge": dc.charge,
                    "spin": dc.spin,
                    "ligands": [{"smiles": lig.smiles, "donors": lig.donors, "sites": lig.sites} for lig in dc.ligands],
                }, indent=2))
                return 0
            print(f"{args.name.strip().upper()}: {dc.metal} {dc.geometry} charge={dc.charge} spin={dc.spin}")
            for i, lig in enumerate(dc.ligands, 1):
                print(f"  ligand {i}: {lig.smiles}  donors={lig.donors}")
            return 0

        if args.json:
            print(json.dumps({
                "name": ldef.name,
                "smiles": ldef.smiles,
                "charge": ldef.charge,
                "category": ldef.category,
                "default_mode": implied_mode or ldef.default_mode,
                "aliases": list(ldef.aliases),
                "modes": {
                    k: {
                        "donors": [list(g) for g in bm.donors],
                        "denticity": bm.denticity,
                        "summary": bm.summary,
                        "aliases": list(bm.aliases),
                    }
                    for k, bm in ldef.modes.items()
                },
                "notes": ldef.notes,
            }, indent=2))
            return 0
        print(f"{ldef.name}  ({ldef.category}, charge={ldef.charge}, smiles={ldef.smiles})")
        if implied_mode:
            print(f"  Alias implies mode: {implied_mode}")
        if ldef.aliases:
            print(f"  Aliases: {', '.join(ldef.aliases)}")
        if ldef.notes:
            print(f"  Notes: {ldef.notes}")
        for mode_name, bm in ldef.modes.items():
            mark = "*" if mode_name == ldef.default_mode else " "
            alias_txt = f"  aliases=[{', '.join(bm.aliases)}]" if bm.aliases else ""
            print(f"  {mark} {ldef.name}@{mode_name}: donors={[list(g) for g in bm.donors]}  {bm.summary}{alias_txt}")
        return 0

    rows = lib.list_ligands(query=args.query, category=args.category)
    if args.json:
        print(json.dumps({
            "ligands": [
                {
                    "name": ldef.name,
                    "smiles": ldef.smiles,
                    "charge": ldef.charge,
                    "category": ldef.category,
                    "default_mode": ldef.default_mode,
                    "modes": list(ldef.modes.keys()),
                    "aliases": list(ldef.aliases),
                }
                for ldef in rows
            ],
            "debug_complexes": sorted(lib.DEBUG_COMPLEXES.keys()),
        }, indent=2))
        return 0
    print(f"{'Name':12s} {'q':>3s} {'Category':13s} {'Modes'}")
    print("-" * 72)
    for ldef in rows:
        mode_labels = [f"{m}*" if m == ldef.default_mode else m for m in ldef.modes]
        print(f"{ldef.name:12s} {ldef.charge:>3d} {ldef.category:13s} {', '.join(mode_labels)}")
    print(f"\n{len(rows)} ligands (* = default mode). Debug complexes (--debug-complex): {', '.join(sorted(lib.DEBUG_COMPLEXES))}")
    return 0


def _add_solve_parser(subparsers) -> None:
    db_hint = str(default_db_path() or PRODUCTION_DB_NAME)
    sp = subparsers.add_parser("solve", help="Generate conformers from a debug complex, ligand library/SMILES, or tmQM dataset")
    # Predefined debug complex or dataset batch mode
    sp.add_argument("--debug-complex", default=None, help="Solve a predefined complex from interface/ligand_library.py (e.g. CBTYCO, OJIMIF).")
    sp.add_argument("--dataset", default=None, help="Directory containing tmQM-format input files.")
    sp.add_argument("--split", default="examples", help="Dataset split name (default: examples).")
    sp.add_argument("--codes", nargs="*", default=None, help="Optional subset of CSD codes in dataset mode.")
    # SMILES / library single-complex mode
    sp.add_argument("--metal", default=None, help="Metal element symbol (single-complex mode, e.g. Fe).")
    sp.add_argument("--geometry", default=None, help="Coordination geometry (e.g. octahedral).")
    sp.add_argument("--charge", type=int, default=0, help="Total formal complex charge (default: 0).")
    sp.add_argument("--spin", type=int, default=1, help="Spin multiplicity 2S+1 (default: 1).")
    sp.add_argument("--oxidation-state", type=int, default=None, help="Optional metal oxidation state override.")
    sp.add_argument(
        "--ligand",
        action="append",
        default=[],
        help="Ligand alias/name (e.g. 'bpy', 'dmso@kappa1_O', 'bpy*3'), 'SMILES:[[0],[1]]', or JSON (repeatable).",
    )
    sp.add_argument("--out", default=None, help="Output .xyz file or directory (SMILES mode).")
    sp.add_argument("--out-dir", default=None, help="Output directory (required in --dataset mode).")
    sp.add_argument("--by-class", action="store_true", help="Write one XYZ file per isomer class.")
    # Shared solver options
    sp.add_argument("--db", default=None, help=f"Path to constraints DB (default: {db_hint}).")
    sp.add_argument("--n-conformers", type=int, default=None, help="Max conformers kept.")
    sp.add_argument("--beam-width", type=int, default=16, help="Torsion beam width per ligand (default: 16).")
    sp.add_argument("--n-ik-steps", type=int, default=200, help="IK refinement iterations (default: 200).")
    sp.add_argument("--n-coord-samples", type=int, default=8, help="Coordination-sphere distance samples (default: 8).")
    sp.add_argument("--seed", type=int, default=11, help="Random seed (default: 11).")
    sp.add_argument("--device", default="auto", help="Compute device: auto, cpu, or cuda (default: auto).")
    sp.add_argument("--embed", default=None, choices=["gfn2", "mmff", "etkdg"], help="Ligand embed method.")
    sp.add_argument("--fast-embed", action="store_true", help="Alias for --embed etkdg.")
    sp.add_argument("--no-symmetry-enumeration", action="store_true", help="Disable site-assignment enumeration.")
    sp.add_argument("--out-per-class", action="store_true", default=None, help="Keep conformers per isomer class.")
    sp.add_argument("--out-per-class-k", type=int, default=None, help="Conformers kept per class (default: 8).")
    sp.add_argument("--out-per-class-window", type=float, default=None, help="Per-class nat window (default: 35).")
    sp.add_argument("--out-window", type=float, default=None, help="Global nat score window (default: 200).")
    sp.add_argument("--dedup-rmsd", type=float, default=None, help="Heavy-atom symmetry RMSD threshold (A).")
    sp.add_argument("--dedup-max-dev", type=float, default=None, help="Max per-atom deviation threshold (A, 0=off).")
    sp.add_argument("-v", "--verbose", action="store_true", help="Enable INFO logging.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jacobik", description="Jacobik transition-metal complex conformer generator")
    sub = parser.add_subparsers(dest="command")
    _add_solve_parser(sub)

    ck = sub.add_parser("check", help="Validate runtime dependencies (xtb, constraints DB, compute device)")
    ck.add_argument("--db", default=None, help="Optional explicit constraints DB path to check.")
    ck.add_argument("--device", default=None, help="Optional compute device to check (e.g. cuda, cpu).")
    ck.add_argument("--no-xtb", action="store_true", help="Skip checking for external xtb binary.")

    fp = sub.add_parser("fingerprint", help="Print the resolved production configuration fingerprint")
    fp.add_argument("--json", action="store_true", help="Print full profile snapshot as JSON.")

    lg = sub.add_parser("ligands", help="List or inspect curated ligands, aliases, binding modes, and debug complexes")
    lg.add_argument("name", nargs="?", default=None, help="Optional ligand name, alias, or debug complex (e.g. dmso, bpy, CBTYCO).")
    lg.add_argument("--query", "-q", default=None, help="Filter ligands by substring in names, aliases, modes, or notes.")
    lg.add_argument("--category", "-c", default=None, help="Filter ligands by category (e.g. bidentate, phosphine, haptic).")
    lg.add_argument("--json", action="store_true", help="Print results as JSON.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point installed as ``jacobik``."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not args_list:
        parser.print_help()
        return 0
    args = parser.parse_args(args_list)

    if args.command == "fingerprint":
        snap = _profile.snapshot()
        if args.json:
            print(json.dumps(snap, indent=2, default=str))
        else:
            print(snap["fingerprint"])
        return 0

    if args.command == "check":
        try:
            info = check_environment(
                require_xtb=not args.no_xtb,
                constraints_db=args.db,
                device=args.device,
            )
        except JacobikEnvironmentError as exc:
            print(f"Environment check failed: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(info, indent=2))
        return 0

    if args.command == "ligands":
        try:
            return _cmd_ligands(args)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            print(f"Ligand library error: {exc}", file=sys.stderr)
            return 1

    if args.command == "solve":
        logging.basicConfig(
            level=logging.INFO if args.verbose else logging.WARNING,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        try:
            if args.debug_complex is not None:
                return _solve_debug_complex(args)
            if args.dataset is not None:
                if not args.out_dir:
                    parser.error("--out-dir is required with --dataset")
                return _solve_dataset(args)
            if args.metal and args.geometry and args.ligand:
                return _solve_smiles(args)
            parser.error(
                "solve requires --debug-complex NAME, --metal M --geometry G --ligand ..., "
                "or --dataset DIR --out-dir DIR"
            )
        except (JacobikEnvironmentError, KeyError, ValueError, FileNotFoundError) as exc:
            print(f"Solve failed: {exc}", file=sys.stderr)
            return 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
