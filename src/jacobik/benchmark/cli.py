"""Command-line entry point for acquiring and benchmarking tmQM."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    # The established report uses a few Unicode glyphs. Keep Windows cp1252
    # consoles from aborting an otherwise valid long benchmark run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(prog="jacobik-benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download the public tmQM dataset")
    fetch.add_argument("data_dir")
    fetch.add_argument("--splits", nargs="+", default=["X1"], choices=["X1", "X2", "X3"])
    fetch.add_argument("--force", action="store_true")

    run = sub.add_parser("run", help="reconstruct tmQM complexes and compare structures")
    run.add_argument("data_dir")
    run.add_argument("--split", default="X1", choices=["X1", "X2", "X3"])
    run.add_argument("--limit", type=int)
    run.add_argument("--output", default="benchmark.json")
    run.add_argument("--constraints-db")
    run.add_argument("--beam-width", type=int, default=16)
    run.add_argument("--n-ik-steps", type=int, default=200)
    run.add_argument("--n-coord-samples", type=int, default=8)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run.add_argument("--max-sym-perms", type=int, default=2048)
    run.add_argument("--mace", action="store_true", help="enable MACE-POLAR-1 energies")
    run.add_argument("--mace-model", choices=["polar-1-s", "polar-1-m", "polar-1-l"], default="polar-1-m")

    args = parser.parse_args()
    if args.command == "fetch":
        from .dataset import fetch_tmqm
        hashes = fetch_tmqm(args.data_dir, tuple(args.splits), force=args.force)
        for name, digest in hashes.items():
            print(f"{digest}  {name}")
        return

    # Delegate to the established streaming reconstruction benchmark; this
    # wrapper supplies the modern energy backend and a stable installed CLI.
    import json
    from dataclasses import asdict
    from pathlib import Path
    from jacobik.vis.benchmark import run_benchmark

    summary = run_benchmark(
        args.data_dir, args.split, constraints_db=args.constraints_db,
        limit=args.limit, beam_width=args.beam_width,
        n_ik_steps=args.n_ik_steps, n_coord_samples=args.n_coord_samples,
        seed=args.seed, device=args.device, max_sym_perms=args.max_sym_perms,
        use_mace=args.mace, mace_model=args.mace_model,
    )
    summary.print_summary()
    output = Path(args.output)
    output.write_text(json.dumps([asdict(r) for r in summary.records], indent=2), encoding="utf-8")
    print(f"Records saved to {output}")


if __name__ == "__main__":
    main()
