#!/usr/bin/env python3
"""
Conformer generation CLI wrapper delegating to ``jacobik.main``.

Examples::

  python interface/jacobik_cli.py --debug-complex CBTYCO --out-dir ./output --n-conformers 3
  python interface/jacobik_cli.py --metal Fe --geometry octahedral --charge 2 --spin 1 --ligand "bpy*3"
  python interface/jacobik_cli.py ligands --query thiocyanate
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List, Optional

if importlib.util.find_spec("jacobik") is None:
    _repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_repo / "src"))

import jacobik  # noqa: E402

_SUBCOMMANDS = {"solve", "check", "fingerprint", "ligands"}


def main(argv: Optional[List[str]] = None) -> int:
    """Delegate CLI invocation to ``jacobik.main``, defaulting flat flags to ``solve``."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] not in _SUBCOMMANDS and args_list[0] not in ("-h", "--help"):
        args_list = ["solve", *args_list]
    return jacobik.main(args_list)


if __name__ == "__main__":
    raise SystemExit(main())
