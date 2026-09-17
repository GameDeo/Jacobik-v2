#!/usr/bin/env python3
"""
Standalone reproduction + verification for the _is_empty_dist() type-
whitelist bug in stats.py, using real code against the real live
constraints.json (no mocking of the resolver logic).

Loaded via importlib + fake package registration (not `import jacobik...`)
because stats.py does `from jacobik.topology.donor_hierarchy import ...`,
and going through the real jacobik package __init__ chain would pull in
engine.py -> torch, which isn't installed in this shell. donor_hierarchy.py
itself has no heavy deps, so we register it under the expected dotted name
and let stats.py's relative-looking absolute import resolve normally.
"""
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # .../Jacobik v3
SRC = ROOT / "src"


def load_module(dotted_name, file_path):
    spec = importlib.util.spec_from_file_location(dotted_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Register empty namespace packages so `from jacobik.topology.donor_hierarchy
# import X` resolves without executing jacobik/__init__.py or
# jacobik/topology/__init__.py (which might import other heavy stuff).
for pkg in ("jacobik", "jacobik.topology", "jacobik.stats"):
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = []  # mark as package
        sys.modules[pkg] = m

load_module("jacobik.topology.donor_hierarchy", SRC / "jacobik" / "topology" / "donor_hierarchy.py")
stats = load_module("jacobik.stats.stats", SRC / "jacobik" / "stats" / "stats.py")

print("Loaded stats.py standalone OK")
print()

# ---- Reproduce the exact bug -----------------------------------------------
print("=== _is_empty_dist() type whitelist check ===")
import inspect

src_lines = inspect.getsource(stats._is_empty_dist)
print(src_lines)

# ---- Load the LIVE constraints.json and try real lookups ------------------
CONSTRAINTS = ROOT / "src" / "jacobik" / "constraints.json"
db = stats.load_db(CONSTRAINTS, n_min=1)
lookup = stats.StatsLookup(db, n_min=30)

print("=== Live dist() calls (should reproduce the user's failures) ===")

test_cases = [
    ("Q2", ("square_plan", "different_ligand", "Pt", "Hal_Cl", "Hal_Cl")),
    ("Q3", ("N_amine", "Co", "octahedral")),
    ("Q2", ("octahedral", "chelate_short", "Co", "N_amine", "N_amine")),
    ("Q3", ("C_carbonyl", "Fe", "trig_bipyr")),
]

for quantity, key in test_cases:
    try:
        fd, resolved_key, candidates = lookup.dist_trace(quantity, key)
        print(f"{quantity} {key} -> OK type={fd.type} n={fd.n} resolved_key={resolved_key}")
    except Exception as exc:
        print(f"{quantity} {key} -> FAILED: {exc}")

print()
print("=== Direct entry check: does the exact key exist and what does")
print("    _is_empty_dist() say about it? ===")
for quantity, key in test_cases:
    q_data = db.data.get(quantity, {})
    fd = q_data.get(key)
    if fd is None:
        print(f"{quantity} {key}: NOT IN db.data (unexpected)")
    else:
        print(
            f"{quantity} {key}: type={fd.type!r} n={fd.n} "
            f"_is_empty_dist={stats._is_empty_dist(fd)}"
        )
