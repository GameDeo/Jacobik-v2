#!/usr/bin/env python3
"""
Run the Q2/Q3 GMM-on-cos(angle) refit to a TEMP output path (does NOT touch
the live constraints.json -- fit_all()'s `database` dict starts empty, so
writing straight to the live out_path would silently delete every other
quantity's entries). merge_constraints.py does the actual safe merge
afterwards.

Loaded via importlib (not `import jacobik.stats.fitter`) because fitter.py
has no jacobik-internal imports of its own, but going through the jacobik
package __init__ chain would pull in engine.py -> torch, which isn't
installed in this shell.
"""
import importlib.util
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[1]  # .../Jacobik v3
FITTER_PATH = ROOT / "src" / "jacobik" / "stats" / "fitter.py"
EXPANDED_DIR = ROOT / "data" / "out" / "expanded"
TEMP_OUT = ROOT / "_patch_scripts" / "constraints_q2q3_refit.json"

spec = importlib.util.spec_from_file_location("jacobik_fitter_standalone", FITTER_PATH)
fitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fitter)

print(f"Loaded fitter from {FITTER_PATH}")
print(f"_DIST_TYPE = {fitter._DIST_TYPE}")
assert fitter._DIST_TYPE["Q2"] == "gmm_cos", "fitter.py patch did not apply -- Q2 still vmm"
assert fitter._DIST_TYPE["Q3"] == "gmm_cos", "fitter.py patch did not apply -- Q3 still vmm"
assert hasattr(fitter, "fit_gmm_cosangle"), "fit_gmm_cosangle missing"

t0 = time.perf_counter()
summary = fitter.fit_all(
    expanded_dir=EXPANDED_DIR,
    out_path=TEMP_OUT,
    quantities=["Q2", "Q3"],
)
elapsed = time.perf_counter() - t0
print(f"\nDONE in {elapsed:.1f}s")
print(summary)
