"""Tiny MACE-POLAR-1-S device check for the D-B Colab allocation."""
import json
import numpy as np
from tools.runkit import mace

ev = mace.Evaluator(device="cuda", dtype="float64", threads=1)
energy = ev.energies_kj([(["H", "H"], np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]), 0, 1)])
print(json.dumps({"energy_kj": energy.tolist(), "device": ev.device, "model_sha256": ev.model_sha256, "stats": ev.stats}))
