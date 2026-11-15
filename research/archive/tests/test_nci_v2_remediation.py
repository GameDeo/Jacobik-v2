"""Tests for the R1/R3/R4 remediation: whitelist graph keys, code-state hash, coordination integrity."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "src", REPO / "refit"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from refit.nci_v2_frame import ligand_graph_key, ligand_key_from_smiles, load_whitelist_keys
from nci_v2_provenance import code_state_sha256

SPEC = REPO / "data/refit/nci_v2/pilot_v1/spec.json"


def test_whitelist_keys_frozen_and_rederivable():
    spec = json.loads(SPEC.read_text())
    keys = load_whitelist_keys(spec)  # asserts SMILES -> key matches the frozen key
    assert keys["PPh3"] == keys["PCy3"]  # H-less heavy-atom key: documented key-twins
    assert keys["IMes"] == keys["SIMes"]
    assert len(set(keys.values())) < len(keys)


def test_ligand_key_ignores_aromaticity_but_not_connectivity():
    assert ligand_key_from_smiles("c1ccncc1") == ligand_key_from_smiles("C1CCNCC1")
    assert ligand_key_from_smiles("c1ccncc1") != ligand_key_from_smiles("Cc1ccccc1")


def test_spurious_ring_contact_changes_key():
    """A Wiberg 1,4 ring contact is an extra edge and yields a different key; the frame must drop it."""
    elements = {i: "C" for i in range(6)}
    ring = [(i, (i + 1) % 6) for i in range(6)]
    assert ligand_graph_key(elements, ring) != ligand_graph_key(elements, ring + [(0, 3)])


def test_code_state_hash_covers_untracked_files(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    assert code_state_sha256(tmp_path) == "clean"
    (tmp_path / "refit").mkdir()
    new = tmp_path / "refit" / "b.py"
    new.write_text("y = 1\n")
    h1 = code_state_sha256(tmp_path)
    new.write_text("y = 2\n")
    h2 = code_state_sha256(tmp_path)
    assert h1 not in ("clean", "unknown") and h1 != h2


def test_code_state_hash_covers_tools_directory(tmp_path):
    """Verify that an untracked or modified file under tools/ changes code_state_sha256."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"], cwd=tmp_path, check=True)
    assert code_state_sha256(tmp_path) == "clean"

    # Untracked file under tools/
    (tmp_path / "tools").mkdir()
    tool_file = tmp_path / "tools" / "script.py"
    tool_file.write_text("print('tool')\n")
    h_untracked = code_state_sha256(tmp_path)
    assert h_untracked not in ("clean", "unknown")

    # Track and commit file under tools/
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "add tools"], cwd=tmp_path, check=True)
    assert code_state_sha256(tmp_path) == "clean"

    # Modify tracked file under tools/
    tool_file.write_text("print('tool modified')\n")
    h_modified = code_state_sha256(tmp_path)
    assert h_modified not in ("clean", "unknown")
    assert h_modified != h_untracked


def _cp_sandwich(slip: float) -> tuple:
    """Zr with one eta5 C5 ring 2.2 A below; `slip` shifts the ring sideways (eta5 -> eta3/eta1)."""
    ang = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    ring = np.c_[1.2 * np.cos(ang), 1.2 * np.sin(ang), np.full(5, -2.2)]
    ref = np.vstack([[0.0, 0.0, 0.0], ring, [[2.4, 0.0, 0.5]]])
    gen = ref.copy()
    gen[1:6, 0] += slip
    return ["Zr", "C", "C", "C", "C", "C", "Cl"], ref, gen


def test_coordination_integrity_flags_haptic_slip_only():
    pytest.importorskip("torch")
    from nci_v2_pilot import coordination_integrity

    sym, ref, _ = _cp_sandwich(0.0)
    ok = coordination_integrity(sym, ref, ref[None])
    assert ok["haptic_group_sizes"] == [5] and ok["n_sigma_donors"] == 1 and bool(ok["intact"][0])
    _, _, slipped = _cp_sandwich(0.9)
    bad = coordination_integrity(sym, ref, slipped[None])
    assert bad["haptic_slip_A"][0] > 0.3 and not bool(bad["intact"][0])
