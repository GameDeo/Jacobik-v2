# Changelog

One line per change: date, what, why, how it was validated. Newest first.

- **2026-11-14 (`v1.0a`, fingerprint `2c87e2940af2`):** Production baseline consolidating P2R search/stereo, DS fitting/typing (`donor_identity_reconcile` merged into `donor_assignment`), packaged default DB (`constraints_2026-11-12.npz`), `interface/ligand_library.py`, compacted `dev/tests/` suite, MIT licensing, and clean docs; validated by Perlmutter `v1.0a` gate (`PERFORMANCE.md`), Perlmutter DB rebuild (`61319813`, all 419,631 active entries identical vs `constraints_2026-11-12.npz` excluding 13 retired `NCI` entries), and local `ws` fingerprint + unit checks.
