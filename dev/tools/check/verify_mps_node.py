"""Verify the four-runner MPS smoke outputs; report power for human review."""
import json
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1])
errors = []
for gpu in range(4):
    directory = root / f"g{gpu}"
    try:
        mps = json.loads((directory / "mps.json").read_text())
        events = [json.loads(line) for line in (directory / "run_events.jsonl").read_text().splitlines()]
        telemetry = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
        power = [r["gpu_power_w"] for r in telemetry if r.get("gpu_power_w") is not None]
        done = [r for r in events if r.get("kind") == "parent_done" and r.get("ok")]
        records = []
        for archive in directory.glob("run_pack*.tar"):
            with tarfile.open(archive) as tar:
                for member in tar:
                    if member.isfile() and member.name.startswith("C/records/") and member.name.endswith(".json"):
                        records.append(json.load(tar.extractfile(member)))
        if not mps.get("stopped") or mps.get("left"):
            errors.append(f"g{gpu}: MPS shutdown not clean: {mps}")
        if len(done) < 2:
            errors.append(f"g{gpu}: expected two successful lane events; got {len(done)}")
        if len(records) < 2 or any(record.get("status") != "ok" for record in records):
            errors.append(f"g{gpu}: expected two ok control records; got {[r.get('status') for r in records]}")
        if not power:
            errors.append(f"g{gpu}: no power samples")
        print(f"g{gpu}: successful parents={len(done)}, power max={max(power) if power else None} W, mps={mps}")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"g{gpu}: {exc}")
for error in errors:
    print(error, file=sys.stderr)
sys.exit(bool(errors))
