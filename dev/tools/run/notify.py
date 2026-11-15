"""Discord job notifications (stdlib only; never fails the caller).

The webhook URL is a secret and is never committed: it is read from $JK_DISCORD_WEBHOOK or from
~/.config/jacobik/discord_webhook (chmod 600). Without one every command is a silent no-op.

    python -m tools.run.notify msg "text"
    python -m tools.run.notify file PATH [PATH ...] --text "text"     # attachments, gzipped when large
    python -m tools.run.notify watch --out DIR --every 900 [--label NAME]   # periodic stats until killed
    python -m tools.run.notify stats --out DIR                       # print the stats line (no send)

Stats are read from the tools.run outputs under DIR: every */g*/run_plan.json (parents) and run_events.jsonl
(parent_done ok / failed), plus DIR/gpu_power.csv (last sample per GPU) and DIR/load.txt.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path

LIMIT = 9_500_000          # Discord webhook upload limit is 10 MB without boosts
UA = "jacobik-notify/1.0"


def webhook() -> str | None:
    url = os.environ.get("JK_DISCORD_WEBHOOK")
    if not url:
        f = Path.home() / ".config" / "jacobik" / "discord_webhook"
        if f.is_file():
            url = f.read_text().strip()
    return url or None


def _post(body: bytes, ctype: str) -> bool:
    url = webhook()
    if not url:
        return False
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": ctype, "User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return 200 <= r.status < 300
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(float(e.headers.get("Retry-After", 2)))
                continue
            print(f"notify: HTTP {e.code}", file=sys.stderr)
            return False
        except Exception as e:  # network down, DNS, timeout: never fail the job
            print(f"notify: {type(e).__name__}: {e}", file=sys.stderr)
            time.sleep(2)
    return False


def send(text: str) -> bool:
    return _post(json.dumps({"content": text[:1990]}).encode(), "application/json")


def _attachment(path: Path) -> tuple[str, bytes]:
    data = path.read_bytes()
    if len(data) <= LIMIT:
        return path.name, data
    z = gzip.compress(data)
    if len(z) <= LIMIT:
        return path.name + ".gz", z
    return path.name + ".tail", data[-LIMIT:]


def send_files(paths: list[Path], text: str = "") -> bool:
    files = [_attachment(p) for p in paths if p.is_file()]
    if not files:
        return send(text + "\n(no log files found)")
    # one message per file keeps each under the size limit
    ok = True
    for i, (name, data) in enumerate(files):
        b = uuid.uuid4().hex
        payload = json.dumps({"content": (text if i == 0 else "")[:1990]})
        parts = [f'--{b}\r\nContent-Disposition: form-data; name="payload_json"\r\n'
                 f'Content-Type: application/json\r\n\r\n{payload}\r\n'.encode(),
                 f'--{b}\r\nContent-Disposition: form-data; name="files[0]"; filename="{name}"\r\n'
                 f'Content-Type: application/octet-stream\r\n\r\n'.encode(), data, f'\r\n--{b}--\r\n'.encode()]
        ok &= _post(b"".join(parts), f"multipart/form-data; boundary={b}")
    return ok


def stats(out: Path, t0: float | None = None) -> str:
    """Per phase: parents done / parents in the phase. Under a shared claim queue every GPU's plan lists all parents, so
    the phase total is the largest plan and 'done' is the union of parents finished on any GPU."""
    phases: dict[str, dict] = {}
    # phase layouts: <out>/<phase>/g<N>/run_plan.json (D3 jobs) and <out>/g<N>/run_plan.json (pm-full gates: one phase)
    for plan in sorted(list(out.glob("*/g*/run_plan.json")) + list(out.glob("g*/run_plan.json"))):
        try:
            n = len(json.loads(plan.read_text()).get("parents", []))
        except Exception:
            continue
        phase_dir = plan.parent.parent
        ph = phases.setdefault(phase_dir.name if phase_dir != out else "gate", dict(n=0, last={}, claim=False))
        claimed = (phase_dir / "claims").is_dir()
        ph["claim"] = claimed
        ph["n"] = max(ph["n"], n) if claimed else ph["n"] + n
        ev = plan.parent / "run_events.jsonl"
        if ev.is_file():
            for line in ev.read_text(errors="replace").splitlines():
                if '"parent_done"' in line:      # retries / per-arm events count once per parent
                    try:
                        e = json.loads(line)
                        ph["last"][e.get("code")] = ph["last"].get(e.get("code"), True) and bool(e.get("ok"))
                    except Exception:
                        pass
    lines = []
    if t0:
        lines.append(f"elapsed {int((time.time() - t0) / 60)} min")
    for name, ph in phases.items():
        bad = sum(1 for v in ph["last"].values() if not v)
        lines.append(f"`{name}` {len(ph['last'])}/{ph['n']}" + (f" ({bad} failed)" if bad else ""))
    rows = phases
    gp = out / "gpu_power.csv"
    if gp.is_file():
        last = gp.read_text().strip().splitlines()[-4:]
        try:
            w = [float(l.split(",")[1].split()[0]) for l in last]
            lines.append("GPU W " + " / ".join(f"{x:.0f}" for x in w))
        except Exception:
            pass
    ld = out / "load.txt"
    if ld.is_file():
        s = ld.read_text().strip().splitlines()
        if s:
            lines.append("load " + s[-1].split("load average:")[-1].strip())
    if not phases:
        lines.append("no phase started yet")
    return " | ".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.run.notify")
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("msg"); m.add_argument("text")
    f = sub.add_parser("file"); f.add_argument("paths", nargs="+", type=Path); f.add_argument("--text", default="")
    w = sub.add_parser("watch"); w.add_argument("--out", type=Path, required=True)
    w.add_argument("--every", type=int, default=900); w.add_argument("--label", default="")
    s = sub.add_parser("stats"); s.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "msg":
            send(a.text)
        elif a.cmd == "file":
            send_files(a.paths, a.text)
        elif a.cmd == "stats":
            print(stats(a.out))
        elif a.cmd == "watch":
            t0 = time.time()
            while True:
                time.sleep(max(60, a.every))
                send(f"{a.label} RUNNING | " + stats(a.out, t0))
    except KeyboardInterrupt:
        pass
    except Exception as e:  # never fail the job
        print(f"notify: {type(e).__name__}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
