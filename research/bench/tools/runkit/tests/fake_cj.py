"""A fake cj.py for the colab tests: both a module (balance, remote_py, START_BALANCE, CAP_UNITS) and a CLI.
State lives in the directory $FAKE_CJ_STATE: down.log (one line per `down`), up_mode (ok|fail), files/ (served by `get`),
balance (a float), jobs (text: lines `tag done|running`)."""
import json
import os
import sys
import time
from pathlib import Path

START_BALANCE = 100.0
CAP_UNITS = 50.0


def _st() -> Path:
    return Path(os.environ["FAKE_CJ_STATE"])


def balance():
    p = _st() / "balance"
    return (float(p.read_text()) if p.exists() else 90.0), 0, "fake"


def remote_py(name, code, timeout=120):
    jobs = {}
    p = _st() / "jobs"
    if p.exists():
        for ln in p.read_text().splitlines():
            t, s = ln.split()
            jobs[t] = s == "done"
    return "RK_STATUS " + json.dumps(dict(uptime="up", gpu=(_st() / "gpu").read_text() if (_st() / "gpu").exists() else "0 %, 10 MiB",
                                          jobs=jobs, counts={}))


def main():
    cmd = sys.argv[1]
    a = sys.argv[2:]
    if cmd == "up":
        time.sleep(0.3)
        if (_st() / "up_mode").exists() and (_st() / "up_mode").read_text().strip() == "fail":
            print("fingerprint check FAILED")
            sys.exit(1)
        print("READY: fake on L4", flush=True)
        time.sleep(float(os.environ.get("FAKE_UP_LINGER", "0")))
    elif cmd == "down":
        with open(_st() / "down.log", "a") as fh:
            fh.write(a[a.index("--name") + 1] + "\n")
        print("stopped")
    elif cmd == "get":
        remote, local = a[-2], a[-1]
        base = os.path.basename(remote.rstrip("/"))
        out = Path(local) / base
        out.mkdir(parents=True, exist_ok=True)
        src = _st() / "files"
        for f in src.glob("*") if src.exists() else []:
            (out / f.name).write_bytes(f.read_bytes())
    elif cmd in ("run", "wait"):
        print("ok")
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
