"""Colab job helper: provision one Colab GPU VM, ship a code ref plus the read-only inputs, run jobs in the background, fetch results.

Every Colab CLI call has a hard timeout and retries (the CLI websocket drops: "Connection was lost"). Local host only; Python 3.11 (pixi env).

  cj.py credits                                   balance, spent against the cap
  cj.py up   --name N [--gpu L4] --repo PATH --ref REF   new VM, bundle REF of the git repo at PATH, upload, build env, verify fingerprint
  cj.py push --name N LOCAL REMOTE                chunked upload of one local file
  cj.py run  --name N --tag T [--env K=V ...] -- CMD...   background shell job in /content/jk, log /content/jobs/T.log
  cj.py wait --name N --tag T [--max-min M]       poll until the job ends (keeps the VM busy); prints the log tail and exit code
  cj.py tail --name N --tag T [--bytes B]
  cj.py get  --name N REMOTE LOCAL_DIR            tar REMOTE (file or dir) on the VM, download, extract into LOCAL_DIR
  cj.py down --name N                             stop the VM (always do this when done)

Rules enforced here: at most MAX_VMS VMs at once (all agents together); a credit cap (CAP_UNITS spent since START_BALANCE) blocks `up` and `run`;
the tmQM/lnQM CSV is shipped sanitized (CSD_code, CSD_years, SMILES only: no QM property columns).
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

MAIN = "/mnt/e/Dev/Projects/jacobik_2026-10-02_full"
WORK = os.path.join(MAIN, "research/gpuwork/out/colab")
START_BALANCE = 530.79
CAP_UNITS = 180.0
MAX_VMS = 4  # owner 2026-10-26: spin up VMs as needed (agents share this limit); always `down`
CHUNK = "15M"
FINGERPRINT = "a47283387172"  # P2R fold-in (prepared 2026-11-13); supersedes d5e2f4113497
PATH_ENV = os.path.expanduser("~/.local/bin") + ":" + os.path.expanduser("~/.pixi/bin") + ":" + os.environ.get("PATH", "")
REMOTE_ENV = {
    "PYTHONPATH": "/content/jk/src:/content/jk:/content/jk/refit:/content/jk/research/refit:/content/jk/research/bench:/content/jk/research/bench/tools/jdev/budget",
    "JACOBIK_XTB": "/content/jk/.pixi/envs/linux/bin/xtb",
    "JACOBIK_DATA_ROOT": "/content/jk/data",
}
DATA_FILES = [
    # panel definitions and reference energies: the VM has no research/data, so the bench falls back to
    # data/refit/nci_v2/scaled_v1 (locally a symlink to research/data/refit_nci_v2/scaled_v1; copy2 follows it)
    "data/refit/nci_v2/pilot_v1/spec.json",
    "data/refit/nci_v2/pilot_v1/frame.json",
    "data/refit/nci_v2/scaled_v1/spec.json",
    "data/refit/nci_v2/scaled_v1/frame.json",
    "data/refit/nci_v2/scaled_v1/manifest.json",
    "data/refit/nci_v2/scaled_v1/ref_mace_cache.json",
    "data/db/constraints_2026-11-12.npz",
    "data/db/constraints_2026-11-12.meta.json",
]
DATA_GLOBS = ["data/ref/tmQM/xyz/*", "data/ref/tmQM/bo/*", "data/ref/lnQM/xyz/*", "data/ref/lnQM/bo/*"]
CSVS = ["data/ref/tmQM/tmQM_y.csv", "data/ref/lnQM/tmQM_y.csv"]
CSV_KEEP = ("CSD_code", "SMILES")


def colab(args, timeout=120, stdin=None, retries=3, check=True):
    """Run one Colab CLI call with a hard timeout; retry on timeouts and dropped connections."""
    last = None
    for attempt in range(retries):
        try:
            r = subprocess.run(["colab", "--auth=adc"] + args, input=stdin, capture_output=True, text=True,
                               timeout=timeout, env=dict(os.environ, PATH=PATH_ENV))
            out = r.stdout + r.stderr
            if r.returncode == 0 and "Connection was lost" not in out and "Traceback" not in r.stderr:
                return r.stdout
            last = out[-1500:]
        except subprocess.TimeoutExpired:
            last = "timeout after %ds" % timeout
        time.sleep(10 * (attempt + 1))
    if check:
        sys.exit("colab %s failed after %d tries: %s" % (" ".join(args[:3]), retries, last))
    return None


def remote_py(name, code, timeout=120):
    return colab(["exec", "-s", name], timeout=timeout, stdin=code)


def remote_sh(name, cmd, timeout=120):
    code = "import subprocess\nr = subprocess.run(%r, shell=True, capture_output=True, text=True)\nprint(r.stdout[-6000:]); print(r.stderr[-3000:])\n" % cmd
    return remote_py(name, code, timeout)


def balance():
    out = colab(["usage"], timeout=60)
    m = re.search(r"Current balance:\s*([\d.]+)", out)
    a = re.search(r"Active assignments:\s*(\d+)", out)
    return float(m.group(1)), int(a.group(1)) if a else 0, out


def guard_credits():
    bal, active, _ = balance()
    spent = START_BALANCE - bal
    if spent >= CAP_UNITS:
        sys.exit("credit cap reached: spent %.2f of %.0f units; ask the owner" % (spent, CAP_UNITS))
    return bal, active


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_bundle(repo, ref):
    commit = subprocess.check_output(["git", "-C", repo, "rev-parse", ref], text=True).strip()
    out = os.path.join(WORK, "bundles", commit[:12] + ".tgz")
    if os.path.exists(out):
        return out, commit
    stage = os.path.join(WORK, "bundles", commit[:12])
    shutil.rmtree(stage, ignore_errors=True)
    os.makedirs(stage)
    subprocess.run("git -C %s archive --format=tar %s | tar -x -C %s" % (repo, commit, stage), shell=True, check=True)
    files = list(DATA_FILES) + [os.path.relpath(p, MAIN) for g in DATA_GLOBS for p in sorted(glob.glob(os.path.join(MAIN, g)))]
    for rel in files:
        os.makedirs(os.path.dirname(os.path.join(stage, rel)), exist_ok=True)
        shutil.copy2(os.path.join(MAIN, rel), os.path.join(stage, rel))
    for rel in CSVS:
        with open(os.path.join(MAIN, rel), newline="") as fi, open(os.path.join(stage, rel), "w", newline="") as fo:
            w = csv.writer(fo, delimiter=";")
            w.writerow(CSV_KEEP)
            for row in csv.DictReader(fi, delimiter=";"):
                w.writerow([row[k] for k in CSV_KEEP])
    shutil.copy2(os.path.join(MAIN, "research/gpuwork/colab/pip_extras.txt"), os.path.join(stage, "pip_extras.txt"))
    json.dump(dict(repo=repo, ref=ref, commit=commit), open(os.path.join(stage, "COLAB_BUNDLE.json"), "w"))
    with tarfile.open(out, "w:gz") as tf:
        tf.add(stage, arcname=".")
    shutil.rmtree(stage)
    return out, commit


def push(name, local, remote):
    parts = local + ".parts"
    shutil.rmtree(parts, ignore_errors=True)
    os.makedirs(parts)
    subprocess.run(["split", "-b", CHUNK, "-d", "-a", "3", local, os.path.join(parts, "p")], check=True)
    names = sorted(os.listdir(parts))
    remote_sh(name, "rm -rf /content/_up && mkdir -p /content/_up")
    for n in names:
        colab(["upload", "-s", name, os.path.join(parts, n), "/content/_up/" + n], timeout=300)
    out = remote_sh(name, "cat /content/_up/p* > %s && rm -rf /content/_up && sha256sum %s" % (remote, remote), timeout=300)
    shutil.rmtree(parts)
    want = sha256(local)
    if want not in out:
        sys.exit("upload checksum mismatch for %s: %s" % (local, out[-300:]))
    print("pushed %s -> %s (%d chunks, sha256 %s)" % (local, remote, len(names), want[:12]))


SETUP = r'''
set -x
mkdir -p /content/jk /content/jobs && tar -xzf /content/bundle.tgz -C /content/jk
nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; free -g | head -2
command -v pixi || curl -fsSL https://pixi.sh/install.sh | bash
export PATH=$HOME/.pixi/bin:$PATH
command -v nvidia-smi >/dev/null || export CONDA_OVERRIDE_CUDA=12.4  # CPU runtime: let the cuda-locked env resolve (runkit probe; untested)
mkdir -p $HOME/.pixi && cat > $HOME/.pixi/config.toml <<'PIXICFG'
[mirrors]
"https://conda.anaconda.org/conda-forge" = ["https://prefix.dev/conda-forge", "https://conda.anaconda.org/conda-forge"]
"https://conda.anaconda.org/bioconda" = ["https://prefix.dev/bioconda", "https://conda.anaconda.org/bioconda"]
PIXICFG
export PIXI_NO_TELEMETRY=1
# The lock pins every file by sha256; bound stalled HTTPS reads and retry using the package cache.
cd /content/jk && for i in 1 2 3; do timeout 600 pixi install -e linux --frozen && break; echo "CJ pixi install attempt $i failed or stalled (rc $?)"; done
.pixi/envs/linux/bin/python -m pip install --no-deps -r pip_extras.txt
git config --global --add safe.directory /content/jk
git init -q && git -c user.email=colab@local -c user.name=colab add -A src refit research tools tests pixi.toml pixi.lock >/dev/null && git -c user.email=colab@local -c user.name=colab commit -qm colab-snapshot
.pixi/envs/linux/bin/xtb --version | grep -i version
JACOBIK_PROFILE=u1 JACOBIK_PROFILE_OVERRIDES='{"nci": false, "deterministic_embed": true}' PYTHONPATH=/content/jk/src .pixi/envs/linux/bin/python -c "import torch, mace; from jacobik.core import profile as P; s = P.snapshot(); print('torch', torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'); print('FINGERPRINT', s['fingerprint'])"
echo SETUP_DONE
'''


def cmd_up(a):
    bal, active = guard_credits()
    if active >= MAX_VMS:
        sys.exit("%d Colab VMs already active (limit %d, shared by all agents); stop one first: %s"
                 % (active, MAX_VMS, balance()[2]))
    bundle, commit = build_bundle(os.path.abspath(a.repo), a.ref)
    print("bundle", bundle, "commit", commit[:12])
    colab(["new", "-s", a.name] + ([] if a.gpu in ("none", "cpu") else ["--gpu", a.gpu]), timeout=600, retries=2)  # --gpu none: CPU runtime (runkit probe)
    push(a.name, bundle, "/content/bundle.tgz")
    remote_py(a.name, "open('/content/setup.sh','w').write(%r)\nimport subprocess\nsubprocess.Popen('nohup bash /content/setup.sh > /content/setup.log 2>&1 &', shell=True)\nprint('setup started')\n" % SETUP)
    # Frozen linux environments are large; allow slow first-time Colab disk extraction to finish.
    log = wait_marker(a.name, "/content/setup.log", "SETUP_DONE", max_min=100)
    fp = re.search(r"FINGERPRINT (\w+)", log)
    print(log[-1200:])
    if not fp or fp.group(1) != FINGERPRINT:
        sys.exit("fingerprint check FAILED (want %s); VM left up for inspection, stop it with `down`" % FINGERPRINT)
    json.dump(dict(name=a.name, gpu=a.gpu, commit=commit, up_at=time.time()), open(os.path.join(WORK, "session_%s.json" % a.name), "w"))
    print("READY: %s on %s, commit %s, fingerprint %s" % (a.name, a.gpu, commit[:12], fp.group(1)))


def wait_marker(name, path, marker, max_min):
    t0 = time.time()
    while True:
        out = colab(["exec", "-s", name], timeout=90, stdin="print(open(%r).read()[-4000:])\n" % path, check=False) or ""
        if marker in out:
            return out
        if time.time() - t0 > max_min * 60:
            return out + "\n[cj] wait limit %d min reached; job still running" % max_min
        time.sleep(30)


def cmd_run(a):
    guard_credits()
    env = dict(REMOTE_ENV)
    for kv in a.env or []:
        k, v = kv.split("=", 1)
        env[k] = v
    cmd = " ".join(a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd)
    exports = " ".join("export %s=%s;" % (k, json.dumps(v)) for k, v in env.items())
    job = "cd /content/jk; export PATH=/content/jk/.pixi/envs/linux/bin:$PATH; %s (%s) > /content/jobs/%s.log 2>&1; echo \"CJ_EXIT $?\" >> /content/jobs/%s.log" % (exports, cmd, a.tag, a.tag)
    remote_py(a.name, "import subprocess\nsubprocess.Popen(%r, shell=True)\nprint('launched %s')\n" % (job, a.tag))
    print("launched", a.tag)


def cmd_wait(a):
    out = wait_marker(a.name, "/content/jobs/%s.log" % a.tag, "CJ_EXIT", a.max_min)
    print(out[-3000:])
    m = re.search(r"CJ_EXIT (\d+)", out)
    bal, _, _ = balance()
    print("[cj] balance %.2f (spent %.2f of cap %.0f)" % (bal, START_BALANCE - bal, CAP_UNITS))
    sys.exit(0 if m and m.group(1) == "0" else 1)


def cmd_tail(a):
    print(colab(["exec", "-s", a.name], timeout=90, stdin="print(open('/content/jobs/%s.log').read()[-%d:])\n" % (a.tag, a.bytes)))


def cmd_get(a):
    os.makedirs(a.local_dir, exist_ok=True)
    base = os.path.basename(a.remote.rstrip("/"))
    remote_tgz = "/content/_get_%s.tgz" % base
    remote_sh(a.name, "cd %s && tar -czf %s %s && ls -la %s" % (os.path.dirname(a.remote.rstrip("/")) or "/", remote_tgz, base, remote_tgz), timeout=600)
    local_tgz = os.path.join(a.local_dir, "_get_%s.tgz" % base)
    colab(["download", "-s", a.name, remote_tgz, local_tgz], timeout=900)
    with tarfile.open(local_tgz) as tf:
        tf.extractall(a.local_dir, filter="data")
    os.remove(local_tgz)
    print("fetched %s -> %s" % (a.remote, os.path.join(a.local_dir, base)))


def cmd_down(a):
    colab(["stop", "-s", a.name], timeout=180)
    print(balance()[2])


def cmd_credits(a):
    bal, active, out = balance()
    print(out.strip())
    print("spent since START_BALANCE: %.2f of cap %.0f" % (START_BALANCE - bal, CAP_UNITS))


def main():
    os.makedirs(os.path.join(WORK, "bundles"), exist_ok=True)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("up"); s.add_argument("--name", required=True); s.add_argument("--gpu", default="L4")
    s.add_argument("--repo", required=True); s.add_argument("--ref", default="HEAD"); s.set_defaults(f=cmd_up)
    s = sp.add_parser("push"); s.add_argument("--name", required=True); s.add_argument("local"); s.add_argument("remote")
    s.set_defaults(f=lambda a: push(a.name, a.local, a.remote))
    s = sp.add_parser("run"); s.add_argument("--name", required=True); s.add_argument("--tag", required=True)
    s.add_argument("--env", action="append"); s.add_argument("cmd", nargs=argparse.REMAINDER); s.set_defaults(f=cmd_run)
    s = sp.add_parser("wait"); s.add_argument("--name", required=True); s.add_argument("--tag", required=True)
    s.add_argument("--max-min", type=float, default=60); s.set_defaults(f=cmd_wait)
    s = sp.add_parser("tail"); s.add_argument("--name", required=True); s.add_argument("--tag", required=True)
    s.add_argument("--bytes", type=int, default=3000); s.set_defaults(f=cmd_tail)
    s = sp.add_parser("get"); s.add_argument("--name", required=True); s.add_argument("remote"); s.add_argument("local_dir")
    s.set_defaults(f=cmd_get)
    s = sp.add_parser("down"); s.add_argument("--name", required=True); s.set_defaults(f=cmd_down)
    s = sp.add_parser("credits"); s.set_defaults(f=cmd_credits)
    a = p.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
