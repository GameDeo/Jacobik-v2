"""Isolated, content-addressed Step 3–4 development artifacts (no production writes)."""
from __future__ import annotations
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / 'data_dev'
MAIN = Path('/mnt/e/Dev/Projects/jacobik_2026-10-02_full')


def cpu_only():
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    import torch
    torch.set_num_threads(1)


def plain(x):
    import numpy as np
    import torch
    if dataclasses.is_dataclass(x):
        return plain(dataclasses.asdict(x))
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    return x


def canonical(x):
    return json.dumps(plain(x), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    if not path.resolve().is_relative_to(DEV.resolve()):
        raise ValueError('Step 3–4 outputs must stay under data_dev/')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_bytes(canonical(obj) + b'\n')
    temp.replace(path)


def write_native_json(path, obj):
    """Stream an already-JSON-native DB without recursively copying its arrays."""
    path = Path(path)
    if not path.resolve().is_relative_to(DEV.resolve()):
        raise ValueError('Step 3–4 outputs must stay under data_dev/')
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    encoder = json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False)
    with temp.open('w') as f:
        for part in encoder.iterencode(obj):
            f.write(part)
        f.write('\n')
    temp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def code_hash():
    h = hashlib.sha256()
    for folder in ('src', 'refit', 'tests', 'tools'):
        for p in sorted((ROOT / folder).rglob('*.py')):
            h.update(str(p.relative_to(ROOT)).encode() + b'\0')
            h.update(p.read_bytes())
    return h.hexdigest()


class Run:
    """Completed directories are immutable; resume checks inputs, code AND output hashes."""
    def __init__(self, stage, spec, inputs=(), options=None, out=None):
        cpu_only()
        self.started = time.perf_counter()
        import numpy, scipy, torch
        self.identity = dict(stage=stage, spec_sha256=digest(spec), code_sha256=code_hash(),
                             inputs={str(Path(p).resolve()): digest(p) for p in sorted(set(map(Path, inputs)))},
                             options=options or {}, environment=dict(python=platform.python_version(),
                             numpy=numpy.__version__, scipy=scipy.__version__, torch=torch.__version__,
                             device='cpu', threads=1))
        self.id = hashlib.sha256(canonical(self.identity)).hexdigest()[:20]
        self.path = Path(out) if out else DEV / f'{stage}_{self.id}'
        if not self.path.resolve().is_relative_to(DEV.resolve()):
            raise ValueError('output escapes data_dev')
        self.resumed = False
        mp = self.path / 'manifest.json'
        if mp.exists():
            m = load_json(mp)
            if m['identity'] != plain(self.identity):
                raise ValueError('resume provenance mismatch')
            if m['status'] == 'complete':
                for rel, sha in m['outputs'].items():
                    if digest(self.path / rel) != sha:
                        raise ValueError(f'resume output checksum mismatch: {rel}')
                self.resumed = True
        elif self.path.exists() and any(self.path.iterdir()):
            raise ValueError('refusing unrecognized nonempty run directory')
        if not self.resumed:
            self.path.mkdir(parents=True, exist_ok=True)
            write_json(mp, dict(identity=self.identity, status='running'))
            write_json(self.path / 'spec.json', load_json(spec))

    def finish(self, units, limitations=()):
        if code_hash() != self.identity['code_sha256']:
            raise ValueError('source changed while stage was running; outputs remain incomplete')
        for path, expected in self.identity['inputs'].items():
            if digest(path) != expected:
                raise ValueError(f'input changed while stage was running: {path}')
        outputs = {str(p.relative_to(self.path)): digest(p) for p in sorted(self.path.rglob('*'))
                   if p.is_file() and p.name != 'manifest.json'}
        report = dict(id=self.id, identity=self.identity, status='complete', outputs=outputs,
                      wall_s=time.perf_counter()-self.started,
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      work_units=units, limitations=list(limitations), production_ready=False)
        write_json(self.path / 'manifest.json', report)
        write_json(DEV / 'records' / f'{self.identity["stage"]}_{self.id}.json',
                   dict(id=self.id, status='completed', mode='motion_scan' if 'response' in self.identity['stage'] else 'audit',
                        question='Step3–4 development harness only', cohort_manifest=self.identity['inputs'],
                        generation=self.identity, artifacts=outputs, metrics=report,
                        conclusion='Development evidence; requires audit', limitations=list(limitations)))
        return report


def load_surfaces(path):
    from research.archive.code.interactions.nci_v2.nci_v2 import SurfaceArtifact
    raw = load_json(path)
    return {fam: SurfaceArtifact(**a) for fam, a in raw.items()}
