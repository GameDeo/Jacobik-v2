"""Frozen search identity and fail-closed D2 provenance validation."""
import hashlib
import json
import subprocess
from pathlib import Path

CONFIG = Path('research/campaigns/D3_ARMS.json')
SEARCH = {'start_kabsch': True, 'start_kabsch_pivot': True}
PRODUCTION = 'a47283387172'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def fk_prior_settings(flags):
    """Effective solve defaults and profile controls; FK prior is out of curvature scope."""
    import inspect
    from jacobik.builder.engine import ComplexBuilder
    from jacobik.core import profile as P
    defaults = inspect.signature(ComplexBuilder.solve).parameters
    settings = {k: defaults[k].default for k in
                ('bend_penalty', 'bend_scale', 'bend_sigma_deg', 'bend_weight')}
    settings['bend_free_deg'] = flags.get('bend_free_deg', P.FIELDS['bend_free_deg'][2])
    settings['profile'] = {k: v for k, v in P.PRODUCTION_CONFIG.items()
                           if 'bend' in k or k.startswith('haptic_')}
    return settings


def verify_fk_priors(identities):
    settings = [i.get('fk_prior_settings') for i in identities.values()]
    if not settings or any(not s or s != settings[0] for s in settings):
        raise ValueError('FK bend prior settings missing or differ across arms')


def verify_runtime_identity(arm):
    if arm.get('fk_prior_settings') != fk_prior_settings(arm['flags']):
        raise ValueError('runtime FK bend prior settings mismatch')
    actual = db_identity(arm['db'], False, arm['flags'])
    for key, value in actual.items():
        if arm.get(key) != value:
            raise ValueError('remote DB/sidecar identity mismatch: ' + key)


def db_identity(path, require_build=True, flags=None):
    path = Path(path)
    from jacobik.core import container as dbio
    embedded, tables = dbio.read_container_head(path)
    if embedded is not None:
        # self-describing container (D2 build): meta and tables are inside the file, whose sha256 is the identity
        meta, meta_path = embedded, None
    else:
        meta_path = path.with_suffix('.meta.json')
        meta = json.loads(meta_path.read_text())  # missing sidecar is always fatal
    if require_build:
        build = meta.get('build', meta)
        if not build.get('code_commit') or not isinstance(build.get('flags'), dict) or not build.get('input_hashes'):
            raise ValueError('D2 meta requires code_commit, flags, input_hashes')
        commit = build['code_commit']
        if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
            raise ValueError('D2 code_commit must be a full git SHA')
        if not all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
                   for v in build['input_hashes'].values()):
            raise ValueError('D2 input_hashes must contain SHA256 values')
    result = dict(db_sha256=sha(path), meta_sha256=sha(meta_path) if meta_path else 'embedded')
    if (flags or {}).get('q7_bound_centre'):
        if embedded is not None:
            if 'bends' not in (tables or {}):
                raise ValueError('q7_bound_centre requires the embedded bends table')
            result['bends_sha256'] = 'embedded'
        else:
            bends = path.with_suffix('.bends.json')
            result['bends_sha256'] = sha(bends)  # missing required sidecar is fatal
    return result


def validate(config, dryrun=False):
    from jacobik.core import profile as P
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    model_sha = sha(config['mace_path'])
    result = {}
    for name, spec in config['arms'].items():
        if name in ('D', 'DS') and not dryrun and any(spec.get(k) is None for k in ('db', 'flags', 'fingerprint')):
            raise ValueError('D2 placeholders are unresolved')
        flags = dict(spec['flags'] or {})
        if dryrun and name in ('D', 'DS'):
            flags = SEARCH.copy() if name == 'DS' else {}
        if set(flags)-set(P.FIELDS):raise ValueError('unknown or hardcoded production flag override')
        db = config['arms']['C']['db'] if dryrun else spec['db']
        fp = P.fingerprint({**P.PRODUCTION_CONFIG, **{k: v[2] for k, v in P.FIELDS.items()}, **flags})
        expected = ('ddca4fedde5b' if name in ('S', 'DS') else PRODUCTION) if dryrun else spec['fingerprint']
        if fp != expected:
            raise ValueError(f'{name}: fingerprint {fp} differs from {expected}')
        result[name] = dict(arm=name, flags=flags, db=str(Path(db).resolve()), fingerprint=fp,
            **db_identity(db, require_build=name in ('D', 'DS') and not dryrun, flags=flags),
            fk_prior_settings=fk_prior_settings(flags),
            mace_sha256=model_sha, code_commit=commit, label='DRYRUN' if dryrun else 'D3')
    verify_fk_priors(result)
    expected=config.get('expected_control_identity',{})
    for key,source in [('db_sha256','constraints_db_sha256'),('meta_sha256','constraints_meta_sha256'),('mace_sha256','mace_model_sha256')]:
        if source in expected and result['C'][key]!=expected[source]:raise ValueError('control identity differs from C-2b production: '+key)
    for a, b in [('C', 'S'), ('C', 'Cprime'), ('D', 'DS')]:
        if all(n in result for n in (a, b)):
            for key in ('db_sha256', 'meta_sha256', 'bends_sha256', 'mace_sha256', 'code_commit'):
                if result[a].get(key) != result[b].get(key):
                    raise ValueError(f'{a}/{b} identity mismatch: {key}')
    if result['S']['flags'] != SEARCH or result['C']['flags'] or result['Cprime']['flags']:
        raise ValueError('C/Cprime/S flags differ from registered identities')
    if result['DS']['flags'] != dict(result['D']['flags'], **SEARCH):
        raise ValueError('DS must be D + P')
    return result
