import json
import numpy as np

from tools.jgen.a4t import relax_run


def test_a4t_resume_reuses_energy_and_retries_failed(tmp_path, monkeypatch):
    data, prev, out = (tmp_path / n for n in ('data', 'prev', 'out'))
    for p in (data / 'xyz' / 'CASE', data / 'records', data / 'relax', prev):
        p.mkdir(parents=True)
    def put(path, value):
        path.write_text(json.dumps(value))
    xyz = [[0., 0., 0.]]
    manifest = {'n_rows': 3, 'rows': [{'file': 'pool.xyz'}] * 3,
                'ref_symbols': ['Fe'], 'gen_to_ref_idx': [0], 'ref_X': xyz}
    put(data / 'xyz' / 'CASE' / 'CASE_control_s11_pool.json', manifest)
    put(data / 'records' / 'CASE_11.json', {'status': 'ok', 'fingerprint': relax_run.A4_FINGERPRINT,
                                            'energies_kj': [0, 1, 2]})
    put(data / 'charge_mult.json', {'CASE': {'charge': 0, 'mult': 1}})
    put(data / 'relax' / 'references.json', {})
    good = {'status': 'ok', 'X': xyz, 'E_mace_kj': 7.5}
    put(prev / 'CASE_11.json', {'code': 'CASE', 'seed': 11, 'rows': {'0': good, '1': {'status': 'failed'}}})
    ref = {'X': xyz, 'E_ref0_mace_kj': 1., 'E_refrel_mace_kj': 2.}
    put(prev / 'references.json', {'CASE': ref})
    monkeypatch.setattr(relax_run, '_read_frames', lambda _: [({'row_index': i}, ['Fe'], np.array(xyz)) for i in range(3)])
    monkeypatch.setattr(relax_run.xtb, 'require', lambda: None)
    calls = []
    def opt(jobs, **kwargs):
        calls.extend(jobs)
        return [{'status': 'ok', 'X': np.array(xyz), 'converged': True, 'cycles': 5}] * len(jobs)
    monkeypatch.setattr(relax_run.xtb, 'opt_many', opt)
    class Evaluator:
        stats = {}
        def energies_kj(self, structures):
            assert len(structures) == 2
            return [3., 4.]
    metrics = relax_run.run(data, out, ['CASE'], seeds=(11,), resume=prev, evaluator=Evaluator())
    rows = json.loads((out / 'CASE_11.json').read_text())['rows']
    assert len(calls) == 2 and metrics['n_rows_resumed'] == 1
    assert rows['0']['E_mace_kj'] == good['E_mace_kj'] and rows['0']['X'] == good['X']
    assert json.loads((out / 'references.json').read_text())['CASE'] == ref


def test_a4b_resume_reuses_energy_and_retries_failed(tmp_path, monkeypatch):
    from tools.jgen.a4b import relax_run as b
    data, prev, out = (tmp_path / n for n in ('data', 'prev', 'out'))
    for p in (data / 'xyz' / 'CASE', data / 'records', prev):
        p.mkdir(parents=True)
    def put(path, value):
        path.write_text(json.dumps(value))
    xyz = [[0., 0., 0.]]
    manifest = {'code': 'CASE', 'seed': 11, 'rows': [{'file': 'pool.xyz'}] * 3,
                'ref_symbols': ['Fe'], 'gen_to_ref_idx': [0], 'ref_X': xyz}
    mp = data / 'xyz' / 'CASE' / 'CASE_control_s11_pool.json'
    put(mp, manifest); put(data / 'records' / 'CASE_11.json', {})
    good = {'status': 'ok', 'X': xyz, 'E_mace_kj': 7.5, 'cycles': 5, 'X0_sha': b._x0_sha(xyz)}
    other = dict(good, X0_sha=b._x0_sha([[1., 0., 0.]]))   # same (code, seed, row), another unrelaxed structure
    put(prev / 'CASE_11.json', {'code': 'CASE', 'seed': 11, 'rows': {'0': good, '1': {'status': 'failed'}, '2': other}})
    put(prev / 'references.json', {'CASE': {'status': 'ok', 'X': xyz, 'E_mace_kj': 2., 'E_unrel_mace_kj': 1.}})
    monkeypatch.setattr(b, 'select', lambda *_: [0, 1, 2])
    monkeypatch.setattr(b, '_read_frames', lambda _: [({}, ['Fe'], np.array(xyz)) for _ in range(3)])
    monkeypatch.setattr(b, 'reference_context', lambda *_: (None, None, None, None))
    monkeypatch.setattr(b.xtb, 'require', lambda: None)
    monkeypatch.setattr(b.geom, 'sym_rmsd_to_ref', lambda *a, **k: [0.])
    monkeypatch.setattr(b.geom, 'find_metal', lambda *_: 0)
    monkeypatch.setattr(b.geom, 'rcov', lambda *_: 1.)
    calls = []
    def opt(jobs, **kwargs):
        calls.extend(jobs)
        return [{'status': 'ok', 'X': np.array(xyz), 'converged': True, 'cycles': 5}] * len(jobs)
    monkeypatch.setattr(b.xtb, 'opt_many', opt)
    class Evaluator:
        stats = {}
        def energies_kj(self, structures):
            assert len(structures) == 2
            return [3., 4.]
    metrics = b.process([mp], out, {'CASE': {'charge': 0, 'mult': 1}}, resume=prev, evaluator=Evaluator())
    rows = json.loads((out / 'CASE_11.json').read_text())['rows']
    assert len(calls) == 2 and metrics['n_reused_rows'] == 1 and metrics['n_reused_refs'] == 1
    assert rows['0']['E_mace_kj'] == good['E_mace_kj'] and rows['0']['X'] == good['X']
