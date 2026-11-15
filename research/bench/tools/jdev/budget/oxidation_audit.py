"""Compare query-time oxidation deduction with the extractor's labels, without solving.

Dev mode uses the exact benchmark ligand inputs. Training mode streams cleaned
training structures and uses the same ligand mapper; --limit allows a measured
sample without re-extracting observations. Every unmappable input is counted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import zlib

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO
from pools import set_cases, setup


def training_stream(parser, shard, limit):
    """Stable hash sample per shard, selected before costly geometry assembly."""
    if not limit:
        yield from parser.stream_split(shard)
        return
    from fit.parser import _iter_xyz_blocks, _iter_bo_blocks, _assemble_raw, _attach_qm_properties
    from jacobik.topology.mapping import assemble
    xyz = parser.data_dir / 'xyz' / f'tmQM_{shard}.xyz'
    bo = parser.data_dir / 'bo' / f'tmQM_{shard}.BO'
    codes = [comment.split('CSD_code = ', 1)[1].split(' |', 1)[0] for comment, _ in _iter_xyz_blocks(xyz)]
    chosen = set(sorted(codes, key=lambda c: zlib.crc32(f'oxidation-audit-2026-10-18:{c}'.encode()))[:limit])
    for (comment, atoms), (code, bonds) in zip(_iter_xyz_blocks(xyz), _iter_bo_blocks(bo)):
        if code in chosen:
            csv_row = parser.csv_index.get(code)
            mol = assemble(_assemble_raw(comment, atoms, code, bonds, csv_row))
            if csv_row is not None:
                _attach_qm_properties(mol, csv_row)
            yield mol


def row(mol, ligands, set_name, models=None):
    from jacobik.core.chemistry import deduce_oxidation_state, ligand_ionic_charge
    os_, confidence, notes = deduce_oxidation_state(mol.metal_symbol, mol.charge, ligands)
    result = dict(set=set_name, code=mol.csd_code, metal=mol.metal_symbol, charge=mol.charge,
                label=mol.oxidation_state, deduced=os_, confidence=confidence, notes=notes,
                smiles=mol.smiles, ligands=[dict(smiles=l.smiles, donors=l.donors,
                ionic_charge=ligand_ionic_charge(l.smiles, l.donors)) for l in ligands])
    if models:
        result['deductions'] = {model: deduce_oxidation_state(mol.metal_symbol, mol.charge, ligands, model=model)
                                for model in models}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', action='store_true')
    p.add_argument('--limit', type=int, default=0, help='hash-sampled structures per training shard; 0 = all')
    p.add_argument('--shards', nargs='*')
    p.add_argument('--models', nargs='+', choices=['legacy', 'training'])
    p.add_argument('--out', required=True)
    a = p.parse_args()
    setup()
    from tools.jdev.cases import get_case
    rows, skipped = [], Counter()
    if a.training:
        from fit.parser import TMQMParser
        from jacobik.builder.ligands import _build_ligand_groups, ligand_inputs_from_mol
        root = REPO / 'data/refit/clean/tmQM'
        holdout = json.loads((REPO / 'src/fit/data/benchmark_holdout_codes.json').read_text())
        excluded = {c for k, codes in holdout.items() if not k.startswith('_') for c in codes}
        shards = a.shards or [f.stem.removeprefix('tmQM_') for f in sorted((root / 'xyz').glob('*.xyz'))]
        parser = TMQMParser(root)
        for shard in shards:
            for m in training_stream(parser, shard, a.limit):
                if m.csd_code in excluded:
                    skipped['benchmark_excluded'] += 1
                    continue
                if m.oxidation_state is None:
                    skipped['unknown_training_label'] += 1
                    continue
                try:
                    ligs = ligand_inputs_from_mol(m, _build_ligand_groups(m, m.smiles))
                    if not ligs:
                        skipped['unmappable_ligands'] += 1
                        continue
                    rows.append(row(m, ligs, shard, a.models))
                except Exception as exc:
                    skipped[type(exc).__name__] += 1
            print(f'{shard}: {len(rows)} mapped, skips {dict(skipped)}', flush=True)
    else:
        for name in ('dev32', 'n48'):
            for code, spec in set_cases(name):
                c = get_case(spec)
                rows.append(row(c.tmqm_mol, c.ligand_inputs, name, a.models))
    known = [r for r in rows if r['label'] is not None]
    bad = [r for r in known if r['label'] != r['deduced']]
    summary = dict(total=len(rows), known_labels=len(known), matching=len(known)-len(bad),
                   mismatching=len(bad), skipped=dict(skipped))
    if a.models:
        summary['by_model'] = {model: dict(matching=sum(r['deductions'][model][0] == r['label'] for r in known),
                                          mismatching=sum(r['deductions'][model][0] != r['label'] for r in known))
                               for model in a.models}
        if len(a.models) == 2:
            before, after = a.models
            summary['fixed'] = sum(r['deductions'][before][0] != r['label'] and
                                   r['deductions'][after][0] == r['label'] for r in known)
            summary['regressed'] = sum(r['deductions'][before][0] == r['label'] and
                                       r['deductions'][after][0] != r['label'] for r in known)
    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(summary=summary, rows=rows), indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)
    for r in bad[:20]:
        print(f"{r['code']} {r['metal']} label={r['label']} deduced={r['deduced']} {r['ligands']}", flush=True)


if __name__ == '__main__':
    main()
