"""Render the final C-2 report from committed measurement/analysis JSON."""
from __future__ import annotations
import json
from pathlib import Path

BASE=Path('research/campaigns')


def num(x,scale=1):return '—' if x is None else f'{x*scale:.2f}'


def estimate(d,scale=1):
    if d.get('mean') is None:return '—'
    ci=d.get('ci')
    return num(d['mean'],scale)+(f" [{num(ci[0],scale)}, {num(ci[1],scale)}]" if ci else '')


def main():
    p0=json.loads((BASE/'c2/C2_PHASE0.json').read_text())
    p1=json.loads((BASE/'c2/C2_PHASE1.json').read_text())
    p2=json.loads((BASE/'c2/C2_PHASE2.json').read_text())
    placement=p0['measurements']['placement'];decision=placement['placement_decision']
    rings=p0['measurements']['rings'];tiny=p0['measurements']['tiny_pool']['records']
    arms=p2['arms'];selected=[x['arm'] for x in arms if x['selected']]
    lines=[f"# C-2 coverage report — {p2.get('analysis_date','2026-11-03')}",'',p2['status_text'],'',
        '## Provenance','',
        'Worktree `jgen-c`, branch `jgen/c-coverage`, base `b0134a2`. Production fingerprint remains `5f74b14b57ed`. '
        'Screen producer revision `b9ce924`; subsequent revisions correct measurement scope and analysis/accounting without changing the running arm algorithms. '
        'All gates use unrelaxed MACE energies and crystal geometry. No post-generation relaxation was performed.','',
        'Population: 120 unique cal/tail parents after the locked exclusion list. The five validation-derived tail codes were excluded before loading inputs. '
        'Primary statistics exclude FEKKAJ and NEPNEF; both remain in tables and the failure ledger. Seeds 11–13. '
        'Every arm and its control ran in the same invocation on the local RTX 2060 SUPER, NVIDIA driver 595.99.02 / CUDA driver API 13.2. '
        f"The local environment reports PyTorch {p2['platform'].get('torch_version','unknown')} / CUDA runtime {p2['platform'].get('torch_cuda_runtime','unknown')}.",'',
        'The registered contract and settings precede arm runs. Bootstrap: 10,000 parent-cluster resamples, seed 11; seeds are averaged within parent before resampling. '
        'Undefined energies/correlations are not imputed; paired sample counts and failures are reported. Coverage failures stay in the denominator.','',
        f"Measured screen wall, including MACE/export/startup: {num(p2.get('screen_wall_s'),1/60)} minutes. "
        f"Colab spend so far: {num(p2.get('units_spent',0))} units. No production defaults, protected DBs or goldens were changed.",'',
        '| Arm | Flags / budget | Fingerprint |','|---|---|---|',
        *[f"| {a} | {p2['arm_identity'][a]['definition']} | `{p2['arm_identity'][a]['fingerprint']}` |" for a in ['C','K','U','T','S1']],'',
        '## Phase 0 placement','',
        f"{placement['n_success']}/120 successful parents; {placement['n_initial_rows']:,} initial rows across all valid native E/Z variants. "
        f"{decision['parents_above_03']}/120 parents ({num(decision['fraction'],100)}%) have median row-maximum offset >0.3 Å. "
        f"Pooled median row maximum: {num(decision['original_median'])} → {num(decision['fitted_median'])} Å "
        f"({num(decision['reduction'],100)}% reduction). Build start_kabsch by the parent-fraction condition; the alternative 30% reduction condition fails.",'',
        'Per-parent/ligand-class/overall donor-offset, M–D-distance and M–D-error quantiles are in C2_PHASE0.json. '
        'K retains orientation diversification, pairing the Kabsch random draws in the diagnostic. Its actual rerun matches the paired-fit median. '
        'Placement counts all candidate initialization slots before class-budget selection; the tiny-pool trace separately reports actual class-budget rows.','',
        'The initial helper captured only the first native E/Z variant. A scope audit identified nine multi-variant parents, '
        'which were remeasured in C and K. There are 133 native parent/variant configurations, 132 valid (NEPNEF fails GFN2). '
        'The corrected build rule remains 101/120. Original diagnostic records are retained. The Phase 2 runner used native enumeration from its first invocation.','',
        '## Phase 0 tiny pools','',
        'Counts are seeds 11/12/13. All seven tiny-pool parents have one native E/Z variant. '
        'All their solved rows are finite and embedding-valid; production has no additional row-level embedding-validity rejection. '
        'Filters: class-budget split, finite scores/geometry, global 200-nat window, best-first symmetric 0.5-Å dedup, and output cap.','',
        '| Parent | Solved | In window | Distinct before window | Distinct in window | Returned | Classification |',
        '|---|---|---|---|---|---|---|']
    for record in tiny[:7]:
        values=lambda key:'/'.join(str(s['counts'][key]) for s in record['seeds'])
        lines.append('| '+record['code']+' | '+' | '.join(values(k) for k in ['solved_B','inside_score_window','distinct_before_window','distinct_inside_window','after_output_cap'])+' | '+record['classification']+' |')
    lines+=['',
        'QAPFAS/MIGTEE/ZAZREZ have nearly symmetry-equivalent sampled starts despite many formal torsions; '
        'this is provisional sampled-space class (a), not proof of a globally small conformational space. '
        'UYOVUA and MIVXAR are (a-prime), with large rings frozen by topology. SIVBUV has diverse starts that collapse in IK: class (c), unresolved. '
        'HOXJUZ seed 12 loses two distinct rows to the window: class (b), motivating pool_window_fill.','',
        'The five preselected normal controls are AJAQAD, AWONOR, AYOGUR, AZUSIA and BIBFUN. '
        'Their full stage counts, score spreads and pairwise RMSD distributions are retained. Pair distributions use all pairs up to 4096, otherwise a seed-11 uniform sample; '
        'float64 reductions avoid cancellation on nearly identical rows.','',
        '## Phase 0 rings','',
        'Internal nonaromatic 5–8 rings dominate on 2/9 (DOHTOM, HOHZAG), below the required four. FEKKAJ is 48.79%, not rounded up. '
        'ring_conf_pool is omitted. Across the population, 27 parents and 33 ligands contain qualifying internal rings. '
        'Chelate domination on UYOVUA alone is below the clarified ≥5/9 stop threshold; the earlier any-parent stop was an auditor specification ambiguity.','',
        '| Parent | Best seed / row | Shell RMSD Å | Internal 5–8 share | Chelate share | Aromatic/other share |',
        '|---|---|---:|---:|---:|---:|']
    for r in rings['per_parent']:
        shares=r['type_shares']
        lines.append(f"| {r['code']} | {r['best_seed']} / {r['best_row']} | {num(r['shell_rmsd'])} | {num(shares.get('internal_5_8',0),100)}% | {num(shares.get('chelate',0),100)}% | {num(shares.get('aromatic_or_other',0),100)}% |")
    lines+=['',
        'Ring-type shares overlap and are nonadditive. CIXYUD/DUHHIX symmetry caps are retained. '
        'UYOVUA’s 13-ring is outside the macrocycle path; production FK confirms two base torsions and no macrocycle joints. '
        'Three ligands have nonaromatic ≥9 rings outside that path: LOPCAW (16), MIVXAR (16), UYOVUA (13). '
        'Each has four input donors and fails the all-single-bond test. Actual path status comes from production decomposition, including its small-donor-ring fallback. '
        'Eligibility and rigidity counts use the crystal-input topology; the separate UYOVUA check uses actual production FK. No topology change was made.','',
        'Aromatic rings with share >20% use the stored A4b pool atom map, the selected row’s embed variant for attachment identification, '
        'and physical initial conformers across all native variants in that seed’s pool. The initial free-embed mapping/first-variant calculation was corrected before opening Phase 2 comparisons.','',
        '| Parent | Ring share | Ligand / joint | Row−crystal rad | Closest initial seed rad | ≤0.2 rad? |',
        '|---|---:|---|---:|---:|---|']
    for parent in placement['aromatic']:
        for ring in parent['rings']:
            if not ring['attachments']:lines.append(f"| {parent['code']} | {num(ring['share'],100)}% | none | — | — | no attachment DOF |")
            for j in ring['attachments']:
                lines.append(f"| {parent['code']} | {num(ring['share'],100)}% | {j['ligand']} / {j['joint']} | {num(j['difference_rad'])} | {num(j['closest_seed_rad'])} | {j['within_02']} |")
    lines+=['','## Flags, runner and validation','',
        '- K: start_kabsch fits each row after torsion/bend seeding, preserving orientation diversification.','- U: beam_union ports the C-1 target-aware beam, rebuilt per seed/candidate/sample; unions it with the free-ligand beam. '
        'Target share 0.5, wrapped duplicate tolerance 1e-6 rad, group batch cap 16; all registered and read at import. '
        'Duplicates advance within the same ranked beam; remaining unique seeds precede unavoidable padding.',
        '- T: pool_window_fill retains finite distinct rows beyond the score window up to the existing cap. Ranking, top-1 and solved-row budget are unchanged. '
        'This addresses window loss, not SIVBUV collapse.','- S1: runner seed budget single_w48, with W/seed_budget and row-steps in every record. '
        'Each independent W48 seed is compared with C’s W16 seed-11–13 union. Three wide seeds provide noise replicates.','',
        'All flags default off; default fingerprint stays 5f74b14b57ed. Three-parent --cpu-identity checks match exported coordinates, scores and MACE energies; '
        'the full-precision NPZ supplement matches coordinates/scores bit-for-bit against b0134a2. Nine focused tests pass. '
        'HOXJUZ seed 11 full-200-step replay/eager fast FK, replay/eager reference FK and CPU package-loop parity pass with zero score differences. '
        'No replayed source path changed, so solver_replay.py is unchanged.','',
        'Full CPU suite: 374 passed, 85 skipped, 3 xfailed, 2 missing-worktree-input failures. The multiseed test passes with the shared-DB override; '
        'the unrelated reader-path failure remains recorded. CUDA tests were excluded from that concurrent suite to preserve one GPU job.','',
        '**Golden check FAILED:** 8/245 arrays in each production/u1 profile differ under one-worker CPU execution; all 231 rescore arrays match. '
        'The unchanged b0134a2 engine reproduces the same eight solve-array differences (OJIMIF, CBTYCO, SOKVET, YAVFEG) under that CPU layout. '
        'No reference was changed.','',
        '## Phase 2 results','',
        'Values are primary-population means over parent seed means; differences are arm minus C, with parent-clustered 95% CIs. '
        'Coverage differences are percentage points. Spearman is score vs energy (ideal correlation is negative); undefined values remain null. '
        'For S1, C means refer to the three-seed union.','']
    labels=[('floor','Floor ΔE kJ/mol',1),('top1','Top-1 ΔE kJ/mol',1),('min_rmsd','Minimum heavy RMSD Å',1),
        ('hit1','Coverage ≤1 Å',100),('hit05','Coverage ≤0.5 Å',100),('min_shell','Minimum shell RMSD Å',1),
        ('shell03','Shell ≤0.3 Å',100),('shell05','Shell ≤0.5 Å',100),('distinct','Distinct rows',1),('rho','Spearman ρ',1)]
    for arm in arms:
        lines += [f"### {arm['arm']}",'','| Metric | C | Arm | Paired difference [95% CI] | N parents |','|---|---:|---:|---:|---:|']
        for key,label,scale in labels:
            m=arm['metrics'][key]
            lines.append(f"| {label} | {num(m['control_mean'],scale)} | {num(m['arm_mean'],scale)} | {estimate(m,scale)} | {m['n']} |")
        g=arm['guard'];ci=arm['wall_ci']
        lines += ['',f"Wall ratio: {num(arm['wall_ratio'])}x"+(f" [{num(ci[0])}, {num(ci[1])}]" if ci else '')+'. '
            +f"Guard: trips {g['trips']}, mirror {g['mirror']}, arm rotation null {g['null']}, C rotation null {g['control_null']}. "
            +f"Equal-compute prerequisite: {arm['equal_compute_passed']}. Failures: {', '.join(arm['failures']) or 'none'}.",'']
        cases=p2.get('regression_cases',{}).get(arm['arm'],{})
        if cases:lines += [f"Trip parents: {', '.join(cases['trips']) or 'none'}. Mirror parents: {', '.join(cases['mirror']) or 'none'}.",'']
    lines+=['| Arm | Floor/coverage | Top-1 | Guard | Wall | Selected |','|---|---|---|---|---|---|']
    for arm in arms:
        gates=arm['gates'];status=lambda b:'PASS' if b else 'FAIL'
        lines.append('| '+arm['arm']+' | '+' | '.join(status(gates[k]) for k in ['primary','top1','guard','wall'])+' | '+str(arm['selected'])+' |')
    lines+=['',f"Selection outcome: {', '.join(selected) or 'no qualifying arm'}.",'',
        'End-to-end wall includes prepare, seed generation, solve and assembly, excluding MACE/export. '
        'Shared preparation is charged equally. Every independent S1 seed pays full preparation, including native E/Z preparation; only the three-replicate seed/solve total is divided by three. '
        'Raw timing/row-step ledgers are preserved; finalise.py reconstructs metadata after both record writers finish. '
        'The three-parent pilot projected 93 minutes including MACE/export; the actual 190.68 minutes was about twice that estimate. '
        'Large-pool/symmetry-heavy parents and the S1 resource retries were not represented adequately by the pilot.','',
        '## Arm-specific measurements','',p2.get('arm_specific_text','See machine-readable records.'),'',
        '## Named cases (Phase 2)','',
        'Each cell is top-1 / floor unrelaxed ΔE in kJ/mol. Phase 3 was not run; these are local-screen measurements.','',
        '| Parent | Seed | C | K | U | T | S1 |','|---|---:|---:|---:|---:|---:|---:|']
    for code,seeds in p2.get('named_cases',{}).items():
        for seed,records in seeds.items():
            cells=[]
            for arm in ['C','K','U','T','S1']:
                r=records[arm];cells.append(num(r.get('top1_dE'))+' / '+num(r.get('floor_dE')) if r['status']=='ok' else 'failed')
            lines.append(f"| {code} | {seed} | "+' | '.join(cells)+' |')
    lines += ['',
        '## Failures and confirmation','',p2.get('failure_text','See the machine-readable failure ledger.'),'',
        p2.get('confirmation_text','No confirmation results yet.'),'',
        'Machine-readable evidence: c2/C2_PHASE0.json, C2_PHASE1.json, C2_PHASE2.json. '
        'Raw pools, timing ledgers, initialization arrays, failure logs and per-parent results are in shared research/data/jgen/campaign/C2/.','']
    (BASE/'C2_REPORT.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
