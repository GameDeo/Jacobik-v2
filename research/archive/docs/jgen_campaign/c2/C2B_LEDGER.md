# C-2b variant ledger — 2026-11-03

Guardrails and initial entries were written before C-2b measurements. All
development uses seen seeds 11–13, cal/tail minus the locked exclusion list.
No named-parent condition occurs in the generator. Eight non-control
configurations were assessed: four new (P/F/FU/KU), four carried (K/U/T/S1).
C is the common control, not a trial. Partial attempts and dropped leads count.

| Variant | Exact change and mechanism | What ran | Development result | Disposition at freeze |
|---|---|---|---|---|
| K reference | `start_kabsch=true`; C-2 seeded fit with original orientation/pivot handling | Complete C-2 panel reanalysis; paired pilot and seven-case follow-up | Inherited primary top-1 CI upper +2.148 misses +2; artefact-free guard 2 trips / 6 mirrors / null 1. JEGPUJ 45/57/80 versus C 140/141/153 | Finalist, route B; no qualification claim |
| P / K-pivot | K plus `start_kabsch_pivot=true`; transport seeded fit translation to FK's original donor-atom pivot | Synthetic before/after proof; actual-engine traces; all native-config placement census; paired pilot; seven-case follow-up | Pooled row-max median 2.8071 ->0.3057 A versus K; >0.3-A change in 102 parents. Helps FELSAU; loses some K energy gain on JEGPUJ; CAXTAX floor worsens despite better placement | Finalist, route D; must still pass energy non-inferiority |
| F / K-fit | P plus `start_kabsch_diversity=0.0`; remove well-posed orientation perturbation, preserve rank-deficient rotations and all RNG draws | Actual-engine traces, full placement census; first three completed paired-pilot parents; QEZVEY recovery attempt aborted | Median 0.2486 A, only six fewer parents over 0.3 A than P. JEGPUJ top-1 115/95/139, weaker than K and P despite tighter placement | Dropped from shortlist; exploration loss is not justified by smaller offsets |
| FU / KU-fit | F plus `beam_union=true`, target share 0.5 | First three paired-pilot parents; QEZVEY recovery attempt aborted | Does not rescue JEGPUJ from F's weaker result. Confounds union with removal of orientation exploration | Dropped; replace with clean KU test, not a numerical retuning |
| KU | K plus `beam_union=true`, share 0.5, tolerance 1e-6 rad, group batch 16; default pivot and diversity | Predeclared seven-case C/K/P/KU run, all 84 records valid; 21 per arm, seeds 11–13 | JEGPUJ top-1 56/45/51; SIVBUV 11/4/6 rows versus C 2/2/2, seed-13 floor 44.8 versus 66.5. CAXTAX floor improves; ranking still imperfect | Finalist, route B |
| U reference | `beam_union=true`, share 0.5; unchanged C-2 implementation | Complete C-2 reanalysis under route A; exercised inside KU | Geometric improvement but no average energy/coverage qualification; top-1 CI upper +4.567. Artefact-free guard 2 trips /5 mirrors /null1 passes | No standalone finalist; retained in KU |
| S1 | Production flags, `single_w48`; one W48 experiment equals three W16 seeds; three noise replicates | Complete C-2 pool reanalysis; 20-draw size-matched rho; paired-pilot first three parents plus aborted QEZVEY attempt | Matched rho delta -0.0691 [-0.1006,-0.0397], n115. Inherited top-1 upper +4.67 still misses +2. No valid new speed inference | Finalist, route C; fresh confirmation decides |
| T | `pool_window_fill=true`; retain distinct finite output rows beyond the score window, existing cap | Complete C-2 export storage/usefulness reanalysis | 72 pools gain rows; no added row beats C floor; 26 have an extra row within 1 A, without new aggregate coverage. Storage 1.0547x for those exports | Experimental user output option, default off; no efficacy finalist or fold-in claim |

All efficacy finalists retain pool_window_fill=false. K/P/KU use W16 and
start_kabsch_diversity=1.0. S1 uses W48 and production search flags. No other
numeric beam/orientation budget was tried. Final fingerprints/settings/routes
are in c2/C2B_FINALISTS.json and C2B_CONTRACT.md.

## Attempt history and failures

1. The original actual-engine placement trace assumed `Ligand.smiles` and
   failed. Stopped own PID; retained placement_attempt1 and the original log.
   Corrected instrumentation and repeated the same named cases/settings. This
   was not a new algorithm variant or an outcome-based parameter change.
2. Actual-engine trace cases: JEGPUJ, SIVBUV, UYOVUA, MIVXAR, LOPCAW, XACMOE,
   FELSAU, CAXTAX, NEPNEF. Seven single-native-variant parents independently
   match the paired reconstruction exactly (84 quantile vectors); LOPCAW's
   native scope differs and NEPNEF fails refinement.
3. Before the placement census, a three-worker pilot used QEZVEY, HOXJUZ,
   JEGPUJ, XACMOE and BABNAV. Its conservative direct-mean wall estimate was
   logged first (33 min, allow 45). Reused those five records and measured the
   other 115 parents, all native variants: 133 configurations, 195,456 rows,
   NEPNEF sole failure. These are input geometry measurements, not timings used
   to establish a speedup.
4. The GPU pilot was declared C/K/P/F/FU/S1 on BABNAV, HOXJUZ, JEGPUJ, QEZVEY,
   XACMOE, seeds 11–13. First three parents completed (54 records). QEZVEY
   logged two graph-capture OOM warnings and continued eagerly. After the
   owner's severe-contention caution, the controller stopped own PID 79709 at
   2,488.13 elapsed process seconds. QEZVEY's 18 unexported and XACMOE's 18
   unreached records are explicit controller stubs, null energies, retained
   reasons/provenance. No rerun of QEZVEY and no aggregate/ratio from that
   incomplete invocation. The initial 10–20 min pilot estimate failed.
5. Clean KU follow-up cases were declared before their results: JEGPUJ,
   SIVBUV, FELSAU, CAXTAX, UYOVUA, MIVXAR, LOPCAW, C/K/P/KU, seeds 11–13.
   It began after the earlier process exited and nvidia-smi confirmed GPU
   release. All 84 records valid; identical row-steps per parent across arms.
   Existing LOPCAW SMILES-alignment/feasibility replacement warnings retained.
   No pivot-correction, graph-capture or solver-OOM fallback in this follow-up.
6. NEPNEF actual xTB trace: source ligands have formal charge -7, one-Li capped
   input charge -6; SCC/optimisation failure, no xtbopt.xyz. Generator returns
   degraded conformers in a two-step smoke; benchmark refuses the invalid
   fallback. Retain for D, no per-parent charge/embed patch.
7. Macrocycles: UYOVUA/MIVXAR/LOPCAW rejected by all-single/aromatic-band rules.
   No mixed-bond ring flag was attempted. SIVBUV collapse is not fixed by P
   alone; KU mitigates it in the declared cases. Scorer/DB issues are recorded
   for D, not modified here.
8. Golden layout diagnosis: eight configured partitions reproduce all 245
   arrays/profile; unchanged one-worker source fails eight arrays. OJIMIF
   partitions executed serially still match bit-for-bit. Checker pins the
   recorded layout, references unchanged. This is infrastructure, not a new
   search variant.

## Guardrail accounting

No seeds 21–23 generated or evaluated. No val/reserve/locked tuning. FEKKAJ and
NEPNEF remain the original primary exclusions; guard artefacts fixed before
new data (NUHPEL, MIDQEY, UMUBAF, YUCZUS, NUWFAM, CAFWIQ). No extra exclusion
was added. No scorer/DB or production default changed. No Colab VM was started;
spend 0 units. Current timing estimates/ratios are provisional under the
owner-reported 100% CPU/refit contention; no speedup claim.

The owner also requests long scheduled/completion-driven checks for unattended
runs (e.g. 20 min), rather than frequent polling. Confirmation follows that
preference after audit clearance. Freeze means stop; no confirmation here.
