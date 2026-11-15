# P2R Step 2: coverage baseline on production `d5e2f4113497`, results

The contract is `BASELINE_CONTRACT.md` (written before these results). The data is in
`research/data/p2r/baseline/pm_post_61204770/`. On Perlmutter it is in `$JK_SCR/runs/p2r-solve` and
`$JK_SCR/runs/p2r-post-61204770`.

## Run
- **Solves:** Perlmutter login A100s (login10, then login28 after the 9-minute cap; resumed). 579 + 305 s wall. 360/360
  unit records.
- **Post:** billed CPU node, job 61204770, debug QOS, 12.75 min (0.21 node-hours). Before it, job 61204387 failed in
  preflight after 9 s: the sbatch lacked `JK_SRC`, and `jk_preflight.sh` did not know the npz sidecar name. Both are
  fixed.
- **Relaxations:** 1449 rows plus 116 references. A4t added 2458 full-pool rows for 26 cal parents (C-MISS on every
  seed). 4464 MACE single points.
- **Parents with no solve: 4/120 (3.3 %, under the 5 % stop rule).** All stay in the denominators.
  - NEPNEF: known GFN2 embed fallback (`fallback_codes.json`).
  - UYOVUA: **new production regression.** It solved under every earlier production. The GFN2 embed refinement places
    the Li caps of the two convergent amide N- donors of the N4 macrocycle 0.57 Å apart, so the xtb SCC fails and the
    bench refuses the MMFF embed. The fix belongs in `engine.py` (owner).
  - QEZVEY: over the per-parent wall budget (300 s); the known cost case.
  - SILROY: over the per-parent wall budget (180 s).
- No unit was built without GFN2.

## Headline (parent-clustered bootstrap, 10k, seed 20261112)
| Measure | cal | primary root |
|---|---|---|
| **C-MISS, failed parents counted as misses** (contract denominator) | **0.350** [0.272, 0.433] (n=120) | **0.330** [0.251, 0.413] (n=109) |
| C-MISS over the 112 evaluated cal parents | 0.321 [0.244, 0.402] | 0.311 [0.233, 0.395] (n=103) |
| C-MISS with mirror completion (diagnostic) | 0.292 [0.217, 0.372] | 0.282 [0.204, 0.362] |
| C-TOP1 / C-RANK | 0.426 / 0.253 | |
| Top-1 relaxed at or below the reference, any class: ≤ 0 / ≤ +5 kJ/mol | 0.253 / 0.414 | |
| `strict_any` / `strict_all` | 0.768 / 0.589 | |

- The retrospective on old production gave 0.319, so production coverage is unchanged in practice. It is not a
  paired comparison.
- The tail-only parents (4 evaluated) are too few to read: C-MISS 2/4 (DAMVAQ, HOXJUZ, both not offered).
- Strata (cal, evaluated):
  - valid 0.333 (94);
  - repaired 0.074 (9);
  - ambiguous 0.571 (7);
  - unrepaired 0 (2).

## Miss split (input to Step 3)
- **Not offered: 0.107** [0.057, 0.164]. No unrelaxed pool row is the reference class. 14 parents:
  - FETTAZ, FUTSOD, GOTWET, HEDJOP, JUCBIV, LIPDEU, LISVOC, PANBOY, YOSXIP and ZOJDEH on all seeds;
  - HUNGOP and MEJYIN on 2 seeds;
  - NUWFAM and WEYJIT on 1 seed.
- **Offered: 0.214** [0.149, 0.286]. The class is in the pool, but no relaxed row hits. Named among them: HOHZAG,
  DOHTOM, FEKKAJ, XACMOE, NOYTUS, BABNAV, CAQLEP, CIXYUD, DUHHIX, EFOXII, EZIBUL, FEXWUC, JUPDON, NUCKUS.
  - "Offered" is the class at ANY core RMSD. It mixes rows that relaxed into a different basin of the right class
    (strain or misrank) with rows never relaxed because the class was there only beyond 1.5 Å core RMSD. Step 3
    splits these by the nearest same-class core RMSD and by the A4t full-pool relaxations.
- **Enantiomer-only units: 10**: NOYTUS ×3, YOSXIP ×3, NUHPEL, TIKZOF, WEYJIT, XEVDAE. These are what mirror completion
  would recover (0.321 to 0.292).

## Diagnostics
- **Classes per unit** in the unrelaxed pool: median 2, max 58.
  - The 5-row cap dropped other-class representatives in 103/336 cal units. Same-class rows come first, so the drops
    do not affect hits; they limit class completeness.
- **Garbage** (topology change or > +50 kJ/mol over the relaxed reference): 11.6 % of relaxed rows; 75 units have at least one.
- **Why relaxed rows are not the reference class:**
  - enantiomer 115;
  - contacts 98;
  - descriptors chi 55;
  - arrangement 54;
  - E/Z 16;
  - topology 2.
- **Hits by RMSD 322, by energy 246.**

## Not done here
Step 3 classifies each miss (not offered / offered-unreached / reached-but-strained or misranked / input / embed), runs
the reachability probe on the named cases, and checks remote-stereo coupling.
