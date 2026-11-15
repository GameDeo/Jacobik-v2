# P2R Step 3: where the misses come from (interim, 2026-11-12)

Inputs:
- the baseline (`BASELINE_REPORT.md`, post job 61204770);
- the A4t full-pool relaxations of the 26 cal parents that miss on every seed, evaluated with A4c (energy offsets of
  A4t-relaxed rows filled from `E_mace_kj` minus the same relaxed-reference energy);
- a lineage run with per-start coordinates (`JACOBIK_LINEAGE_XYZ=1`, `a4c.lineage_class`, PM `$JK_SCR/runs/p2r-lin*`,
  same production, seeds 11-13) on the not-offered parents.

All on Perlmutter login nodes; no billed time.

## 1. Offered misses (the class is in the unrelaxed pool; 72 cal units)
- **64 units: right class, different basin.** The relaxed same-class rows are at 0.3-1.1 Å core RMSD and +5 to
  +140 kJ/mol above the relaxed reference. Every row converged, and the references are GFN2 minima already (1 cycle),
  so these are real basins, not relaxation artefacts.
- 5 units: the class appears only beyond 1.5 Å core RMSD, so it was never relaxed.
- 3 units: the class rows relaxed out of the class (OMAGAK E/Z, XACMOE a centre).

**Full pool (A4t, the 26 parents missed on all seeds): is the reference basin anywhere in the pool?**

| Group | Parents | Reading |
|---|---|---|
| A pool row hits; the 5-row relax set did not pick it | JUPDON (2/3 seeds), NOYTUS (3/3), XEVDAE (1/3) | selection / ranking (small) |
| Class present, best same-class basin above +5 kJ/mol | ZEFXAM +5.0, NUCKUS +5.5, EFOXII +10, CAQLEP +10, BABNAV +13, GOTWET +15, FEXWUC +21, HOHZAG +21, EZIBUL +23, CIXYUD +37, FEKKAJ +58, DUHHIX +86, ZAQNUE +86 | **reference basin within the class never reached** |
| Class absent from the whole pool | the 10 below | enumeration / culling / input |

## 2. Not-offered parents: does any solver start ever land in the class? (lineage, all starts of every pass)

| Verdict | Parents | Reading |
|---|---|---|
| Only the enantiomer is produced | ZOJDEH (128/128 starts), YOSXIP (~270/512), LIPDEU (~100/256), WEYJIT s13 | mirror completion |
| Same-class starts exist but are not kept | FETTAZ: 8-13 per seed, all `beyond_cap` (output cap reached first) | cap / allocation |
| | HUNGOP s12/13: all `merged` into a kept row of another class (the RMSD dedup does not see the class) | dedup merges classes |
| No start ever in the class, input valid | stereocentre differs: FUTSOD, LISVOC, PANBOY; metal contact set differs: DAMVAQ, GOTWET, HEDJOP, HOXJUZ; MEJYIN (E/Z, contacts) | enumeration / binding mode (to inspect) |
| Input ambiguous (reference not compatible with the target typing) | JUCBIV, NUWFAM, HUNGOP | input artefact, reported separately |

## Reading so far
- **The largest category is "right class, reference basin not reached".** That covers the 64 offered units and 13 of
  the 26 all-seed parents.
  - The classes are offered. The search inside a class does not reach the reference basin, which is often tens of
    kJ/mol lower.
  - This points at within-class search: core conformations, ring puckers, chelate conformers, strain. It does not point
    at beam allocation across classes.
- **Enumeration or binding mode is a real but smaller share:** 7 valid parents.
  - Missing stereocentre configurations (3) are what the near-product stereo rule addresses.
  - Different contact sets (4) need a per-case look at whether they are binding-mode alternatives the generator should
    offer.
- **Mirror completion: 4 parents** on the not-offered side; with NOYTUS etc., 10 units in the baseline.
- **Cull and dedup losses: 2 parents.**
  - FETTAZ: the output cap drops same-class starts.
  - HUNGOP: the dedup merges across classes. A class-aware dedup and output cap (Step 4 items 4 and 6) fixes both.
- **Selection: 3 parents.** The full pool has the hit; the relax set did not take it.

## 3. Unrelaxed quality (the product is the unrelaxed rows; owner 2026-11-12)
The relaxed results are a measuring instrument. A wider relax set is information, never a fix.
`a4c.unrelaxed` gives, per unit, the core RMSD of the nearest same-class unrelaxed row to the reference (all 351
baseline units, 2 errors).

| Relaxed bucket | Units | Nearest unrelaxed same-class core RMSD, median [IQR] | Within 0.3 Å unrelaxed |
|---|---|---|---|
| C-TOP1 | 148 | 0.26 [0.19, 0.41] | 0.58 |
| C-RANK | 83 | 0.34 [0.25, 0.50] | 0.37 |
| C-MISS | 72 | 0.65 [0.49, 0.91] | 0.01 |
| All with a same-class row | 304 | 0.36 | 0.39 |

- About 119/351 units (0.34) have an unrelaxed same-class row within 0.3 Å core RMSD.
- The relaxed metric covers about 0.68 of the evaluated units. The gap is relaxation pulling rows from 0.3-0.6 Å into
  the basin.
- **Owner ruling (2026-11-12): no unrelaxed coverage gate.**
  - Coverage is judged on relaxed structures, which removes variation outside the solver's control.
  - The product is the unrelaxed rows. Each should lie in a real basin and close to it, and together they should cover
    the basins. Unrelaxed closeness (core RMSD to the basin a row relaxes into) and garbage are quality diagnostics.
  - Relaxation is never a fix.
- The top-1 row, when it is the reference class, has median core 0.35 (C-TOP1) and 1.12 (C-RANK).

### Own-basin quality of the unrelaxed rows (`a4c.own_basin`, 1434 baseline relaxed rows)
The question is how far each unrelaxed row is from the basin it relaxes into.

| Rows | Heavy RMSD to own basin, median / p90 / ≤ 0.5 Å | Core RMSD, median / ≤ 0.3 Å | MACE energy drop on relaxation, median / p90 / ≤ 25 kJ/mol | Class unchanged by relaxation |
|---|---|---|---|---|
| all | 0.18 / 0.49 / 0.90 | 0.14 / 0.86 | 43 / 158 / 0.29 | 0.874 |
| top-1 | 0.15 / 0.38 / 0.93 | 0.12 / 0.90 | 39 / 107 / 0.37 | 0.896 |

- **Geometrically, the unrelaxed rows sit close to their own basins.**
- **Energetically, they are strained:** a median of 43 kJ/mol is released on relaxation, consistent with the A4 strain
  split (about 2/3 of it reachable in solver space, 1/3 in donor-adjacent bonds and angles).
- **12.6 % of rows change class on relaxation.** These rows do not belong to the class their geometry first shows, a
  direct product defect.
- Unrelaxed-quality targets for Step 4: lower strain (energy drop) and fewer class changes at equal coverage.

## Reachability probe: first attempt is biased (recorded, not used)
`a4c.refscore` re-scores pool rows with the production ranking score. The check passes: top-1 row score minus rescore
minus its bend prior is about 1e-7 on MUFFEA.
- The raw tmQM geometry scores below every pool row even where production hits (MUFFEA: 1.58 against 1.92).
- The reason: the solver's rows are local maxima of the score; the crystal geometry is not optimised under it.
- So a fair probe has to fit the solver DOFs to the reference and then ascend the score locally from there. That is a
  new tool, not yet built.

## Still to do in Step 3
1. Reachability probe on the "basin not reached" parents (HOHZAG, ZAQNUE, DUHHIX, FEKKAJ, EZIBUL, ZEFXAM):
   - start from the relaxed reference and see whether the score holds it (search lost it) or drifts away (score surface);
   - check which core feature differs (ring pucker, chelate conformation, donor-adjacent torsion).
2. The four valid contact-set parents (DAMVAQ, GOTWET, HEDJOP, HOXJUZ) by hand.
3. The remote-stereo coupling check.
4. Register the Step 4 targets from the final shares.
