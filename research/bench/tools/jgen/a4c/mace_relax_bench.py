"""Batched MACE-POLAR-1-S relaxation of the best rows per class: steps, wall, and where the minima land (GPU).

    python -m tools.jgen.a4c.mace_relax_bench --trace TRACE_DIR --run SOLVE_RUN [SOLVE_RUN ...] --arm cfg
        --inputs A4B_INPUTS --out OUT.json [--n 96] [--batch-atoms 6000] [--fmax 0.05] [--max-steps 300] [--dtype float64]

Rows: the `class_best` rows of `a4c.settle_trace` (TRACE_DIR/rows/*.json), so steps compare with the GFN2 traces.
Each row's unrelaxed geometry X0 and the tmQM reference are relaxed together in one batch: per-structure L-BFGS
(memory 10, max step 0.2 A per atom, no line search, as ASE's LBFGS), one batched MACE energy+force call per step for
every structure not yet converged (max |F| <= fmax eV/A), graphs rebuilt every step (the neighbour list moves).
Reported: steps per row, wall per step and in total (graph build vs model), steps of the references (a control: they
are GFN2 minima), and per unit the MACE energy of the relaxed row minus the relaxed reference and the heavy-atom RMSD
between them. Never fails the batch: a structure whose force call fails is dropped and recorded.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

EV2KJ = 96.485


def _kabsch(A, B):
    a, b = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(a.T @ b)
    S[-1] *= np.sign(np.linalg.det(Vt.T @ U.T))
    return float(np.sqrt(max((a ** 2).sum() + (b ** 2).sum() - 2 * S.sum(), 0.0) / len(a)))


class LBFGS:
    """ASE-style L-BFGS without line search for one structure (positions in A, forces in eV/A)."""

    def __init__(self, n, memory=10, maxstep=0.2, h0=70.0):
        self.s, self.y, self.rho = [], [], []
        self.m, self.maxstep, self.h0 = memory, maxstep, h0
        self.r0 = self.f0 = None

    def step(self, r, f):
        r, f = r.reshape(-1), f.reshape(-1)
        if self.r0 is not None:
            s, y = r - self.r0, self.f0 - f
            sy = float(s @ y)
            if sy > 1e-10:
                self.s.append(s); self.y.append(y); self.rho.append(1.0 / sy)
                if len(self.s) > self.m:
                    self.s.pop(0); self.y.pop(0); self.rho.pop(0)
        q = -f.copy()
        a = []
        for s, y, rho in zip(reversed(self.s), reversed(self.y), reversed(self.rho)):
            ai = rho * float(s @ q); a.append(ai); q -= ai * y
        z = q / self.h0
        for (s, y, rho), ai in zip(zip(self.s, self.y, self.rho), reversed(a)):
            b = rho * float(y @ z); z += s * (ai - b)
        dr = -z.reshape(-1, 3)
        longest = np.sqrt((dr ** 2).sum(1)).max()
        if longest > self.maxstep:
            dr *= self.maxstep / longest
        self.r0, self.f0 = r.copy(), f.copy()
        return r.reshape(-1, 3) + dr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True); ap.add_argument("--run", required=True, nargs="+")
    ap.add_argument("--arm", default="cfg"); ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=96); ap.add_argument("--batch-atoms", type=int, default=6000)
    ap.add_argument("--fmax", type=float, default=0.05); ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--dtype", default="float64")
    a = ap.parse_args()
    import torch
    from ase import Atoms
    from mace import data as mace_data
    from mace.tools import torch_geometric, torch_tools
    from jacobik.core.energy import MACEPolarEvaluator
    from jacobik.core.env import MACE_MODEL_PATH
    from tools.jgen.bench.pool_export import _read_frames

    cm = json.loads((Path(a.inputs) / "charge_mult.json").read_text())
    rows = []
    for p in sorted((Path(a.trace) / "rows").glob("*.json")):
        r = json.loads(p.read_text())
        t = [t for t in r.get("traces", []) if t["role"] == "class_best" and t["engine"] == "ancopt" and not t["error"]]
        if r.get("error") or not t or r["code"] not in cm:
            continue
        rows.append((r["code"], r["seed"], t[0]["row"], t[0]["cycles"], t[0]["e_minus_ref"][-1]))
    rows = rows[:a.n]
    S = []                                                           # structures: (unit, kind, syms, X, q, mult)
    for code, seed, i, gcyc, goff in rows:
        for run in a.run:
            mp = Path(run) / a.arm / "xyz" / code / f"{code}_{a.arm}_s{seed}_pool.json"
            if mp.exists():
                break
        m = json.loads(mp.read_text())
        g2r = np.asarray(m["gen_to_ref_idx"], int)
        frames = [np.asarray(fr[2], float)[g2r] for f in dict.fromkeys(r["file"] for r in m["rows"]) for fr in _read_frames(mp.parent / f)]
        q, mult = int(cm[code]["charge"]), int(cm[code]["mult"])
        syms = list(m["ref_symbols"])
        S.append(dict(unit=(code, seed), kind="row", syms=syms, X=frames[i].copy(), X0=frames[i].copy(), q=q, mult=mult, gfn2_cycles=gcyc, gfn2_off=goff))
        S.append(dict(unit=(code, seed), kind="ref", syms=syms, X=np.asarray(m["ref_X"], float), X0=np.asarray(m["ref_X"], float), q=q, mult=mult))
    print(f"rows {len(rows)} structures {len(S)} atoms {sum(len(s['syms']) for s in S)}", flush=True)

    ev = MACEPolarEvaluator(model="polar-1-s", device="cuda", dtype=a.dtype, model_path=str(MACE_MODEL_PATH), mode="energy-only", threads=1).load()
    calc = ev._get_calculator()
    calc.arrays_keys.update({calc.charges_key: "charges"})
    keyspec = mace_data.KeySpecification(info_keys=calc.info_keys, arrays_keys=calc.arrays_keys)
    model = calc.models[0]
    mdt = next(model.parameters()).dtype

    def graph(s):
        at = Atoms(symbols=s["syms"], positions=s["X"], pbc=False)
        at.info.update(charge=s["q"], spin=s["mult"], external_field=[0.0, 0.0, 0.0])
        with torch_tools.default_dtype(calc.default_dtype):
            cfg = mace_data.config_from_atoms(at, key_specification=keyspec, head_name=calc.head)
            return mace_data.AtomicData.from_config(cfg, z_table=calc.z_table, cutoff=calc.r_max, heads=calc.available_heads)

    def ef(idx):                                                     # batched energy (eV) and forces (eV/A)
        out_e, out_f = {}, {}
        chunks, cur, n = [], [], 0
        for k in idx:
            na = len(S[k]["syms"])
            if cur and n + na > a.batch_atoms:
                chunks.append(cur); cur, n = [], 0
            cur.append(k); n += na
        if cur:
            chunks.append(cur)
        tg = tm = 0.0
        for ch in chunks:
            t0 = time.perf_counter()
            data = [graph(S[k]) for k in ch]
            b = next(iter(torch_geometric.dataloader.DataLoader(dataset=data, batch_size=len(data), shuffle=False))).to(calc.device)
            for key in b.keys:
                v = b[key]
                if torch.is_tensor(v) and torch.is_floating_point(v):
                    b[key] = v.to(dtype=mdt)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            out = model(b.to_dict(), compute_force=True, compute_virials=False, compute_stress=False, training=False)
            E = out["energy"].detach().to("cpu", torch.float64).numpy() * calc.energy_units_to_eV
            F = out["forces"].detach().to("cpu", torch.float64).numpy() * calc.energy_units_to_eV
            torch.cuda.synchronize(); t2 = time.perf_counter()
            tg += t1 - t0; tm += t2 - t1
            off = 0
            for j, k in enumerate(ch):
                na = len(S[k]["syms"]); out_e[k] = float(E[j]); out_f[k] = F[off:off + na]; off += na
        return out_e, out_f, tg, tm

    opt = {k: LBFGS(len(s["syms"])) for k, s in enumerate(S)}
    active = list(range(len(S)))
    steps = {k: 0 for k in active}
    E0, Ef = {}, {}
    t_graph = t_model = 0.0
    per_step = []
    t_start = time.perf_counter()
    for it in range(a.max_steps + 1):
        if not active:
            break
        ts = time.perf_counter()
        try:
            e, f, tg, tm = ef(active)
        except RuntimeError as exc:
            print("force call failed:", exc); break
        t_graph += tg; t_model += tm
        nxt = []
        for k in active:
            if it == 0:
                E0[k] = e[k]
            Ef[k] = e[k]
            fmax = float(np.sqrt((f[k] ** 2).sum(1)).max())
            if fmax <= a.fmax or it == a.max_steps:
                continue
            S[k]["X"] = opt[k].step(S[k]["X"], f[k]); steps[k] += 1
            nxt.append(k)
        per_step.append(dict(it=it, n_active=len(active), wall=time.perf_counter() - ts, graph=tg, model=tm))
        active = nxt
    wall = time.perf_counter() - t_start
    res = []
    by_unit = {}
    for k, s in enumerate(S):
        by_unit.setdefault(s["unit"], {})[s["kind"]] = k
    for u, d in by_unit.items():
        r, f_ = d["row"], d["ref"]
        heavy = np.array([x != "H" for x in S[r]["syms"]])
        res.append(dict(code=u[0], seed=u[1], steps_row=steps[r], steps_ref=steps[f_], converged_row=steps[r] < a.max_steps,
                        gfn2_cycles=S[r]["gfn2_cycles"], gfn2_off_vs_ref_kj=S[r]["gfn2_off"],
                        mace_released_row_kj=(E0[r] - Ef[r]) * EV2KJ, mace_released_ref_kj=(E0[f_] - Ef[f_]) * EV2KJ,
                        mace_off_vs_ref_kj=(Ef[r] - Ef[f_]) * EV2KJ,
                        rmsd_heavy_row_min_ref_min=_kabsch(S[r]["X"][heavy], S[f_]["X"][heavy]),
                        rmsd_heavy_ref_moved=_kabsch(S[f_]["X0"][heavy], S[f_]["X"][heavy])))
    summ = dict(n_rows=len(res), n_structures=len(S), wall_s=wall, t_graph_s=t_graph, t_model_s=t_model, n_iterations=len(per_step),
                gpu=torch.cuda.get_device_name(0), dtype=a.dtype, fmax=a.fmax, batch_atoms=a.batch_atoms)
    Path(a.out).write_text(json.dumps(dict(summary=summ, units=res, per_step=per_step), default=float))
    q = lambda v: f"med {np.median(v):.1f} p75 {np.percentile(v, 75):.1f} p90 {np.percentile(v, 90):.1f}"
    print(json.dumps(summ))
    print("steps row:", q([r["steps_row"] for r in res]), "| GFN2 cycles same rows:", q([r["gfn2_cycles"] for r in res]))
    print("steps ref (GFN2 minima):", q([r["steps_ref"] for r in res]), "| ref heavy RMSD moved", q([r["rmsd_heavy_ref_moved"] for r in res]))
    print("MACE row min - ref min kJ/mol:", q([r["mace_off_vs_ref_kj"] for r in res]), "| share <= 0", np.mean([r["mace_off_vs_ref_kj"] <= 0 for r in res]).round(2),
          "<= 5", np.mean([r["mace_off_vs_ref_kj"] <= 5 for r in res]).round(2))
    print("GFN2 row min - ref (same units):", q([r["gfn2_off_vs_ref_kj"] for r in res]))
    print("heavy RMSD row min vs ref min:", q([r["rmsd_heavy_row_min_ref_min"] for r in res]))
    print(f"wall {wall:.1f} s for {len(S)} structures ({len(res)} rows + refs): {wall / len(S):.2f} s per structure; graph {t_graph:.1f} s, model {t_model:.1f} s")


if __name__ == "__main__":
    main()
