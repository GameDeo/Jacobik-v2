"""Full-pool GFN2 relaxation and CUDA MACE pass for frozen A4b pool rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os

#: production fingerprint the A4 records must carry (A4b ran on 273f2ccbf892; reruns set JACOBIK_A4_FINGERPRINT)
A4_FINGERPRINT = os.environ.get("JACOBIK_A4_FINGERPRINT", "273f2ccbf892")

import numpy as np

from tools.jgen.bench.pool_export import _read_frames
from tools.jgen.a4b.relax_run import to_ref_order
from tools.runkit import mace, xtb


def run(data_root, out, parents, nproc=4, device="cuda", seeds=(11,12,13), evaluator=None, resume=None):
    """resume: a previous A4t relax dir; its successful and capped rows and references are
    reused with their stored MACE energies, so xtb and MACE run only for rows and references it does not have."""
    from tools.jgen.a4b.relax_run import load_prior
    prior_t, prior_tref = load_prior(resume)
    resumed = 0
    data, out = Path(data_root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    xtb.require()
    cm = json.loads((data / "charge_mult.json").read_text())
    pending, jobs, job_pending, refs, failures = [], [], [], {}, []
    reused = 0
    for code in parents:
        for seed in seeds:
            try:
                mp = data / "xyz" / code / f"{code}_control_s{seed}_pool.json"
                manifest = json.loads(mp.read_text())
                rec = json.loads((data / "records" / f"{code}_{seed}.json").read_text())
                if rec.get("status") != "ok" or rec.get("fingerprint") != A4_FINGERPRINT:
                    raise ValueError("A4b record status/fingerprint mismatch")
                if int(manifest["n_rows"]) != len(manifest["rows"]) or len(rec["energies_kj"]) != len(manifest["rows"]):
                    raise ValueError("pool/record row counts differ")
                symbols = manifest["ref_symbols"]
                mapping = manifest["gen_to_ref_idx"]
                q, mult = int(cm[code]["charge"]), int(cm[code]["mult"])
                refs.setdefault(code, dict(symbols=symbols, X=np.asarray(manifest["ref_X"], float), q=q, mult=mult))
                frames = list(_read_frames(mp.parent / manifest["rows"][0]["file"]))
                if len(frames) != len(manifest["rows"]): raise ValueError("pool XYZ frame count mismatch")
                oldp = data / "relax" / f"{code}_{seed}.json"
                old = json.loads(oldp.read_text()).get("rows", {}) if oldp.exists() else {}
                for i, (header, _, Xgen) in enumerate(frames):
                    if int(header["row_index"]) != i: raise ValueError("XYZ row index mismatch")
                    X = to_ref_order(Xgen, mapping)
                    prior = old.get(str(i))
                    pt = prior_t.get((code, int(seed), i))
                    if pt is not None and pt.get("status") in ("ok", "capped") and \
                            (pt.get("status") != "ok" or (pt.get("X") is not None and pt.get("E_mace_kj") is not None)):
                        initial = dict(**{k: v for k, v in pt.items() if k != "reused_from"},
                                       reused_from=pt.get("reused_from") or "A4t", _resumed=True)
                        resumed += 1
                    elif prior is not None and prior.get("status") in ("ok", "capped"):
                        # Any A4b attempt is reused, including capped rows (which remain NaN); never retry a row.
                        initial = dict(**prior, reused_from="A4b")
                        reused += 1
                    else:
                        initial = {"status": "pending"}
                        jobs.append((symbols, X, q, mult)); job_pending.append(len(pending))
                    pending.append(dict(code=code, seed=seed, row=i, symbols=symbols, X=X,
                                        q=q, mult=mult, initial=initial))
            except Exception as exc:
                failures.append(dict(code=code, seed=seed, status="failed", reason=f"{type(exc).__name__}: {exc}"))

    ref_list_all = list(refs.items())
    ref_keep = {c: prior_tref[c] for c, _ in ref_list_all if c in prior_tref and prior_tref[c].get("X") is not None
                and prior_tref[c].get("E_ref0_mace_kj") is not None and prior_tref[c].get("E_refrel_mace_kj") is not None}
    ref_list = [(c, v) for c, v in ref_list_all if c not in ref_keep]
    refjobs = [(v["symbols"], v["X"], v["q"], v["mult"]) for _, v in ref_list]
    # one pool for pool rows and references (independent jobs, same results), largest first: one tail, not two
    alljobs = jobs + refjobs
    order = sorted(range(len(alljobs)), key=lambda i: -len(alljobs[i][0]))
    allres_sorted = xtb.opt_many([alljobs[i] for i in order], workers=max(1, nproc - 1),
                                 workroot=str(out / "tmp"), cycles=200) if alljobs else []
    allres = [None] * len(alljobs)
    for i, r in zip(order, allres_sorted): allres[i] = r
    relaxed_sorted = allres[:len(jobs)]
    result_by_pending = {job_pending[i]: r for i, r in enumerate(relaxed_sorted)}
    ref_results = allres[len(jobs):]
    structures, labels = [], []
    emap = {}
    for i, (code, ref) in enumerate(ref_list):
        structures.append((ref["symbols"], ref["X"], ref["q"], ref["mult"])); labels.append(("ref0", i))
        r = ref_results[i]
        if r.get("status") == "ok" and r.get("X") is not None:
            structures.append((ref["symbols"], np.asarray(r["X"]), ref["q"], ref["mult"])); labels.append(("ref", i))
    # MACE energies for every relaxed pool row are measured on the same L4 as the references.
    for i, p in enumerate(pending):
        r = p["initial"] if i not in result_by_pending else result_by_pending[i]
        p["relax"] = r
        if r.get("_resumed"):
            if r.get("status") == "ok":
                emap[("row", i)] = float(r["E_mace_kj"])
            continue
        if r.get("status") == "ok" and r.get("X") is not None:
            structures.append((p["symbols"], np.asarray(r["X"]), p["q"], p["mult"])); labels.append(("row", i))
    ev = evaluator if evaluator is not None else mace.Evaluator(device=device)
    energies = ev.energies_kj(structures) if structures else np.zeros(0)
    emap.update({label: float(value) for label, value in zip(labels, energies)})
    oldrefs = json.loads((data / "relax" / "references.json").read_text())
    refout = {}
    for i, (code, ref) in enumerate(ref_list):
        rr = ref_results[i]
        refout[code] = dict(**{k:v for k,v in rr.items() if k != "X"},
                            X=None if rr.get("X") is None else np.asarray(rr["X"]).tolist(),
                            E_ref0_mace_kj=emap.get(("ref0", i)), E_refrel_mace_kj=emap.get(("ref", i)),
                            E_refrel_A4b_kj=oldrefs.get(code, {}).get("E_mace_kj"))
    refout.update(ref_keep)
    refout = {c: refout[c] for c, _ in ref_list_all}
    (out / "references.json").write_text(json.dumps(refout, allow_nan=False))
    for i, p in enumerate(pending):
        r = p["relax"]
        status = "capped" if (r.get("status") == "capped" or (r.get("cycles") is not None and int(r["cycles"]) >= 200 and not r.get("converged"))) else ("ok" if r.get("status") == "ok" else "failed")
        E = emap.get(("row", i)) if status == "ok" else None
        row = dict(**{k:v for k,v in r.items() if k not in ("X", "E_mace_kj", "status", "reused_from", "_resumed")},
                   status=status, X=None if r.get("X") is None else np.asarray(r["X"]).tolist(),
                   E_mace_kj=E, reused_from=r.get("reused_from"))
        path = out / f"{p['code']}_{p['seed']}.json"
        d = json.loads(path.read_text()) if path.exists() else dict(code=p["code"], seed=p["seed"], rows={})
        d["rows"][str(p["row"])] = row
        path.write_text(json.dumps(d, allow_nan=False))
    (out / "failures.json").write_text(json.dumps(failures, indent=2))
    metrics = dict(n_pool_rows=len(pending), n_relax_new=len(jobs), n_relax_reused=reused,
                   n_references=len(ref_list_all), n_references_resumed=len(ref_keep), n_rows_resumed=resumed,
                   n_manifest_failures=len(failures),
                   xtb_wall_sum_s=sum(float(x.get("wall_s") or 0) for x in relaxed_sorted + ref_results),
                   mace=ev.stats, parents=list(parents))
    (out / "run_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--parents", required=True); p.add_argument("--nproc", type=int, default=4); p.add_argument("--device", choices=("cuda","cpu"),default="cuda"); p.add_argument("--seeds",default="11,12,13")
    a = p.parse_args(); parents = json.loads(Path(a.parents).read_text())
    print(json.dumps(run(a.data, a.out, parents, a.nproc, a.device, tuple(map(int,a.seeds.split(",")))), indent=2))


if __name__ == "__main__": main()
