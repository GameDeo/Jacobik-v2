"""CPU reproduction of the failed control solves: BODCAY, LIYTAQ, OBABAX (read-only, CPU only).

Manifest status (frame_v1/manifest.json): BODCAY and LIYTAQ failed with 'Solver exception: ValueError: Bad
Conformer Id' (no pool); OBABAX ended as 'oom' ('CUDA Out of Memory on solve_seeds after retrying max_rows down to
128'), no pool. For each code this script (i) labels the SMILES with d14, (ii) runs the builder and the pre-beam
set-up for seed 11 on CPU (multiseed.prepare, the same path the control solve takes), catching any exception, and
(iii) when prepare succeeds, runs multiseed.seed_rows to see whether the beam stage reaches the same error.
The exception type, message and the innermost repository frame are recorded. The OBABAX OOM is a GPU memory
event and is not reproduced on CPU; the CPU result only says whether the set-up stage completes.

    python tools/jgen/diag/d18_failed_parent_repro.py --codes BODCAY,LIYTAQ,OBABAX --out-json <path>
"""
import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ctxio  # noqa: E402
import d14_input_consistency as d14  # noqa: E402


def innermost_repo_frame(exc):
    tb = traceback.extract_tb(exc.__traceback__)
    repo = [f for f in tb if "/src/" in f.filename or "/refit/" in f.filename or "/tools/" in f.filename]
    last = repo[-1] if repo else (tb[-1] if tb else None)
    if last is None:
        return None
    return dict(file=os.path.relpath(last.filename, ctxio.REPO), line=last.lineno, func=last.name,
                code=(last.line or "")[:160])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    import json
    idx = ctxio.frame_index()
    results = []
    for code in a.codes.split(","):
        rec = dict(code=code, frame=idx.get(code))
        mol = ctxio.load_parent(code, idx)
        if mol is None:
            rec["error"] = "not loadable"
            results.append(rec)
            continue
        lab = d14.classify(mol)
        rec.update(label=lab["label"], hard=lab["hard"], soft=lab["soft"], smiles=mol.smiles)
        try:
            from jacobik.core.env import build_builder_and_inputs
            builder, _, berr = build_builder_and_inputs(mol)
            rec["builder_error"] = berr
            if builder is None:
                results.append(rec)
                continue
            _, ctx, asg_w, perr = ctxio.load_ctx(mol, a.seed)
            rec["prepare_ok"] = ctx is not None
            if ctx is None:
                rec["prepare_error"] = perr
                results.append(rec)
                continue
            from tools.jdev import multiseed
            multiseed.seed_rows(builder, ctx, asg_w[0], asg_w[1], [a.seed])
            rec["seed_rows_ok"] = True
        except Exception as exc:  # the point is to record it
            rec["exception"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
            rec["innermost_repo_frame"] = innermost_repo_frame(exc)
        results.append(rec)
    print(ctxio.to_json(dict(results=results)))
    if a.out_json:
        with open(a.out_json, "w") as fh:
            fh.write(ctxio.to_json(dict(results=results)))


if __name__ == "__main__":
    main()
