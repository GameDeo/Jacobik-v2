"""S7 prevalence (research/archive/docs/roadmap/roadmap_generation_v1.md S7 step 3): for every parent of the frame (or the codes given), the donor
count of the solver shell with ring_closure_dative_donors off (legacy) and on, and whether the fix changes the parent.
Reads SMILES and geometry only (tools/jgen/inputs.py). Writes JSON to stdout.

    python -m tools.jgen diag d33_s7_prevalence [CODE ...] > out.json
"""
import json
import re
import sys

RC = re.compile(r"(?:->|<-)\d")


def main():
    from jacobik.topology import mapping as M
    from tools.jgen import inputs

    shards = inputs.frame_shards()
    codes = sys.argv[1:] or sorted(shards)
    smiles = inputs.tmqm_smiles(inputs.CLEAN / "tmQM")
    rows = []
    for code in codes:
        rec = dict(code=code, ring_closure_dative=bool(RC.search(smiles.get(code) or "")))
        try:
            shard, ds = shards[code]
            counts = {}
            for flag in (False, True):
                M.RING_CLOSURE_DATIVE_DONORS = flag
                mol = inputs.load_mol(code, shard, ds)
                if mol is None:
                    raise RuntimeError("no geometry")
                counts[flag] = sorted(tuple(s.donor_indices) for s in mol.coord_sites)
            M.RING_CLOSURE_DATIVE_DONORS = False
            rec.update(n_sites_legacy=len(counts[False]), n_sites_on=len(counts[True]),
                       changed=counts[False] != counts[True])
        except Exception as exc:  # recorded, never raised
            rec.update(error=f"{type(exc).__name__}: {str(exc)[:160]}", changed=False)
        rows.append(rec)
    n = len(rows)
    summary = dict(parents=n, with_ring_closure_dative=sum(r["ring_closure_dative"] for r in rows),
                   changed=sum(r["changed"] for r in rows), errors=sum("error" in r for r in rows),
                   changed_codes=[r["code"] for r in rows if r["changed"]])
    print(json.dumps(dict(summary=summary, rows=rows), indent=1))


if __name__ == "__main__":
    main()
