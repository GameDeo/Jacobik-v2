"""Step 6 damaged-trial DBs: the candidate DB with one refitted (damaged) entry swapped in.

For each stressed key and each mechanism (observed_input_error, parent_removal) the registered trial whose fixed-pool ranking moved most (lowest
minimum-over-seeds Spearman in ``*.rank.json`` of the pilot read-out, ties by trial order) is selected by this rule, before any generation result exists.
The base DB's entry at the resolved key is replaced field by field with the trial's refitted mixture; reference diagnostics (``ref_ll``, ``ref_sd``) are kept.
This is the same substitution the fixed-pool scorer makes, so the generation test and the fixed-pool test damage the DB identically. Other keys that
pool this key's parents are not re-derived (stated limit). Each output is a self-describing container plus ``<stem>.meta.json`` naming the damage.

    python make_damaged_dbs.py PIPELINE_DIR STRESS_DIR OUT_DIR [--base D-final]
"""
import argparse, hashlib, json, sys
from pathlib import Path

CASES = {  # label -> (stress file stem, DB entry key)
    "LUQVOK_Q1": ("luqvok_q1", "Q1:('Pd', 1, 2, 'O_carboxylate', 'square_plan', 'O_carboxylate')"),
    "LIPDEU_QO": ("lipdeu_qo", "QO:(2, 4)"),
}
SCENARIOS = ("observed_input_error", "parent_removal")


def select(stress: Path, stem: str, scenario: str):
    """(trial_id, min_spearman) of the worst-ranking trial of ``scenario``; first in file order on ties."""
    rows = json.loads((stress / f"{stem}.rank.json").read_text())["rows"]
    entries = {}
    for line in (stress / f"{stem}.entries.jsonl").read_text().splitlines():
        e = json.loads(line); entries[(e["scenario"], e["replicate"], e["affected_count"])] = e
    best = None
    for r in rows:
        if r["scenario"] != scenario:
            continue
        m = min(s["spearman"] for s in r["seeds"])
        if best is None or m < best[0]:
            best = (m, r)
    m, r = best
    e = entries[(r["scenario"], r["replicate"], r["affected"])]
    return e["trial_id"], m, e["entry"], r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline", type=Path); ap.add_argument("stress", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--base", default="D-final")
    a = ap.parse_args()
    sys.path.insert(0, "src")
    from jacobik.core import container as dbio
    base = a.pipeline / a.base / "constraints.json"
    meta_head, tables = dbio.read_container_head(base)
    raw = json.loads(base.read_text())
    entries = {k: v for k, v in raw.items() if not k.startswith("_")}
    base_sha = dbio.sha256_file(base)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for label, (stem, key) in CASES.items():
        for scenario in SCENARIOS:
            tid, spearman, entry, r = select(a.stress, stem, scenario)
            new = dict(entries[key]); new.update(entry)
            db = dict(entries); db[key] = new
            name = f"{label}_{'obs' if scenario == 'observed_input_error' else 'rem'}"
            meta = dict(meta_head)
            meta["damage"] = dict(base_db=a.base, base_sha256=base_sha, key=key, scenario=scenario, trial_id=tid,
                                  affected_count=r["affected"], replicate=r["replicate"], min_seed_spearman=spearman,
                                  rule="lowest minimum-over-seeds Spearman of the pilot ranking read-out within the mechanism",
                                  substitution="entry at the resolved key replaced; pooled descendants not re-derived")
            path = a.out / f"{name}.json"
            path.write_text(dbio.dumps_container(db, meta, tables))
            sidecar = json.loads((a.pipeline / a.base / "constraints.meta.json").read_text()); sidecar["damage"] = meta["damage"]
            path.with_suffix(".meta.json").write_text(json.dumps(sidecar, sort_keys=True, indent=1))
            manifest.append(dict(name=name, path=str(path), sha256=dbio.sha256_file(path), **meta["damage"],
                                 old_k=len(entries[key].get("means", [])), new_k=len(new.get("means", []))))
            print(name, tid, f"spearman {spearman:.4f}", "k", manifest[-1]["old_k"], "->", manifest[-1]["new_k"], flush=True)
    (a.out / "damaged_manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
