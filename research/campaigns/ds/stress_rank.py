"""Ranking-level read-out of the stress pilot: how much does the whole fixed-pool ranking move, not only the top-1?

Reuses the registered trials and refitted entries of ``stress_run.py``; for every trial and seed compares the trial's pool scores with the clean (base DB)
scores: Spearman rank correlation, overlap of the top-10 sets, rank of the clean top-1 under the trial score, and the energy of the trial top-10 mean.

    python stress_rank.py PIPELINE_DIR ENTRIES.jsonl OUT.json --case LUQVOK --quantity Q1 --key "<resolved key>" [--base D-final]
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stress_run as SR   # installs the build profile before any jacobik import
import numpy as np
from scipy.stats import spearmanr
from fit.pipeline import confidence as stress_driver


def metrics(clean, trial, k=10):
    out = []
    for s in range(clean["scores"].shape[0]):
        ok = clean["scores"][s] > -1e8
        c, t, e = clean["scores"][s][ok], trial["scores"][s][ok], clean["energies_kj"][s][ok]
        top_c, top_t = set(np.argsort(-c)[:k]), set(np.argsort(-t)[:k])
        rank_of_clean_top1 = int((t > t[np.argmax(c)]).sum())
        srt = np.sort(c)[::-1]
        out.append(dict(n=int(ok.sum()), spearman=float(spearmanr(c, t).statistic) if np.ptp(t) > 0 and np.ptp(c) > 0 else 1.0,
                        top10_overlap=len(top_c & top_t) / k, clean_top1_rank_under_trial=rank_of_clean_top1,
                        trial_top10_mean_energy_change_kj=float(e[list(top_t)].mean() - e[list(top_c)].mean()),
                        clean_margin_top1_top2=float(srt[0] - srt[1]), score_range=float(np.ptp(c))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pipeline", type=Path); ap.add_argument("entries", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--case", required=True); ap.add_argument("--quantity", required=True); ap.add_argument("--key", required=True)
    ap.add_argument("--base", default="D-final"); ap.add_argument("--ref-dir", type=Path, default=SR.MAIN / "data/ref")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--replicates", type=int, default=32); ap.add_argument("--limit", type=int)
    ap.add_argument("--scenarios", nargs="*", default=["observed_input_error", "parent_removal"])
    a = ap.parse_args()
    reg, trials, key = SR._setup(a)
    from tools.jgen.ds import stress_trial as ST
    score_fn, clean, info = ST.prepare(ST.manifest_row(SR.MAIN, a.case, "D"), resolved_key=key if a.quantity == "QO" else eval(key), quantity=a.quantity,
                                       score_db=str(a.pipeline / a.base / "constraints.json"))
    entries = {json.loads(l)["trial_id"]: json.loads(l) for l in a.entries.open()}
    rows = []
    for t in trials:
        e = entries.get(t["trial_id"])
        if e is None or "error" in e:
            continue
        m = metrics(clean, score_fn(stress_driver.fitdist_kwargs(e["entry"])))
        rows.append(dict(scenario=t["scenario"], affected=t["affected_count"], replicate=t["replicate"], k=e["entry"].get("k"), seeds=m))
    a.out.write_text(json.dumps(dict(info=info, rows=rows), default=str))
    print("done", len(rows))


if __name__ == "__main__":
    main()
