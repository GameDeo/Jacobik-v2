"""NCI-on ranking on a fixed small cohort; stored pools/energies, no solves.

Only Q6 and NCI change. Keep stored assignment-dependent terms and bend prior,
verify reconstructed Q6, and cross-check the delta with full scorer components.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import DB, REPO, pool_energy, pool_path, stats_from_db
from pools import setup, set_cases

COHORT = {
    "hbond": ["PEWBOL", "YIYXEJ", "CBTYCO", "BULGAT"],
    "stacking": ["POZSIG", "JEFJUZ", "XEPKAF", "SOKVET", "FETSUU", "GUYFAH"],
    "controls": ["OJIMIF", "HUYNEU"],
}
CANDIDATE = REPO / "data/refit/nci/2026-10-18_bound_fix/constraints_nci_2026-10-18.json"


def run(out):
    import pandas as pd
    import torch
    from jacobik.core import profile
    from jacobik.ik.geometry import GeometryExtractor
    from jacobik.ik.scorer import Scorer
    from research.archive.code.interactions import nci
    from jacobik.topology.decompose import decompose_ligand
    from tools.jdev.cases import get_case
    from tools.jdev.start import starting_geometry
    from research.archive.code.refit.nci_refit import digest
    from tools.jdev.env import tree_hash

    assert profile.ACTIVE_NAME == "u1" and profile.get("oxidation_model") == "training"
    assert not nci.NCI_ON and not profile.get("bvs")
    torch.set_num_threads(1)
    start = time.perf_counter()
    fit = json.loads((CANDIDATE.parent / "fit_summary.json").read_text())
    assert digest(DB) == fit["base_sha256"] and digest(CANDIDATE) == fit["candidate_sha256"]
    stats = stats_from_db(CANDIDATE)
    names = {c: (n, s) for s in ("dev32", "n48") for c, n in set_cases(s)}
    rows, arrays, cases = [], {}, {}
    for group, codes in COHORT.items():
        for code in codes:
            name, set_name = names[code]
            case = get_case(name)
            # Reference coordinates are used only to obtain the existing atom mapping
            # and topology. Scores below are exclusively at stored pool coordinates.
            with starting_geometry(case, "full_oracle"):
                mol, _, _ = case.make_builder()._build_shell(False, False)
            for lig in mol.ligands:
                decompose_ligand(mol, lig)
            ge = GeometryExtractor(mol, torch.device("cpu"))
            scorer = Scorer(mol, ge, stats, torch.device("cpu"))
            runtime = nci.build_runtime(mol, stats.nci_tables(), "cpu")
            assert scorer._q6_pmf is not None and scorer._q6_qweight == 1.0
            case_rows = []
            for seed in (11, 12, 13):
                f = pool_path("etkdg_u1_os", set_name, code, seed)
                energy = pool_energy(f, "mace64")
                with np.load(f) as z:
                    assert [a.symbol for a in mol.atoms] == z["symbols"].tolist(), code
                    fields = json.loads(str(z["profile"]))["fields"]
                    assert fields["oxidation_model"] == "training" and not fields["nci"]
                    assert len(energy) == len(z["X"]) and np.isfinite(energy).all()
                    Z = float(z["Z"])
                    assert scorer._agg_norm() == Z, (code, scorer._agg_norm(), Z)
                    X = torch.tensor(z["X"], dtype=torch.float32)
                    nci.NCI_ON = False
                    scorer._nci = None
                    with torch.no_grad():
                        full_base, base = scorer.score(X)
                        nci.NCI_ON = True
                        scorer._nci = runtime
                        full_cand, cand = scorer.score(X)
                    nci.NCI_ON = False
                    scorer._nci = None
                    assert all(torch.equal(base[k], cand[k]) for k in base if k != "Q6"), code
                    q6_diff = cand["Q6"] - base["Q6"]
                    contribution = cand.get("NCI", torch.zeros(len(X)))
                    delta = q6_diff + contribution
                    full_delta_error = float((full_cand-full_base-delta).abs().max())
                    assert full_delta_error < 1e-4, (code, full_delta_error)
                    q6_error = float(np.abs(base["Q6"].numpy()-z["terms_Q6"]/Z).max())
                    assert q6_error < 1e-4, (code, seed, q6_error)
                    assert float(q6_diff.max()) < 1e-4, "removing attractive wells cannot improve Q6"
                    scores = z["score"].copy()
                    changed = scores + delta.numpy()
                    ib, ic = int(scores.argmax()), int(changed.argmax())
                    by_type = {t: nci.evaluate({t: rt}, X).numpy() for t, rt in (runtime or {}).items()}
                    def details(i):
                        return dict(index=i, energy=float(energy[i]), nci_nats=float(contribution[i])*Z,
                                    q6_change_nats=float(q6_diff[i])*Z,
                                    nci_types_nats={t: float(v[i]) for t, v in by_type.items()})
                    row = dict(group=group, case=code, set=set_name, seed=seed, conformers=len(scores),
                               base_index=ib, cand_index=ic, changed_pick=ib != ic,
                               base_energy=float(energy[ib]), cand_energy=float(energy[ic]),
                               energy_change=float(energy[ic]-energy[ib]), floor=float(energy.min()),
                               base_regret=float(energy[ib]-energy.min()), cand_regret=float(energy[ic]-energy.min()),
                               q6_base_max_error=q6_error, full_delta_max_error=full_delta_error,
                               base=details(ib), candidate=details(ic),
                               max_score_change=float(np.abs(delta.numpy()).max()),
                               active_candidates={t: rt["n"] for t, rt in (runtime or {}).items()})
                    rows.append(row)
                    case_rows.append(row)
                    for key, val in dict(base_score=scores, candidate_score=changed, q6_delta=q6_diff.numpy(),
                                         nci=contribution.numpy(), energy=energy).items():
                        arrays[f"{code}__s{seed}__{key}"] = val
            cases[code] = dict(group=group, changed_picks=sum(r["changed_pick"] for r in case_rows),
                               base_mean=float(np.mean([r["base_energy"] for r in case_rows])),
                               candidate_mean=float(np.mean([r["cand_energy"] for r in case_rows])),
                               mean_change=float(np.mean([r["energy_change"] for r in case_rows])),
                               base_median=float(np.median([r["base_energy"] for r in case_rows])),
                               candidate_median=float(np.median([r["cand_energy"] for r in case_rows])))
            print(code, json.dumps(cases[code]), flush=True)
    checks = dict(pools=len(rows), cases=len(cases), q6_base_max_error=max(r["q6_base_max_error"] for r in rows),
                  full_delta_max_error=max(r["full_delta_max_error"] for r in rows),
                  changed_picks=sum(r["changed_pick"] for r in rows),
                  mean_energy_change=float(np.mean([r["energy_change"] for r in rows])),
                  better_cases_over_10=sum(r["mean_change"] < -10 for r in cases.values()),
                  worse_cases_over_10=sum(r["mean_change"] > 10 for r in cases.values()))
    report = dict(checks=checks, cases=cases, pools=rows, cohort=COHORT, elapsed_s=time.perf_counter()-start,
                  source=tree_hash(), base_db_sha256=fit["base_sha256"], candidate_db_sha256=fit["candidate_sha256"],
                  scope="selected-case ranking only; exact Q6/NCI delta on stored total scores, unchanged assignment terms and bend prior; no solves/new energies; not dev80 effect estimate")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    np.savez_compressed(out.with_suffix(".npz"), **arrays)
    pd.DataFrame([{k: v for k, v in r.items() if k not in ("base", "candidate", "active_candidates")} for r in rows]).to_csv(out.with_suffix(".csv"), index=False)
    print(json.dumps(checks), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    setup()
    run(a.out)


if __name__ == "__main__":
    main()
