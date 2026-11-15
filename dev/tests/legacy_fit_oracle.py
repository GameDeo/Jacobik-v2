"""Reference oracle for the D2 pipeline tests: the pre-D2 serial ``fit_database`` (master d0395ca, src/fit/build_db.py),
kept verbatim except that it lives here. ``tests/test_d2_pipeline.py`` compares the shared-pool ``fit`` stage with it on the
same merged observations."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

PRODUCTION_QUANTITIES = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "QO", "Q6P"]


def _read_observations(obs_dir: Path, quantity: str):
    import pandas as pd

    # Match the refit's canonical concatenation order. QO's seeded GMM
    # initialization is sensitive to row order, even though each key's seed is
    # fixed. Alphabetical order would put LNf before X1.
    files = [obs_dir / f"observations_{quantity}_{split}.parquet"
             for split in ("X1", "X2", "X3", "LNf")]
    files = [p for p in files if p.exists()]
    if not files:
        raise FileNotFoundError(f"No observations_{quantity}_*.parquet files in {obs_dir}")
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def fit_database(obs_dir: Path, hist_dir: Path, *, workers: int, quantities: Sequence[str] | None = None,
                 charge_bad: set[str] | None = None, locked: set[str] | None = None,
                 work_dir: Path | None = None, production_db: dict | None = None,
                 eta2_obs_dir: Path):
    """Fit the database from archived raw geometry observations and histogram accumulators."""
    from fit import q1, q2, q3, q4, q5, q6, q6p, qo
    from fit.expander import expand_quantity
    import pandas as pd

    selected = set(quantities or PRODUCTION_QUANTITIES)
    db: dict[str, Any] = {}
    rows: dict[str, int] = {}
    eta2_reproduction = {"checked": False}
    work_dir = work_dir or (obs_dir / ".build_work")
    filtered_dir = work_dir / "filtered_observations"
    expanded_dir = work_dir / "expanded_observations"
    filtered_dir.mkdir(parents=True, exist_ok=True)
    expanded_dir.mkdir(parents=True, exist_ok=True)
    for q, module in (("Q1", q1), ("Q2", q2), ("Q3", q3), ("Q4", q4),
                      ("Q5", q5), ("Q6", q6)):
        if q not in selected:
            continue
        # Apply whole-parent flags before the single global ancestor reservoir,
        # matching the production expander's cap across the complete fit frame.
        from fit.sanity import filter_fit_frame
        filtered_count = 0
        filtered_parts = []
        for split in ("X1", "X2", "X3", "LNf"):
            src = obs_dir / f"observations_{q}_{split}.parquet"
            if not src.exists():
                continue
            df_part = filter_fit_frame(pd.read_parquet(src), charge_bad=charge_bad, locked=locked)
            filtered_count += len(df_part)
            filtered_parts.append(df_part)
        if not filtered_parts:
            raise FileNotFoundError(f"No observation tables for {q} in {obs_dir}")
        pd.concat(filtered_parts, ignore_index=True).to_parquet(
            filtered_dir / f"observations_{q}.parquet", index=False)
        expand_quantity(filtered_dir, expanded_dir, quantities=[q])
        rows[q] = filtered_count
        expanded_path = expanded_dir / f"expanded_{q}.parquet"
        if not expanded_path.exists():
            raise FileNotFoundError(f"No expanded observations for {q} from {obs_dir}")
        df = pd.read_parquet(expanded_path)
        if q in ("Q1", "Q2"):
            fitted = module.fit_from_observations(df, n_workers=workers)
        else:
            df.to_parquet(expanded_dir / f"expanded_{q}.parquet", index=False)
            fitted = module.fit_all(expanded_dir)
        db.update(fitted)

    if "QO" in selected:
        df = _read_observations(eta2_obs_dir, "QO")
        from fit.sanity import filter_fit_frame
        df = filter_fit_frame(df, charge_bad=charge_bad, locked=locked)
        rows["QO"] = int(len(df))
        from fit import qo as _qo
        db.update(_qo.fit_from_observations(df, n_jobs=workers))

    if "Q6P" in selected:
        # q6hist_m13_*.npz is a distinct family and also matches the broad
        # q6hist_*.npz glob. Keep it out of the non-M13 contact fit.
        contact = sorted(p for p in hist_dir.glob("q6hist_*.npz")
                         if not p.name.startswith("q6hist_m13_"))
        m13 = sorted(hist_dir.glob("q6hist_m13_*.npz"))
        db.update(q6p.fit_from_observations(q6p.load_hist(contact), q6p.load_hist(m13)))
        rows["Q6P_histograms"] = len(contact) + len(m13)

    return db, rows, eta2_reproduction


