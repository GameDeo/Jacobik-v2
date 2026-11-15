"""P5: zip each group into data/_archive/<group>.zip (with MANIFEST.txt), verify, then delete the originals."""
import shutil, zipfile
from pathlib import Path
D = Path(r"E:\Dev\Projects\jacobik_2026-10-02_full\data")
E = D / "energy_scale_refit"
G = {
 "energy_scale_v6": [D / "energy_scale"],
 "q6_ablation": [D / "q6_ablation"],
 "benchmark_runs_2026-10-02_to_15": [D / "benchmark_runs" / x for x in ("2026-10-02", "2026-10-03", "2026-10-06", "2026-10-13", "2026-10-15_refit1")],
 "pools_abandoned_arms": [E / "pools" / x for x in ("etkdg_u1", "etkdg_u1_mass", "etkdg_u1_lean", "etkdg_u1_lean_m13", "etkdg_u1_lean_pmf", "etkdg_u1_lean_contact", "etkdg_u1_lean_ez", "etkdg_u1_lean_sur")],
 "energy_scale_refit_old_analyses": [E / x for x in ("curvature", "curvature_solve", "q6diag", "validity", "ensemble_stats", "start_error", "_xtbtmp")],
 "refit_superseded_dbs": [D / "refit" / x for x in ("final_no_state", "final_run1_m_sub_early", "final_run2")],
}
out = D / "_archive"; out.mkdir(exist_ok=True)
for g, paths in G.items():
    paths = [p for p in paths if p.exists()]
    if not paths:
        continue
    files = [f for p in paths for f in ([p] if p.is_file() else sorted(p.rglob("*"))) if f.is_file()]
    z = out / f"{g}.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("MANIFEST.txt", f"archived 2026-10-17 (cleanup P5), originals deleted\nroot: data/\n" +
                    "\n".join(f"{f.relative_to(D).as_posix()}\t{f.stat().st_size}" for f in files))
        for f in files:
            zf.write(f, f.relative_to(D).as_posix())
    with zipfile.ZipFile(z) as zf:
        assert zf.testzip() is None and len(zf.namelist()) == len(files) + 1
    for p in paths:
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    print(f"{g}: {len(files)} files -> {z.stat().st_size / 1e6:.0f} MB", flush=True)
