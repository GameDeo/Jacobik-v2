"""Archive the verified 2026-10-18 DB leftovers, without deleting source files.

Each archive has a size/SHA256 manifest and is read back before completion.
Remove the originals separately, after reviewing the report.
"""
import hashlib
import json
from pathlib import Path
import zipfile

REPO = Path(__file__).resolve().parents[4]


def main():
    final = REPO / "data/refit/final"
    groups = {
        "refit_addon_dbs_2026-10-18": [
            final / f"constraints_refit_q6p5_{kind}{suffix}"
            for kind in ("m13", "bvs", "nci") for suffix in (".json", ".meta.json")
        ],
        "v6db_eval_2026-10-18": [REPO / "dbs/v6db_eval"],
        "refit_final_verify_2026-10-18": [REPO / "data/refit/final_verify"],
    }
    out = REPO / "data/_archive"
    out.mkdir(exist_ok=True)
    for group, roots in groups.items():
        files = sorted(f for p in roots for f in ([p] if p.is_file() else p.rglob("*")) if f.is_file())
        if not files:
            continue
        archive = out / f"{group}.zip"
        if archive.exists():
            raise FileExistsError(archive)
        manifest = []
        for p in files:
            with p.open("rb") as fh:
                digest = hashlib.file_digest(fh, "sha256").hexdigest()
            manifest.append(dict(path=p.relative_to(REPO).as_posix(), bytes=p.stat().st_size, sha256=digest))
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            z.writestr("MANIFEST.txt", "2026-10-18; paths relative to repository root\n" + json.dumps(manifest, indent=2))
            for item in manifest:
                z.write(REPO / item["path"], item["path"])
        with zipfile.ZipFile(archive) as z:
            assert len(z.namelist()) == len(manifest) + 1
            for item in manifest:
                info = z.getinfo(item["path"])
                assert info.file_size == item["bytes"]
                with z.open(item["path"]) as fh:
                    assert hashlib.file_digest(fh, "sha256").hexdigest() == item["sha256"]
        print(f"VERIFIED {archive}: {len(files)} files, {archive.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
