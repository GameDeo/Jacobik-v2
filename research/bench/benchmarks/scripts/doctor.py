"""Report dependency imports and platform support without running a benchmark."""
import importlib
import importlib.metadata
import platform
import shutil
import sys


def main():
    print(f"Python: {sys.version.split()[0]}; platform: {platform.platform()}")
    failed = []
    for module, distribution in [
        ("architector", "architector"), ("xtb.interface", "xtb"),
        ("openbabel.openbabel", "openbabel"), ("pynauty", "pynauty"),
        ("ase", "ase"), ("mendeleev", "mendeleev"), ("rdkit", "rdkit"),
    ]:
        try:
            importlib.import_module(module)
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                version = "version unavailable (conda package)"
            print(f"OK {module}: {version}")
        except Exception as exc:
            failed.append(module)
            print(f"FAIL {module}: {type(exc).__name__}: {exc}")
    print(f"CREST (optional; disabled in protocol): {shutil.which('crest') or 'not installed'}")
    if failed:
        print("The Architector runtime is not ready. Use the Linux default environment.")
        if sys.platform == "win32":
            print("Windows audit environment intentionally excludes Architector; conda-forge pynauty has no win-64 build.")
        return 1
    print("Dependency imports pass. Run the smoke task to verify a real build.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
