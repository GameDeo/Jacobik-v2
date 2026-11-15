"""Compatibility CLI for the production extraction entry point in ``src/fit``; adds the tail panel to the target parents."""
import sys

from fit import typing_reextract as _impl
from tools.jgen.bench.panels import TAIL

configure = _impl.configure  # paths are CLI arguments now (--data-root --out-dir --frame --locked)
MODES = _impl.MODES
SPLITS = _impl.SPLITS
run = _impl.run


def target_codes():
    return _impl.target_codes(extra=TAIL)


def main():
    if "--extra-codes" not in sys.argv:
        sys.argv += ["--extra-codes", *TAIL]
    _impl.main()


if __name__ == "__main__":
    main()
