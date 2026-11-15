"""jdev -- one entry point for routine Jacobik development tasks. See dev/tools/jdev/README.md or `python -m tools.jdev -h`.

The one-off modules that stay under ``research/bench/tools/jdev`` remain importable as ``tools.jdev.*`` (path extension)."""
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
