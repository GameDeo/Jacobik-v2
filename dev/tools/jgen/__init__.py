"""jgen runtime modules the dev tools stand on (bench runner, panels, inputs). The analysis scripts stay under
``research/bench/tools/jgen`` and remain importable as ``tools.jgen.*`` (path extension)."""
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
