"""Core package initialization for the agrostats project."""

from pathlib import Path
import sys


# Ensure the repository's src/ directory is importable when running modules directly.
_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


__all__ = [
    "io",
    "normalize",
    "features",
    "validate",
    "train",
    "utils",
]


from . import features, io, normalize, train, utils, validate  # noqa: E402  (import after sys.path tweak)
