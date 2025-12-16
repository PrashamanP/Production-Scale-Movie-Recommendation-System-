"""
Helper module to import classes from the `pop-gen` package whose name
contains a hyphen (not a valid Python identifier).
"""

from importlib import util
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
POP_GEN_DIR = CURRENT_DIR / "pop-gen"
INFER_PATH = POP_GEN_DIR / "infer_popgen.py"

_SPEC = util.spec_from_file_location("models.pop_gen_infer", INFER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not locate PopGenreRecommender at {INFER_PATH}")

_MODULE = util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

PopGenreRecommender = _MODULE.PopGenreRecommender

__all__ = ["PopGenreRecommender"]
