__version__ = "1.2.0"

import sys as _sys
import warnings as _warnings

# Guard against the known import-order segfault: valis must be imported before
# any pytorch-related package (torch, torchvision, kornia, etc.).  If those
# packages are already in sys.modules they must have been imported beforehand,
# which is the unsafe ordering.  Raise a clear error instead of segfaulting.
# (When torch is not installed this guard is never triggered.)
_TORCH_RELATED = {"torch", "torchvision", "kornia", "einops", "timm"}
_already_imported = _TORCH_RELATED.intersection(_sys.modules)
if _already_imported:
    raise ImportError(
        "valis must be imported before pytorch-related packages "
        f"({', '.join(sorted(_already_imported))}). "
        "Move 'import valis' to the top of your script, before any torch/kornia imports."
    )

from . import affine_optimizer
from . import feature_detectors
from . import feature_matcher
from . import non_rigid_registrars
from . import preprocessing
from . import registration
from . import serial_non_rigid
from . import serial_rigid
from . import slide_io
from . import slide_tools
from . import valtils
from . import viz
from . import warp_tools
from . import micro_rigid_registrar

__all__ = [
    "affine_optimizer",
    "feature_detectors",
    "feature_matcher",
    "non_rigid_registrars",
    "preprocessing",
    "registration",
    "serial_non_rigid",
    "serial_rigid",
    "slide_io",
    "slide_tools",
    "valtils",
    "viz",
    "warp_tools",
    "micro_rigid_registrar",
]
