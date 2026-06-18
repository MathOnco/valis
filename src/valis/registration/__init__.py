"""Registration package.

This package replaces the monolithic ``registration.py`` module.  All public
names are re-exported here so that existing code such as::

    from valis.registration import Valis, Slide, load_registrar
    from valis import registration; registration.CROP_OVERLAP

continues to work unchanged.

Sub-module layout
-----------------
_constants.py   Module-level constants, default parameters, and shared imports.
state.py        ``CropMode``, ``DisplacementField``, ``RegistrationConfig``,
                ``load_registrar``.
slide.py        ``Slide`` class.
pipeline.py     ``Valis`` class.
"""

# Re-export everything from _constants so ``registration.CROP_OVERLAP`` etc. work
from ._constants import (
    CONVERTED_IMG_DIR,
    PROCESSED_IMG_DIR,
    RIGID_REG_IMG_DIR,
    NON_RIGID_REG_IMG_DIR,
    DEFORMATION_FIELD_IMG_DIR,
    OVERLAP_IMG_DIR,
    REG_RESULTS_DATA_DIR,
    MICRO_REG_DIR,
    DISPLACEMENT_DIRS,
    MASK_DIR,
    DEFAULT_BRIGHTFIELD_CLASS,
    DEFAULT_BRIGHTFIELD_PROCESSING_ARGS,
    DEFAULT_FLOURESCENCE_CLASS,
    DEFAULT_FLOURESCENCE_PROCESSING_ARGS,
    DEFAULT_NORM_METHOD,
    DEFAULT_FD,
    DEFAULT_TRANSFORM_CLASS,
    DEFAULT_MATCHER,
    DEFAULT_MATCHER_FOR_SORTING,
    DEFAULT_SIMILARITY_METRIC,
    DEFAULT_AFFINE_OPTIMIZER_CLASS,
    DEFAULT_MAX_PROCESSED_IMG_SIZE,
    DEFAULT_MAX_IMG_DIM,
    DEFAULT_THUMBNAIL_SIZE,
    DEFAULT_MAX_NON_RIGID_REG_SIZE,
    DEFAULT_MAX_MICRO_REG_SIZE,
    TILER_THRESH_GB,
    DEFAULT_NR_TILE_WH,
    AFFINE_OPTIMIZER_KEY,
    TRANSFORMER_KEY,
    SIM_METRIC_KEY,
    FD_KEY,
    MATCHER_KEY,
    MATCHER_FOR_SORTING_KEY,
    NAME_KEY,
    IMAGES_ORDERD_KEY,
    REF_IMG_KEY,
    QT_EMMITER_KEY,
    TFORM_SRC_SHAPE_KEY,
    TFORM_DST_SHAPE_KEY,
    TFORM_MAT_KEY,
    CHECK_REFLECT_KEY,
    NON_RIGID_REG_CLASS_KEY,
    NON_RIGID_REG_PARAMS_KEY,
    NON_RIGID_USE_XY_KEY,
    NON_RIGID_COMPOSE_KEY,
    DEFAULT_NON_RIGID_CLASS,
    DEFAULT_NON_RIGID_KWARGS,
    DEFAULT_COMPRESSION,
    CropMode,
    CROP_OVERLAP,
    CROP_REF,
    CROP_NONE,
    WARP_ANNO_MSG,
    CONVERT_MSG,
    DENOISE_MSG,
    PROCESS_IMG_MSG,
    NORM_IMG_MSG,
    TRANSFORM_MSG,
    PREP_NON_RIGID_MSG,
    MEASURE_MSG,
    SAVING_IMG_MSG,
)

# Re-export shared types
from .state import (
    DisplacementField,
    RegistrationConfig,
    load_registrar,
)

# Re-export the two main classes
from .slide import Slide
from .pipeline import Valis

__all__ = [
    # Main classes
    "Valis",
    "Slide",
    # Support classes
    "CropMode",
    "DisplacementField",
    "RegistrationConfig",
    # Functions
    "load_registrar",
    # Constants — directory names
    "CONVERTED_IMG_DIR",
    "PROCESSED_IMG_DIR",
    "RIGID_REG_IMG_DIR",
    "NON_RIGID_REG_IMG_DIR",
    "DEFORMATION_FIELD_IMG_DIR",
    "OVERLAP_IMG_DIR",
    "REG_RESULTS_DATA_DIR",
    "MICRO_REG_DIR",
    "DISPLACEMENT_DIRS",
    "MASK_DIR",
    # Constants — crop modes (string aliases kept for backward compat)
    "CROP_OVERLAP",
    "CROP_REF",
    "CROP_NONE",
    # Constants — defaults
    "DEFAULT_BRIGHTFIELD_CLASS",
    "DEFAULT_BRIGHTFIELD_PROCESSING_ARGS",
    "DEFAULT_FLOURESCENCE_CLASS",
    "DEFAULT_FLOURESCENCE_PROCESSING_ARGS",
    "DEFAULT_NORM_METHOD",
    "DEFAULT_FD",
    "DEFAULT_TRANSFORM_CLASS",
    "DEFAULT_MATCHER",
    "DEFAULT_MATCHER_FOR_SORTING",
    "DEFAULT_SIMILARITY_METRIC",
    "DEFAULT_AFFINE_OPTIMIZER_CLASS",
    "DEFAULT_MAX_PROCESSED_IMG_SIZE",
    "DEFAULT_MAX_IMG_DIM",
    "DEFAULT_THUMBNAIL_SIZE",
    "DEFAULT_MAX_NON_RIGID_REG_SIZE",
    "DEFAULT_MAX_MICRO_REG_SIZE",
    "TILER_THRESH_GB",
    "DEFAULT_NR_TILE_WH",
    "DEFAULT_NON_RIGID_CLASS",
    "DEFAULT_NON_RIGID_KWARGS",
    "DEFAULT_COMPRESSION",
    # Constants — kwarg keys
    "AFFINE_OPTIMIZER_KEY",
    "TRANSFORMER_KEY",
    "SIM_METRIC_KEY",
    "FD_KEY",
    "MATCHER_KEY",
    "MATCHER_FOR_SORTING_KEY",
    "NAME_KEY",
    "IMAGES_ORDERD_KEY",
    "REF_IMG_KEY",
    "QT_EMMITER_KEY",
    "TFORM_SRC_SHAPE_KEY",
    "TFORM_DST_SHAPE_KEY",
    "TFORM_MAT_KEY",
    "CHECK_REFLECT_KEY",
    "NON_RIGID_REG_CLASS_KEY",
    "NON_RIGID_REG_PARAMS_KEY",
    "NON_RIGID_USE_XY_KEY",
    "NON_RIGID_COMPOSE_KEY",
    # Constants — messages
    "WARP_ANNO_MSG",
    "CONVERT_MSG",
    "DENOISE_MSG",
    "PROCESS_IMG_MSG",
    "NORM_IMG_MSG",
    "TRANSFORM_MSG",
    "PREP_NON_RIGID_MSG",
    "MEASURE_MSG",
    "SAVING_IMG_MSG",
]
