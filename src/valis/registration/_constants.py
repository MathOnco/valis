"""Constants, default parameters, and imports shared across the registration package."""

import logging
from typing import Optional, Union

logging.basicConfig(level=logging.WARNING)

import torch
import kornia
import einops

import traceback
import re
import os
import numpy as np
import pathlib
from skimage import transform, exposure, filters
from skimage import color as skcolor
from time import time
import tqdm
import pandas as pd
import pickle
import colour
import pyvips
from scipy import ndimage
import shapely
from copy import deepcopy
from pprint import pformat
import json
from colorama import Fore
from itertools import chain
import cv2
import matplotlib.pyplot as plt

from .. import feature_matcher
from .. import serial_rigid
from .. import feature_detectors
from .. import non_rigid_registrars
from .. import valtils
from .. import preprocessing
from .. import slide_tools
from .. import slide_io
from .. import viz
from .. import warp_tools
from .. import serial_non_rigid

logger = logging.getLogger(__name__)

pyvips.cache_set_max(0)

# Destination directories #
CONVERTED_IMG_DIR = "images"
PROCESSED_IMG_DIR = "processed"
RIGID_REG_IMG_DIR = "rigid_registration"
NON_RIGID_REG_IMG_DIR = "non_rigid_registration"
DEFORMATION_FIELD_IMG_DIR = "deformation_fields"
OVERLAP_IMG_DIR = "overlaps"
REG_RESULTS_DATA_DIR = "data"
MICRO_REG_DIR = "micro_registration"
DISPLACEMENT_DIRS = os.path.join(REG_RESULTS_DATA_DIR, "displacements")
MASK_DIR = "masks"

# Default image processing #
DEFAULT_BRIGHTFIELD_CLASS = preprocessing.OD
DEFAULT_BRIGHTFIELD_PROCESSING_ARGS = {
    "adaptive_eq": False
}  # {'c': preprocessing.DEFAULT_COLOR_STD_C, "h": 0}
DEFAULT_FLOURESCENCE_CLASS = preprocessing.ChannelGetter
DEFAULT_FLOURESCENCE_PROCESSING_ARGS = {"channel": "dapi", "adaptive_eq": True}
DEFAULT_NORM_METHOD = "img_stats"

# Default rigid registration parameters #
DEFAULT_FD = feature_detectors.VggFD
DEFAULT_TRANSFORM_CLASS = transform.SimilarityTransform

try:
    DEFAULT_MATCHER = feature_matcher.LightGlueMatcher(
        match_filter_method=feature_matcher.DEFAULT_RANSAC_NAME,
        feature_detector=feature_detectors.DiskFD(),
    )
except ImportError:
    DEFAULT_MATCHER = feature_matcher.Matcher(
        match_filter_method=feature_matcher.DEFAULT_RANSAC_NAME,
        feature_detector=feature_detectors.VggFD(),
    )

DEFAULT_MATCHER_FOR_SORTING = feature_matcher.Matcher(
    match_filter_method=feature_matcher.DEFAULT_RANSAC_NAME,
    feature_detector=feature_detectors.VggFD(),
)
DEFAULT_SIMILARITY_METRIC = "n_matches"
DEFAULT_AFFINE_OPTIMIZER_CLASS = None
DEFAULT_MAX_PROCESSED_IMG_SIZE = 512
DEFAULT_MAX_IMG_DIM = 1024
DEFAULT_THUMBNAIL_SIZE = 512
DEFAULT_MAX_NON_RIGID_REG_SIZE = 2048
DEFAULT_MAX_MICRO_REG_SIZE = 4096
DEFAULT_MIN_RIGID_MATCHES = (
    0  # 0 disables the safeguard; opt-in via `min_rigid_matches`
)

# Tiled non-rigid registration arguments
TILER_THRESH_GB = 10
DEFAULT_NR_TILE_WH = 512

# Rigid registration kwarg keys #
AFFINE_OPTIMIZER_KEY = "affine_optimizer"
TRANSFORMER_KEY = "transformer"
SIM_METRIC_KEY = "similarity_metric"
FD_KEY = "feature_detector"
MATCHER_KEY = "matcher"
MATCHER_FOR_SORTING_KEY = "matcher_for_sorting"
NAME_KEY = "name"
IMAGES_ORDERD_KEY = "imgs_ordered"
REF_IMG_KEY = "reference_img_f"
QT_EMMITER_KEY = "qt_emitter"
TFORM_SRC_SHAPE_KEY = "transformation_src_shape_rc"
TFORM_DST_SHAPE_KEY = "transformation_dst_shape_rc"
TFORM_MAT_KEY = "M"
CHECK_REFLECT_KEY = "check_for_reflections"
MIN_RIGID_MATCHES_KEY = "min_rigid_matches"

# Rigid registration kwarg keys #
NON_RIGID_REG_CLASS_KEY = "non_rigid_reg_class"
NON_RIGID_REG_PARAMS_KEY = "non_rigid_reg_params"
NON_RIGID_USE_XY_KEY = "moving_to_fixed_xy"
NON_RIGID_COMPOSE_KEY = "compose_transforms"

# Default non-rigid registration parameters #
DEFAULT_NON_RIGID_CLASS = non_rigid_registrars.OpticalFlowWarper()
DEFAULT_NON_RIGID_KWARGS = {}

# Cropping options
import sys as _sys

if _sys.version_info >= (3, 11):
    from enum import StrEnum as _StrEnum
else:
    from enum import Enum as _Enum

    class _StrEnum(str, _Enum):
        pass


class CropMode(_StrEnum):
    """How to crop registered images.

    Being a StrEnum, string literals ("overlap", "reference", "all") are still
    accepted wherever a CropMode is expected, so existing code is unaffected.
    """

    OVERLAP = "overlap"
    """Crop to the area where all images overlap."""
    REFERENCE = "reference"
    """Crop to the area overlapping with the reference image."""
    NONE = "all"
    """No cropping — use all pixels."""


# Keep module-level aliases for backward compatibility
CROP_OVERLAP = CropMode.OVERLAP
CROP_REF = CropMode.REFERENCE
CROP_NONE = CropMode.NONE

DEFAULT_COMPRESSION = pyvips.enums.ForeignTiffCompression.DEFLATE
# Messages
WARP_ANNO_MSG = "Warping annotations"
CONVERT_MSG = "Converting images"
DENOISE_MSG = "Denoising images"
PROCESS_IMG_MSG = "Processing images"
NORM_IMG_MSG = "Normalizing images"
TRANSFORM_MSG = "Finding rigid transforms"
PREP_NON_RIGID_MSG = "Preparing images for non-rigid registration"
MEASURE_MSG = "Measuring error"
SAVING_IMG_MSG = "Saving images"

PROCESS_IMG_MSG, NORM_IMG_MSG, DENOISE_MSG = valtils.pad_strings(
    [PROCESS_IMG_MSG, NORM_IMG_MSG, DENOISE_MSG]
)
