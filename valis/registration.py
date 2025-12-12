"""
Classes and functions to register a collection of images
"""
import logging
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
from colorama import Fore
import jpype

from . import feature_matcher
from . import serial_rigid
from . import feature_detectors
from . import non_rigid_registrars
from . import valtils
from . import preprocessing
from . import slide_tools
from . import slide_io
from . import viz
from . import warp_tools
from . import serial_non_rigid

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
DEFAULT_BRIGHTFIELD_PROCESSING_ARGS = {"adaptive_eq": False} #{'c': preprocessing.DEFAULT_COLOR_STD_C, "h": 0}
DEFAULT_FLOURESCENCE_CLASS = preprocessing.ChannelGetter
DEFAULT_FLOURESCENCE_PROCESSING_ARGS = {"channel": "dapi", "adaptive_eq": True}
DEFAULT_NORM_METHOD = "img_stats"

# Default rigid registration parameters #
DEFAULT_FD = feature_detectors.VggFD
DEFAULT_TRANSFORM_CLASS = transform.SimilarityTransform
DEFAULT_MATCHER = feature_matcher.LightGlueMatcher(match_filter_method=feature_matcher.DEFAULT_RANSAC_NAME, feature_detector=feature_detectors.DiskFD())
DEFAULT_MATCHER_FOR_SORTING = feature_matcher.Matcher(match_filter_method=feature_matcher.DEFAULT_RANSAC_NAME, feature_detector=feature_detectors.VggFD())
DEFAULT_SIMILARITY_METRIC = "n_matches"
DEFAULT_AFFINE_OPTIMIZER_CLASS = None
DEFAULT_MAX_PROCESSED_IMG_SIZE = 512
DEFAULT_MAX_IMG_DIM = 1024
DEFAULT_THUMBNAIL_SIZE = 512
DEFAULT_MAX_NON_RIGID_REG_SIZE = 2048
DEFAULT_MAX_MICRO_REG_SIZE = 4096

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

# Rigid registration kwarg keys #
NON_RIGID_REG_CLASS_KEY = "non_rigid_reg_class"
NON_RIGID_REG_PARAMS_KEY = "non_rigid_reg_params"
NON_RIGID_USE_XY_KEY = "moving_to_fixed_xy"
NON_RIGID_COMPOSE_KEY = "compose_transforms"

# Default non-rigid registration parameters #
DEFAULT_NON_RIGID_CLASS = non_rigid_registrars.OpticalFlowWarper()
DEFAULT_NON_RIGID_KWARGS = {}

# Cropping options
CROP_OVERLAP = "overlap"
CROP_REF = "reference"
CROP_NONE = "all"

DEFAULT_COMPRESSION=pyvips.enums.ForeignTiffCompression.DEFLATE
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

PROCESS_IMG_MSG, NORM_IMG_MSG, DENOISE_MSG = valtils.pad_strings([PROCESS_IMG_MSG, NORM_IMG_MSG, DENOISE_MSG])


def init_jvm(jar=None, mem_gb=10):
    """Initialize JVM for BioFormats
    """
    slide_io.init_jvm(jar=None, mem_gb=10)


def kill_jvm():
    """Kill JVM for BioFormats
    """
    slide_io.kill_jvm()


def load_registrar(src_f):
    """Load a Valis object

    Parameters
    ----------
    src_f : string
        Path to pickled Valis object

    Returns
    -------
    registrar : Valis

        Valis object used for registration

    """
    try:
        registrar = pickle.load(open(src_f, 'rb'))
    except jpype._core.JVMNotRunning:
        init_jvm()
        registrar = pickle.load(open(src_f, 'rb'))

    data_dir = registrar.data_dir
    read_data_dir = os.path.split(src_f)[0]

    # If registrar has moved, will need to update paths to results
    # and displacement fields
    if data_dir != read_data_dir:
        new_dst_dir = os.path.split(read_data_dir)[0]
        registrar.dst_dir = new_dst_dir
        registrar.set_dst_paths()

        for slide_obj in registrar.slide_dict.values():
            slide_obj.update_results_img_paths()

    return registrar


class Slide(object):
    """Stores registration info and warps slides/points

    `Slide` is a class that stores registration parameters
    and other metadata about a slide. Once registration has been
    completed, `Slide` is also able warp the slide and/or points
    using the same registration parameters. Warped slides can be saved
    as ome.tiff images with valid ome-xml.

    Attributes
    ----------
    src_f : str
        Path to slide.

    image: ndarray
        Image to registered. Taken from a level in the image pyramid.
        However, image may be resized to fit within the `max_image_dim_px`
        argument specified when creating a `Valis` object.

    val_obj : Valis
        The "parent" object that registers all of the slide.

    reader : SlideReader
        Object that can read slides and collect metadata.

    original_xml : str
        Xml string created by bio-formats

    img_type : str
        Whether the image is "brightfield" or "fluorescence"

    is_rgb : bool
        Whether or not the slide is RGB.

    slide_shape_rc : tuple of int
        Dimensions of the largest resolution in the slide, in the form
        of (row, col).

    series : int
        Slide series to be read

    slide_dimensions_wh : ndarray
        Dimensions of all images in the pyramid (width, height).

    resolution : float
        Physical size of each pixel.

    units : str
        Physical unit of each pixel.

    name : str
        Name of the image. Usually `img_f` but with the extension removed.

    processed_img : ndarray
        Image used to perform registration

    rigid_reg_mask : ndarray
        Mask of convex hulls covering tissue in unregistered image.
        Could be used to mask `processed_img` before rigid registration

    non_rigid_reg_mask : ndarray
        Created by combining rigidly warped `rigid_reg_mask` in all
        other slides.

    stack_idx : int
        Position of image in sorted Z-stack

    processed_img_f : str
        Path to thumbnail of the processed `image`.

    rigid_reg_img_f : str
        Path to thumbnail of rigidly aligned `image`.

    non_rigid_reg_img_f : str
        Path to thumbnail of non-rigidly aligned `image`.

    processed_img_shape_rc : tuple of int
        Shape (row, col) of the processed image used to find the
        transformation parameters. Maximum dimension will be less or
        equal to the `max_processed_image_dim_px` specified when
        creating a `Valis` object. As such, this may be smaller than
        the image's shape.

    aligned_slide_shape_rc : tuple of int
        Shape (row, col) of aligned slide, based on the dimensions in the 0th
        level of they pyramid. In

    reg_img_shape_rc : tuple of int
        Shape (row, col) of the registered image

    M : ndarray
        Rigid transformation matrix that aligns `image` to the previous
        image in the stack. Found using the processed copy of `image`.

    bk_dxdy : ndarray
        (2, N, M) numpy array of pixel displacements in
        the x and y directions. dx = bk_dxdy[0], and dy=bk_dxdy[1]. Used
        to warp images. Found using the rigidly aligned version of the
        processed image.

    fwd_dxdy : ndarray
        Inverse of `bk_dxdy`. Used to warp points.

    _bk_dxdy_f : str
        Path to file containing bk_dxdy, if saved

    _fwd_dxdy_f : str
        Path to file containing fwd_dxdy, if saved

    _bk_dxdy_np : ndarray
        `bk_dxdy` as a numpy array. Only not None if `bk_dxdy` becomes
        associated with a file

    _fwd_dxdy_np : ndarray
        `fwd_dxdy` as a numpy array. Only not None if `fwd_dxdy` becomes
        associated with a file

    stored_dxdy : bool
        Whether or not the non-rigid displacements are saved in a file
        Should only occur if image is very large.

    fixed_slide : Slide
        Slide object to which this one was aligned.

    xy_matched_to_prev : ndarray
        Coordinates (x, y) of features in `image` that had matches in the
        previous image. Will have shape (N, 2)

    xy_in_prev : ndarray
        Coordinates (x, y) of features in the previous that had matches
        to those in `image`. Will have shape (N, 2)

    xy_matched_to_prev_in_bbox : ndarray
        Subset of `xy_matched_to_prev` that were within `overlap_mask_bbox_xywh`.
        Will either have shape (N, 2) or (M, 2), with M < N.

    xy_in_prev_in_bbox : ndarray
        Subset of `xy_in_prev` that were within `overlap_mask_bbox_xywh`.
        Will either have shape (N, 2) or (M, 2), with M < N.

    crop : str
        Crop method

    bg_px_pos_rc : tuple
        Position of pixel that has the background color

    bg_color : list, optional
        Color of background pixels

    is_empty : bool
        True if the image is empty (i.e. contains only 1 value)

    """

    def __init__(self, src_f, image, val_obj, reader, name=None):
        """
        Parameters
        ----------
        src_f : str
            Path to slide.

        image: ndarray
            Image to registered. Taken from a level in the image pyramid.
            However, image may be resized to fit within the `max_image_dim_px`
            argument specified when creating a `Valis` object.

        val_obj : Valis
            The "parent" object that registers all of the slide.

        reader : SlideReader
            Object that can read slides and collect metadata.

        name : str, optional
            Name of slide. If None, it will be `src_f` with the extension removed

        """

        self.src_f = src_f
        self.image = image
        self.val_obj = val_obj
        self.reader = reader

        # Metadata #
        self.is_rgb = reader.metadata.is_rgb
        self.img_type = reader.guess_image_type()
        self.slide_shape_rc = reader.metadata.slide_dimensions[0][::-1]
        self.series = reader.series
        self.slide_dimensions_wh = reader.metadata.slide_dimensions
        self.resolution = np.mean(reader.metadata.pixel_physical_size_xyu[0:2])
        self.units = reader.metadata.pixel_physical_size_xyu[2]
        self.original_xml = reader.metadata.original_xml

        if self.is_rgb and self.image.dtype != np.uint8:
            self.image = exposure.rescale_intensity(self.image, out_range=np.uint8)

        if name is None:
            name = valtils.get_name(src_f)

        self.name = name

        # To be filled in during registration #
        self.processed_img = None
        self.rigid_reg_mask = None
        self.non_rigid_reg_mask = None
        self.stack_idx = None

        self.aligned_slide_shape_rc = None
        self.processed_img_shape_rc = None
        self.reg_img_shape_rc = None
        self.M = None
        self.bk_dxdy = None
        self.fwd_dxdy = None

        self.stored_dxdy = False
        self._bk_dxdy_f = None
        self._fwd_dxdy_f = None
        self._bk_dxdy_np = None
        self._fwd_dxdy_np = None
        self.processed_img_f = None
        self.rigid_reg_img_f = None
        self.non_rigid_reg_img_f = None

        self.fixed_slide = None
        self.xy_matched_to_prev = None
        self.xy_in_prev = None
        self.xy_matched_to_prev_in_bbox = None
        self.xy_in_prev_in_bbox = None

        self.crop = None
        self.bg_px_pos_rc = (0, 0)
        self.bg_color = None

        self.is_empty = self.check_if_empty(image)

        self.processed_crop_bbox = None
        self.uncropped_processed_img_shape_rc = None
        self.rigid_cropped = False
        self.M_for_cropped = None
        self.rigid_reg_cropped_shape_rc = None

    def __repr__(self):
        repr_str = (f'<{self.__class__.__name__}, name = {self.name}>'
                    f', width={self.slide_dimensions_wh[0][0]}'
                    f', height={self.slide_dimensions_wh[0][1]}'
                    f', channels={self.reader.metadata.n_channels}'
                    f', levels={len(self.slide_dimensions_wh)}'
                    f', RGB={self.is_rgb}'
                    f', dtype={self.image.dtype}>'
                    )
        return (repr_str)

    def check_if_empty(self, img):
        """Check if the image is empty

        Return
        ------
        is_empty : bool
            Whether or not the image is empty

        """

        is_empty = img.min() == img.max()

        return is_empty

    def slide2image(self, level, series=None, xywh=None):
        """Convert slide to image

        Parameters
        -----------
        level : int
            Pyramid level

        series : int, optional
            Series number. Defaults to 0

        xywh : tuple of int, optional
            The region to be sliced from the slide. If None,
            then the entire slide will be converted. Otherwise
            xywh is the (top left x, top left y, width, height) of
            the region to be sliced.

        Returns
        -------
        img : ndarray
            An image of the slide or the region defined by xywh

        """

        img = self.reader.slide2image(level=level, series=series, xywh=xywh)

        return img

    def slide2vips(self, level, series=None, xywh=None):
        """Convert slide to pyvips.Image

        Parameters
        -----------
        level : int
            Pyramid level

        series : int, optional
            Series number. Defaults to 0

        xywh : tuple of int, optional
            The region to be sliced from the slide. If None,
            then the entire slide will be converted. Otherwise
            xywh is the (top left x, top left y, width, height) of
            the region to be sliced.

        Returns
        -------
        vips_slide : pyvips.Image
            An of the slide or the region defined by xywh

        """

        vips_img = self.reader.slide2vips(level=level, series=series, xywh=xywh)

        return vips_img

    def get_aligned_to_ref_slide_crop_xywh(self, ref_img_shape_rc, ref_M, scaled_ref_img_shape_rc=None):
        """Get bounding box used to crop slide to fit in reference image

        Parameters
        ----------
        ref_img_shape_rc : tuple of int
            shape of reference image used to find registration parameters, i.e. processed image)

        ref_M : ndarray
            Transformation matrix for the reference image

        scaled_ref_img_shape_rc : tuple of int, optional
            shape of scaled image with shape `img_shape_rc`, i.e. slide corresponding
            to the image used to find the registration parameters.

        Returns
        -------
        crop_xywh : tuple of int
            Bounding box of crop area (XYWH)

        mask : ndarray
            Mask covering reference image

        """

        mask , _ = self.val_obj.get_crop_mask(CROP_REF)

        if scaled_ref_img_shape_rc is not None:
            sxy = np.array([*scaled_ref_img_shape_rc[::-1]]) / np.array([*ref_img_shape_rc[::-1]])
        else:
            scaled_ref_img_shape_rc = ref_img_shape_rc
            sxy = np.ones(2)

        reg_txy = -ref_M[0:2, 2]
        slide_xywh = (*reg_txy*sxy, *scaled_ref_img_shape_rc[::-1])

        return slide_xywh, mask

    def get_overlap_crop_xywh(self, warped_img_shape_rc, scaled_warped_img_shape_rc=None):
        """Get bounding box used to crop slide to where all slides overlap

        Parameters
        ----------
        warped_img_shape_rc : tuple of int
            shape of registered image

        warped_scaled_img_shape_rc : tuple of int, optional
            shape of scaled registered image (i.e. registered slied)

        Returns
        -------
        crop_xywh : tuple of int
            Bounding box of crop area (XYWH)

        """
        mask , mask_bbox_xywh = self.val_obj.get_crop_mask(CROP_OVERLAP)

        if scaled_warped_img_shape_rc is not None:
            sxy = np.array([*scaled_warped_img_shape_rc[::-1]]) / np.array([*warped_img_shape_rc[::-1]])
        else:
            sxy = np.ones(2)

        to_slide_transformer = transform.SimilarityTransform(scale=sxy)
        overlap_bbox = warp_tools.bbox2xy(mask_bbox_xywh)
        scaled_overlap_bbox = to_slide_transformer(overlap_bbox)
        scaled_overlap_xywh = warp_tools.xy2bbox(scaled_overlap_bbox)

        scaled_overlap_xywh[2:] = np.ceil(scaled_overlap_xywh[2:])
        scaled_overlap_xywh = tuple(scaled_overlap_xywh.astype(int))

        return scaled_overlap_xywh, mask

    def get_crop_xywh(self, crop, out_shape_rc=None):
        """Get bounding box used to crop aligned slide

        Parameters
        ----------

        out_shape_rc : tuple of int, optional
            If crop is "reference", this should be the shape of scaled reference image, such
            as the unwarped slide that corresponds to the unwarped processed reference image.

            If crop is "overlap", this should be the shape of the registered slides.


        Returns
        -------
        crop_xywh : tuple of int
            Bounding box of crop area (XYWH)

        mask : ndarray
            Mask, before crop
        """

        ref_slide = self.val_obj.get_ref_slide()
        if crop == CROP_REF:
            transformation_shape_rc = np.array(ref_slide.processed_img_shape_rc)
            crop_xywh, mask = self.get_aligned_to_ref_slide_crop_xywh(ref_img_shape_rc=transformation_shape_rc,
                                                                      ref_M=ref_slide.M,
                                                                      scaled_ref_img_shape_rc=out_shape_rc)
        elif crop == CROP_OVERLAP:
            transformation_shape_rc = np.array(ref_slide.reg_img_shape_rc)
            crop_xywh, mask = self.get_overlap_crop_xywh(warped_img_shape_rc=transformation_shape_rc,
                                                         scaled_warped_img_shape_rc=out_shape_rc)

        return crop_xywh, mask

    def get_crop_method(self, crop):
        """Get string or logic defining how to crop the image
        """
        if crop is True:
            crop_method = self.crop
        else:
            crop_method = crop

        do_crop = crop_method in [CROP_REF, CROP_OVERLAP]

        if do_crop:
            return crop_method
        else:
            return False

    def get_bg_color_px_pos(self, cspace="Hunter LAB"):
        """Get position of pixel that has color used for background
        """
        if self.img_type == slide_tools.IHC_NAME:
            # RGB. Get brightest pixel
            mean_rgb, color_mask, filtered_label_counts, color_clusterer = preprocessing.find_dominant_colors(self.image, cspace=cspace, return_xy_clusterer=True)
            mean_jab = preprocessing.rgb2jab(mean_rgb, cspace=cspace)
            mean_jch = colour.models.Jab_to_JCh(mean_jab)

            # Find highest luminosity (L) and lowest colorfulness
            bg_idx = np.lexsort([mean_jch[:, 1], -mean_jch[:, 0]])[0] # Last column sorted 1st. Returns ascending order
            self.bg_color = mean_rgb[bg_idx, :]

        else:
            # IF. Get darkest pixel
            sum_img = self.image.sum(axis=2)
            bg_px = np.unravel_index(np.argmin(sum_img, axis=None), sum_img.shape)

            self.bg_px_pos_rc = bg_px
            self.bg_color = list(self.image[bg_px])

    def update_results_img_paths(self):
        n_digits = len(str(self.val_obj.size))
        stack_id = str.zfill(str(self.stack_idx), n_digits)

        self.processed_img_f = os.path.join(self.val_obj.processed_dir, self.name + ".png")
        self.rigid_reg_img_f = os.path.join(self.val_obj.reg_dst_dir, f"{stack_id}_f{self.name}.png")
        self.non_rigid_reg_img_f = os.path.join(self.val_obj.non_rigid_dst_dir, f"{stack_id}_f{self.name}.png")
        if self.stored_dxdy:
            bk_dxdy_f, fwd_dxdy_f = self.get_displacement_f()
            self._bk_dxdy_f = bk_dxdy_f
            self._fwd_dxdy_f = fwd_dxdy_f

    def get_displacement_f(self):
        bk_dxdy_f = os.path.join(self.val_obj.displacements_dir, f"{self.name}_bk_dxdy.tiff")
        fwd_dxdy_f = os.path.join(self.val_obj.displacements_dir, f"{self.name}_fwd_dxdy.tiff")

        return bk_dxdy_f, fwd_dxdy_f

    def get_bk_dxdy(self):
        if self._bk_dxdy_np is None and not self.stored_dxdy:
            return None

        elif self.stored_dxdy:
            bk_dxdy_f, _ = self.get_displacement_f()
            cropped_bk_dxdy = pyvips.Image.new_from_file(bk_dxdy_f)
            full_bk_dxdy = self.val_obj.pad_displacement(cropped_bk_dxdy,
                self.val_obj._full_displacement_shape_rc,
                self.val_obj._non_rigid_bbox)

        else:
            if np.any(self._bk_dxdy_np.shape[1:2] != self.val_obj._full_displacement_shape_rc):
                full_bk_dxdy = self.val_obj.pad_displacement(self._bk_dxdy_np,
                    self.val_obj._full_displacement_shape_rc,
                    self.val_obj._non_rigid_bbox)
            else:
                full_bk_dxdy = self._bk_dxdy_np

        return full_bk_dxdy


    def set_bk_dxdy(self, bk_dxdy):
        """
        Only set if an array
        """
        if not isinstance(bk_dxdy, pyvips.Image):
            self._bk_dxdy_np = bk_dxdy
        else:
            print(f"Cannot set bk_dxdy when data is type {type(bk_dxdy)}")

    bk_dxdy = property(fget=get_bk_dxdy,
                       fset=set_bk_dxdy,
                       doc="Get and set backwards displacements")

    def get_fwd_dxdy(self):
        if self._fwd_dxdy_np is None and not self.stored_dxdy:
            return None

        elif self.stored_dxdy:
            _, fwd_dxdy_f = self.get_displacement_f()
            cropped_fwd_dxdy = pyvips.Image.new_from_file(fwd_dxdy_f)
            full_fwd_dxdy = self.val_obj.pad_displacement(cropped_fwd_dxdy,
                self.val_obj._full_displacement_shape_rc,
                self.val_obj._non_rigid_bbox)

        else:
            if np.any(self._fwd_dxdy_np.shape[1:2] != self.val_obj._full_displacement_shape_rc):
                full_fwd_dxdy = self.val_obj.pad_displacement(self._fwd_dxdy_np,
                    self.val_obj._full_displacement_shape_rc,
                    self.val_obj._non_rigid_bbox)
            else:
                full_fwd_dxdy = self._fwd_dxdy_np

        return full_fwd_dxdy


    def set_fwd_dxdy(self, fwd_dxdy):
        if not isinstance(fwd_dxdy, pyvips.Image):
            self._fwd_dxdy_np = fwd_dxdy
        else:
            print(f"Cannot set fwd_dxdy when data is type {type(fwd_dxdy)}")

    fwd_dxdy = property(fget=get_fwd_dxdy,
                        fset=set_fwd_dxdy,
                        doc="Get forward displacements")

    def warp_img(self, img=None, non_rigid=True, crop=True, interp_method="bicubic"):
        """Warp an image using the registration parameters

        img : ndarray, optional
            The image to be warped. If None, then Slide.image
            will be warped.

        non_rigid : bool
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied.

        crop: bool, str
            How to crop the registered images. If `True`, then the same crop used
            when initializing the `Valis` object will be used. If `False`, the
            image will not be cropped. If "overlap", the warped slide will be
            cropped to include only areas where all images overlapped.
            "reference" crops to the area that overlaps with the reference image,
            defined by `reference_img_f` when initialzing the `Valis object`.

        interp_method : str
            Interpolation method used when warping slide. Default is "bicubic"

        Returns
        -------
        warped_img : ndarray
            Warped copy of `img`

        """

        if img is None:
            img = self.image

        if non_rigid:
            dxdy = self.bk_dxdy
        else:
            dxdy = None

        if isinstance(img, pyvips.Image):
            img_shape_rc = (img.height, img.width)
            img_dim = img.bands
        else:
            img_shape_rc = img.shape[0:2]
            img_dim = img.ndim

        ref_slide = self.val_obj.get_ref_slide()

        if self == ref_slide and crop == CROP_REF and np.all(warp_tools.get_shape(img)[0:2] == self.processed_img_shape_rc):
            # Save on computation time and avoid interpolation/rounding issues and return the original image
            return img

        if not np.all(img_shape_rc == self.processed_img_shape_rc):
            msg = ("scaling transformation for image with different shape. "
                   "However, without knowing all of other image's shapes, "
                   "the scaling may not be the same for all images, and so "
                   "may not overlap."
                   )
            valtils.print_warning(msg)
            same_shape = False
            img_scale_rc = np.array(img_shape_rc)/(np.array(self.processed_img_shape_rc))
            out_shape_rc = self.val_obj.get_aligned_slide_shape(img_scale_rc)

        else:
            same_shape = True
            out_shape_rc = self.reg_img_shape_rc

        if isinstance(crop, bool) or isinstance(crop, str):
            crop_method = self.get_crop_method(crop)
            if crop_method is not False:
                if crop_method == CROP_REF:
                    if not same_shape:
                        scaled_shape_rc = np.array(ref_slide.processed_img_shape_rc)*img_scale_rc
                    else:
                        scaled_shape_rc = ref_slide.processed_img_shape_rc
                elif crop_method == CROP_OVERLAP:
                    scaled_shape_rc = out_shape_rc

                bbox_xywh, _ = self.get_crop_xywh(crop=crop_method, out_shape_rc=scaled_shape_rc)
            else:
                bbox_xywh = None

        elif isinstance(crop[0], (int, float)) and len(crop) == 4:
            bbox_xywh = crop
        else:
            bbox_xywh = None

        if img_dim == self.image.ndim:
            bg_color = self.bg_color
        else:
            bg_color = None

        warped_img = \
            warp_tools.warp_img(img, M=self.M,
                                bk_dxdy=dxdy,
                                out_shape_rc=out_shape_rc,
                                transformation_src_shape_rc=self.processed_img_shape_rc,
                                transformation_dst_shape_rc=self.reg_img_shape_rc,
                                bbox_xywh=bbox_xywh,
                                bg_color=bg_color,
                                interp_method=interp_method)

        return warped_img

    def warp_img_from_to(self, img, to_slide_obj,
                         dst_slide_level=0, non_rigid=True, interp_method="bicubic", bg_color=None):

        """Warp an image from this slide onto another unwarped slide

        Note that if `img` is a labeled image/mask then it is recommended to set `interp_method` to "nearest"

        Parameters
        ----------
        img : ndarray, pyvips.Image
            Image to warp. Should be a scaled version of the same one used for registration

        to_slide_obj : Slide
            Slide to which the points will be warped. I.e. `xy`
            will be warped from this Slide to their position in
            the unwarped slide associated with `to_slide_obj`.

        dst_slide_level: int, tuple, optional
            Pyramid level of the slide/image that `img` will be warped on to

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied.

        """

        if np.issubdtype(type(dst_slide_level), np.integer):
            to_slide_src_shape_rc = to_slide_obj.slide_dimensions_wh[dst_slide_level][::-1]
            aligned_slide_shape = self.val_obj.get_aligned_slide_shape(dst_slide_level)
        else:

            to_slide_src_shape_rc = np.array(dst_slide_level)

            dst_scale_rc = (to_slide_src_shape_rc/np.array(to_slide_obj.processed_img_shape_rc))
            aligned_slide_shape = np.round(dst_scale_rc*np.array(to_slide_obj.reg_img_shape_rc)).astype(int)

        if non_rigid:
            from_bk_dxdy = self.bk_dxdy
            to_fwd_dxdy = to_slide_obj.fwd_dxdy

        else:
            from_bk_dxdy = None
            to_fwd_dxdy = None

        warped_img = \
            warp_tools.warp_img_from_to(img,
                                        from_M=self.M,
                                        from_transformation_src_shape_rc=self.processed_img_shape_rc,
                                        from_transformation_dst_shape_rc=self.reg_img_shape_rc,
                                        from_dst_shape_rc=aligned_slide_shape,
                                        from_bk_dxdy=from_bk_dxdy,
                                        to_M=to_slide_obj.M,
                                        to_transformation_src_shape_rc=to_slide_obj.processed_img_shape_rc,
                                        to_transformation_dst_shape_rc=to_slide_obj.reg_img_shape_rc,
                                        to_src_shape_rc=to_slide_src_shape_rc,
                                        to_fwd_dxdy=to_fwd_dxdy,
                                        bg_color=bg_color,
                                        interp_method=interp_method
                                        )

        return warped_img

    @valtils.deprecated_args(crop_to_overlap="crop")
    def warp_slide(self, level, non_rigid=True, crop=True,
                   src_f=None, interp_method="bicubic", reader=None):
        """Warp a slide using registration parameters

        Parameters
        ----------
        level : int
            Pyramid level to be warped

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied. Default is True

        crop: bool, str
            How to crop the registered images. If `True`, then the same crop used
            when initializing the `Valis` object will be used. If `False`, the
            image will not be cropped. If "overlap", the warped slide will be
            cropped to include only areas where all images overlapped.
            "reference" crops to the area that overlaps with the reference image,
            defined by `reference_img_f` when initialzing the `Valis object`.

        src_f : str, optional
           Path of slide to be warped. If None (the default), Slide.src_f
           will be used. Otherwise, the file to which `src_f` points to should
           be an alternative copy of the slide, such as one that has undergone
           processing (e.g. stain segmentation), has a mask applied, etc...

        interp_method : str
            Interpolation method used when warping slide. Default is "bicubic"

        """
        if src_f is None:
            src_f = self.src_f

        if non_rigid:
            bk_dxdy = self.bk_dxdy
        else:
            bk_dxdy = None

        if level != 0:
            if not np.issubdtype(type(level), np.integer):
                msg = "Need slide level to be an integer indicating pyramid level"
                valtils.print_warning(msg)
            aligned_slide_shape = self.val_obj.get_aligned_slide_shape(level)
        else:
            aligned_slide_shape = self.aligned_slide_shape_rc

        if isinstance(crop, bool) or isinstance(crop, str):
            crop_method = self.get_crop_method(crop)
            if crop_method is not False:
                if crop_method == CROP_REF:
                    ref_slide = self.val_obj.get_ref_slide()
                    scaled_aligned_shape_rc = ref_slide.slide_dimensions_wh[level][::-1]

                elif crop_method == CROP_OVERLAP:
                    scaled_aligned_shape_rc = aligned_slide_shape

                slide_bbox_xywh, _ = self.get_crop_xywh(crop=crop_method,
                                                        out_shape_rc=scaled_aligned_shape_rc)

                if crop_method == CROP_REF:
                    assert np.all(slide_bbox_xywh[2:] == scaled_aligned_shape_rc[::-1])
                    if src_f == self.src_f and self == ref_slide:
                        # Shouldn't need to warp, but do checks just in case
                        no_rigid = True
                        no_non_rigid = True
                        if self.M is not None:
                            sxy = (scaled_aligned_shape_rc/self.processed_img_shape_rc)[::-1]
                            scaled_txy = sxy*self.M[:2, 2]
                            no_transforms = all(self.M[:2, :2].reshape(-1) == [1, 0, 0, 1])
                            crop_to_origin = np.all(np.abs(slide_bbox_xywh[0:2] + scaled_txy) < 1)
                            no_rigid = no_transforms and crop_to_origin

                        if self.bk_dxdy is not None:
                            no_non_rigid = self.bk_dxdy.min() == 0 and self.bk_dxdy.max() == 0

                        if no_rigid and no_non_rigid:
                            # Don't need to warp, so return original reference image
                            ref_img = self.reader.slide2vips(level=level)
                            return ref_img

            else:
                slide_bbox_xywh = None

        elif isinstance(crop[0], (int, float)) and len(crop) == 4:
            slide_bbox_xywh = crop
        else:
            slide_bbox_xywh = None

        if src_f == self.src_f:
            bg_color = self.bg_color
        else:
            bg_color = None

        if reader is None:
            reader = self.reader

        warped_slide = slide_tools.warp_slide(src_f, M=self.M,
                                              transformation_src_shape_rc=self.processed_img_shape_rc,
                                              transformation_dst_shape_rc=self.reg_img_shape_rc,
                                              aligned_slide_shape_rc=aligned_slide_shape,
                                              dxdy=bk_dxdy, level=level, series=self.series,
                                              interp_method=interp_method,
                                              bbox_xywh=slide_bbox_xywh,
                                              bg_color=bg_color,
                                              reader=reader)
        return warped_slide

    def warp_and_save_slide(self, dst_f, level=0, non_rigid=True,
                            crop=True, src_f=None,
                            channel_names=None,
                            colormap=slide_io.CMAP_AUTO,
                            interp_method="bicubic",
                            tile_wh=None,
                            compression=DEFAULT_COMPRESSION,
                            Q=100,
                            pyramid=True,
                            reader=None):

        """Warp and save a slide

        Slides will be saved in the ome.tiff format.

        Parameters
        ----------
        dst_f : str
            Path to were the warped slide will be saved.

        level : int
            Pyramid level to be warped

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied. Default is True

        crop: bool, str
            How to crop the registered images. If `True`, then the same crop used
            when initializing the `Valis` object will be used. If `False`, the
            image will not be cropped. If "overlap", the warped slide will be
            cropped to include only areas where all images overlapped.
            "reference" crops to the area that overlaps with the reference image,
            defined by `reference_img_f` when initializing the `Valis object`.

        channel_names : list, optional
            List of channel names. If None, then Slide.reader
            will attempt to find the channel names associated with `src_f`.

        colormap : dict, optional
            Dictionary of channel colors, where the key is the channel name, and the value the color as rgb255.
            If None (default), the channel colors from `current_ome_xml_str` will be used, if available.
            If None, and there are no channel colors in the `current_ome_xml_str`, then no colors will be added

        src_f : str, optional
            Path of slide to be warped. If None (the default), Slide.src_f
            will be used. Otherwise, the file to which `src_f` points to should
            be an alternative copy of the slide, such as one that has undergone
            processing (e.g. stain segmentation), has a mask applied, etc...

        interp_method : str
            Interpolation method used when warping slide. Default is "bicubic"

        tile_wh : int, optional
            Tile width and height used to save image

        compression : str
            Compression method used to save ome.tiff. See pyips for more details.

        Q : int
            Q factor for lossy compression

        pyramid : bool
            Whether or not to save an image pyramid.
        """

        if src_f is None:
            src_f = self.src_f

        if reader is None:
            if src_f != self.src_f:
                slide_reader_cls = slide_io.get_slide_reader(src_f)
                reader = slide_reader_cls(src_f)
            else:
                reader = self.reader

        warped_slide = self.warp_slide(level=level,
                                       non_rigid=non_rigid,
                                       crop=crop,
                                       interp_method=interp_method,
                                       src_f=src_f,
                                       reader=reader)

        # Get ome-xml #
        ref_slide = self.val_obj.get_ref_slide()
        pixel_physical_size_xyu = ref_slide.reader.scale_physical_size(level)

        ome_xml_obj = slide_io.update_xml_for_new_img(img=warped_slide,
                                                      reader=reader,
                                                      level=level,
                                                      channel_names=channel_names,
                                                      colormap=colormap,
                                                      pixel_physical_size_xyu=pixel_physical_size_xyu)

        ome_xml = ome_xml_obj.to_xml()

        out_shape_wh = warp_tools.get_shape(warped_slide)[0:2][::-1]
        tile_wh = slide_io.get_tile_wh(reader=reader,
                                       level=level,
                                       out_shape_wh=out_shape_wh)

        slide_io.save_ome_tiff(warped_slide, dst_f=dst_f, ome_xml=ome_xml,
                               tile_wh=tile_wh, compression=compression,
                               Q=Q, pyramid=pyramid)


    def warp_xy(self, xy, M=None, slide_level=0, pt_level=0,
                non_rigid=True, crop=True):
        """Warp points using registration parameters

        Warps `xy` to their location in the registered slide/image

        Parameters
        ----------
        xy : ndarray
            (N, 2) array of points to be warped. Must be x,y coordinates

        slide_level: int, tuple, optional
            Pyramid level of the slide. Used to scale transformation matrices.
            Can also be the shape of the warped image (row, col) into which
            the points should be warped. Default is 0.

        pt_level: int, tuple, optional
            Pyramid level from which the points origingated. For example, if
            `xy` are from the centroids of cell segmentation performed on the
            full resolution image, this should be 0. Alternatively, the value can
            be a tuple of the image's shape (row, col) from which the points came.
            For example, if `xy` are  bounding box coordinates from an analysis on
            a lower resolution image, then pt_level is that lower resolution
            image's shape (row, col). Default is 0.

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied. Default is True.

        crop: bool, str
            Apply crop to warped points by shifting points to the mask's origin.
            Note that this can result in negative coordinates, but might be useful
            if wanting to draw the coordinates on the registered slide, such as
            annotation coordinates.

            If `True`, then the same crop used
            when initializing the `Valis` object will be used. If `False`, the
            image will not be cropped. If "overlap", the warped slide will be
            cropped to include only areas where all images overlapped.
            "reference" crops to the area that overlaps with the reference image,
            defined by `reference_img_f` when initialzing the `Valis object`.

        """
        if M is None:
            M = self.M

        if np.issubdtype(type(pt_level), np.integer):
            pt_dim_rc = self.slide_dimensions_wh[pt_level][::-1]
        else:
            pt_dim_rc = np.array(pt_level)

        if np.issubdtype(type(slide_level), np.integer):
            if slide_level != 0:
                if np.issubdtype(type(slide_level), np.integer):
                    aligned_slide_shape = self.val_obj.get_aligned_slide_shape(slide_level)
                else:
                    aligned_slide_shape = np.array(slide_level)
            else:
                aligned_slide_shape = self.aligned_slide_shape_rc
        else:
            aligned_slide_shape = np.array(slide_level)

        if non_rigid:
            fwd_dxdy = self.fwd_dxdy
        else:
            fwd_dxdy = None

        warped_xy = warp_tools.warp_xy(xy, M=M,
                                       transformation_src_shape_rc=self.processed_img_shape_rc,
                                       transformation_dst_shape_rc=self.reg_img_shape_rc,
                                       src_shape_rc=pt_dim_rc,
                                       dst_shape_rc=aligned_slide_shape,
                                       fwd_dxdy=fwd_dxdy)

        crop_method = self.get_crop_method(crop)
        if crop_method is not False:
            if crop_method == CROP_REF:
                ref_slide = self.val_obj.get_ref_slide()
                if isinstance(slide_level, int):
                    scaled_aligned_shape_rc = ref_slide.slide_dimensions_wh[slide_level][::-1]
                else:
                    if len(slide_level) == 2:
                        scaled_aligned_shape_rc = slide_level
            elif crop_method == CROP_OVERLAP:
                scaled_aligned_shape_rc = aligned_slide_shape

            crop_bbox_xywh, _ = self.get_crop_xywh(crop_method, scaled_aligned_shape_rc)
            warped_xy -= crop_bbox_xywh[0:2]

        return warped_xy

    def warp_xy_from_to(self, xy, to_slide_obj, src_slide_level=0, src_pt_level=0,
                        dst_slide_level=0, non_rigid=True):

        """Warp points from this slide to another unwarped slide

        Takes a set of points found in this unwarped slide, and warps them to
        their position in the unwarped "to" slide.

        Parameters
        ----------
        xy : ndarray
            (N, 2) array of points to be warped. Must be x,y coordinates

        to_slide_obj : Slide
            Slide to which the points will be warped. I.e. `xy`
            will be warped from this Slide to their position in
            the unwarped slide associated with `to_slide_obj`.

        src_pt_level: int, tuple, optional
            Pyramid level of the slide/image in which `xy` originated.
            For example, if `xy` are from the centroids of cell segmentation
            performed on the unwarped full resolution image, this should be 0.
            Alternatively, the value can be a tuple of the image's shape (row, col)
            from which the points came. For example, if `xy` are  bounding
            box coordinates from an analysis on a lower resolution image,
            then pt_level is that lower resolution image's shape (row, col).

        dst_slide_level: int, tuple, optional
            Pyramid level of the slide/image in to `xy` will be warped.
            Similar to `src_pt_level`, if `dst_slide_level` is an int then
            the points will be warped to that pyramid level. If `dst_slide_level`
            is the "to" image's shape (row, col), then the points will be warped
            to their location in an image with that same shape.

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied.

        """

        if np.issubdtype(type(src_pt_level), np.integer):
            src_pt_dim_rc = self.slide_dimensions_wh[src_pt_level][::-1]
        else:
            src_pt_dim_rc = np.array(src_pt_level)

        if np.issubdtype(type(dst_slide_level), np.integer):
            to_slide_src_shape_rc = to_slide_obj.slide_dimensions_wh[dst_slide_level][::-1]
        else:
            to_slide_src_shape_rc = np.array(dst_slide_level)

        if src_slide_level != 0:
            if np.issubdtype(type(src_slide_level), np.integer):
                aligned_slide_shape = self.val_obj.get_aligned_slide_shape(src_slide_level)
            else:
                aligned_slide_shape = np.array(src_slide_level)
        else:
            aligned_slide_shape = self.aligned_slide_shape_rc

        if non_rigid:
            src_fwd_dxdy = self.fwd_dxdy
            dst_bk_dxdy = to_slide_obj.bk_dxdy

        else:
            src_fwd_dxdy = None
            dst_bk_dxdy = None

        xy_in_unwarped_to_img = \
            warp_tools.warp_xy_from_to(xy=xy,
                                       from_M=self.M,
                                       from_transformation_dst_shape_rc=self.reg_img_shape_rc,
                                       from_transformation_src_shape_rc=self.processed_img_shape_rc,
                                       from_dst_shape_rc=aligned_slide_shape,
                                       from_src_shape_rc=src_pt_dim_rc,
                                       from_fwd_dxdy=src_fwd_dxdy,
                                       to_M=to_slide_obj.M,
                                       to_transformation_src_shape_rc=to_slide_obj.processed_img_shape_rc,
                                       to_transformation_dst_shape_rc=to_slide_obj.reg_img_shape_rc,
                                       to_src_shape_rc=to_slide_src_shape_rc,
                                       to_dst_shape_rc=aligned_slide_shape,
                                       to_bk_dxdy=dst_bk_dxdy
                                       )

        return xy_in_unwarped_to_img

    def warp_geojson(self, geojson_f, M=None, slide_level=0, pt_level=0,
                non_rigid=True, crop=True):
        """Warp geometry using registration parameters

        Warps geometries to their location in the registered slide/image

        Parameters
        ----------
        geojson_f : str
            Path to geojson file containing the annotation geometries. Assumes
            coordinates are in pixels.

        slide_level: int, tuple, optional
            Pyramid level of the slide. Used to scale transformation matrices.
            Can also be the shape of the warped image (row, col) into which
            the points should be warped. Default is 0.

        pt_level: int, tuple, optional
            Pyramid level from which the points origingated. For example, if
            `xy` are from the centroids of cell segmentation performed on the
            full resolution image, this should be 0. Alternatively, the value can
            be a tuple of the image's shape (row, col) from which the points came.
            For example, if `xy` are  bounding box coordinates from an analysis on
            a lower resolution image, then pt_level is that lower resolution
            image's shape (row, col). Default is 0.

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied. Default is True.

        crop: bool, str
            Apply crop to warped points by shifting points to the mask's origin.
            Note that this can result in negative coordinates, but might be useful
            if wanting to draw the coordinates on the registered slide, such as
            annotation coordinates.

            If `True`, then the same crop used
            when initializing the `Valis` object will be used. If `False`, the
            image will not be cropped. If "overlap", the warped slide will be
            cropped to include only areas where all images overlapped.
            "reference" crops to the area that overlaps with the reference image,
            defined by `reference_img_f` when initialzing the `Valis object`.

        """
        if M is None:
            M = self.M

        if np.issubdtype(type(pt_level), np.integer):
            pt_dim_rc = self.slide_dimensions_wh[pt_level][::-1]
        else:
            pt_dim_rc = np.array(pt_level)

        if np.issubdtype(type(slide_level), np.integer):
            if slide_level != 0:
                if np.issubdtype(type(slide_level), np.integer):
                    aligned_slide_shape = self.val_obj.get_aligned_slide_shape(slide_level)
                else:
                    aligned_slide_shape = np.array(slide_level)
            else:
                aligned_slide_shape = self.aligned_slide_shape_rc
        else:
            aligned_slide_shape = np.array(slide_level)

        if non_rigid:
            fwd_dxdy = self.fwd_dxdy
        else:
            fwd_dxdy = None

        with open(geojson_f) as f:
            annotation_geojson = json.load(f)

        crop_method = self.get_crop_method(crop)
        if crop_method is not False:
            if crop_method == CROP_REF:
                ref_slide = self.val_obj.get_ref_slide()
                if isinstance(slide_level, int):
                    scaled_aligned_shape_rc = ref_slide.slide_dimensions_wh[slide_level][::-1]
                else:
                    if len(slide_level) == 2:
                        scaled_aligned_shape_rc = slide_level
            elif crop_method == CROP_OVERLAP:
                scaled_aligned_shape_rc = aligned_slide_shape

            crop_bbox_xywh, _ = self.get_crop_xywh(crop_method, scaled_aligned_shape_rc)
            shift_xy = crop_bbox_xywh[0:2]
        else:
            shift_xy = None

        warped_features = [None]*len(annotation_geojson["features"])
        for i, ft in tqdm.tqdm(enumerate(annotation_geojson["features"]), desc=WARP_ANNO_MSG, unit="annotation"):
            geom = shapely.geometry.shape(ft["geometry"])
            warped_geom = warp_tools.warp_shapely_geom(geom, M=M,
                                            transformation_src_shape_rc=self.processed_img_shape_rc,
                                            transformation_dst_shape_rc=self.reg_img_shape_rc,
                                            src_shape_rc=pt_dim_rc,
                                            dst_shape_rc=aligned_slide_shape,
                                            fwd_dxdy=fwd_dxdy,
                                            shift_xy=shift_xy)
            warped_ft = deepcopy(ft)
            warped_ft["geometry"] = shapely.geometry.mapping(warped_geom)
            warped_features[i] = warped_ft

        warped_geojson = {"type":annotation_geojson["type"], "features":warped_features}

        return warped_geojson

    def warp_geojson_from_to(self, geojson_f, to_slide_obj, src_slide_level=0, src_pt_level=0,
                             dst_slide_level=0, non_rigid=True):
        """Warp geoms in geojson file from annotation slide to another unwarped slide

        Takes a set of geometries found in this annotation slide, and warps them to
        their position in the unwarped "to" slide.

        Parameters
        ----------
        geojson_f : str
            Path to geojson file containing the annotation geometries. Assumes
            coordinates are in pixels.

        to_slide_obj : Slide
            Slide to which the points will be warped. I.e. `xy`
            will be warped from this Slide to their position in
            the unwarped slide associated with `to_slide_obj`.

        src_pt_level: int, tuple, optional
            Pyramid level of the slide/image in which `xy` originated.
            For example, if `xy` are from the centroids of cell segmentation
            performed on the unwarped full resolution image, this should be 0.
            Alternatively, the value can be a tuple of the image's shape (row, col)
            from which the points came. For example, if `xy` are  bounding
            box coordinates from an analysis on a lower resolution image,
            then pt_level is that lower resolution image's shape (row, col).

        dst_slide_level: int, tuple, optional
            Pyramid level of the slide/image in to `xy` will be warped.
            Similar to `src_pt_level`, if `dst_slide_level` is an int then
            the points will be warped to that pyramid level. If `dst_slide_level`
            is the "to" image's shape (row, col), then the points will be warped
            to their location in an image with that same shape.

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied.

        Returns
        -------
        warped_geojson : dict
            Dictionry of warped geojson geometries

        """

        if np.issubdtype(type(src_pt_level), np.integer):
            src_pt_dim_rc = self.slide_dimensions_wh[src_pt_level][::-1]
        else:
            src_pt_dim_rc = np.array(src_pt_level)

        if np.issubdtype(type(dst_slide_level), np.integer):
            to_slide_src_shape_rc = to_slide_obj.slide_dimensions_wh[dst_slide_level][::-1]
        else:
            to_slide_src_shape_rc = np.array(dst_slide_level)

        if src_slide_level != 0:
            if np.issubdtype(type(src_slide_level), np.integer):
                aligned_slide_shape = self.val_obj.get_aligned_slide_shape(src_slide_level)
            else:
                aligned_slide_shape = np.array(src_slide_level)
        else:
            aligned_slide_shape = self.aligned_slide_shape_rc

        if non_rigid:
            src_fwd_dxdy = self.fwd_dxdy
            dst_bk_dxdy = to_slide_obj.bk_dxdy

        else:
            src_fwd_dxdy = None
            dst_bk_dxdy = None

        with open(geojson_f) as f:
            annotation_geojson = json.load(f)

        warped_features = [None]*len(annotation_geojson["features"])
        for i, ft in tqdm.tqdm(enumerate(annotation_geojson["features"]), desc=WARP_ANNO_MSG, unit="annotation"):
            geom = shapely.geometry.shape(ft["geometry"])
            warped_geom = warp_tools.warp_shapely_geom_from_to(geom=geom,
                                            from_M=self.M,
                                            from_transformation_dst_shape_rc=self.reg_img_shape_rc,
                                            from_transformation_src_shape_rc=self.processed_img_shape_rc,
                                            from_dst_shape_rc=aligned_slide_shape,
                                            from_src_shape_rc=src_pt_dim_rc,
                                            from_fwd_dxdy=src_fwd_dxdy,
                                            to_M=to_slide_obj.M,
                                            to_transformation_src_shape_rc=to_slide_obj.processed_img_shape_rc,
                                            to_transformation_dst_shape_rc=to_slide_obj.reg_img_shape_rc,
                                            to_src_shape_rc=to_slide_src_shape_rc,
                                            to_dst_shape_rc=aligned_slide_shape,
                                            to_bk_dxdy=dst_bk_dxdy
                                            )

            warped_ft = deepcopy(ft)
            warped_ft["geometry"] = shapely.geometry.mapping(warped_geom)
            warped_features[i] = warped_ft

        warped_geojson = {"type":annotation_geojson["type"], "features":warped_features}

        return warped_geojson

    def pad_cropped_processed_img(self):
        """
        Pad cropped processed image to have original dimensions
        """
        vips_img = warp_tools.numpy2vips(self.processed_img)

        padded = vips_img.embed(self.processed_crop_bbox[0], self.processed_crop_bbox[1],
                    self.uncropped_processed_img_shape_rc[1], self.uncropped_processed_img_shape_rc[0],
                    extend=pyvips.enums.Extend.BLACK
                    )
        scaled_padded = warp_tools.resize_img(padded, self.processed_img_shape_rc)
        scaled_padded_np = warp_tools.vips2numpy(scaled_padded)

        return scaled_padded_np


class Valis(object):
    """Reads, registers, and saves a series of slides/images

    Implements the registration pipeline described in
    "VALIS: Virtual Alignment of pathoLogy Image Series" by Gatenbee et al.
    This pipeline will read images and whole slide images (WSI) using pyvips,
    bioformats, or openslide, and so should work with a wide variety of formats.
    VALIS can perform both rigid and non-rigid registration. The registered slides
    can be saved as ome.tiff slides that can be used in downstream analyses. The
    ome.tiff format is opensource and widely supported, being readable in several
    different programming languages (Python, Java, Matlab, etc...) and software,
    such as QuPath or HALO.

    The pipeline is fully automated and goes as follows:

    1. Images/slides are converted to numpy arrays. As WSI are often
    too large to fit into memory, these images are usually lower resolution
    images from different pyramid levels.

    2. Images are processed to single channel images. They are then
    normalized to make them look as similar as possible.

    3. Image features are detected and then matched between all pairs of image.

    4. If the order of images is unknown, they will be optimally ordered
    based on their feature similarity

    5. Rigid registration is performed serially, with each image being
    rigidly aligned to the previous image in the stack.

    6. Non-rigid registration is then performed either by 1) aliging each image
    towards the center of the stack, composing the deformation fields
    along the way, or 2) using groupwise registration that non-rigidly aligns
    the images to a common frame of reference.

    7. Error is measured by calculating the distance between registered
    matched features.

    The transformations found by VALIS can then be used to warp the full
    resolution slides. It is also possible to merge non-RGB registered slides
    to create a highly multiplexed image. These aligned and/or merged slides
    can then be saved as ome.tiff images using pyvips.

    In addition to warping images and slides, VALIS can also warp point data,
    such as cell centoids or ROI coordinates.

    Attributes
    ----------
    name : str
        Descriptive name of registrar, such as the sample's name.

    src_dir: str
        Path to directory containing the slides that will be registered.

    dst_dir : str
        Path to where the results should be saved.

    original_img_list : list of ndarray
        List of images converted from the slides in `src_dir`

    name_dict : dictionary
        Key=full path to image, value = name used to look up `Slide` in `Valis.slide_dict`

    slide_dims_dict_wh :
        Dictionary of slide dimensions. Only needed if dimensions not
        available in the slide/image's metadata.

    resolution_xyu: tuple
        Physical size per pixel and the unit.

    image_type : str
        Type of image, i.e. "brightfield" or "fluorescence"

    series : int
        Slide series to that was read.

    size : int
        Number of images to align

    aligned_img_shape_rc : tuple of int
        Shape (row, col) of aligned images

    aligned_slide_shape_rc : tuple of int
        Shape (row, col) of the aligned slides

    slide_dict : dict of Slide
        Dictionary of Slide objects, each of which contains information
        about a slide, and methods to warp it.

    brightfield_procsseing_fxn_str: str
        Name of function used to process brightfield images.

    if_procsseing_fxn_str : str
        Name of function used to process fluorescence images.

    max_image_dim_px : int
        Maximum width or height of images that will be saved.
        This limit is mostly to keep memory in check.

    max_processed_image_dim_px : int
        Maximum width or height of processed images. An important
        parameter, as it determines the size of of the image in which
        features will be detected and displacement fields computed.

    reference_img_f : str
        Filename of image that will be treated as the center of the stack.
        If None, the index of the middle image will be the reference.

    reference_img_idx : int
        Index of slide that corresponds to `reference_img_f`, after
        the `img_obj_list` has been sorted during rigid registration.

    align_to_reference : bool
        Whether or not images should be aligne to a reference image
        specified by `reference_img_f`. Will be set to True if
        `reference_img_f` is provided.

    crop: str, optional
        How to crop the registered images.

    rigid_registrar : SerialRigidRegistrar
        SerialRigidRegistrar object that performs the rigid registration.

    rigid_reg_kwargs : dict
        Dictionary of keyward arguments passed to
        `serial_rigid.register_images`.

    feature_descriptor_str : str
        Name of feature descriptor.

    feature_detector_str : str
        Name of feature detector.

    transform_str : str
        Name of rigid transform

    similarity_metric : str
        Name of similarity metric used to order slides.

    match_filter_method : str
        Name of method used to filter out poor feature matches.

    non_rigid_registrar : SerialNonRigidRegistrar
        SerialNonRigidRegistrar object that performs serial
        non-rigid registration.

    non_rigid_reg_kwargs : dict
        Dictionary of keyward arguments passed to
        `serial_non_rigid.register_images`.

    non_rigid_registrar_cls : NonRigidRegistrar
        Uninstantiated NonRigidRegistrar class that will be used
        by `non_rigid_registrar` to calculate the deformation fields
        between images.

    non_rigid_reg_class_str : str
        Name of the of class `non_rigid_registrar_cls` belongs to.

    thumbnail_size : int
        Maximum width or height of thumbnails that show results

    original_overlap_img : ndarray
        Image showing how original images overlap before registration.
        Created by merging coloring the inverted greyscale copies of each
        image, and then merging those images.

    rigid_overlap_img : ndarray
        Image showing how images overlap after rigid registration.

    non_rigid_overlap_img : ndarray
        Image showing how images overlap after rigid + non-rigid registration.

    has_rounds : bool
        Whether or not the contents of `src_dir` contain subdirectories that
        have single images spread across multiple files. An example would be
        .ndpis images.

    norm_method : str
        Name of method used to normalize the processed images

    target_processing_stats : ndarray
        Array of processed images' stats used to normalize all images

    summary_df : pd.Dataframe
        Pandas dataframe containing information about the results, such
        as the error, shape of aligned slides, time to completion, etc...

    start_time : float
        The time at which registation was initiated.

    end_rigid_time : float
        The time at which rigid registation was completed.

    end_non_rigid_time : float
        The time at which non-rigid registation was completed.

    error_df : Dataframe
        Dataframe containing information about registration error

    qt_emitter : PySide2.QtCore.Signal
        Used to emit signals that update the GUI's progress bars

    _non_rigid_bbox : list
        Bounding box of area in which non-rigid registration was conducted

    _full_displacement_shape_rc : tuple
        Shape of full displacement field. Would be larger than `_non_rigid_bbox`
        if non-rigid registration only performed in a masked region

    _dup_names_dict : dictionary
        Dictionary describing which images would have been assigned duplicate
        names. Key= duplicated name, value=list of paths to images which
        would have been assigned the same name

    _empty_slides : dictionary
        Dictionary of `Slide` objects that have empty images. Ignored during
        registration but added back at the end


    Examples
    --------

    Basic example using default parameters

    >>> from valis import registration, data
    >>> slide_src_dir = data.dcis_src_dir
    >>> results_dst_dir = "./slide_registration_example"
    >>> registered_slide_dst_dir = "./slide_registration_example/registered_slides"

    Perform registration

    >>> rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    View results in "./slide_registration_example".
    If they look good, warp and save the slides as ome.tiff

    >>> registrar.warp_and_save_slides(registered_slide_dst_dir)

    This example shows how to register CyCIF images and then merge
    to create a high dimensional ome.tiff slide

    >>> registrar = registration.Valis(slide_src_dir, results_dst_dir)
    >>> rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    Create function to get marker names from each slides' filename

    >>> def cnames_from_filename(src_f):
    ...     f = valtils.get_name(src_f)
    ...     return ["DAPI"] + f.split(" ")[1:4]
    ...
    >>> channel_name_dict = {f:cnames_from_filename(f) for f in  registrar.original_img_list}
    >>> merged_img, channel_names, ome_xml = registrar.warp_and_merge_slides(merged_slide_dst_f, channel_name_dict=channel_name_dict)

    View ome.tiff, located at merged_slide_dst_f

    """
    @valtils.deprecated_args(max_non_rigid_registartion_dim_px="max_non_rigid_registration_dim_px", img_type="image_type")
    def __init__(self, src_dir, dst_dir, series=None, name=None, image_type=None,
                 feature_detector_cls=None,
                 transformer_cls=DEFAULT_TRANSFORM_CLASS,
                 affine_optimizer_cls=DEFAULT_AFFINE_OPTIMIZER_CLASS,
                 similarity_metric=DEFAULT_SIMILARITY_METRIC,
                 matcher=DEFAULT_MATCHER,
                 matcher_for_sorting=DEFAULT_MATCHER_FOR_SORTING,
                 imgs_ordered=False,
                 non_rigid_registrar_cls=DEFAULT_NON_RIGID_CLASS,
                 non_rigid_reg_params=DEFAULT_NON_RIGID_KWARGS,
                 compose_non_rigid=False,
                 img_list=None,
                 reference_img_f=None,
                 align_to_reference=False,
                 do_rigid=True,
                 crop=None,
                 create_masks=True,
                 denoise_rigid=False,
                 crop_for_rigid_reg=True,
                 check_for_reflections=False,
                 resolution_xyu=None,
                 slide_dims_dict_wh=None,
                 max_image_dim_px=DEFAULT_MAX_IMG_DIM,
                 max_processed_image_dim_px=DEFAULT_MAX_PROCESSED_IMG_SIZE,
                 max_non_rigid_registration_dim_px=DEFAULT_MAX_NON_RIGID_REG_SIZE,
                 thumbnail_size=DEFAULT_THUMBNAIL_SIZE,
                 norm_method=DEFAULT_NORM_METHOD,
                 micro_rigid_registrar_cls=None,
                 micro_rigid_registrar_params={},
                 qt_emitter=None):

        """
        src_dir: str
            Path to directory containing the slides that will be registered.

        dst_dir : str
            Path to where the results should be saved.

        name : str, optional
            Descriptive name of registrar, such as the sample's name

        series : int, optional
            Slide series to that was read. If None, series will be set to 0.

        image_type : str, optional
            The type of image, either "brightfield", "fluorescence",
            or "multi". If None, VALIS will guess `image_type`
            of each image, based on the number of channels and datatype.
            Will assume that RGB = "brightfield",
            otherwise `image_type` will be set to "fluorescence".

        feature_detector_cls : FeatureDD, optional
            FeatureDD object or class that detects and computes
            image features.
            available feature_detectors are found in the `feature_detectors`
            module. If a desired feature detector is not available,
            one can be created by subclassing `feature_detectors.FeatureDD`.

        transformer_cls : scikit-image Transform class, optional
            Uninstantiated scikit-image transformer used to find
            transformation matrix that will warp each image to the target
            image. Default is SimilarityTransform

        affine_optimizer_cls : AffineOptimzer class, optional
            Uninstantiated AffineOptimzer that will minimize a
            cost function to find the optimal affine transformations.
            If a desired affine optimization is not available,
            one can be created by subclassing `affine_optimizer.AffineOptimizer`.

        similarity_metric : str, optional
            Metric used to calculate similarity between images, which is in
            turn used to build the distance matrix used to sort the images.
            Can be "n_matches", or a string to used as
            distance in spatial.distance.cdist. "n_matches"
            is the number of matching features between image pairs.

        matcher_for_sorting: feature_matcher.Matcher
            Matcher that will be used to construct image dissimilarity matrix
            constructed using feature matches

        matcher: feature_matcher.Matcher
            Matcher that will be used to match images after sorting

        imgs_ordered : bool, optional
            Boolean defining whether or not the order of images in img_dir
            are already in the correct order. If True, then each filename should
            begin with the number that indicates its position in the z-stack. If
            False, then the images will be sorted by ordering a feature distance
            matix. Default is False.

        reference_img_f : str, optional
            Filename of image that will be treated as the center of the stack.
            If None, the index of the middle image will be the reference.

        align_to_reference : bool, optional
            If `False`, images will be non-rigidly aligned serially towards the
            reference image. If `True`, images will be non-rigidly aligned
            directly to the reference image. If `reference_img_f` is None,
            then the reference image will be the one in the middle of the stack.

        non_rigid_registrar_cls : NonRigidRegistrar, optional
            Uninstantiated NonRigidRegistrar class that will be used to
            calculate the deformation fields between images. See
            the `non_rigid_registrars` module for a desciption of available
            methods. If a desired non-rigid registration method is not available,
            one can be implemented by subclassing.NonRigidRegistrar.
            If None, then only rigid registration will be performed

        non_rigid_reg_params: dictionary, optional
            Dictionary containing key, value pairs to be used to initialize
            `non_rigid_registrar_cls`.
            In the case where simple ITK is used by the, params should be
            a SimpleITK.ParameterMap. Note that numeric values need to be
            converted to strings. See the NonRigidRegistrar classes in
            `non_rigid_registrars` for the available non-rigid registration
            methods and arguments.

        compose_non_rigid : bool, optional
            Whether or not to compose non-rigid transformations. If `True`,
            then an image is non-rigidly warped before aligning to the
            adjacent non-rigidly aligned image. This allows the transformations
            to accumulate, which may bring distant features together but could
            also  result  in un-wanted deformations, particularly around the edges.
            If `False`, the image not warped before being aaligned to the adjacent
            non-rigidly aligned image. This can reduce unwanted deformations, but
            may not bring distant features together.

        img_list : list, dictionary, optional
            List of images to be registered. However, it can also be a dictionary,
            in which case the key: value pairs are full_path_to_image: name_of_image,
            where name_of_image is the key that can be used to access the image from
            Valis.slide_dict.

        do_rigid: bool, dictionary, optional
            Whether or not to perform rigid registration. If `False`, rigid
            registration will be skipped.

            If `do_rigid` is a dictionary, it should contain inverse transformation
            matrices to rigidly align images to the specificed by `reference_img_f`.
            M will be estimated for images that are not in the dictionary.
            Each key is the filename of the image associated with the transformation matrix,
            and value is a dictionary containing the following values:
                `M` : (required) a 3x3 inverse transformation matrix as a numpy array.
                      Found by determining how to align fixed to moving.
                      If `M` was found by determining how to align moving to fixed,
                      then `M` will need to be inverted first.
                `transformation_src_shape_rc` : (optional) shape (row, col) of image used to find the rigid transformation.
                      If not provided, then it is assumed to be the shape of the level 0 slide
                `transformation_dst_shape_rc` : (optional) shape of registered image.
                      If not provided, this is assumed to be the shape of the level 0 reference slide.

        crop: str, optional
            How to crop the registered images. "overlap" will crop to include
            only areas where all images overlapped. "reference" crops to the
            area that overlaps with a reference image, defined by
            `reference_img_f`. This option can be used even if `reference_img_f`
            is `None` because the reference image will be set as the one at the center
            of the stack.

            If both `crop` and `reference_img_f` are `None`, `crop`
            will be set to "overlap". If `crop` is None, but `reference_img_f`
            is defined, then `crop` will be set to "reference".

        create_masks : bool, optional
            Whether or not to create and apply masks for registration.
            Can help focus alignment on the tissue, but can sometimes
            mask too much if there is a lot of variation in the image.

        denoise_rigid : bool, optional
            Whether or not to denoise processed images before rigid registion.
            Note that un-denoised images are used in the non-rigid registration

        crop_for_rigid_reg : bool, optional
            Whether or not to crop the images used for rigid registration. If `True`,
            then higher resolution images may be used for rigid registeration, as valis
            will "zoom" in to the area around the mask created by `ImageProcesser.create_mask()`,
            and slice out that region and resize it to have a maximum dimension the same
            as `max_processed_image_dim_px`. If `False`, the full image will be used, although
            the tissue may be at a lower resolution.

        check_for_reflections : bool, optional
            Determine if alignments are improved by relfecting/mirroring/flipping
            images. Optional because it requires re-detecting features in each version
            of the images and then re-matching features, and so can be time consuming and
            not always necessary.

        resolution_xyu: tuple, optional
            Physical size per pixel and the unit. If None (the default), these
            values will be determined for each slide using the slides' metadata.
            If provided, this physical pixel sizes will be used for all of the slides.
            This option is available in case one cannot easily access to the original
            slides, but does have the information on pixel's physical units.

        slide_dims_dict_wh : dict, optional
            Key= slide/image file name,
            value= dimensions = [(width, height), (width, height), ...] for each level.
            If None (the default), the slide dimensions will be pulled from the
            slides' metadata. If provided, those values will be overwritten. This
            option is available in case one cannot easily access to the original
            slides, but does have the information on the slide dimensions.

        max_image_dim_px : int, optional
            Maximum width or height of images that will be saved.
            This limit is mostly to keep memory in check.

        max_processed_image_dim_px : int, optional
            Maximum width or height of processed images. An important
            parameter, as it determines the size of of the image in which
            features will be detected and displacement fields computed.

        max_non_rigid_registration_dim_px : int, optional
             Maximum width or height of images used for non-rigid registration.
             Larger values may yeild more accurate results, at the expense of
             speed and memory. There is also a practical limit, as the specified
             size may be too large to fit in memory.

        mask_dict : dictionary
            Dictionary where key = overlap type (all, overlap, or reference), and
            value = (mask, mask_bbox_xywh)

        thumbnail_size : int, optional
            Maximum width or height of thumbnails that show results

        norm_method : str
            Name of method used to normalize the processed images. Options
            are None when normalization is not desired, "histo_match" for
            histogram matching and "img_stats" for normalizing by image statistics.
            See preprocessing.match_histograms and preprocessing.norm_khan
            for details.

        iter_order : list of tuples
            Each element of `iter_order` contains a tuple of stack
            indices. The first value is the index of the moving/current/from
            image, while the second value is the index of the moving/next/to
            image.

        micro_rigid_registrar_cls : MicroRigidRegistrar, optional
            Class used to perform higher resolution rigid registration. If `None`,
            this step is skipped.

        micro_rigid_registrar_params : dictionary
            Dictionary of keyword arguments used intialize the `MicroRigidRegistrar`

        qt_emitter : PySide2.QtCore.Signal, optional
            Used to emit signals that update the GUI's progress bars

        """

        # Get name, based on src directory
        if name is None:
            if src_dir.endswith(os.path.sep):
                name = os.path.split(src_dir[:-1])[1]
            else:
                name = os.path.split(src_dir)[1]
        self.name = name.replace(" ", "_")

        # Set paths #
        self.src_dir = src_dir
        self.dst_dir = os.path.join(dst_dir, self.name)
        self.name_dict = None

        if img_list is not None:
            if isinstance(img_list, dict):
                # Key=original file name, value=name
                self.original_img_list = list(img_list.keys())
                self.name_dict = img_list
            elif hasattr(img_list, "__iter__"):
                self.original_img_list = list(img_list)
            else:
                msg = (f"Cannot upack `img_list`, which is type {type(img_list).__name__}. "
                       "Please provide an iterable object (list, tuple, array, etc...) that has the location of the images")
                valtils.print_warning(msg, rgb=Fore.RED)
        else:
            self.get_imgs_in_dir()

        if reference_img_f is not None:
            img_names = [valtils.get_name(f) for f in self.original_img_list]

            if valtils.get_name(reference_img_f) not in img_names:
                if not imgs_ordered:
                    # Reference image not in img_list, but not ordered. So just append
                    ref_img_src_dir = len(os.path.split(reference_img_f)[0])
                    if len(ref_img_src_dir) == 0:
                        # Only provided file name. Assuming it's located it src directory
                        reference_img_f = os.path.join(self.dst_dir, reference_img_f)
                    self.original_img_list.append(reference_img_f)
                else:
                    print_f = os.path.split(reference_img_f)[1]
                    msg = (f"{print_f} specified as the reference image, "
                           f"but it is not in `img_list`. Not possible to append {print_f} to `img_list` "
                           f"because `imgs_ordered`={imgs_ordered}. "
                           f"Please add {print_f} to `img_list` at the desired location, "
                           f"or set `imgs_ordered`=False if order of image alignment is not already known")
                    valtils.print_warning(msg, rgb=Fore.RED)
                    return None


        self.original_img_list = [str(x) for x in self.original_img_list]
        if self.name_dict is None:
            self.name_dict = self.get_img_names(self.original_img_list)

        self.check_for_duplicated_names(self.original_img_list)

        valtils.sort_nicely(self.original_img_list)

        self.set_dst_paths()

        # Some information may already be provided #
        self.slide_dims_dict_wh = slide_dims_dict_wh
        self.resolution_xyu = resolution_xyu
        self.image_type = image_type

        # Results fields #
        self.series = series
        self.size = 0
        self.aligned_img_shape_rc = None
        self.aligned_slide_shape_rc = None
        self.slide_dict = {}

        # Fields related to image pre-processing #
        self.brightfield_procsseing_fxn_str = None
        self.if_procsseing_fxn_str = None

        if max_image_dim_px < max_processed_image_dim_px:
            msg = f"max_image_dim_px is {max_image_dim_px} but needs to be less or equal to {max_processed_image_dim_px}. Setting max_image_dim_px to {max_processed_image_dim_px}"
            valtils.print_warning(msg)
            max_image_dim_px = max_processed_image_dim_px

        self.max_image_dim_px = max_image_dim_px
        self.max_processed_image_dim_px = max_processed_image_dim_px
        self.max_non_rigid_registration_dim_px = max_non_rigid_registration_dim_px

        # Setup rigid registration #
        self.reference_img_idx = None
        self.reference_img_f = reference_img_f
        self.align_to_reference = align_to_reference
        self.iter_order = None

        self.do_rigid = do_rigid
        self.crop_for_rigid_reg = crop_for_rigid_reg
        self.rigid_registrar = None
        self.micro_rigid_registrar_cls = micro_rigid_registrar_cls
        self.micro_rigid_registrar_params = micro_rigid_registrar_params
        self.denoise_rigid = denoise_rigid

        if feature_detector_cls is not None:
            if isinstance(feature_detector_cls, type):
                fd = feature_detector_cls()
            else:
                fd = feature_detector_cls

            if matcher.feature_detector is not None:
                msg = (f"Replacing feature detector in {matcher.__class__.__name__} with {fd.__class__.__name__}, "
                       f"which was previously {matcher.feature_detector.__class__.__name__}. "
                       f"To avoid this, set `feature_detector_cls=None` when initializing the `Valis` object")
                valtils.print_warning(msg, None)
            matcher.feature_detector = fd
        else:
            fd = None

        self._set_rigid_reg_kwargs(name=name,
                                   feature_detector=fd,
                                   similarity_metric=similarity_metric,
                                   matcher=matcher,
                                   matcher_for_sorting=matcher_for_sorting,
                                   transformer=transformer_cls,
                                   affine_optimizer=affine_optimizer_cls,
                                   imgs_ordered=imgs_ordered,
                                   reference_img_f=reference_img_f,
                                   check_for_reflections=check_for_reflections,
                                   qt_emitter=qt_emitter)


        # Setup non-rigid registration #
        self.non_rigid_registrar = None
        self.non_rigid_registrar_cls = non_rigid_registrar_cls
        self._non_rigid_bbox = None
        self._full_displacement_shape_rc = None

        if crop is None:
            if reference_img_f is None:
                self.crop = CROP_OVERLAP
            else:
                self.crop = CROP_REF
        else:
            self.crop = crop

        self.compose_non_rigid = compose_non_rigid
        if non_rigid_registrar_cls is not None:
            self._set_non_rigid_reg_kwargs(name=name,
                                           non_rigid_reg_class=non_rigid_registrar_cls,
                                           non_rigid_reg_params=non_rigid_reg_params,
                                           reference_img_f=reference_img_f,
                                           compose_non_rigid=compose_non_rigid,
                                           qt_emitter=qt_emitter)

        # Info realted to saving images to view results #
        self.mask_dict = None
        self.create_masks = create_masks

        self.thumbnail_size = thumbnail_size
        self.original_overlap_img = None
        self.rigid_overlap_img = None
        self.non_rigid_overlap_img = None
        self.micro_reg_overlap_img = None

        self.has_rounds = False
        self.norm_method = norm_method
        self.target_processing_hist = None
        self.target_processing_stats = None
        self.norm_percentiles = np.array([1, 5, 50, 95, 99])
        self.summary_df = None
        self.start_time = None
        self.end_rigid_time = None
        self.end_non_rigid_time = None

        self.error_df = None

        self._empty_slides = {}

    def __repr__(self):
        repr_str = (f'<{self.__class__.__name__}, name = {self.name}>'
                    f', size={self.size}>'
                    )
        return (repr_str)

    def _set_rigid_reg_kwargs(self, name, feature_detector, similarity_metric,
                              matcher, matcher_for_sorting, transformer, affine_optimizer,
                              imgs_ordered, reference_img_f,
                              check_for_reflections, qt_emitter):

        """Set rigid registration kwargs
        Keyword arguments will be passed to `serial_rigid.register_images`

        """

        if affine_optimizer is not None:
            afo = affine_optimizer(transform=transformer.__name__)
        else:
            afo = affine_optimizer

        self.rigid_reg_kwargs = {NAME_KEY: name,
                                 FD_KEY: feature_detector,
                                 SIM_METRIC_KEY: similarity_metric,
                                 TRANSFORMER_KEY: transformer(),
                                 MATCHER_KEY: matcher,
                                 MATCHER_FOR_SORTING_KEY: matcher_for_sorting,
                                 AFFINE_OPTIMIZER_KEY: afo,
                                 REF_IMG_KEY: reference_img_f,
                                 IMAGES_ORDERD_KEY: imgs_ordered,
                                 CHECK_REFLECT_KEY: check_for_reflections,
                                 QT_EMMITER_KEY: qt_emitter
                                 }

        # Save methods as strings since some objects cannot be pickled #

        self.feature_descriptor_str = self.rigid_reg_kwargs[MATCHER_KEY].feature_detector.kp_descriptor_name
        self.feature_detector_str = self.rigid_reg_kwargs[MATCHER_KEY].feature_detector.kp_detector_name

        self.transform_str = self.rigid_reg_kwargs[TRANSFORMER_KEY].__class__.__name__
        self.similarity_metric = self.rigid_reg_kwargs[SIM_METRIC_KEY]
        self.match_filter_method = matcher.__class__.__name__
        self.imgs_ordered = imgs_ordered

    def _set_non_rigid_reg_kwargs(self, name, non_rigid_reg_class, non_rigid_reg_params,
                                  reference_img_f, compose_non_rigid, qt_emitter):
        """Set non-rigid registration kwargs
        Keyword arguments will be passed to `serial_non_rigid.register_images`

        """

        if non_rigid_reg_class is not None:
            if isinstance(non_rigid_reg_class, type):
                if non_rigid_reg_params is None:
                    nr = non_rigid_reg_class()
                else:
                    nr = non_rigid_reg_class(**non_rigid_reg_params)
            else:
                nr = non_rigid_reg_class
        else:
            nr = None

        self.non_rigid_reg_kwargs = {NAME_KEY: name,
                                     NON_RIGID_REG_CLASS_KEY: nr,
                                     NON_RIGID_REG_PARAMS_KEY: non_rigid_reg_params,
                                     REF_IMG_KEY: reference_img_f,
                                     QT_EMMITER_KEY: qt_emitter,
                                     NON_RIGID_COMPOSE_KEY: compose_non_rigid
                                     }

        self.non_rigid_reg_class_str = nr.__class__.__name__

    def _add_empty_slides(self):

        # Fill in missing attributes
        for slide_name, slide_obj in self._empty_slides.items():

            slide_obj.processed_img_shape_rc = slide_obj.image.shape[0:2]
            slide_obj.aligned_slide_shape_rc = self.aligned_slide_shape_rc
            slide_obj.reg_img_shape_rc = self.aligned_img_shape_rc

            slide_obj.processed_img = np.zeros(slide_obj.processed_img_shape_rc)
            slide_obj.rigid_reg_mask = np.full(slide_obj.processed_img_shape_rc, 255)
            slide_obj.non_rigid_reg_mask = np.full(slide_obj.reg_img_shape_rc, 255)

            slide_obj.M = np.eye(3)

            slide_obj.stack_idx = self.size
            self.size += 1
            self.slide_dict[slide_name] = slide_obj

    def get_imgs_in_dir(self):
        """Get all images in Valis.src_dir

        """

        full_path_list = [os.path.join(self.src_dir, f) for f in os.listdir(self.src_dir)]
        self.original_img_list = []
        img_names = []
        for f in full_path_list:
            if os.path.isfile(f):
                if slide_tools.get_img_type(f) is not None:
                    self.original_img_list.append(f)
                    img_names.append(valtils.get_name(f))

        for f in full_path_list:
            if os.path.isdir(f):
                dir_name = os.path.split(f)[1]
                is_round, master_slide = slide_tools.determine_if_staining_round(f)
                if is_round:
                    self.original_img_list.append(master_slide)

                else:
                    # Some formats, like .mrxs have the main file but
                    # data in a subdirectory with the same name
                    escape_dir = re.escape(dir_name) # Remove special characters
                    matching_f = [ff for ff in full_path_list if re.search(escape_dir, ff) is not None and os.path.split(ff)[1] != dir_name]

                    if len(matching_f) == 1:
                        if not matching_f[0] in self.original_img_list:
                            # Make sure that file not already in list
                            self.original_img_list.extend(matching_f)
                            img_names.append(dir_name)

                    elif len(matching_f) > 1:
                        msg = f"found {len(matching_f)} matches for {dir_name}: {', '.join(matching_f)}"
                        valtils.print_warning(msg, rgb=Fore.RED)
                    elif len(matching_f) == 0:
                        msg = f"Can't find slide file associated with {dir_name}"
                        valtils.print_warning(msg, rgb=Fore.RED)

        # Final check to make sure that all images are readable
        self.original_img_list = [f for f in self.original_img_list if slide_tools.get_slide_extension(f) in slide_io.ALL_READABLE_FORMATS and len(slide_tools.get_slide_extension(f)) > 0]

    def set_dst_paths(self):
        """Set paths to where the results will be saved.

        """

        self.img_dir = os.path.join(self.dst_dir, CONVERTED_IMG_DIR)
        self.processed_dir = os.path.join(self.dst_dir, PROCESSED_IMG_DIR)
        self.reg_dst_dir = os.path.join(self.dst_dir, RIGID_REG_IMG_DIR)
        self.non_rigid_dst_dir = os.path.join(self.dst_dir, NON_RIGID_REG_IMG_DIR)
        self.deformation_field_dir = os.path.join(self.dst_dir, DEFORMATION_FIELD_IMG_DIR)
        self.overlap_dir = os.path.join(self.dst_dir, OVERLAP_IMG_DIR)
        self.data_dir = os.path.join(self.dst_dir, REG_RESULTS_DATA_DIR)
        self.displacements_dir = os.path.join(self.dst_dir, DISPLACEMENT_DIRS)
        self.micro_reg_dir = os.path.join(self.dst_dir, MICRO_REG_DIR)
        self.mask_dir = os.path.join(self.dst_dir, MASK_DIR)

    def get_slide(self, src_f):
        """Get Slide

        Get the Slide associated with `src_f`.
        Slide store registration parameters and other metadata about
        the slide associated with `src_f`. Slide can also:

        * Convert the slide to a numpy array (Slide.slide2image)
        * Convert the slide to a pyvips.Image (Slide.slide2vips)
        * Warp the slide (Slide.warp_slide)
        * Save the warped slide as an ome.tiff (Slide.warp_and_save_slide)
        * Warp an image of the slide (Slide.warp_img)
        * Warp points (Slide.warp_xy)
        * Warp points in one slide to their position in another unwarped slide (Slide.warp_xy_from_to)
        * Access slide ome-xml (Slide.original_xml)

        See Slide for more details.

        Parameters
        ----------
        src_f : str
            Path to the slide, or name assigned to slide (see Valis.name_dict)

        Returns
        -------
        slide_obj : Slide
            Slide associated with src_f

        """

        default_name = valtils.get_name(src_f)

        if src_f in self.name_dict.keys():
            # src_f is full path to image
            assigned_name = self.name_dict[src_f]
        elif src_f in self.name_dict.values():
            # src_f is name of image
            assigned_name = src_f
        else:
            # src_f isn't in name_dict
            assigned_name = None

        if default_name in self.slide_dict:
            # src_f is the image name or file name
            slide_obj = self.slide_dict[default_name]

        elif assigned_name in self.slide_dict:
            # src_f is full path and name was looked up
            slide_obj = self.slide_dict[assigned_name]

        elif src_f in self.slide_dict:
            # src_f is the name of the slide
            slide_obj = self.slide_dict[src_f]

        elif default_name in self._dup_names_dict:
            # default name has multiple matches
            n_matching = len(self._dup_names_dict[default_name])
            possible_names_dict = {f: self.name_dict[f] for f in self._dup_names_dict[default_name]}

            msg = (f"\n{src_f} matches {n_matching} images in this dataset:\n"
                   f"{pformat(self._dup_names_dict[default_name])}"
                   f"\n\nPlease see `Valis.name_dict` to find correct name in "
                   f"the dictionary. Either key (filenmae) or value (assigned name) will work:\n"
                   f"{pformat(possible_names_dict)}")

            valtils.print_warning(msg, rgb=Fore.RED)
            slide_obj = None

        return slide_obj

    def get_ref_slide(self):
        ref_slide = self.get_slide(self.reference_img_f)

        return ref_slide

    def get_img_names(self, img_list):
        """
        Check that each image will have a unique name, which is based on the file name.
        Images that would otherwise have the same name are assigned extra ids, starting at 0.
        For example, if there were three images named "HE.tiff", they would be
        named "HE_0", "HE_1", and "HE_2".

        Parameters
        ----------

        img_list : list
            List of image names

        Returns
        -------
        name_dict : dict
            Dictionary, where key= full path to image, value = image name used as
            key in Valis.slide_dict

        """

        img_df = pd.DataFrame({"img_f": img_list,
                               "name": [valtils.get_name(f) for f in img_list]})

        names_dict = {f: valtils.get_name(f) for f in img_list}
        count_df = img_df["name"].value_counts()
        dup_idx = np.where(count_df.values > 1)[0]
        if len(dup_idx) > 0:
            for i in dup_idx:
                dup_name = count_df.index[i]
                dup_paths = img_df["img_f"][img_df["name"] == dup_name]
                z = len(str(len(dup_paths)))

                msg = f"Detected {len(dup_paths)} images that would be named {dup_name}"
                valtils.print_warning(msg, rgb=Fore.RED)

                for j, p in enumerate(dup_paths):
                    new_name = f"{names_dict[p]}_{str(j).zfill(z)}"
                    msg = f"Renmaing {p} to {new_name} in Valis.slide_dict)"
                    valtils.print_warning(msg)
                    names_dict[p] = new_name

        return names_dict

    def check_for_duplicated_names(self, img_list):
        """
        Create dictionary that tracks which files
        might be assigned the same name, which
        can happen if the filenames (minus the rest of the path) are the same
        """
        default_names_dict = {}
        for f in img_list:
            default_name = valtils.get_name(f)
            if default_name not in default_names_dict:
                default_names_dict[default_name] = [f]
            else:
                default_names_dict[default_name].append(f)

        self._dup_names_dict = {k: v for k, v in default_names_dict.items() if len(v) > 1}

    def create_img_reader_dict(self, reader_dict=None, default_reader=None, series=None):

        if reader_dict is None:
            named_reader_dict = {}
        else:
            named_reader_dict = {valtils.get_name(f): reader_dict[f] for f in reader_dict.keys()}

        for i, slide_f in enumerate(self.original_img_list):
            slide_name = valtils.get_name(slide_f)
            if slide_name not in named_reader_dict:
                if default_reader is None:
                    try:
                        slide_reader_cls = slide_io.get_slide_reader(slide_f, series=series)
                    except Exception as e:
                        traceback_msg = traceback.format_exc()
                        msg = f"Attempting to get reader for {slide_f} created the following error:\n{e}"
                        valtils.print_warning(msg, rgb=Fore.RED, traceback_msg=traceback_msg)
                else:
                    slide_reader_cls = default_reader

                slide_reader_kwargs = {"series": series}
            else:
                slide_reader_info = named_reader_dict[slide_name]

                if isinstance(slide_reader_info, list) or isinstance(slide_reader_info, tuple):
                    if len(slide_reader_info) == 2:
                        slide_reader_cls, slide_reader_kwargs = slide_reader_info
                    elif len(slide_reader_info) == 1:
                        # Provided reader, but no kwargs
                        slide_reader_cls = slide_reader_info[0]
                        slide_reader_kwargs = {}
                elif isinstance(slide_reader_info, slide_io.ImageReader):
                    slide_reader_kwargs = {}
                    slide_reader = slide_reader_info
                else:
                    # Provided reader, but no kwargs
                    slide_reader = slide_reader_info
                    slide_reader_kwargs = {}
            try:
                slide_reader = slide_reader_cls(src_f=slide_f, **slide_reader_kwargs)
            except Exception as e:
                traceback_msg = traceback.format_exc()
                msg = f"Attempting to read {slide_f} created the following error:\n{e}"
                valtils.print_warning(msg, rgb=Fore.RED, traceback_msg=traceback_msg)

            named_reader_dict[slide_name] = slide_reader

        return named_reader_dict

    def convert_imgs(self, series=None, reader_dict=None, reader_cls=None):
        """Convert slides to images and create dictionary of Slides.

        series : int, optional
            Slide series to be read. If None, the series with largest image will be read

        reader_cls : SlideReader, optional
            Uninstantiated SlideReader class that will convert
            the slide to an image, and also collect metadata.

        reader_dict: dict, optional
            Dictionary specifying which readers to use for individual images.
            The keys, value pairs are image filename and instantiated `slide_io.SlideReader`
            to use to read that file. Valis will try to find an appropritate reader
            for any omitted files, or will use `reader_cls` as the default.

        """

        named_reader_dict = self.create_img_reader_dict(reader_dict=reader_dict,
                                                        default_reader=reader_cls,
                                                        series=series)
        img_types = []
        self.size = 0
        for f in tqdm.tqdm(self.original_img_list, desc=CONVERT_MSG, unit="image"):
            slide_name = valtils.get_name(f)
            reader = named_reader_dict[slide_name]
            slide_dims = reader.metadata.slide_dimensions
            levels_in_range = np.where(slide_dims.max(axis=1) <= self.max_image_dim_px)[0]

            if len(levels_in_range) > 0:
                level = levels_in_range[0] - 1
            else:
                level = len(slide_dims) - 1

            level = max(level, 0)  # Avoid negative level

            vips_img = reader.slide2vips(level=level)

            scaling = np.min(self.max_image_dim_px/np.array([vips_img.width, vips_img.height]))
            if scaling < 1:
                vips_img = warp_tools.rescale_img(vips_img, scaling)

            img = warp_tools.vips2numpy(vips_img)




            slide_name = self.name_dict[f]
            slide_obj = Slide(f, img, self, reader, name=slide_name)
            slide_obj.crop = self.crop

            # Will overwrite data if provided. Can occur if reading images, not the actual slides #
            if self.slide_dims_dict_wh is not None:
                matching_slide = [k for k in self.slide_dims_dict_wh.keys()
                                  if valtils.get_name(k) == slide_obj.name][0]

                slide_dims = self.slide_dims_dict_wh[matching_slide]
                if slide_dims.ndim == 1:
                    slide_dims = np.array([[slide_dims]])
                slide_obj.slide_shape_rc = slide_dims[0][::-1]

            if self.resolution_xyu is not None:
                slide_obj.resolution = np.mean(self.resolution_xyu[0:2])
                slide_obj.units = self.resolution_xyu[2]

            if slide_obj.is_empty:
                msg = f"{slide_obj.name} appears to be empty and will be skipped during registration"
                valtils.print_warning(msg)
                self._empty_slides[slide_obj.name] = slide_obj
                continue

            img_types.append(slide_obj.img_type)
            self.slide_dict[slide_obj.name] = slide_obj
            self.size += 1

        if self.image_type is None:
            unique_img_types = list(set(img_types))
            if len(unique_img_types) > 1:
                self.image_type = slide_tools.MULTI_MODAL_NAME
            else:
                self.image_type = unique_img_types[0]

        self.check_img_max_dims()

    def check_img_max_dims(self):
        """Ensure that all images have similar sizes.

        `max_image_dim_px` will be set to the maximum dimension of the
        smallest image if that value is less than max_image_dim_px

        """

        og_img_sizes_wh = np.array([slide_obj.image.shape[0:2][::-1] for slide_obj in self.slide_dict.values()])
        img_max_dims = og_img_sizes_wh.max(axis=1)
        min_max_wh = img_max_dims.min()
        scaling_for_og_imgs = min_max_wh/img_max_dims

        if np.any(scaling_for_og_imgs < 1):
            msg = f"Smallest image is less than max_image_dim_px. parameter max_image_dim_px is being set to {min_max_wh}"
            valtils.print_warning(msg)
            self.max_image_dim_px = min_max_wh
            for slide_obj in self.slide_dict.values():
                # Rescale images
                scaling = self.max_image_dim_px/max(slide_obj.image.shape[0:2])
                assert scaling <= self.max_image_dim_px
                if scaling < 1:
                    slide_obj.image = warp_tools.rescale_img(slide_obj.image, scaling)

        if self.max_processed_image_dim_px > self.max_image_dim_px:
            msg = f"parameter max_processed_image_dim_px also being updated to {self.max_image_dim_px}"
            valtils.print_warning(msg)
            self.max_processed_image_dim_px = self.max_image_dim_px

    def create_original_composite_img(self, rigid_registrar):
        """Create imaage showing how images overlap before registration
        """

        min_r = np.inf
        max_r = 0
        min_c = np.inf
        max_c = 0
        composite_img_list = [None] * self.size

        thumbnail_s = np.min(self.thumbnail_size/np.array(rigid_registrar.img_obj_list[0].padded_shape_rc))
        for i, img_obj in enumerate(rigid_registrar.img_obj_list):
            img = img_obj.image
            padded_img = transform.warp(img, img_obj.T, preserve_range=True,
                                        output_shape=img_obj.padded_shape_rc)

            composite_img_list[i] = warp_tools.rescale_img(padded_img, scaling=thumbnail_s)

            img_corners_rc = warp_tools.get_corners_of_image(img.shape[0:2])
            warped_corners_xy = warp_tools.warp_xy(img_corners_rc[:, ::-1], img_obj.T)
            min_r = min(warped_corners_xy[:, 1].min(), min_r)
            max_r = max(warped_corners_xy[:, 1].max(), max_r)
            min_c = min(warped_corners_xy[:, 0].min(), min_c)
            max_c = max(warped_corners_xy[:, 0].max(), max_c)

        overlap_img = self.draw_overlap_img(img_list=composite_img_list)
        min_r = int(min_r*thumbnail_s)
        max_r = int(np.ceil(max_r*thumbnail_s))
        min_c = int(min_c*thumbnail_s)
        max_c = int(np.ceil(max_c*thumbnail_s))
        overlap_img = overlap_img[min_r:max_r, min_c:max_c]

        return overlap_img

    def measure_original_mmi(self, img1, img2):
        """Measure Mattes mutation inormation between 2 unregistered images.
        """

        dst_rc = np.max([img1.shape, img2.shape], axis=1)
        padded_img_list = [None] * self.size
        for i, img in enumerate([img1, img2]):
            T = warp_tools.get_padding_matrix(img.shape, dst_rc)
            padded_img = transform.warp(img, T, preserve_range=True, output_shape=dst_rc)
            padded_img_list[i] = padded_img

        og_mmi = warp_tools.mattes_mi(padded_img_list[0], padded_img_list[1])

        return og_mmi

    def create_img_processor_dict(self, brightfield_processing_cls=DEFAULT_BRIGHTFIELD_CLASS,
                                  brightfield_processing_kwargs=DEFAULT_BRIGHTFIELD_PROCESSING_ARGS,
                                  if_processing_cls=DEFAULT_FLOURESCENCE_CLASS,
                                  if_processing_kwargs=DEFAULT_FLOURESCENCE_PROCESSING_ARGS,
                                  processor_dict=None):
        """Create dictionary to get processors for each image

        Create dictionary to get processors for each image. If an image is not in `processing_dict`,
        this function will try to guess the modality and then assign a default processor.

        Parameters
        ----------
        brightfield_processing_cls : ImageProcesser
            ImageProcesser to pre-process brightfield images to make them look as similar as possible.
            Should return a single channel uint8 image.

        brightfield_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `brightfield_processing_cls`

        if_processing_cls : ImageProcesser
            ImageProcesser to pre-process immunofluorescent images to make them look as similar as possible.
            Should return a single channel uint8 image.

        if_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `if_processing_cls`

        processor_dict : dict
            Each key should be the filename of the image, and the value either a subclassed
            preprocessing.ImageProcessor, or a list, where the 1st element is the processor,
            and the second element a dictionary of keyword arguments passed to ImageProcesser.process_img.
            If `None`, then this function will assign a processor to each image.

        Returns
        -------
        named_processing_dict : Dict
            Each key is the name of the slide, and the value is a list, where
            the 1st element is the processor, and the second element a dictionary
            of keyword arguments passed to ImageProcesser.process_img

        """

        if processor_dict is None:
            named_processing_dict = {}
        else:
            named_processing_dict = {self.get_slide(f).name: processor_dict[f] for f in processor_dict.keys()}

        for i, slide_obj in enumerate(self.slide_dict.values()):

            if slide_obj.name in named_processing_dict:
                slide_p = named_processing_dict[slide_obj.name]
                if isinstance(slide_p, list):
                    if len(slide_p) == 2:
                        slide_p, slide_kwargs = slide_p
                    elif len(slide_p) == 1:
                        # Provided processor, but no kwargs
                        slide_kwargs = {}
                else:
                    # Provided processor, but no kwargs
                    slide_kwargs = {}

                named_processing_dict[slide_obj.name] = [slide_p, slide_kwargs]

            else:
                # Processor not provided, so assign one based on inferred modality
                is_ihc = slide_obj.img_type == slide_tools.IHC_NAME
                if is_ihc:
                    processing_cls = brightfield_processing_cls
                    processing_kwargs = brightfield_processing_kwargs

                else:
                    processing_cls = if_processing_cls
                    processing_kwargs = if_processing_kwargs

                named_processing_dict[slide_obj.name] = [processing_cls, processing_kwargs]

        return named_processing_dict

    def get_roi_for_processing(self, slide_obj, processing_cls, mask=None):

        # First, create mask from whole image
        if mask is None:
            mask_level = slide_tools.get_level_idx(slide_obj.slide_dimensions_wh, self.max_processed_image_dim_px) - 1
            mask_level = max(0, mask_level)
            mask_generator = processing_cls(image=slide_obj.image,
                                    src_f=slide_obj.src_f,
                                    level=mask_level,
                                    series=slide_obj.series,
                                    reader=slide_obj.reader)

            mask = mask_generator.create_mask()

        mask_s = warp_tools.get_shape(mask)[0:2]/warp_tools.get_shape(slide_obj.image)[0:2]
        assert np.isclose(mask_s[0], mask_s[1], atol=10**-2), print("mask does not appear to based on scaled copy of Slide's image")
        if np.any(mask.shape[0:2] != slide_obj.image.shape[0:2]):
            mask = warp_tools.resize_img(mask, slide_obj.image.shape[0:2])

        mask_bbox = warp_tools.xy2bbox(warp_tools.mask2xy(mask))
        small_cropped_shape_wh = mask_bbox[2:]

        # Use mask to crop image

        # Determine how large image needs to be so that cropped region has same max dimension as the current image
        crop_s = np.min(self.max_processed_image_dim_px/small_cropped_shape_wh)
        crop_max_dim = np.max(np.array(mask.shape[0:2])*crop_s)

        # Draw bbox around crop
        crop_level = slide_tools.get_level_idx(slide_obj.slide_dimensions_wh, crop_max_dim) - 1
        crop_level = max(0, crop_level)
        crop_resize_s = np.min(crop_max_dim/slide_obj.slide_dimensions_wh[crop_level])
        img_to_crop = warp_tools.rescale_img(slide_obj.slide2vips(level=crop_level), scaling=crop_resize_s)

        crop_bbox = mask_bbox*crop_s
        cropped_vips = img_to_crop.extract_area(*crop_bbox)

        cropped = warp_tools.vips2numpy(cropped_vips)
        uncropped_shape_rc = warp_tools.get_shape(img_to_crop)[0:2]
        original_shape_rc = warp_tools.get_shape(mask)[0:2]

        return cropped, mask, original_shape_rc, uncropped_shape_rc, crop_bbox

    def process_imgs(self, processor_dict):
        if os.path.exists(self.processed_dir):
            n_in_processed_dir = len(os.listdir(self.processed_dir))
            if n_in_processed_dir != self.size:
                msg = (f"It appears the destination directory, {self.dst_dir}, is being reused, but that the number of images (n={self.size})"
                   f" in this run differs than previous runs (n={n_in_processed_dir})."
                   f" This may result in confusing naming of thumbnail files, but the actual registration should be unaffected."
                   f" To avoid this, please either delete {self.dst_dir}, or change the"
                   f" `name` parameter when initializing the Valis object to something besides `None` or '{self.name}'")
                valtils.print_warning(msg)

        pathlib.Path(self.processed_dir).mkdir(exist_ok=True, parents=True)

        for i, slide_obj in enumerate(tqdm.tqdm(self.slide_dict.values(), desc=PROCESS_IMG_MSG, unit="image")):

            processing_cls, processing_kwargs = processor_dict[slide_obj.name]

            if self.crop_for_rigid_reg:
                slide_obj.rigid_cropped = True
                img_to_process, mask, uncropped_unscaled_processed_shape_rc, uncropped_shape_rc, crop_bbox = self.get_roi_for_processing(slide_obj, processing_cls)
            else:
                slide_obj.rigid_cropped = False
                # Create later: mask, original_processed_shape_rc, uncropped_shape_rc, crop_bbox
                img_shape_rc = warp_tools.get_shape(slide_obj.image)[0:2]
                if np.max(img_shape_rc) > self.max_processed_image_dim_px:
                    processing_s = np.min(self.max_processed_image_dim_px/img_shape_rc)
                    img_to_process = warp_tools.rescale_img(slide_obj.image, processing_s)
                else:
                    img_to_process = slide_obj.image

            processing_level = slide_tools.get_level_idx(slide_obj.slide_dimensions_wh, self.max_processed_image_dim_px) - 1
            processing_level = max(0, processing_level)
            processor = processing_cls(image=img_to_process,
                                        src_f=slide_obj.src_f,
                                        level=processing_level,
                                        series=slide_obj.series,
                                        reader=slide_obj.reader)
            try:
                processed_img = processor.process_image(**processing_kwargs)
            except TypeError:
                # processor.process_image doesn't take kwargs
                processed_img = processor.process_image()

            processed_img = exposure.rescale_intensity(processed_img, out_range=(0, 255)).astype(np.uint8)

            # Ensure processed image shape is within specified limit
            processed_shape_rc = warp_tools.get_shape(processed_img)[0:2]
            if np.max(processed_shape_rc) > self.max_processed_image_dim_px:
                s = np.min(self.max_processed_image_dim_px/processed_shape_rc)
                processed_img = warp_tools.rescale_img(processed_img, s)
                processed_shape_rc = warp_tools.get_shape(processed_img)[0:2]

            if not self.crop_for_rigid_reg:
                uncropped_shape_rc = processed_shape_rc
                uncropped_unscaled_processed_shape_rc = processed_shape_rc
                crop_bbox = np.array([0, 0, *processed_shape_rc[::-1]])

                if self.create_masks:
                    mask = processor.create_mask()
                    mask_s = warp_tools.get_shape(mask)[0:2]/processed_shape_rc
                    assert np.isclose(mask_s[0], mask_s[1], atol=10**-2), print("mask does not appear to based on scaled copy of Slide's image")
                    if np.any(mask.shape[0:2] != processed_shape_rc):
                        mask = warp_tools.resize_img(mask, processed_shape_rc)

                else:
                    mask = np.full(processor.original_shape_rc, 255, dtype=np.uint8)

            # Set attributes related to processed image's shape
            processed_f_out = os.path.join(self.processed_dir, slide_obj.name + ".png")
            slide_obj.processed_img_f = processed_f_out
            slide_obj.processed_img = processed_img
            slide_obj.processed_img_shape_rc = uncropped_unscaled_processed_shape_rc
            slide_obj.rigid_reg_mask = mask
            slide_obj.uncropped_processed_img_shape_rc = uncropped_shape_rc
            slide_obj.processed_crop_bbox = crop_bbox

            warp_tools.save_img(processed_f_out, processed_img)

            # Save thumbnails of mask
            if self.crop_for_rigid_reg or self.create_masks:
                pathlib.Path(self.mask_dir).mkdir(exist_ok=True, parents=True)
                thumbnail_mask = self.create_thumbnail(mask, thumbnail_size=self.thumbnail_size)
                if slide_obj.img_type == slide_tools.IHC_NAME:
                    thumbnail_img = self.create_thumbnail(slide_obj.image, thumbnail_size=self.thumbnail_size)
                else:
                    thumbnail_img = self.create_thumbnail(slide_obj.pad_cropped_processed_img(), thumbnail_size=self.thumbnail_size)

                thumbnail_mask_outline = viz.draw_outline(thumbnail_img, thumbnail_mask)
                outline_f_out = os.path.join(self.mask_dir, f'{slide_obj.name}.png')
                warp_tools.save_img(outline_f_out, thumbnail_mask_outline)

        if self.norm_method is not None:
            self.target_processing_hist, self.target_processing_stats = self.normalize_images()

    def crop_rigid_reg_mask(self, slide_obj, mask=None):
        if mask is None:
            mask = slide_obj.rigid_reg_mask

        if not slide_obj.rigid_cropped:
            return mask

        vips_mask = warp_tools.numpy2vips(mask)
        scaled_mask = warp_tools.resize_img(vips_mask, slide_obj.uncropped_processed_img_shape_rc, interp_method="nearest")
        cropped_mask = scaled_mask.extract_area(*slide_obj.processed_crop_bbox)
        if isinstance(mask, np.ndarray):
            cropped_mask = warp_tools.vips2numpy(cropped_mask)

        return cropped_mask

    def denoise_images(self):
        for i, slide_obj in enumerate(tqdm.tqdm(self.slide_dict.values(), desc=DENOISE_MSG, unit="image")):
            if slide_obj.rigid_reg_mask is None:
                is_ihc = slide_obj.img_type == slide_tools.IHC_NAME
                _, tissue_mask = preprocessing.create_tissue_mask(slide_obj.image, is_ihc)
                mask_bbox = warp_tools.xy2bbox(warp_tools.mask2xy(tissue_mask))
                c0, r0 = mask_bbox[:2]
                c1, r1 = mask_bbox[:2] + mask_bbox[2:]
                denoise_mask = np.zeros_like(tissue_mask)
                denoise_mask[r0:r1, c0:c1] = 255
            else:
                denoise_mask = slide_obj.rigid_reg_mask

            denoise_mask = self.crop_rigid_reg_mask(slide_obj=slide_obj, mask=denoise_mask)
            denoised = preprocessing.denoise_img(slide_obj.processed_img, mask=denoise_mask)
            warp_tools.save_img(slide_obj.processed_img_f, denoised)

    def normalize_images(self, all_histogram=None, all_img_stats=None):
        """Normalize intensity values in images
        """
        img_list = [None] * self.size
        mask_list = [None] * self.size
        for i, slide_obj in enumerate(self.slide_dict.values()):
            vips_img = pyvips.Image.new_from_file(slide_obj.processed_img_f)
            img = warp_tools.vips2numpy(vips_img)
            img_list[i] = img
            mask_list[i] = self.crop_rigid_reg_mask(slide_obj)

        if all_histogram is None or all_img_stats is None:
            all_histogram, all_img_stats = preprocessing.collect_img_stats(img_list, mask_list=mask_list)

        for i, slide_obj in enumerate(tqdm.tqdm(self.slide_dict.values(), desc=NORM_IMG_MSG, unit="image")):
            img = img_list[i]
            if self.norm_method == "histo_match":
                normed_img = preprocessing.match_histograms(img, all_histogram)
            elif self.norm_method == "img_stats":
                normed_img = preprocessing.norm_img_stats(img, all_img_stats)

            normed_img = exposure.rescale_intensity(normed_img, out_range=(0, 255)).astype(np.uint8)
            slide_obj.processed_img = normed_img

            warp_tools.save_img(slide_obj.processed_img_f, normed_img)

        return all_histogram, all_img_stats

    def create_thumbnail(self, img, rescale_color=False, thumbnail_size=DEFAULT_THUMBNAIL_SIZE):
        """Create thumbnail image to view results
        """

        is_vips = isinstance(img, pyvips.Image)

        img_shape = warp_tools.get_shape(img)
        scaling = np.min(thumbnail_size/np.array(img_shape[:2]))
        if scaling < 1:
            thumbnail = warp_tools.rescale_img(img, scaling)
        else:
            thumbnail = img

        if rescale_color is True:
            if is_vips:
                # Convert to numpy to rescale
                thumbnail = warp_tools.vips2numpy(img)
            thumbnail = exposure.rescale_intensity(thumbnail, out_range=(0, 255)).astype(np.uint8)

            if is_vips:
                # Convert back to pyvips
                thumbnail = warp_tools.numpy2vips(thumbnail)

        return thumbnail

    def draw_overlap_img(self, img_list, blending="weighted"):
        """Create image showing the overlap of registered images
        blending="weighted"
        blending="light"
        """

        cmap = viz.jzazbz_cmap()
        overlap_img = viz.create_overlap_img(img_list, cmap=cmap, blending=blending)

        overlap_img = exposure.equalize_adapthist(overlap_img)
        overlap_img = exposure.rescale_intensity(overlap_img, out_range=(0, 255)).astype(np.uint8)
        return overlap_img

    def get_ref_img_mask(self):
        """Create mask that covers reference image

        Returns
        -------
        mask : ndarray
            Mask that covers reference image in registered images
        mask_bbox_xywh : tuple of int
            XYWH of mask in reference image

        """
        ref_slide = self.get_ref_slide()
        ref_shape_wh = ref_slide.processed_img_shape_rc[::-1]

        uw_mask = np.full(ref_shape_wh[::-1], 255, dtype=np.uint8)
        mask = warp_tools.warp_img(uw_mask, ref_slide.M,
                                   out_shape_rc=ref_slide.reg_img_shape_rc)

        reg_txy = -ref_slide.M[0:2, 2]
        mask_bbox_xywh = np.array([*reg_txy, *ref_shape_wh])

        return mask, mask_bbox_xywh

    def get_all_overlap_mask(self):
        """Create mask that covers all tissue


        Returns
        -------
        mask : ndarray
            Mask that covers reference image in registered images
        mask_bbox_xywh : tuple of int
            XYWH of mask in reference image

        """

        ref_slide = self.get_ref_slide()
        combo_mask = np.zeros(self.aligned_img_shape_rc, dtype=int)
        for slide_obj in self.slide_dict.values():
            warped_img_mask = warp_tools.warp_img(slide_obj.rigid_reg_mask,
                                                  M=slide_obj.M,
                                                  out_shape_rc=slide_obj.reg_img_shape_rc,
                                                  interp_method="nearest")

            combo_mask[warped_img_mask > 0] += 1

        temp_mask = 255*filters.apply_hysteresis_threshold(combo_mask, 0.5, self.size-0.5).astype(np.uint8)
        if temp_mask.max() == 0:
            lt, ht, _  = filters.threshold_multiotsu(combo_mask, 4)
            temp_mask = 255*filters.apply_hysteresis_threshold(combo_mask, lt, ht).astype(np.uint8)

        mask = 255*ndimage.binary_fill_holes(temp_mask).astype(np.uint8)
        mask = preprocessing.mask2contours(mask)

        mask_bbox_xywh = warp_tools.xy2bbox(warp_tools.mask2xy(mask))

        return mask, mask_bbox_xywh

    def get_null_overlap_mask(self):
        """Create mask that covers all of the image.
        Not really a mask


        Returns
        -------
        mask : ndarray
            Mask that covers reference image in registered images
        mask_bbox_xywh : tuple of int
            XYWH of mask in reference image

        """
        reg_shape = self.aligned_img_shape_rc
        mask = np.full(reg_shape, 255, dtype=np.uint8)
        mask_bbox_xywh = np.array([0, 0, reg_shape[1], reg_shape[0]])

        return mask, mask_bbox_xywh

    def create_crop_masks(self):
        """Create masks based on rigid registration

        """
        mask_dict = {}
        mask_dict[CROP_REF] = self.get_ref_img_mask()
        mask_dict[CROP_OVERLAP] = self.get_all_overlap_mask()
        mask_dict[CROP_NONE] = self.get_null_overlap_mask()
        self.mask_dict = mask_dict

    def get_crop_mask(self, overlap_type):
        """Get overlap mask and bounding box

        Returns
        -------
        mask : ndarray
            Mask

        mask_xywh : tuple
            XYWH for bounding box around mask

        """
        if overlap_type is None:
            overlap_type = CROP_NONE

        return self.mask_dict[overlap_type]

    def extract_rigid_transforms_from_serial_rigid(self, rigid_registrar):
        """
        If rigid transforms were found on cropped images, they will need to be
        altered to account for cropping and scaling.
        """

        slide_M_dict = {}
        matches_dict = {}
        cropped_M_dict = {}

        ref_slide = self.get_ref_slide()
        for moving_idx, fixed_idx in rigid_registrar.iter_order:
            img_obj = rigid_registrar.img_obj_list[moving_idx]
            prev_img_obj = rigid_registrar.img_obj_list[fixed_idx]

            if fixed_idx == rigid_registrar.reference_img_idx:
                prev_M = np.eye(3)
                slide_M_dict[ref_slide.name] = prev_M
                cropped_M_dict[ref_slide.name] = prev_M

            slide_obj = self.get_slide(img_obj.name)
            prev_slide_obj = self.get_slide(prev_img_obj.name)

            match_info = img_obj.match_dict[prev_img_obj]

            # Put points back in uncropped images
            rx, ry = img_obj.reflection_M[[0, 1], [0, 1]] < 0
            any_reflections = any([rx, ry])
            if any_reflections:
                uncropped_reflect_M = warp_tools.get_reflection_M(rx, ry, slide_obj.processed_img_shape_rc)
                kp1_xy = warp_tools.warp_xy(match_info.matched_kp1_xy, img_obj.reflection_M)
            else:
                kp1_xy = match_info.matched_kp1_xy

            s = np.array(slide_obj.processed_img_shape_rc)/np.array(slide_obj.uncropped_processed_img_shape_rc)
            kp1_xy_in_uncropped_scaled = s*(kp1_xy + slide_obj.processed_crop_bbox[0:2])

            prev_s = np.array(prev_slide_obj.processed_img_shape_rc)/np.array(prev_slide_obj.uncropped_processed_img_shape_rc)
            kp2_xy_in_uncropped_scaled = prev_s*(match_info.matched_kp2_xy + prev_slide_obj.processed_crop_bbox[0:2])
            kp2_xy_in_uncropped_warped = warp_tools.warp_xy(kp2_xy_in_uncropped_scaled, M=prev_M)

            # Estimate transform
            M_tform = rigid_registrar.transformer
            M_tform.estimate(kp2_xy_in_uncropped_warped, kp1_xy_in_uncropped_scaled)

            if any_reflections:
                scaled_M = uncropped_reflect_M @ M_tform.params
            else:
                scaled_M = M_tform.params

            scaled_M = rigid_registrar.check_M(kp1_xy_in_uncropped_scaled, kp2_xy_in_uncropped_warped, scaled_M)
            prev_M = scaled_M


            slide_M_dict[slide_obj.name] = scaled_M
            cropped_M_dict[slide_obj.name] = img_obj.M

            # Update match dictionary
            uncropped_matches = {slide_obj.name: kp1_xy_in_uncropped_scaled,
                                 prev_slide_obj.name: kp2_xy_in_uncropped_scaled}

            matches_dict[slide_obj.name] = uncropped_matches

        # Determine size of output images and any padding need to get them all to fit
        min_x = np.inf
        max_x = 0
        min_y = np.inf
        max_y = 0
        for slide_obj in self.slide_dict.values():
            temp_M = slide_M_dict[slide_obj.name]
            corners_xy = warp_tools.get_corners_of_image(slide_obj.processed_img_shape_rc)[:, ::-1]
            warped_corners = warp_tools.warp_xy(corners_xy, M=temp_M)

            min_x = np.min([np.min(warped_corners[:, 0]), min_x])
            max_x = np.max([np.max(warped_corners[:, 0]), max_x])
            min_y = np.min([np.min(warped_corners[:, 1]), min_y])
            max_y = np.max([np.max(warped_corners[:, 1]), max_y])

        # Determine size of registered image.
        pad_T = np.identity(3)
        pad_T[0, 2] = min_x
        pad_T[1, 2] = min_y

        w = int(np.ceil(max_x - min_x))
        h = int(np.ceil(max_y - min_y))
        registerd_out_shape_rc = np.array([h, w])

        for slide_obj in self.slide_dict.values():
            M = slide_M_dict[slide_obj.name] @ pad_T
            slide_M_dict[slide_obj.name] = M

        cropped_registerd_out_shape_rc = rigid_registrar.img_obj_list[0].registered_shape_rc

        return slide_M_dict, registerd_out_shape_rc, cropped_M_dict, cropped_registerd_out_shape_rc, matches_dict


    def get_cropped_img_for_rigid_warp(self, slide_obj):
        level = slide_tools.get_level_idx(slide_obj.slide_dimensions_wh, np.max(slide_obj.uncropped_processed_img_shape_rc))
        if level > 0:
            level -= 1

        vips_img = slide_obj.slide2vips(level)
        vips_img = warp_tools.resize_img(vips_img, slide_obj.uncropped_processed_img_shape_rc)
        vips_cropped_img = vips_img.extract_area(*slide_obj.processed_crop_bbox)
        cropped_img = warp_tools.vips2numpy(vips_cropped_img)

        return cropped_img

    def rigid_register_partial(self, tform_dict=None):
        """Perform rigid registration using provided parameters

        Still sorts images by similarity for use with non-rigid registration.

        tform_dict : dictionary
            Dictionary with rigid registration parameters. Each key is the image's file name, and
            the values are another dictionary with transformation parameters:
                M: 3x3 inverse transformation matrix. Found by determining how to align fixed to moving.
                    If M was found by determining how to align moving to fixed, then it will need to be inverted

                transformation_src_shape_rc: shape (row, col) of image used to find the rigid transformation. If
                    not provided, then it is assumed to be the shape of the level 0 slide
                transformation_dst_shape_rc: shape of registered image. If not presesnt, but a reference was provided
                    and `transformation_src_shape_rc` was not provided, this is assumed to be the shape of the reference slide

            If None, then all rigid M will be the identity matrix
        """


        # Still need to sort images #
        rigid_registrar = serial_rigid.SerialRigidRegistrar(self.processed_dir,
                                        imgs_ordered=self.imgs_ordered,
                                        reference_img_f=self.reference_img_f,
                                        name=self.name,
                                        align_to_reference=self.align_to_reference)

        feature_detector = self.rigid_reg_kwargs[FD_KEY]
        matcher = self.rigid_reg_kwargs[MATCHER_KEY]
        matcher_for_sorting = self.rigid_reg_kwargs[MATCHER_FOR_SORTING_KEY]
        similarity_metric = self.rigid_reg_kwargs[SIM_METRIC_KEY]
        transformer = self.rigid_reg_kwargs[TRANSFORMER_KEY]

        # print("\n======== Detecting features\n")
        rigid_registrar.generate_img_obj_list(feature_detector, valis_obj=self)

        if self.create_masks:
            # Remove feature points outside of mask
            for img_obj in rigid_registrar.img_obj_dict.values():
                slide_obj = self.get_slide(img_obj.name)
                features_in_mask_idx = warp_tools.get_xy_inside_mask(xy=img_obj.kp_pos_xy, mask=slide_obj.rigid_reg_mask)
                if len(features_in_mask_idx) > 0:
                    img_obj.kp_pos_xy = img_obj.kp_pos_xy[features_in_mask_idx, :]
                    img_obj.desc = img_obj.desc[features_in_mask_idx, :]


        # print("\n======== Matching images\n")
        if rigid_registrar.aleady_sorted:
            rigid_registrar.match_sorted_imgs(matcher_for_sorting, keep_unfiltered=False, valis_obj=self)

            for i, img_obj in enumerate(rigid_registrar.img_obj_list):
                img_obj.stack_idx = i

        else:
            rigid_registrar.match_imgs(matcher_for_sorting, keep_unfiltered=False, valis_obj=self)

            # print("\n======== Sorting images\n")
            rigid_registrar.build_metric_matrix(metric=similarity_metric)
            rigid_registrar.sort()

        rigid_registrar.get_iter_order()
        rigid_registrar.distance_metric_name = matcher.metric_name
        rigid_registrar.distance_metric_type = matcher.metric_type

        do_rematch = (matcher_for_sorting.__class__.__name__ != matcher.__class__.__name__) \
            or (matcher_for_sorting.feature_detector.__class__.__name__ != matcher.feature_detector.__class__.__name__)
        if do_rematch:
            msg = (f"Images sorted using {matcher_for_sorting.__class__.__name__} features. "
                   f"Will now use {matcher.__class__.__name__} to match images using {matcher.feature_detector.__class__.__name__} features")
            # Images sorted with different feature_detector, but need to matched using matcher's feature_detector
            valtils.print_warning(msg, None)
            rigid_registrar.rematch(matcher_obj=matcher, keep_unfiltered=False, valis_obj=self)

        if rigid_registrar.size > 2:
            rigid_registrar.update_match_dicts_with_neighbor_filter(transformer, matcher)

        if self.reference_img_f is not None:
            ref_name = self.name_dict[self.reference_img_f]
        else:
            ref_name = valtils.get_name(rigid_registrar.reference_img_f)
            if self.do_rigid is not False:
                msg = " ".join([f"Best to specify `{REF_IMG_KEY}` when manually providing `{TFORM_MAT_KEY}`.",
                       f"Setting this image to be {ref_name}"])

                valtils.print_warning(msg)

        # Get output shapes #
        if tform_dict is None:
            named_tform_dict = {o.name: {"M":np.eye(3)} for o in rigid_registrar.img_obj_list}
        else:
            named_tform_dict = {valtils.get_name(k):v for k, v in tform_dict.items()}

        # Get output shapes #
        rigid_ref_obj = rigid_registrar.img_obj_dict[ref_name]
        ref_slide_obj = self.get_ref_slide()
        if ref_name in named_tform_dict.keys():
            ref_tforms = named_tform_dict[ref_name]
            if TFORM_SRC_SHAPE_KEY in ref_tforms:
                ref_tform_src_shape_rc = ref_tforms[TFORM_SRC_SHAPE_KEY]
            else:
                ref_tform_src_shape_rc = ref_slide_obj.slide_dimensions_wh[0][::-1]

            if TFORM_DST_SHAPE_KEY in ref_tforms:
                temp_out_shape_rc = ref_tforms[TFORM_DST_SHAPE_KEY]
            else:
                # Assume M was found by aligning to level 0 reference
                temp_out_shape_rc = ref_slide_obj.slide_dimensions_wh[0][::-1]

            ref_to_reg_sxy = (np.array(rigid_ref_obj.image.shape)/np.array(ref_tform_src_shape_rc))[::-1]
            out_rc = np.round(temp_out_shape_rc*ref_to_reg_sxy).astype(int)

        else:
            out_rc = rigid_ref_obj.image.shape

        scaled_M_dict = {}
        for img_name, img_tforms in named_tform_dict.items():
            matching_rigid_obj = rigid_registrar.img_obj_dict[img_name]
            matching_slide_obj = self.slide_dict[img_name]

            if TFORM_SRC_SHAPE_KEY in img_tforms:
                og_src_shape_rc = img_tforms[TFORM_SRC_SHAPE_KEY]
            else:
                og_src_shape_rc = matching_slide_obj.slide_dimensions_wh[0][::-1]

            temp_M = img_tforms[TFORM_MAT_KEY]
            if temp_M.shape[0] == 2:
                temp_M = np.vstack([temp_M, [0, 0, 1]])

            if TFORM_DST_SHAPE_KEY in img_tforms:
                og_dst_shape_rc = img_tforms[TFORM_DST_SHAPE_KEY]
            else:
                og_dst_shape_rc = ref_slide_obj.slide_dimensions_wh[0][::-1]

            img_corners_xy = warp_tools.get_corners_of_image(matching_rigid_obj.image.shape)[::-1]
            warped_corners = warp_tools.warp_xy(img_corners_xy, M=temp_M,
                                    transformation_src_shape_rc=og_src_shape_rc,
                                    transformation_dst_shape_rc=og_dst_shape_rc,
                                    src_shape_rc=matching_rigid_obj.image.shape,
                                    dst_shape_rc=out_rc)
            M_tform = transform.ProjectiveTransform()
            M_tform.estimate(warped_corners, img_corners_xy)
            for_reg_M = M_tform.params
            scaled_M_dict[matching_rigid_obj.name] = for_reg_M
            matching_rigid_obj.M = for_reg_M

        # Find M if not provided
        for moving_idx, fixed_idx in tqdm.tqdm(rigid_registrar.iter_order, desc=TRANSFORM_MSG, unit="image"):
            img_obj = rigid_registrar.img_obj_list[moving_idx]
            if img_obj.name in scaled_M_dict:
                print(f"Skipping {img_obj.name}, which has affine transform provided.")
                continue

            prev_img_obj = rigid_registrar.img_obj_list[fixed_idx]
            img_obj.fixed_obj = prev_img_obj

            print(f"finding M for {img_obj.name}, which is being aligned to {prev_img_obj.name}")

            if fixed_idx == rigid_registrar.reference_img_idx:
                prev_M = np.eye(3)

            to_prev_match_info = img_obj.match_dict[prev_img_obj]
            src_xy = to_prev_match_info.matched_kp1_xy
            dst_xy = warp_tools.warp_xy(to_prev_match_info.matched_kp2_xy, prev_M)

            transformer.estimate(dst_xy, src_xy)
            img_obj.M = transformer.params

            prev_M = img_obj.M

        # Add registered image
        for img_obj in rigid_registrar.img_obj_list:
            img_obj.M_inv = np.linalg.inv(img_obj.M)

            img_obj.registered_img = warp_tools.warp_img(img=img_obj.image,
                                                        M=img_obj.M,
                                                        out_shape_rc=out_rc)

            img_obj.registered_shape_rc = img_obj.registered_img.shape[0:2]

        return rigid_registrar

    def rigid_register(self):
        """Rigidly register slides

        Also saves thumbnails of rigidly registered images.

        Returns
        -------
        rigid_registrar : SerialRigidRegistrar
            SerialRigidRegistrar object that performed the rigid registration.

        """

        if self.denoise_rigid:
            self.denoise_images()

        print("\n==== Rigid registration\n")
        if self.do_rigid is True:
            rigid_registrar = serial_rigid.register_images(img_dir=self.processed_dir,
                                                           align_to_reference=self.align_to_reference,
                                                           valis_obj=self,
                                                           **self.rigid_reg_kwargs)
        else:
            if isinstance(self.do_rigid, dict):
                # User provided transforms
                rigid_tforms = self.do_rigid
            elif self.do_rigid is False:
                # Skip rigid registration
                rigid_tforms = None

            rigid_registrar = self.rigid_register_partial(tform_dict=rigid_tforms)

        self.end_rigid_time = time()
        self.rigid_registrar = rigid_registrar

        if rigid_registrar is False:
            msg = "Rigid registration failed"
            valtils.print_warning(msg, rgb=Fore.RED)

            return False

        self.reference_img_idx = rigid_registrar.reference_img_idx

        ref_slide = self.slide_dict[valtils.get_name(rigid_registrar.reference_img_f)]
        self.reference_img_f = ref_slide.src_f

        rigid_transform_dict, rigid_reg_shape, cropped_M_dict, cropped_registerd_out_shape_rc, rigid_matches_dict = \
              self.extract_rigid_transforms_from_serial_rigid(rigid_registrar)

        self.aligned_img_shape_rc = rigid_reg_shape
        n_digits = len(str(rigid_registrar.size))
        for slide_reg_obj in rigid_registrar.img_obj_list:
            slide_obj = self.slide_dict[slide_reg_obj.name]
            slide_obj.M_for_cropped = cropped_M_dict[slide_obj.name]
            slide_obj.rigid_reg_cropped_shape_rc = cropped_registerd_out_shape_rc
            slide_obj.M = rigid_transform_dict[slide_obj.name]
            slide_obj.reg_img_shape_rc = rigid_reg_shape
            slide_obj.stack_idx = slide_reg_obj.stack_idx
            slide_obj.rigid_reg_img_f = os.path.join(self.reg_dst_dir,
                                                     str.zfill(str(slide_obj.stack_idx), n_digits) + "_" + slide_obj.name + ".png")
            if slide_obj.image.ndim > 2:
                # Won't know if single channel image is processed RGB (bight bg) or IF channel (dark bg)
                slide_obj.get_bg_color_px_pos()

            if slide_reg_obj.stack_idx == self.reference_img_idx:
                continue

            if slide_reg_obj.fixed_obj is None:
                fixed_name = ref_slide.name
            else:
                fixed_name = slide_reg_obj.fixed_obj.name

            fixed_slide = self.slide_dict[fixed_name]
            slide_obj.fixed_slide = fixed_slide

            match_dict = rigid_matches_dict[slide_obj.name]
            slide_obj.xy_matched_to_prev = match_dict[slide_obj.name]
            slide_obj.xy_in_prev = match_dict[fixed_slide.name]

        self.create_crop_masks()
        overlap_mask, overlap_mask_bbox_xywh = self.get_crop_mask(self.crop)

        overlap_mask_bbox_xywh = overlap_mask_bbox_xywh.astype(int)

        # Create original overlap image #
        pathlib.Path(self.overlap_dir).mkdir(exist_ok=True, parents=True)
        self.original_overlap_img = self.create_original_composite_img(rigid_registrar)
        original_overlap_img_fout = os.path.join(self.overlap_dir, self.name + "_original_overlap.png")
        warp_tools.save_img(original_overlap_img_fout,  self.original_overlap_img)

        pathlib.Path(self.reg_dst_dir).mkdir(exist_ok=True, parents=True)

        if self.denoise_rigid:

            # Processed image may have been denoised for rigid registration. Replace with unblurred image
            for img_obj in rigid_registrar.img_obj_list:
                matching_slide = self.slide_dict[img_obj.name]
                reg_img = warp_tools.warp_img(matching_slide.processed_img, M=img_obj.M, out_shape_rc=img_obj.registered_shape_rc)
                img_obj.registered_img = reg_img
                img_obj.image = matching_slide.processed_img

        rigid_img_list = [self.create_thumbnail(img_obj.registered_img, thumbnail_size=self.thumbnail_size) for img_obj in rigid_registrar.img_obj_list]
        thumbnail_s = np.min(np.array(rigid_img_list[0].shape)/np.array(rigid_registrar.img_obj_list[0].registered_img.shape[0:2]))
        self.rigid_overlap_img = self.draw_overlap_img(img_list=rigid_img_list)

        rigid_overlap_img_fout = os.path.join(self.overlap_dir, self.name + "_rigid_overlap.png")
        warp_tools.save_img(rigid_overlap_img_fout, self.rigid_overlap_img, thumbnail_size=self.thumbnail_size)

        # Overwrite black and white processed images #
        for slide_name, slide_obj in self.slide_dict.items():
            slide_reg_obj = rigid_registrar.img_obj_dict[slide_name]

            if not slide_obj.is_rgb:
                img_to_warp = slide_obj.pad_cropped_processed_img()
            else:
                img_to_warp = slide_obj.image
            img_to_warp = warp_tools.resize_img(img_to_warp, slide_obj.processed_img_shape_rc)

            warped_img = slide_obj.warp_img(img_to_warp, non_rigid=False, crop=self.crop)
            warp_tools.save_img(slide_obj.rigid_reg_img_f, warped_img.astype(np.uint8), thumbnail_size=self.thumbnail_size)

            # Replace processed image with a thumbnail #
            warp_tools.save_img(slide_obj.processed_img_f, slide_reg_obj.image, thumbnail_size=self.thumbnail_size)

        return rigid_registrar

    def micro_rigid_register(self):
        micro_rigid_registar = self.micro_rigid_registrar_cls(val_obj=self, **self.micro_rigid_registrar_params)
        micro_rigid_registar.register()

        # Not all pairs will have keept high resolution M, so re-estimate M based on final matches
        slide_idx, slide_names = list(zip(*[[slide_obj.stack_idx, slide_obj.name] for slide_obj in self.slide_dict.values()]))
        slide_order = np.argsort(slide_idx) # sorts ascending
        slide_list = [self.slide_dict[slide_names[i]] for i in slide_order]
        ref_slide = self.get_ref_slide()
        for moving_idx, fixed_idx in self.iter_order:
            slide_obj = slide_list[moving_idx]
            fixed_slide = slide_list[fixed_idx]

            if fixed_idx == self.reference_img_idx:
                prev_M = ref_slide.M

            src_xy = slide_obj.xy_matched_to_prev
            dst_xy = warp_tools.warp_xy(slide_obj.xy_in_prev, prev_M)
            transformer = getattr(transform, self.transform_str)()
            transformer.estimate(dst_xy, src_xy)
            slide_obj.M = transformer.params

            prev_M = transformer.params


        # Draw in same order as regular rigid registration
        draw_list = [self.slide_dict[img_obj.name] for img_obj in self.rigid_registrar.img_obj_list]

        thumbnail_s = np.min(self.thumbnail_size/np.array(draw_list[0].reg_img_shape_rc))
        rigid_img_list = [warp_tools.rescale_img(slide_obj.warp_img(slide_obj.pad_cropped_processed_img(), non_rigid=False), scaling=thumbnail_s) for slide_obj in draw_list]

        self.micro_rigid_overlap_img = self.draw_overlap_img(rigid_img_list)

        micro_rigid_overlap_img_fout = os.path.join(self.overlap_dir, self.name + "_micro_rigid_overlap.png")
        warp_tools.save_img(micro_rigid_overlap_img_fout, self.micro_rigid_overlap_img)

        # Overwrite rigid registration results and update rigid registrar
        for slide_name, slide_obj in self.slide_dict.items():
            if not slide_obj.is_rgb:
                img_to_warp = slide_obj.processed_img
            else:
                img_to_warp = slide_obj.image
            img_to_warp = warp_tools.resize_img(img_to_warp, slide_obj.processed_img_shape_rc)
            warped_img = slide_obj.warp_img(img_to_warp, non_rigid=False, crop=self.crop)
            warp_tools.save_img(slide_obj.rigid_reg_img_f, warped_img.astype(np.uint8), thumbnail_size=self.thumbnail_size)

            if slide_obj.fixed_slide is None:
                continue
            fixed_slide = slide_obj.fixed_slide
            fixed_rigid_obj = self.rigid_registrar.img_obj_dict[fixed_slide.name]

            rigid_img_obj = self.rigid_registrar.img_obj_dict[slide_obj.name]
            rigid_img_obj.M = slide_obj.M
            rigid_img_obj.M_inv = np.linalg.inv(slide_obj.M)
            rigid_img_obj.registered_img = slide_obj.warp_img(img_to_warp, non_rigid=False, crop=False)

            rigid_img_obj.match_dict[fixed_rigid_obj].matched_kp1_xy = slide_obj.xy_matched_to_prev
            rigid_img_obj.match_dict[fixed_rigid_obj].matched_kp2_xy = slide_obj.xy_in_prev
            rigid_img_obj.match_dict[fixed_rigid_obj].n_matches = slide_obj.xy_in_prev.shape[0]

            fixed_rigid_obj.match_dict[rigid_img_obj].matched_kp1_xy = slide_obj.xy_in_prev
            fixed_rigid_obj.match_dict[rigid_img_obj].matched_kp2_xy = slide_obj.xy_matched_to_prev
            fixed_rigid_obj.match_dict[rigid_img_obj].n_matches = slide_obj.xy_in_prev.shape[0]

    def draw_matches(self, dst_dir, thumbnail_size=DEFAULT_THUMBNAIL_SIZE):
        """Draw and save images of matching features

        Parameters
        ----------
        dst_dir : str
            Where to save the images of the matched features
        """

        dst_dir = str(dst_dir)
        pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)

        slide_idx, slide_names = list(zip(*[[slide_obj.stack_idx, slide_obj.name] for slide_obj in self.slide_dict.values()]))
        slide_order = np.argsort(slide_idx) # sorts ascending
        slide_list = [self.slide_dict[slide_names[i]] for i in slide_order]
        for moving_idx, fixed_idx in self.iter_order:
            moving_slide = slide_list[moving_idx]
            fixed_slide = slide_list[fixed_idx]

            # RGB draw images
            if moving_slide.image.ndim == 3 and moving_slide.is_rgb:
                moving_draw_img = warp_tools.resize_img(moving_slide.image, moving_slide.processed_img_shape_rc)
            else:
                moving_draw_img = moving_slide.pad_cropped_processed_img()

            if fixed_slide.image.ndim == 3 and fixed_slide.is_rgb:
                fixed_draw_img = warp_tools.resize_img(fixed_slide.image, fixed_slide.processed_img_shape_rc)
            else:
                fixed_draw_img = fixed_slide.pad_cropped_processed_img()

            all_matches_img = viz.draw_matches(src_img=moving_draw_img, kp1_xy=moving_slide.xy_matched_to_prev,
                                               dst_img=fixed_draw_img,  kp2_xy=moving_slide.xy_in_prev,
                                               rad=3, alignment='horizontal')
            all_matches_img = self.create_thumbnail(all_matches_img, thumbnail_size=thumbnail_size)
            matches_f_out = os.path.join(dst_dir, f"{self.name}_{moving_slide.name}_to_{fixed_slide.name}_matches.png")
            warp_tools.save_img(matches_f_out, all_matches_img)

    def create_non_rigid_reg_mask(self):
        """
        Get mask for non-rigid registration
        """
        print("Creating non-rigid mask")
        if self.create_masks:
            non_rigid_mask = self._create_mask_from_processed()
        else:
            non_rigid_mask = self._create_non_rigid_reg_mask_from_bbox()

        for slide_obj in self.slide_dict.values():
            slide_obj.non_rigid_reg_mask = non_rigid_mask

        # Save thumbnail of mask
        ref_slide = self.get_ref_slide()
        if ref_slide.img_type == slide_tools.IHC_NAME:
            ref_img = warp_tools.resize_img(ref_slide.image, ref_slide.processed_img_shape_rc)
        else:
            ref_img = ref_slide.pad_cropped_processed_img()

        warped_ref_img = ref_slide.warp_img(img=ref_img, non_rigid=False, crop=CROP_REF)

        pathlib.Path(self.mask_dir).mkdir(exist_ok=True, parents=True)
        thumbnail_img = self.create_thumbnail(warped_ref_img, thumbnail_size=self.thumbnail_size)

        draw_mask = warp_tools.resize_img(non_rigid_mask, ref_slide.reg_img_shape_rc, interp_method="nearest")
        _, overlap_mask_bbox_xywh = self.get_crop_mask(CROP_REF)
        draw_mask = warp_tools.crop_img(draw_mask, overlap_mask_bbox_xywh.astype(int))
        thumbnail_mask = self.create_thumbnail(draw_mask, thumbnail_size=self.thumbnail_size)

        thumbnail_mask_outline = viz.draw_outline(thumbnail_img, thumbnail_mask)
        outline_f_out = os.path.join(self.mask_dir, f'{self.name}_non_rigid_mask.png')
        warp_tools.save_img(outline_f_out, thumbnail_mask_outline)

    def _create_non_rigid_reg_mask_from_bbox(self, slide_list=None):
        """Mask will be bounding box of image overlaps

        """
        ref_slide = self.get_ref_slide()
        combo_mask = np.zeros(ref_slide.reg_img_shape_rc, dtype=int)

        if slide_list is None:
            slide_list = list(self.slide_dict.values())

        for slide_obj in slide_list:
            img_bbox = np.full(slide_obj.processed_img_shape_rc, 255, dtype=np.uint8)
            rigid_mask = slide_obj.warp_img(img_bbox, non_rigid=False, crop=False, interp_method="nearest")
            combo_mask[rigid_mask > 0] += 1

        n = len(slide_list)
        overlap_mask = (combo_mask == n).astype(np.uint8)
        overlap_bbox = warp_tools.xy2bbox(warp_tools.mask2xy(overlap_mask))
        c0, r0 = overlap_bbox[:2]
        c1, r1 = overlap_bbox[:2] + overlap_bbox[2:]

        non_rigid_mask = np.zeros_like(overlap_mask)
        non_rigid_mask[r0:r1, c0:c1] = 255

        return non_rigid_mask

    def _create_mask_from_processed(self, slide_list=None):
        if slide_list is None:
            slide_list = list(self.slide_dict.values())

        combo_mask = np.zeros(self.aligned_img_shape_rc, dtype=int)
        summary_img = np.zeros(self.aligned_img_shape_rc)

        for i, slide_obj in enumerate(slide_list):
            # Determine where images overlap
            rigid_mask = slide_obj.warp_img(slide_obj.rigid_reg_mask, non_rigid=False, crop=False, interp_method="nearest")
            combo_mask[rigid_mask > 0] += 1

            # Caclulate running average
            padded_processed = slide_obj.pad_cropped_processed_img()
            padded_processed[slide_obj.rigid_reg_mask == 0] = 0
            padded_processed[slide_obj.rigid_reg_mask > 0] = exposure.rescale_intensity(padded_processed[slide_obj.rigid_reg_mask > 0], out_range=(0, 255))
            for_summary = slide_obj.warp_img(padded_processed, non_rigid=False, crop=False).astype(float)
            for_summary = exposure.rescale_intensity(for_summary, out_range=(0, 1))
            for_summary = exposure.equalize_adapthist(for_summary)
            summary_img = np.dstack([for_summary, summary_img]).max(axis=2)

        summary_img /= summary_img.max()
        hyst_thresh = min(self.size-0.5, 2)
        combo_mask = 255*filters.apply_hysteresis_threshold(combo_mask, 0.5, hyst_thresh).astype(np.uint8) # At least 2 masks are touching

        # Remake masks, weighting by summary image
        weighted_combo_mask = np.zeros(self.aligned_img_shape_rc, dtype=int)
        weighted_mask_list = [None] * self.size
        for i, slide_obj in enumerate(slide_list):
            warped_processed = slide_obj.warp_img(slide_obj.pad_cropped_processed_img(), non_rigid=False, crop=False).astype(float)
            if combo_mask.max() > 0:
                warped_processed[combo_mask == 0] = 0
            weighted_processed = summary_img*(warped_processed/warped_processed.max())
            weighted_processed = exposure.equalize_adapthist(weighted_processed)
            wt, _ = filters.threshold_multiotsu(weighted_processed)
            weighted_mask = 255*(weighted_processed > wt).astype(np.uint8)
            weighted_mask = preprocessing.mask2contours(weighted_mask, 1)
            weighted_mask_list[i] = weighted_mask
            weighted_combo_mask[weighted_mask > 0] += 1

        temp_non_rigid_mask = 255*filters.apply_hysteresis_threshold(weighted_combo_mask, 0.5, hyst_thresh).astype(np.uint8) # At least 2 masks are touching
        overlap_mask = preprocessing.mask2bbox_mask(temp_non_rigid_mask)

        return overlap_mask

    def _create_non_rigid_reg_mask_from_rigid_masks(self, slide_list=None):
        """
        Get mask that will cover all tissue. Use hysteresis thresholding to ignore
        masked regions found in only 1 image.

        """

        if slide_list is None:
            slide_list = list(self.slide_dict.values())

        combo_mask = np.zeros(self.aligned_img_shape_rc, dtype=int)
        for i, slide_obj in enumerate(slide_list):
            rigid_mask = slide_obj.warp_img(slide_obj.rigid_reg_mask, non_rigid=False, crop=False, interp_method="nearest")
            combo_mask[rigid_mask > 0] += 1

        temp_mask = 255*filters.apply_hysteresis_threshold(combo_mask, 0.5, self.size-0.5).astype(np.uint8)

        # Draw convex hull around each region
        final_mask = 255*ndimage.binary_fill_holes(temp_mask).astype(np.uint8)
        final_mask = preprocessing.mask2contours(final_mask)

        return final_mask

    def pad_displacement(self, dxdy, out_shape_rc, bbox_xywh):

        is_array = not isinstance(dxdy, pyvips.Image)
        if is_array:
            vips_dxdy = warp_tools.numpy2vips(np.dstack(dxdy))
        else:
            vips_dxdy = dxdy

        if bbox_xywh is None:
            full_dxdy = vips_dxdy
        else:
            full_dxdy = vips_dxdy.embed(bbox_xywh[0], bbox_xywh[1],
                                out_shape_rc[1], out_shape_rc[0],
                                extend=pyvips.enums.Extend.BLACK,
                                background=[0,0])

        if is_array:
            full_dxdy = warp_tools.vips2numpy(full_dxdy)
            full_dxdy = np.array([full_dxdy[..., 0], full_dxdy[..., 1]])

        return full_dxdy

    def get_nr_tiling_params(self, non_rigid_registrar_cls,
                             processor_dict,
                             img_specific_args,
                             tile_wh):
        """Get extra parameters need for tiled non-rigid registration

        processor_dict : dict
            Each key should be the filename of the image, and the value either a subclassed
            preprocessing.ImageProcessor, or a list, where the 1st element is the processor,
            and the second element a dictionary of keyword arguments passed to the processor.
            If `None`, then a default processor will be used for each image based on
            the inferred modality.
        """
        if img_specific_args is None:
            img_specific_args = {}

        for slide_obj in self.slide_dict.values():

            processing_cls, processing_kwargs = processor_dict[slide_obj.name]
            tiler_rigid_processing_kwargs = deepcopy(processing_kwargs)

            # Add registration parameters
            tiled_non_rigid_reg_params = {}
            tiled_non_rigid_reg_params[non_rigid_registrars.NR_CLS_KEY] = non_rigid_registrar_cls
            if self.norm_method is not None:
                tiled_non_rigid_reg_params[non_rigid_registrars.NR_STATS_KEY] = self.target_processing_stats
            tiled_non_rigid_reg_params[non_rigid_registrars.NR_TILE_WH_KEY] = tile_wh

            tiled_non_rigid_reg_params[non_rigid_registrars.NR_PROCESSING_CLASS_KEY] = processing_cls
            tiled_non_rigid_reg_params[non_rigid_registrars.NR_PROCESSING_KW_KEY] = tiler_rigid_processing_kwargs
            tiled_non_rigid_reg_params[non_rigid_registrars.NR_PROCESSING_INIT_KW_KEY] = {"src_f": slide_obj.src_f,
                                                                                          "series": slide_obj.series,
                                                                                          "reader": deepcopy(slide_obj.reader)
                                                                                          }
            img_specific_args[slide_obj.name] = tiled_non_rigid_reg_params

        non_rigid_registrar_cls = non_rigid_registrars.NonRigidTileRegistrar

        return non_rigid_registrar_cls, img_specific_args

    def prep_images_for_large_non_rigid_registration(self, max_img_dim,
                                                     processor_dict,
                                                     updating_non_rigid=False,
                                                     mask=None, rgb=False):

        """Scale and process images for non-rigid registration using larger images

        Parameters
        ----------
        max_img_dim : int, optional
            Maximum size of image to be used for non-rigid registration. If None, the whole image
            will be used  for non-rigid registration

        processor_dict : dict
            Each key should be the filename of the image, and the value either a subclassed
            preprocessing.ImageProcessor, or a list, where the 1st element is the processor,
            and the second element a dictionary of keyword arguments passed to the processor.
            If `None`, then a default processor will be used for each image based on
            the inferred modality.

        updating_non_rigid : bool, optional
            If `True`, the slide's current non-rigid registration will be applied
            The new displacements found using these larger images can therefore be used
            to update existing dxdy. If `False`, only the rigid transform will be applied,
            so this will be the first non-rigid transformation.

        mask : ndarray, optional
            Binary image indicating where to perform the non-rigid registration. Should be
            based off an already registered image.

        Returns
        -------
        img_dict : dictionary
            Dictionary that can be passed to a non-rigid registrar

        max_img_dim : int
            Maximum size of image to do non-rigid registration on. May be different
            if the requested size was too big

        scaled_non_rigid_mask : ndarray
            Scaled mask to use for non-rigid registration

        full_out_shape : ndarray of int
            Shape (row, col) of the warped images, without cropping

        mask_bbox_xywh : list
            Bounding box of `mask`. If `mask` is None, then so will `mask_bbox_xywh`

        """

        warp_full_img = max_img_dim is None
        if not warp_full_img:
            all_max_dims = [np.any(np.max(slide_obj.slide_dimensions_wh, axis=1) >= max_img_dim) for slide_obj in self.slide_dict.values()]
            if not np.all(all_max_dims):
                img_maxes = [np.max(slide_obj.slide_dimensions_wh, axis=1)[0] for slide_obj in self.slide_dict.values()]
                smallest_img_max = np.min(img_maxes)
                msg = (f"Requested size of images for non-rigid registration was {max_img_dim}. "
                    f"However, not all images are this large. Setting `max_non_rigid_registration_dim_px` to "
                    f"{smallest_img_max}, which is the largest dimension of the smallest image")
                valtils.print_warning(msg)
                max_img_dim = smallest_img_max

        ref_slide = self.get_ref_slide()

        max_s = np.min(ref_slide.slide_dimensions_wh[0]/np.array(ref_slide.processed_img_shape_rc[::-1]))
        if mask is None:
            if warp_full_img:
                s = max_s
            else:
                s = np.min(max_img_dim/np.array(ref_slide.processed_img_shape_rc))
        else:
            # Determine how big image would have to be to get mask with maxmimum dimension = max_img_dim
            if isinstance(mask, pyvips.Image):
                mask_shape_rc = np.array((mask.height, mask.width))
            else:
                mask_shape_rc = np.array(mask.shape[0:2])

            to_reg_mask_sxy = (mask_shape_rc/np.array(ref_slide.reg_img_shape_rc))[::-1]
            if not np.all(to_reg_mask_sxy == 1):
                # Resize just in case it's huge. Only need bounding box
                reg_size_mask = warp_tools.resize_img(mask, ref_slide.reg_img_shape_rc, interp_method="nearest")
            else:
                reg_size_mask = mask
            reg_size_mask_xy = warp_tools.mask2xy(reg_size_mask)
            to_reg_mask_bbox_xywh = list(warp_tools.xy2bbox(reg_size_mask_xy))
            to_reg_mask_wh = np.round(to_reg_mask_bbox_xywh[2:]).astype(int)
            if warp_full_img:
                s = max_s
            else:
                s = np.min(max_img_dim/np.array(to_reg_mask_wh))

        if s < max_s:
            full_out_shape = self.get_aligned_slide_shape(s)
        else:
            full_out_shape = self.get_aligned_slide_shape(0)

        if mask is None:
            out_shape = full_out_shape
            mask_bbox_xywh = None
        else:
            # If masking, the area will be smaller. Get bounding box
            mask_sxy = (full_out_shape/mask_shape_rc)[::-1]
            mask_bbox_xywh = list(warp_tools.xy2bbox(mask_sxy*reg_size_mask_xy))
            mask_bbox_xywh[2:] = np.round(mask_bbox_xywh[2:]).astype(int)
            mask_bbox_max_xy = np.array(mask_bbox_xywh[0:2]) + np.array(mask_bbox_xywh[2:])
            if np.any(mask_bbox_max_xy > full_out_shape[::-1]):
                # due to rounding , bbox is too big
                mask_shift_xy = mask_bbox_max_xy - full_out_shape[::-1] + 1
                mask_shift_xy[mask_shift_xy < 0] = 0
                mask_bbox_xywh[2:] -= mask_shift_xy

            out_shape = mask_bbox_xywh[2:][::-1]

            if not isinstance(mask, pyvips.Image):
                vips_micro_reg_mask = warp_tools.numpy2vips(mask)
            else:
                vips_micro_reg_mask = mask
            vips_micro_reg_mask = warp_tools.resize_img(vips_micro_reg_mask, full_out_shape, interp_method="nearest")
            vips_micro_reg_mask = warp_tools.crop_img(img=vips_micro_reg_mask, xywh=mask_bbox_xywh)

        if ref_slide.reader.metadata.bf_datatype is not None:
            np_dtype = slide_tools.BF_FORMAT_NUMPY_DTYPE[ref_slide.reader.metadata.bf_datatype]
        else:
            # Assuming images not read by bio-formats are RGB read using from openslide or png, jpeg, etc...
            np_dtype = "uint8"

        displacement_gb = self.size*warp_tools.calc_memory_size_gb(full_out_shape, 2, "float32")
        processed_img_gb = self.size*warp_tools.calc_memory_size_gb(out_shape, 1, "uint8")
        img_gb = self.size*warp_tools.calc_memory_size_gb(out_shape, ref_slide.reader.metadata.n_channels, np_dtype)

        # Size of full displacement fields, all larger processed images, and an image that will be processed
        estimated_gb = img_gb + displacement_gb + processed_img_gb
        use_tiler = False
        if estimated_gb > TILER_THRESH_GB:
            # Avoid having huge displacement fields saved in registrar.
            use_tiler = True

        scaled_img_list = [None] * self.size # Note, will actually warp after normalization if not using tiler
        scaled_mask_list = [None] * self.size
        img_names_list = [None] * self.size
        img_f_list = [None] * self.size
        slide_mask_list = [None] * self.size

        # print("\n======== Preparing images for non-rigid registration\n")
        for slide_obj in tqdm.tqdm(self.slide_dict.values(), desc=PREP_NON_RIGID_MSG, unit="image"):
            # Get image to warp. Likely a larger image scaled down to specified shape #
            src_img_shape_rc, src_M = warp_tools.get_src_img_shape_and_M(transformation_src_shape_rc=slide_obj.processed_img_shape_rc,
                                                                            transformation_dst_shape_rc=slide_obj.reg_img_shape_rc,
                                                                            dst_shape_rc=full_out_shape,
                                                                            M=slide_obj.M)

            if max_img_dim is not None:
                closest_img_levels = np.where(np.max(slide_obj.slide_dimensions_wh, axis=1) < np.max(src_img_shape_rc))[0]
                if len(closest_img_levels) > 0:
                    closest_img_level = closest_img_levels[0] - 1
                else:
                    closest_img_level = len(slide_obj.slide_dimensions_wh) - 1
            else:
                closest_img_level = 0

            closest_img_level = max(closest_img_level, 0)

            vips_level_img = slide_obj.slide2vips(closest_img_level)
            img_to_warp = warp_tools.resize_img(vips_level_img, src_img_shape_rc)

            if updating_non_rigid:
                dxdy = slide_obj.bk_dxdy
            else:
                dxdy = None

            # Get mask covering tissue
            temp_slide_mask = slide_obj.warp_img(slide_obj.rigid_reg_mask, non_rigid=dxdy is not None, crop=False, interp_method="nearest")
            temp_slide_mask = warp_tools.numpy2vips(temp_slide_mask)
            slide_mask = warp_tools.resize_img(temp_slide_mask, full_out_shape, interp_method="nearest")
            if mask_bbox_xywh is not None:
                slide_mask = warp_tools.crop_img(slide_mask, mask_bbox_xywh)

            # Get mask that covers image
            temp_processing_mask = pyvips.Image.black(img_to_warp.width, img_to_warp.height).invert()
            processing_mask = warp_tools.warp_img(img=temp_processing_mask, M=slide_obj.M,
                bk_dxdy=dxdy,
                transformation_src_shape_rc=slide_obj.processed_img_shape_rc,
                transformation_dst_shape_rc=slide_obj.reg_img_shape_rc,
                out_shape_rc=full_out_shape,
                bbox_xywh=mask_bbox_xywh,
                interp_method="nearest")

            if not use_tiler and rgb and slide_obj.is_rgb:
                warped_img = warp_tools.warp_img(img=img_to_warp, M=slide_obj.M,
                    bk_dxdy=dxdy,
                    transformation_src_shape_rc=slide_obj.processed_img_shape_rc,
                    transformation_dst_shape_rc=slide_obj.reg_img_shape_rc,
                    out_shape_rc=full_out_shape,
                    bbox_xywh=mask_bbox_xywh,
                    bg_color=slide_obj.bg_color)
                warped_img_np = warp_tools.vips2numpy(warped_img)
                scaled_img_list[slide_obj.stack_idx] = warped_img_np

            elif not use_tiler and not rgb:
                # Process image using same method for rigid registration
                img_to_warp_np = warp_tools.vips2numpy(img_to_warp)
                processing_cls, processing_kwargs = processor_dict[slide_obj.name]
                non_rigid_processing_kwargs = deepcopy(processing_kwargs)
                processor = processing_cls(image=img_to_warp_np,
                                           src_f=slide_obj.src_f,
                                           level=closest_img_level,
                                           series=slide_obj.series,
                                           reader=slide_obj.reader)

                try:
                    processed_img = processor.process_image(**non_rigid_processing_kwargs)
                except TypeError:
                    # processor.process_image doesn't take kwargs
                    processed_img = processor.process_image()
                warped_img = exposure.rescale_intensity(processed_img, out_range=(0, 255)).astype(np.uint8)
                scaled_img_list[slide_obj.stack_idx] = processed_img
            else:
                if not warp_full_img:
                    warped_img = warp_tools.warp_img(img=img_to_warp, M=slide_obj.M,
                                bk_dxdy=dxdy,
                                transformation_src_shape_rc=slide_obj.processed_img_shape_rc,
                                transformation_dst_shape_rc=slide_obj.reg_img_shape_rc,
                                out_shape_rc=full_out_shape,
                                bbox_xywh=mask_bbox_xywh,
                                bg_color=slide_obj.bg_color)
                else:
                    warped_img = slide_obj.warp_slide(0, non_rigid=updating_non_rigid, crop=mask_bbox_xywh)
                scaled_img_list[slide_obj.stack_idx] = warped_img

            # Get mask
            if mask is not None:
                slide_mask = (vips_micro_reg_mask==0).ifthenelse(0, slide_mask)

            # Update lists
            img_f_list[slide_obj.stack_idx] = slide_obj.src_f
            img_names_list[slide_obj.stack_idx] = slide_obj.name
            scaled_mask_list[slide_obj.stack_idx] = processing_mask
            slide_mask_list[slide_obj.stack_idx] = slide_mask


        # Normalize images. Since they are ROI, probably have different image stats
        # Warp after normalization, since padding after warping can create a lot of empty space that throws off normalization
        if not use_tiler and self.norm_method is not None and not rgb:

            all_histogram, all_img_stats = preprocessing.collect_img_stats(scaled_img_list)
            for i, img in enumerate(scaled_img_list):
                if self.norm_method == "histo_match":
                    normed_img = preprocessing.match_histograms(img, all_histogram)
                elif self.norm_method == "img_stats":
                    normed_img = preprocessing.norm_img_stats(img, all_img_stats)
                else:
                    print(f"Don't recognize `norm_metthod`={self.norm_method}")
                    normed_img = img

                normed_img = exposure.rescale_intensity(normed_img, out_range=(0, 255)).astype(np.uint8)

                slide_obj = self.get_slide(img_f_list[i])
                processed_warped_img = warp_tools.warp_img(img=normed_img, M=slide_obj.M,
                    bk_dxdy=dxdy,
                    transformation_src_shape_rc=slide_obj.processed_img_shape_rc,
                    transformation_dst_shape_rc=slide_obj.reg_img_shape_rc,
                    out_shape_rc=full_out_shape,
                    bbox_xywh=mask_bbox_xywh)

                scaled_img_list[i] = processed_warped_img


        img_dict = {serial_non_rigid.IMG_LIST_KEY: scaled_img_list,
                    serial_non_rigid.IMG_F_LIST_KEY: img_f_list,
                    serial_non_rigid.MASK_LIST_KEY: scaled_mask_list,
                    serial_non_rigid.IMG_NAME_KEY: img_names_list
                    }

        if ref_slide.non_rigid_reg_mask is not None:
            vips_nr_mask = warp_tools.numpy2vips(ref_slide.non_rigid_reg_mask)
            scaled_non_rigid_mask = warp_tools.resize_img(vips_nr_mask, full_out_shape, interp_method="nearest")
            if mask is not None:
                scaled_non_rigid_mask = scaled_non_rigid_mask.extract_area(*mask_bbox_xywh)
                scaled_non_rigid_mask = (vips_micro_reg_mask == 0).ifthenelse(0, scaled_non_rigid_mask)
            if not use_tiler:
                scaled_non_rigid_mask = warp_tools.vips2numpy(scaled_non_rigid_mask)
        else:
            scaled_non_rigid_mask = None

        if mask is not None:
            final_max_img_dim = np.max(mask_bbox_xywh[2:])
        else:
            final_max_img_dim = max_img_dim

        return img_dict, final_max_img_dim, scaled_non_rigid_mask, full_out_shape, mask_bbox_xywh, use_tiler

    def clean_dxdy(self):

        ref_slide = self.get_ref_slide()
        for slide_obj in self.slide_dict.values():
            if slide_obj == ref_slide:
                continue
            # Find where there are non-rigid displacement creates tears
            img_mask = np.full(slide_obj.processed_img_shape_rc, 255, dtype=np.uint8)
            r_warped_mask = warp_tools.warp_img(img_mask,
                                                M=slide_obj.M,
                                                out_shape_rc=slide_obj.reg_img_shape_rc,
                                                transformation_src_shape_rc=slide_obj.processed_img_shape_rc,
                                                transformation_dst_shape_rc=slide_obj.reg_img_shape_rc)

            nr_warped_mask = slide_obj.warp_img(img_mask, crop=False)

            tears = r_warped_mask - nr_warped_mask
            tears[tears != 255] = 0
            large_tears = warp_tools.resize_img(tears, slide_obj.bk_dxdy[0].shape, interp_method="nearest")
            inv_tears = warp_tools.warp_img(large_tears, bk_dxdy=slide_obj.fwd_dxdy, interp_method="nearest")

            # Find regions that are in tissue mask but outside of non-rigid mask
            rigid_reg_mask = slide_obj.warp_img(slide_obj.rigid_reg_mask, non_rigid=False, crop=False)

            rigid_bbox = warp_tools.xy2bbox(warp_tools.mask2xy(rigid_reg_mask))
            c0, r0 = rigid_bbox[:2]
            c1, r1 = rigid_bbox[:2] + rigid_bbox[2:]
            rigid_bbox_mask = np.zeros_like(rigid_reg_mask)
            rigid_bbox_mask[r0:r1, c0:c1] = 255
            temp_missing_mask = cv2.bitwise_xor(slide_obj.non_rigid_reg_mask, rigid_reg_mask)
            r_nr_intersection = cv2.bitwise_and(slide_obj.non_rigid_reg_mask, rigid_reg_mask)
            combined_mask = cv2.bitwise_or(r_nr_intersection, temp_missing_mask)
            small_missing_mask = cv2.bitwise_xor(slide_obj.non_rigid_reg_mask, combined_mask)

            inpaint_mask = cv2.bitwise_or(inv_tears, missing_mask)
            inpaint_mask[inpaint_mask != 0] = 255

            if inpaint_mask.max() == 0:
                print(f"no defects in {slide_obj.name}")
                continue

            cv_inpaint_method = cv2.INPAINT_NS
            inpainted_bk_dx = cv2.inpaint(slide_obj.bk_dxdy[0].astype(np.float32), inpaint_mask, 3, cv_inpaint_method)
            inpainted_bk_dy = cv2.inpaint(slide_obj.bk_dxdy[1].astype(np.float32), inpaint_mask, 3, cv_inpaint_method)
            inpainted_bk_dxdy = np.array([inpainted_bk_dx, inpainted_bk_dy])

            warped_rigid_reg_mask = slide_obj.warp_img(slide_obj.rigid_reg_mask, non_rigid=True, crop=False)
            warped_rigid_reg_mask = warp_tools.resize_img(warped_rigid_reg_mask, slide_obj.bk_dxdy[0].shape, interp_method="nearest")
            large_rigid_mask = warp_tools.resize_img(rigid_reg_mask, slide_obj.bk_dxdy[0].shape, interp_method="nearest")

            reg_mask = cv2.bitwise_or(warped_rigid_reg_mask, large_rigid_mask)
            reg_mask = warp_tools.resize_img(reg_mask, slide_obj.bk_dxdy[0].shape, interp_method="nearest")

            inpainted_bk_dxdy[0][reg_mask == 0] = 0
            inpainted_bk_dxdy[1][reg_mask == 0] = 0

            inpainted_fwd_dxdy = warp_tools.get_inverse_field(inpainted_bk_dxdy)
            inpainted_bk_dxdy = np.array([warp_tools.crop_img(inpainted_bk_dxdy[0], self._non_rigid_bbox), warp_tools.crop_img(inpainted_bk_dxdy[1], self._non_rigid_bbox)])
            inpainted_fwd_dxdy = np.array([warp_tools.crop_img(inpainted_fwd_dxdy[0], self._non_rigid_bbox), warp_tools.crop_img(inpainted_fwd_dxdy[1], self._non_rigid_bbox)])

            slide_obj.bk_dxdy = inpainted_bk_dxdy
            slide_obj.fwd_dxdy = np.array(inpainted_fwd_dxdy)


    def non_rigid_register(self, rigid_registrar, processor_dict):

        """Non-rigidly register slides

        Non-rigidly register slides after performing rigid registration.
        Also saves thumbnails of non-rigidly registered images and deformation
        fields.

        Parameters
        ----------
        rigid_registrar : SerialRigidRegistrar
            SerialRigidRegistrar object that performed the rigid registration.

        processor_dict : dict
            Each key should be the filename of the image, and the value either a subclassed
            preprocessing.ImageProcessor, or a list, where the 1st element is the processor,
            and the second element a dictionary of keyword arguments passed to the processor.
            If `None`, then a default processor will be used for each image based on
            the inferred modality.
        Returns
        -------
        non_rigid_registrar : SerialNonRigidRegistrar
            SerialNonRigidRegistrar object that performed serial
            non-rigid registration.

        """


        ref_slide = self.get_ref_slide()

        self.create_non_rigid_reg_mask()
        non_rigid_reg_mask = ref_slide.non_rigid_reg_mask
        non_rigid_reg_mask_bbox = warp_tools.xy2bbox(warp_tools.mask2xy(non_rigid_reg_mask))
        cropped_mask_shape_rc = non_rigid_reg_mask_bbox[2:][::-1]
        nr_on_scaled_img = self.max_processed_image_dim_px != self.max_non_rigid_registration_dim_px or \
            (non_rigid_reg_mask is not None and np.any(cropped_mask_shape_rc != ref_slide.reg_img_shape_rc))

        using_tiler = False

        nr_obj = self.non_rigid_reg_kwargs[NON_RIGID_REG_CLASS_KEY]
        nr_in_rgb = nr_obj.rgb
        img_specific_args = {}

        if nr_on_scaled_img:
            # Use higher resolution and/or roi for non-rigid
            nr_reg_src, max_img_dim, non_rigid_reg_mask, full_out_shape_rc, mask_bbox_xywh, using_tiler = \
                self.prep_images_for_large_non_rigid_registration(max_img_dim=self.max_non_rigid_registration_dim_px,
                                                                  processor_dict=processor_dict,
                                                                  mask=non_rigid_reg_mask,
                                                                  rgb=nr_in_rgb)

            self._non_rigid_bbox = mask_bbox_xywh
            self.max_non_rigid_registration_dim_px = max_img_dim

            if using_tiler:
                non_rigid_registrar_cls, img_specific_args = self.get_nr_tiling_params(self.non_rigid_reg_kwargs[NON_RIGID_REG_CLASS_KEY],
                                                                                       processor_dict=processor_dict,
                                                                                       img_specific_args=None,
                                                                                       tile_wh=DEFAULT_NR_TILE_WH)

                # Update args to use tiled non-rigid registrar
                self.non_rigid_reg_kwargs[NON_RIGID_REG_CLASS_KEY] = non_rigid_registrar_cls

        else:
            nr_reg_src = rigid_registrar
            full_out_shape_rc = ref_slide.reg_img_shape_rc
            if not self.crop_for_rigid_reg:
                self._non_rigid_bbox = None

        self._full_displacement_shape_rc = full_out_shape_rc

        non_rigid_registrar = serial_non_rigid.register_images(src=nr_reg_src,
                                                               align_to_reference=self.align_to_reference,
                                                               img_params = img_specific_args,
                                                               **self.non_rigid_reg_kwargs)
        self.end_non_rigid_time = time()

        for d in  [self.non_rigid_dst_dir, self.deformation_field_dir]:
            pathlib.Path(d).mkdir(exist_ok=True, parents=True)
        self.non_rigid_registrar = non_rigid_registrar

        if self.crop_for_rigid_reg and not nr_on_scaled_img:

            # If using cropped rigid registrar, will need to calculate self._non_rigid_bbox
            # Also need to reshape and scale dxdy to be proportional to registered shape
            bk_for_crop_for_rigid_reg = {}
            fwd_for_crop_for_rigid_reg = {}

            ref_src = (ref_slide.uncropped_processed_img_shape_rc/ref_slide.processed_img_shape_rc)
            rescaled_dxdy_shape_rc = np.ceil(np.array(ref_slide.reg_img_shape_rc)*ref_src).astype(int)

            bbox_x = np.inf
            bbox_y = np.inf
            bbox_w = 0
            bbox_h = 0

            for nr_obj in non_rigid_registrar.non_rigid_obj_list:
                slide_obj = self.get_slide(nr_obj.name)
                rigid_obj = rigid_registrar.img_obj_dict[nr_obj.name]
                crop_T = np.eye(3)
                crop_T[0:2, 2] = slide_obj.processed_crop_bbox[0:2]
                inv_M = np.linalg.inv(crop_T @ rigid_obj.M)

                uncropped_corners_xy = warp_tools.get_corners_of_image(slide_obj.uncropped_processed_img_shape_rc)[:, ::-1]
                warped_uncropped_corners = warp_tools.warp_xy(uncropped_corners_xy,
                                                            M=slide_obj.M,
                                                            transformation_src_shape_rc=slide_obj.processed_img_shape_rc,
                                                            transformation_dst_shape_rc=slide_obj.reg_img_shape_rc,
                                                            src_shape_rc=slide_obj.uncropped_processed_img_shape_rc,
                                                            dst_shape_rc = rescaled_dxdy_shape_rc
                                                            )
                dispalcement_transformer = transform.ProjectiveTransform()
                dispalcement_transformer.estimate(warped_uncropped_corners, uncropped_corners_xy)
                displacement_M = dispalcement_transformer.params

                bk_dxdy_in_original = warp_tools.warp_img(np.dstack(nr_obj.bk_dxdy),
                                        M=inv_M,
                                        out_shape_rc=slide_obj.uncropped_processed_img_shape_rc)
                warped_bk_dxdy = warp_tools.warp_img(bk_dxdy_in_original, M=displacement_M, out_shape_rc=rescaled_dxdy_shape_rc)
                bk_for_crop_for_rigid_reg[slide_obj.name] = np.array([warped_bk_dxdy[..., 0], warped_bk_dxdy[..., 1]])

                fwd_dxdy_in_original = warp_tools.warp_img(np.dstack(nr_obj.fwd_dxdy),
                                        M=inv_M,
                                        out_shape_rc=slide_obj.uncropped_processed_img_shape_rc)
                warped_fwd_dxdy = warp_tools.warp_img(fwd_dxdy_in_original, M=displacement_M, out_shape_rc=rescaled_dxdy_shape_rc)
                fwd_for_crop_for_rigid_reg[slide_obj.name] = np.array([warped_fwd_dxdy[..., 0], warped_fwd_dxdy[..., 1]])

                displacement_corners = warp_tools.get_corners_of_image(nr_obj.bk_dxdy[0].shape)[:, ::-1]
                displacement_bbox_in_rigid = warp_tools.xy2bbox(warp_tools.warp_xy(displacement_corners, M=inv_M @ displacement_M))

                bbox_x = min(displacement_bbox_in_rigid[0], bbox_x)
                bbox_y = min(displacement_bbox_in_rigid[1], bbox_y)
                bbox_w = max(displacement_bbox_in_rigid[2], bbox_w)
                bbox_h = max(displacement_bbox_in_rigid[3], bbox_h)

            full_out_shape_rc = rescaled_dxdy_shape_rc
            self._full_displacement_shape_rc = full_out_shape_rc
            self._non_rigid_bbox = np.array([bbox_x, bbox_y, bbox_w, bbox_h])

        # Draw overlap image #
        overlap_mask, overlap_mask_bbox_xywh = self.get_crop_mask(self.crop)
        overlap_mask_bbox_xywh = overlap_mask_bbox_xywh.astype(int)

        thumbnail_s = np.min(self.thumbnail_size/warp_tools.get_shape(non_rigid_registrar.non_rigid_obj_list[0].registered_img)[0:2])
        non_rigid_img_list = [warp_tools.rescale_img(nr_img_obj.registered_img, thumbnail_s) for nr_img_obj in non_rigid_registrar.non_rigid_obj_list]
        if isinstance(non_rigid_img_list[0], pyvips.Image):
            non_rigid_img_list = [warp_tools.vips2numpy(x) for x in non_rigid_img_list]

        if non_rigid_img_list[0].ndim == 3:
            # Non-rigid performed on RGB images
            non_rigid_img_list = [ (255*(1 - skcolor.rgb2gray(img))).astype(np.uint8) for img in non_rigid_img_list]

        self.non_rigid_overlap_img  = self.draw_overlap_img(img_list=non_rigid_img_list)

        overlap_img_fout = os.path.join(self.overlap_dir, self.name + "_non_rigid_overlap.png")
        warp_tools.save_img(overlap_img_fout, self.non_rigid_overlap_img)

        n_digits = len(str(self.size))
        for slide_name, slide_obj in self.slide_dict.items():
            img_save_id = str.zfill(str(slide_obj.stack_idx), n_digits)
            slide_nr_reg_obj = non_rigid_registrar.non_rigid_obj_dict[slide_name]

            if not using_tiler:
                if self.crop_for_rigid_reg and not nr_on_scaled_img:
                    slide_obj.bk_dxdy = bk_for_crop_for_rigid_reg[slide_obj.name]
                    slide_obj.fwd_dxdy = fwd_for_crop_for_rigid_reg[slide_obj.name]
                else:
                    slide_obj.bk_dxdy = np.array(slide_nr_reg_obj.bk_dxdy)
                    slide_obj.fwd_dxdy = np.array(slide_nr_reg_obj.fwd_dxdy)
            else:
                # save displacements as images
                pathlib.Path(self.displacements_dir).mkdir(exist_ok=True, parents=True)
                slide_obj.stored_dxdy = True
                bk_dxdy_f, fwd_dxdy_f = slide_obj.get_displacement_f()
                slide_obj._bk_dxdy_f = bk_dxdy_f
                slide_obj._fwd_dxdy_f = fwd_dxdy_f
                # Save space by only writing the necessary areas. Most displacements may be 0
                if np.all(warp_tools.get_shape(slide_nr_reg_obj.bk_dxdy)[0:2][::-1] > mask_bbox_xywh[2:]):
                    cropped_bk_dxdy = slide_nr_reg_obj.bk_dxdy.extract_area(*mask_bbox_xywh)
                    cropped_fwd_dxdy = slide_nr_reg_obj.fwd_dxdy.extract_area(*mask_bbox_xywh)
                else:
                    cropped_bk_dxdy = slide_nr_reg_obj.bk_dxdy
                    cropped_fwd_dxdy = slide_nr_reg_obj.fwd_dxdy

                cropped_bk_dxdy.cast("float").tiffsave(slide_obj._bk_dxdy_f, compression="lzw", lossless=True, tile=True, bigtiff=True)
                cropped_fwd_dxdy.cast("float").tiffsave(slide_obj._fwd_dxdy_f, compression="lzw", lossless=True, tile=True, bigtiff=True)

            slide_obj.nr_rigid_reg_img_f = os.path.join(self.non_rigid_dst_dir, img_save_id + "_" + slide_obj.name + ".png")

        for slide_name, slide_obj in self.slide_dict.items():
            img_save_id = str.zfill(str(slide_obj.stack_idx), n_digits)
            slide_nr_reg_obj = non_rigid_registrar.non_rigid_obj_dict[slide_name]
            if not slide_obj.is_rgb:
                img_to_warp = slide_obj.pad_cropped_processed_img()
            else:
                img_to_warp = slide_obj.image
            img_to_warp = warp_tools.resize_img(img_to_warp, slide_obj.processed_img_shape_rc)
            warped_img = slide_obj.warp_img(img_to_warp, non_rigid=True, crop=self.crop)
            warp_tools.save_img(slide_obj.nr_rigid_reg_img_f, warped_img, thumbnail_size=self.thumbnail_size)

            # Draw displacements on image actually used in non-rigid. Might be higher resolution
            if not isinstance(slide_nr_reg_obj.bk_dxdy, pyvips.Image):
                draw_dxdy = np.dstack(slide_nr_reg_obj.bk_dxdy)
            else:
                #pyvips
                draw_dxdy = slide_nr_reg_obj.bk_dxdy

            dxdy_shape = warp_tools.get_shape(draw_dxdy)
            thumbnail_scaling = np.min(self.thumbnail_size/np.array(dxdy_shape[0:2]))
            thumbnail_bk_dxdy = self.create_thumbnail(draw_dxdy, thumbnail_size=self.thumbnail_size)
            thumbnail_bk_dxdy *= float(thumbnail_scaling)

            if isinstance(thumbnail_bk_dxdy, pyvips.Image):
                thumbnail_bk_dxdy = warp_tools.vips2numpy(thumbnail_bk_dxdy)

            draw_img = warp_tools.resize_img(slide_nr_reg_obj.registered_img, thumbnail_bk_dxdy[..., 0].shape)
            if isinstance(draw_img, pyvips.Image):
                draw_img = warp_tools.vips2numpy(draw_img)

            draw_img = exposure.rescale_intensity(draw_img, out_range=(0, 255))

            if draw_img.ndim == 2:
                draw_img = np.dstack([draw_img] * 3)

            thumbanil_deform_grid = viz.draw_displacement_vector_field(dxdy=[-thumbnail_bk_dxdy[..., 0], -thumbnail_bk_dxdy[..., 1]],
                                                                                img=draw_img,
                                                                                spacing=10)

            deform_img_f = os.path.join(self.deformation_field_dir, img_save_id + "_" + slide_obj.name + ".png")
            warp_tools.save_img(deform_img_f, thumbanil_deform_grid, thumbnail_size=self.thumbnail_size)

        return non_rigid_registrar

    def measure_error(self):
        """Measure registration error

        Error is measured as the distance between matched features
        after registration.

        Returns
        -------
        summary_df : Dataframe
            `summary_df` contains various information about the registration.

            The "from" column is the name of the image, while the "to" column
            name of the image it was aligned to. "from" is analagous to "moving"
            or "current", while "to" is analgous to "fixed" or "previous".

            Columns begining with "original" refer to error measurements of the
            unregistered images. Those beginning with "rigid" or "non_rigid" refer
            to measurements related to rigid or non-rigid registration, respectively.

            Columns beginning with "mean" are averages of error measurements. In
            the case of errors based on feature distances (i.e. those ending in "D"),
            the mean is weighted by the number of feature matches between "from" and "to".

            Columns endining in "D" indicate the median distance between matched
            features in "from" and "to".

            Columns ending in "rTRE" indicate the target registration error between
            "from" and "to".

            Columns ending in "mattesMI" contain measurements of the Mattes mutual
            information between "from" and "to".

            "processed_img_shape" indicates the shape (row, column) of the processed
            image actually used to conduct the registration

            "shape" is the shape of the slide at full resolution

            "aligned_shape" is the shape of the registered full resolution slide

            "physical_units" are the names of the pixels physcial unit, e.g. u'\u00B5m'

            "resolution" is the physical unit per pixel

            "name" is the name assigned to the Valis instance

            "rigid_time_minutes" is the total number of minutes it took
            to convert the images and then rigidly align them.

            "non_rigid_time_minutes" is the total number of minutes it took
            to convert the images, and then perform rigid -> non-rigid registration.

        """

        path_list = [None] * (self.size)
        all_og_d = [None] * (self.size)
        all_og_tre = [None] * (self.size)

        all_rigid_d = [None] * (self.size)
        all_rigid_tre = [None] * (self.size)

        all_nr_d = [None] * (self.size)
        all_nr_tre = [None] * (self.size)

        all_n = [None] * (self.size)
        from_list = [None] * (self.size)
        to_list = [None] * (self.size)
        shape_list = [None] * (self.size)
        processed_img_shape_list = [None] * (self.size)
        unit_list = [None] * (self.size)
        resolution_list = [None] * (self.size)

        slide_obj_list = list(self.slide_dict.values())
        outshape = slide_obj_list[0].aligned_slide_shape_rc

        ref_slide = self.get_ref_slide()
        ref_diagonal = np.sqrt(np.sum(np.power(ref_slide.processed_img_shape_rc, 2)))

        measure_idx = []
        for slide_obj in tqdm.tqdm(self.slide_dict.values(), desc=MEASURE_MSG, unit="image"):
            i = slide_obj.stack_idx
            slide_name = slide_obj.name

            shape_list[i] = tuple(slide_obj.slide_shape_rc)
            processed_img_shape_list[i] = tuple(slide_obj.processed_img_shape_rc)
            unit_list[i] = ref_slide.units
            resolution_list[i] = ref_slide.resolution
            from_list[i] = slide_name
            path_list[i] = slide_obj.src_f

            if slide_obj.name == ref_slide.name or slide_obj.is_empty:
                continue

            measure_idx.append(i)
            prev_slide_obj = slide_obj.fixed_slide
            to_list[i] = prev_slide_obj.name

            img_T = warp_tools.get_padding_matrix(slide_obj.processed_img_shape_rc,
                                                  slide_obj.reg_img_shape_rc)

            prev_T = warp_tools.get_padding_matrix(prev_slide_obj.processed_img_shape_rc,
                                                   prev_slide_obj.reg_img_shape_rc)


            prev_kp_in_slide = prev_slide_obj.warp_xy(slide_obj.xy_in_prev,
                                                     M=prev_T,
                                                     pt_level= prev_slide_obj.processed_img_shape_rc,
                                                     non_rigid=False)

            current_kp_in_slide = slide_obj.warp_xy(slide_obj.xy_matched_to_prev,
                                                    M=img_T,
                                                    pt_level= slide_obj.processed_img_shape_rc,
                                                    non_rigid=False)

            og_d = warp_tools.calc_d(prev_kp_in_slide, current_kp_in_slide)

            og_rtre = og_d/ref_diagonal
            median_og_tre = np.median(og_rtre)
            og_d *= slide_obj.resolution
            median_d_og = np.median(og_d)

            all_og_d[i] = median_d_og
            all_og_tre[i] = median_og_tre


            prev_warped_rigid = prev_slide_obj.warp_xy(slide_obj.xy_in_prev,
                                                       M=prev_slide_obj.M,
                                                       pt_level= prev_slide_obj.processed_img_shape_rc,
                                                       non_rigid=False)

            current_warped_rigid = slide_obj.warp_xy(slide_obj.xy_matched_to_prev,
                                                     M=slide_obj.M,
                                                     pt_level= slide_obj.processed_img_shape_rc,
                                                     non_rigid=False)


            rigid_d = warp_tools.calc_d(prev_warped_rigid, current_warped_rigid)
            rtre = rigid_d/ref_diagonal
            median_rigid_tre = np.median(rtre)
            rigid_d *= slide_obj.resolution
            median_d_rigid = np.median(rigid_d)

            all_rigid_d[i] = median_d_rigid
            all_n[i] = len(rigid_d)
            all_rigid_tre[i] = median_rigid_tre

            if slide_obj.bk_dxdy is not None:
                prev_warped_nr = prev_slide_obj.warp_xy(slide_obj.xy_in_prev,
                                                        M=prev_slide_obj.M,
                                                        pt_level= prev_slide_obj.processed_img_shape_rc,
                                                        non_rigid=True)

                current_warped_nr = slide_obj.warp_xy(slide_obj.xy_matched_to_prev,
                                                      M=slide_obj.M,
                                                      pt_level= slide_obj.processed_img_shape_rc,
                                                      non_rigid=True)

                nr_d =  warp_tools.calc_d(prev_warped_nr, current_warped_nr)
                nrtre = nr_d/ref_diagonal
                mean_nr_tre = np.median(nrtre)

                nr_d *= slide_obj.resolution
                median_d_nr = np.median(nr_d)
                all_nr_d[i] = median_d_nr
                all_nr_tre[i] = mean_nr_tre

        weights = np.array(all_n)[measure_idx]
        mean_og_d = np.average(np.array(all_og_d)[measure_idx], weights=weights)
        median_og_tre = np.average(np.array(all_og_tre)[measure_idx], weights=weights)

        mean_rigid_d = np.average(np.array(all_rigid_d)[measure_idx], weights=weights)
        median_rigid_tre = np.average(np.array(all_rigid_tre)[measure_idx], weights=weights)

        rigid_min = (self.end_rigid_time - self.start_time)/60

        self.summary_df = pd.DataFrame({
            "filename": path_list,
            "from":from_list,
            "to": to_list,
            "original_D": all_og_d,
            "original_rTRE": all_og_tre,
            "rigid_D": all_rigid_d,
            "rigid_rTRE": all_rigid_tre,
            "non_rigid_D": all_nr_d,
            "non_rigid_rTRE": all_nr_tre,
            "processed_img_shape": processed_img_shape_list,
            "shape": shape_list,
            "aligned_shape": [tuple(outshape)]*self.size,
            "mean_original_D": [mean_og_d]*self.size,
            "mean_rigid_D": [mean_rigid_d]*self.size,
            "physical_units":unit_list,
            "resolution":resolution_list,
            "name": [self.name]*self.size,
            "rigid_time_minutes" : [rigid_min]*self.size
        })

        if any([d for d in all_nr_d if d is not None]):
            mean_nr_d = np.average(np.array(all_nr_d)[measure_idx], weights=weights)
            mean_nr_tre = np.average(np.array(all_nr_tre)[measure_idx], weights=weights)
            non_rigid_min = (self.end_non_rigid_time - self.start_time)/60

            self.summary_df["mean_non_rigid_D"] = [mean_nr_d]*self.size
            self.summary_df["non_rigid_time_minutes"] = [non_rigid_min]*self.size

        return self.summary_df

    def register(self, brightfield_processing_cls=DEFAULT_BRIGHTFIELD_CLASS,
                 brightfield_processing_kwargs=DEFAULT_BRIGHTFIELD_PROCESSING_ARGS,
                 if_processing_cls=DEFAULT_FLOURESCENCE_CLASS,
                 if_processing_kwargs=DEFAULT_FLOURESCENCE_PROCESSING_ARGS,
                 processor_dict=None,
                 reader_cls=None,
                 reader_dict=None):

        """Register a collection of images

        This function will convert the slides to images, pre-process and normalize them, and
        then conduct rigid registration. Non-rigid registration will then be performed if the
        `non_rigid_registrar_cls` argument used to initialize the Valis object was not None.

        In addition to the objects returned, the desination directory (i.e. `dst_dir`)
        will contain thumbnails so that one can visualize the results: converted image
        thumbnails will be in "images/"; processed images in "processed/";
        rigidly aligned images in "rigid_registration/"; non-rigidly aligned images in "non_rigid_registration/";
        non-rigid deformation field images (i.e. warped grids colored by the direction and magntidue)
        of the deformation) will be in ""deformation_fields/". The size of these thumbnails
        is determined by the `thumbnail_size` argument used to initialze this object.

        One can get a sense of how well the registration worked by looking
        in the "overlaps/", which shows how the images overlap before
        registration, after rigid registration, and after non-rigid registration. Each image
        is created by coloring an inverted greyscale version of the processed images, and then
        blending those images.

        The "data/" directory will contain a pickled copy of this registrar, which can be
        later be opened (unpickled) and used to warp slides and/or point data.

        "data/" will also contain the `summary_df` saved as a csv file.


        Parameters
        ----------
        brightfield_processing_cls : preprocessing.ImageProcesser
            preprocessing.ImageProcesser used to pre-process brightfield images to make
            them look as similar as possible.

        brightfield_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `brightfield_processing_cls`

        if_processing_cls : preprocessing.ImageProcesser
            preprocessing.ImageProcesser used to pre-process immunofluorescent images
            to make them look as similar as possible.

        if_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `if_processing_cls`

        processor_dict : dict, optional
            Each key should be the filename of the image, and the value either a subclassed
            preprocessing.ImageProcessor, or a list, where the 1st element is the processor,
            and the second element a dictionary of keyword arguments passed to the processor.
            If `None`, then a default processor will be used for each image based on
            the inferred modality.

        reader_cls : SlideReader, optional
            Uninstantiated SlideReader class that will convert
            the slide to an image, and also collect metadata. If None (the default),
            the appropriate SlideReader will be found by `slide_io.get_slide_reader`.
            This option is provided in case the slides cannot be opened by a current
            SlideReader class. In this case, the user should create a subclass of
            SlideReader. See slide_io.SlideReader for details.

        reader_dict: dict, optional
            Dictionary specifying which readers to use for individual images. The
            keys should be the image's filename, and the values the instantiated slide_io.SlideReader
            to use to read that file. Valis will try to find an appropritate reader
            for any omitted files, or will use `reader_cls` as the default.

        Returns
        -------
        rigid_registrar : SerialRigidRegistrar
            SerialRigidRegistrar object that performed the rigid registration.
            This object can be pickled if so desired

        non_rigid_registrar : SerialNonRigidRegistrar
            SerialNonRigidRegistrar object that performed serial
            non-rigid registration. This object can be pickled if so desired.

        summary_df : Dataframe
            `summary_df` contains various information about the registration.

            The "from" column is the name of the image, while the "to" column
            name of the image it was aligned to. "from" is analagous to "moving"
            or "current", while "to" is analgous to "fixed" or "previous".

            Columns begining with "original" refer to error measurements of the
            unregistered images. Those beginning with "rigid" or "non_rigid" refer
            to measurements related to rigid or non-rigid registration, respectively.

            Columns beginning with "mean" are averages of error measurements. In
            the case of errors based on feature distances (i.e. those ending in "D"),
            the mean is weighted by the number of feature matches between "from" and "to".

            Columns endining in "D" indicate the median distance between matched
            features in "from" and "to".

            Columns ending in "TRE" indicate the target registration error between
            "from" and "to".

            Columns ending in "mattesMI" contain measurements of the Mattes mutual
            information between "from" and "to".

            "processed_img_shape" indicates the shape (row, column) of the processed
            image actually used to conduct the registration

            "shape" is the shape of the slide at full resolution

            "aligned_shape" is the shape of the registered full resolution slide

            "physical_units" are the names of the pixels physcial unit, e.g. u'\u00B5m'

            "resolution" is the physical unit per pixel

            "name" is the name assigned to the Valis instance

            "rigid_time_minutes" is the total number of minutes it took
            to convert the images and then rigidly align them.

            "non_rigid_time_minutes" is the total number of minutes it took
            to convert the images, and then perform rigid -> non-rigid registration.

        """

        self.start_time = time()
        try:
            print("\n==== Converting images\n")
            self.convert_imgs(series=self.series, reader_cls=reader_cls, reader_dict=reader_dict)

            print("\n==== Processing images\n")
            slide_processors = self.create_img_processor_dict(brightfield_processing_cls=brightfield_processing_cls,
                                            brightfield_processing_kwargs=brightfield_processing_kwargs,
                                            if_processing_cls=if_processing_cls,
                                            if_processing_kwargs=if_processing_kwargs,
                                            processor_dict=processor_dict)

            self.brightfield_procsseing_fxn_str = brightfield_processing_cls.__name__
            self.if_processing_fxn_str = if_processing_cls.__name__
            self.process_imgs(processor_dict=slide_processors)

            # print("\n==== Rigid registration\n")
            rigid_registrar = self.rigid_register()
            aligned_slide_shape_rc = self.get_aligned_slide_shape(0)
            self.aligned_slide_shape_rc = aligned_slide_shape_rc
            self.iter_order = rigid_registrar.iter_order
            for slide_obj in self.slide_dict.values():
                slide_obj.aligned_slide_shape_rc = aligned_slide_shape_rc

            if self.micro_rigid_registrar_cls is not None:
                print("\n==== Micro-rigid registration\n")
                self.micro_rigid_register()

            if rigid_registrar is False:
                return None, None, None

            if self.non_rigid_registrar_cls is not None:
                print("\n==== Non-rigid registration\n")
                non_rigid_registrar = self.non_rigid_register(rigid_registrar, slide_processors)

            else:
                non_rigid_registrar = None


            self._add_empty_slides()

            print("\n==== Measuring error\n")
            error_df = self.measure_error()
            self.error_df = error_df
            self.cleanup()

            pathlib.Path(self.data_dir).mkdir(exist_ok=True,  parents=True)
            f_out = os.path.join(self.data_dir, self.name + "_registrar.pickle")
            self.reg_f = f_out
            pickle.dump(self, open(f_out, 'wb'))

            data_f_out = os.path.join(self.data_dir, self.name + "_summary.csv")
            error_df.to_csv(data_f_out, index=False)

        except Exception as e:
            traceback_msg = traceback.format_exc()
            valtils.print_warning(e, rgb=Fore.RED, traceback_msg=traceback_msg)
            if slide_io.ome is not None:
                # Only kill JVM if it has been initialized
                kill_jvm()
            return None, None, None


        return rigid_registrar, non_rigid_registrar, error_df

    def cleanup(self):
        """Remove objects that can't be pickled
        """
        self.rigid_reg_kwargs[FD_KEY] = None
        self.rigid_reg_kwargs[AFFINE_OPTIMIZER_KEY] = None
        self.rigid_reg_kwargs[MATCHER_KEY] = None
        self.rigid_reg_kwargs[MATCHER_FOR_SORTING_KEY] = None
        if hasattr(self, "non_rigid_reg_kwargs"):
            self.non_rigid_reg_kwargs[NON_RIGID_REG_CLASS_KEY] = None
        self.non_rigid_registrar_cls = None
        self.rigid_registrar = None
        self.micro_rigid_registrar_cls = None
        self.non_rigid_registrar = None

    @valtils.deprecated_args(max_non_rigid_registartion_dim_px="max_non_rigid_registration_dim_px")
    def register_micro(self, brightfield_processing_cls=DEFAULT_BRIGHTFIELD_CLASS,
                 brightfield_processing_kwargs=DEFAULT_BRIGHTFIELD_PROCESSING_ARGS,
                 if_processing_cls=DEFAULT_FLOURESCENCE_CLASS,
                 if_processing_kwargs=DEFAULT_FLOURESCENCE_PROCESSING_ARGS,
                 processor_dict=None,
                 max_non_rigid_registration_dim_px=DEFAULT_MAX_MICRO_REG_SIZE,
                 non_rigid_registrar_cls=DEFAULT_NON_RIGID_CLASS,
                 non_rigid_reg_params=DEFAULT_NON_RIGID_KWARGS,
                 reference_img_f=None, align_to_reference=False, mask=None, tile_wh=DEFAULT_NR_TILE_WH):
        """Improve alingment of microfeatures by performing second non-rigid registration on larger images

        Caclculates additional non-rigid deformations using a larger image

        Parameters
        ----------
        brightfield_processing_cls : preprocessing.ImageProcesser
            preprocessing.ImageProcesser used to pre-process brightfield images to make
            them look as similar as possible.

        brightfield_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `brightfield_processing_cls`

        if_processing_cls : preprocessing.ImageProcesser
            preprocessing.ImageProcesser used to pre-process immunofluorescent images
            to make them look as similar as possible.

        if_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `if_processing_cls`

        max_non_rigid_registration_dim_px : int, optional
             Maximum width or height of images used for non-rigid registration.
             If None, then the full sized image will be used. However, this
             may take quite some time to complete.

        reference_img_f : str, optional
            Filename of image that will be treated as the center of the stack.
            If None, the index of the middle image will be the reference, and
            images will be aligned towards it. If provided, images will be
            aligned to this reference.

        align_to_reference : bool, optional
            If `False`, images will be non-rigidly aligned serially towards the
            reference image. If `True`, images will be non-rigidly aligned
            directly to the reference image. If `reference_img_f` is None,
            then the reference image will be the one in the middle of the stack.

        non_rigid_registrar_cls : NonRigidRegistrar, optional
            Uninstantiated NonRigidRegistrar class that will be used to
            calculate the deformation fields between images. See
            the `non_rigid_registrars` module for a desciption of available
            methods. If a desired non-rigid registration method is not available,
            one can be implemented by subclassing.NonRigidRegistrar.

        non_rigid_reg_params: dictionary, optional
            Dictionary containing key, value pairs to be used to initialize
            `non_rigid_registrar_cls`.
            In the case where simple ITK is used by the, params should be
            a SimpleITK.ParameterMap. Note that numeric values nedd to be
            converted to strings. See the NonRigidRegistrar classes in
            `non_rigid_registrars` for the available non-rigid registration
            methods and arguments.

        """

        if self.max_non_rigid_registration_dim_px >= max_non_rigid_registration_dim_px:
            if self.max_non_rigid_registration_dim_px > max_non_rigid_registration_dim_px:
                comp = "larger than the"
            else:
                comp = "equal to the"
            msg = (f"Have already non-rigidly aligned images with size = {self.max_non_rigid_registration_dim_px} "
                   f"which is {comp} specified size of {max_non_rigid_registration_dim_px}. "
                   f"Please set `max_non_rigid_registration_dim_px` to a larger value.")
            valtils.print_warning(msg)
            return None, self.error_df

        # Remove empty slides
        for empty_slide_name, empty_slide in self._empty_slides.items():
            del self.slide_dict[empty_slide_name]
            self.size -= 1

        ref_slide = self.get_ref_slide()
        if mask is None:
            if ref_slide.non_rigid_reg_mask is not None:
                mask = ref_slide.non_rigid_reg_mask.copy()

        slide_processors = self.create_img_processor_dict(brightfield_processing_cls=brightfield_processing_cls,
                                brightfield_processing_kwargs=brightfield_processing_kwargs,
                                if_processing_cls=if_processing_cls,
                                if_processing_kwargs=if_processing_kwargs,
                                processor_dict=processor_dict)


        if non_rigid_registrar_cls is not None:
            if isinstance(non_rigid_registrar_cls, type):
                if non_rigid_reg_params is None:
                    non_rigid_registrar_obj = non_rigid_registrar_cls()
                else:
                    nr = non_rigid_registrar_obj(**non_rigid_reg_params)
            else:
                non_rigid_registrar_obj = non_rigid_registrar_cls

        reg_in_rgb = non_rigid_registrar_obj.rgb
        nr_reg_src, max_img_dim, non_rigid_reg_mask, full_out_shape_rc, mask_bbox_xywh, using_tiler = \
            self.prep_images_for_large_non_rigid_registration(max_img_dim=max_non_rigid_registration_dim_px,
                                                              processor_dict=slide_processors,
                                                              updating_non_rigid=True,
                                                              mask=mask,
                                                              rgb=reg_in_rgb)

        img_specific_args = None
        write_dxdy = isinstance(ref_slide.bk_dxdy, pyvips.Image)

        if using_tiler:
            # Have determined that these images will be too big
            msg = (f"Registration would more than {TILER_THRESH_GB} GB if all images opened in memory. "
                    f"Will use NonRigidTileRegistrar to register cooresponding tiles to reduce memory consumption, "
                    f"but this method is experimental")

            valtils.print_warning(msg)

            write_dxdy = True
            non_rigid_registrar_cls, img_specific_args = self.get_nr_tiling_params(non_rigid_registrar_obj,
                                                                                   processor_dict=slide_processors,
                                                                                   img_specific_args=img_specific_args,
                                                                                   tile_wh=tile_wh)
            non_rigid_registrar_obj = non_rigid_registrar_cls()

        print("\n==== Performing microregistration\n")
        non_rigid_registrar = serial_non_rigid.register_images(src=nr_reg_src,
                                                               non_rigid_reg_class=non_rigid_registrar_cls,
                                                               non_rigid_reg_params=non_rigid_reg_params,
                                                               reference_img_f=reference_img_f,
                                                               mask=non_rigid_reg_mask,
                                                               align_to_reference=align_to_reference,
                                                               name=self.name,
                                                               img_params=img_specific_args
                                                               )

        pathlib.Path(self.micro_reg_dir).mkdir(exist_ok=True, parents=True)
        out_shape = full_out_shape_rc
        n_digits = len(str(self.size))
        micro_reg_imgs = [None] * self.size

        # Update displacements
        for slide_obj in self.slide_dict.values():

            if slide_obj == ref_slide:
                continue

            nr_obj = non_rigid_registrar.non_rigid_obj_dict[slide_obj.name]
            # Will be combining original and new dxdy as pyvips Images
            if not isinstance(slide_obj.bk_dxdy[0], pyvips.Image):
                vips_current_bk_dxdy = warp_tools.numpy2vips(np.dstack(slide_obj.bk_dxdy)).cast("float")
                vips_current_fwd_dxdy = warp_tools.numpy2vips(np.dstack(slide_obj.fwd_dxdy)).cast("float")
            else:
                vips_current_bk_dxdy = slide_obj.bk_dxdy
                vips_current_fwd_dxdy = slide_obj.fwd_dxdy

            if not isinstance(nr_obj.bk_dxdy, pyvips.Image):
                vips_new_bk_dxdy = warp_tools.numpy2vips(np.dstack(nr_obj.bk_dxdy)).cast("float")
                vips_new_fwd_dxdy = warp_tools.numpy2vips(np.dstack(nr_obj.fwd_dxdy)).cast("float")
            else:
                vips_new_bk_dxdy = nr_obj.bk_dxdy
                vips_new_fwd_dxdy = nr_obj.fwd_dxdy

            if np.any(non_rigid_registrar.shape != full_out_shape_rc):
                # Micro-registration performed on sub-region. Need to put in full image
                vips_new_bk_dxdy = self.pad_displacement(vips_new_bk_dxdy, full_out_shape_rc, mask_bbox_xywh)
                vips_new_fwd_dxdy = self.pad_displacement(vips_new_fwd_dxdy, full_out_shape_rc, mask_bbox_xywh)

            # Scale original dxdy to match scaled shape of new dxdy
            slide_sxy = (np.array(out_shape)/np.array([vips_current_bk_dxdy.height, vips_current_bk_dxdy.width]))[::-1]
            if not np.all(slide_sxy == 1):
                scaled_bk_dx = float(slide_sxy[0])*vips_current_bk_dxdy[0]
                scaled_bk_dy = float(slide_sxy[1])*vips_current_bk_dxdy[1]
                vips_current_bk_dxdy = scaled_bk_dx.bandjoin(scaled_bk_dy)
                vips_current_bk_dxdy = warp_tools.resize_img(vips_current_bk_dxdy, out_shape)

                scaled_fwd_dx = float(slide_sxy[0])*vips_current_fwd_dxdy[0]
                scaled_fwd_dy = float(slide_sxy[1])*vips_current_fwd_dxdy[1]
                vips_current_fwd_dxdy = scaled_fwd_dx.bandjoin(scaled_fwd_dy)
                vips_current_fwd_dxdy = warp_tools.resize_img(vips_current_fwd_dxdy, out_shape)

            vips_updated_bk_dxdy = vips_current_bk_dxdy + vips_new_bk_dxdy
            vips_updated_fwd_dxdy = vips_current_fwd_dxdy + vips_new_fwd_dxdy

            if not write_dxdy:
                # Will save numpy dxdy as Slide attributes
                np_updated_bk_dxdy = warp_tools.vips2numpy(vips_updated_bk_dxdy)
                np_updated_fwd_dxdy = warp_tools.vips2numpy(vips_updated_fwd_dxdy)

                slide_obj.bk_dxdy = np.array([np_updated_bk_dxdy[..., 0], np_updated_bk_dxdy[..., 1]])
                slide_obj.fwd_dxdy = np.array([np_updated_fwd_dxdy[..., 0], np_updated_fwd_dxdy[..., 1]])
            else:
                pathlib.Path(self.displacements_dir).mkdir(exist_ok=True, parents=True)
                slide_obj.stored_dxdy = True

                bk_dxdy_f, fwd_dxdy_f = slide_obj.get_displacement_f()
                slide_obj._bk_dxdy_f = bk_dxdy_f
                slide_obj._fwd_dxdy_f = fwd_dxdy_f

                # Save space by only writing the necessary areas. Most displacements may be 0
                cropped_bk_dxdy = vips_updated_bk_dxdy.extract_area(*mask_bbox_xywh)
                cropped_fwd_dxdy = vips_updated_fwd_dxdy.extract_area(*mask_bbox_xywh)

                if not os.path.exists(slide_obj._bk_dxdy_f):
                    cropped_bk_dxdy.cast("float").tiffsave(slide_obj._bk_dxdy_f, compression="lzw", lossless=True, tile=True, bigtiff=True)

                else:
                    # Don't seem to be able to overwrite directly because also accessing it?
                    disp_dir, temp_bk_f = os.path.split(slide_obj._bk_dxdy_f)
                    full_temp_dx_f = os.path.join(disp_dir, f".temp_{temp_bk_f}")
                    cropped_bk_dxdy.cast("float").tiffsave(full_temp_dx_f, compression="lzw", lossless=True, tile=True, bigtiff=True)
                    os.remove(slide_obj._bk_dxdy_f)
                    os.rename(full_temp_dx_f, slide_obj._bk_dxdy_f)

                if not os.path.exists(slide_obj._fwd_dxdy_f):
                    cropped_fwd_dxdy.cast("float").tiffsave(slide_obj._fwd_dxdy_f, compression="lzw", lossless=True, tile=True, bigtiff=True)
                else:
                    disp_dir, temp_fwd_f = os.path.split(slide_obj._fwd_dxdy_f)
                    full_temp_fwd_f = os.path.join(disp_dir, f".temp_{temp_fwd_f}")
                    cropped_fwd_dxdy.cast("float").tiffsave(full_temp_fwd_f, compression="lzw", lossless=True, tile=True, bigtiff=True)
                    os.remove(slide_obj._fwd_dxdy_f)
                    os.rename(full_temp_fwd_f, slide_obj._fwd_dxdy_f)

        # Update dxdy padding attributes here, in the event that previous displacements were also saved as files
        # Updating these attributes earlier will cause errors
        self._non_rigid_bbox = mask_bbox_xywh
        self._full_displacement_shape_rc = full_out_shape_rc
        for slide_obj in self.slide_dict.values():
            if not slide_obj.is_rgb:
                img_to_warp = slide_obj.pad_cropped_processed_img()
            else:
                img_to_warp = slide_obj.image

            img_to_warp = warp_tools.resize_img(img_to_warp, slide_obj.processed_img_shape_rc)
            micro_reg_img = slide_obj.warp_img(img_to_warp, non_rigid=True, crop=self.crop)

            img_save_id = str.zfill(str(slide_obj.stack_idx), n_digits)
            micro_fout = os.path.join(self.micro_reg_dir, f"{img_save_id}_{slide_obj.name}.png")
            micro_thumb = self.create_thumbnail(micro_reg_img, thumbnail_size=self.thumbnail_size)
            warp_tools.save_img(micro_fout, micro_thumb)

            processed_micro_reg_img = slide_obj.warp_img(slide_obj.pad_cropped_processed_img())
            thumbnail_s = np.min(self.thumbnail_size/np.array(processed_micro_reg_img.shape[0:2]))
            micro_reg_imgs[slide_obj.stack_idx] = warp_tools.rescale_img(processed_micro_reg_img, thumbnail_s)

        # Add empty slides back and save results
        for empty_slide_name, empty_slide in self._empty_slides.items():
            self.slide_dict[empty_slide_name] = empty_slide
            self.size += 1

        pickle.dump(self, open(self.reg_f, 'wb'))

        micro_overlap = self.draw_overlap_img(micro_reg_imgs)
        self.micro_reg_overlap_img = micro_overlap
        overlap_img_fout = os.path.join(self.overlap_dir, self.name + "_micro_reg.png")
        warp_tools.save_img(overlap_img_fout, micro_overlap)

        print("\n==== Measuring error\n")
        error_df = self.measure_error()
        self.error_df = error_df
        data_f_out = os.path.join(self.data_dir, self.name + "_summary.csv")
        error_df.to_csv(data_f_out, index=False)

        return non_rigid_registrar, error_df

    def get_aligned_slide_shape(self, level):
        """Get size of aligned images

        Parameters
        ----------
        level : int, float
            If `level` is an integer, then it is assumed that `level` is referring to
            the pyramid level that will be warped.

            If `level` is a float, it is assumed `level` is how much to rescale the
            registered image's size.

        """

        ref_slide = self.get_ref_slide()

        if np.issubdtype(type(level), np.integer):
            n_levels = len(ref_slide.slide_dimensions_wh)
            if level >= n_levels:
                msg = (f"requested to scale transformation for pyramid level {level}, ",
                    f"but the image only has {n_levels} (starting from 0). ",
                    f"Will use level {level-1}, which is the smallest level")
                valtils.print_warning(msg)
                level = level - 1

            slide_shape_rc = ref_slide.slide_dimensions_wh[level][::-1]
            s_rc = (slide_shape_rc/np.array(ref_slide.processed_img_shape_rc))
        else:
            s_rc = level

        aligned_out_shape_rc = np.ceil(np.array(ref_slide.reg_img_shape_rc)*s_rc).astype(int)

        return aligned_out_shape_rc

    def get_sorted_img_f_list(self):
        img_idx = [slide_obj.stack_idx for slide_obj in self.slide_dict.values()]
        img_order = np.argsort(img_idx)
        src_f_list = [self.original_img_list[i] for i in img_order]

        return src_f_list

    @valtils.deprecated_args(perceputally_uniform_channel_colors="colormap")
    def warp_and_save_slides(self, dst_dir, level=0, non_rigid=True,
                             crop=True,
                             colormap=slide_io.CMAP_AUTO,
                             interp_method="bicubic",
                             tile_wh=None, compression=DEFAULT_COMPRESSION, Q=100, pyramid=True):

        f"""Warp and save all slides

        Each slide will be saved as an ome.tiff. The extension of each file will
        be changed to ome.tiff if it is not already.

        Parameters
        ----------
        dst_dir : str
            Path to were the warped slides will be saved.

        level : int, optional
            Pyramid level to be warped. Default is 0, which means the highest
            resolution image will be warped and saved.

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied. Default is True

        crop: bool, str
            How to crop the registered images. If `True`, then the same crop used
            when initializing the `Valis` object will be used. If `False`, the
            image will not be cropped. If "overlap", the warped slide will be
            cropped to include only areas where all images overlapped.
            "reference" crops to the area that overlaps with the reference image,
            defined by `reference_img_f` when initialzing the `Valis object`.

        colormap : list
            List of RGB colors (0-255) to use for channel colors.
            If 'auto' (the default), the original channel colors ` will be used, if available.
            If `None`, no channel colors will be assigned.

        interp_method : str
            Interpolation method used when warping slide. Default is "bicubic"

        tile_wh : int, optional
            Tile width and height used to save image

        compression : str, optional
            Compression method used to save ome.tiff. See pyips for more details.

        Q : int
            Q factor for lossy compression

        """
        pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)

        src_f_list = self.get_sorted_img_f_list()
        cmap_is_str = False
        named_color_map = None
        if colormap is not None:
            if isinstance(colormap, str) and colormap == slide_io.CMAP_AUTO:
                cmap_is_str = True
            else:
                named_color_map = {self.get_slide(x).name:colormap[x] for x in colormap.keys()}

        for src_f in tqdm.tqdm(src_f_list, desc=SAVING_IMG_MSG, unit="image"):
            slide_obj = self.get_slide(src_f)
            slide_cmap = None
            is_rgb = slide_obj.reader.metadata.is_rgb
            if is_rgb:
                updated_channel_names = None
            elif colormap is not None:
                chnl_names = slide_obj.reader.metadata.channel_names
                updated_channel_names = slide_io.check_channel_names(chnl_names, is_rgb, nc=slide_obj.reader.metadata.n_channels)
                try:
                    if not cmap_is_str and named_color_map is not None:
                        slide_cmap = named_color_map[slide_obj.name]
                    else:
                        slide_cmap = colormap

                    slide_cmap = slide_io.check_colormap(colormap=slide_cmap, channel_names=updated_channel_names)
                except Exception as e:
                    traceback_msg = traceback.format_exc()
                    msg = f"Could not create colormap for the following reason:{e}"
                    valtils.print_warning(msg, traceback_msg=traceback_msg)

            dst_f = os.path.join(dst_dir, slide_obj.name + ".ome.tiff")

            slide_obj.warp_and_save_slide(dst_f=dst_f, level=level,
                                          non_rigid=non_rigid,
                                          crop=crop,
                                          src_f=slide_obj.src_f,
                                          interp_method=interp_method,
                                          colormap=slide_cmap,
                                          tile_wh=tile_wh,
                                          compression=compression,
                                          channel_names=updated_channel_names,
                                          Q=Q,
                                          pyramid=pyramid)


    @valtils.deprecated_args(perceputally_uniform_channel_colors="colormap")
    def warp_and_merge_slides(self, dst_f=None, level=0, non_rigid=True,
                              crop=True, channel_name_dict=None,
                              src_f_list=None, colormap=slide_io.CMAP_AUTO,
                              drop_duplicates=True, tile_wh=None,
                              interp_method="bicubic", compression=DEFAULT_COMPRESSION,
                              Q=100, pyramid=True):

        """Warp and merge registered slides

        Parameters
        ----------
        dst_f : str, optional
            Path to were the warped slide will be saved. If None, then the slides will be merged
            but not saved.

        level : int, optional
            Pyramid level to be warped. Default is 0, which means the highest
            resolution image will be warped and saved.

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied. Default is True

        crop: bool, str
            How to crop the registered images. If `True`, then the same crop used
            when initializing the `Valis` object will be used. If `False`, the
            image will not be cropped. If "overlap", the warped slide will be
            cropped to include only areas where all images overlapped.
            "reference" crops to the area that overlaps with the reference image,
            defined by `reference_img_f` when initialzing the `Valis object`.

        channel_name_dict : dict of lists, optional.
            key =  slide file name, value = list of channel names for that slide. If None,
            the the channel names found in each slide will be used.

        src_f_list : list of str, optionaal
            List of paths to slide to be warped. If None (the default), Valis.original_img_list
            will be used. Otherwise, the paths to which `src_f_list` points to should
            be an alternative copy of the slides, such as ones that have undergone
            processing (e.g. stain segmentation), had a mask applied, etc...

        colormap : list
            List of RGB colors (0-255) to use for channel colors

        drop_duplicates : bool, optional
            Whether or not to drop duplicate channels that might be found in multiple slides.
            For example, if DAPI is in multiple slides, then the only the DAPI channel in the
            first slide will be kept.

        tile_wh : int, optional
            Tile width and height used to save image

        interp_method : str
            Interpolation method used when warping slide. Default is "bicubic"

        compression : str
            Compression method used to save ome.tiff . Default is lzw, but can also
            be jpeg or jp2k. See pyips for more details.

        Q : int
            Q factor for lossy compression

        pyramid : bool
            Whether or not to save an image pyramid.

        Returns
        -------
        merged_slide : pyvips.Image
            Image with all channels merged. If `drop_duplicates` is True, then this
            will only contain unique channels.

        all_channel_names : list of str
            Name of each channel in the image

        ome_xml : str
            OME-XML string containing the slide's metadata

        """

        if channel_name_dict is not None:
            channel_name_dict_by_name = {valtils.get_name(k):channel_name_dict[k] for k in channel_name_dict}
        else:
            channel_name_dict_by_name = {slide_obj.name: [f"{c} ({slide_obj.name})" for c in slide_obj.reader.metadata.channel_names]
                                        for slide_obj in self.slide_dict.values()}

        if src_f_list is None:
            # Save in the sorted order. Will still be original order if imgs_ordered= True
            src_f_list = self.get_sorted_img_f_list()

        all_channel_names = []
        merged_slide = None

        expected_channel_order = list(chain.from_iterable([channel_name_dict_by_name[valtils.get_name(f)] for f in src_f_list]))
        if drop_duplicates:
            expected_channel_order = list(dict.fromkeys(expected_channel_order))

        for f in src_f_list:
            slide_name = valtils.get_name(os.path.split(f)[1])
            slide_obj = self.slide_dict[slide_name]

            warped_slide = slide_obj.warp_slide(level, non_rigid=non_rigid,
                                                crop=crop,
                                                interp_method=interp_method)

            keep_idx = list(range(warped_slide.bands))
            slide_channel_names = channel_name_dict_by_name[slide_obj.name]

            if drop_duplicates:
                keep_idx = [idx for idx  in range(len(slide_channel_names)) if
                            slide_channel_names[idx] not in all_channel_names]

            if len(keep_idx) == 0:
                msg= f"Have already added all channels in {slide_channel_names}. Ignoring {slide_name}"
                valtils.print_warning(msg)
                continue

            if drop_duplicates and warped_slide.bands != len(keep_idx):
                keep_channels = [warped_slide[c] for c in keep_idx]
                slide_channel_names = [slide_channel_names[idx] for idx in keep_idx]
                if len(keep_channels) == 1:
                    warped_slide = keep_channels[0]
                else:
                    warped_slide = keep_channels[0].bandjoin(keep_channels[1:])
            print(f"merging {', '.join(slide_channel_names)} from {slide_obj.name}")

            if merged_slide is None:
                merged_slide = warped_slide
            else:
                merged_slide = merged_slide.bandjoin(warped_slide)

            all_channel_names.extend(slide_channel_names)

        if merged_slide.bands == 1:
            merged_slide = merged_slide.copy(interpretation="b-w")
        else:
            merged_slide = merged_slide.copy(interpretation="multiband")

        assert all_channel_names == expected_channel_order

        if colormap is not None:
            cmap_dict = slide_io.check_colormap(colormap, all_channel_names)
        else:
            cmap_dict = None

        slide_obj = self.get_ref_slide()
        px_phys_size = slide_obj.reader.scale_physical_size(level)
        bf_dtype = slide_io.vips2bf_dtype(merged_slide.format)
        out_xyczt = slide_io.get_shape_xyzct((merged_slide.width, merged_slide.height), merged_slide.bands)

        ome_xml_obj = slide_io.create_ome_xml(out_xyczt, bf_dtype, is_rgb=False,
                                              pixel_physical_size_xyu=px_phys_size,
                                              channel_names=all_channel_names,
                                              colormap=cmap_dict)
        ome_xml = ome_xml_obj.to_xml()

        if dst_f is not None:
            dst_dir = os.path.split(dst_f)[0]
            pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)

            ref_slide = self.get_ref_slide()
            if tile_wh is None:
                tile_wh = slide_io.get_tile_wh(reader=ref_slide.reader,
                                    level=level,
                                    out_shape_wh=out_xyczt[0:2])

            slide_io.save_ome_tiff(merged_slide, dst_f=dst_f,
                                   ome_xml=ome_xml,tile_wh=tile_wh,
                                   compression=compression, Q=Q, pyramid=pyramid)

        return merged_slide, all_channel_names, ome_xml



