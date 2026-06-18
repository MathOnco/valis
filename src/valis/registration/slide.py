"""Slide class — stores registration state and warps images/points.

Import ``Slide`` from ``valis.registration`` rather than this sub-module.
"""

import logging
from typing import Optional, Union

import json
import os
import pathlib
from copy import deepcopy

import colour
import cv2
import numpy as np
import pyvips
import shapely
import tqdm
from skimage import exposure, transform

from .. import warp_tools
from .. import slide_io
from .. import slide_tools
from .. import preprocessing
from .. import valtils
from ._constants import (
    CROP_OVERLAP,
    CROP_REF,
    CROP_NONE,
    CropMode,
    DEFAULT_COMPRESSION,
    WARP_ANNO_MSG,
)
from .state import DisplacementField

logger = logging.getLogger(__name__)


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
        repr_str = (
            f"<{self.__class__.__name__}, name = {self.name}>"
            f", width={self.slide_dimensions_wh[0][0]}"
            f", height={self.slide_dimensions_wh[0][1]}"
            f", channels={self.reader.metadata.n_channels}"
            f", levels={len(self.slide_dimensions_wh)}"
            f", RGB={self.is_rgb}"
            f", dtype={self.image.dtype}>"
        )
        return repr_str

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

    def get_aligned_to_ref_slide_crop_xywh(
        self, ref_img_shape_rc, ref_M, scaled_ref_img_shape_rc=None
    ):
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

        mask, _ = self.val_obj.get_crop_mask(CROP_REF)

        if scaled_ref_img_shape_rc is not None:
            sxy = np.array([*scaled_ref_img_shape_rc[::-1]]) / np.array(
                [*ref_img_shape_rc[::-1]]
            )
        else:
            scaled_ref_img_shape_rc = ref_img_shape_rc
            sxy = np.ones(2)

        reg_txy = -ref_M[0:2, 2]
        slide_xywh = (*reg_txy * sxy, *scaled_ref_img_shape_rc[::-1])

        return slide_xywh, mask

    def get_overlap_crop_xywh(
        self, warped_img_shape_rc, scaled_warped_img_shape_rc=None
    ):
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
        mask, mask_bbox_xywh = self.val_obj.get_crop_mask(CROP_OVERLAP)

        if scaled_warped_img_shape_rc is not None:
            sxy = np.array([*scaled_warped_img_shape_rc[::-1]]) / np.array(
                [*warped_img_shape_rc[::-1]]
            )
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
            crop_xywh, mask = self.get_aligned_to_ref_slide_crop_xywh(
                ref_img_shape_rc=transformation_shape_rc,
                ref_M=ref_slide.M,
                scaled_ref_img_shape_rc=out_shape_rc,
            )
        elif crop == CROP_OVERLAP:
            transformation_shape_rc = np.array(ref_slide.reg_img_shape_rc)
            crop_xywh, mask = self.get_overlap_crop_xywh(
                warped_img_shape_rc=transformation_shape_rc,
                scaled_warped_img_shape_rc=out_shape_rc,
            )

        return crop_xywh, mask

    def get_crop_method(self, crop):
        """Get string or logic defining how to crop the image"""
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
        """Get position of pixel that has color used for background"""
        if self.img_type == slide_tools.IHC_NAME:
            # RGB. Get brightest pixel
            mean_rgb, color_mask, filtered_label_counts, color_clusterer = (
                preprocessing.find_dominant_colors(
                    self.image, cspace=cspace, return_xy_clusterer=True
                )
            )
            mean_jab = preprocessing.rgb2jab(mean_rgb, cspace=cspace)
            mean_jch = colour.models.Jab_to_JCh(mean_jab)

            # Find highest luminosity (L) and lowest colorfulness
            bg_idx = np.lexsort([mean_jch[:, 1], -mean_jch[:, 0]])[
                0
            ]  # Last column sorted 1st. Returns ascending order
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

        self.processed_img_f = os.path.join(
            self.val_obj.processed_dir, self.name + ".png"
        )
        self.rigid_reg_img_f = os.path.join(
            self.val_obj.reg_dst_dir, f"{stack_id}_f{self.name}.png"
        )
        self.non_rigid_reg_img_f = os.path.join(
            self.val_obj.non_rigid_dst_dir, f"{stack_id}_f{self.name}.png"
        )
        if self.stored_dxdy:
            bk_dxdy_f, fwd_dxdy_f = self.get_displacement_f()
            self._bk_dxdy_f = bk_dxdy_f
            self._fwd_dxdy_f = fwd_dxdy_f

    def get_displacement_f(self):
        bk_dxdy_f = os.path.join(
            self.val_obj.displacements_dir, f"{self.name}_bk_dxdy.tiff"
        )
        fwd_dxdy_f = os.path.join(
            self.val_obj.displacements_dir, f"{self.name}_fwd_dxdy.tiff"
        )

        return bk_dxdy_f, fwd_dxdy_f

    def get_bk_dxdy(self):
        if self._bk_dxdy_np is None and not self.stored_dxdy:
            return None

        elif self.stored_dxdy:
            bk_dxdy_f, _ = self.get_displacement_f()
            cropped_bk_dxdy = pyvips.Image.new_from_file(bk_dxdy_f)
            full_bk_dxdy = self.val_obj.pad_displacement(
                cropped_bk_dxdy,
                self.val_obj._full_displacement_shape_rc,
                self.val_obj._non_rigid_bbox,
            )

        else:
            if np.any(
                self._bk_dxdy_np.shape[1:2] != self.val_obj._full_displacement_shape_rc
            ):
                full_bk_dxdy = self.val_obj.pad_displacement(
                    self._bk_dxdy_np,
                    self.val_obj._full_displacement_shape_rc,
                    self.val_obj._non_rigid_bbox,
                )
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
            logger.error(f"Cannot set bk_dxdy when data is type {type(bk_dxdy)}")

    bk_dxdy = property(
        fget=get_bk_dxdy, fset=set_bk_dxdy, doc="Get and set backwards displacements"
    )

    def get_fwd_dxdy(self):
        if self._fwd_dxdy_np is None and not self.stored_dxdy:
            return None

        elif self.stored_dxdy:
            _, fwd_dxdy_f = self.get_displacement_f()
            cropped_fwd_dxdy = pyvips.Image.new_from_file(fwd_dxdy_f)
            full_fwd_dxdy = self.val_obj.pad_displacement(
                cropped_fwd_dxdy,
                self.val_obj._full_displacement_shape_rc,
                self.val_obj._non_rigid_bbox,
            )

        else:
            if np.any(
                self._fwd_dxdy_np.shape[1:2] != self.val_obj._full_displacement_shape_rc
            ):
                full_fwd_dxdy = self.val_obj.pad_displacement(
                    self._fwd_dxdy_np,
                    self.val_obj._full_displacement_shape_rc,
                    self.val_obj._non_rigid_bbox,
                )
            else:
                full_fwd_dxdy = self._fwd_dxdy_np

        return full_fwd_dxdy

    def set_fwd_dxdy(self, fwd_dxdy):
        if not isinstance(fwd_dxdy, pyvips.Image):
            self._fwd_dxdy_np = fwd_dxdy
        else:
            logger.error(f"Cannot set fwd_dxdy when data is type {type(fwd_dxdy)}")

    fwd_dxdy = property(
        fget=get_fwd_dxdy, fset=set_fwd_dxdy, doc="Get forward displacements"
    )

    def warp_img(
        self,
        img: Optional[np.ndarray] = None,
        non_rigid: bool = True,
        crop: Union[bool, str, "CropMode"] = True,
        interp_method: str = "bicubic",
    ) -> np.ndarray:
        """Warp an image using the registration parameters

        img : ndarray, optional
            The image to be warped. If None, then Slide.image
            will be warped.

        non_rigid : bool
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied.

        crop: bool, str, or CropMode
            How to crop the registered images. If `True`, then the same crop used
            when initializing the `Valis` object will be used. If `False`, the
            image will not be cropped. If "overlap" or CropMode.OVERLAP, the warped
            slide will be cropped to include only areas where all images overlapped.
            "reference" or CropMode.REFERENCE crops to the area that overlaps with
            the reference image, defined by `reference_img_f` when initializing
            the `Valis` object.

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

        if (
            self == ref_slide
            and crop == CROP_REF
            and np.all(warp_tools.get_shape(img)[0:2] == self.processed_img_shape_rc)
        ):
            # Save on computation time and avoid interpolation/rounding issues and return the original image
            return img

        if not np.all(img_shape_rc == self.processed_img_shape_rc):
            msg = (
                "scaling transformation for image with different shape. "
                "However, without knowing all of other image's shapes, "
                "the scaling may not be the same for all images, and so "
                "may not overlap."
            )
            logger.warning(msg)
            same_shape = False
            img_scale_rc = np.array(img_shape_rc) / (
                np.array(self.processed_img_shape_rc)
            )
            out_shape_rc = self.val_obj.get_aligned_slide_shape(img_scale_rc)

        else:
            same_shape = True
            out_shape_rc = self.reg_img_shape_rc

        if isinstance(crop, bool) or isinstance(crop, str):
            crop_method = self.get_crop_method(crop)
            if crop_method is not False:
                if crop_method == CROP_REF:
                    if not same_shape:
                        scaled_shape_rc = (
                            np.array(ref_slide.processed_img_shape_rc) * img_scale_rc
                        )
                    else:
                        scaled_shape_rc = ref_slide.processed_img_shape_rc
                elif crop_method == CROP_OVERLAP:
                    scaled_shape_rc = out_shape_rc

                bbox_xywh, _ = self.get_crop_xywh(
                    crop=crop_method, out_shape_rc=scaled_shape_rc
                )
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

        warped_img = warp_tools.warp_img(
            img,
            M=self.M,
            bk_dxdy=dxdy,
            out_shape_rc=out_shape_rc,
            transformation_src_shape_rc=self.processed_img_shape_rc,
            transformation_dst_shape_rc=self.reg_img_shape_rc,
            bbox_xywh=bbox_xywh,
            bg_color=bg_color,
            interp_method=interp_method,
        )

        return warped_img

    def warp_img_from_to(
        self,
        img,
        to_slide_obj,
        dst_slide_level=0,
        non_rigid=True,
        interp_method="bicubic",
        bg_color=None,
    ):
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
            to_slide_src_shape_rc = to_slide_obj.slide_dimensions_wh[dst_slide_level][
                ::-1
            ]
            aligned_slide_shape = self.val_obj.get_aligned_slide_shape(dst_slide_level)
        else:

            to_slide_src_shape_rc = np.array(dst_slide_level)

            dst_scale_rc = to_slide_src_shape_rc / np.array(
                to_slide_obj.processed_img_shape_rc
            )
            aligned_slide_shape = np.round(
                dst_scale_rc * np.array(to_slide_obj.reg_img_shape_rc)
            ).astype(int)

        if non_rigid:
            from_bk_dxdy = self.bk_dxdy
            to_fwd_dxdy = to_slide_obj.fwd_dxdy

        else:
            from_bk_dxdy = None
            to_fwd_dxdy = None

        warped_img = warp_tools.warp_img_from_to(
            img,
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
            interp_method=interp_method,
        )

        return warped_img

    @valtils.deprecated_args(crop_to_overlap="crop")
    def warp_slide(
        self,
        level,
        non_rigid=True,
        crop=True,
        src_f=None,
        interp_method="bicubic",
        reader=None,
    ):
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
                logger.warning(msg)
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

                slide_bbox_xywh, _ = self.get_crop_xywh(
                    crop=crop_method, out_shape_rc=scaled_aligned_shape_rc
                )

                if crop_method == CROP_REF:
                    assert np.all(slide_bbox_xywh[2:] == scaled_aligned_shape_rc[::-1])
                    if src_f == self.src_f and self == ref_slide:
                        # Shouldn't need to warp, but do checks just in case
                        no_rigid = True
                        no_non_rigid = True
                        if self.M is not None:
                            sxy = (
                                scaled_aligned_shape_rc / self.processed_img_shape_rc
                            )[::-1]
                            scaled_txy = sxy * self.M[:2, 2]
                            no_transforms = all(
                                self.M[:2, :2].reshape(-1) == [1, 0, 0, 1]
                            )
                            crop_to_origin = np.all(
                                np.abs(slide_bbox_xywh[0:2] + scaled_txy) < 1
                            )
                            no_rigid = no_transforms and crop_to_origin

                        if self.bk_dxdy is not None:
                            no_non_rigid = (
                                self.bk_dxdy.min() == 0 and self.bk_dxdy.max() == 0
                            )

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

        warped_slide = slide_tools.warp_slide(
            src_f,
            M=self.M,
            transformation_src_shape_rc=self.processed_img_shape_rc,
            transformation_dst_shape_rc=self.reg_img_shape_rc,
            aligned_slide_shape_rc=aligned_slide_shape,
            dxdy=bk_dxdy,
            level=level,
            series=self.series,
            interp_method=interp_method,
            bbox_xywh=slide_bbox_xywh,
            bg_color=bg_color,
            reader=reader,
        )
        return warped_slide

    def warp_and_save_slide(
        self,
        dst_f,
        level=0,
        non_rigid=True,
        crop=True,
        src_f=None,
        channel_names=None,
        colormap=slide_io.CMAP_AUTO,
        interp_method="bicubic",
        tile_wh=None,
        compression=DEFAULT_COMPRESSION,
        Q=100,
        pyramid=True,
        reader=None,
    ):
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

        warped_slide = self.warp_slide(
            level=level,
            non_rigid=non_rigid,
            crop=crop,
            interp_method=interp_method,
            src_f=src_f,
            reader=reader,
        )

        # Get ome-xml #
        ref_slide = self.val_obj.get_ref_slide()
        pixel_physical_size_xyu = ref_slide.reader.scale_physical_size(level)

        ome_xml_obj = slide_io.update_xml_for_new_img(
            img=warped_slide,
            reader=reader,
            level=level,
            channel_names=channel_names,
            colormap=colormap,
            pixel_physical_size_xyu=pixel_physical_size_xyu,
        )

        ome_xml = ome_xml_obj.to_xml()

        out_shape_wh = warp_tools.get_shape(warped_slide)[0:2][::-1]
        tile_wh = slide_io.get_tile_wh(
            reader=reader, level=level, out_shape_wh=out_shape_wh
        )

        slide_io.save_ome_tiff(
            warped_slide,
            dst_f=dst_f,
            ome_xml=ome_xml,
            tile_wh=tile_wh,
            compression=compression,
            Q=Q,
            pyramid=pyramid,
        )

    def warp_xy(
        self,
        xy: np.ndarray,
        M: Optional[np.ndarray] = None,
        slide_level: Union[int, tuple[int, int]] = 0,
        pt_level: Union[int, tuple[int, int]] = 0,
        non_rigid: bool = True,
        crop: Union[bool, str, "CropMode"] = True,
    ) -> np.ndarray:
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
                    aligned_slide_shape = self.val_obj.get_aligned_slide_shape(
                        slide_level
                    )
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

        warped_xy = warp_tools.warp_xy(
            xy,
            M=M,
            transformation_src_shape_rc=self.processed_img_shape_rc,
            transformation_dst_shape_rc=self.reg_img_shape_rc,
            src_shape_rc=pt_dim_rc,
            dst_shape_rc=aligned_slide_shape,
            fwd_dxdy=fwd_dxdy,
        )
        crop_method = self.get_crop_method(crop)
        if crop_method is not False:
            if crop_method == CROP_REF:
                ref_slide = self.val_obj.get_ref_slide()
                if isinstance(slide_level, int):
                    scaled_aligned_shape_rc = ref_slide.slide_dimensions_wh[
                        slide_level
                    ][::-1]
                else:
                    if len(slide_level) == 2:
                        scaled_aligned_shape_rc = slide_level
            elif crop_method == CROP_OVERLAP:
                scaled_aligned_shape_rc = aligned_slide_shape

            crop_bbox_xywh, _ = self.get_crop_xywh(crop_method, scaled_aligned_shape_rc)
            warped_xy -= crop_bbox_xywh[0:2]

        return warped_xy

    def warp_xy_from_to(
        self,
        xy,
        to_slide_obj,
        src_slide_level=0,
        src_pt_level=0,
        dst_slide_level=0,
        non_rigid=True,
    ):
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
            to_slide_src_shape_rc = to_slide_obj.slide_dimensions_wh[dst_slide_level][
                ::-1
            ]
        else:
            to_slide_src_shape_rc = np.array(dst_slide_level)

        if src_slide_level != 0:
            if np.issubdtype(type(src_slide_level), np.integer):
                aligned_slide_shape = self.val_obj.get_aligned_slide_shape(
                    src_slide_level
                )
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

        xy_in_unwarped_to_img = warp_tools.warp_xy_from_to(
            xy=xy,
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
            to_bk_dxdy=dst_bk_dxdy,
        )

        return xy_in_unwarped_to_img

    def warp_geojson(
        self,
        geojson_f: Union[str, pathlib.Path],
        M: Optional[np.ndarray] = None,
        slide_level: Union[int, tuple[int, int]] = 0,
        pt_level: Union[int, tuple[int, int]] = 0,
        non_rigid: bool = True,
        crop: Union[bool, str, "CropMode"] = True,
    ) -> dict:
        """Warp geometry using registration parameters

        Warps geometries to their location in the registered slide/image

        Parameters
        ----------
        geojson_f : str or Path
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
                    aligned_slide_shape = self.val_obj.get_aligned_slide_shape(
                        slide_level
                    )
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
                    scaled_aligned_shape_rc = ref_slide.slide_dimensions_wh[
                        slide_level
                    ][::-1]
                else:
                    if len(slide_level) == 2:
                        scaled_aligned_shape_rc = slide_level
            elif crop_method == CROP_OVERLAP:
                scaled_aligned_shape_rc = aligned_slide_shape

            crop_bbox_xywh, _ = self.get_crop_xywh(crop_method, scaled_aligned_shape_rc)
            shift_xy = crop_bbox_xywh[0:2]
        else:
            shift_xy = None

        warped_features = [None] * len(annotation_geojson["features"])
        for i, ft in tqdm.tqdm(
            enumerate(annotation_geojson["features"]),
            desc=WARP_ANNO_MSG,
            unit="annotation",
        ):
            geom = shapely.geometry.shape(ft["geometry"])
            warped_geom = warp_tools.warp_shapely_geom(
                geom,
                M=M,
                transformation_src_shape_rc=self.processed_img_shape_rc,
                transformation_dst_shape_rc=self.reg_img_shape_rc,
                src_shape_rc=pt_dim_rc,
                dst_shape_rc=aligned_slide_shape,
                fwd_dxdy=fwd_dxdy,
                shift_xy=shift_xy,
            )
            warped_ft = deepcopy(ft)
            warped_ft["geometry"] = shapely.geometry.mapping(warped_geom)
            warped_features[i] = warped_ft

        warped_geojson = {
            "type": annotation_geojson["type"],
            "features": warped_features,
        }

        return warped_geojson

    def warp_geojson_from_to(
        self,
        geojson_f,
        to_slide_obj,
        src_slide_level=0,
        src_pt_level=0,
        dst_slide_level=0,
        non_rigid=True,
    ):
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
            to_slide_src_shape_rc = to_slide_obj.slide_dimensions_wh[dst_slide_level][
                ::-1
            ]
        else:
            to_slide_src_shape_rc = np.array(dst_slide_level)

        if src_slide_level != 0:
            if np.issubdtype(type(src_slide_level), np.integer):
                aligned_slide_shape = self.val_obj.get_aligned_slide_shape(
                    src_slide_level
                )
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

        warped_features = [None] * len(annotation_geojson["features"])
        for i, ft in tqdm.tqdm(
            enumerate(annotation_geojson["features"]),
            desc=WARP_ANNO_MSG,
            unit="annotation",
        ):
            geom = shapely.geometry.shape(ft["geometry"])
            warped_geom = warp_tools.warp_shapely_geom_from_to(
                geom=geom,
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
                to_bk_dxdy=dst_bk_dxdy,
            )

            warped_ft = deepcopy(ft)
            warped_ft["geometry"] = shapely.geometry.mapping(warped_geom)
            warped_features[i] = warped_ft

        warped_geojson = {
            "type": annotation_geojson["type"],
            "features": warped_features,
        }

        return warped_geojson

    def pad_cropped_processed_img(self):
        """
        Pad cropped processed image to have original dimensions
        """
        vips_img = warp_tools.numpy2vips(self.processed_img)

        padded = vips_img.embed(
            self.processed_crop_bbox[0],
            self.processed_crop_bbox[1],
            self.uncropped_processed_img_shape_rc[1],
            self.uncropped_processed_img_shape_rc[0],
            extend=pyvips.enums.Extend.BLACK,
        )
        scaled_padded = warp_tools.resize_img(padded, self.processed_img_shape_rc)
        scaled_padded_np = warp_tools.vips2numpy(scaled_padded)

        return scaled_padded_np
