"""Virtual Alignment of pathoLogy Image Series

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

*. Images/slides are converted to numpy arrays. As WSI are often
too large to fit into memory, these images are usually lower resolution
images from different pyramid levels.

*. Images are processed to single channel images. They are then
normalized to make them look as similar as possible.

*. Image features are detected and then matched between all pairs of image.

*. If the order of images is unknown, they will be optimally ordered
based on their feature similarity

*. Rigid registration is performed serially, with each image being
rigidly aligned to the previous image in the stack. VALIS uses feature
detection to match and align images, but one can optionally perform
a final step that maximizes the mutual information betweeen each
pair of images.

* Non-rigid registration is then performed either by 1) aliging each image
towards the center of the stack, composing the deformation fields
along the way, or 2) using groupwise registration that non-rigidly aligns
the images to a common frame of reference.

* Error is measured by calculating the distance between registered
matched features.

The transformations found by VALIS can then be used to warp the full
resolution slides. It is also possible to merge non-RGB registered slides
to create a highly multiplexed image. These aligned and/or merged slides
can then be saved as ome.tiff images using pyvips.

In addition to warping images and slides, VALIS can also warp point data,
such as cell centoids or ROI coordinates.

Examples
--------

In this example, the slides that need to be aligned are located in
"../resources/slides/ihc_mrxs". Registration results will be saved
in  "./slide_registration_example", in which will be 6 folders:

1. *data* contains 2 files:
    * a summary spreadsheet of the alignment results, such
    as the registration error between each pair of slides, their
    dimensions, physical units, etc...

    * a pickled version of the registrar. This can be reloaded
    (unpickled) and used later. For example, one could perform
    the registration locally, but then use the pickled object
    to warp and save the slides on an HPC. Or, one could perform
    the registration and use the registrar later to warp
    points in the slide.

2. *overlaps* contains thumbnails showing the how the images
    would look if stacked without being registered, how they
    look after rigid registration, and how they would look
    after non-rigid registration.

3. *rigid_registration* shows thumbnails of how each image
    looks after performing rigid registration.

4. *non_rigid_registration* shows thumbnaials of how each
    image looks after non-rigid registration.

5. *deformation_fields* contains images showing what the
    non-rigid deformation would do to a triangular mesh.
    These can be used to get a better sense of how the
    images were altered by non-rigid warping

6. *processed* shows thumnails of the processed images.
    This are thumbnails of the images that are actually
    used to perform the registration. The pre-processing
    and normalization methods should try to make these
    images look as similar as possible.

After registraation is complete, one can view the
results to determine if they are acceptable. If they
are, then one can warp and save all of the slides.

>>> from valis import registration
>>> slide_src_dir = "/path/to/slides"
>>> results_dst_dir = "./slide_registration_example"
>>> registered_slide_dst_dir = "./slide_registration_example/registered_slides"

Perform registration

>>> registrar = registration.Valis(slide_src_dir, results_dst_dir)
>>> rigid_registrar, non_rigid_registrar, error_df = registrar.register()

View results in "./slide_registration_example".

If they look good, warp and save the slides as ome.tiff

>>> registrar.warp_and_save_slides(registered_slide_dst_dir)

Don't forget to kill the JVM

>>> registration.kill_jvm()


This next example shows how to align a series of CyCIF images and then
merge them into a single ome.tiff image

>>> from valis import registration
>>> slide_src_dir = "/path/to/slides"
>>> results_dst_dir = "./slide_merging_example"
>>> merged_slide_dst_f = "./slide_merging_example/merged_slides.ome.tiff"

Create a Valis object and use it to register the slides

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

VALIS can also be used to warp point data. Here it will register 2 TMA cores and then
warp cell positions located in a separate .csv

>>> import numpy as np
>>> import pandas as pd
>>> import pathlib
>>> import matplotlib.pyplot as plt
>>> slide_src_dir = "path/to/slides"
>>> point_data_dir = "path/to/cell_positions"
>>> results_dst_dir = "./point_warping_example"

Create a Valis object and use it to register the slides in slide_src_dir

>>> registrar = registration.Valis(slide_src_dir, results_dst_dir)
>>> rigid_registrar, non_rigid_registrar, error_df = registrar.register()

Warp points, which are located in csv files

>>> point_data_list = list(pathlib.Path(point_data_dir).rglob("*.csv"))
>>> for f in point_data_list:
...     # Get Slide associated with the slide from which the point data originated
...     # Point data and image have similar file names
...     fname = os.path.split(f)[1]
...     corresponding_img = fname.split(".tif")[0]
...     slide_obj = registrar.get_slide(corresponding_img)
...
...     # Read data and calculate cell centroids (x, y)
...     points_df = pd.read_csv(f)
...     x = np.mean(points_df[["XMin", "XMax"]], axis=1).values
...     y = np.mean(points_df[["YMin", "YMax"]], axis=1).values
...     xy = np.dstack([x, y])[0]
...
...     # Use Slide to warp the coordinates
...     warped_xy = slide_obj.warp_xy(xy)
...     # Update dataframe with registered cell centroids
...     points_df[["registered_x", "registered_y"]] = warped_xy
...
...     # Save updated dataframe
...     pt_f_out = os.path.split(f)[1].replace(".csv", "_registered.csv")
...     full_pt_f_out = os.path.join(results_dst_dir, pt_f_out)
...     points_df.to_csv(full_pt_f_out, index=False)
>>> valis.kill_jvm() # Kill the JVM

In most cases the default paramters work well. However, one may wish to try
non-default parameters. In this case, we will use KAZE for
feature detection and description, and SimpleElastix for non-rigid warping.

>>> from valis import feature_detectors, non_rigid_registrars
>>> feature_detector = feature_detectors.KazeFD
>>> non_rigid_registrar = non_rigid_registrars.SimpleElastixWarper
>>> slide_src_dir = data.ihc2_src_dir
>>> registrar = registration.Valis(slide_src_dir, results_dst_dir,
...                         feature_detector_cls=feature_detector,
...                         non_rigid_registrar_cls=non_rigid_registrar)

rigid_registrar, non_rigid_registrar, error_df = registrar.register()

Finlly, VALIS can also be used to convert slides to ome.tiff. Here
is an example of converting a large tiff to ome.tiff. Note that
bioformats does not seem to read this file correctly, so it is a
good case where conversion to ome.tiff mmight be needed. Takes
approximately 5 minutes.

>>> from valis import slide_io
>>> slide_src_f = "path/to/slide"
>>> converted_slide_f = "./slide_conversion_example/Beg_P1_48_C1D15_06S17081023.ome.tiff"
>>> slide_io.convert_to_ome_tiff(slide_src_f, converted_slide_f, level=0, perceputally_uniform_channel_colors=True)

After access to slides is no longer needed, be sure to kill the JVM

>>> registration.kill_jvm()

Attributes
----------
CONVERTED_IMG_DIR : str
    Where thumnails of the converted images will be saved

PROCESSED_IMG_DIR : str
    Where thumnails of the processed images will be saved

RIGID_REG_IMG_DIR : str
    Where thumnails of the rigidly aligned images will be saved

NON_RIGID_REG_IMG_DIR : str
    Where thumnails of the non-rigidly aligned images will be saved

DEFORMATION_FIELD_IMG_DIR : str
    Where thumnails of the non-rigid deformation fields will be saved

OVERLAP_IMG_DIR : str
    Where thumnails of the image overlaps will be saved

REG_RESULTS_DATA_DIR : str
    Where the summary and pickled Valis object will be saved

DEFAULT_BRIGHTFIELD_CLASS : preprocessing.ImageProcesser
    Default ImageProcesser class used to process brightfield images

DEFAULT_BRIGHTFIELD_PROCESSING_ARGS : dict
    Dictionary of keyward arguments passed to DEFAULT_BRIGHTFIELD_CLASS.process_image()

DEFAULT_FLOURESCENCE_CLASS : preprocessing.ImageProcesser
    Default ImageProcesser class used to process immunofluorescence images

DEFAULT_FLOURESCENCE_PROCESSING_ARGS : dict
    Dictionary of keyward arguments passed to DEFAULT_FLOURESCENCE_PROCESSING_ARGS.process_image()

DEFAULT_NORM_METHOD : str
    Default image normalization method

DEFAULT_FD : feature_detectors.FeatureDD
    Default feature detector class

DEFAULT_TRANSFORM_CLASS : skimage.transform.GeometricTransform
    Default rigid transformer class

DEFAULT_MATCH_FILTER : str
    Default match filtering method to use

DEFAULT_SIMILARITY_METRIC : str
    Default similarity metric used to compare images

DEFAULT_AFFINE_OPTIMIZER_CLASS : affine_optimizer.AffineOptimizer
    Default affine optimaer to use. If None, affine optimization will not be performed

DEFAULT_NON_RIGID_CLASS : non_rigid_registrars.NonRigidRegistrar
    Default non-rigid registrar. If None, non-rigid registration will not be performed

DEFAULT_NON_RIGID_KWARGS : dict
    Default parameters used to intialize DEFAULT_NON_RIGID_CLASS

"""

import traceback
import re
import os
import numpy as np
import pathlib
from skimage import io, transform, exposure
from time import time
import tqdm
import pandas as pd
import pickle
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

# Destination directories #
CONVERTED_IMG_DIR = "images"
PROCESSED_IMG_DIR = "processed"
RIGID_REG_IMG_DIR = "rigid_registration"
NON_RIGID_REG_IMG_DIR = "non_rigid_registration"
DEFORMATION_FIELD_IMG_DIR = "deformation_fields"
OVERLAP_IMG_DIR = "overlaps"
REG_RESULTS_DATA_DIR = "data"

# Default image processing #
DEFAULT_BRIGHTFIELD_CLASS = preprocessing.ColorfulStandardizer
DEFAULT_BRIGHTFIELD_PROCESSING_ARGS = {'c': preprocessing.DEFAULT_COLOR_STD_C, "h": 0}
DEFAULT_FLOURESCENCE_CLASS = preprocessing.ChannelGetter
DEFAULT_FLOURESCENCE_PROCESSING_ARGS = {"channel": "dapi", "adaptive_eq": True}
DEFAULT_NORM_METHOD = "img_stats"

# Default rigid registration parameters #
DEFAULT_FD = feature_detectors.VggFD
DEFAULT_TRANSFORM_CLASS = transform.SimilarityTransform
DEFAULT_MATCH_FILTER = feature_matcher.RANSAC_NAME
DEFAULT_SIMILARITY_METRIC = "n_matches"
DEFAULT_AFFINE_OPTIMIZER_CLASS = None  # affine_optimizer.AffineOptimizerMattesMI
DEFAULT_MAX_PROCESSED_IMG_SIZE = 850
DEFAULT_MAX_IMG_DIM = 850
DEFAULT_THUMBNAIL_SIZE = 500


# Rigid registration kwarg keys #
AFFINE_OPTIMIZER_KEY = "affine_optimizer"
TRANSFORMER_KEY = "transformer"
SIM_METRIC_KEY = "similarity_metric"
FD_KEY = "feature_detector"
MATCHER_KEY = "matcher"
NAME_KEY = "name"
IMAGES_ORDERD_KEY = "imgs_ordered"
QT_EMMITER_KEY = "qt_emitter"

# Rigid registration kwarg keys #
NON_RIGID_REG_CLASS_KEY = "non_rigid_reg_class"
NON_RIGID_REG_PARAMS_KEY = "non_rigid_reg_params"
NON_RIGID_REG_REF_IMG_KEY = "ref_img_name"
NON_RIGID_USE_XY_KEY = "moving_to_fixed_xy"

# Default non-rigid registration parameters #
DEFAULT_NON_RIGID_CLASS = non_rigid_registrars.OpticalFlowWarper
DEFAULT_NON_RIGID_KWARGS = {}


def init_jvm():
    """Initialize JVM for BioFormats
    """
    slide_io.init_jvm()


def kill_jvm():
    """Kill JVM for BioFormats
    """
    slide_io.kill_jvm()


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

    overlap_mask_bbox_xywh : ndarray
        Bounding box of overlap_mask, which covers intersection of all rigidly
        aligned images. Values are [min_x, min_y, width, height]

    overlap_mask : ndarray
        Mask that covers intersection of all rigidly aligned images.

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

    """

    def __init__(self, src_f, image, val_obj, reader):
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

        """

        self.src_f = src_f
        self.image = image
        self.val_obj = val_obj
        self.reader = reader

        # Metadata #
        self.img_type = reader.guess_image_type()
        self.is_rgb = reader.metadata.is_rgb
        self.slide_shape_rc = reader.metadata.slide_dimensions[0][::-1]
        self.series = reader.metadata.series
        self.slide_dimensions_wh = reader.metadata.slide_dimensions
        self.resolution = np.mean(reader.metadata.pixel_physical_size_xyu[0:2])
        self.units = reader.metadata.pixel_physical_size_xyu[2]
        self.original_xml = reader.metadata.original_xml

        self.name = valtils.get_name(src_f)

        # To be filled in during registration #
        self.processed_img_f = None
        self.rigid_reg_img_f = None
        self.stack_idx = None
        self.non_rigid_reg_img_f = None
        self.aligned_slide_shape_rc = None

        self.processed_img_shape_rc = None
        self.reg_img_shape_rc = None
        self.M = None
        self.bk_dxdy = None
        self.fwd_dxdy = None
        self.overlap_mask_bbox_xywh = None
        self.overlap_mask = None
        self.xy_matched_to_prev = None
        self.xy_in_prev = None
        self.xy_matched_to_prev_in_bbox = None
        self.xy_in_prev_in_bbox = None

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

    def warp_img(self, img=None, non_rigid=True):
        """Warp an image using the registration parameters

        img : ndarray, optional
            The image to be warped. If None, then Slide.image
            will be warped.

        non_rigid : bool
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied.

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

        if not np.all(img.shape[0:2] == self.processed_img_shape_rc):
            img_scale_rc = np.array(img.shape[0:2])/(np.array(self.processed_img_shape_rc))
            out_shape_rc = np.ceil(np.array(self.reg_img_shape_rc)*img_scale_rc).astype(int)
        else:
            out_shape_rc = self.reg_img_shape_rc

        warped_img = warp_tools.warp_img(img, M=self.M,
                                         bk_dxdy=dxdy,
                                         out_shape_rc=out_shape_rc,
                                         transformation_src_shape_rc=self.processed_img_shape_rc,
                                         transformation_dst_shape_rc=self.reg_img_shape_rc)

        return warped_img

    def warp_slide(self, level, non_rigid=True, crop_to_overlap=True,
                   src_f=None, bg_color=None, interp_method="bicubic"):
        """Warp a slide using registration parameters

        Parameters
        ----------
        level : int
            Pyramid level to be warped

        non_rigid : bool, optional
            Whether or not to conduct non-rigid warping. If False,
            then only a rigid transformation will be applied. Default is True

        crop_to_overlap : bool, optional
            Whether or not to crop the warped slide so that it fits within
            Slide.overlap_mask_bbox_xywh. Default is True.

        src_f : str, optional
           Path of slide to be warped. If None (the default), Slide.src_f
           will be used. Otherwise, the file to which `src_f` points to should
           be an alternative copy of the slide, such as one that has undergone
           processing (e.g. stain segmentation), has a mask applied, etc...

        bg_color : ndarray, str, optional
            Color used to fill in black background. If "auto",
            the color will be the same as the most luminescent regions of the slide.
            Alternatively, the RGB values can be provided, but should be between 0-255.
            Default is None, which means the background will not be colored.

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
                print("Need slide level to be an integer indicating pyramid level")
            aligned_slide_shape = self.val_obj.get_aligned_slide_shape(level)
        else:
            aligned_slide_shape = self.aligned_slide_shape_rc

        warped_slide = slide_tools.warp_slide(src_f, M=self.M,
                                              in_shape_rc=self.processed_img_shape_rc,
                                              aligned_img_shape_rc=self.reg_img_shape_rc,
                                              aligned_slide_shape_rc=aligned_slide_shape,
                                              dxdy=bk_dxdy, level=level, series=self.series,
                                              interp_method=interp_method, bg_color=bg_color)

        if crop_to_overlap:
            to_slide_scaling = np.array([warped_slide.width, warped_slide.height]) / np.array(self.reg_img_shape_rc[::-1])
            to_slide_transformer = transform.SimilarityTransform(scale=to_slide_scaling)
            overlap_bbox = warp_tools.bbox2xy(self.overlap_mask_bbox_xywh)
            slide_overlap_bbox = to_slide_transformer(overlap_bbox)
            slide_overlap_xywh = warp_tools.xy2bbox(slide_overlap_bbox)
            slide_overlap_xywh[0:2] = np.floor(slide_overlap_xywh[0:2])
            slide_overlap_xywh[2:] = np.ceil(slide_overlap_xywh[2:])
            slide_overlap_xywh = tuple(slide_overlap_xywh.astype(int))

            warped_slide = warped_slide.crop(*slide_overlap_xywh)

        return warped_slide

    def warp_and_save_slide(self, dst_f, level=0, non_rigid=True,
                            crop_to_overlap=True, src_f=None,
                            channel_names=None,
                            perceputally_uniform_channel_colors=False,
                            bg_color=None, interp_method="bicubic",
                            tile_wh=None, compression="lzw"):

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

        crop_to_overlap : bool, optional
            Whether or not to crop the warped slide so that it fits within
            Slide.overlap_mask_bbox_xywh. Default is True.

        channel_names : list, optional
            List of channel names. If None, then Slide.reader
            will attempt to find the channel names associated with `src_f`.

        src_f : str, optional
           Path of slide to be warped. If None (the deffault), Slide.src_f
           will be used. Otherwise, the file to which `src_f` points to should
           be an alternative copy of the slide, such as one that has undergone
           processing (e.g. stain segmentation), has a mask applied, etc...

        bg_color : ndarray, str, optional
            Color used to fill in black background. If "auto",
            the color will be the same as the most luminescent regions of the slide.
            Alternatively, the RGB values can be provided, but should be between 0-255.
            Default is None, which means the background will not be colored.

        interp_method : str
            Interpolation method used when warping slide. Default is "bicubic"

        tile_wh : int, optional
            Tile width and height used to save image

        compression : str
            Compression method used to save ome.tiff . Default is lzw, but can also
            be jpeg or jp2k. See pyips for more details.

        """

        warped_slide = self.warp_slide(level=level, non_rigid=non_rigid,
                                       crop_to_overlap=crop_to_overlap,
                                       interp_method=interp_method, bg_color=bg_color)

        # Get ome-xml #
        slide_meta = self.reader.metadata
        if slide_meta.pixel_physical_size_xyu[2] == slide_io.PIXEL_UNIT:
            px_phys_size = None
        else:
            px_phys_size = self.reader.scale_physical_size(level)

        if channel_names is None:
            if src_f is None:
                channel_names = slide_meta.channel_names
            else:
                reader_cls = slide_io.get_slide_reader(src_f)
                reader = reader_cls(src_f)
                channel_names = reader.metadata.channel_names

        bf_dtype = slide_io.vips2bf_dtype(warped_slide.format)
        out_xyczt = slide_io.get_shape_xyzct((warped_slide.width, warped_slide.height), warped_slide.bands)
        ome_xml_obj = slide_io.update_xml_for_new_img(slide_meta.original_xml,
                                                      new_xyzct=out_xyczt,
                                                      bf_dtype=bf_dtype,
                                                      is_rgb=self.is_rgb,
                                                      pixel_physical_size_xyu=px_phys_size,
                                                      channel_names=channel_names,
                                                      perceputally_uniform_channel_colors=perceputally_uniform_channel_colors
                                                      )

        ome_xml = ome_xml_obj.to_xml()
        if tile_wh is None:
            tile_wh = slide_meta.optimal_tile_wh
            if level != 0:
                down_sampling = np.mean(slide_meta.slide_dimensions[level]/slide_meta.slide_dimensions[0])
                tile_wh = int(np.round(tile_wh*down_sampling))
                tile_wh = tile_wh - (tile_wh % 16)  # Tile shape must be multiple of 16
                if tile_wh < 16:
                    tile_wh = 16
                if np.any(np.array(out_xyczt[0:2]) < tile_wh):
                    tile_wh = min(out_xyczt[0:2])

        slide_io.save_ome_tiff(warped_slide, dst_f=dst_f, ome_xml=ome_xml,
                               tile_wh=tile_wh, compression=compression)

    def warp_xy(self, xy, M=None, slide_level=0, pt_level=0, non_rigid=True):
        """Warp points using registration parameters

        Warps `xy` to their location in the registered slide/image

        Parameters
        ----------
        xy : ndarray
            (N, 2) array of points to be warped. Must be x,y coordinates

        slide_level: int, tuple, optional
            Pyramid level of the slide. Used to scale transformation matrices.
            Can also be the shape of the unwarped image (row, col) into which
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

        """

        if M is None:
            M = self.M

        if np.issubdtype(type(pt_level), np.integer):
            pt_dim_rc = self.slide_dimensions_wh[pt_level][::-1]
        else:
            pt_dim_rc = np.array(pt_level)

        if slide_level != 0:
            if np.issubdtype(type(slide_level), np.integer):
                aligned_slide_shape = self.val_obj.get_aligned_slide_shape(slide_level)
            else:
                aligned_slide_shape = np.array(slide_level)
        else:
            aligned_slide_shape = self.aligned_slide_shape_rc

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

        return warped_xy

    def warp_xy_from_to(self, xy, to_slide_obj, src_pt_level=0,
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
            dst_shape_rc = to_slide_obj.slide_dimensions_wh[dst_slide_level][::-1]
        else:
            dst_shape_rc = np.array(dst_slide_level)

        if non_rigid:
            src_fwd_dxdy = self.fwd_dxdy
            dst_bk_dxdy = to_slide_obj.bk_dxdy

        else:
            src_fwd_dxdy = None
            dst_bk_dxdy = None

        xy_in_unwarped_to_img = \
            warp_tools.warp_xy_from_to(xy=xy,
                                       from_M=self.M,
                                       registered_img_shape_rc=self.reg_img_shape_rc,
                                       from_transformation_src_shape_rc=self.processed_img_shape_rc,
                                       from_src_shape_rc=src_pt_dim_rc,
                                       from_fwd_dxdy=src_fwd_dxdy,
                                       to_M=to_slide_obj.M,
                                       to_transformation_src_shape_rc=to_slide_obj.processed_img_shape_rc,
                                       to_src_shape_rc=dst_shape_rc,
                                       to_bk_dxdy=dst_bk_dxdy
                                       )

        return xy_in_unwarped_to_img


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

    aligned_slide_shape_rc: tuple of int
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

    summary_df : pd.Dataframe
        Pandas dataframe containing information about the results, such
        as the error, shape of aligned slides, time to completion, etc...

    start_time : float
        The time at which registation was initiated.

    end_rigid_time : float
        The time at which rigid registation was completed.

    end_non_rigid_time : float
        The time at which non-rigid registation was completed.

    qt_emitter : PySide2.QtCore.Signal
        Used to emit signals that update the GUI's progress bars

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

    def __init__(self, src_dir, dst_dir, series=None, name=None, img_type=None,
                 feature_detector_cls=DEFAULT_FD, transformer_cls=DEFAULT_TRANSFORM_CLASS,
                 affine_optimizer_cls=DEFAULT_AFFINE_OPTIMIZER_CLASS,
                 similarity_metric=DEFAULT_SIMILARITY_METRIC,
                 match_filter_method=DEFAULT_MATCH_FILTER, imgs_ordered=False,
                 non_rigid_registrar_cls=DEFAULT_NON_RIGID_CLASS,
                 non_rigid_reg_params=DEFAULT_NON_RIGID_KWARGS, img_list=None,
                 resolution_xyu=None, slide_dims_dict_wh=None,
                 max_image_dim_px=DEFAULT_MAX_IMG_DIM,
                 max_processed_image_dim_px=DEFAULT_MAX_PROCESSED_IMG_SIZE,
                 thumbnail_size=DEFAULT_THUMBNAIL_SIZE,
                 norm_method=DEFAULT_NORM_METHOD, qt_emitter=None):

        """
        src_dir: str
            Path to directory containing the slides that will be registered.

        dst_dir : str
            Path to where the results should be saved.

        name : str, optional
            Descriptive name of registrar, such as the sample's name

        series : int, optional
            Slide series to that was read. If None, series will be set to 0.

        img_type : str, optional
            The type of image, either "brightfield", "fluorescence",
            or "multi". If None, VALIS will guess `img_type`
            of each image, based on the number of channels and datatype.
            Will assume that RGB = "brightfield",
            otherwise `img_type` will be set to "fluorescence".

        feature_detector_cls : FeatureDD, optional
            Uninstantiated FeatureDD object that detects and computes
            image features. Default is VggFD. The
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

        match_filter_method: str, optional
            "GMS" will use filter_matches_gms() to remove poor matches.
            This uses the Grid-based Motion Statistics (GMS) or RANSAC.

        imgs_ordered : bool, optional
            Boolean defining whether or not the order of images in img_dir
            are already in the correct order. If True, then each filename should
            begin with the number that indicates its position in the z-stack. If
            False, then the images will be sorted by ordering a feature distance
            matix. Default is False.

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
            a SimpleITK.ParameterMap. Note that numeric values nedd to be
            converted to strings. See the NonRigidRegistrar classes in
            `non_rigid_registrars` for the available non-rigid registration
            methods and arguments.

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

        thumbnail_size : int, optional
            Maximum width or height of thumbnails that show results

        norm_method : str
            Name of method used to normalize the processed images. Options
            are "histo_match" for histogram matching, "img_stats" for normalizing by
            image statistics. See preprocessing.match_histograms
            and preprocessing.norm_khan for details.

        qt_emitter : PySide2.QtCore.Signal, optional
            Used to emit signals that update the GUI's progress bars

        """

        if name is None:
            name = os.path.split(src_dir)[1]
        self.name = name.replace(" ", "_")

        # Set paths #
        self.src_dir = src_dir
        self.dst_dir = os.path.join(dst_dir, self.name)
        if img_list is not None:
            self.original_img_list = img_list
        else:
            self.get_imgs_in_dir()
        self.set_dst_paths()

        # Some information may already be provided #
        self.slide_dims_dict_wh = slide_dims_dict_wh
        self.resolution_xyu = resolution_xyu
        self.image_type = img_type

        # Results fields #
        if series is None:
            series = 0
        self.series = series
        self.size = 0
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

        # Setup rigid registration #
        self.rigid_registrar = None
        self._set_rigid_reg_kwargs(name=name,
                                   feature_detector=feature_detector_cls,
                                   similarity_metric=similarity_metric,
                                   match_filter_method=match_filter_method,
                                   transformer=transformer_cls,
                                   affine_optimizer=affine_optimizer_cls,
                                   imgs_ordered=imgs_ordered,
                                   qt_emitter=qt_emitter)

        # Setup non-rigid registration #
        self.non_rigid_registrar = None
        self.non_rigid_registrar_cls = non_rigid_registrar_cls
        if non_rigid_registrar_cls is not None:
            self._set_non_rigid_reg_kwargs(name=name,
                                           non_rigid_reg_class=non_rigid_registrar_cls,
                                           non_rigid_reg_params=non_rigid_reg_params,
                                           qt_emitter=qt_emitter)

        # Info realted to saving images to view results #
        self.thumbnail_size = thumbnail_size
        self.original_overlap_img = None
        self.rigid_overlap_img = None
        self.non_rigid_overlap_img = None

        self.has_rounds = False
        self.norm_method = norm_method
        self.summary_df = None
        self.start_time = None
        self.end_rigid_time = None
        self.end_non_rigid_time = None

    def _set_rigid_reg_kwargs(self, name, feature_detector, similarity_metric,
                              match_filter_method, transformer, affine_optimizer,
                              imgs_ordered, qt_emitter):

        """Set rigid registration kwargs
        Keyword arguments will be passed to `serial_rigid.register_images`

        """

        matcher = feature_matcher.Matcher(match_filter_method=match_filter_method)
        if affine_optimizer is not None:
            afo = affine_optimizer(transform=transformer.__name__)
        else:
            afo = affine_optimizer

        self.rigid_reg_kwargs = {NAME_KEY: name,
                                 FD_KEY: feature_detector(),
                                 SIM_METRIC_KEY: similarity_metric,
                                 TRANSFORMER_KEY: transformer(),
                                 MATCHER_KEY: matcher,
                                 AFFINE_OPTIMIZER_KEY: afo,
                                 IMAGES_ORDERD_KEY: imgs_ordered,
                                 QT_EMMITER_KEY: qt_emitter
                                 }

        # Save methods as strings since some objects cannot be pickled #
        self.feature_descriptor_str = self.rigid_reg_kwargs[FD_KEY].kp_descriptor_name
        self.feature_detector_str = self.rigid_reg_kwargs[FD_KEY].kp_detector_name
        self.transform_str = self.rigid_reg_kwargs[TRANSFORMER_KEY].__class__.__name__
        self.similarity_metric = self.rigid_reg_kwargs[SIM_METRIC_KEY]
        self.match_filter_method = match_filter_method

    def _set_non_rigid_reg_kwargs(self, name, non_rigid_reg_class, non_rigid_reg_params, qt_emitter):
        """Set non-rigid registration kwargs
        Keyword arguments will be passed to `serial_non_rigid.register_images`

        """

        self.non_rigid_reg_kwargs = {NAME_KEY: name,
                                     NON_RIGID_REG_CLASS_KEY: non_rigid_reg_class,
                                     NON_RIGID_REG_PARAMS_KEY: non_rigid_reg_params,
                                     QT_EMMITER_KEY: qt_emitter
                                     }

        self.non_rigid_reg_class_str = self.non_rigid_reg_kwargs[NON_RIGID_REG_CLASS_KEY].__name__

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
                    matching_f = [ff for ff in full_path_list if re.search(dir_name, ff) is not None and os.path.split(ff)[1] != dir_name]
                    if len(matching_f) == 1:
                        if not matching_f[0] in self.original_img_list:
                            #Make sure that file not already in list
                            self.original_img_list.extend(matching_f)
                            img_names.append(dir_name)

                    elif len(matching_f) > 1:
                        msg = f"found {len(matching_f)} matches for {dir_name}: {', '.join(matching_f)}"
                        valtils.print_warning(msg)
                    elif len(matching_f) == 0:
                        msg = f"Can't find slide file associated with {dir_name}"
                        valtils.print_warning(msg)

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

    def convert_imgs(self, series=None, reader_cls=None):
        """Convert slides to images and create dictionary of Slides.

        series : int
            Slide series to be read.

        reader_cls : SlideReader, optional
            Uninstantiated SlideReader class that will convert
            the slide to an image, and also collect metadata.

        """

        if series is None:
            series = 0

        img_types = []
        self.size = 0
        for f in tqdm.tqdm(self.original_img_list):
            if reader_cls is None:
                reader_cls = slide_io.get_slide_reader(f, series=series)
            reader = reader_cls(f, series=series)
            slide_dims = reader.metadata.slide_dimensions
            levels_in_range = np.where(slide_dims.max(axis=1) < self.max_image_dim_px)[0]
            if len(levels_in_range) > 0:
                level = levels_in_range[0]
            else:
                level = len(slide_dims) - 1

            img = reader.slide2image(level=level, series=series)

            scaling = np.min(self.max_image_dim_px/np.array(img.shape[0:2]))
            if scaling < 1:
                img = warp_tools.rescale_img(img, scaling)

            slide_obj = Slide(f, img, self, reader)
            img_types.append(slide_obj.img_type)
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
            print(f"Smallest image is less than max_image_dim_px. parameter max_image_dim_px is being set to {min_max_wh}")
            self.max_image_dim_px = min_max_wh
            for slide_obj in self.slide_dict.values():
                # Rescale images
                scaling = self.max_image_dim_px/max(slide_obj.image.shape[0:2])
                assert scaling <= self.max_image_dim_px
                if scaling < 1:
                    slide_obj.image =  warp_tools.rescale_img(slide_obj.image, scaling)

        if self.max_processed_image_dim_px > self.max_image_dim_px:
            print(f"parameter max_processed_image_dim_px also being updated to {self.max_image_dim_px}")
            self.max_processed_image_dim_px = self.max_image_dim_px

    def create_original_composite_img(self, rigid_registrar):
        """Create imaage showing how images overlap before registration
        """

        min_r = np.inf
        max_r = 0
        min_c = np.inf
        max_c = 0
        composite_img_list = [None] * self.size
        for i, img_obj in enumerate(rigid_registrar.img_obj_list):
            img = img_obj.image
            padded_img = transform.warp(img, img_obj.T, preserve_range=True,
                                        output_shape=img_obj.padded_shape_rc)

            composite_img_list[i] = padded_img

            img_corners_rc = warp_tools.get_corners_of_image(img.shape[0:2])
            warped_corners_xy = warp_tools.warp_xy(img_corners_rc[:, ::-1], img_obj.T)
            min_r = min(warped_corners_xy[:, 1].min(), min_r)
            max_r = max(warped_corners_xy[:, 1].max(), max_r)
            min_c = min(warped_corners_xy[:, 0].min(), min_c)
            max_c = max(warped_corners_xy[:, 0].max(), max_c)

        composite_img = np.dstack(composite_img_list)
        cmap = viz.jzazbz_cmap()
        # cmap = viz.cam16ucs_cmap()
        channel_colors = viz.get_n_colors(cmap, composite_img.shape[2])
        overlap_img = viz.color_multichannel(composite_img, channel_colors, rescale_channels=True)

        min_r = int(min_r)
        max_r = int(np.ceil(max_r))
        min_c = int(min_c)
        max_c = int(np.ceil(max_c))
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

    def process_imgs(self, brightfield_processing_cls, brightfield_processing_kwargs,
                     if_processing_cls, if_processing_kwargs):

        f"""Process images to make them look as similar as possible

        Images will also be normalized after images are processed

        Parameters
        ----------
        brightfield_processing_cls : ImageProcesser
            ImageProcesser to pre-process brightfield images to make them look as similar as possible.
            Should return a single channel uint8 image. The default function is
            {DEFAULT_BRIGHTFIELD_CLASS.__name__} will be used for
            `img_type` = {slide_tools.IHC_NAME}. {DEFAULT_BRIGHTFIELD_CLASS.__name__}
            is located in the preprocessing module.

        brightfield_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `ihc_processing_fxn`

        if_processing_fxn : ImageProcesser
            ImageProcesser to pre-process immunofluorescent images to make them look as similar as possible.
            Should return a single channel uint8 image. If None, then {DEFAULT_FLOURESCENCE_CLASS.__name__}
            will be used for `img_type` = {slide_tools.IF_NAME}. {DEFAULT_FLOURESCENCE_CLASS.__name__} is
            located in the preprocessing module.

        if_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `if_processing_fxn`

        """

        pathlib.Path(self.processed_dir).mkdir(exist_ok=True, parents=True)
        if self.norm_method is not None:
            if self.norm_method == "histo_match":
                ref_histogram = np.zeros(256, dtype=np.int)
            else:
                all_v = [None]*self.size

        for i, slide_obj in enumerate(tqdm.tqdm(self.slide_dict.values())):
            if slide_obj.img_type == slide_tools.IHC_NAME:
                processing_cls = brightfield_processing_cls
                processing_kwargs = brightfield_processing_kwargs
            else:
                processing_cls = if_processing_cls
                processing_kwargs = if_processing_kwargs


            levels_in_range = np.where(slide_obj.slide_dimensions_wh.max(axis=1) < self.max_processed_image_dim_px)[0]
            if len(levels_in_range) > 0:
                level = levels_in_range[0]
            else:
                level = len(slide_obj.slide_dimensions_wh) - 1
            processor = processing_cls(image=slide_obj.image, src_f=slide_obj.src_f, level=level, series=slide_obj.series)

            try:
                processed_img = processor.process_image(**processing_kwargs)
            except TypeError:
                # processor.process_image doesn't take kwargs
                processed_img = processor.process_image()

            processed_img = exposure.rescale_intensity(processed_img, out_range=(0, 255)).astype(np.uint8)
            scaling = np.min(self.max_processed_image_dim_px/np.array(processed_img.shape[0:2]))
            if scaling < 1:
                processed_img = warp_tools.rescale_img(processed_img, scaling)

            processed_f_out = os.path.join(self.processed_dir, slide_obj.name + ".png")
            slide_obj.processed_img_f = processed_f_out
            slide_obj.processed_img_shape_rc = processed_img.shape

            io.imsave(processed_f_out, processed_img)
            if self.norm_method is not None:
                if self.norm_method == "histo_match":
                    img_hist, _ = np.histogram(processed_img, bins=256)
                    ref_histogram += img_hist
                else:
                    all_v[i] = processed_img.reshape(-1)


        if self.norm_method is not None:
            if self.norm_method == "histo_match":
                target_stats = ref_histogram
            else:
                all_v = np.hstack(all_v)
                # target_stats = preprocessing.get_channel_stats(all_v, 5, 95)
                target_stats = all_v

            self.normalize_images(target_stats)

    def normalize_images(self, target):
        """Normalize intensity values in images

        Parameters
        ----------
        target : ndarray
            Target statistics used to normalize images

        """
        print("\n==== Normalizing images\n")
        for i, slide_obj in enumerate(tqdm.tqdm(self.slide_dict.values())):
            img = io.imread(slide_obj.processed_img_f, True)
            if self.norm_method == "histo_match":
                normed_img = preprocessing.match_histograms(img, target)
            elif self.norm_method == "img_stats":
                normed_img = preprocessing.norm_img_stats(img, target)
            normed_img = exposure.rescale_intensity(normed_img, out_range=(0, 255)).astype(np.uint8)

            slide_obj.processed_img_shape_rc = normed_img.shape
            io.imsave(slide_obj.processed_img_f, normed_img)

    def create_thumbnail(self, img, rescale=True):
        """Create thumbnail image to view results
        """
        scaling = np.min(self.thumbnail_size/np.array(img.shape[:2]))
        if scaling < 1:
            thumbnail = warp_tools.rescale_img(img, scaling)
        else:
            thumbnail = img
        if rescale:
            thumbnail = exposure.rescale_intensity(thumbnail, out_range=(0, 255)).astype(np.uint8)

        return thumbnail

    def draw_overlap_img(self, img_list):
        """Create image showing the overlap of registered images
        """

        composite_img = np.dstack(img_list)
        cmap = viz.jzazbz_cmap()
        # cmap = viz.cam16ucs_cmap()

        channel_colors = viz.get_n_colors(cmap, composite_img.shape[2])
        overlap_img = viz.color_multichannel(composite_img, channel_colors, rescale_channels=True)
        overlap_img = exposure.equalize_adapthist(overlap_img)
        overlap_img = exposure.rescale_intensity(overlap_img, out_range=(0, 255)).astype(np.uint8)

        return overlap_img

    def rigid_register(self):
        """Rigidly register slides

        Also saves thumbnails of rigidly registered images.

        Returns
        -------
        rigid_registrar : SerialRigidRegistrar
            SerialRigidRegistrar object that performed the rigid registration.

        """

        rigid_registrar = serial_rigid.register_images(self.processed_dir,
                                                       **self.rigid_reg_kwargs)

        self.end_rigid_time = time()

        self.rigid_registrar = rigid_registrar
        if rigid_registrar is False:
            msg = "Rigid registration failed"
            valtils.print_warning(msg)

            return False

        # Draw and save overlap image #
        overlap_min_r = rigid_registrar.overlap_mask_bbox_xywh[1]
        overlap_min_c = rigid_registrar.overlap_mask_bbox_xywh[0]
        overlap_max_r = rigid_registrar.overlap_mask_bbox_xywh[1] + rigid_registrar.overlap_mask_bbox_xywh[3]
        overlap_max_c = rigid_registrar.overlap_mask_bbox_xywh[0] + rigid_registrar.overlap_mask_bbox_xywh[2]

        rigid_img_list = [img_obj.registered_img for img_obj in rigid_registrar.img_obj_list]
        rigid_overlap_img = self.draw_overlap_img(rigid_img_list)[overlap_min_r:overlap_max_r, overlap_min_c:overlap_max_c]
        self.rigid_overlap_img = self.create_thumbnail(rigid_overlap_img, self.thumbnail_size)


        pathlib.Path(self.overlap_dir).mkdir(exist_ok=True, parents=True)
        rigid_overlap_img_fout = os.path.join(self.overlap_dir, self.name + "_rigid_overlap.png")
        io.imsave(rigid_overlap_img_fout, self.rigid_overlap_img)

        # Create original overlap image #
        original_overlap_img = self.create_original_composite_img(rigid_registrar)
        self.original_overlap_img = self.create_thumbnail(original_overlap_img, self.thumbnail_size)

        original_overlap_img_fout = os.path.join(self.overlap_dir, self.name + "_original_overlap.png")
        io.imsave(original_overlap_img_fout, self.original_overlap_img)


        pathlib.Path(self.reg_dst_dir).mkdir(exist_ok=  True, parents=  True)
        # Update attributes in slide_obj #
        n_digits = len(str(rigid_registrar.size))
        prev_img_obj = None
        for slide_reg_obj in rigid_registrar.img_obj_list:
            slide_obj = self.slide_dict[slide_reg_obj.name]
            slide_obj.M = slide_reg_obj.M
            slide_obj.stack_idx = slide_reg_obj.stack_idx
            slide_obj.reg_img_shape_rc = slide_reg_obj.registered_img.shape
            slide_obj.rigid_reg_img_f = os.path.join(self.reg_dst_dir,
                                                     str.zfill(str(slide_obj.stack_idx), n_digits) + "_" + slide_obj.name + ".png")
            slide_obj.overlap_mask_bbox_xywh = rigid_registrar.overlap_mask_bbox_xywh
            slide_obj.overlap_mask = rigid_registrar.overlap_mask
            if prev_img_obj is not None:
                match_dict = slide_reg_obj.match_dict[prev_img_obj]
                slide_obj.xy_matched_to_prev = match_dict.matched_kp1_xy
                slide_obj.xy_in_prev = match_dict.matched_kp2_xy

                # Get points in overlap box #
                prev_kp_warped_for_bbox_test = warp_tools.warp_xy(slide_obj.xy_in_prev, M=slide_obj.M)
                _, prev_kp_in_bbox_idx = \
                    warp_tools.get_pts_in_bbox(prev_kp_warped_for_bbox_test, rigid_registrar.overlap_mask_bbox_xywh)

                current_kp_warped_for_bbox_test = \
                    warp_tools.warp_xy(slide_obj.xy_matched_to_prev, M=slide_obj.M)

                _, current_kp_in_bbox_idx = \
                    warp_tools.get_pts_in_bbox(current_kp_warped_for_bbox_test, rigid_registrar.overlap_mask_bbox_xywh)

                matched_kp_in_bbox = np.intersect1d(prev_kp_in_bbox_idx, current_kp_in_bbox_idx)
                slide_obj.xy_matched_to_prev_in_bbox =  slide_obj.xy_matched_to_prev[matched_kp_in_bbox] # Found using processed img
                slide_obj.xy_in_prev_in_bbox = slide_obj.xy_in_prev[matched_kp_in_bbox] # Found using processed img. Image at idx=i-1

            prev_img_obj = slide_reg_obj


        # Overwrite black and white processed images #
        for slide_name, slide_obj in self.slide_dict.items():
            slide_reg_obj = rigid_registrar.img_obj_dict[slide_name]
            if not slide_obj.is_rgb:
                img_to_warp = slide_reg_obj.image
            else:
                img_to_warp = slide_obj.image

            warped_img = slide_obj.warp_img(img_to_warp, non_rigid=False)
            if warped_img.ndim == 2:
                warped_img[slide_obj.overlap_mask == 0] = 0
            else:
                warped_img[slide_obj.overlap_mask == 0] = [0] * warped_img.shape[2]

            warped_img = warped_img[overlap_min_r:overlap_max_r, overlap_min_c:overlap_max_c]
            warped_img = self.create_thumbnail(warped_img, self.thumbnail_size)
            io.imsave(slide_obj.rigid_reg_img_f, warped_img.astype(np.uint8))

            # Replace processed image with a thumbnail #
            io.imsave(slide_obj.processed_img_f, self.create_thumbnail(slide_reg_obj.image, self.thumbnail_size))

        return rigid_registrar

    def non_rigid_register(self, rigid_registrar):
        """Non-rigidly register slides

        Non-rigidly register slides after performing rigid registration.
        Also saves thumbnails of non-rigidly registered images and deformation
        fields.

        Parameters
        ----------
        rigid_registrar : SerialRigidRegistrar
            SerialRigidRegistrar object that performed the rigid registration.

        Returns
        -------
        non_rigid_registrar : SerialNonRigidRegistrar
            SerialNonRigidRegistrar object that performed serial
            non-rigid registration.

        """

        self.non_rigid_reg_kwargs["mask"] = rigid_registrar.overlap_mask
        non_rigid_registrar = serial_non_rigid.register_images(src=rigid_registrar,
                                                               **self.non_rigid_reg_kwargs)
        self.end_non_rigid_time = time()

        for d in  [self.non_rigid_dst_dir, self.deformation_field_dir]:
            pathlib.Path(d).mkdir(exist_ok=True, parents=True)
        self.non_rigid_registrar = non_rigid_registrar

        # Draw overlap image #
        overlap_min_r = rigid_registrar.overlap_mask_bbox_xywh[1]
        overlap_min_c = rigid_registrar.overlap_mask_bbox_xywh[0]
        overlap_max_r = rigid_registrar.overlap_mask_bbox_xywh[1] + rigid_registrar.overlap_mask_bbox_xywh[3]
        overlap_max_c = rigid_registrar.overlap_mask_bbox_xywh[0] + rigid_registrar.overlap_mask_bbox_xywh[2]
        non_rigid_img_list = [nr_img_obj.registered_img for nr_img_obj in non_rigid_registrar.non_rigid_obj_list]
        non_rigid_overlap_img = self.draw_overlap_img(non_rigid_img_list)[overlap_min_r:overlap_max_r, overlap_min_c:overlap_max_c]
        self.non_rigid_overlap_img = self.create_thumbnail(non_rigid_overlap_img, self.thumbnail_size)

        overlap_img_fout = os.path.join(self.overlap_dir, self.name + "_non_rigid_overlap.png")
        io.imsave(overlap_img_fout, self.non_rigid_overlap_img)

        n_digits = len(str(self.size))
        for slide_name, slide_obj in self.slide_dict.items():
            img_save_id = str.zfill(str(slide_obj.stack_idx), n_digits)
            slide_nr_reg_obj = non_rigid_registrar.non_rigid_obj_dict[slide_name]
            slide_obj.bk_dxdy = slide_nr_reg_obj.bk_dxdy
            slide_obj.fwd_dxdy = slide_nr_reg_obj.fwd_dxdy
            slide_obj.nr_rigid_reg_img_f = os.path.join(self.non_rigid_dst_dir, img_save_id + "_" + slide_obj.name + ".png")

            scaling = np.min(self.thumbnail_size/np.array(slide_obj.reg_img_shape_rc[:2]))
            thumbnail_bk_dxdy = self.create_thumbnail(np.dstack(slide_obj.bk_dxdy), rescale=False)

            thumbnail_bk_dxdy *= scaling
            thumbanil_deform_grid = viz.color_displacement_tri_grid(thumbnail_bk_dxdy[..., 0],
                                                                    thumbnail_bk_dxdy[..., 1], n_grid_pts=20)

            deform_img_f = os.path.join(self.deformation_field_dir, img_save_id + "_" + slide_obj.name + ".png")

            io.imsave(deform_img_f, thumbanil_deform_grid)


        for slide_name, slide_obj in self.slide_dict.items():
            slide_nr_reg_obj = non_rigid_registrar.non_rigid_obj_dict[slide_name]
            img_save_id = str.zfill(str(slide_obj.stack_idx), n_digits)
            if not slide_obj.is_rgb:
                img_to_warp = rigid_registrar.img_obj_dict[slide_name].image
            else:
                img_to_warp = slide_obj.image


            warped_img = slide_obj.warp_img(img_to_warp, non_rigid=True)
            if warped_img.ndim == 2:
                warped_img[slide_obj.overlap_mask == 0] = 0
            else:
                warped_img[slide_obj.overlap_mask == 0] = [0] * warped_img.shape[2]

            warped_img = warped_img[overlap_min_r:overlap_max_r, overlap_min_c:overlap_max_c]

            warped_img = self.create_thumbnail(warped_img, self.thumbnail_size)
            io.imsave(slide_obj.nr_rigid_reg_img_f, warped_img.astype(np.uint8))


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

        path_list = [None] * (self.size)
        all_og_d = [None] * (self.size)
        all_og_tre = [None] * (self.size)
        all_og_mi = [None] * (self.size)

        all_rigid_d = [None] * (self.size)
        all_rigid_tre = [None] * (self.size)
        all_rigid_mi = [None] * (self.size)

        all_nr_d = [None] * (self.size)
        all_nr_tre = [None] * (self.size)
        all_nr_mi = [None] * (self.size)

        all_n = [None] * (self.size)
        from_list = [None] * (self.size)
        to_list = [None] * (self.size)
        shape_list = [None] * (self.size)
        processed_img_shape_list = [None] * (self.size)
        unit_list = [None] * (self.size)
        resolution_list = [None] * (self.size)

        prev_img_obj = None
        prev_slide_obj = None
        prev_rigid_img = None
        prev_non_rigid_img = None
        slide_obj_list = [self.slide_dict[img_obj.name] for img_obj in self.rigid_registrar.img_obj_list]

        for i, slide_obj in enumerate(tqdm.tqdm(slide_obj_list)):
            slide_name = slide_obj.name

            img_obj = self.rigid_registrar.img_obj_dict[slide_name]
            shape_list[i] = tuple(slide_obj.slide_shape_rc)
            processed_img_shape_list[i] = tuple(slide_obj.processed_img_shape_rc)
            unit_list[i] = slide_obj.units
            resolution_list[i] = slide_obj.resolution
            from_list[i] = slide_name
            path_list[i] = slide_obj.src_f
            rigid_img = img_obj.registered_img

            if prev_img_obj is None:
                prev_img_obj = img_obj
                prev_slide_obj = slide_obj
                outshape = slide_obj.aligned_slide_shape_rc
                prev_rigid_img = rigid_img
                if slide_obj.bk_dxdy is not None:
                    prev_non_rigid_img = self.non_rigid_registrar.non_rigid_obj_dict[slide_name].registered_img
                continue

            to_list[i] = prev_slide_obj.name

            prev_kp_in_slide = prev_slide_obj.warp_xy(slide_obj.xy_in_prev_in_bbox,
                                                     M=prev_img_obj.T,
                                                     pt_level= prev_slide_obj.processed_img_shape_rc,
                                                     non_rigid=False)

            current_kp_in_slide = slide_obj.warp_xy(slide_obj.xy_matched_to_prev_in_bbox,
                                                    M=img_obj.T,
                                                    pt_level= slide_obj.processed_img_shape_rc,
                                                    non_rigid=False)

            og_d = np.sqrt(np.sum((prev_kp_in_slide - current_kp_in_slide)**2, axis=1))
            og_rtre = og_d/np.sqrt(np.sum(np.power(outshape, 2)))
            median_og_tre = np.median(og_rtre)
            og_d *= slide_obj.resolution
            median_d_og = np.median(og_d)

            og_mmi = self.measure_original_mmi(img_obj.image, prev_img_obj.image)
            all_og_d[i] = median_d_og
            all_og_tre[i] = median_og_tre
            all_og_mi[i] = og_mmi

            prev_warped_rigid = prev_slide_obj.warp_xy(slide_obj.xy_in_prev_in_bbox,
                                                       M=prev_slide_obj.M,
                                                       pt_level= prev_slide_obj.processed_img_shape_rc,
                                                       non_rigid=False)

            current_warped_rigid = slide_obj.warp_xy(slide_obj.xy_matched_to_prev_in_bbox,
                                                     M=slide_obj.M,
                                                     pt_level= slide_obj.processed_img_shape_rc,
                                                     non_rigid=False)

            rigid_d = np.sqrt(np.sum((prev_warped_rigid - current_warped_rigid)**2, axis=1))
            rtre = rigid_d/np.sqrt(np.sum(np.power(outshape, 2)))
            median_rigid_tre = np.median(rtre)
            rigid_d *= slide_obj.resolution

            median_d_rigid = np.median(rigid_d)
            rigid_mi =  warp_tools.mattes_mi(rigid_img, prev_rigid_img, mask=self.rigid_registrar.overlap_mask)

            all_rigid_d[i] = median_d_rigid
            all_n[i] = len(rigid_d)
            all_rigid_tre[i] = median_rigid_tre
            all_rigid_mi[i] = rigid_mi


            if slide_obj.bk_dxdy is not None:

                prev_warped_nr = prev_slide_obj.warp_xy(slide_obj.xy_in_prev_in_bbox,
                                                        M=prev_slide_obj.M,
                                                        pt_level= prev_slide_obj.processed_img_shape_rc,
                                                        non_rigid=True)

                current_warped_nr = slide_obj.warp_xy(slide_obj.xy_matched_to_prev_in_bbox,
                                                      M=slide_obj.M,
                                                      pt_level= slide_obj.processed_img_shape_rc,
                                                      non_rigid=True)

                nr_d =  np.sqrt(np.sum((prev_warped_nr - current_warped_nr)**2, axis=1))
                nrtre = nr_d/np.sqrt(np.sum(np.power(outshape, 2)))
                mean_nr_tre = np.median(nrtre)

                non_rigid_img = self.non_rigid_registrar.non_rigid_obj_dict[slide_name].registered_img
                non_rigid_mi =  warp_tools.mattes_mi(non_rigid_img, prev_non_rigid_img, mask=self.rigid_registrar.overlap_mask)

                nr_d *= slide_obj.resolution
                median_d_nr = np.median(nr_d)
                all_nr_d[i] = median_d_nr
                all_nr_tre[i] = mean_nr_tre
                all_nr_mi[i] = non_rigid_mi

                prev_non_rigid_img  = non_rigid_img


            prev_img_obj = img_obj
            prev_slide_obj = slide_obj
            prev_rigid_img  = rigid_img


        mean_og_d = np.average(all_og_d[1:], weights=all_n[1:])
        median_og_tre = np.average(all_og_tre[1:], weights=all_n[1:])

        mean_rigid_d = np.average(all_rigid_d[1:], weights=all_n[1:])
        median_rigid_tre = np.average(all_rigid_tre[1:], weights=all_n[1:])

        rigid_min = (self.end_rigid_time - self.start_time)/60

        self.summary_df = pd.DataFrame({
            "filename": path_list,
            "from":from_list,
            "to": to_list,
            "original_D": all_og_d,
            "original_TRE": all_og_tre,
            "original_mattesMI": all_og_mi,
            "rigid_D": all_rigid_d,
            "rigid_TRE": all_rigid_tre,
            "rigid_mattesMI": all_rigid_mi,
            "non_rigid_D": all_nr_d,
            "non_rigid_TRE": all_rigid_tre,
            "non_rigid_mattesMI": all_nr_mi,
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

        if self.non_rigid_registrar is not None:
            mean_nr_d = np.average(all_nr_d[1:], weights=all_n[1:])
            mean_nr_tre = np.average(all_nr_tre[1:], weights=all_n[1:])
            non_rigid_min = (self.end_non_rigid_time - self.start_time)/60

            self.summary_df["mean_non_rigid_D"] = [mean_nr_d]*self.size
            self.summary_df["non_rigid_time_minutes"] = [non_rigid_min]*self.size

        return self.summary_df

    def register(self, brightfield_processing_cls=DEFAULT_BRIGHTFIELD_CLASS,
                 brightfield_processing_kwargs=DEFAULT_BRIGHTFIELD_PROCESSING_ARGS,
                 if_processing_cls=DEFAULT_FLOURESCENCE_CLASS,
                 if_processing_kwargs=DEFAULT_FLOURESCENCE_PROCESSING_ARGS,
                 reader_cls=None):

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
        brightfield_processing_fxn : callable
            Function to pre-process brightfield images to make them look as similar as possible.
            Should return a single channel uint8 image.

        brightfield_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `ihc_processing_fxn`

        if_processing_fxn : callable
            Function to pre-process immunofluorescent images to make them look as similar as possible.
            Should return a single channel uint8 image.

        if_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `if_processing_fxn`

        reader_cls : SlideReader, optional
            Uninstantiated SlideReader class that will convert
            the slide to an image, and also collect metadata. If None (the default),
            the appropriate SlideReader will be found by `slide_io.get_slide_reader`.
            This option is provided in case the slides cannot be opened by a current
            SlideReader class. In this case, the user should create a subclass of
            SlideReader. See slide_io.SlideReader for details.

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
            self.convert_imgs(series=self.series, reader_cls=reader_cls)

            print("\n==== Processing images\n")
            self.brightfield_procsseing_fxn_str = brightfield_processing_cls.__name__
            self.if_processing_fxn_str = if_processing_cls.__name__
            self.process_imgs(brightfield_processing_cls, brightfield_processing_kwargs,
                            if_processing_cls, if_processing_kwargs)

            print("\n==== Rigid registraration\n")
            rigid_registrar = self.rigid_register()
            if rigid_registrar is False:
                return None, None, None

            if self.non_rigid_registrar_cls is not None:
                print("\n==== Non-rigid registraration\n")
                non_rigid_registrar = self.non_rigid_register(rigid_registrar)
            else:
                non_rigid_registrar = None

            print("\n==== Measuring error\n")
            aligned_slide_shape_rc = self.get_aligned_slide_shape()
            self.aligned_slide_shape_rc = aligned_slide_shape_rc
            for slide_obj in self.slide_dict.values():
                slide_obj.aligned_slide_shape_rc = aligned_slide_shape_rc

            error_df = self.measure_error()
            self.cleanup()

            pathlib.Path(self.data_dir).mkdir(exist_ok=True,  parents=True)
            f_out = os.path.join(self.data_dir, self.name + "_registrar.pickle")
            pickle.dump(self, open(f_out, 'wb'))

            data_f_out = os.path.join(self.data_dir, self.name + "_summary.csv")
            error_df.to_csv(data_f_out, index=False)
        except Exception as e:
            valtils.print_warning(e)
            print(traceback.format_exc())
            kill_jvm()
            return None, None, None


        return rigid_registrar, non_rigid_registrar, error_df

    def cleanup(self):
        """Remove objects that can't be pickled
        """
        self.rigid_reg_kwargs["feature_detector"] = None
        self.rigid_reg_kwargs["affine_optimizer"] = None
        self.non_rigid_registrar_cls = None
        self.rigid_registrar = None
        self.non_rigid_registrar = None

    def get_aligned_slide_shape(self, level=0):
        """Determine the shape of aligned slide at the spefified level
        """

        if level == 0:
            if self.aligned_slide_shape_rc is not None:
                return self.aligned_slide_shape_rc

        aligned_slide_s = 0
        for slide_obj in self.slide_dict.values():
            if np.issubdtype(type(level), np.integer):
                slide_shape_rc = slide_obj.slide_dimensions_wh[level][::-1]
            else:
                slide_shape_rc = np.array(level)

            sy,  sx = slide_shape_rc/np.array(slide_obj.processed_img_shape_rc)
            M = warp_tools.scale_M(slide_obj.M, sx, sy)
            slide_corners_rc = warp_tools.get_corners_of_image(slide_shape_rc)
            warped_corners_xy = warp_tools.warp_xy(slide_corners_rc[:, ::-1], M)

            slide_w = int(np.ceil(np.max(warped_corners_xy[:, 0]) - np.min(warped_corners_xy[:, 0])))
            slide_h = int(np.ceil(np.max(warped_corners_xy[:, 1]) - np.min(warped_corners_xy[:, 1])))

            slide_sysx = np.array([slide_h, slide_w])/np.array(slide_obj.reg_img_shape_rc)

            aligned_slide_s = max(aligned_slide_s, max(slide_sysx))

        aligned_out_shape_rc = np.ceil(np.array(slide_obj.reg_img_shape_rc)*aligned_slide_s).astype(int)

        return aligned_out_shape_rc

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
            Path to the slide

        Returns
        -------
        slide_obj : Slide
            Slide associated with src_f

        """

        slide_name = valtils.get_name(src_f)
        slide_obj =  self.slide_dict[slide_name]

        return slide_obj

    def warp_and_save_slides(self, dst_dir, level = 0, non_rigid=True,
                             crop_to_overlap=True,
                             perceputally_uniform_channel_colors=False,
                             bg_color=None, interp_method="bicubic",
                             tile_wh=None, compression="lzw"):

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

        crop_to_overlap : bool, optional
            Whether or not to crop the warped slide so that it fits within
            Slide.overlap_mask_bbox_xywh. Default is True.

        bg_color : ndarray, str, optional
            Color used to fill in black background. If "auto",
            the color will be the same as the most luminescent regions of the slide.
            Alternatively, the RGB values can be provided, but should be between 0-255.
            Default is None, which means the background will not be colored.

        interp_method : str
            Interpolation method used when warping slide. Default is "bicubic"

        tile_wh : int, optional
            Tile width and height used to save image

        compression : str, optional
            Compression method used to save ome.tiff . Default is lzw, but can also
            be jpeg or jp2k. See pyips for more details.

        """
        pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)

        for slide_obj in tqdm.tqdm(self.slide_dict.values()):
            dst_f = os.path.join(dst_dir, slide_obj.name + ".ome.tiff")
            slide_obj.warp_and_save_slide(dst_f=dst_f, level = level,
                                          non_rigid=non_rigid,
                                          crop_to_overlap=crop_to_overlap,
                                          perceputally_uniform_channel_colors=perceputally_uniform_channel_colors,
                                          bg_color=bg_color, interp_method=interp_method,
                                          tile_wh=tile_wh, compression=compression)

    def warp_and_merge_slides(self, dst_f=None, level=0, non_rigid=True,
                              crop_to_overlap=True, channel_name_dict=None,
                              src_f_list=None, perceputally_uniform_channel_colors=False,
                              drop_duplicates=True, tile_wh=None,
                              interp_method="bicubic", compression="lzw"):

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

        crop_to_overlap : bool, optional
            Whether or not to crop the warped slide so that it fits within
            Slide.overlap_mask_bbox_xywh. Default is True.

        channel_name_dict : dict of lists, optional.
            key =  slide file name, value = list of channel names for that slide. If None,
            the the channel names found in each slide will be used.

        src_f_list : list of str, optionaal
            List of paths to slide to be warped. If None (the default), Valis.original_img_list
            will be used. Otherwise, the paths to which `src_f_list` points to should
            be an alternative copy of the slides, such as ones that have undergone
            processing (e.g. stain segmentation), had a mask applied, etc...

        perceputally_uniform_channel_colors : bool, optional
            Whether or not to add perceptually uniform channel colors.

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

        if src_f_list is None:
            src_f_list = self.original_img_list

        all_channel_names = []
        merged_slide = None

        for f in src_f_list:
            slide_name = valtils.get_name(os.path.split(f)[1])
            slide_obj = self.slide_dict[slide_name]

            warped_slide = slide_obj.warp_slide(level, non_rigid=non_rigid,
                                                crop_to_overlap=crop_to_overlap,
                                                interp_method=interp_method)

            keep_idx = list(range(warped_slide.bands))
            if channel_name_dict is not None:
                slide_channel_names = channel_name_dict_by_name[slide_obj.name]

                if drop_duplicates:
                    keep_idx = [idx for idx  in range(len(slide_channel_names)) if
                                slide_channel_names[idx] not in all_channel_names]

            else:
                slide_channel_names = slide_obj.reader.metadata.channel_names
                slide_channel_names = [c + " (" + slide_name + ")" for c in  slide_channel_names]

            if drop_duplicates and warped_slide.bands != len(keep_idx):
                keep_channels = [warped_slide[c] for c in keep_idx]
                slide_channel_names = [slide_channel_names[idx] for idx in keep_idx]
                if len(keep_channels) == 1:
                    warped_slide = keep_channels[0]
                else:
                    warped_slide = keep_channels[0].bandjoin(keep_channels[1:])
            print(f"merging {', '.join(slide_channel_names)}")

            if merged_slide is None:
                merged_slide = warped_slide
            else:
                merged_slide = merged_slide.bandjoin(warped_slide)

            all_channel_names.extend(slide_channel_names)


        px_phys_size = slide_obj.reader.scale_physical_size(level)
        bf_dtype = slide_io.vips2bf_dtype(merged_slide.format)
        out_xyczt = slide_io.get_shape_xyzct((merged_slide.width, merged_slide.height), merged_slide.bands)
        ome_xml_obj = slide_io.create_ome_xml(out_xyczt, bf_dtype, is_rgb=False,
                                              pixel_physical_size_xyu=px_phys_size,
                                              channel_names=all_channel_names,
                                              perceputally_uniform_channel_colors=perceputally_uniform_channel_colors)
        ome_xml = ome_xml_obj.to_xml()

        if dst_f is not None:
            dst_dir = os.path.split(dst_f)[0]
            pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)
            if tile_wh is None:
                tile_wh = slide_obj.reader.metadata.optimal_tile_wh
                if level != 0:
                    down_sampling = np.mean(slide_obj.slide_dimensions_wh[level]/slide_obj.slide_dimensions_wh[0])
                    tile_wh = int(np.round(tile_wh*down_sampling))
                    tile_wh = tile_wh - (tile_wh % 16)  # Tile shape must be multiple of 16
                    if tile_wh < 16:
                        tile_wh = 16
                    if np.any(np.array(out_xyczt[0:2]) < tile_wh):
                        tile_wh = min(out_xyczt[0:2])

            slide_io.save_ome_tiff(merged_slide, dst_f=dst_f,
                                   ome_xml=ome_xml,tile_wh=tile_wh,
                                   compression=compression)

        return merged_slide, all_channel_names, ome_xml

