"""
Methods to work with slides, after being opened using slide_io

"""

import os
import pyvips
import numpy as np
import colour
import re
import imghdr
from collections import Counter
from . import warp_tools
from . import slide_io
from .import viz

IHC_NAME = "brightfield"
IF_NAME = "fluorescence"
MULTI_MODAL_NAME = "multi"
TYPE_IMG_NAME = "img"
TYPE_SLIDE_NAME = "slide"
BG_AUTO_FILL_STR = "auto"

NUMPY_FORMAT_VIPS_DTYPE = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
    }


VIPS_FORMAT_NUMPY_DTYPE = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

NUMPY_FORMAT_BF_DTYPE = {'uint8': 'uint8',
                         'int8': 'int8',
                         'uint16': 'uint16',
                         'int16': 'int16',
                         'uint32': 'uint32',
                         'int32': 'int32',
                         'float32': 'float',
                         'float64': 'double'}

BF_FORMAT_NUMPY_DTYPE = {v:k for k, v in NUMPY_FORMAT_BF_DTYPE.items()}


def vips2numpy(vi):
    """
    https://github.com/libvips/pyvips/blob/master/examples/pil-numpy-pyvips.py

    """

    img = np.ndarray(buffer=vi.write_to_memory(),
                     dtype=VIPS_FORMAT_NUMPY_DTYPE[vi.format],
                     shape=[vi.height, vi.width, vi.bands])
    if vi.bands == 1:
        img = img[..., 0]

    return img


def numpy2vips(a, pyvips_interpretation=None):
    """

    """

    if a.ndim > 2:
        height, width, bands = a.shape
    else:
        height, width = a.shape
        bands = 1

    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      NUMPY_FORMAT_VIPS_DTYPE[str(a.dtype)])
    # maybe a try catch is better here, but could be slower performance-wise
    if pyvips_interpretation is not None:
        vi = vi.copy(interpretation=pyvips_interpretation)
    return vi


def get_slide_extension(src_f):
    """Get slide format

    Parameters
    ----------
    src_f : str
        Path to slide

    Returns
    -------
    slide_format : str
        Slide format.

    """

    f = os.path.split(src_f)[1]
    if re.search(".ome.tif", f):
        format_split = -2
    else:
        format_split = -1
    slide_format = "." + ".".join(f.split(".")[format_split:])

    return slide_format


def get_img_type(img_f):
    """Determine if file is a slide or an image

    Parameters
    ----------
    img_f : str
        Path to image

    Returns
    -------
    kind : str
        Type of file, either 'image', 'slide', or None if they type
        could not be determined

    """

    if os.path.isdir(img_f):
        return None

    f_extension = get_slide_extension(img_f)
    what_img = imghdr.what(img_f)
    if slide_io.BF_READABLE_FORMATS is None:
        slide_io.init_jvm()

    can_use_bf = f_extension in slide_io.BF_READABLE_FORMATS
    can_use_openslide = slide_io.check_to_use_openslide(img_f)
    is_tiff = f_extension == ".tiff" or f_extension == ".tif"
    can_use_skimage = ".".join(f_extension.split(".")[1:]) == what_img and not is_tiff

    kind = None
    if can_use_skimage:
        kind = TYPE_IMG_NAME
    elif can_use_bf or can_use_openslide:
        kind = TYPE_SLIDE_NAME

    return kind


def determine_if_staining_round(src_dir):
    """Determine if path contains an image split across different files

    Checks to see if files in the directory belong to a single image.
    An example is a folder of several .ndpi images, with a single .ndpis
    file. This method assumes that if there is a single file that has
    a different extension than the other images then the path contains
    a set of files (e.g. 3 .npdi images) that can be read using a
    single file (e.g. 1 .ndpis image).

    Parameters
    ----------
    src_dir : str
        Path to directory containing the images

    Returns
    -------
    multifile_img : bool
        Whether or not the path contains an image split across different files

    master_img_f : str
        Name of file that can be used to open all images in `src_dir`

    """

    if not os.path.isdir(src_dir):
        multifile_img = False
        master_img_f = None
    else:
        f_list = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if get_img_type(os.path.join(src_dir, f)) is not None]
        extensions = [get_slide_extension(f) for f in f_list]
        format_counts = Counter(extensions)
        format_count_values = list(format_counts.values())
        n_formats = len(format_count_values)
        if n_formats > 1 and min(format_count_values) == 1:
            multifile_img = True
            master_img_format = list(format_counts.keys())[np.argmin(format_count_values)]
            master_img_file_idx = extensions.index(master_img_format)
            master_img_f = f_list[master_img_file_idx]
        else:
            multifile_img = False
            master_img_f = None

    return multifile_img, master_img_f


def um_to_px(um, um_per_px):
    """Conver mircon to pixel
    """
    return um * 1/um_per_px


def warp_slide(src_f, transformation_src_shape_rc, transformation_dst_shape_rc,
               aligned_slide_shape_rc, M=None, dxdy=None,
               level=0, series=None, interp_method="bicubic",
               bbox_xywh=None, bg_color=None):
    """ Warp a slide

    Warp slide according to `M` and/or `non_rigid_dxdy`

    Parameters
    ----------
    src_f : str
        Path to slide

    transformation_src_shape_rc : (N, M)
        Shape of the image used to find the transformations

    transformation_dst_shape_rc : (int, int)
        Shape of image with shape `in_shape_rc`, after being warped,
        i.e. the shape of the registered image.

    aligned_slide_shape_rc : (int, int)
        Shape of the warped slide.

    scaled_out_shape_rc : optional, (int, int)
        Shape of scaled image (with shape out_shape_rc) after warping

    M : ndarray, optional
        3x3 Affine transformation matrix to perform rigid warp

    dxdy : ndarray, optional
        An array containing the x-axis (column) displacement,
        and y-axis (row) displacement applied after the rigid transformation

    level : int, optional
        Pyramid level

    series : int, optional
        Series number

    interp_method : str, optional

    bbox_xywh : tuple
        Bounding box to crop warped slide. Should be in refernce the
        warped slide

    Returns
    -------
    vips_warped : pyvips.Image
        A warped copy of the slide specified by `src_f`

    """
    reader_cls = slide_io.get_slide_reader(src_f, series=series)
    reader = reader_cls(src_f, series=series)
    if series is None:
        series = reader.series

    vips_slide = reader.slide2vips(level=level, series=series)
    if M is None and dxdy is None:
        return vips_slide

    vips_warped = warp_tools.warp_img(img=vips_slide, M=M, bk_dxdy=dxdy,
                                      transformation_dst_shape_rc=transformation_dst_shape_rc,
                                      out_shape_rc=aligned_slide_shape_rc,
                                      transformation_src_shape_rc=transformation_src_shape_rc,
                                      bbox_xywh=bbox_xywh,
                                      bg_color=bg_color,
                                      interp_method=interp_method)

    return vips_warped


def turbo_channel_colors(n_colors):
    """Channel colors using the Turbo colormap

    Gets channel colors from the Turbo colormap
    https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html
    These are not percepually uniform, but are better than jet.

    Parameters
    ----------
    n_colors : int
        Number of colors needed.

    Returns
    --------
    channel_colors : ndarray
        RGB values for each of the `n_colors`

    """

    turbo = viz.turbo_cmap()[40:-40]
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        cam16 = colour.convert(turbo, 'sRGB', "CAM16UCS")
        cam16[..., 0] *= 1.1
        brighter_turbo = colour.convert(cam16, "CAM16UCS", 'sRGB')

    brighter_turbo = np.clip(brighter_turbo, 0, 1)

    channel_colors = viz.get_n_colors(brighter_turbo, n_colors)
    channel_colors = (255*channel_colors).astype(np.uint8)

    return channel_colors


def perceptually_uniform_channel_colors(n_colors):
    """Channel colors using a perceptually uniform colormap

    Gets perceptually uniform channel colors using the
    JzAzBz colormap.

    See https://www.osapublishing.org/DirectPDFAccess/BA34298D-D6DF-42BA-A704279555676BA8_368272/oe-25-13-15131.pdf?da=1&id=368272&seq=0&mobile=no

    Parameters
    ----------
    n_colors : int
        Number of colors needed.

    Returns
    --------
    channel_colors : ndarray
        RGB values for each of the `n_colors`

    """
    cmap = viz.jzazbz_cmap()
    channel_colors = viz.get_n_colors(cmap, n_colors)
    channel_colors = (channel_colors*255).astype(np.uint8)

    return channel_colors
