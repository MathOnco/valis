"""
Methods to work with slides, after being opened using slide_io

"""

import os
import pyvips
import numpy as np
import colour
from matplotlib import colormaps
import re
from PIL import Image
from collections import Counter
from . import warp_tools
from . import slide_io
from . import viz
from . import preprocessing
from . import valtils

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

# See slide_io.bf_to_numpy_dtype
BF_DTYPE_PIXEL_TYPE = {'uint8':1,
                       'int8': 0,
                       'uint16': 3,
                       'int16': 2,
                       'uint32': 5,
                       'int32': 4,
                       'float': 6,
                       'double': 7
                       }

CZI_FORMAT_NUMPY_DTYPE = {
    "gray8": "uint8",
    "gray16": "uint16",
    "gray32": "uint32",
    "gray32float": "float32",
    "bgr24": "uint8",
    "bgr48": "uint16",
    "invalid": "uint8",
}

CZI_FORMAT_TO_BF_FORMAT = {k:NUMPY_FORMAT_BF_DTYPE[v] for k,v in CZI_FORMAT_NUMPY_DTYPE.items()}

BF_FORMAT_NUMPY_DTYPE = {v:k for k, v in NUMPY_FORMAT_BF_DTYPE.items()}


def vips2numpy(vi):
    """
    https://github.com/libvips/pyvips/blob/master/examples/pil-numpy-pyvips.py

    """
    try:
        img = vi.numpy()
    except:
        img = np.ndarray(buffer=vi.write_to_memory(),
                        dtype=VIPS_FORMAT_NUMPY_DTYPE[vi.format],
                        shape=[vi.height, vi.width, vi.bands])
        if vi.bands == 1:
            img = img[..., 0]

    return img


def numpy2vips(a, pyvips_interpretation=None):
    """

    """
    try:
        vi = pyvips.Image.new_from_array(a)

    except Exception as e:
        if a.ndim > 2:
            height, width, bands = a.shape
        else:
            height, width = a.shape
            bands = 1

        linear = a.reshape(width * height * bands)
        if linear.dtype.byteorder == ">":
            # vips seems to expect the array to be little endian, but `a` is big endian
            linear = linear.byteswap(inplace=False)

        vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                        NUMPY_FORMAT_VIPS_DTYPE[a.dtype.name])

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

    f = os.path.split(src_f)[1].lower()
    if (re.search(".ome", f) is not None and re.search(".tif*", f) is not None):
        slide_format = ".ome.tiff"
    else:
        if re.search(".nii.gz", f):
            format_split = -2
        else:
            format_split = -1

        slide_format = "." + ".".join(f.split(".")[format_split:])

    return slide_format


def get_level_idx(dims_wh, max_dim):
    possible_levels = np.where(np.max(dims_wh, axis=1) <= max_dim)[0]
    if len(possible_levels):
        level = possible_levels[0]
    else:
        level = len(dims_wh) - 1

    level = max(0, level)

    return level


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

    kind = None
    if os.path.isdir(img_f):
        return kind

    f_extension = get_slide_extension(str(img_f))

    if f_extension.lower() == '.ds_store':
        return kind

    is_ome_tiff = slide_io.check_is_ome(str(img_f))
    is_czi = f_extension == ".czi"

    can_use_pil = False
    if not is_ome_tiff:
        try:
            with valtils.HiddenPrints():
                pil_image = Image.open(str(img_f))
            can_use_pil = True
        except:
            pass

    can_use_vips = slide_io.check_to_use_vips(str(img_f))
    if not is_ome_tiff and (can_use_pil or can_use_vips):
        return TYPE_IMG_NAME

    can_use_openslide = slide_io.check_to_use_openslide(str(img_f))
    if can_use_openslide or is_ome_tiff or is_czi:
        return TYPE_SLIDE_NAME

    # Finally, see if Bioformats can read slide.
    if slide_io.BF_READABLE_FORMATS is None:
        slide_io.init_jvm()
    can_use_bf = f_extension in slide_io.BF_READABLE_FORMATS
    if can_use_bf:
        return TYPE_SLIDE_NAME

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

        f_list = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if get_img_type(os.path.join(src_dir, f)) is not None and not f.startswith(".")]
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
               bbox_xywh=None, bg_color=None, reader=None):
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
    if reader is None:
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


def get_matplotlib_channel_colors(n_colors, name="gist_rainbow", min_lum=0.5, min_c=0.2):
    """Get channel colors using matplotlib colormaps

    Parameters
    ----------
    n_colors : int
        Number of colors needed.

    name : str
        Name of matplotlib colormap

    min_lum : float
        Minimum luminosity allowed

    min_c : float
        Minimum colorfulness allowed

    Returns
    --------
    channel_colors : ndarray
        RGB values for each of the `n_colors`

    """
    n = 200
    if n_colors > n:
        n = n_colors
    all_colors = colormaps[name](np.linspace(0, 1, n))[..., 0:3]

    # Only allow bright colors #
    jch = preprocessing.rgb2jch(all_colors)
    all_colors = all_colors[(jch[..., 0] >= min_lum) & (jch[..., 1] >= min_c)]
    channel_colors = viz.get_n_colors(all_colors, n_colors)
    channel_colors = (255*channel_colors).astype(np.uint8)

    return channel_colors


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
