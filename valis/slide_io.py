"""Methods and classes to read and write slides in the .ome.tiff format

"""

import os
from skimage import io, transform
import pyvips
import numpy as np
from PIL import Image
import pathlib
import re
import multiprocessing
from pqdm.threads import pqdm
from scipy import stats
from bs4 import BeautifulSoup
from statistics import mode
import time
import sys
import re
import itertools
import xml.etree.ElementTree as elementTree
import unicodedata
import ome_types
import jpype
from aicspylibczi import CziFile
from tqdm import tqdm
import scyjava
from difflib import get_close_matches
import traceback

from colorama import Fore
from . import valtils
from . import slide_tools
from . import warp_tools
from . import valtils


pyvips.cache_set_max(0)

CMAP_AUTO = "auto"
"""
str: Default argument to get channel colors.
"""

DEFAULT_COMPRESSION = pyvips.enums.ForeignTiffCompression.DEFLATE
"""
Default tiff compression method
"""

MAX_TILE_SIZE = 2**10
"""int: maximum tile used to read or write images"""

BF_RDR = "bioformats"
"""str: Name of Bioformats reader."""

VIPS_RDR = "libvips"
"""str: Name of pyvips reader"""

OPENSLIDE_RDR = "openslide"
"""str: Name of OpenSlide reader"""

IMG_RDR = "skimage"
"""str: Name of image reader"""

# PIXEL_UNIT = "px"
PIXEL_UNIT = "pixel"
"""str: Physical unit when the unit can't be found in the metadata"""

MICRON_UNIT = u'\u00B5m'
"""str: Phyiscal unit for micron/micrometers"""

ALL_OPENSLIDE_READABLE_FORMATS = [".svs", ".tif", ".vms", ".vmu", ".ndpi", ".scn", ".mrxs", ".tiff", ".svslide", ".bif"]
"""list: File extensions that OpenSlide can read"""


# VIPS_READABLE_FORMATS = [".jpg", ".jpeg", ".jpe", ".jif", ".jfif", ".jfi",
#                            ".tif", ".tiff", ".png", ".webp", ".heif", ".heifs",
#                            ".heic", ".heics", ".avci", ".avcs", ".avif", ".hif", ".avif",
#                            ".fits", ".fit", ".fts", ".exr", ".pdf", ".svg", ".hdr",
#                            ".pbm", ".pgm", ".ppm", ".pfm",
#                            ".csv", ".gif", ".img", ".vips",
#                            ".nii", ".nii.gz",
#                            ".dzi" ".xml", ".dcm", ".ome.tiff", ".ome.tif"]

# VIPS_READABLE_FORMATS = pyvips.get_suffixes()
VIPS_READABLE_FORMATS = [*pyvips.get_suffixes(), *ALL_OPENSLIDE_READABLE_FORMATS, ".ome.tiff", ".ome.tif"]
"""list: File extensions that libvips can read. See https://github.com/libvips/libvips
"""


VIPS_RGB_FORMATS = [x.lower() for x in dir(pyvips.enums.Interpretation) if re.search("rgb", x.lower()) is not None]
"""list: List of libvips rgb formats
"""

BF_READABLE_FORMATS = None
"""list: File extensions that Bioformats can read.
   Filled in after initializing JVM"""

OPENSLIDE_ONLY = None
"""list: File extensions that OpenSlide can read but Bioformats can't.
   Filled in after initializingJVM"""

FormatTools = None
"""Bioformats FormatTools.
   Created after initializing JVM"""

BF_UNIT = None
"""Bioformats UNITS.
   Created after initializing JVM."""

BF_MICROMETER = None
"""Bioformats Unit mircometer object.
   Created after initializing JVM."""

ome = None
"""Bioformats ome from bioforamts_jar.
   Created after initializing JVM."""

loci = None
"""Bioformats loci from bioforamts_jar.
   Created after initializing JVM."""

OME_TYPES_PARSER = "lxml"

"""
NOTE: Commented out block is how to use boformats with javabrdige.
However, on conda, javabridge isn't available for python 3.9.
If using, remember to put bftools/bioformats_package.jar in the
source directory.

Keeping the code just in case need to use javabridge again.
"""
# Bioformats + Javabridge #
#---------------------------------------#
#
# try:
#     bf_jar = os.path.join(pathlib.Path(__file__).parent, "bftools/bioformats_package.jar")
# except Exception:
#     # Running interactively
#     bf_jar = os.path.join(os.getcwd(), "bftools/bioformats_package.jar")
#
#
# def init_jvm_javabridge():
#     """Initialize JVM for BioFormats
#     """
#
#     if javabridge.get_env() is None:
#
#         all_jars = javabridge.JARS + [bf_jar]
#         javabridge.start_vm(class_path=all_jars, max_heap_size="10G", run_headless=True)
#
#         myloglevel = "ERROR"
#         rootLoggerName = javabridge.get_static_field("org/slf4j/Logger", "ROOT_LOGGER_NAME", "Ljava/lang/String;")
#         rootLogger = javabridge.static_call("org/slf4j/LoggerFactory", "getLogger",
#                                             "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
#         logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level", myloglevel, "Lch/qos/logback/classic/Level;")
#         javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)
#
#         msg = "JVM has been initialize. Be sure to call valis.kill_jvm() or slide_io.kill_jvm() at the end of your script"
#         valtils.print_warning(msg, warning_type=None, rgb=valtils.Fore.GREEN)
#
#         # Fill in global variables that can only be created after initializing the JVM
#
#         global FormatTools
#         global BF_UNIT
#         global BF_MICROMETER
#         global BF_READABLE_FORMATS
#         global OPENSLIDE_ONLY
#
#         FormatTools = javabridge.JClassWrapper("loci.formats.FormatTools")
#         BF_UNIT = javabridge.JClassWrapper("ome.units.UNITS")
#         BF_MICROMETER = BF_UNIT.MICROMETER
#         BF_READABLE_FORMATS = get_bf_readable_formats_javabridge()
#         OPENSLIDE_ONLY = list(set(ALL_OPENSLIDE_READABLE_FORMATS).difference(set(BF_READABLE_FORMATS)))
#
#
# def kill_jvm_javabridge():
#     """Kill JVM for BioFormats
#     """
#     javabridge.kill_vm()
#     msg = "JVM has been killed. If this was due to an error, then a new Python session will need to be started"
#     valtils.print_warning(msg, warning_type=None, rgb=valtils.Fore.GREEN)
#
#
# def get_bf_readable_formats_javabridge():
#     """Get extensions of formats that BioFormats can read
#     """
#     if javabridge.get_env() is None:
#         init_jvm_javabridge()
#
#     env = javabridge.get_env()
#     base_reader = javabridge.make_instance('loci/formats/ImageReader', '()V')
#     readers = javabridge.jutil.call(base_reader, 'getReaders',
#                                     '()[Lloci/formats/IFormatReader;')
#     all_readers = env.get_object_array_elements(readers)
#     readable_formats = []
#     f_append = readable_formats.append
#     for format_reader in all_readers:
#         j_suffixes = javabridge.get_env().get_object_array_elements(
#             javabridge.jutil.call(
#                 format_reader, 'getSuffixes',
#                 '()[Ljava/lang/String;'))
#
#         for js in j_suffixes:
#             suffix = javabridge.to_string(js)
#             if len(suffix) > 0:
#                 f_append("." + suffix)
#
#     javabridge.jutil.call(base_reader, 'close', '()V')
#
#     return readable_formats
#---------------------------------------#

# Bioformats/scyjava + Jpype #
#--------------------#


def init_jvm(jar=None, mem_gb=10):
    """Initialize JVM for BioFormats

    Parameters
    ----------
    mem_gb : int
        Amount of memory, in GB, for JVM
    """
    import jpype
    if not jpype.isJVMStarted():
        global FormatTools
        global BF_MICROMETER
        global OPENSLIDE_ONLY
        global BF_READABLE_FORMATS
        global ome
        global loci

        if jar is None:

            # Check if jar is bundled with source code, like in a Docker image
            # Can use instead of using maven to download, which requires an unblocked connection
            parent_dir = pathlib.Path(__file__).parent.resolve()
            local_bf_jar = os.path.join(parent_dir, "bioformats_package.jar")
            if os.path.exists(local_bf_jar):
                jar = local_bf_jar

        if jar is not None:
            jpype.addClassPath(jar)
            jpype.startJVM(f"-Djava.awt.headless=true -Xmx{mem_gb}G", classpath=jar)

        else:
            scyjava.config.endpoints.extend(['ome:formats-gpl', 'ome:jxrlib-all'])
            scyjava.start_jvm([f"-Xmx{mem_gb}G"])

        loci = jpype.JPackage("loci")
        ome = jpype.JPackage("ome")
        loci.common.DebugTools.setRootLevel("ERROR")

        FormatTools = loci.formats.FormatTools
        BF_MICROMETER = ome.units.UNITS.MICROMETER
        BF_READABLE_FORMATS = get_bf_readable_formats()
        OPENSLIDE_ONLY = list(set(ALL_OPENSLIDE_READABLE_FORMATS).difference(set(BF_READABLE_FORMATS)))

        msg = (f"JVM has been initialized. "
               f"Be sure to call registration.kill_jvm() "
               f"or slide_io.kill_jvm() at the end of your script.")
        valtils.print_warning(msg, warning_type=None, rgb=valtils.Fore.GREEN)


def kill_jvm():
    """Kill JVM for BioFormats
    """
    try:
        scyjava.shutdown_jvm()
        msg = "JVM has been killed. If this was due to an error, then a new Python session will need to be started"
        valtils.print_warning(msg, warning_type=None, rgb=valtils.Fore.GREEN)

    except NameError:
        pass


def get_bioformats_version():
    v = loci.formats.FormatTools.VERSION

    return v


def get_bf_readable_formats():
    """Get extensions of formats that BioFormats can read

    Returns
    -------
    readable_formats : list of str
        List of formats that can be read by Bioformats

    """

    if not jpype.isJVMStarted():
        init_jvm()

    baseReader = loci.formats.ImageReader()
    readers = baseReader.getReaders()
    read_range = range(1, readers.length)
    readable_formats = ["." + str(f) for l in [list(readers[i].getSuffixes()) for i in read_range] for f in l if len(f) > 0]
    baseReader.close()

    return readable_formats


def bf_to_numpy_dtype(bf_pixel_type, little_endian):
    """Get numpy equivalent of the bioformats pixel type

    Adapted from the python-bioformats package

    Parameters
    ----------
    bf_pixel_type : int
        Integer indicating the Bioformats pixel type

    little_endian : bool
        Whether or not the image is little endian

    Returns
    -------
    dtype : numpy.dtype
        Numpy dtype

    scale : int
        Maximum value of `dtype`

    """

    if bf_pixel_type == FormatTools.INT8:
        # FormatTools.INT8 = 0
        dtype = np.int8
        scale = 255

    elif bf_pixel_type == FormatTools.UINT8:
        # FormatTools.UINT8 = 1
        dtype = np.uint8
        scale = 255

    elif bf_pixel_type == FormatTools.UINT16:
        # FormatTools.UINT16 = 3
        dtype = '<u2' if little_endian else '>u2'
        scale = 65535

    elif bf_pixel_type == FormatTools.INT16:
        # FormatTools.INT16 = 2
        dtype = '<i2' if little_endian else '>i2'
        scale = 65535

    elif bf_pixel_type == FormatTools.UINT32:
        # FormatTools.UINT32 = 5
        dtype = '<u4' if little_endian else '>u4'
        scale = 2**32

    elif bf_pixel_type == FormatTools.INT32:
        # FormatTools.INT32 = 4
        dtype = '<i4' if little_endian else '>i4'
        scale = 2**32-1

    elif bf_pixel_type == FormatTools.FLOAT:
        # FormatTools.FLOAT = 6
        dtype = '<f4' if little_endian else '>f4'
        scale = 1

    elif bf_pixel_type == FormatTools.DOUBLE:
        # FormatTools.DOUBLE = 7
        dtype = '<f8' if little_endian else '>f8'
        scale = 1

    return dtype, scale


def vips2bf_dtype(vips_format):
    """Get bioformats equivalent of the pyvips pixel type

    Parameters
    ----------
    vips_format : str
        Format of the pyvips.Image

    Returns
    -------
    bf_dtype : str
        String format of Bioformats datatype

    """

    np_dtype = slide_tools.VIPS_FORMAT_NUMPY_DTYPE[vips_format]
    bf_dtype = slide_tools.NUMPY_FORMAT_BF_DTYPE[str(np_dtype().dtype)]

    return bf_dtype


def bf2vips_dtype(bf_dtype):
    """Get bioformats equivalent of the pyvips pixel type

    Parameters
    ----------
    bf_dtype : str
        String format of Bioformats datatype

    Returns
    -------
    vips_format : str
        Format of the pyvips.Image

    """

    np_type = slide_tools.BF_FORMAT_NUMPY_DTYPE[bf_dtype]
    vips_format = slide_tools.NUMPY_FORMAT_VIPS_DTYPE[np_type]

    return vips_format


def check_czi_jpegxr(src_f):
    f_extension = slide_tools.get_slide_extension(src_f)
    if f_extension != ".czi":
        return False

    czi = CziFile(src_f)
    is_czi_jpgxr = False
    comp_tree = czi.meta.findall(".//OriginalCompressionMethod")
    if len(comp_tree) > 0:
        is_czi_jpgxr = comp_tree[0].text.lower() == "jpgxr"

    return is_czi_jpgxr


def check_to_use_openslide(src_f):
    """Determine if OpenSlide can be used to read the slide

    Parameters
    ----------
    src_f : str
        Path to slide

    Returns
    -------
    use_openslide : bool
        Whether or not OpenSlide can be used to read the slide.
        This can happen if the file format is not readable by
        OpenSlide, or if pyvips wasn't installed with OpenSlide
        support.

    """

    use_openslide = False
    img_format = slide_tools.get_slide_extension(src_f)
    if img_format in ALL_OPENSLIDE_READABLE_FORMATS:
        try:
            vips_img = pyvips.Image.new_from_file(src_f)
            vips_fields = vips_img.get_fields()
            if "openslide.level-count" in vips_fields:
                use_openslide = True
        except pyvips.error.Error as e:
            traceback_msg = traceback.format_exc()
            valtils.print_warning(e, traceback_msg=traceback_msg)

    return use_openslide


def get_ome_obj(x):
    """Get ome_types.model.ome.OME object

    Paramters
    ---------
    x: str
        Either OME-XML or path to ome.tiff

    Returns
    -------
    ome_obj : ome_types.model.ome.OME

    """

    x = str(x)
    is_file = os.path.exists(x)
    ome_obj = None
    try:
        # Try to use ome-types
        if is_file:
            ome_fxn = ome_types.from_tiff
        else:
            ome_fxn = ome_types.from_xml

        ome_obj = ome_fxn(x)

        if len(ome_obj.images) == 0:
            ome_obj = ome_fxn(x, parser=OME_TYPES_PARSER)

    except Exception as e:
        # Sometimes the image description found by `ome_types.from_tiff`
        # does not contain the ome-xml. Seems to be the case for ImageJ exports, at least
        if is_file:
            try:
                bf_rdr, bf_meta = get_bioformats_reader_and_meta(x)
                meta_xml = bf_meta.dumpXML()
                meta_xml = str(meta_xml)
                ome_obj = ome_types.from_xml(meta_xml)
                if len(ome_obj.images) == 0:
                    ome_obj = ome_types.from_xml(meta_xml, parser=OME_TYPES_PARSER)

            except Exception as e:
                if ome_fxn == ome_types.from_tiff:
                    valtils.print_warning(f"Could not get OME-XML for image {x}, due to the following error: {e}")
                else:
                    valtils.print_warning(f"Could not get OME-XML, due to the following error: {e}")

    return ome_obj


def check_is_ome(src_f):
    is_ome = re.search(".ome", src_f) is not None and re.search(".tif*", src_f) is not None
    if is_ome:
        # Verify that image is valid ome.tiff
        ome_obj = get_ome_obj(src_f)
        if ome_obj is None:
            is_ome = False

    return is_ome


def check_to_use_vips(src_f):

    f_extension = slide_tools.get_slide_extension(src_f)
    can_use_pyvips = f_extension.lower() in VIPS_READABLE_FORMATS

    return can_use_pyvips


def check_to_use_bioformats(src_f, series=None):
    """Check if bioformats can be used to read metadata and/or image

    """
    init_jvm()
    img_format = slide_tools.get_slide_extension(src_f)
    use_bf = img_format in BF_READABLE_FORMATS

    can_get_metadata = use_bf
    can_read_img = use_bf
    if use_bf:

        err_msg = f"Error using Bioformats to read {os.path.split(src_f)[-1]}. Will try to use a different reader"
        try:
            # Try to get metadata
            bf_reader = BioFormatsSlideReader(src_f, series=series)
        except Exception as e:
            valtils.print_warning(err_msg)
            can_get_metadata = False
            can_read_img = False

    if can_get_metadata:
        try:
            # Can get metadata, try reading small slice
            test_read_level = len(bf_reader.metadata.slide_dimensions) - 1
            bf_reader.slide2vips(level=test_read_level, xywh=(0, 0, 5, 5))
        except Exception as e:
            valtils.print_warning(err_msg)
            can_read_img = False

    return can_get_metadata, can_read_img


def check_flattened_pyramid_tiff(src_f, check_with_bf=False):
    """Determine if a tiff is a flattened pyramid

    Determines if a slide is pyramid where each page/plane is a channel
    in the pyramid. An example would be one where the plane dimensions are
    something like
    [(600, 600), (600, 600), (600, 600), (300, 300), (300, 300), (300, 300)]
    for a 3 channel image with  2  pyramid levels. It seems that bioformats
    does not recognize these as pyramid images.

    Parameters
    ----------
    src_f : str
        Path to slide

    Returns
    -------
    is_flattended_pyramid : bool
        Whether  or not the slide is a flattened pyramid

    can_use_bf : bool
        Whether or not Bioformats will read the slide in the same way

    slide_dimensions : ndarray
        Dimensions (width, height) for each level in the  pyramid

    levels_start_idx : ndarray
        The indices indicating which pages/planes start the next
        pyramid level

    n_channels : int
        Number of channels in the slide

    """

    vips_img = pyvips.Image.new_from_file(src_f)
    vips_fields = vips_img.get_fields()

    is_flattended_pyramid = False
    can_use_bf = False

    if 'n-pages' in vips_fields:
        n_pages = vips_img.get("n-pages")
        all_areas = []
        all_dims = []
        all_n_channels = []
        level_starts = []
        prev_area = None
        for i in range(n_pages):
            try:
                page = pyvips.Image.new_from_file(src_f, page=i)
            except pyvips.error.Error as e:
                print(f"error at page {i}: {e}")
                continue

            w = page.width
            h = page.height
            nc = page.bands
            img_area = w*h*nc

            all_areas.append(img_area)
            all_dims.append([w, h])
            all_n_channels.append(nc)

            if prev_area is None:
                prev_area = img_area
                level_starts.append(0)

            else:
                if prev_area != img_area:
                    level_starts.append(i)

            prev_area = img_area

        level_starts = np.array(level_starts)
        area_diff = np.diff(all_areas)
        most_common_channel_count = mode(all_n_channels)

        unique_areas, _ = np.unique(all_areas, return_index=True)
        n_zero_diff = len(np.where(area_diff == 0)[0])
        if most_common_channel_count == 1 and n_zero_diff >= len(unique_areas):
            is_flattended_pyramid = True

        if is_flattended_pyramid:
            nchannels_per_each_level = np.diff(level_starts)
            last_level_channel_count = np.sum(all_n_channels[level_starts[-1]:])
            nchannels_per_each_level = np.hstack([nchannels_per_each_level,
                                                  last_level_channel_count])

            if last_level_channel_count == 3 and nchannels_per_each_level[0] != 3:
                # last level is probably a thumbnail
                nchannels_per_each_level = nchannels_per_each_level[:-1]
            n_channels = mode(nchannels_per_each_level)
            levels_start_idx = level_starts[np.where(nchannels_per_each_level==n_channels)[0]]
            slide_dimensions = np.array(all_dims)[levels_start_idx]

        else:
            slide_dimensions = all_dims
            levels_start_idx = np.arange(0, len(slide_dimensions))
            n_channels = most_common_channel_count

    else:
        return is_flattended_pyramid, can_use_bf, None, None, None
    # Now check if Bioformats reads it similarly #
    if check_with_bf:
        with valtils.HiddenPrints():
            bf_reader = BioFormatsSlideReader(src_f)
        bf_levels = len(bf_reader.metadata.slide_dimensions)
        bf_channels = bf_reader.metadata.n_channels
        can_use_bf = bf_levels >= len(slide_dimensions) and bf_channels == n_channels

    return is_flattended_pyramid, can_use_bf, slide_dimensions, levels_start_idx, n_channels


def check_xml_img_match(xml, vips_img, metadata, series=0):
    """ Make sure that provided xml and image match.
    If there is a mismatch (i.e. channel number), the values in the image take precedence
    """
    ome_obj = get_ome_obj(xml)
    if len(ome_obj.images) > 0:
        ome_img = ome_obj.images[series].pixels
        ome_nc = ome_img.size_c
        ome_nt = ome_img.size_t
        ome_nz = ome_img.size_z
        ome_size_x = ome_img.size_x
        ome_size_y = ome_img.size_y
        ome_dtype = ome_img.type.name.lower()

        total_pages = ome_nc*ome_nt*ome_nz
    else:
        msg = f"ome-xml for {metadata.name} does not contain any metadata for any images"
        valtils.print_warning(msg)
        ome_nc = None
        ome_nt = None
        ome_nz = None
        ome_size_x = None
        ome_size_y = None
        ome_dtype = None
        total_pages = vips_img.bands

    vips_nc = vips_img.bands
    vips_size_x = vips_img.width
    vips_size_y = vips_img.height
    np_dtype = slide_tools.VIPS_FORMAT_NUMPY_DTYPE[vips_img.format]
    vips_bf_dtype = slide_tools.NUMPY_FORMAT_BF_DTYPE[str(np_dtype().dtype)].lower()

    if total_pages != vips_img.bands:
            msg = (f"For {metadata.name}, the ome-xml states there should be {total_pages} pages, but there is/are {vips_img.bands} pages in the image",
                   f"Assuming that all pages are channels (not time points or Z-axes), so updating the metadata to have {total_pages} channels")
            metadata.n_channels = vips_nc
            metadata.n_t = 1
            metadata.n_z = 1
            if ome_nc is not None :
                valtils.print_warning(msg)

    if ome_size_x != vips_size_x:
        msg = f"For {metadata.name}, the ome-xml states the width should be {ome_size_x}, but the image has a width of {vips_size_x}"
        if ome_size_x is not None:
            valtils.print_warning(msg)

    if ome_size_y != vips_size_y:
        msg = f"For {metadata.name}, the ome-xml states the height should be {ome_size_y}, but the image has a width of {vips_size_y}"
        if ome_size_y is not None:
            valtils.print_warning(msg)

    if ome_dtype != vips_bf_dtype:
        msg = f"For {metadata.name}, the ome-xml states the image type should be {ome_dtype}, but the image has type of {vips_bf_dtype}"
        metadata.bf_datatype = vips_bf_dtype
        metadata.bf_pixel_type = slide_tools.BF_DTYPE_PIXEL_TYPE[vips_bf_dtype]

        if ome_dtype is not None:
            valtils.print_warning(msg)

    return metadata


def get_bioformats_reader_and_meta(src_f):
    init_jvm()
    rdr = loci.formats.ImageReader()
    factory = loci.common.services.ServiceFactory()
    OMEXMLService_class = loci.formats.services.OMEXMLService

    service = factory.getInstance(OMEXMLService_class)
    ome_meta = service.createOMEXMLMetadata()
    rdr.setMetadataStore(ome_meta)
    rdr.setFlattenedResolutions(False)
    rdr.setId(src_f)
    meta = rdr.getMetadataStore()

    return rdr, meta


def metadata_from_xml(xml, name, server, series=0, metadata=None):
    """
    Use ome-types to extract metadata from xml.
    """

    ome_info = get_ome_obj(xml)
    ome_img = ome_info.images[series]

    if metadata is None:
        metadata = MetaData(name=name, server=server, series=series)

    if ome_img.pixels.big_endian is not None:
        metadata.is_little_endian = ome_img.pixels.big_endian == False

    has_channel_info = len(ome_img.pixels.channels) > 0

    if has_channel_info:
        samples_per_pixel = ome_img.pixels.channels[0].samples_per_pixel
        if samples_per_pixel is None:
            samples_per_pixel = ome_img.pixels.size_c
        metadata.is_rgb = samples_per_pixel == 3 and \
            ome_img.pixels.type.value == 'uint8' and \
            len(ome_img.pixels.channels) == 1
    else:
        # No channel info, so guess based on image shape and datatype
        metadata.is_rgb = ome_img.pixels.type.value == 'uint8' and ome_img.pixels.size_c == 3

    if ome_img.pixels.physical_size_x is not None:
        metadata.pixel_physical_size_xyu = (ome_img.pixels.physical_size_x, ome_img.pixels.physical_size_y, MICRON_UNIT)
    else:
        metadata.pixel_physical_size_xyu = (1, 1, PIXEL_UNIT)

    metadata.n_channels = ome_img.pixels.size_c
    metadata.n_z = ome_img.pixels.size_z
    metadata.n_t = ome_img.pixels.size_t
    metadata.original_xml = ome_info.to_xml()
    metadata.bf_datatype = ome_img.pixels.type.value
    metadata.bf_pixel_type = slide_tools.BF_DTYPE_PIXEL_TYPE[metadata.bf_datatype]

    if not metadata.is_rgb:
        if has_channel_info:
            metadata.channel_names = [ome_img.pixels.channels[i].name for i in range(metadata.n_channels)]
            metadata.channel_names = check_channel_names(metadata.channel_names, metadata.is_rgb, metadata.n_channels, src_f=name)
        else:
            metadata.channel_names = get_default_channel_names(metadata.n_channels, src_f=name)

    return metadata


def openslide_desc_2_omexml(vips_img):
    """Get basic metatad using openslide and convert to ome-xml

    """
    assert "openslide.vendor" in vips_img.get_fields(), "image does not appear to be openslide metadata"
    img_shape_wh = warp_tools.get_shape(vips_img)[0:2][::-1]
    x, y, z, c, t = get_shape_xyzct(shape_wh=img_shape_wh, n_channels=vips_img.bands)

    np_dtype = slide_tools.VIPS_FORMAT_NUMPY_DTYPE[vips_img.format]
    bf_datatype = slide_tools.NUMPY_FORMAT_BF_DTYPE[str(np_dtype().dtype)]

    new_img = ome_types.model.Image(
        id=f"Image:0",
        pixels=ome_types.model.Pixels(
            id="Pixels:0",
            size_x=x,
            size_y=y,
            size_z=z,
            size_c=c,
            size_t=t,
            type=bf_datatype,
            dimension_order='XYZCT',
            physical_size_x = eval(vips_img.get('openslide.mpp-x')),
            physical_size_x_unit = MICRON_UNIT,
            physical_size_y = eval(vips_img.get('openslide.mpp-y')),
            physical_size_y_unit = MICRON_UNIT,
            metadata_only=True
        )
    )

    # Should always be rgb, but checking anyway
    is_rgb = vips_img.interpretation in VIPS_RGB_FORMATS
    if is_rgb:
        rgb_channel = ome_types.model.Channel(id='Channel:0:0', samples_per_pixel=3)
        new_img.pixels.channels = [rgb_channel]

    new_ome = ome_types.OME()
    new_ome.images.append(new_img)

    img_xml = new_ome.to_xml()

    return img_xml



# Read slides #
class MetaData(object):
    """Store slide metadata

    To be filled in by a SlideReader object

    Attributes
    ----------

    name : str
        Name of slide.

    series : int
        Series number.

    server : str
        String indicating what was used to read the metadata.

    slide_dimensions :
        Dimensions of all images in the pyramid (width, height).

    is_rgb : bool
        Whether or not the image is RGB.

    pixel_physical_size_xyu :
        Physical size per pixel and the unit.

    channel_names : list
        List of channel names. None if image is RGB

    n_channels : int
        Number of channels.

    original_xml : str
        Xml string created by bio-formats

    bf_datatype : str
        String indicating bioformats image datatype

    optimal_tile_wh : int
        Tile width and height used to open and/or save image

    """

    def __init__(self, name, server, series=0):
        """

        Parameters
        ----------
        name : str
            Name of slide.

        server : str, optional
            String indicating what was used to read the metadata.

        series : int, optional
            Series number.

        """

        self.name = name
        self.series = series
        self.server = server
        self.slide_dimensions = []
        self.is_rgb = None
        self.pixel_physical_size_xyu = []
        self.channel_names = None
        self.n_channels = 0
        self.n_z = 1
        self.n_t = 1
        self.original_xml = None
        self.bf_datatype = None
        self.bf_pixel_type = None
        self.is_little_endian = None
        self.optimal_tile_wh = 1024


class SlideReader(object):
    """Read slides and get metadata

    Attributes
    ----------
    slide_f : str
        Path to slide

    metadata : MetaData
        MetaData containing some basic metadata about the slide

    series : int
        Image series

    """

    def __init__(self, src_f, *args, **kwargs):
        """
        Parameters
        -----------
        src_f : str
            Path to slide

        """

        self.src_f = str(src_f)
        self.metadata = None
        self.series = 0

    def slide2vips(self, level, xywh=None, *args, **kwargs):
        """Convert slide to pyvips.Image

        Parameters
        -----------
        level : int
            Pyramid level

        xywh : tuple of int, optional
            The region to be sliced from the slide. If None,
            then the entire slide will be converted. Otherwise
            xywh is the (top left x, top left y, width, height) of
            the region to be sliced.

        Returns
        -------
        vips_slide : pyvips.Image
            An  of the slide or the region defined by xywh

        """

    def slide2image(self, level, xywh=None, *args, **kwargs):
        """Convert slide to image

        Parameters
        -----------
        level : int
            Pyramid level

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

    def guess_image_type(self):
        f"""Guess if image is {slide_tools.IHC_NAME} or {slide_tools.IF_NAME}

            Brightfield : RGB or uint8 + 3 channels (after removing alpha)
            Immunofluorescence: != 3 channels and not RGB

        Returns
        -------
        img_type : str
            Image type

        """

        if self.metadata.is_rgb:
            img_type = slide_tools.IHC_NAME
        else:
            img_type = slide_tools.IF_NAME

        return img_type

    def scale_physical_size(self, level):
        """Get resolution pyramid level

        Scale resolution to be for requested pyramid level

        Parameters
        ----------
        level : int
            Pyramid level

        Returns
        -------
        level_xy_per_px: tuple

        """

        level_0_shape = self.metadata.slide_dimensions[0]
        level_shape = self.metadata.slide_dimensions[level]
        scale_x = level_0_shape[0]/level_shape[0]
        scale_y = level_0_shape[1]/level_shape[1]

        level_xy_per_px = (scale_x * self.metadata.pixel_physical_size_xyu[0],
                           scale_y * self.metadata.pixel_physical_size_xyu[1],
                           self.metadata.pixel_physical_size_xyu[2])

        return level_xy_per_px

    def create_metadata(self):
        """ Create and fill in a MetaData object

        Returns
        -------
        metadata : MetaData
            MetaData object containing metadata about slide

        """

    def get_channel_index(self, channel):

        if isinstance(channel, int):
            matching_channel_idx = channel

        elif isinstance(channel, str):
            cnames = [x.upper() for x in self.metadata.channel_names]
            try:
                best_match = get_close_matches(channel.upper(), cnames)[0]
                matching_channel_idx = cnames.index(best_match)
                if best_match.upper() != channel.upper():
                    msg = f"Cannot find exact match to channel '{channel}' in {valtils.get_name(self.src_f)}. Using channel {best_match}"
                    valtils.print_warning(msg)
            except Exception as e:
                traceback_msg = traceback.format_exc()
                matching_channel_idx = 0
                msg = (f"Cannot find channel '{channel}' in {valtils.get_name(self.src_f)}."
                       f" Available channels are {self.metadata.channel_names}."
                       f" Using channel number {matching_channel_idx}, which has name {self.metadata.channel_names[matching_channel_idx]}")

                valtils.print_warning(msg)

        else:
            matching_channel_idx = 0

        return matching_channel_idx

    def get_channel(self, level, series, channel):
        """Get channel from slide

        Parameters
        ----------
        level : int
            Pyramid level

        series  : int
            Series number

        channel : str, int
            Either the name of the channel (string), or the index of the channel (int)

        Returns
        -------
        img_channel : ndarray
            Specified channel sliced from the slide/image

        """

        matching_channel_idx = self.get_channel_index(channel)
        image = self.slide2image(level=level, series=series)
        img_channel = image[..., matching_channel_idx]

        return img_channel

    def _check_rgb(self, *args, **kwargs):
        """Determine if image is RGB

        Returns
        -------
        is_rgb : bool
            Whether or not the image is RGB

        """

    def _get_channel_names(self, *args, **kwargs):
        """Get names of each channel

        Get list of channel names

        Returns
        -------
        channel_names : list
            List of channel names

        """

    def _get_slide_dimensions(self, *args, **kwargs):
        """Get dimensions of slide at all pyramid levels

        Returns
        -------
        slide_dims : ndarray
            Dimensions of all images in the pyramid (width, height).

        """

    def _get_pixel_physical_size(self, *args, **kwargs):
        """Get resolution of slide

        Returns
        -------
        res_xyu : tuple
            Physical size per pixel and the unit, e.g. u'\u00B5m'

        Notes
        -----
            If physical unit is micron, it must be u'\u00B5m',
            not mu (u'\u03bcm') or u.

        """


class BioFormatsSlideReader(SlideReader):
    """Read slides using BioFormats

    Uses the packages jpype and bioformats-jar

    """
    def __init__(self, src_f, series=None, *args, **kwargs):
        """
        Parameters
        -----------
        src_f : str
            Path to slide

        series : int
            The series to be read. If `series` is None, the the `series`
            will be set to the series associated with the largest image.

        """

        init_jvm()

        self.meta_list = [None]
        super().__init__(src_f=src_f, *args, **kwargs)

        try:
            self.meta_list = self.create_metadata()
        except Exception as e:
            traceback_msg = traceback.format_exc()
            valtils.print_warning(e, traceback_msg=traceback_msg)
            kill_jvm()

        self.n_series = len(self.meta_list)
        if series is None:
            img_areas = [np.multiply(*meta.slide_dimensions[0]) for meta in self.meta_list]
            series = np.argmax(img_areas)
            if len(img_areas) > 1:
                msg = (f"No series provided. "
                       f"Selecting series with largest image, "
                       f"which is series {series}")

                valtils.print_warning(msg, warning_type=None, rgb=valtils.Fore.GREEN)

        self._series = series
        self.series = series

    def _set_series(self, series):
        self._series = series
        self.metadata = self.meta_list[series]

    def _get_series(self):
        return self._series

    series = property(fget=_get_series,
                      fset=_set_series,
                      doc="Slide series")

    def get_tiles_parallel(self, level, tile_bbox_list, pixel_type, series=0, z=0, t=0):
        """Get tiles to slice from the slide

        """

        n_tiles = len(tile_bbox_list)
        tile_array = [None] * n_tiles

        def tile2vips_threaded(idx):
            xywh = tile_bbox_list[idx]
            # javabridge.attach()
            # jpype.attachThreadToJVM()
            jpype.java.lang.Thread.attach()
            try:
                tile = self.slide2image(level, series, xywh=tuple(xywh), z=z, t=t)
            except Exception as e:
                traceback_msg = traceback.format_exc()
                valtils.print_warning(e, traceback_msg=traceback_msg, rgb=Fore.RED)
                pass
            # javabridge.detach()
            # jpype.detachThreadFromJVM()
            jpype.java.lang.Thread.detach()

            tile_array[idx] = slide_tools.numpy2vips(tile, self.metadata.pyvips_interpretation)


        n_cpu = valtils.get_ncpus_available() - 1
        res = pqdm(range(n_tiles), tile2vips_threaded, n_jobs=n_cpu, unit="tiles", leave=None)

        return tile_array

    def slide2vips(self, level, series=None, xywh=None, tile_wh=None, z=0, t=0, *args, **kwargs):
        """Convert slide to pyvips.Image

        This method uses Bioformats to slice tiles from the slides, and then
        stitch them together using pyvips.

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

        tile_wh : int, optional
            Size of tiles used to contstruct `vips_slide`

        Returns
        -------
        vips_slide : pyvips.Image
            An  of the slide or the region defined by xywh

        """

        if series is None:
            series = self.series

        else:
            self.series = series

        rdr, meta = self._get_bf_objects()
        pixel_type, drange = bf_to_numpy_dtype(rdr.getPixelType(),
                                               rdr.isLittleEndian())

        slide_shape_wh = self.metadata.slide_dimensions[level]

        if tile_wh is None:
            tile_wh = rdr.getOptimalTileWidth()
        rdr.close()

        tile_wh = min(tile_wh, MAX_TILE_SIZE)
        if np.any(slide_shape_wh < tile_wh):
            tile_wh = min(slide_shape_wh)

        tile_bbox = warp_tools.get_grid_bboxes(slide_shape_wh[::-1],
                                               tile_wh, tile_wh, inclusive=True)

        n_across = len(np.unique(tile_bbox[:, 0]))

        print(f"Converting slide to pyvips image")
        vips_slide = pyvips.Image.arrayjoin(
                                  self.get_tiles_parallel(level, tile_bbox_list=tile_bbox, pixel_type=pixel_type, series=series, z=z, t=t),
                                  across=n_across).crop(0, 0, *slide_shape_wh)
        if xywh is not None:
            vips_slide = vips_slide.extract_area(*xywh)

        return vips_slide

    def slide2image(self, level, series=None, xywh=None, z=0, t=0, *args, **kwargs):
        """Convert slide to image

        Parameters
        -----------
        level : int
            Pyramid level

        series : int, optional
            Series number. Defaults to 1

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

        if series is None:
            series = self.series

        else:
            self.series = series

        rdr, meta = self._get_bf_objects()

        rdr.setSeries(series)
        rdr.setResolution(level)
        if xywh is None:
            x = 0
            y = 0
            w = rdr.getSizeX()
            h = rdr.getSizeY()
            xywh = (x, y, w, h)

        if rdr.isRGB():
            img = self._read_rgb(rdr=rdr, xywh=xywh, z=z, t=t)

        else:
            img = self._read_multichannel(rdr=rdr, xywh=xywh, z=z, t=t)

        rdr.close()

        return img

    def create_metadata(self):
        rdr, meta = self._get_bf_objects()
        meta_xml = meta.dumpXML()
        try:
            n_series = rdr.getSeriesCount()
            i0 = rdr.getSeries()
            slide_format = f"{BF_RDR}_{rdr.getFormat()}"
            meta_list = [None] * n_series
            for i in range(n_series):
                rdr.setSeries(i)
                series_name = str(meta.getImageName(i))
                temp_name = f"{os.path.split(self.src_f)[1]}_{series_name}".strip("_")
                full_name = f"{temp_name}_Series_{i}"
                full_name = full_name.replace(" ", "_")

                series_meta = MetaData(full_name, slide_format, series=i)

                series_meta.is_rgb = self._check_rgb(rdr)
                series_meta.channel_names = self._get_channel_names(rdr, meta)
                series_meta.n_channels = int(rdr.getSizeC())
                series_meta.n_z = rdr.getSizeZ()
                series_meta.n_t = rdr.getSizeT()
                series_meta.slide_dimensions = self._get_slide_dimensions(rdr)
                if series_meta.is_rgb:
                    series_meta.pyvips_interpretation = 'srgb'
                elif series_meta.n_channels == 1:
                    series_meta.pyvips_interpretation = 'b-w'
                else:
                    series_meta.pyvips_interpretation = 'multiband'

                series_meta.pixel_physical_size_xyu = self._get_pixel_physical_size(rdr, meta)
                series_meta.bf_pixel_type = str(rdr.getPixelType())
                series_meta.is_little_endian = rdr.isLittleEndian()
                series_meta.original_xml = str(meta_xml)
                series_meta.bf_datatype = str(FormatTools.getPixelTypeString(rdr.getPixelType()))
                series_meta.optimal_tile_wh = int(rdr.getOptimalTileWidth())

                meta_list[i] = series_meta

            i0 = rdr.setSeries(i0)
            rdr.close()

        except Exception as e:
            traceback_msg = traceback.format_exc()
            valtils.print_warning(e, traceback_msg=traceback_msg)
            rdr.close()

        return meta_list

    def _read_rgb(self, rdr, xywh, z=0, t=0):

        np_dtype, drange = bf_to_numpy_dtype(rdr.getPixelType(),
                                             rdr.isLittleEndian())

        buffer = rdr.openBytes(0, *xywh)
        img = np.frombuffer(bytes(buffer), np_dtype)
        nrgb = rdr.getRGBChannelCount()
        _, _, w, h = xywh

        if rdr.isInterleaved():
            img = img.reshape(h, w, nrgb)
        else:
            img = img.reshape(nrgb, h, w)
            img = np.transpose(img, (1, 2, 0))

        if img.shape[2] > 3:
            img = img[0:3]

        return img

    def _read_multichannel(self, rdr, xywh, z=0, t=0):
        _, _, w, h = xywh
        n_channels = rdr.getSizeC()
        np_dtype, drange = bf_to_numpy_dtype(rdr.getPixelType(),
                                             rdr.isLittleEndian())

        if n_channels > 1:
            img = np.zeros((h, w, n_channels), dtype=np_dtype)
        else:
            img = None

        for i in range(n_channels):
            idx = rdr.getIndex(z, i, t)  # ZCT
            buffer = rdr.openBytes(idx, *xywh)
            if img is None:
                img = np.frombuffer(bytes(buffer), np_dtype).reshape((h, w))
            else:
                img[..., i] = np.frombuffer(bytes(buffer), np_dtype).reshape((h, w))

        return img

    def _get_bf_objects(self):
        """Get Bioformat objects

        Returns
        -------

        rdr : IFormatReader
            IFormatReader object that is a property of a bioformats.ImageReader.

        meta : loci.formats.ome.OMEPyramidStore
            Used to read metadata

        Notes
        -----
        Be sure to close rdr with rdr.close() when it's no longer needed

        """
        # Javabridge #
        #------------#
        # env = javabridge.jutil.get_env()
        # rdr = javabridge.JWrapper(javabridge.make_instance(
        #                           'loci/formats/ImageReader', '()V')
        #                           )

        # factory = javabridge.JWrapper(javabridge.make_instance(
        #                              'loci/common/services/ServiceFactory', '()V')
        #                              )

        # OMEXMLService_class = \
        #     env.find_class('loci/formats/services/OMEXMLService').as_class_object()

        # Jpype #
        #-------#

        rdr, meta = get_bioformats_reader_and_meta(self.src_f)

        return rdr, meta

    def _check_rgb(self, rdr):
        """Determine if image is RGB

        Returns
        -------
        is_rgb : bool
            Whether or not the image is RGB

        """

        return rdr.isRGB()

    def _get_slide_dimensions(self, rdr):
        """Get dimensions of slide at all pyramid levels

        Parameters
        ----------
        rdr : IFormatReader
            IFormatReader object

        Returns
        -------
        slide_dims : ndarray
            Dimensions of all images in the pyramid (width, height).

        Notes
        -----
        Using javabridge and python-bioformmats, this can be accessed as follows
        `
        bf_slide = bioformats.ImageReader(slide_f)
        bf_img_reader = javabridge.JWrapper(bf_slide.rdr.o)

        Or
        with bioformats.ImageReader(slide_f) as bf_slide:
            bf_img_reader = javabridge.JWrapper(bf_slide.rdr.o)

        """

        r0 = rdr.getResolution()
        n_res = rdr.getResolutionCount()
        slide_dims = [None] * n_res
        for j in range(n_res):
            rdr.setResolution(j)
            slide_dims[j] = [rdr.getSizeX(), rdr.getSizeY()]

        slide_dims = np.array(slide_dims)

        rdr.setResolution(r0)

        return slide_dims

    def _get_pixel_physical_size(self, rdr, meta):
        """Get resolutions for each series

        Parameters
        ----------
        rdr : IFormatReader
            IFormatReader object.

        meta : loci.formats.ome.OMEPyramidStore
            Used to read metadata

        Returns
        -------
        res_xyu : tuple
            Physical size per pixel and the unit, e.g. u'\u00B5m'

        """
        current_series = rdr.getSeries()
        temp_x_res = meta.getPixelsPhysicalSizeX(current_series)
        if temp_x_res is not None:
            x_res = float(temp_x_res.value(BF_MICROMETER).doubleValue())
            y_res = float(meta.getPixelsPhysicalSizeY(current_series).value(BF_MICROMETER).doubleValue())
            phys_unit = str(BF_MICROMETER.getSymbol())
        else:
            x_res = 1
            y_res = 1
            phys_unit = PIXEL_UNIT

        res_xyu = (x_res, y_res, phys_unit)

        return res_xyu

    def _get_channel_names(self, rdr, meta):
        """Get channel names of image
        Parameters
        ----------
        rdr : IFormatReader
            IFormatReader object

        meta : loci.formats.ome.OMEPyramidStore
            Used to read metadata.

        Returns
        -------
        channel_names : list
            List of channel names.

        """

        nc = rdr.getSizeC()
        current_series = rdr.getSeries()
        if rdr.isRGB():
            channel_names = None
        else:
            channel_names = [""] * nc
            for i in range(nc):
                channel_names[i] = str(meta.getChannelName(current_series, i))

        return channel_names


class VipsSlideReader(SlideReader):
    """Read slides using pyvips
    Pyvips includes OpenSlide and so can read those formats as well.

    Attributes
    ----------
    use_openslide : bool
        Whether or not openslide can be used to read this slide.

    is_ome : bool
        Whether ot not the side is an ome.tiff.

    Notes
    -----
    When using openslide, lower levels can only be read without distortion,
    if pixman version 0.40.0 is installed. As of Oct 7, 2021, Macports only has
    pixman version 0.38, which produces distorted lower level images. If using
    macports may need to install from source do  "./configure --prefix=/opt/local/"
    when installing from source.

    """
    def __init__(self, src_f, *args, **kwargs):
        super().__init__(src_f=src_f, *args, **kwargs)
        self.use_openslide = check_to_use_openslide(self.src_f)
        self.is_ome = check_is_ome(self.src_f)
        self.metadata = self.create_metadata()
        self.verify_xml()

    def create_metadata(self):

        vips_img = pyvips.Image.new_from_file(self.src_f)

        if "image-description" in vips_img.get_fields():
            is_image_J = re.search("imagej", vips_img.get("image-description").lower()) is not None
            if is_image_J:
                slide_meta = self._get_metadata_bf()
                return slide_meta

        if self.use_openslide:
            server = OPENSLIDE_RDR
        else:
            server = VIPS_RDR

        meta_name = f"{os.path.split(self.src_f)[1]}_Series(0)".strip("_")
        slide_meta = MetaData(meta_name, server)

        slide_meta.is_rgb = self._check_rgb(vips_img)
        slide_meta.n_channels = vips_img.bands
        if (slide_meta.is_rgb and vips_img.hasalpha() >= 1) or self.use_openslide:
            # Will also remove alpha channel after reading
            slide_meta.n_channels = vips_img.bands - vips_img.hasalpha()
            vips_img = vips_img[0:3]

        slide_meta.slide_dimensions = self._get_slide_dimensions(vips_img)
        img_xml = self._get_xml(vips_img)
        if img_xml is not None:
            try:
               slide_meta = metadata_from_xml(xml=img_xml,
                                              name=slide_meta.name,
                                              server=server,
                                              metadata=slide_meta)
            except Exception as e:
                slide_meta = self._get_metadata_vips(slide_meta, vips_img)

        else:
            slide_meta = self._get_metadata_vips(slide_meta, vips_img)

        if slide_meta.is_rgb:
            slide_meta.channel_names = None

        if self.is_ome:
            toilet_roll = pyvips.Image.new_from_file(self.src_f, n=-1, subifd=-1)
            page = pyvips.Image.new_from_file(self.src_f, n=1, subifd=-1, access='random')
            n_pages = toilet_roll.height/page.height
            if n_pages > 1:
                slide_meta.n_channels = int(n_pages)

        return slide_meta

    def verify_xml(self):
        img_xml = self.metadata.original_xml
        if img_xml is not None and not self.use_openslide:
            # Don't check openslide images, as metadata counts alpha channel
            try:
                ome_info = get_ome_obj(img_xml)
                assert len(ome_info.images) > 0
            except:
                return None
            read_img = self.slide2vips(0)
            self.metadata = check_xml_img_match(xml=img_xml, vips_img=read_img, metadata=self.metadata, series=self.series)

    def _get_metadata_vips(self, slide_meta, vips_img):
        slide_meta.n_channels = vips_img.bands
        slide_meta.channel_names = self._get_channel_names(vips_img, n_channels=slide_meta.n_channels)
        slide_meta.pixel_physical_size_xyu = self._get_pixel_physical_size(vips_img)
        np_dtype = slide_tools.VIPS_FORMAT_NUMPY_DTYPE[vips_img.format]
        slide_meta.bf_datatype = slide_tools.NUMPY_FORMAT_BF_DTYPE[str(np_dtype().dtype)]
        slide_meta.bf_pixel_type = slide_tools.BF_DTYPE_PIXEL_TYPE[slide_meta.bf_datatype]
        slide_meta.is_little_endian = sys.byteorder.startswith("l")
        slide_meta.original_xml = self._get_xml(vips_img)
        slide_meta.optimal_tile_wh = get_tile_wh(self, 0, warp_tools.get_shape(vips_img)[0:2][::-1])


        return slide_meta

    def _get_metadata_bf(self):
        with valtils.HiddenPrints():
            bf_reader = BioFormatsSlideReader(self.src_f)

        # slide_meta.channel_names = bf_reader.metadata.channel_names # None if RGB
        # # Need to update the n_channels based on bioformats metadata if toilet roll .ome.tiff
        # slide_meta.n_channels = bf_reader.metadata.n_channels
        # slide_meta.pixel_physical_size_xyu = bf_reader.metadata.pixel_physical_size_xyu
        # slide_meta.bf_pixel_type = bf_reader.metadata.bf_pixel_type
        # slide_meta.is_little_endian = bf_reader.metadata.is_little_endian
        # slide_meta.original_xml = bf_reader.metadata.original_xml
        # slide_meta.bf_datatype = bf_reader.metadata.bf_datatype
        # slide_meta.optimal_tile_wh = bf_reader.metadata.optimal_tile_wh

        return bf_reader.metadata

    def _slide2vips_ome_one_series(self, level, *args, **kwargs):
        """Use pyvips to read an ome.tiff image that has only 1 series

        Pyvips throws an error when trying to read other series
        because they may have a different shape than the 1st one
        https://github.com/libvips/pyvips/issues/262

        Parameters
        -----------
        level : int
            Pyramid level

        xywh : tuple of int, optional
            The region to be sliced from the slide. If None,
            then the entire slide will be converted. Otherwise
            xywh is the (top left x, top left y, width, height) of
            the region to be sliced.

        Returns
        -------
        vips_slide : pyvips.Image
            An  of the slide or the region defined by xywh

        """

        toilet_roll = pyvips.Image.new_from_file(self.src_f, n=-1, subifd=level-1)
        page = pyvips.Image.new_from_file(self.src_f, n=1, subifd=level-1, access='random')
        if page.interpretation in VIPS_RGB_FORMATS:
            vips_slide = page
        else:
            page_height = page.height
            pages = [toilet_roll.crop(0, y, toilet_roll.width, page_height) for
                     y in range(0, toilet_roll.height, page_height)]

            vips_slide = pages[0].bandjoin(pages[1:])
            if vips_slide.bands == 1:
                vips_slide = vips_slide.copy(interpretation="b-w")
            else:
                vips_slide = vips_slide.copy(interpretation="multiband")


        return vips_slide

    def slide2vips(self, level, xywh=None, *args, **kwargs):
        """Convert slide to pyvips.Image

        Parameters
        -----------
        level : int
            Pyramid level

        xywh : tuple of int, optional
            The region to be sliced from the slide. If None,
            then the entire slide will be converted. Otherwise
            xywh is the (top left x, top left y, width, height) of
            the region to be sliced.

        Returns
        -------
        vips_slide : pyvips.Image
            An  of the slide or the region defined by xywh

        """
        if level < 0:
            print(f"level is negative {level} for {self.src_f}")
        level = max(0, level)

        if self.use_openslide:
            vips_slide = pyvips.Image.new_from_file(self.src_f, level=level, access='random')[0:3]

        elif self.is_ome:
            vips_slide = self._slide2vips_ome_one_series(level=level, *args, **kwargs)

        else:
            try:
                vips_slide = pyvips.Image.new_from_file(self.src_f, subifd=level-1, access='random')
            except Exception as e:
                if level > 0 and len(self.metadata.slide_dimensions) > 1:
                    # Pyramid image but each level is a page, not a SubIFD
                    vips_slide = pyvips.Image.new_from_file(self.src_f, page=level, access='random')
                else:
                    # Regular images like png or jpeg don't have SubIFD or pages
                    vips_slide = pyvips.Image.new_from_file(self.src_f, access='random')

        if self.metadata.is_rgb and vips_slide.hasalpha() >= 1:
            # Remove alpha channel
            vips_slide = vips_slide.flatten()

        if xywh is not None:
            vips_slide = vips_slide.extract_area(*xywh)

        return vips_slide

    def slide2image(self, level, xywh=None, *args, **kwargs):
        """Convert slide to image

        Parameters
        -----------
        level : int
            Pyramid level.

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

        vips_slide = self.slide2vips(level=level, xywh=xywh, *args, **kwargs)
        vips_img = slide_tools.vips2numpy(vips_slide)

        return vips_img

    def _check_rgb(self, vips_img):
        """Determine if image is RGB

        See https://github.com/libvips/libvips/issues/580#issuecomment-272804812
        3 types of RGB
        srgb = 0-255, uint8
        rgb16 =  0-65535, uint16
        scrgb = 0-1, float

        Parameters
        ----------
        vips_img : pyvips.Image
            pyvips.Image of slide.

        Returns
        -------
        is_rgb : bool
            Whether or not the image is RGB.

        """

        is_rgb = vips_img.interpretation in VIPS_RGB_FORMATS

        return is_rgb

    def _get_xml(self, vips_img):
        img_desc = None
        vips_fields = vips_img.get_fields()

        if "openslide.vendor" in vips_fields:
            img_desc = openslide_desc_2_omexml(vips_img)

        elif "image-description" in vips_fields:
            img_desc = vips_img.get("image-description")

        return img_desc

    def _get_channel_names(self, vips_img, *args, **kwargs):
        """Get names of each channel

        Get list of channel names.

        Parameters
        ----------
        vips_img : pyvips.Image
            pyvips.Image of slide.

        Returns
        -------
        channel_names : list
            List of channel naames.

        """

        vips_fields = vips_img.get_fields()
        channel_names = None
        if 'n-pages' in vips_fields and "image-description" in vips_fields:
            n_pages = vips_img.get("n-pages")
            channel_names = []
            for i in range(n_pages):
                page = pyvips.Image.new_from_file(self.src_f, page=i)
                page_fields = page.get_fields()
                if "image-description" not in page_fields:
                    continue

                page_metadata = page.get("image-description")

                with valtils.HiddenPrints():
                    page_soup = BeautifulSoup(page_metadata, features="lxml")

                cname = page_soup.find("name")

                if cname is not None:
                    if cname.text not in channel_names:
                        channel_names.append(cname.text)

        is_vips_rgb = vips_img.interpretation in VIPS_RGB_FORMATS
        if (channel_names is None or len(channel_names) == 0) and not is_vips_rgb:
            channel_names = get_default_channel_names(vips_img.bands,
                                                      src_f=self.src_f)
        return channel_names

    def _get_slide_dimensions(self, vips_img):
        """Get dimensions of slide at all pyramid levels using openslide.

        Parameters
        ----------
        vips_img : pyvips.Image
            pyvips.Image of slide.

        Returns
        -------
        slide_dims : ndarray
            Dimensions of all images in the pyramid (width, height).

        """

        if self.use_openslide:
            slide_dimensions = self._get_slide_dimensions_openslide(vips_img)

        elif self.is_ome:
            slide_dimensions = self._get_slide_dimensions_ometiff(vips_img)

        else:
            slide_dimensions = self._get_slide_dimensions_vips(vips_img)

        return slide_dimensions

    def _get_slide_dimensions_ometiff_bf(self, *args):
        with valtils.HiddenPrints():
            bf_reader = BioFormatsSlideReader(self.src_f)

        return np.array(bf_reader.metadata.slide_dimensions)

    def _get_slide_dimensions_ometiff(self, vips_img, *args):

        if "n-subifds" not in vips_img.get_fields():
            # non-pyramid ome.tiff
            slide_dims_wh = self._get_slide_dimensions_vips(vips_img)
            _, unique_dim_idx = np.unique(slide_dims_wh, axis=0, return_index=True)
            slide_dims_wh = np.array([slide_dims_wh[i] for i in sorted(unique_dim_idx)])

            return slide_dims_wh

        n_levels = vips_img.get("n-subifds") + 1
        slide_dims_wh = [None] * n_levels
        for i in range(0, n_levels):
            page = pyvips.Image.new_from_file(self.src_f, n=1, subifd=i-1)
            slide_dims_wh[i] = np.array([page.width, page.height])

        slide_dims_wh = np.array(slide_dims_wh)

        return slide_dims_wh

    def _get_slide_dimensions_openslide(self, vips_img):
        """Get dimensions of slide at all pyramid levels using openslide

        Parameters
        ----------
        vips_img : pyvips.Image
            pyvips.Image of slide

        Returns
        -------
        slide_dims : ndarray
            Dimensions of all images in the pyramid (width, height).

        """

        n_levels = eval(vips_img.get('openslide.level-count'))
        slide_dims = np.array([[eval(vips_img.get(f"openslide.level[{i}].width")),
                              eval(vips_img.get(f"openslide.level[{i}].height"))]
                              for i in range(n_levels)])

        return slide_dims

    def _get_slide_dimensions_vips(self, vips_img):
        """Get dimensions of slide at all pyramid levels using vips

        Parameters
        ----------
        vips_img : pyvips.Image
            pyvips.Image of slide

        Returns
        -------
        slide_dims : ndarray
            Dimensions of all images in the pyramid (width, height).

        """

        vips_fields = vips_img.get_fields()
        if 'n-pages' in vips_fields:
            n_pages = vips_img.get("n-pages")
            all_dims = []
            all_channels = []
            for i in range(n_pages):
                try:
                    page = pyvips.Image.new_from_file(self.src_f, page=i)
                except pyvips.error.Error as e:
                    print(f"error at page {i}: {e}")

                w = page.width
                h = page.height
                c = page.bands

                all_dims.append([w, h])
                all_channels.append(c)

            try:
                most_common_channel_count = stats.mode(all_channels, keepdims=True)[0][0]
            except:
                most_common_channel_count = stats.mode(all_channels)[0][0]

            all_dims = np.array(all_dims)
            keep_idx = np.where(all_channels == most_common_channel_count)[0]
            slide_dims = all_dims[keep_idx]

        else:
            slide_dims = [[vips_img.width, vips_img.height]]

        return np.array(slide_dims)

    def _get_pixel_physical_size(self, vips_img):
        """Get resolution of slide

        Parameters
        ----------
        vips_img : pyvips.Image
            pyvips.Image of slide

        Returns
        -------
        res_xyu : tuple
            Physical size per pixel and the unit, e.g. u'\u00B5m'

        Notes
        -----
            If physical unit is micron, it must be u'\u00B5m',
            not mu (u'\u03bcm') or u.

        """

        res_xyu = None
        if self.use_openslide:
            x_res = eval(vips_img.get('openslide.mpp-x'))
            y_res = eval(vips_img.get('openslide.mpp-y'))
            vips_img.get('slide-associated-images')
            phys_unit = MICRON_UNIT
        else:
            x_res = vips_img.get("xres")
            y_res = vips_img.get("yres")
            has_units = "resolution-unit" in vips_img.get_fields()
            if x_res != 0 and y_res != 0 and has_units:
                # in vips, x_res and y_res are px/mm (https://www.libvips.org/API/current/VipsImage.html#VipsImage--xres)
                # Need to convert to um/px
                x_res = (1/x_res)*(10**3)
                y_res = (1/y_res)*(10**3)
                phys_unit = MICRON_UNIT
            else:
                # Default value is 0, so not provided
                x_res = 1
                y_res = 1
                phys_unit = PIXEL_UNIT

        res_xyu = (x_res, y_res, phys_unit)

        return res_xyu


class FlattenedPyramidReader(VipsSlideReader):
    """Read flattened pyramid using pyvips
    Read slide pyramids where each page/plane is a channel in the pyramid.
    An example would be one where the plane dimensions are
    something like
    [(600, 600), (600, 600), (600, 600), (300, 300), (300, 300), (300, 300)]
    for a 3 channel image with  2  pyramid levels. It seems that bioformats
    does not recognize these as pyramid images.

    """

    def __init__(self, src_f, *args, **kwargs):
        super().__init__(src_f, *args, **kwargs)
        # BF Datatype may not match min/max values in the image
        # e.g. datatype is uint32, but min and max are floats
        self.metadata.img_dtype = None
        self.metadata.img_dtype = self._get_dtype()


    def create_metadata(self):
        is_flattended_pyramid, bf_reads_flat, slide_dimensions,\
        levels_start_idx, n_channels = \
            check_flattened_pyramid_tiff(self.src_f)

        assert is_flattended_pyramid and not bf_reads_flat, "Trying to use FlattenedPyramidReader but slide is not a flattened pyramid"

        meta_name = f"{os.path.split(self.src_f)[1]}_Series(0)".strip("_")
        server = VIPS_RDR
        slide_meta = MetaData(meta_name, server)
        slide_meta.slide_dimensions = slide_dimensions
        slide_meta.n_channels = n_channels
        slide_meta.levels_start_idx = levels_start_idx

        vips_img = pyvips.Image.new_from_file(self.src_f)
        slide_meta.is_rgb = self._check_rgb(vips_img)
        slide_meta.n_pages = self._get_page_count(vips_img)

        img_xml = self._get_xml(vips_img)
        can_read_xml = False
        if img_xml is not None:
            try:
               slide_meta = metadata_from_xml(xml=img_xml,
                                              name=slide_meta.name,
                                              server=server,
                                              metadata=slide_meta)
               can_read_xml = True
            except Exception as e:
                slide_meta = self._get_metadata_vips(slide_meta, vips_img)

        else:
            slide_meta = self._get_metadata_vips(slide_meta, vips_img)

        if slide_meta.is_rgb:
            slide_meta.channel_names = None

        if can_read_xml:
            # Verify basic info of read image matches xml
            read_img = self.slide2vips(0)
            slide_meta = check_xml_img_match(img_xml, read_img, slide_meta, series=self.series)

        return slide_meta

    def slide2vips(self, level, xywh=None, *args, **kwargs):
        """Convert slide to pyvips.Image

        Parameters
        -----------
        level : int
            Pyramid level

        xywh : tuple of int, optional
            The region to be sliced from the slide. If None,
            then the entire slide will be converted. Otherwise
            xywh is the (top left x, top left y, width, height) of
            the region to be sliced.

        Returns
        -------
        vips_slide : pyvips.Image
            An  of the slide or the region defined by xywh

        """

        level_start = self.metadata.levels_start_idx[level]
        vips_slide = None
        level_shape = self.metadata.slide_dimensions[level]
        for i in range(level_start, self.metadata.n_pages):
            page = pyvips.Image.new_from_file(self.src_f, page=i, access='random')
            page_shape = np.array([page.width, page.height])
            if not np.all(page_shape == level_shape):
                continue

            if vips_slide is None:
                vips_slide = page
            else:
                vips_slide = vips_slide.bandjoin(page)

        if xywh is not None:
            vips_slide = vips_slide.extract_area(*xywh)

        if self.metadata.bf_datatype != self.metadata.img_dtype and self.metadata.img_dtype is not None:
            # Min/max/response datatypes in xml don't match values image.
            msg = (f"Bio-formats datatype is {self.metadata.bf_datatype}, "
                   f"but min/max/response values in xml are {self.metadata.img_dtype}. "
                   f"Converting to {self.metadata.img_dtype}"
                   )
            valtils.print_warning(msg)
            vips_dtype = bf2vips_dtype(self.metadata.img_dtype)
            vips_slide = vips_slide.copy(format=vips_dtype)
            self.bf_datatype = self.metadata.img_dtype

        if vips_slide.bands == 1:
            vips_slide = vips_slide.copy(interpretation="b-w")
        elif vips_slide.bands == 3 and vips_slide.format == 'uchar':
            vips_slide = vips_slide.copy(interpretation="srgb")
        else:
            vips_slide = vips_slide.copy(interpretation="multiband")

        return vips_slide

    def slide2image(self, level, xywh=None, *args, **kwargs):
        vips_slide = self.slide2vips(level=level, xywh=xywh, *args, **kwargs)
        try:
            vips_img = slide_tools.vips2numpy(vips_slide)
        except pyvips.error.Error as e:
            # Big hack for when get the error "tiff2vips: out of order read" even with random access
            out_shape_wh = self.metadata.slide_dimensions[level]
            msg1 = f"pyvips.error.Error: {e} when converting pvips.Image to numpy array"
            msg2 = f"Will try to resize level 0 to have shape {out_shape_wh} and convert"
            valtils.print_warning(msg1)
            valtils.print_warning(msg2, None)

            s = np.mean(out_shape_wh/self.metadata.slide_dimensions[0])
            l0_slide = self.slide2vips(level=0, xywh=xywh, *args, **kwargs)
            resized = l0_slide.resize(s)
            vips_img = slide_tools.vips2numpy(resized)
            if not np.all(vips_img.shape[0:2][::-1] == out_shape_wh):
                vips_img = transform.resize(vips_img, output_shape=out_shape_wh[::-1], preserve_range=True)

        return vips_img

    def _get_channel_names(self, vips_img, *args, **kwargs):
        """Get names of each channel

        Get list of channel names.

        Parameters
        ----------
        vips_img : pyvips.Image
            pyvips.Image of slide.

        Returns
        -------
        channel_names : list
            List of channel naames.

        """

        def cname_from_tag_name(soup):
            "Ex .qptiff"

            cnames = soup.findAll(re.compile("^Name$", re.I))
            if len(cnames) > 0:
                cnames = [c.text for c in cnames]

            return cnames

        def cname_from_tag_channel(soup):
            "Ex. Indica labs HALO"

            cnames = soup.findAll(re.compile("^Channel$", re.I))

            name_tags = ["Name", "name"]
            name_tag = [t for t in name_tags if cnames[0].has_attr(t)]
            if len(name_tag) == 1:
                name_tag = name_tag[0]
                names = [c.get(name_tag) for c in cnames]
            else:
                names = []

            return names

        default_channel_names = get_default_channel_names(vips_img.bands,
                                                          src_f=self.src_f)

        vips_fields = vips_img.get_fields()
        if "image-description" in vips_fields:
            img_desc = vips_img.get("image-description")
            soup = BeautifulSoup(img_desc, features="xml")

            channel_names = cname_from_tag_name(soup)
            if len(channel_names) == 0:
                channel_names = cname_from_tag_channel(soup)

            if len(channel_names) == 0:
                channel_names = default_channel_names

        else:
            channel_names = default_channel_names

        return channel_names

    def _get_page_count(self, vips_img):
        vips_fields = vips_img.get_fields()
        if 'n-pages' in vips_fields:
            n_pages = vips_img.get("n-pages")
        else:
            n_pages = 0

        return n_pages


    def _get_dtype(self):
        """Get Bio-Formats datatype from values in metadata.

        For example, BF metadata may have image datatype as
        uint32, but in the image descriiption, min/max/resppnse,
        are floats. This will determine if the slide should be cast
        to a different dataatype to match values in metadata.

        """
        smallest_level = len(self.metadata.slide_dimensions) - 1
        vips_img = self.slide2vips(smallest_level)
        vips_fields = vips_img.get_fields()
        current_bf_dtype = vips2bf_dtype(vips_img.format)
        if 'n-pages' in vips_fields:
            page = pyvips.Image.new_from_file(self.src_f, page=0)
            page_metadata = page.get("image-description")

            page_soup = BeautifulSoup(page_metadata, features="lxml")
            channels = page_soup.findAll("channel")
            response = page_soup.findAll("response")
            if len(channels) > 0:
                # Indica Labs tiff
                dtypes = [None] * len(channels)
                for i, chnl in enumerate(channels):
                    if chnl.has_attr("max"):
                        max_v = eval(chnl["max"])
                        dtypes[i] = max_v.__class__.__name__

            elif len(response) > 0:
                # PerkinElmer-QPI tiff
                dtypes = [None] * len(response)
                for i, r in enumerate(response):
                    v = eval(r.getText("response"))
                    dtypes[i] = np.array([v]).dtype
                    dtypes[i] = v.__class__.__name__
            else:
                return current_bf_dtype

            unique_dtypes = set(dtypes)
            if len(unique_dtypes) > 1:
                msg = "More than 1 datatype. Will not try to scale values"
                valtils.print_warning(msg)
                img_dtype = None
            else:
                img_dtype = dtypes[0]

            vals_are_floats = re.search("float", img_dtype) is not None
            img_is_int = re.search("int", current_bf_dtype) is not None
            if vals_are_floats and img_is_int:
                max_v = vips_img.max()

                bf_px_num_type = FormatTools.pixelTypeFromString(self.metadata.bf_datatype)
                temp_np_type, max_v_for_type = bf_to_numpy_dtype(bf_px_num_type, self.metadata.is_little_endian)
                if temp_np_type.endswith('4'):
                    np_type = "float32"
                elif temp_np_type.endswith('8'):
                    np_type = "float64"

                bf_type = slide_tools.NUMPY_FORMAT_BF_DTYPE[np_type]
        else:
            bf_type = current_bf_dtype

        return bf_type


class CziJpgxrReader(SlideReader):
    """Read slides and get metadata

    Attributes
    ----------
    slide_f : str
        Path to slide

    metadata : MetaData
        MetaData containing some basic metadata about the slide

    series : int
        Image series

    """
    def __init__(self, src_f, series=None, *args, **kwargs):
        """
        Parameters
        -----------
        src_f : str
            Path to slide

        series : int
            The series to be read. If `series` is None, the the `series`
            will be set to the series associated with the largest image.

        """
        try:
            from aicspylibczi import CziFile
        except Exception as e:
            traceback_msg = traceback.format_exc()
            msg = "Please install aicspylibczi"
            valtils.print_warning(msg, traceback_msg=traceback_msg)

        czi_reader = CziFile(src_f)
        self.original_meta_dict = valtils.etree_to_dict(czi_reader.meta)

        self.is_bgr = False
        self.meta_list = [None]

        super().__init__(src_f=src_f, *args, **kwargs)

        try:
            self.meta_list = self.create_metadata()
        except Exception as e:
            traceback_msg = traceback.format_exc()
            valtils.print_warning(e, traceback_msg=traceback_msg)

        self.n_series = len(self.meta_list)
        if series is None:
            img_areas = [np.multiply(*meta.slide_dimensions[0]) for meta in self.meta_list]
            series = np.argmax(img_areas)
            if len(img_areas) > 1:
                msg = (f"No series provided. "
                       f"Selecting series with largest image, "
                       f"which is series {series}")

                valtils.print_warning(msg, warning_type=None, rgb=valtils.Fore.GREEN)

        self._series = series
        self.series = series

    def _set_series(self, series):
        self._series = series
        self.metadata = self.meta_list[series]

    def _get_series(self):
        return self._series

    series = property(fget=_get_series,
                      fset=_set_series,
                      doc="Slide scene")

    def _read_whole_img(self, level=0, xywh=None, *args, **kwargs):
        """

        Return
        ------
        img : ndarray

        """

        czi_reader = CziFile(self.src_f)
        img, shp = czi_reader.read_image(S=self.series)
        shp = dict(shp)
        if img.ndim == 4 and shp["T"] == 1:
            img = img[0]

        if self.is_bgr:
            img = img[..., ::-1]

        vips_img = warp_tools.numpy2vips(img)

        return vips_img

    def _read_mosaic(self, level=0, xywh=None, *args, **kwargs):
        # Note, tried multiprocessing (see below), but get blank image.
        czi_reader = CziFile(self.src_f)

        out_shape_wh = self.metadata.slide_dimensions[0]
        tile_bboxes = czi_reader.get_all_mosaic_tile_bounding_boxes(C=0)

        vips_img = pyvips.Image.black(*out_shape_wh, bands=self.metadata.n_channels)
        print(f"Building CZI mosaic for {valtils.get_name(self.src_f)}")
        for tile_info, tile_bbox in tqdm(tile_bboxes.items()):
            m = tile_info.m_index
            x = tile_bbox.x
            y = tile_bbox.y

            np_tile, tile_dims = czi_reader.read_image(S=self.series, M=m)

            slice_dims = [v - 1 for k, v in tile_dims if k not in ["Y", "X", "A"]]

            np_tile = np_tile[(*slice_dims, ...)]
            if self.is_bgr:
                np_tile = np_tile[..., ::-1]

            vips_tile = warp_tools.numpy2vips(np_tile)
            vips_img = vips_img.insert(vips_tile, x, y)

        return vips_img

    def slide2vips(self, level=0, xywh=None, *args, **kwargs):
        try:
            # Image is mosaic
            vips_img = self._read_mosaic(level=level, xywh=xywh,*args, **kwargs)

        except Exception as e:
            print(e)
            print("Reading whole image")
            vips_img = self._read_whole_img(level=level, xywh=xywh,*args, **kwargs)

        czi_reader = CziFile(self.src_f)
        if xywh is not None:
            vips_img = vips_img.extract_area(*xywh)

        if level != 0:
            scaling = self.metadata._zoom_levels[level]
            vips_img = warp_tools.rescale_img(vips_img, scaling)
        if self.is_bgr:
            vips_img = vips_img.copy(interpretation="srgb")

        np_type = slide_tools.CZI_FORMAT_TO_BF_FORMAT[czi_reader.pixel_type]
        vips_type = slide_tools.NUMPY_FORMAT_VIPS_DTYPE[np_type]
        vips_img = vips_img.cast(vips_type)

        return vips_img

    def slide2image(self, level, xywh=None, *args, **kwargs):
        """Convert slide to image

        Parameters
        -----------
        level : int
            Pyramid level

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

        vips_img = self.slide2vips(level=level, xywh=xywh, *args, **kwargs)
        np_img = warp_tools.vips2numpy(vips_img)

        return np_img

    def create_metadata(self):
        """ Create and fill in a MetaData object

        Returns
        -------
        metadata : MetaData
            MetaData object containing metadata about slide

        """

        czi_reader = CziFile(self.src_f)
        dims_dict = czi_reader.get_dims_shape()

        n_scenes = len(dims_dict)
        meta_list = [None] * n_scenes
        phys_size = self._get_pixel_physical_size()

        with valtils.HiddenPrints():
            bf_reader = BioFormatsSlideReader(self.src_f)
            original_xml = bf_reader.metadata.original_xml

        for i in range(n_scenes):

            temp_name = f"{os.path.split(self.src_f)[1]}".strip("_")
            full_name = f"{temp_name}_Scene_{i}"

            series_meta = MetaData(full_name, "aicspylibczi", series=i)

            series_meta.is_rgb = self._check_rgb()

            if series_meta.is_rgb:
                n_channels = dims_dict[i]["A"][1]
                series_meta.pyvips_interpretation = 'srgb'
            else:
                n_channels = dims_dict[i]["C"][1]
                if n_channels == 1:
                    series_meta.pyvips_interpretation = 'b-w'
                else:
                    series_meta.pyvips_interpretation = 'multiband'

            series_meta.n_channels = n_channels
            series_meta.slide_dimensions = self._get_slide_dimensions(i)
            series_meta.bf_datatype = slide_tools.CZI_FORMAT_TO_BF_FORMAT[czi_reader.pixel_type]
            series_meta.channel_names = self._get_channel_names(meta=series_meta)

            series_meta.pixel_physical_size_xyu = phys_size
            series_meta.original_xml = original_xml
            series_meta._zoom_levels = self._get_zoom_levels(i)

            meta_list[i] = series_meta


        return meta_list

    def _get_img_meta_dict(self):
        return self.original_meta_dict["ImageDocument"]["Metadata"]["Information"]["Image"]

    def _check_rgb(self, *args, **kwargs):
        """Determine if image is RGB

        Returns
        -------
        is_rgb : bool
            Whether or not the image is RGB

        """
        czi_reader = CziFile(self.src_f)
        self.is_bgr = czi_reader.pixel_type.startswith("bgr")
        _is_rgb = czi_reader.pixel_type.startswith("rgb")
        is_rgb  =_is_rgb or self.is_bgr

        return is_rgb

    def _get_channel_names_aics(self, meta, *args, **kwargs):
        """Get names of each channel

        Get list of channel names

        Returns
        -------
        channel_names : list
            List of channel names

        """

        if meta.is_rgb:
            return None

        img_dict = self._get_img_meta_dict()
        channels = img_dict["Dimensions"]["Channels"]
        if "Channel" in channels:
            channels = channels["Channel"]

        if meta.n_channels == 1 and "@Name" in channels:
            channel_names = [channels["@Name"]]

            return channel_names

        if isinstance(channels, dict):
            channels = list(channels.values())

        try:
            all_channel_ids = [x["@Id"].split(":") for x in channels]
            all_channel_ids = [eval(x["@Id"].split(":")[1]) for x in channels]
            max_c = max([eval(img_dict["SizeC"]), max(all_channel_ids)+1])
            channel_names = [None] * max_c
        except:
            channel_names = [None] * eval(img_dict["SizeC"])

        for chnl_attr in channels:
            chnl_name = chnl_attr["@Name"]
            chnl_idx = eval(chnl_attr["@Id"].split(":")[1])
            channel_names[chnl_idx] = chnl_name

        channel_names = [x for x in channel_names if x is not None]

        return channel_names


    def _get_channel_names_bf(self, meta, *args, **kwargs):
        """Get names of each channel

        Get list of channel names

        Returns
        -------
        channel_names : list
            List of channel names

        """
        if meta.is_rgb:
            return None


        with valtils.HiddenPrints():
            bf_reader = BioFormatsSlideReader(self.src_f)

        rdr, bf_meta = bf_reader._get_bf_objects()
        channel_names = bf_reader._get_channel_names(rdr, bf_meta)

        return channel_names

    def _get_channel_names(self, meta, *args, **kwargs):
        channel_names = self._get_channel_names_aics(meta)
        return channel_names

    def _get_slide_dimensions(self, scene=0, *args, **kwargs):
        """Get dimensions of slide at all pyramid levels

        Returns
        -------
        slide_dims : ndarray
            Dimensions of all images in the pyramid (width, height).

        """
        zoom_levels = self._get_zoom_levels(scene)

        czi_reader = CziFile(self.src_f)
        scene_bbox = czi_reader.get_all_scene_bounding_boxes()[scene]
        scence_l0_wh = np.array([scene_bbox.w, scene_bbox.h])
        slide_dimensions = np.round(scence_l0_wh*zoom_levels[..., np.newaxis]).astype(int)

        return slide_dimensions

    def _get_zoom_levels(self, scene=0):

        img_dict = self._get_img_meta_dict()
        if "S" not in img_dict["Dimensions"]:
            # No pyramid levels
            return np.array([1.0])

        scene_dict = img_dict["Dimensions"]["S"]["Scenes"]["Scene"]
        if scene in scene_dict or isinstance(scene_dict, list):
            pyramid_info = scene_dict[scene]["PyramidInfo"]
        else:
            pyramid_info = scene_dict["PyramidInfo"]

        n_levels = eval(pyramid_info["PyramidLayersCount"])
        downsampling = eval(pyramid_info["MinificationFactor"])
        zoom_levels = (1/downsampling)**(np.arange(0, n_levels))

        return zoom_levels

    def _get_pixel_physical_size(self, *args, **kwargs):
        """Get resolution of slide

        Returns
        -------
        res_xyu : tuple
            Physical size per pixel and the unit, e.g. u'\u00B5m'

        Notes
        -----
            If physical unit is micron, it must be u'\u00B5m',
            not mu (u'\u03bcm') or u.

        """

        physical_sizes = self.original_meta_dict["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]

        physical_size_xyu = [None] * 3
        physical_unit = physical_sizes[0]["DefaultUnitFormat"]
        physical_size_xyu[2] = physical_unit

        if physical_unit == u'\u00B5m':
            physical_scaling = 10**6
        elif physical_unit == "mm":
            physical_scaling = 10**3
        elif physical_unit == "cm":
            physical_scaling = 10**2
        else:
            physical_scaling = 1

        for ps in physical_sizes:
            if ps["@Id"] == "X":
                physical_size_xyu[0] = eval(ps["Value"])*physical_scaling
            elif ps["@Id"] == "Y":
                physical_size_xyu[1] = eval(ps["Value"])*physical_scaling

        return tuple(physical_size_xyu)


class ImageReader(SlideReader):
    """Read image using scikit-image

    """

    def __init__(self, src_f, *args, **kwargs):
        super().__init__(src_f, *args, **kwargs)
        self.metadata = self.create_metadata()

    def create_metadata(self):
        server = IMG_RDR
        meta_name = f"{os.path.split(self.src_f)[1]}_Series(0)".strip("_")
        slide_meta = MetaData(meta_name, server)
        pil_img = Image.open(self.src_f)

        slide_meta.is_rgb = self._check_rgb(pil_img)
        slide_meta.channel_names = self._get_channel_names(pil_img)
        slide_meta.n_channels = self._get_n_channels(pil_img)
        slide_meta.pixel_physical_size_xyu = [1, 1, PIXEL_UNIT]
        slide_meta.slide_dimensions = self._get_slide_dimensions(pil_img)

        f_extension = slide_tools.get_slide_extension(self.src_f)
        if f_extension in BF_READABLE_FORMATS:
            with valtils.HiddenPrints():
                bf_reader = BioFormatsSlideReader(self.src_f)

            slide_meta.original_xml = bf_reader.metadata.original_xml
            slide_meta.bf_datatype = bf_reader.metadata.bf_datatype
        pil_img.close()

        return slide_meta

    def slide2vips(self, xywh=None, *args, **kwargs):
        img = self.slide2image(xywh=xywh, *args, **kwargs)
        vips_img = slide_tools.numpy2vips(img)

        return vips_img

    def slide2image(self, xywh=None, *args, **kwargs):
        img = io.imread(self.src_f)

        if xywh is not None:
            xywh = np.array(xywh)
            start_c, start_r = xywh[0:2]
            end_c, end_r = xywh[0:2] + xywh[2:]
            img = img[start_r:end_r, start_c:end_c]

        return img

    def _get_slide_dimensions(self, pil_img, *args, **kwargs):
        """
        """
        img_dims = np.array([[pil_img.width, pil_img.height]])

        return img_dims

    def _get_n_channels(self, pil_img, *args, **kwargs):

        n_channels = len(pil_img.getbands())

        return n_channels

    def _check_rgb(self, pil_img, *args, **kwargs):

        is_rgb = pil_img.mode == 'RGB'

        return is_rgb

    def _get_channel_names(self, pil_img, *args, **kwargs):
        is_rgb = pil_img.mode == 'RGB'
        if is_rgb:
            channel_names = None
        else:
            channel_names = pil_img.getbands()
        return channel_names


def get_slide_reader(src_f, series=None):
    """Get appropriate SlideReader

    If a slide can be read by openslide and bioformats, VipsSlideReader will be used
    because it can be opened as a pyvips.Image.

    Parameters
    ----------
    src_f : str
        Path to slide

    series : int, optional
        The series to be read. If `series` is None, the the `series`
        will be set to the series associated with the largest image.
        In cases where there is only 1 image in the file, `series`
        will be 0.

    Returns
    -------
    reader: SlideReader
        SlideReader class that can read the slide and and convert them to
        images or pyvips.Images at the specified level and series. They
        also contain a `MetaData` object that contains information about
        the slide, like dimensions at each level, physical units, etc...

    Notes
    -----
    pyvips will be used to open ome-tiff images when `series` is 0

    """

    src_f = str(src_f)
    f_extension = slide_tools.get_slide_extension(src_f)
    is_ome_tiff = check_is_ome(src_f)
    is_tiff = re.search(".tif*", f_extension) is not None and not is_ome_tiff
    is_czi = f_extension == ".czi"

    is_flattened_tiff = False
    bf_reads_flat = False
    if is_tiff:
        is_flattened_tiff, _ = check_flattened_pyramid_tiff(src_f, check_with_bf=False)[0:2]

    one_series = True
    if is_ome_tiff:
        ome_obj = get_ome_obj(src_f)
        one_series = len(ome_obj.images) <= 1

    can_use_vips = check_to_use_vips(src_f)
    can_use_openslide = check_to_use_openslide(src_f) # Checks openslide is installed

    # Give preference to vips/openslide since it will be fastest
    if (can_use_vips or can_use_openslide) and one_series and series in [0, None] and not is_flattened_tiff:
        return VipsSlideReader

    if is_czi:
        is_jpegxr = check_czi_jpegxr(src_f)
        is_m1_mac = valtils.check_m1_mac()
        if is_m1_mac and is_jpegxr:
            msg = "Will likely be errors using Bioformats to read a JPEGXR compressed CZI on this Apple M1 machine. Will use CziJpgxrReader instead."
            return CziJpgxrReader

    # Check to see if Bio-formats will work
    init_jvm()
    can_read_meta_bf, can_read_img_bf = check_to_use_bioformats(src_f, series=series)
    can_use_bf = can_read_meta_bf and can_read_img_bf
    if is_flattened_tiff:
        _, bf_reads_flat = check_flattened_pyramid_tiff(src_f, check_with_bf=True)[0:2]
        # Give preference to BioFormatsSlideReader since it will be faster
        if bf_reads_flat and can_read_img_bf:
            return BioFormatsSlideReader
        else:
            return FlattenedPyramidReader

    if is_czi:
        if can_read_img_bf:
            # Bio-formats should be able to read CZI
            return BioFormatsSlideReader
        else:
            # Bio-formats unable to read CZI. Check if it is due to jpgxr compression
            czi = CziFile(src_f)
            comp_tree = czi.meta.findall(".//OriginalCompressionMethod")
            if len(comp_tree) > 0:
                is_czi_jpgxr = comp_tree[0].text.lower() == "jpgxr"
            else:
                is_czi_jpgxr = False

            if is_czi_jpgxr:
                return CziJpgxrReader
            else:
                msg = f"Unable to find reader to open {os.path.split(src_f)[-1]}"
                valtils.print_warning(msg, rgb=Fore.RED)

                return None

    if can_use_bf:
        return BioFormatsSlideReader

    # Try using scikit-image. Not ideal if image is large
    try:
        ImageReader(src_f)
        return ImageReader
    except:
        pass

    msg = f"Unable to find reader to open {os.path.split(src_f)[-1]}"
    valtils.print_warning(msg, rgb=Fore.RED)

    return None


# Write slides to ome.tiff #

def remove_control_chars(s):
    """Remove control characters

    Control characters shouldn't be in some strings, like channel names.
    This will remove them

    Parameters
    ----------
    s : str
        String to check

    Returns
    -------
    control_char_removed : str
        `s` with control characters removed.

    """

    control_chars = ''.join(map(chr, itertools.chain(range(0x00,0x20), range(0x7f,0xa0))))
    control_char_re = re.compile('[%s]' % re.escape(control_chars))
    control_char_removed = control_char_re.sub('', s)

    return control_char_removed


def get_shape_xyzct(shape_wh, n_channels, nt=1, nz=1):
    """Get image shape in XYZCT format

    Parameters
    ----------
    shape_wh : tuple of int
        Width and heigth of image

    n_channels : int
        Number of channels in the image

    Returns
    -------
    xyzct : tuple of int
        XYZCT shape of the image

    """

    xyzct = (*shape_wh, nz, n_channels, nt)
    return xyzct


def create_channel(channel_id, name=None, color=None, samples_per_pixel=1):
    """Create an ome-xml channel

    Parameters
    ----------
    channel_id : int
        Channel number

    name : str, optinal
        Channel name

    color : tuple of int
        Channel color

    Returns
    -------
    new_channel : ome_types.model.channel.Channel
        Channel object

    """

    if name is not None:
        unicode_name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore')
        decoded_name = unicode_name.decode('unicode_escape')
        decoded_name = remove_control_chars(decoded_name)

    else:
        decoded_name = None

    new_channel = ome_types.model.Channel(id=f"Channel:{channel_id}")
    new_channel.samples_per_pixel = samples_per_pixel
    if name is not None:
        new_channel.name = decoded_name
    if color is not None:

        if len(color) == 3:
            new_channel.color = tuple([*color, 1])
        elif len(color) == 4:
            if color[3] == 0:
                # color has alpha, won't be shown because it's 0
                color = tuple([*color[:3], 1])
        new_channel.color = tuple(color)

    return new_channel


def get_colormap(channel_names, is_rgb, series=0, original_xml=None):

    if is_rgb:
        colormap = {None: (255, 255, 255)}

    else:
        nc = len(channel_names)
        channel_colors = slide_tools.get_matplotlib_channel_colors(nc)
        default_colormap = {channel_names[i]: channel_colors[i] for i in range(nc)}

        if original_xml is not None:
            try:
                # Try to get original colors
                og_ome = get_ome_obj(original_xml)

                ome_img = og_ome.images[series]
                colormap = {c.name: c.color.as_rgb_tuple() for c in ome_img.pixels.channels}
                all_rgb = set(list(colormap.values()))
                nc = len(ome_img.pixels.channels)
                if len(all_rgb) < nc:
                    # Channels do not have unique colors
                    colormap = default_colormap
            except Exception as e:
                print(e)
                # Can't get original colors
                colormap = default_colormap
        else:
            colormap = default_colormap

    return colormap


def check_colormap(colormap, channel_names):
    """Make sure colormap is valid
    If colormap is a dictionary, make sure all `channel_names` are in keys of colormap.
    If colormap is a list, create a dictionary colormap

    Returns
    -------
    updated_colormap : dict
        Colormap dictionary or None if colormap was invalid
    """

    msg = None
    updated_colormap = colormap
    if channel_names is None or len(channel_names) == 0:
        return None

    if isinstance(colormap, str) and colormap == CMAP_AUTO:
        updated_colormap = get_colormap(channel_names, is_rgb=False)

    elif isinstance(colormap, list) or isinstance(colormap, np.ndarray) or isinstance(colormap, tuple):
        if np.array(colormap).ndim == 1 and len(channel_names) == 1:
            # colormap is an array for a single channel
            updated_colormap = np.array([updated_colormap])
        if len(updated_colormap) < len(channel_names):
            msg = f"Not enough colors in colormap. Only {len(updated_colormap)} colors provided, but there are {len(channel_names)} channels"
            updated_colormap = {channel_names[i]: updated_colormap[i] for i in range(len(channel_names))}

    elif isinstance(colormap, dict):

        missing_channels = set(channel_names) - set(colormap.keys())

        if len(missing_channels) != 0:
            msg = f"Missing colors in colormap for the following channels: {missing_channels}"

    elif colormap is not None:
        msg = (f"Colormap must be {CMAP_AUTO}, "
               f"a list of colors with the same length as `channel_names`, ",
               f"a dictionary (key=channel name, value=rgb color), ",
               f"or `None`")

    if msg is not None:
        msg += ". Will not try to add channel colors"
        updated_colormap = None
        valtils.print_warning(msg)

    return updated_colormap


def get_default_channel_names(nc, src_f=None):

    if src_f is not None and nc == 1:
        default_channel_names = [valtils.get_name(src_f)]
    else:
        default_channel_names = [f"C{i+1}" for i in range(nc)]

    return default_channel_names


def check_channel_names(channel_names, is_rgb, nc, src_f=None):

    if is_rgb:
        return None

    default_channel_names = get_default_channel_names(nc, src_f=src_f)

    if channel_names is None:
        channel_names = []

    if len(channel_names) == 0 and nc > 0:
        updated_channel_names = default_channel_names
    else:
        updated_channel_names = [channel_names[i] if
                                 (channel_names[i] is not None and channel_names[i] != "None")
                                 else default_channel_names[i] for i in range(nc)]

    renamed_channels = set(updated_channel_names) - set(channel_names)
    if len(renamed_channels) > 0:
        msg = f"some non-RGB channel names were `None` or not provided. Renamed channels are: {sorted(list(renamed_channels))}"
        print(msg)

    return updated_channel_names


def create_ome_xml(shape_xyzct, bf_dtype, is_rgb, pixel_physical_size_xyu=None, channel_names=None, colormap=CMAP_AUTO):
    """Create new ome-xmml object

    Parameters
    -------
    shape_xyzct : tuple of int
        XYZCT shape of image

    bf_dtype : str
        String format of Bioformats datatype

    is_rgb : bool
        Whether or not the image is RGB

    pixel_physical_size_xyu : tuple, optional
        Physical size per pixel and the unit.

    channel_names : list, optional
        List of channel names.

    colormap : dict, str, optional
        Dictionary of channel colors, where the key is the channel name, and the value the color as rgb255.
        If "auto" (default), the channel colors from `current_ome_xml_str` will be used, if available.
        If `None`, channel colors will not be assigned.

    Returns
    -------
    new_ome : ome_types.model.OME
        ome_types.model.OME object containing ome-xml metadata

    """

    # Minimal ome-xml
    x, y, z, c, t = shape_xyzct
    new_ome = ome_types.OME()
    new_img = ome_types.model.Image(
        id="Image:0",
        pixels=ome_types.model.Pixels(
            id="Pixels:0",
            size_x=x,
            size_y=y,
            size_z=z,
            size_c=c,
            size_t=t,
            type=bf_dtype,
            dimension_order='XYZCT',
            metadata_only=True
        )
    )

    if pixel_physical_size_xyu is not None:
        phys_x, phys_y, phys_u = pixel_physical_size_xyu
        new_img.pixels.physical_size_x = phys_x
        new_img.pixels.physical_size_x_unit = phys_u
        new_img.pixels.physical_size_y = phys_y
        new_img.pixels.physical_size_y_unit = phys_u

    if is_rgb:
        rgb_channel = ome_types.model.Channel(id='Channel:0:0', samples_per_pixel=3)
        new_img.pixels.channels = [rgb_channel]

    else:
        updated_channel_names = check_channel_names(channel_names, is_rgb, nc=c)
        if isinstance(colormap, str) and colormap == CMAP_AUTO:
            colormap = get_colormap(updated_channel_names, is_rgb)

        if colormap is not None:
            colormap = check_colormap(colormap, updated_channel_names)
            try:
                if isinstance(colormap, dict):
                    channels = [create_channel(i, name=updated_channel_names[i], color=colormap[updated_channel_names[i]]) for i in range(c)]
                elif isinstance(colormap, np.ndarray) or isinstance(colormap, list) or isinstance(colormap, tuple):
                    channels = [create_channel(i, name=updated_channel_names[i], color=colormap[i]) for i in range(c)]
            except KeyError as e:
                msg = f"Mismatch between channel names and keys in colormap. Cannot find channel name {e} in colormap"
                if colormap is not None:
                    msg += f", which has keys: {list(colormap.keys())}"
                msg += ". Saving without colormap. To avoid this error, please provide valid colormap, or set colormap=None."
                valtils.print_warning(msg)
                channels = [create_channel(i, name=updated_channel_names[i]) for i in range(c)]
                #Mismatch between channel names and keys in colormap
        else:
            channels = [create_channel(i, name=updated_channel_names[i]) for i in range(c)]

        new_img.pixels.channels = channels

    new_ome = ome_types.model.OME()
    new_ome.images.append(new_img)

    return new_ome


def get_tile_wh(reader, level, out_shape_wh, default_wh=512):
    """Get tile width and height to write image
    If predefined optimal tile wh is too large, will try to
    find the size that minimizes the overhangs
    """

    if reader.metadata is None:
        tile_wh = default_wh
    else:
        slide_meta = reader.metadata
        if slide_meta.optimal_tile_wh is None:
            tile_wh = default_wh
        else:
            tile_wh = slide_meta.optimal_tile_wh

    if level != 0:
        down_sampling = np.mean(slide_meta.slide_dimensions[level]/slide_meta.slide_dimensions[0])
        tile_wh = int(np.round(tile_wh*down_sampling))
        tile_wh = tile_wh - (tile_wh % 16)  # Tile shape must be multiple of 16
        if tile_wh < 16:
            tile_wh = 16

    if np.any(np.array(out_shape_wh[0:2]) < tile_wh):
        # Tile is too big for the image. Get niggest tile size that fit in image
        if tile_wh < default_wh:
            min_wh = 16
        else:
            min_wh = default_wh

        max_tile_exp = np.floor(np.log2(np.min(out_shape_wh)))
        if max_tile_exp <= np.log2(min_wh):
            min_wh = 16

        possible_wh = 2**np.arange(np.log2(min_wh), max_tile_exp+1)
        overhangs = np.array([np.max(np.ceil(out_shape_wh/wh)*wh - out_shape_wh) for wh in possible_wh])
        min_overhang = np.min(overhangs)
        tile_wh = int(np.max(possible_wh[overhangs == min_overhang]))

    return tile_wh


def update_xml_for_new_img(img, reader, level=0, channel_names=None, colormap=CMAP_AUTO, pixel_physical_size_xyu=None):
    """Update dimensions ome-xml metadata

    Used to create a new ome-xmlthat reflects changes in an image, such as its shape

    If `current_ome_xml_str` is invalid or None, a new ome-xml will be created

    Parameters
    -------
    img : ndarry or pyvips.Image
        Image for which xml will be generated. Used to determine shape and datatype.

    reader : SlideReader
        SlideReader used to open `img`. Will use this to extract other metadata, including
        the original xml.

    channel_names : list, optional
        List of channel names.

    colormap : dict, optional
        Dictionary of channel colors, where the key is the channel name, and the value the color as rgb255.
        If "auto" (the default), the channel colors from `current_ome_xml_str` will be used, if available.
        If None, and there are no channel colors in the `current_ome_xml_str`, then no colors will be added

    Returns
    -------
    new_ome : ome_types.model.OME
        ome_types.model.OME object containing ome-xml metadata

    """

    slide_meta = reader.metadata
    img_h, img_w, _ = warp_tools.get_shape(img)

    nc = slide_meta.n_channels
    new_xyzct = get_shape_xyzct((img_w, img_h), nc, nt=slide_meta.n_t, nz=slide_meta.n_z)
    current_ome_xml_str = slide_meta.original_xml
    is_rgb = slide_meta.is_rgb
    series = slide_meta.series

    if isinstance(img, pyvips.Image):
        bf_dtype = vips2bf_dtype(img.format)
    else:
        bf_dtype = slide_tools.NUMPY_FORMAT_BF_DTYPE[str(img.dtype)]

    if channel_names is None:
        channel_names = slide_meta.channel_names
    updated_channel_names = check_channel_names(channel_names, is_rgb, nc=nc)

    if pixel_physical_size_xyu is None:
        if slide_meta.pixel_physical_size_xyu[2] == PIXEL_UNIT:
            pixel_physical_size_xyu = None
        else:
            pixel_physical_size_xyu = reader.scale_physical_size(level)

    if not is_rgb:
        if isinstance(colormap, str) and colormap == CMAP_AUTO:
            colormap = get_colormap(updated_channel_names, is_rgb=is_rgb, series=series, original_xml=current_ome_xml_str)

        colormap = check_colormap(colormap, channel_names=updated_channel_names)

    og_valid_xml = True
    og_ome = None
    if current_ome_xml_str is not None:
        try:
            elementTree.fromstring(current_ome_xml_str)
            og_ome = get_ome_obj(current_ome_xml_str)
        except elementTree.ParseError as e:
            traceback_msg = traceback.format_exc()
            msg = "xml in original file is invalid or missing. Will create one"
            valtils.print_warning(msg, traceback_msg=traceback_msg)
            og_valid_xml = False

    else:
        og_valid_xml = False

    temp_new_ome = create_ome_xml(shape_xyzct=new_xyzct, bf_dtype=bf_dtype, is_rgb=is_rgb,
                                  pixel_physical_size_xyu=pixel_physical_size_xyu,
                                  channel_names=updated_channel_names, colormap=colormap)

    if og_valid_xml and og_ome is not None:
        new_ome = og_ome.copy()
        new_ome.images = temp_new_ome.images
    else:
        new_ome = temp_new_ome

    return new_ome


@valtils.deprecated_args(perceputally_uniform_channel_colors="colormap")
def warp_and_save_slide(src_f, dst_f, transformation_src_shape_rc, transformation_dst_shape_rc,
                        aligned_slide_shape_rc, M=None, dxdy=None,
                        level=0, series=None, interp_method="bicubic",
                        bbox_xywh=None, bg_color=None, colormap=None, channel_names=None,
                        tile_wh=None, compression=DEFAULT_COMPRESSION, Q=100, pyramid=True, reader=None):

    """ Warp and save a slide

    Warp slide according to `M` and/or `dxdy`, then save as an ome.tiff image.

    Parameters
    ----------
    src_f : str
        Path to slide

    dst_f : str
        Path indicating where the warped slide will be saved.

    transformation_src_shape_rc : (int, int)
        Shape of the image used to find the rigid transformations (row, col)

    transformation_dst_shape_rc : (int, int)
        Shape of image with shape `in_shape_rc`, after being warped,
        i.e. the shape of the registered image.

    aligned_slide_shape_rc : (int, int)
        Shape of the warped slide (row, col)

    M : ndarray, optional
        3x3 Affine transformation matrix to perform rigid warp.
        Found by aligning the target/fixed image to source/moving image.
        If `M` was found the other way around, then `M` will need to be inverted
        using np.linalg.inv()

    dxdy : list, optional
        A list containing the x-axis (column) displacement and y-axis (row) displacements.
        Found by aligning the target/fixed image to source/moving image.
        If `dxdy` was found the other way around, then `dxdy` will need to be inverted,
        which can be done using `warp_tools.get_inverse_field`

    level : int, optional
        Pyramid level to warp an save

    series : int, optional
        Series number of image

    interp_method : str, optional
        Interpolation method

    bbox_xywh : tuple
        Bounding box to crop warped slide. Should be in reference the
        warped slide.

    bg_color : optional, list
        Background color, if `None`, then the background color will be black

    colormap : dict, optional
        Dictionary of channel colors, where the key is the channel name, and the value the color as rgb255.
        If "auto" (default), the channel colors from `current_ome_xml_str` will be used, if available.
        If `None`, no channel colors will be added.

    tile_wh : int, optional
        Tile width and height used to save image

    compression : str
        Compression method used to save ome.tiff .
        be jpeg or jp2k. See https://libvips.github.io/pyvips/enums.html#pyvips.enums.ForeignTiffCompression for more details.

    Q : int
        Q factor for lossy compression

    pyramid : bool
        Whether or not to save an image pyramid.

    reader: SlideReader, optional
        Instantiated SlideReader to use to read image. If `None`,
        `get_slide_reader` will be used to find the appropriate reader.
    """

    warped_slide = slide_tools.warp_slide(src_f=src_f,
                                          transformation_src_shape_rc=transformation_src_shape_rc,
                                          transformation_dst_shape_rc=transformation_dst_shape_rc,
                                          aligned_slide_shape_rc=aligned_slide_shape_rc,
                                          M=M,
                                          dxdy=dxdy,
                                          level=level,
                                          series=series,
                                          interp_method=interp_method,
                                          bbox_xywh=bbox_xywh,
                                          bg_color=bg_color,
                                          reader=reader)

    # Get OMEXML and update with new dimensions
    if reader is None:
        reader_cls = get_slide_reader(src_f, series=series) # Get slide reader class
        reader = reader_cls(src_f, series=series) # Get reader

    ome_xml_obj = update_xml_for_new_img(img=warped_slide,
                                         reader=reader,
                                         level=level,
                                         channel_names=channel_names,
                                         colormap=colormap)

    ome_xml = ome_xml_obj.to_xml()

    out_shape_wh = warp_tools.get_shape(warped_slide)[0:2][::-1]
    tile_wh = get_tile_wh(reader=reader,
                          level=level,
                          out_shape_wh=out_shape_wh)

    save_ome_tiff(warped_slide, dst_f=dst_f, ome_xml=ome_xml,
                  tile_wh=tile_wh, compression=compression, Q=Q, pyramid=pyramid)


def save_ome_tiff(img, dst_f, ome_xml=None, tile_wh=512, compression=DEFAULT_COMPRESSION, Q=100, pyramid=True):
    """Save an image in the ome.tiff format using pyvips

    Parameters
    ---------
    img : pyvips.Image, ndarray
        Image to be saved. If a numpy array is provided, it will be converted
        to a pyvips.Image.

    ome_xml : str, optional
        ome-xml string describing image's metadata. If None, it will be created

    tile_wh : int
        Tile shape used to save `img`. Used to create a square tile, so `tile_wh`
        is both the width and height.

    compression : str
        Compression method used to save ome.tiff . See pyips for more details.

    Q : int
        Q factor for lossy compression

    pyramid : bool
        Whether or not to save an image pyramid.

    """
    compression = compression.lower()

    dst_dir = os.path.split(dst_f)[0]
    pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)

    if not isinstance(img, pyvips.vimage.Image):
        img = slide_tools.numpy2vips(img)

    if img.format in ["float", "double"] and compression in [pyvips.enums.ForeignTiffCompression.JP2K, pyvips.enums.ForeignTiffCompression.JPEG]:
        msg = f"Image has type {img.format} but compression method {compression} will convert image to uint8. To avoid this, change compression 'lzw', 'deflate', or 'none' "
        valtils.print_warning(msg)
        if compression == "jp2k":
            compression = "jpeg"
            msg = f"Float images can't be saved using {compression} compression. Please change to another method, such as 'lzw', 'deflate', or 'none' "
            valtils.print_warning(msg)

            return None

    dst_f_extension = slide_tools.get_slide_extension(dst_f)
    if dst_f_extension != ".ome.tiff":
        dst_dir, out_f = os.path.split(dst_f)
        new_out_f = out_f.split(dst_f_extension)[0] + ".ome.tiff"
        new_dst_f = os.path.join(dst_dir, new_out_f)
        msg = f"{out_f} is not an ome.tiff. Changing dst_f to {new_dst_f}"
        valtils.print_warning(msg)
        dst_f = new_dst_f

    # Get ome-xml metadata #
    xyzct = get_shape_xyzct((img.width, img.height), img.bands)

    og_interpretation = img.interpretation
    is_rgb = og_interpretation in VIPS_RGB_FORMATS

    bf_dtype = vips2bf_dtype(img.format)
    if ome_xml is None:
        # Create minimal ome-xml
        ome_xml_obj = create_ome_xml(shape_xyzct=xyzct, bf_dtype=bf_dtype, is_rgb=is_rgb)
    else:
        # Verify that vips image and ome-xml match
        ome_xml_obj = get_ome_obj(ome_xml)
        ome_img = ome_xml_obj.images[0].pixels
        total_pages = ome_img.size_c*ome_img.size_z*ome_img.size_t

        match_dict = {"same_x": ome_img.size_x == img.width,
                      "same_y": ome_img.size_y == img.height,
                      "total_pages": total_pages == img.bands,
                      "same_type": ome_img.type.name.lower() == bf_dtype
                      }

        if not all(list(match_dict.values())):
            msg = f"mismatch in ome-xml and image: {str(match_dict)}. Will create ome-xml"
            valtils.print_warning(msg)
            ome_xml_obj = create_ome_xml(xyzct, bf_dtype, is_rgb)

    ome_xml_obj.creator = f"pyvips version {pyvips.__version__}"
    ome_metadata = ome_xml_obj.to_xml()

    # Save ome-tiff using vips #
    image_height = img.height
    image_bands = img.bands
    if is_rgb:
        img = img.copy(interpretation=og_interpretation)
    else:
        img = pyvips.Image.arrayjoin(img.bandsplit(), across=1)
        img = img.copy(interpretation="b-w")

    img.set_type(pyvips.GValue.gint_type, "page-height", image_height)
    img.set_type(pyvips.GValue.gstr_type, "image-description", ome_metadata)

    # Set up progress bar #
    bar_len = 100
    if is_rgb:
        total = 100
    else:
        total = 100*image_bands
    tic = time.time()

    save_ome_tiff.n_complete = -1
    save_ome_tiff.current_im = None
    def eval_handler(im, progress):
        if save_ome_tiff.current_im != progress.im:
            save_ome_tiff.n_complete += 1
        save_ome_tiff.current_im = progress.im
        count = save_ome_tiff.n_complete*100 + progress.percent
        filled_len = int(round(bar_len * count / float(total)))
        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        toc = time.time()
        processing_time_h = round((toc - tic)/(60), 3)

        sys.stdout.write('[%s] %s%s %s %s %s\r' % (bar, percents, '%', 'in', processing_time_h, "minutes"))
        sys.stdout.flush()

    try:
        img.set_progress(True)
        img.signal_connect("eval", eval_handler)
    except pyvips.error.Error:
        msg = "Unable to create progress bar for pyvips. May need to update libvips to >= 8.11"
        valtils.print_warning(msg)

    print(f"saving {dst_f} ({img.width} x {image_height} and {image_bands} channels)")

    # Write image #
    tile_wh = tile_wh - (tile_wh % 16)  # Tile shape must be multiple of 16

    if min(xyzct[0:2]) < tile_wh:
        tile = False
    else:
        tile = True

    if np.any(np.array(xyzct[0:2]) < tile_wh):
        # Image is smaller than the tile #
        min_dim = min(xyzct[0:2])
        tile_wh = int((min_dim - min_dim % 16))

    if tile_wh < 16:
        tile_wh = 16

    print("")

    lossless = Q == 100
    rgbjpeg = compression in ["jp2k", "jpeg"] and img.interpretation in VIPS_RGB_FORMATS
    subifd = pyramid # a pyramid will still be created if subifd = True and pyramid = False
    img.tiffsave(dst_f, compression=compression, tile=tile,
                 tile_width=tile_wh, tile_height=tile_wh,
                 pyramid=pyramid, subifd=subifd, bigtiff=True,
                 lossless=lossless, Q=Q, rgbjpeg=rgbjpeg)

    # Print total time to completion #
    toc = time.time()
    processing_time_seconds = toc-tic
    processing_time, processing_time_unit = valtils.get_elapsed_time_string(processing_time_seconds)

    bar = '=' * bar_len
    sys.stdout.write('[%s] %s%s %s %s %s\r' % (bar, 100.0, '%', 'in', processing_time, processing_time_unit))
    sys.stdout.flush()
    sys.stdout.write('\nComplete\n')
    print("")


@valtils.deprecated_args(perceputally_uniform_channel_colors="colormap")
def convert_to_ome_tiff(src_f, dst_f, level, series=None, xywh=None,
                        colormap=CMAP_AUTO, tile_wh=None, compression=DEFAULT_COMPRESSION, Q=100, pyramid=True, reader=None):
    """Convert an image to an ome.tiff image

    Saves a new copy of the image as a tiled pyramid ome.tiff with valid ome-xml.
    Uses pyvips to save the image. Currently only writes a single series.

    Parameters
    ----------
    src_f : str
        Path to image to be converted

    dst_f : str
        Path indicating where the image should be saved.

    level : int
        Pyramid level to be converted.

    series : str
        Series to be converted.

    xywh : tuple of int, optional
        The region of the slide to be converted. If None,
        then the entire slide will be converted. Otherwise
        xywh is the (top left x, top left y, width, height) of
        the region to be sliced.

    colormap : dict, optional
        Dictionary of channel colors, where the key is the channel name, and the value the color as rgb255.
        If None (default), the channel colors from `current_ome_xml_str` will be used, if available.
        If None, and there are no channel colors in the `current_ome_xml_str`, then no colors will be added

    tile_wh : int
        Tile shape used to save the image. Used to create a square tile,
        so `tile_wh` is both the width and height.

    compression : str
        Compression method used to save ome.tiff. See pyips for more details.

    Q : int
        Q factor for lossy compression

    pyramid : bool
        Whether or not to save an image pyramid.

    """

    dst_dir = os.path.join(dst_f)[0]
    pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)

    if reader is None:
        reader_cls = get_slide_reader(src_f, series)
        reader = reader_cls(src_f, series=series)
    slide_meta = reader.metadata
    if series is None:
        series = reader.metadata.series

    vips_img = reader.slide2vips(level=level, series=series, xywh=xywh)

    ome_obj = update_xml_for_new_img(img=vips_img,
                                     reader=reader,
                                     level=level,
                                     channel_names=slide_meta.channel_names,
                                     colormap=colormap)

    ome_obj.creator = f"pyvips version {pyvips.__version__}"
    ome_xml_str = ome_obj.to_xml()
    if tile_wh is None:
        tile_wh = slide_meta.optimal_tile_wh

    if tile_wh > MAX_TILE_SIZE:
        tile_wh = MAX_TILE_SIZE

    save_ome_tiff(img=vips_img, dst_f=dst_f, ome_xml=ome_xml_str, tile_wh=tile_wh, compression=compression, Q=Q, pyramid=pyramid)
