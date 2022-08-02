"""Methods and classes to read and write slides in the .ome.tiff format

"""

from csv import excel
import os
from skimage import util, io, transform
import pyvips
import numpy as np
from PIL import Image
import pathlib
import re
import multiprocessing
from joblib import Parallel, delayed, parallel_backend
import imghdr
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
import bioformats_jar
from tqdm import tqdm
import scyjava

from . import valtils
from . import slide_tools
from . import warp_tools

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

PIXEL_UNIT = "px"
"""str: Physical unit when the unit can't be found in the metadata"""

MICRON_UNIT = u'\u00B5m'
"""str: Phyiscal unit for micron/micrometers"""

ALL_OPENSLIDE_READABLE_FORMATS = [".svs", ".tif", ".vms", ".vmu", ".ndpi", ".scn", ".mrxs", ".tiff", ".svslide", ".bif"]
"""list: File extensions that OpenSlide can read"""

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

        if jar is not None:
            jpype.addClassPath(jar)
        elif jar is None and hasattr(bioformats_jar, "JAR"):
            jpype.addClassPath(bioformats_jar.JAR)
        else:
            scyjava.config.endpoints.append('ome:formats-gpl')

        try:
            # jpype.startJVM(f"-Djava.awt.headless=true -Xmx{mem_gb}G", classpath=bioformats_jar.JAR)
            scyjava.start_jvm([f"-Xmx{mem_gb}G"])
        except ModuleNotFoundError:
            bioformats_jar.start_jvm(memory=f"{mem_gb}G")

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
        jpype.shutdownJVM()
        msg = "JVM has been killed. If this was due to an error, then a new Python session will need to be started"
        valtils.print_warning(msg, warning_type=None, rgb=valtils.Fore.GREEN)

    except NameError:
        pass


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
        dtype = np.int8
        scale = 255

    elif bf_pixel_type == FormatTools.UINT8:
        dtype = np.uint8
        scale = 255

    elif bf_pixel_type == FormatTools.UINT16:
        dtype = '<u2' if little_endian else '>u2'
        scale = 65535

    elif bf_pixel_type == FormatTools.INT16:
        dtype = '<i2' if little_endian else '>i2'
        scale = 65535

    elif bf_pixel_type == FormatTools.UINT32:
        dtype = '<u4' if little_endian else '>u4'
        scale = 2**32

    elif bf_pixel_type == FormatTools.INT32:
        dtype = '<i4' if little_endian else '>i4'
        scale = 2**32-1

    elif bf_pixel_type == FormatTools.FLOAT:
        dtype = '<f4' if little_endian else '>f4'
        scale = 1

    elif bf_pixel_type == FormatTools.DOUBLE:
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
            valtils.print_warning(e)

    return use_openslide


def check_flattened_pyramid_tiff(src_f):
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

    if 'n-pages' in vips_fields:
        n_pages = vips_img.get("n-pages")
        all_areas = [None] * n_pages
        all_dims = [None] * n_pages
        all_n_channels = [None] * n_pages
        level_starts = []
        prev_area = None
        for i in range(n_pages):
            page = pyvips.Image.new_from_file(src_f, page=i)

            w = page.width
            h = page.height
            nc = page.bands
            img_area = w*h*nc

            all_areas[i] = img_area
            all_dims[i] = [w, h]
            all_n_channels[i] = nc

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


    # Now check if Bioformats reads it similarly #
    with valtils.HiddenPrints():
        bf_reader = BioFormatsSlideReader(src_f)
    bf_levels = len(bf_reader.metadata.slide_dimensions)
    bf_channels = bf_reader.metadata.n_channels
    can_use_bf = bf_levels >= len(slide_dimensions) and bf_channels == n_channels

    return is_flattended_pyramid, can_use_bf, slide_dimensions, levels_start_idx, n_channels


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
        List of channel names.

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
        self.original_xml = None
        self.bf_datatype = None
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

        self.src_f = src_f
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
            matching_channels = [i for i in range(self.metadata.n_channels) if
                                 re.search(channel.lower(), self.metadata.channel_names[i].lower())
                                 is not None]

            if len(matching_channels) == 0:
                msg = f"Cannot find channel '{channel}' in {self.src_f}. Using channel 0"
                valtils.print_warning(msg)
                matching_channel_idx = 0

            elif len(matching_channels) > 1:
                all_matching_channels = ", ".join([f"'{self.metadata.channel_names[i]}'" for i in matching_channels])
                msg = f"Fount multiple channels that match '{channel}' in {self.src_f}. These are: {all_matching_channels}. Using channel 0"
                valtils.print_warning(msg)
                matching_channel_idx = 0

            else:
                matching_channel_idx = matching_channels[0]

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
            print(e)
            kill_jvm()

        self.n_series = len(self.meta_list)
        if series is None:
            img_areas = [np.multiply(*meta.slide_dimensions[0]) for meta in self.meta_list]
            series = np.argmax(img_areas)
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

    def get_tiles_parallel(self, level, tile_bbox_list, pixel_type, series=0):
        """Get tiles to slice from the slide

        """

        n_tiles = len(tile_bbox_list)
        tile_array = [None] * n_tiles

        def tile2vips_threaded(idx):
            xywh = tile_bbox_list[idx]
            # javabridge.attach()
            jpype.attachThreadToJVM()
            tile = self.slide2image(level, series, xywh=tuple(xywh))
            # javabridge.detach()
            jpype.detachThreadFromJVM()

            # I don't want uint16 images to be converted to uint8!
            # commenting this out
            # if np.issubdtype(pixel_type, np.unsignedinteger):
            #     tile = util.img_as_ubyte(tile)

            # does it matter that the pyvips interpretation in srgb, not multiband for IF images?
            tile_array[idx] = slide_tools.numpy2vips(tile, self.metadata.pyvips_interpretation)

        n_cpu = multiprocessing.cpu_count() - 1
        with parallel_backend("threading", n_jobs=n_cpu):
            Parallel()(delayed(tile2vips_threaded)(i) for i in tqdm(range(n_tiles)))

        return tile_array

    def slide2vips(self, level, series=None, xywh=None, tile_wh=None, *args, **kwargs):
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

        tile_wh = MAX_TILE_SIZE
        if np.any(slide_shape_wh < tile_wh):
            tile_wh = min(slide_shape_wh)

        tile_bbox = warp_tools.get_grid_bboxes(slide_shape_wh[::-1],
                                               tile_wh, tile_wh, inclusive=True)

        n_across = len(np.unique(tile_bbox[:, 0]))

        print(f"Converting slide to pyvips image")
        vips_slide = pyvips.Image.arrayjoin(
                                  self.get_tiles_parallel(level, tile_bbox, pixel_type, series),
                                  across=n_across).crop(0, 0, *slide_shape_wh)
        if xywh is not None:
            vips_slide = vips_slide.extract_area(*xywh)

        return vips_slide

    def slide2image(self, level, series=None, xywh=None, *args, **kwargs):
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
            img = self._read_rgb(rdr, xywh)

        else:
            img = self._read_multichannel(rdr, xywh)

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
            print(e)
            rdr.close()

        return meta_list

    def _read_rgb(self, rdr, xywh):

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

    def _read_multichannel(self, rdr, xywh):
        _, _, w, h = xywh
        n_channels = rdr.getSizeC()
        np_dtype, drange = bf_to_numpy_dtype(rdr.getPixelType(),
                                             rdr.isLittleEndian())

        if n_channels > 1:
            img = np.zeros((h, w, n_channels), dtype=np_dtype)
        else:
            img = None

        for i in range(n_channels):
            idx = rdr.getIndex(0, i, 0)  # ZCT
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

        rdr = loci.formats.ImageReader()
        factory = loci.common.services.ServiceFactory()
        OMEXMLService_class = loci.formats.services.OMEXMLService

        service = factory.getInstance(OMEXMLService_class)
        ome_meta = service.createOMEXMLMetadata()
        rdr.setMetadataStore(ome_meta)
        rdr.setFlattenedResolutions(False)
        rdr.setId(self.src_f)
        meta = rdr.getMetadataStore()

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
        self.is_ome = False
        self.metadata = self.create_metadata()

    def create_metadata(self):

        if self.use_openslide:
            server = OPENSLIDE_RDR
        else:
            server = VIPS_RDR

        meta_name = f"{os.path.split(self.src_f)[1]}_Series(0)".strip("_")
        f_extension = slide_tools.get_slide_extension(self.src_f)
        if f_extension in BF_READABLE_FORMATS:
            with valtils.HiddenPrints():
                bf_reader = BioFormatsSlideReader(self.src_f)

            self.is_ome = re.search("ome-tiff", bf_reader.metadata.server.lower()) is not None

        slide_meta = MetaData(meta_name, server)
        vips_img = pyvips.Image.new_from_file(self.src_f)

        slide_meta.is_rgb = self._check_rgb(vips_img)
        # Doesn't work for toilet roll images. needs to be updated based on chopping.
        slide_meta.n_channels = vips_img.bands
        if (slide_meta.is_rgb and vips_img.hasalpha() >= 1) or self.use_openslide:
            # Will remove alpha channel after reading
            slide_meta.n_channels = vips_img.bands - vips_img.hasalpha()

        slide_meta.slide_dimensions = self._get_slide_dimensions(vips_img)
        if f_extension in BF_READABLE_FORMATS:
            with valtils.HiddenPrints():
                bf_reader = BioFormatsSlideReader(self.src_f)

            slide_meta.channel_names = bf_reader.metadata.channel_names
            # Need to update the n_channels based on bioformats metadata if toilet roll .ome.tiff
            # Not sure if this will cause problems for other image formats...
            slide_meta.n_channels = len(slide_meta.channel_names)
            slide_meta.pixel_physical_size_xyu = bf_reader.metadata.pixel_physical_size_xyu
            slide_meta.bf_pixel_type = bf_reader.metadata.bf_pixel_type
            slide_meta.is_little_endian = bf_reader.metadata.is_little_endian
            slide_meta.original_xml = bf_reader.metadata.original_xml
            slide_meta.bf_datatype = bf_reader.metadata.bf_datatype
            slide_meta.optimal_tile_wh = bf_reader.metadata.optimal_tile_wh
        else:
            slide_meta.pixel_physical_size_xyu = self._get_pixel_physical_size(vips_img)


        if slide_meta.is_rgb:
            slide_meta.channel_names = None

        return slide_meta

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
        if page.interpretation == "srgb":
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
                # need to modify this slide metadata, assuming that all bands are channels
                # if it was a bioformats compatible file (which .ome.tif is), the metadata.channel_names should be correct
                # do not know if there is a case where the channel_names would not be updated correctly
                # This is too late as the processing script only uses the initialized slide metadata for selecting channels based on channel names...
                self.metadata.n_channels = vips_slide.bands

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

        Parameters
        ----------
        vips_img : pyvips.Image
            pyvips.Image of slide.

        Returns
        -------
        is_rgb : bool
            Whether or not the image is RGB.

        """

        return vips_img.interpretation == "srgb"

    def _get_channel_names(self, vips_img):
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
        if 'n-pages' in vips_fields:
            n_pages = vips_img.get("n-pages")
            channel_names = []
            for i in range(n_pages):
                page = pyvips.Image.new_from_file(self.src_f, page=i)
                page_metadata = page.get("image-description")

                page_soup = BeautifulSoup(page_metadata, features="lxml")
                cname = page_soup.find("name")
                if cname is not None:
                    if cname.text not in channel_names:
                        channel_names.append(cname.text)

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

    def _get_slide_dimensions_ometiff(self, *args):
        with valtils.HiddenPrints():
            bf_reader = BioFormatsSlideReader(self.src_f)

        return np.array(bf_reader.metadata.slide_dimensions)

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
            all_dims = [None] * n_pages
            all_channels = [None] * n_pages
            for i in range(n_pages):
                page = pyvips.Image.new_from_file(self.src_f, page=i)

                w = page.width
                h = page.height
                c = page.bands

                all_dims[i] = [w, h]
                all_channels[i] = c

            all_dims = np.array(all_dims)
            most_common_channel_count = stats.mode(all_channels)[0][0]
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
        f_extension = slide_tools.get_slide_extension(self.src_f)
        if f_extension in BF_READABLE_FORMATS:
            with valtils.HiddenPrints():
                bf_reader = BioFormatsSlideReader(self.src_f)

            channel_names = bf_reader.metadata.channel_names
            slide_meta.pixel_physical_size_xyu = bf_reader.metadata.pixel_physical_size_xyu
            slide_meta.is_little_endian = bf_reader.metadata.is_little_endian
            slide_meta.original_xml = bf_reader.metadata.original_xml
            slide_meta.optimal_tile_wh = bf_reader.metadata.optimal_tile_wh
            slide_meta.bf_datatype = bf_reader.metadata.bf_datatype

        if len(channel_names) != n_channels:
            channel_names = self._get_channel_names(vips_img, n_channels)

        slide_meta.channel_names = channel_names

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

    def _get_channel_names(self, vips_img, n_channels):
        vips_fields = vips_img.get_fields()
        if 'n-pages' in vips_fields:
            page = pyvips.Image.new_from_file(self.src_f, page=0)
            page_metadata = page.get("image-description")

            page_soup = BeautifulSoup(page_metadata, features="lxml")
            channels = page_soup.findAll("channel")
            if len(channels) == 0:
                channel_names = [f"C{i}" for i in range(n_channels)]
            else:
                channel_names = [None] * len(channels)
                for cidx, chnl in enumerate(channels):
                    if chnl.has_attr("name"):
                        channel_names[cidx] = chnl["name"]
                    else:
                        channel_names[cidx] = f"C{cidx}"

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
    because it can be opened as a pyvips.Image. More common formats, like png, jpeg,
    etc... will be opened with scikit-image. Everything else will be opened with
    Bioformats.

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

    init_jvm()

    f_extension = slide_tools.get_slide_extension(src_f)
    what_img = imghdr.what(src_f)
    can_use_openslide = check_to_use_openslide(src_f)
    can_only_use_openslide = f_extension in OPENSLIDE_ONLY
    if can_only_use_openslide and not can_use_openslide:
        msg = (f"file {os.path.split(src_f)[1]} can only be read by OpenSlide, "
               f"which is required to open files with the follwing extensions: {', '.join(OPENSLIDE_ONLY)}."
               f"However, OpenSlide cannot be found. Unable to read this slide."
               )

        valtils.print_warning(msg)

    can_use_bf = f_extension in BF_READABLE_FORMATS and not can_only_use_openslide
    is_tiff = f_extension == ".tiff" or f_extension == ".tif"
    can_use_skimage = ".".join(f_extension.split(".")[1:]) == what_img and not is_tiff
    try:
        pyvips.Image.new_from_file(src_f)
        can_use_pyvips = True
    except:
        can_use_pyvips = False

    fail_msg = f"Can't find reader to open {os.path.split(src_f)[1]}. May need to create a new one by subclassing SlideReader. Returning None"
    if not can_use_openslide and not can_use_bf and not can_use_skimage and not can_use_pyvips:
        valtils.print_warning(fail_msg)

        return None

    if can_use_skimage and not can_use_pyvips:
        reader = ImageReader
        return reader

    n_series = 1
    is_rgb = None
    if can_use_bf:
        with valtils.HiddenPrints():
            bf_reader = BioFormatsSlideReader(src_f)

        n_series = bf_reader.n_series
        is_ometiff = re.search("ome-tiff", bf_reader.metadata.server.lower()) is not None
        is_rgb = bf_reader.metadata.is_rgb

        if series is None:
            series = bf_reader.series
    else:
        is_ometiff = False

    if series is None:
        series = 0

    if is_tiff and not is_ometiff:
        is_flattened_tiff, bf_reads_flat = check_flattened_pyramid_tiff(src_f)[0:2]

    else:
        is_flattened_tiff = False

    if is_flattened_tiff:
        if not bf_reads_flat:
            reader = FlattenedPyramidReader
        else:
            reader = BioFormatsSlideReader

    elif can_only_use_openslide:
        # E.g. .mrxs
        reader = VipsSlideReader

    elif is_ometiff:
        if series == 0 and n_series == 1 and is_rgb and is_rgb is not None:
            # Seems pvips can only read ome.tiff if there is 1 series.
            # Converting a multichannel pyvips.Image is very slow, but is fast for RGB
            reader = VipsSlideReader
        else:
            reader = BioFormatsSlideReader

    elif can_use_pyvips:
        # Use pyvips to open regular formats, like png, jpeg, bmp, etc...
        reader = VipsSlideReader

    elif can_use_bf:
        # Use Bioformats for images that can't be read by pyvips or skimage
        reader = BioFormatsSlideReader

    else:
        valtils.print_warning(fail_msg)
        reader = None

    return reader


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


def get_shape_xyzct(shape_wh, n_channels):
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

    xyzct = (*shape_wh, 1, n_channels, 1)
    return xyzct


def create_channel(channel_id, name=None, color=None):
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
    if name is not None:
        new_channel.name = decoded_name
    if color is not None:
        new_channel.color = tuple([*color, 1])

    return new_channel


def create_ome_xml(shape_xyzct, bf_dtype, is_rgb, pixel_physical_size_xyu=None, channel_names=None, perceputally_uniform_channel_colors=False):
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

    perceputally_uniform_channel_colors : bool
        Whether or not to add perceptually uniform channel colors.

    Returns
    -------
    new_ome : ome_types.model.OME
        ome_types.model.OME object containing ome-xml metadata

    """

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
        if channel_names is not None:
            default_channel_colors = slide_tools.turbo_channel_colors(c)
            channels = [create_channel(i, name=channel_names[i], color=default_channel_colors[i]) for i in range(c)]
            new_img.pixels.channels = channels

        if perceputally_uniform_channel_colors:
            channel_colors = slide_tools.perceptually_uniform_channel_colors(c)
            if len(new_img.pixels.channels) == 0:
                channels = [create_channel(i, color=channel_colors[i]) for i in range(c)]
            else:
                channels = new_img.pixels.channels
                for i, clr in enumerate(channel_colors):
                    channels[i].color = (*clr, 1)

    new_ome = ome_types.model.OME()
    new_ome.images.append(new_img)

    return new_ome


def update_xml_for_new_img(current_ome_xml_str, new_xyzct, bf_dtype, is_rgb, pixel_physical_size_xyu=None, channel_names=None, perceputally_uniform_channel_colors=False):
    """Update dimensions ome-xml metadata

    Used to create a new ome-xml that reflects changes in an image, such as its shape

    If `current_ome_xml_str` is invalid or None, a new ome-xml will be created

    Parameters
    -------
    current_ome_xml_str : str
        ome-xml string that needs to be updated

    new_xyzct : tuple of int
        XYZCT shape of image

    bf_dtype : str
        String format of Bioformats datatype

    is_rgb : bool
        Whether or not the image is RGB

    pixel_physical_size_xyu : tuple, optional
        Physical size per pixel and the unit.

    channel_names : list, optional
        List of channel names.

    perceputally_uniform_channel_colors : bool
        Whether or not to add perceptually uniform channel colors

    Returns
    -------
    new_ome : ome_types.model.OME
        ome_types.model.OME object containing ome-xml metadata

    """

    temp_new_ome = create_ome_xml(new_xyzct, bf_dtype, is_rgb, pixel_physical_size_xyu, channel_names, perceputally_uniform_channel_colors)
    og_valid_xml = True
    if current_ome_xml_str is not None:
        try:
            elementTree.fromstring(current_ome_xml_str)
        except elementTree.ParseError as e:
            print(e)
            msg = "xml in original file is invalid or missing. Will create one"
            valtils.print_warning(msg)
            og_valid_xml = False

    else:
        og_valid_xml = False

    if og_valid_xml:
        new_ome = ome_types.from_xml(current_ome_xml_str)
        new_ome.images = temp_new_ome.images
    else:
        new_ome = temp_new_ome

    return new_ome


def warp_and_save_slide(src_f, dst_f, transformation_src_shape_rc, transformation_dst_shape_rc,
                        aligned_slide_shape_rc, M=None, dxdy=None,
                        level=0, series=None, interp_method="bicubic",
                        bbox_xywh=None, bg_color=None, perceputally_uniform_channel_colors=False,
                        tile_wh=None, compression="lzw"):

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

    perceputally_uniform_channel_colors : bool, optional
        Whether or not to add perceptually uniform channel colors to non-RGB images

    tile_wh : int, optional
        Tile width and height used to save image

    compression : str
        Compression method used to save ome.tiff . Default is lzw, but can also
        be jpeg or jp2k. See pyips for more details.

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
                                          bg_color=bg_color)

    # Get OMEXML and update with new dimensions
    reader_cls = get_slide_reader(src_f, series=series) # Get slide reader class
    reader = reader_cls(src_f, series=series) # Get reader
    slide_meta = reader.metadata
    if slide_meta.pixel_physical_size_xyu[2] == PIXEL_UNIT:
        px_phys_size = None
    else:
        px_phys_size = reader.scale_physical_size(level)

    bf_dtype = vips2bf_dtype(warped_slide.format)
    out_xyczt = get_shape_xyzct((warped_slide.width, warped_slide.height), warped_slide.bands)
    ome_xml_obj = update_xml_for_new_img(slide_meta.original_xml,
                                         new_xyzct=out_xyczt,
                                         bf_dtype=bf_dtype,
                                         is_rgb=reader.metadata.is_rgb,
                                         pixel_physical_size_xyu=px_phys_size,
                                         channel_names=reader.metadata.channel_names,
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

    save_ome_tiff(warped_slide, dst_f=dst_f, ome_xml=ome_xml,
                  tile_wh=tile_wh, compression=compression)


def save_ome_tiff(img, dst_f, ome_xml=None, tile_wh=1024, compression="lzw"):
    """Save an image in the ome.tiff format using pyvips

    Parameters
    ---------
    img : pyvips.Image, ndarray
        Image to be saved. If a numpy array is provided, it will be converted
        to a pyvips.Image.

    ome_xml : str, optional
        ome-xml string describing image's metadata. If None, it will be createdd

    tile_wh : int
        Tile shape used to save `img`. Used to create a square tile, so `tile_wh`
        is both the width and height.

    compression : str
        Compression method used to save ome.tiff . Default is lzw, but can also
        be jpeg or jp2k. See pyips for more details.

    """

    dst_dir = os.path.split(dst_f)[0]
    pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)

    if not isinstance(img, pyvips.vimage.Image):
        img = slide_tools.numpy2vips(img)

    if img.format in ["float", "double"] and compression != "lzw":
        msg = f"Image has type {img.format} but compression method {compression} will convert image to uint8. To avoid this, compression is being changed to 'lzw"
        compression = "lzw"

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
    is_rgb = img.interpretation == "srgb"
    bf_dtype = vips2bf_dtype(img.format)
    if ome_xml is None:
        # Create minimal ome-xml
        ome_xml_obj = create_ome_xml(xyzct, bf_dtype, is_rgb)
    else:
        # Verify that vips image and ome-xml match
        ome_xml_obj = ome_types.from_xml(ome_xml)
        ome_img = ome_xml_obj.images[0].pixels
        match_dict = {"same_x": ome_img.size_x == img.width,
                      "same_y": ome_img.size_y == img.height,
                      "same_c": ome_img.size_c == img.bands,
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
        img = img.copy(interpretation="srgb")
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
    if np.any(np.array(xyzct[0:2]) < tile_wh):
        # Image is smaller than the tile #
        min_dim = min(xyzct[0:2])
        tile_wh = int((min_dim - min_dim %16))
    if tile_wh < 16:
        tile_wh = 16

    print("")
    img.tiffsave(dst_f, compression=compression, tile=True,
                 tile_width=tile_wh, tile_height=tile_wh,
                 pyramid=True, subifd=True, bigtiff=True, lossless=True)

    # Print total time to completion #
    toc = time.time()
    processing_time_seconds = toc-tic
    processing_time, processing_time_unit = valtils.get_elapsed_time_string(processing_time_seconds)

    bar = '=' * bar_len
    sys.stdout.write('[%s] %s%s %s %s %s\r' % (bar, 100.0, '%', 'in', processing_time, processing_time_unit))
    sys.stdout.flush()
    sys.stdout.write('\nComplete\n')
    print("")


def convert_to_ome_tiff(src_f, dst_f, level, series=None, xywh=None,
                        perceputally_uniform_channel_colors=False, tile_wh=None, compression="lzw"):
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

    perceputally_uniform_channel_colors : bool
        Whether or not to add perceptually uniform channel colors

    tile_wh : int
        Tile shape used to save the image. Used to create a square tile,
        so `tile_wh` is both the width and height.

    compression : str
        Compression method used to save ome.tiff . Default is lzw, but can also
        be jpeg or jp2k. See pyips for more details.

    """

    dst_dir = os.path.join(dst_f)[0]
    pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)

    reader_cls = get_slide_reader(src_f, series)
    reader = reader_cls(src_f, series=series)
    slide_meta = reader.metadata
    if series is None:
        series = reader.metadata.series

    vips_img = reader.slide2vips(level=level, series=series, xywh=xywh)
    bf_dtype = vips2bf_dtype(vips_img.format)
    out_xyczt = get_shape_xyzct((vips_img.width, vips_img.height), slide_meta.n_channels)

    if slide_meta.pixel_physical_size_xyu[2] == PIXEL_UNIT:
        px_phys_size = None
    else:
        if level == 0:
            px_phys_size = slide_meta.pixel_physical_size_xyu
        else:
            px_phys_size = reader.scale_physical_size(level)

    ome_obj = update_xml_for_new_img(slide_meta.original_xml,
                                     new_xyzct=out_xyczt,
                                     bf_dtype=bf_dtype,
                                     is_rgb=slide_meta.is_rgb,
                                     pixel_physical_size_xyu=px_phys_size,
                                     channel_names=slide_meta.channel_names,
                                     perceputally_uniform_channel_colors=perceputally_uniform_channel_colors
                                     )

    ome_obj.creator = f"pyvips version {pyvips.__version__}"
    ome_xml_str = ome_obj.to_xml()
    if tile_wh is None:
        tile_wh = slide_meta.optimal_tile_wh

    if tile_wh > MAX_TILE_SIZE:
            tile_wh = MAX_TILE_SIZE

    save_ome_tiff(vips_img, dst_f, ome_xml_str, tile_wh=tile_wh, compression=compression)
