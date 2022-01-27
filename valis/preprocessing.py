"""
Collection of pre-processing methods for aligning images
"""

#import csaps
from scipy.interpolate import Akima1DInterpolator
from skimage import exposure
import numpy as np
import cv2
from skimage import color as skcolor
from . import slide_io
import colour

# DEFAULT_COLOR_STD_C = 0.01 # jzazbz
DEFAULT_COLOR_STD_C = 0.2 # cam16-ucs


class ImageProcesser(object):
    """Process images for registration

    `ImageProcesser` sub-classes processes images to single channel
    images which are then used in image registration.

    Each `ImageProcesser` is initialized with an image, the path to the
    image, the pyramid level, and the series number. These values will
    be set during the registration process.

    `ImageProcesser` must also have a `process_image` method, which is
    called during registration. As `ImageProcesser` has the image and
    and its relevant information (filename, level, series) as attributes,
    it should be able to access and modify the image as needed. However,
    one can also pass extra args and kwargs to `process_image`. As such,
    `process_image` will also need to accept args and kwargs.

    Attributes
    ----------
    image : ndarray
        Image to be processed

    src_f : str
        Path to slide/image.

    level : int
        Pyramid level to be read.

    series : int
        The series to be read.

    """

    def __init__(self, image, src_f, level, series):
        """
        Parameters
        ----------
        image : ndarray
            Image to be processed

        src_f : str
            Path to slide/image.

        level : int
            Pyramid level to be read.

        series : int
            The series to be read.

        """

        self.image = image
        self.src_f = src_f
        self.level = level
        self.series = series

    def process_image(self,  *args, **kwargs):
        """Pre-process image for registration

        Pre-process image for registration. Processed image should
        be a single channel uint8 image.

        Returns
        -------
        processed_img : ndarray
            Single channel processed copy of `image`

        """


class ChannelGetter(ImageProcesser):
    """Select channel from image

    """

    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level,
                         series=series, *args, **kwargs)

    def process_image(self, channel="dapi", adaptive_eq=True, *args, **kwaargs):
        reader_cls = slide_io.get_slide_reader(self.src_f, series=self.series)
        reader = reader_cls(self.src_f)
        chnl = reader.get_channel(channel=channel, level=self.level, series=self.series).astype(float)
        chnl /= chnl.max()
        if adaptive_eq:
            chnl = exposure.equalize_adapthist(chnl)

        chnl = exposure.rescale_intensity(chnl, in_range="image", out_range=(0, 255)).astype(np.uint8)

        return chnl


class ColorfulStandardizer(ImageProcesser):
    """Standardize the colorfulness of the image

    """

    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level,
                         series=series, *args, **kwargs)

    def process_image(self, c=DEFAULT_COLOR_STD_C, invert=True, *args, **kwargs):
        std_rgb = standardize_colorfulness(self.image, c)
        std_g = skcolor.rgb2gray(std_rgb)

        if invert:
            std_g = 255 - std_g
        processed_img = exposure.rescale_intensity(std_g, in_range="image", out_range=(0, 255)).astype(np.uint8)

        return processed_img


class Luminosity(ImageProcesser):
    """Get luminosity of an RGB image

    """

    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level,
                         series=series, *args, **kwargs)

    def process_image(self,  *args, **kwaargs):
        lum = get_luminosity(self.image)
        inv_lum = 255 - lum
        processed_img = exposure.rescale_intensity(inv_lum, in_range="image", out_range=(0, 255)).astype(np.uint8)

        return processed_img


def standardize_colorfulness(img, c=DEFAULT_COLOR_STD_C, h=0):
    """Give image constant colorfulness and hue

    Image is converted to cylindrical CAM-16UCS assigned a constant
    hue and colorfulness, and then coverted back to RGB.

    Parameters
    ----------
    img : ndarray
        Image to be processed
    c : int
        Colorfulness
    h : int
        Hue, in radians (-pi to pi)

    Returns
    -------
    rgb2 : ndarray
        `img` with constant hue and colorfulness

    """
    # Convert to CAM16 #
    eps = np.finfo("float").eps
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        if 1 < img.max() <= 255 and np.issubdtype(img.dtype, np.integer):
            cam = colour.convert(img/255 + eps, 'sRGB', 'CAM16UCS')
        else:
            cam = colour.convert(img + eps, 'sRGB', 'CAM16UCS')

    lum = cam[..., 0]
    cc = np.full_like(lum, c)
    hc = np.full_like(lum, h)
    new_a, new_b = cc * np.cos(hc), cc * np.sin(hc)
    new_cam = np.dstack([lum, new_a+eps, new_b+eps])
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        rgb2 = colour.convert(new_cam, 'CAM16UCS', 'sRGB')
        rgb2 -= eps

    rgb2 = (np.clip(rgb2, 0, 1)*255).astype(np.uint8)

    return rgb2


def get_luminosity(img, **kwargs):
    """Get luminosity of an RGB image
        Converts and RGB image to the CAM16-UCS colorspace, extracts the
        luminosity, and then scales it between 0-255

    Parameters
    ---------
    img : ndarray
        RGB image

    Returns
    -------
    lum : ndarray
        CAM16-UCS luminosity

    """

    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        if 1 < img.max() <= 255 and np.issubdtype(img.dtype, np.integer):
            cam = colour.convert(img/255, 'sRGB', 'CAM16UCS')
        else:
            cam = colour.convert(img, 'sRGB', 'CAM16UCS')

    lum = exposure.rescale_intensity(cam[..., 0], in_range=(0, 1), out_range=(0, 255))

    return lum


def match_histograms(src_image, ref_histogram, bins=256):
    """
    Source: https://automaticaddison.com/how-to-do-histogram-matching-using-opencv/


    This method matches the source image histogram to the
    reference signal
    :param image src_image: The original source image
    :param image  ref_image: The reference image
    :return: image_after_matching
    :rtype: image (array)
    """
    def calculate_cdf(histogram):
        """
        This method calculates the cumulative distribution function
        :param array histogram: The values of the histogram
        :return: normalized_cdf: The normalized cumulative distribution function
        :rtype: array
        """
        # Get the cumulative sum of the elements
        cdf = histogram.cumsum()

        # Normalize the cdf
        normalized_cdf = cdf / float(cdf.max())

        return normalized_cdf

    def calculate_lookup(src_cdf, ref_cdf):
        """
        This method creates the lookup table
        :param array src_cdf: The cdf for the source image
        :param array ref_cdf: The cdf for the reference image
        :return: lookup_table: The lookup table
        :rtype: array
        """
        lookup_table = np.zeros(256)
        lookup_val = 0
        for src_pixel_val in range(len(src_cdf)):
            lookup_val
            for ref_pixel_val in range(len(ref_cdf)):
                if ref_cdf[ref_pixel_val] >= src_cdf[src_pixel_val]:
                    lookup_val = ref_pixel_val
                    break
            lookup_table[src_pixel_val] = lookup_val
        return lookup_table

    # Split the images into the different color channels
    src_hist,  _ = np.histogram(src_image.flatten(), bins)

    # Compute the normalized cdf for the source and reference image
    src_cdf = calculate_cdf(src_hist)
    ref_cdf = calculate_cdf(ref_histogram)

    # Make a separate lookup table for each color
    lookup_table = calculate_lookup(src_cdf, ref_cdf)

    # Use the lookup function to transform the colors of the original
    # source image
    src_after_transform = cv2.LUT(src_image, lookup_table)
    image_after_matching = cv2.convertScaleAbs(src_after_transform)

    return image_after_matching


def get_channel_stats(img):
    img_stats = [None] * 5
    img_stats[0] = np.percentile(img, 1)
    img_stats[1] = np.percentile(img, 5)
    img_stats[2] = np.mean(img)
    img_stats[3] = np.percentile(img, 95)
    img_stats[4] = np.percentile(img, 99)

    return np.array(img_stats)


def norm_img_stats(img, target_stats):
    """Normalize an image

    Image will be normalized to have same stats as `target_stats`

    Based on method in
    "A nonlinear mapping approach to stain normalization in digital histopathology
    images using image-specific color deconvolution.", Khan et al. 2014

    """

    eps = np.finfo(float).resolution
    target_stats = get_channel_stats(target_stats)
    src_stats_flat = get_channel_stats(img)
    eps_array = np.arange(1, len(src_stats_flat)+1)*eps

    # Avoid duplicates and keep in ascending order
    lower_knots = np.array([0])
    upper_knots = np.array([300, 350, 400, 450])
    src_stats_flat = np.hstack([lower_knots, src_stats_flat, upper_knots])
    target_stats_flat = np.hstack([lower_knots, target_stats, upper_knots])

    eps_array = np.arange(len(src_stats_flat)) * eps
    src_stats_flat = np.sort(src_stats_flat + eps_array)
    target_stats_flat = np.sort(target_stats_flat + eps_array)

    # sp = csaps.CubicSmoothingSpline(src_stats_flat, target_stats_flat, smooth=0.995)
    cs = Akima1DInterpolator(src_stats_flat, target_stats_flat)
    normed_img = cs(img.reshape(-1)).reshape(img.shape)

    if img.dtype == np.uint8:
        normed_img = np.clip(normed_img, 0, 255)

    return normed_img

