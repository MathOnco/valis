"""
Collection of pre-processing methods for aligning images
"""
from scipy.interpolate import Akima1DInterpolator
from skimage import exposure, filters, measure, morphology, restoration
import numpy as np
import cv2
from skimage import color as skcolor
import pyvips
import colour
from scipy import ndimage

from . import slide_io
from . import warp_tools

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

    def create_mask(self):
        return np.full(self.image.shape[0:2], 255, dtype=np.uint8)

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

    def create_mask(self):
        _, tissue_mask = create_tissue_mask_from_multichannel(self.image)

        return tissue_mask

    def process_image(self, channel="dapi", adaptive_eq=True, *args, **kwaargs):
        reader_cls = slide_io.get_slide_reader(self.src_f, series=self.series)
        reader = reader_cls(self.src_f)
        if self.image is None:
            chnl = reader.get_channel(channel=channel, level=self.level, series=self.series).astype(float)
        else:
            if self.image.ndim == 2:
                # the image is already the channel
                chnl = self.image
            else:
                chnl_idx = reader.get_channel_index(channel)
                chnl = self.image[..., chnl_idx]
        chnl = exposure.rescale_intensity(chnl, in_range="image", out_range=(0.0, 1.0))

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

    def create_mask(self):
        _, tissue_mask = create_tissue_mask_from_rgb(self.image)

        return tissue_mask

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


    def create_mask(self):
        _, tissue_mask = create_tissue_mask_from_rgb(self.image)

        return tissue_mask

    def process_image(self,  *args, **kwaargs):
        lum = get_luminosity(self.image)
        inv_lum = 255 - lum
        processed_img = exposure.rescale_intensity(inv_lum, in_range="image", out_range=(0, 255)).astype(np.uint8)

        return processed_img


class BgColorDistance(ImageProcesser):
    """Calculate distance between each pixel and the background color

    """

    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level,
                         series=series, *args, **kwargs)

    def create_mask(self):
        _, tissue_mask = create_tissue_mask_from_rgb(self.image)

        return tissue_mask

    def process_image(self,  brightness_q=0.99, *args, **kwargs):

        processed_img, _ = calc_background_color_dist(self.image, brightness_q=brightness_q)
        processed_img = exposure.rescale_intensity(processed_img, in_range="image", out_range=(0, 1))
        processed_img = exposure.equalize_adapthist(processed_img)
        processed_img = exposure.rescale_intensity(processed_img, in_range="image", out_range=(0, 255)).astype(np.uint8)

        return processed_img


def denoise_img(img, mask=None, weight=None):
    if mask is None:
        sigma = restoration.estimate_sigma(img)
        sigma_scale = 40
    else:
        sigma = restoration.estimate_sigma(img[mask != 0])
        sigma_scale = 400

    if weight is None:
        weight=sigma/sigma_scale

    denoised_img = restoration.denoise_tv_chambolle(img, weight=weight)
    denoised_img = exposure.rescale_intensity(denoised_img, out_range="uint8")

    return denoised_img


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


def calc_background_color_dist(img, brightness_q=0.99, mask=None):
    """Create mask that only covers tissue

    #. Find background pixel (most luminescent)
    #. Convert image to CAM16-UCS
    #. Calculate distance between each pixel and background pixel
    #. Threshold on distance (i.e. higher distance = different color)

    Returns
    -------
    cam_d : float
        Distance from background color
    cam : float
        CAM16UCS image

    """
    eps = np.finfo("float").eps
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        if 1 < img.max() <= 255 and np.issubdtype(img.dtype, np.integer):
            cam = colour.convert(img/255 + eps, 'sRGB', 'CAM16UCS')
        else:
            cam = colour.convert(img + eps, 'sRGB', 'CAM16UCS')

    if mask is None:
        brightest_thresh = np.quantile(cam[..., 0], brightness_q)
    else:
        brightest_thresh = np.quantile(cam[..., 0][mask > 0], brightness_q)

    brightest_idx = np.where(cam[..., 0] >= brightest_thresh)
    brightest_pixels = cam[brightest_idx]
    bright_cam = brightest_pixels.mean(axis=0)
    cam_d = np.sqrt(np.sum((cam - bright_cam)**2, axis=2))

    return cam_d, cam


def rgb2jab(rgb, cspace='CAM16UCS'):
    eps = np.finfo("float").eps
    if np.issubdtype(rgb.dtype, np.integer) and rgb.max() > 1:
        rgb01 = rgb/255.0
    else:
        rgb01 = rgb

    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        jab = colour.convert(rgb01+eps, 'sRGB', cspace)

    return jab


def jab2rgb(jab, cspace='CAM16UCS'):
    eps = np.finfo("float").eps
    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        rgb = colour.convert(jab+eps, cspace, 'sRGB')

    return rgb


def rgb2jch(rgb, cspace='CAM16UCS', h_rotation=0):
    jab = rgb2jab(rgb, cspace)
    jch = colour.models.Jab_to_JCh(jab)
    jch[..., 2] += h_rotation

    above_360 = np.where(jch[..., 2] > 360)
    if len(above_360[0]) > 0:
        jch[..., 2][above_360] = jch[..., 2][above_360] - 360

    return jch


def rgb255_to_rgb1(rgb_img):
    if np.issubdtype(rgb_img.dtype, np.integer) or rgb_img.max() > 1:
        rgb01 = rgb_img/255.0
    else:
        rgb01 = rgb_img

    return rgb01


def rgb2od(rgb_img):
    eps = np.finfo("float").eps
    rgb01 = rgb255_to_rgb1(rgb_img)

    od = -np.log10(rgb01 + eps)
    od[od < 0] = 0

    return od


def stainmat2decon(stain_mat_srgb255):

    od_mat = rgb2od(stain_mat_srgb255)

    M = od_mat / np.linalg.norm(od_mat, axis=1, keepdims=True)
    M[np.isnan(M)] = 0
    D = np.linalg.pinv(M)

    return D


def deconvolve_img(rgb_img, D):
    od_img = rgb2od(rgb_img)
    deconvolved_img = np.dot(od_img, D)
    deconvolved_img[deconvolved_img < 0] = 0

    return deconvolved_img


def combine_masks_by_hysteresis(mask_list):
    """
    Combine masks. Keeps areas where they overlap _and_ touch
    """
    m0 = mask_list[0]
    if isinstance(m0, pyvips.Image):
        mshape = np.array([m0.height, m0.width])
    else:
        mshape = m0.shape[0:2]

    to_hyst_mask = np.zeros(mshape)
    for m in mask_list:

        if(isinstance(m, pyvips.Image)):
            np_mask = warp_tools.vips2numpy(m)
        else:
            np_mask = m.copy()

        to_hyst_mask[ np_mask > 0] += 1

    hyst_mask = 255*filters.apply_hysteresis_threshold(to_hyst_mask, 0.5, len(mask_list) - 0.5).astype(np.uint8)

    return hyst_mask


def combine_masks(mask1, mask2, op="or"):
    if not isinstance(mask1, pyvips.Image):
        vmask1 = warp_tools.numpy2vips(mask1)
    else:
        vmask1 = mask1

    if not isinstance(mask2, pyvips.Image):
        vmask2 = warp_tools.numpy2vips(mask2)
    else:
        vmask2 = mask2

    vips_combo_mask = vmask1.bandjoin(vmask2)
    if op == "or":
        combo_mask = vips_combo_mask.bandor()
    else:
        combo_mask = vips_combo_mask.bandand()

    if not isinstance(mask1, pyvips.Image):
        combo_mask = warp_tools.vips2numpy(combo_mask)

    return combo_mask

def remove_small_obj_and_lines_by_dist(mask):
    """
    Will remove smaller objects and thin lines that
    do not interesct with larger objects
    """

    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dst_t = filters.threshold_li(dist_transform[mask > 0])
    temp_sure_fg = 255*(dist_transform >= dst_t).astype(np.uint8)
    sure_mask = combine_masks_by_hysteresis([mask, temp_sure_fg])

    return sure_mask


def create_edges_mask(labeled_img):
    """
    Create two masks, one with objects not touching image borders,
    and a second with objects that do touch the border

    """
    unique_v = np.unique(labeled_img)
    unique_v = unique_v[unique_v != 0]
    if len(unique_v) == 1:
        labeled_img = measure.label(labeled_img)

    img_regions = measure.regionprops(labeled_img)
    inner_mask = np.zeros(labeled_img.shape, dtype=np.uint8)
    edges_mask = np.zeros(labeled_img.shape, dtype=np.uint8)
    for regn in img_regions:
        on_border_idx = np.where((regn.coords[:, 0] == 0) |
                        (regn.coords[:, 0] == labeled_img.shape[0]-1) |
                        (regn.coords[:, 1] == 0) |
                        (regn.coords[:, 1] == labeled_img.shape[1]-1)
                        )[0]
        if len(on_border_idx) == 0:
            inner_mask[regn.coords[:, 0], regn.coords[:, 1]] = 255
        else:
            edges_mask[regn.coords[:, 0], regn.coords[:, 1]] = 255

    return inner_mask, edges_mask


def create_tissue_mask_from_rgb(img, brightness_q=0.99, kernel_size=3, gray_thresh=0.075, light_gray_thresh=0.875, dark_gray_thresh=0.7):
    """Create mask that only covers tissue

    Also remove dark regions on the edge of the slide, which could be artifacts

    Parameters
    ----------
    grey_thresh : float
        Colorfulness values (from JCH) below this are considered "grey", and thus possibly dirt, hair, coverslip edges, etc...

    light_gray_thresh : float
        Upper limit for light gray

    dark_gray_thresh : float
        Upper limit for dark gray

    Returns
    -------
    tissue_mask : ndarray
        Mask covering tissue

    concave_tissue_mask : ndarray
        Similar to `tissue_mask`,  but each region is replaced by a concave hull.
        Covers more area

    """
    # Ignore artifacts that could throw off thresholding. These are often greyish in color

    jch = rgb2jch(img)
    light_greys = 255*((jch[..., 1] < gray_thresh) & (jch[..., 0] < light_gray_thresh)).astype(np.uint8)
    dark_greys = 255*((jch[..., 1] < gray_thresh) & (jch[..., 0] < dark_gray_thresh)).astype(np.uint8)
    grey_mask = combine_masks_by_hysteresis([light_greys, dark_greys])

    color_mask = 255 - grey_mask

    cam_d, cam = calc_background_color_dist(img, brightness_q=brightness_q, mask=color_mask)

    # Reduce intensity of thick horizontal and vertial lines, usually artifacts like edges, streaks, folds, etc...
    vert_knl = np.ones((1, 5))
    no_v_lines = morphology.opening(cam_d, vert_knl)

    horiz_knl = np.ones((5, 1))
    no_h_lines = morphology.opening(cam_d, horiz_knl)
    cam_d_no_lines = np.dstack([no_v_lines, no_h_lines]).min(axis=2)

    # Foreground is where color is different than backaground color
    cam_d_t, _ = filters.threshold_multiotsu(cam_d_no_lines[grey_mask == 0])
    tissue_mask = np.zeros(cam_d_no_lines.shape, dtype=np.uint8)
    tissue_mask[cam_d_no_lines >= cam_d_t] = 255

    concave_tissue_mask = mask2contours(tissue_mask, kernel_size)

    return tissue_mask, concave_tissue_mask


def create_tissue_mask_from_multichannel(img, kernel_size=3):
    """
    Get foreground of multichannel imaage
    """

    tissue_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    if img.ndim > 2:
        for i in range(img.shape[2]):
            chnl_t = np.quantile(img[..., i], 0.01)
            tissue_mask[img[..., i] > chnl_t] = 255

    else:
        t = np.quantile(img, 0.01)
        tissue_mask[img > t] = 255
    tissue_mask = 255*ndimage.binary_fill_holes(tissue_mask).astype(np.uint8)
    concave_tissue_mask = mask2contours(tissue_mask, kernel_size=kernel_size)

    return tissue_mask, concave_tissue_mask


def create_tissue_mask(img, is_rgb=True):
    """
    Returns
    -------
    tissue_mask : ndarray
        Mask covering tissue

    concave_tissue_mask : ndarray
        Similar to `tissue_mask`,  but each region is replaced by a concave hull

    """
    if is_rgb:
        tissue_mask, concave_tissue_mask = create_tissue_mask_from_rgb(img)
    else:
        tissue_mask, concave_tissue_mask = create_tissue_mask_from_multichannel(img)

    return tissue_mask, concave_tissue_mask


def mask2covexhull(mask):
    labeled_mask = measure.label(mask)
    mask_regions = measure.regionprops(labeled_mask)
    concave_mask = np.zeros_like(mask)
    for region in mask_regions:
        r0, c0, r1, c1 = region.bbox
        concave_mask[r0:r1, c0:c1] += region.convex_image.astype(np.uint8)

    concave_mask[concave_mask != 0] = 255
    concave_mask = 255*ndimage.binary_fill_holes(concave_mask).astype(np.uint8)

    return concave_mask


def mask2bbox_mask(mask, merge_bbox=True):
    """
    Replace objects in mask with bounding boxes. If `combine_bbox`
    is True, then bounding boxes will merged if they are touching,
    and the bounding box will be drawn around those overlapping boxes.
    """

    n_regions = -1
    n_prev_regions = 0
    max_iter = 10000
    i = 0
    updated_mask = mask.copy()
    while n_regions != n_prev_regions:
        n_prev_regions = n_regions
        labeled_mask = measure.label(updated_mask)
        bbox_mask = np.zeros_like(updated_mask)
        regions = measure.regionprops(labeled_mask)
        for r in regions:
            r0, c0, r1, c1 = r.bbox
            bbox_mask[r0:r1, c0:c1] = 255

        n_regions = len(regions)
        updated_mask = bbox_mask

        if not merge_bbox:
            break

        i += 1
        if i > max_iter:
            break

    return updated_mask


def mask2contours(mask, kernel_size=3):
    kernel = morphology.disk(kernel_size)
    mask_dilated = cv2.dilate(mask, kernel)
    contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(mask_dilated)
    for cnt in contours:
        cv2.drawContours(contour_mask, [cnt], 0, 255, -1)

    return contour_mask


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


def norm_img_stats(img, target_stats, mask=None):
    """Normalize an image

    Image will be normalized to have same stats as `target_stats`

    Based on method in
    "A nonlinear mapping approach to stain normalization in digital histopathology
    images using image-specific color deconvolution.", Khan et al. 2014

    Assumes that `img` values range between 0-255

    """

    if mask is None:
        src_stats_flat = get_channel_stats(img)
    else:
        src_stats_flat = get_channel_stats(img[mask > 0])

    # Avoid duplicates and keep in ascending order
    lower_knots = np.array([0])
    upper_knots = np.array([300, 350, 400, 450])
    src_stats_flat = np.hstack([lower_knots, src_stats_flat, upper_knots]).astype(float)
    target_stats_flat = np.hstack([lower_knots, target_stats, upper_knots]).astype(float)

    # Add epsilon to avoid duplicate values
    eps = 10*np.finfo(float).resolution
    eps_array = np.arange(len(src_stats_flat)) * eps
    src_stats_flat = src_stats_flat + eps_array
    target_stats_flat = target_stats_flat + eps_array

    # Make sure src stats are in ascending order
    src_order = np.argsort(src_stats_flat)
    src_stats_flat = src_stats_flat[src_order]
    target_stats_flat = target_stats_flat[src_order]

    cs = Akima1DInterpolator(src_stats_flat, target_stats_flat)

    if mask is None:
        normed_img = cs(img.reshape(-1)).reshape(img.shape)
    else:
        normed_img = img.copy()
        normed_img[mask > 0] = cs(img[mask > 0])

    if img.dtype == np.uint8:
        normed_img = np.clip(normed_img, 0, 255)

    return normed_img

