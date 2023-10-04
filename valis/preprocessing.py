"""
Collection of pre-processing methods for aligning images
"""
from scipy.interpolate import Akima1DInterpolator
from skimage import exposure, filters, measure, morphology, restoration
from sklearn.cluster import estimate_bandwidth, MiniBatchKMeans, MeanShift
import numpy as np
import cv2
from skimage import color as skcolor
import pyvips
import colour
from scipy import ndimage
from shapely import LineString
from scipy import stats
from sklearn import cluster
from sklearn.preprocessing import StandardScaler

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

    def process_image(self, c=DEFAULT_COLOR_STD_C, invert=True, adaptive_eq=False, *args, **kwargs):
        std_rgb = standardize_colorfulness(self.image, c)
        std_g = skcolor.rgb2gray(std_rgb)

        if invert:
            std_g = 255 - std_g

        if adaptive_eq:
            std_g = exposure.equalize_adapthist(std_g/255)

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


class StainFlattener(ImageProcesser):
    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level,
                         series=series, *args, **kwargs)

        self.n_colors = -1


    def create_mask(self):

        processed = self.process_image(adaptive_eq=True)

        # Want to ignore black background
        to_thresh_mask = 255*(np.all(self.image > 25, axis=2)).astype(np.uint8)

        low_t, high_t = filters.threshold_multiotsu(processed[to_thresh_mask > 0])
        tissue_mask = 255*filters.apply_hysteresis_threshold(processed, low_t, high_t).astype(np.uint8)

        kernel_size=3
        tissue_mask = mask2contours(tissue_mask, kernel_size)

        return tissue_mask

    def process_image_with_mask(self, n_colors=100, q=95, max_colors=100):
        fg_mask, _ = create_tissue_mask_from_rgb(self.image)
        mean_bg_rgb = np.mean(self.image[fg_mask == 0], axis=0)

        # Get stain vectors
        fg_rgb = self.image[fg_mask > 0]
        fg_to_cluster = rgb2jab(fg_rgb)

        ss = StandardScaler()
        x = ss.fit_transform(fg_to_cluster)

        if n_colors > 0:
            self.n_colors = n_colors
            clusterer = MiniBatchKMeans(n_clusters=n_colors,
                                        reassignment_ratio=0,
                                        n_init=3)
            clusterer.fit(x)
        else:
            k, clusterer = estimate_k(x, max_k=max_colors)
            self.n_colors = k
            # print(f"estimated {k} colors")

        self.clusterer = clusterer
        stain_rgb = jab2rgb(ss.inverse_transform(clusterer.cluster_centers_))
        stain_rgb = np.clip(stain_rgb, 0, 1)

        stain_rgb = np.vstack([255*stain_rgb, mean_bg_rgb])
        D = stainmat2decon(stain_rgb)
        deconvolved = deconvolve_img(self.image, D)

        eps = np.finfo("float").eps
        d_flat = deconvolved.reshape(-1, deconvolved.shape[2])
        dmax = np.percentile(d_flat, q, axis=0)
        for i in range(deconvolved.shape[2]):
            c_dmax  = dmax[i] + eps
            deconvolved[..., i] = np.clip(deconvolved[..., i], 0, c_dmax)
            deconvolved[..., i] /= c_dmax

        summary_img = deconvolved.mean(axis=2)

        return summary_img

    def process_image_all(self, n_colors=100, q=95, max_colors=100):
        img_to_cluster = rgb2jab(self.image)

        ss = StandardScaler()
        x = ss.fit_transform(img_to_cluster.reshape(-1, img_to_cluster.shape[2]))
        if n_colors > 0:
            self.n_colors = n_colors
            clusterer = MiniBatchKMeans(n_clusters=n_colors,
                                        reassignment_ratio=0,
                                        n_init=3)
            clusterer.fit(x)
        else:
            k, clusterer = estimate_k(x, max_k=max_colors)
            self.n_colors = k
            print(f"estimated {k} colors")

        self.clusterer = clusterer
        stain_rgb = jab2rgb(ss.inverse_transform(clusterer.cluster_centers_))
        stain_rgb = np.clip(stain_rgb, 0, 1)

        stain_rgb = 255*stain_rgb
        stain_rgb = np.clip(stain_rgb, 0, 255)
        stain_rgb = np.unique(stain_rgb, axis=0)
        D = stainmat2decon(stain_rgb)
        deconvolved = deconvolve_img(self.image, D)

        d_flat = deconvolved.reshape(-1, deconvolved.shape[2])
        dmax = np.percentile(d_flat, q, axis=0) + np.finfo("float").eps
        for i in range(deconvolved.shape[2]):

            deconvolved[..., i] = np.clip(deconvolved[..., i], 0, dmax[i])
            deconvolved[..., i] /= dmax[i]

        summary_img = deconvolved.mean(axis=2)

        return summary_img

    def process_image(self, n_colors=100, q=95, with_mask=True, adaptive_eq=True, max_colors=100):
        """
        Parameters
        ----------
        n_colors : int
            Number of colors to use for deconvolution. If `n_stains = -1`, then the number
            of colors will be estimated using the K-means "elbow method".

        max_colors : int
            If `n_colors = -1`, this value sets the maximum number of color clusters
        """
        if with_mask:
            processed_img = self.process_image_with_mask(n_colors=n_colors, q=q, max_colors=max_colors)
        else:
            processed_img = self.process_image_all(n_colors=n_colors, q=q, max_colors=max_colors)

        if adaptive_eq:
            processed_img = exposure.equalize_adapthist(processed_img)

        processed_img = exposure.rescale_intensity(processed_img, in_range="image", out_range=(0, 255)).astype(np.uint8)

        return processed_img


class Gray(ImageProcesser):
    """Get luminosity of an RGB image

    """

    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level,
                         series=series, *args, **kwargs)


    def create_mask(self):
        _, tissue_mask = create_tissue_mask_from_rgb(self.image)

        return tissue_mask

    def process_image(self,  *args, **kwaargs):
        g = skcolor.rgb2gray(self.image)
        processed_img = exposure.rescale_intensity(g, in_range="image", out_range=(0, 255)).astype(np.uint8)

        return processed_img


class HEDeconvolution(ImageProcesser):
    """Normalize staining appearence of hematoxylin and eosin (H&E) stained image
    and get the H or E deconvolution image.

    Reference
    ---------
    A method for normalizing histology slides for quantitative analysis. M. Macenko et al., ISBI 2009.

    """

    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level,
                         series=series, *args, **kwargs)

    def create_mask(self):
        _, tissue_mask = create_tissue_mask_from_rgb(self.image)

        return tissue_mask


    def process_image(self, stain="hem", Io=240, alpha=1, beta=0.15, *args, **kwargs):
        """
        Reference
        ---------
        A method for normalizing histology slides for quantitative analysis. M. Macenko et al., ISBI 2009.

        Note
        ----
        Adaptation of the code from https://github.com/schaugf/HEnorm_python.

        """

        normalized_stains_conc = normalize_he(self.image, Io=Io, alpha=alpha, beta=beta)
        processed_img = deconvolution_he(self.image, Io=Io, normalized_concentrations=normalized_stains_conc, stain=stain)

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


def normalize_he(img: np.array, Io: int = 240, alpha: int = 1, beta: int = 0.15):
    """ Normalize staining appearence of H&E stained images.

    Parameters
    ----------
    img : ndarray
        2D RGB image to be transformed, np.array<height, width, ch>.
    Io : int, optional
        The transmitted light intensity. The default value is ``240``.
    alpha : int, optional
        This value is used to get the alpha(th) and (100-alpha)(th) percentile
        as robust approximations of the intensity histogram min and max values.
        The default value, found empirically, is ``1``.
    beta : float, optional
        Threshold value used to remove the pixels with a low OD for stability reasons.
        The default value, found empirically, is ``0.15``.

    Returns
    -------
    normalized_stains_conc : ndarray
        The normalized stains vector, np.array<2, im_height*im_width>.

    """

    max_conc_ref = np.array([1.9705, 1.0308])

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    opt_density = -np.log((img.astype(np.float)+1)/Io)

    # remove transparent pixels
    opt_density_hat = opt_density[~np.any(opt_density<beta, axis=1)]

    # compute eigenvectors
    _, eigvecs = np.linalg.eigh(np.cov(opt_density_hat.T))

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    t_hat = opt_density_hat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(t_hat[:, 1], t_hat[:, 0])

    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100-alpha)

    v_min = eigvecs[:, 1:3].dot(np.array([(np.cos(min_phi), np.sin(min_phi))]).T)
    v_max = eigvecs[:, 1:3].dot(np.array([(np.cos(max_phi), np.sin(max_phi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if v_min[0] > v_max[0]:
        h_e_vector = np.array((v_min[:, 0], v_max[:, 0])).T
    else:
        h_e_vector = np.array((v_max[:, 0], v_min[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    y = np.reshape(opt_density, (-1, 3)).T

    # determine concentrations of the individual stains
    stains_conc = np.linalg.lstsq(h_e_vector, y, rcond=None)[0]

    # normalize stains concentrations
    max_conc = np.array([np.percentile(stains_conc[0, :], 99), np.percentile(stains_conc[1, :],99)])
    tmp = np.divide(max_conc, max_conc_ref)
    normalized_stains_conc = np.divide(stains_conc, tmp[:, np.newaxis])

    return normalized_stains_conc


def deconvolution_he(img: np.array, normalized_concentrations: np.array, stain: str = "hem", Io: int = 240):
    """ Unmix the hematoxylin or eosin channel based on their respective normalized concentrations.

    Parameters
    ----------
    img : ndarray
        2D RGB image to be transformed, np.array<height, width, ch>.
    stain : str
        Either ``hem`` for the hematoxylin stain or ``eos`` for the eosin one.
    Io : int, optional
        The transmitted light intensity. The default value is ``240``.

    Returns
    -------
    out : ndarray
        2D image with a single channel corresponding to the desired stain, np.array<height, width>.

    """
    # define height and width of image
    h, w, _ = img.shape

    # unmix hematoxylin or eosin
    if stain == "hem":
        out = np.multiply(Io, normalized_concentrations[0,:])
    elif stain == "eos":
        out = np.multiply(Io, normalized_concentrations[1,:])
    else:
        raise ValueError(f"Stain ``{stain}`` is unknown.")

    np.clip(out, a_min=0, a_max=255, out=out)
    out = np.reshape(out, (h, w)).astype(np.float32)

    return out


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


def jch2rgb(jch, cspace="CAM16UCS", h_rotation=0):
    eps = np.finfo("float").eps

    c = jch[..., 1]
    h = np.deg2rad(jch[..., 2] - h_rotation)

    a = c*np.cos(h)
    b = c*np.sin(h)

    jab = np.dstack([jch[..., 0], a, b])

    with colour.utilities.suppress_warnings(colour_usage_warnings=True):
        rgb = colour.convert(jab + eps, cspace, 'sRGB')

    rgb = np.clip(rgb, 0, 1)
    rgb = (255*rgb).astype(np.uint8)

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

    eps = np.finfo("float").eps
    M = od_mat / np.linalg.norm(od_mat+eps, axis=1, keepdims=True)
    M[np.isnan(M)] = 0
    D = np.linalg.pinv(M)

    return D


def deconvolve_img(rgb_img, D):
    od_img = rgb2od(rgb_img)
    deconvolved_img = np.dot(od_img, D)
    deconvolved_img[deconvolved_img < 0] = 0

    return deconvolved_img


def thresh_unimodal(x, bins=256):
    """
    https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/unimodal2.pdf

    To threshold
    :param px_vals:
    :param bins:
    :return:
    """
    # Threshold unimodal distribution
    skew = stats.skew(x)
    # Find line from peak to tail
    if skew >= 0:
        counts, bin_edges = np.histogram(x, bins=bins)
    else:
        # Tail is to the left, so reverse values to use this method, which assumes tail is on the right
        counts, bin_edges = np.histogram(-x, bins=bins)

    bin_width = bin_edges[1] - bin_edges[0]
    midpoints = bin_edges[0:-1] + bin_width/2
    hist_line = LineString(np.column_stack([midpoints, counts]))

    peak_bin = np.argmax(counts)
    last_non_zero = np.where(counts > 0)[0][-1]
    if last_non_zero == len(counts) - 1:
        min_bin = last_non_zero
    else:
        min_bin = last_non_zero + 1

    peak_x, min_bin_x = midpoints[peak_bin], midpoints[min_bin]
    peak_y, min_bin_y = counts[peak_bin], counts[min_bin]

    peak_m = (peak_y- min_bin_y)/(peak_x - min_bin_x + np.finfo(float).resolution)
    peak_b = peak_y - peak_m*peak_x
    perp_m = -peak_m + np.finfo(float).resolution
    n_v = len(midpoints)
    d = [-1] * n_v
    all_xi = [-1] * n_v

    for i in range(n_v):

        x1 = midpoints[i]
        if x1 < peak_x:
            continue
        y1 = peak_m*x1 + peak_b
        perp_b = y1 - perp_m*x1
        y2 = 0
        x2 = -perp_b/(perp_m)

        perp_line_obj = LineString([[x1, y1], [x2, y2]])
        if not perp_line_obj.is_valid or not hist_line.is_valid:
            print("perpline is valid", perp_line_obj.is_valid, "hist line is valid", hist_line.is_valid)
            print("perpline xy1, xy2", [x1, y1], [x2, y2], "m=", perp_m)

        intersection = perp_line_obj.intersection(hist_line)
        if intersection.is_empty:
            # No intersection
            continue
        if intersection.geom_type == 'MultiPoint':
            all_x, all_y = LineString(intersection.geoms).xy
            xi = all_x[-1]
            yi = all_y[-1]
        elif intersection.geom_type == 'Point':
            xi, yi = intersection.xy
            xi = xi[0]
            yi = yi[0]
        d[i] = np.sqrt((xi - x1)**2 + (yi - y1)**2)
        all_xi[i] = xi

    max_d_idx = np.argmax(d)
    t = all_xi[max_d_idx]

    if skew < 0:
        t*= -1

    return t


def estimate_k(x, max_k=100, step_size=10):

    if max_k <= 10:
        step_size = 1

    # Create initial cluster list
    potential_c = np.arange(0, max_k, step=step_size)
    # potential_c = np.linspace(2, max_k, n_steps).astype(int)
    if potential_c[-1] != max_k:
        potential_c = np.hstack([potential_c, max_k])
    potential_c[0] = 2
    potential_c = np.unique(potential_c[potential_c > 1])

    almost_done = False
    done = False
    best_k = 2
    best_clst = None
    k_step = step_size
    while not done:
        inertia_list = []
        nc = []
        clst_list = []

        for i in potential_c:

            try:
                clusterer = cluster.MiniBatchKMeans(n_clusters=i, n_init=3)
                clusterer.fit(x)

            except Exception as e:
                continue
            inertia_list.append(clusterer.inertia_)
            nc.append(i)
            clst_list.append(clusterer)

        inertia_list = np.array(inertia_list)

        dy = np.diff(inertia_list)
        intertia_t = thresh_unimodal(dy, int(np.max(potential_c)))
        best_k_idx = np.where(dy >= intertia_t)[0][0] + 1
        best_k = potential_c[best_k_idx]
        best_clst = clst_list[best_k_idx]
        if almost_done:
            done = True
            break

        next_k_range = np.clip([best_k - k_step//2, best_k + k_step//2], 2, max_k)
        kd = np.diff(next_k_range)[0]
        if kd == 0:
            done = True
            almost_done = True
            break
        if kd <= 10:
            k_step = 1
            almost_done = True
        else:
            k_step = step_size

        potential_c = np.arange(next_k_range[0], next_k_range[1], k_step)

    return best_k, best_clst


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

        if isinstance(mask, pyvips.Image):
            np_mask = warp_tools.vips2numpy(mask)
        else:
            np_mask = mask

        src_stats_flat = get_channel_stats(img[np_mask > 0])

    # Avoid duplicates and keep in ascending order
    lower_knots = np.array([0])
    upper_knots = np.array([300, 350, 400, 450])
    src_stats_flat = np.hstack([lower_knots, src_stats_flat, upper_knots]).astype(float)
    target_stats_flat = np.hstack([lower_knots, target_stats, upper_knots]).astype(float)

    # Add epsilon to avoid duplicate values
    eps = 100*np.finfo(float).resolution
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
        fg_px = np.where(np_mask > 0)
        normed_img[fg_px] = cs(img[fg_px])

    if img.dtype == np.uint8:
        normed_img = np.clip(normed_img, 0, 255)

    return normed_img

