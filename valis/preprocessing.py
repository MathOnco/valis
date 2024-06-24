"""
Collection of pre-processing methods for aligning images
"""
from scipy.interpolate import Akima1DInterpolator
from skimage import exposure, filters, measure, morphology, restoration, util
from skimage.feature import peak_local_max
from sklearn.cluster import estimate_bandwidth, MiniBatchKMeans, MeanShift
import numpy as np
import cv2
from skimage import color as skcolor
import pyvips
import colour
from scipy import ndimage
from shapely import LineString
from scipy import stats, spatial, signal
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import shapely
from shapely import ops

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

    def __init__(self, image, src_f, level, series, reader=None):
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

        if reader is None:
            reader_cls = slide_io.get_slide_reader(src_f, series=series)
            reader = reader_cls(src_f, series=series)

        self.reader = reader
        if self.reader.metadata.is_rgb and self.image.dtype != np.uint8:
            self.image = exposure.rescale_intensity(self.image, out_range=np.uint8)

        self.original_shape_rc = warp_tools.get_shape(image)[0:2] # Size of image passed into processor
        self.uncropped_shape_rc = None # Size of uncropped image (bigger than `original_shape_rc`)
        self.crop_bbox = None # bbox (x, y, w, h) of cropped area
        self.cropped = False

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

    def process_image(self, channel="dapi", adaptive_eq=True, invert=False, *args, **kwaargs):
        if self.image is None:
            chnl = self.reader.get_channel(channel=channel, level=self.level, series=self.series).astype(float)
        else:
            if self.image.ndim == 2:
                # the image is already the channel
                chnl = self.image
            else:
                chnl_idx = self.reader.get_channel_index(channel)
                chnl = self.image[..., chnl_idx]

        if adaptive_eq:
            chnl = exposure.rescale_intensity(chnl, in_range="image", out_range=(0.0, 1.0))
            chnl = exposure.equalize_adapthist(chnl)

        chnl = exposure.rescale_intensity(chnl, in_range="image", out_range=(0, 255)).astype(np.uint8)
        if invert:
            chnl = util.invert(chnl)

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


class JCDist(ImageProcesser):
    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level,
                         series=series, *args, **kwargs)

    def create_mask(self):

        _, mask = create_tissue_mask_with_jc_dist(self.image)

        return mask

    def process_image(self, p=99, metric="euclidean", adaptive_eq=True, *args, **kwargs):
        """
        Calculate norm of the OD image
        """

        jcd = jc_dist(self.image, metric=metric, p=p)
        if adaptive_eq:
            jcd = exposure.equalize_adapthist(exposure.rescale_intensity(jcd, out_range=(0, 1)))

        processed = exposure.rescale_intensity(jcd, out_range=np.uint8)

        return processed


class OD(ImageProcesser):
    """Convert the image from RGB to optical density (OD) and calculate pixel norms.
    """

    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level,
                         series=series, *args, **kwargs)



    def create_mask(self):

        _, mask = create_tissue_mask_with_jc_dist(self.image)

        return mask

    def process_image(self, adaptive_eq=False, p=95, *args, **kwargs):
        """
        Calculate norm of the OD image
        """
        eps = np.finfo("float").eps
        img01 = self.image/255
        od = -np.log10(img01 + eps)
        od_norm = np.mean(od, axis=2)
        upper_p = np.percentile(od_norm, p)
        lower_p = 0
        od_clipped = np.clip(od_norm, lower_p, upper_p)

        if adaptive_eq:
            od_clipped = exposure.equalize_adapthist(exposure.rescale_intensity(od_clipped, out_range=(0, 1)))

        processed = exposure.rescale_intensity(od_clipped, out_range=np.uint8)

        return processed


class ColorDeconvolver(ImageProcesser):
    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level,
                         series=series, *args, **kwargs)

    def create_mask(self):

        _, mask = create_tissue_mask_with_jc_dist(self.image)

        return mask

    def process_image(self, cspace="JzAzBz", method="similarity", adaptive_eq=False, return_unmixed=False, *args, **kwargs):
        """
        Process image by enhance stained pixels and subtracting backroung pixels

        Parameters
        ----------
        cspace : str
            Colorspace to use to detect and separate colors using `separate_colors`

        method : str
            How to calculate similarity of each pixel to colors detected in image

        adaptive_eq : bool
            Whether or not to apply adaptive histogram equalization

        Returns
        -------
        processed : np.ndarray
            Processed, single channel image

        """

        unmixed_img, img_colors, fg_color_mask, color_counts = separate_colors(self.image, cspace=cspace, hue_only=False, method=method, min_colorfulness=0)

        main_colors_jab = rgb2jab(img_colors, cspace=cspace)

        main_jab01 = (main_colors_jab - main_colors_jab.min(axis=0))/(main_colors_jab.max(axis=0) - main_colors_jab.min(axis=0))
        bg_jab_idx = np.argmax(main_colors_jab[..., 0]) # BG is brightest
        bg_jab_01 = main_jab01[bg_jab_idx]
        color_weights = spatial.distance.cdist(main_jab01, [bg_jab_01]).T[0]

        fg_thresh = filters.threshold_otsu(color_weights)
        fg_idx = np.where(color_weights > fg_thresh)[0]

        fg_stains = unmixed_img[..., fg_idx]
        fg_norm = np.linalg.norm(fg_stains, axis=2)
        fg_norm = exposure.rescale_intensity(fg_norm, out_range=(0, 1))

        bg_idx = np.where(color_weights <= fg_thresh)[0]
        bg_stains = unmixed_img[..., bg_idx]
        bg_norm = np.linalg.norm(bg_stains, axis=2)
        bg_norm = exposure.rescale_intensity(bg_norm, out_range=(0, 1))

        processed = fg_norm*(1-bg_norm)

        if adaptive_eq:
            processed = exposure.equalize_adapthist(exposure.rescale_intensity(processed, out_range=(0, 1)))

        processed = exposure.rescale_intensity(processed, out_range=np.uint8)

        if return_unmixed:
            return processed, unmixed_img, fg_idx, bg_idx
        else:

            return processed


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


def remove_bg(img, radius, invert=True, algo="rb"):
    _img = exposure.rescale_intensity(img, out_range=(0, 1))
    if invert:
        _img = 1 - _img

    if algo == "rb":
        background = restoration.rolling_ball(_img, radius=radius)
        if invert:
            filtered_inverted = _img - background
            filtered = 1 - filtered_inverted
        else:
            filtered = img - background
    elif algo == "hat":
        filtered = morphology.white_tophat(_img, morphology.disk(radius))
        if invert:
            filtered = 1 - filtered

    filtered = exposure.rescale_intensity(filtered, out_range=(img.min(), img.max()))

    return filtered


def remove_rgb_bg_in_j(img, radius=50, cspace="CAM16UCS", algo="rb", ad_eq=False):
    jab = rgb2jab(img, cspace=cspace)

    filtered_j = remove_bg(jab[..., 0], radius=radius, invert=True, algo=algo)
    if ad_eq:
        filtered_j = exposure.equalize_adapthist(exposure.rescale_intensity(filtered_j, out_range=(0, 1)))
        filtered_j = exposure.rescale_intensity(filtered_j, out_range=(jab[..., 0].min(), jab[..., 0].max()))

    img_no_bg = jab2rgb(np.dstack([filtered_j, jab[..., 1], jab[..., 2]]), cspace=cspace)
    img_no_bg = np.clip(img_no_bg, 0, 1)
    img_no_bg = exposure.rescale_intensity(img_no_bg, out_range=np.uint8)

    return img_no_bg


def remove_bg_each_channel_rgb(img, radius=50):

    image_inverted = util.invert(img)
    background_R = restoration.rolling_ball(image_inverted[..., 0], radius=radius)
    background_G = restoration.rolling_ball(image_inverted[..., 1], radius=radius)
    background_B = restoration.rolling_ball(image_inverted[..., 2], radius=radius)
    background = np.stack([background_R, background_G, background_B], 2)
    filtered_image_inverted = image_inverted - background
    filtered_image = util.invert(filtered_image_inverted)

    return filtered_image


def remove_bg_in_jch(img, radius=50, cspace="JzAzBz", algo="rb"):

    image_inverted = util.invert(img)
    jch = rgb2jch(image_inverted, cspace=cspace)
    filtered_j = remove_bg(jch[..., 0], radius=radius, invert=False, algo=algo)
    filtered_c = remove_bg(jch[..., 1], radius=radius, invert=False, algo=algo)

    img_inverted_no_bg = jch2rgb(np.dstack([filtered_j, filtered_c, jch[..., 2]]), cspace=cspace)
    img_no_bg = util.invert(img_inverted_no_bg)


    return img_no_bg


def denoise_img(img, mask=None, weight=None):
    if mask is None:
        sigma = restoration.estimate_sigma(img)
        sigma_scale = 40
    else:
        sigma = restoration.estimate_sigma(img[mask != 0])
        sigma_scale = 400

    if weight is None:
        weight = sigma/sigma_scale

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


def calc_background_color_dist(img, brightness_q=0.99, mask=None, cspace="CAM16UCS"):
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
            cam = colour.convert(img/255 + eps, 'sRGB', cspace)
        else:
            cam = colour.convert(img + eps, 'sRGB', cspace)

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
    opt_density = -np.log((img.astype(float)+1)/Io)

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
    rgb01 = rgb255_to_rgb1(rgb)
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

    return deconvolved_img


def view_as_scatter(img, cspace_name, cspace_fxn=None, channel_1_idx=None, channel_2_idx=None, channel_3_idx=None, log3d=False, cspace_kwargs=None, mask=None, s=3):
    """View colors in image, transformed used the `cspace_fxn`, as a scatterplot, where the color of each point is the corresponding RGB color

    Useful when trying to find color thresholds

    """

    if mask is None:
        img_flat = img.reshape((-1, 3))
    else:
        img_flat = img[mask > 0]


    unique_colors = np.unique(img_flat, axis=0)
    flat_size = unique_colors.shape[0]

    h = 2
    while flat_size%h != 0:
        h += 1

    w = int(flat_size/h)

    rgb_block = np.reshape(unique_colors, (h, w, 3))
    if cspace_fxn is None:
        cspace = rgb_block
    else:
        if cspace_kwargs is not None:
            cspace = cspace_fxn(rgb_block, **cspace_kwargs)
        else:
            cspace = cspace_fxn(rgb_block)
    if channel_2_idx is None:
        a = cspace[:, :, channel_1_idx]
        a = np.unique(a)
        y = np.random.uniform(-0.01, 0.01, size=a.size)

        plt.scatter(a, y, c=unique_colors, s=s)
        plt.xlabel(cspace_name[channel_1_idx])

    elif channel_3_idx is None:

        a = cspace[:, :, channel_1_idx]
        b = cspace[:, :, channel_2_idx]

        a = a.ravel()
        b = b.ravel()

        plt.scatter(a, b, c=unique_colors/255, s=s)
        plt.xlabel(cspace_name[channel_1_idx])
        plt.ylabel(cspace_name[channel_2_idx])

    else:
        a = cspace[:, :, channel_1_idx]
        b = cspace[:, :, channel_2_idx]
        c = cspace[:, :, channel_3_idx]

        a = a.ravel()
        b = b.ravel()
        c = c.ravel()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(a, b, c, c=unique_colors, depthshade=False, edgecolor=unique_colors, lw=0)
        plt.xlabel(cspace_name[channel_1_idx])
        plt.ylabel(cspace_name[channel_2_idx])
        ax.set_zlabel(cspace_name[channel_3_idx])
        if log3d:
            plt.title("Log")


def seg_jch(img, j_range=[0, 1], c_range=[0, 1], h_range=[0, 360], cspace='CAM16UCS', h_rotation=0):
    """Segment image in a JCH colorspace

    """

    jch_img = rgb2jch(img, cspace=cspace, h_rotation=h_rotation)

    jch_mask_idx = np.where((jch_img[..., 0] >= j_range[0]) & (jch_img[..., 0] < j_range[1]) &
                            (jch_img[..., 1] >= c_range[0]) & (jch_img[..., 1] < c_range[1]) &
                            (jch_img[..., 2] >= h_range[0]) & (jch_img[..., 2] < h_range[1])
                            )

    jch_mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    jch_mask[jch_mask_idx] = 255

    return jch_mask


def calc_shannon(img, n_bins=10, mask=None):
    """Calculate Shannon's entropy for each pixel in `img`

    """

    img01 = exposure.rescale_intensity(img, out_range=(0, 1))
    if img.ndim > 2:
        img01 = img01.reshape(-1, img.shape[2])
    else:
        img01 = img01.reshape(-1)

    if mask is None:
        x = np.round(img01*(n_bins-1)).astype(int)
    else:
        flat_mask = mask.reshape(-1)
        fg_idx = np.where(flat_mask > 0)[0]
        x = np.round(img01*(n_bins-1)).astype(int)[fg_idx]

    unique_x, counts = np.unique(x, return_counts=True, axis=0)
    probs = counts/counts.sum()
    if img.ndim > 2:
        prob_dict = {tuple(unique_x[i]): probs[i] for i in range(len(probs))}
        prob_img = np.array([prob_dict[tuple(k)] for k in x])
    else:
        prob_dict = {unique_x[i]: probs[i] for i in range(len(probs))}
        prob_img = np.array([prob_dict[k] for k in x])

    ent_img = -np.log(prob_img)

    if mask is None:
        prob_img = prob_img.reshape(img.shape[0:2])
        ent_img = ent_img.reshape(img.shape[0:2])
    else:
        _prob_img = np.zeros(np.multiply(*img.shape[0:2]))
        _prob_img[fg_idx] = prob_img
        prob_img = _prob_img.reshape(img.shape[0:2])

        _ent_img = np.zeros(np.multiply(*img.shape[0:2]))
        _ent_img[fg_idx] = ent_img
        ent_img = _ent_img.reshape(img.shape[0:2])

    return ent_img, prob_img


def find_elbow(x, y):
    m = (y[-1] - y[0]) / (x[-1] - x[0])
    c = y[0]

    # make the line
    line = m * x + c

    # get the residuals
    res = y - line

    # get gradient of the residuals
    grad = np.diff(res)

    # get index of minimum gradient
    midx = np.argmin(np.abs(grad))

    return midx


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

    peak_m = (peak_y - min_bin_y)/(peak_x - min_bin_x + np.finfo(float).resolution)
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
        t *= -1

    return t


def estimate_k(x, max_k=100, step_size=10):

    if max_k <= 10:
        step_size = 1

    # Create initial cluster list
    potential_c = np.arange(0, max_k, step=step_size)
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


def combine_masks_by_hysteresis(mask_list, upper_t=None):
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

        to_hyst_mask[np_mask > 0] += 1

    if upper_t is None:
        upper_t = len(mask_list)
    hyst_mask = 255*filters.apply_hysteresis_threshold(to_hyst_mask, 0.5, upper_t - 0.5).astype(np.uint8)

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


def split_shapely_line(line_poly, step_size=10, close_line=False):

    if not line_poly.is_closed and close_line:
        line_coords = np.dstack(line_poly.coords.xy).squeeze()
        closed_coords = np.vstack([line_coords, line_coords[0]])
        to_split = shapely.geometry.LineString(closed_coords)
    else:
        to_split = line_poly

    nseg = int(np.ceil(to_split.length/step_size))
    nseg = max(2, nseg)
    line_seg_idx = np.linspace(0, to_split.length, nseg)
    geom_list = [None] * (nseg-1)
    for i in range(nseg - 1):
        pos = np.linspace(line_seg_idx[i], line_seg_idx[i+1], step_size)
        seg_geom = shapely.geometry.LineString(to_split.interpolate(pos))
        geom_list[i] = seg_geom

    return geom_list


def get_poly_corners(img, tolerance=2):
    contours, _ = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    xy = contours[0].squeeze()

    new_xy = measure.subdivide_polygon(xy, degree=3, preserve_ends=True)
    contours_xy = measure.approximate_polygon(new_xy, tolerance=tolerance)

    return contours_xy


def polygon_tortuosity(img, window_size=3):
    """Calculate tortuosity of a contour in `img`. Should be masked so there is only 1 contour
    """

    contours_xy = get_poly_corners(img)
    n_over = contours_xy.shape[0] % window_size
    if n_over > 0:
        n_append = window_size - contours_xy.shape[0] % window_size
        wrapped_poly = np.vstack([contours_xy, contours_xy[:n_append]])
    else:
        wrapped_poly = contours_xy.copy()

    window_edges = list(range(0, wrapped_poly.shape[0]+1, window_size))

    n = 0
    total_length = 0
    tort_list = []
    for idx1 in range(0, len(window_edges)-1):
        idx2 = idx1 + 1
        window_idx1 = window_edges[idx1]
        window_idx2 = window_edges[idx2]

        verts_in_window = wrapped_poly[window_idx1:window_idx2]
        dist_traveled = np.sqrt(np.sum((verts_in_window[-1] - verts_in_window[0])**2))
        if dist_traveled == 0:
            continue
        path_length = np.sum(np.sqrt(np.sum(np.diff(verts_in_window, axis=0)**2, axis=1)))
        line_arc_chord = (path_length/dist_traveled) - 1
        if np.isclose(path_length, dist_traveled) and line_arc_chord < 0:
            line_arc_chord = 0

        n += 1
        total_length += path_length
        tort_list.append(line_arc_chord)

    if total_length == 0:
        poly_tort = 0
    else:
        poly_tort = ((n-1)/total_length)*sum(tort_list)

    return poly_tort


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
    cleaned_mask = clean_mask(mask=concave_tissue_mask, img=img)

    return tissue_mask, cleaned_mask


def jc_dist(img, cspace="IHLS", p=99, metric="euclidean"):
    """
    Cacluate distance between backround and each pixel
    using a luminosity and colofulness/saturation in a polar colorspace

    Parameters
    ----------
    img : np.ndarray
        RGB image

    cspace: str
        Name of colorspace to use for calculation

    p: int
        Percentile used to determine background values, i.e.
        background pixels have a luminosity greather 99% of other
        pixels. Needs to be between 0-100

    metric: str
        Name of distance metric. Passed to `scipy.spatial.distance.cdist`

    Returns
    -------
    jc_dist : np.ndarray
        Color distance between backround and each pixel

    """

    if cspace.upper() == "IHLS":
        hys = colour.models.RGB_to_IHLS(img) # Hue, luminance, saturation/colorfulness
        j = hys[..., 1]
        c = hys[..., 2]

    else:
        jch = rgb2jch(img, cspace=cspace)
        j = jch[..., 0]
        c = jch[..., 1]

    j01 = exposure.rescale_intensity(j, out_range=(0, 1))
    c01 = exposure.rescale_intensity(c, out_range=(0, 1))
    jc01 = np.dstack([j01, c01]).reshape((-1, 2))

    bg_j = np.percentile(j01, p)
    bg_c = np.percentile(c01, 100-p)

    jc_dist_img = spatial.distance.cdist(jc01, np.array([[bg_j, bg_c]]), metric=metric).reshape(img.shape[0:2])

    return jc_dist_img


def create_tissue_mask_with_jc_dist(img):
    """
    Create tissue mask using JC distance from background

    Parameters
    ----------
    img : np.ndarray
        RGB image

    Returns
    -------
    mask : np.ndarray
        Mask covering tissue

    chull_mask : np.ndarray
        Mask created by drawing a convex hull around each region in
        `mask`

    """

    assert img.ndim == 3, f"`img` needs to be RGB image"
    jc_dist_img = jc_dist(img, metric="chebyshev")
    jc_dist_img[np.isnan(jc_dist_img)] = np.nanmax(jc_dist_img)

    jc_t, _ = filters.threshold_multiotsu(jc_dist_img)
    jc_mask = 255*(jc_dist_img > jc_t).astype(np.uint8)
    jc_dist_img = exposure.equalize_adapthist(exposure.rescale_intensity(jc_dist_img, out_range=(0, 1)))

    img_edges = filters.scharr(jc_dist_img)
    p_t = filters.threshold_otsu(img_edges)
    edges_mask = 255*(img_edges > p_t).astype(np.uint8)

    temp_mask = edges_mask.copy()
    temp_mask[jc_mask == 0] = 0
    temp_mask = mask2contours(temp_mask, 3)

    mask = clean_mask(mask=temp_mask, img=img)
    chull_mask = mask2covexhull(mask)

    return mask, chull_mask


def clean_mask(mask, img, rel_min_size=0.001):
    """
    Remove small objects, regions that are not very colorful (relativey), and retangularly shaped objects
    """

    fg_labeled = measure.label(mask)
    fg_regions = measure.regionprops(fg_labeled)

    if len(fg_regions) == 1:
        return mask

    jch = rgb2jch(img, cspace="JzAzBz")
    c = exposure.rescale_intensity(jch[..., 1], out_range=(0, 1))
    colorfulness_img = np.zeros(mask.shape)

    for i, r in enumerate(fg_regions):
        # Fill in contours that are touching border
        r0, c0, r1, c1 = r.bbox
        r_filled_img = r.image_filled.copy()
        if r0 == 0:
            # Touching top
            lr = np.where(r.image_filled[0, :])[0]
            r_filled_img[0, min(lr):max(lr)] = 255

        if r1 == mask.shape[0]:
            # Touching bottom
            lr = np.where(r.image_filled[-1, :])[0]
            r_filled_img[-1, min(lr):max(lr)] = 255

        if c0 == 0:
            tb = np.where(r.image_filled[:, 0])[0]
            # Touchng left border
            r_filled_img[min(tb):max(tb), 0] = 255

        if c1 == mask.shape[1]:
            # Touchng right border
            tb = np.where(r.image_filled[:, -1])[0]
            r_filled_img[min(tb):max(tb), -1] = 255

        r_filled_img = ndimage.binary_fill_holes(r_filled_img)
        colorfulness_img[r.slice][r_filled_img] = np.max(c[r.slice][r_filled_img])

    color_thresh = filters.threshold_otsu(colorfulness_img[mask > 0])
    color_mask = colorfulness_img > color_thresh
    mask_list = [mask.astype(bool), color_mask]

    feature_mask = combine_masks_by_hysteresis(mask_list)
    if feature_mask.max() == 0:

        feature_mask = np.sum(np.dstack(mask_list), axis=2)
        feature_thresh = len(mask_list)//2
        feature_mask[feature_mask <= feature_thresh] = 0
        feature_mask[feature_mask != 0] = 255

    features_labeled = measure.label(feature_mask)
    feature_regions = measure.regionprops(features_labeled)

    if len(feature_regions) == 1:
        return feature_mask

    region_sizes = np.array([r.area for r in feature_regions])
    min_abs_size = int(rel_min_size*np.multiply(*mask.shape[0:2]))#*kernel_size
    keep_region_idx = np.where(region_sizes > min_abs_size)[0]
    if len(keep_region_idx) == 0:
        biggest_idx = np.argmax([r.area for r in fg_regions])
        keep_region_idx = [biggest_idx]

    # Get final regions
    fg_mask = np.zeros(mask.shape[0:2], np.uint8)
    for i, rid in enumerate(keep_region_idx):
        r = feature_regions[rid]
        fg_mask[r.slice][r.image_filled] = 255

    return fg_mask


def separate_colors(img, cspace="JzAzBz", min_colorfulness=0.005, px_thresh=0.0001, n_hue_bins=360, max_colors=5, n_colors=None, hue_only=False, method="deconvolve"):
    """ Creates an array where each channel corresponds to a color detected by `find_dominant_colors`

    Parameters
    ----------
    img : np.ndarray
        RGB image

    cspace : str
        Colorspace to use to detect and separate colors using `separate_colors`

    min_colorfulness : str
        Pixels with colorfulness/saturation less that this will be exluded.
        Calculated after binning colors.

    px_thresh: float
        Minimal frequency of a color

    n_hue_bins : int
        Number of bins to use when binning hues

    max_colors : int
        Expected maximum number of colors in the image. Number of colors detected
        will be equal to, or less than, this value.

    n_colors : int
        Number of colors in the image. If `None`,  `n_colors`will be determined by clustering

    hue_only : bool
        If `True`, use `find_dominant_hues` for color detection. If, `False`, use `find_dominant_colors`

    method : str
        How to calculate similarity of each pixel to colors detected in image.
        Options are "deconvolve", "norm", "similarity", "svm", "one_class_svm"

    Returns
    -------
    sep_img ; np.ndarray
        Each channel corresponds to similarity/probability of being one of
        the detected colors

    img_colors : np.ndarray
        The colors that were detected

    color_mask : np.ndarray
        Mask indicating which pixels were used for clustering

    color_counts : np.ndarray
        Count of each color in the image

    """
    if hue_only:
        img_colors, color_mask, color_counts = find_dominant_hues(img, cspace=cspace,
                                                min_colorfulness=min_colorfulness,
                                                px_thresh=px_thresh, n_hue_bins=n_hue_bins)
    else:
        img_colors, color_mask, color_counts = find_dominant_colors(img, cspace=cspace,
                                                    min_colorfulness=min_colorfulness,
                                                    px_thresh=px_thresh,
                                                    max_colors=max_colors,
                                                    n_colors=n_colors)

    if method == "deconvolve":
        unmix_D = stainmat2decon(img_colors)
        sep_img = deconvolve_img(img, unmix_D)

    elif method == "norm":
        # Cosine similarity https://www.geeksforgeeks.org/how-to-calculate-cosine-similarity-in-python/
        jab_img = rgb2jab(img, cspace=cspace)
        jab_colors = rgb2jab(img_colors.astype(np.uint8), cspace=cspace)

        jab_flat = jab_img.reshape((-1, 3))
        jab_min = jab_flat.min(axis=0)
        jab_max = jab_flat.max(axis=0)
        jab_range = jab_max - jab_min
        jab01 = (jab_flat - jab_min)/jab_range
        jab_colors01 = (jab_colors - jab_min)/jab_range

        jab_img_norm = np.linalg.norm(jab01, axis=1)
        jab_color_norms = np.linalg.norm(jab_colors01, axis=1)
        sep_img = np.dstack([jab01@jab_colors01[i]/(jab_img_norm*jab_color_norms[i]) for i in range(jab_colors.shape[0])])
        sep_img = sep_img.reshape((*jab_img.shape[0:2], jab_colors.shape[0]))

    elif method == "similarity":
        jab_img = rgb2jab(img, cspace=cspace)
        jab_colors = rgb2jab(img_colors.astype(np.uint8), cspace=cspace)
        jab_flat = jab_img.reshape((-1, 3))
        jab_min = jab_flat.min(axis=0)
        jab_max = jab_flat.max(axis=0)
        jab_range = jab_max - jab_min
        jab01 = (jab_flat - jab_min)/jab_range
        jab_colors01 = (jab_colors - jab_min)/jab_range

        dist_img = np.dstack([spatial.distance.cdist(jab01, jab_colors01[i].reshape(1, -1)).reshape(img.shape[0:2]) for i in range(jab_colors.shape[0])])
        dist_img = exposure.rescale_intensity(dist_img, out_range=(0, 1))
        sep_img = 1 - dist_img

    elif method == "svm":
        jab_img = rgb2jab(img, cspace=cspace)

        jab_flat = jab_img.reshape((-1, 3))
        jab_flat = StandardScaler().fit_transform(jab_flat)
        color_mask_flat = color_mask.reshape(-1)

        training_X = jab_flat[color_mask_flat > -1]
        training_Y = color_mask_flat[color_mask_flat > -1]

        svm = SVC(probability=True)
        svm.fit(training_X, training_Y)
        sep_img = svm.predict_proba(jab_flat).reshape((*img.shape[0:2], img_colors.shape[0]))

    elif method == "one_class_svm":
        jab_img = rgb2jab(img, cspace=cspace)
        jab_flat = jab_img.reshape((-1, 3))
        jab_flat = StandardScaler().fit_transform(jab_flat)
        color_mask_flat = color_mask.reshape(-1)

        sep_img = np.zeros((*img.shape[0:2], img_colors.shape[0]))
        for i in range(sep_img.shape[2]):
            label_X = jab_flat[color_mask_flat == i]
            label_Y = np.zeros(label_X.shape[0], dtype=int)

            other_idx = np.where((color_mask_flat != i) & (color_mask_flat > -1))[0]
            other_X = jab_flat[other_idx]
            other_Y = np.ones(other_X.shape[0], dtype=int)

            chnl_X = np.vstack([label_X, other_X])
            chnl_Y = np.hstack([label_Y, other_Y])
            idx = list(range(len(chnl_Y)))
            np.random.shuffle(idx)

            svm = SVC(probability=True)
            sep_img[..., i] = svm.fit(chnl_X[idx], chnl_Y[idx]).predict_proba(jab_flat)[..., 0].reshape(img.shape[0:2])

    return sep_img, img_colors, color_mask, color_counts


def find_dominant_hues(img, cspace="JzAzBz", min_colorfulness=0.005, px_thresh=0.0001, n_hue_bins=360, min_hue_dist=18, lamb=0):
    jab_img = rgb2jab(img, cspace=cspace)
    jch_img = colour.models.Jab_to_JCh(jab_img)

    if min_colorfulness == "auto":
        min_colorfulness = filters.threshold_otsu(jch_img[..., 1])

    fg_px = np.where(jch_img[..., 1] > min_colorfulness)

    fg_jab = jab_img[fg_px]
    h = jch_img[..., 2][fg_px]
    hue_step = 360//n_hue_bins
    h_hist, h_bins = np.histogram(h, bins=np.arange(0, 360, hue_step))
    if lamb > 0:
        # Smooth curve
        h_hist = signal.cspline1d_eval(signal.cspline1d(h_hist, lamb=lamb), np.arange(0, len(h_bins)))
        h_hist[h_hist < 0] = 0

    img_px_thresh = px_thresh*np.multiply(*img.shape[0:2])
    peak_idx, heights = signal.find_peaks(h_hist, height=img_px_thresh, distance=min_hue_dist)
    h_peaks = h_bins[peak_idx]
    n_peaks = len(h_peaks)
    mean_jch = np.zeros((n_peaks, 3))
    hue_mask = np.full(img.shape[0:2], -1, dtype=int)
    hue_label = 0
    for i in range(n_peaks):
        h_bin_midpoint = h_peaks[i] + hue_step/2
        bin_h_range = np.array([h_bin_midpoint - hue_step, h_bin_midpoint + hue_step])
        bin_h_range = np.clip(bin_h_range, 0, 360)
        in_range_idx = np.where((h >= bin_h_range[0]) & (h < bin_h_range[1]))[0]
        in_range_mean_jab = np.mean(fg_jab[in_range_idx], axis=0)
        in_range_mean_jc = colour.models.Jab_to_JCh(in_range_mean_jab)[0:2]
        in_range_mean_jch = np.hstack([in_range_mean_jc, h_peaks[i]])

        mean_jch[i] = in_range_mean_jch

        hue_mask[fg_px[0][in_range_idx], fg_px[1][in_range_idx]] = hue_label
        hue_label += 1

    mean_rgb_from_jch = jch2rgb(mean_jch, cspace=cspace)[0]
    bin_counts = heights["peak_heights"]
    if mean_rgb_from_jch.shape[0] > 1:
        order_idx = np.argsort(bin_counts)[::-1]
        mean_rgb_from_jch = mean_rgb_from_jch[order_idx]
        bin_counts = bin_counts[order_idx]

    return mean_rgb_from_jch, hue_mask, bin_counts



def find_dominant_colors(img, cspace="JzAzBz", min_colorfulness=0, px_thresh=0.0001, n_bins=50, max_colors=5, n_colors=None, cluster_estimation="unimodal"):
    """ Find most common colors in the image

    Initial colors are detected by converting the image to `cspace`, and then binning the A and B channels.
    Peaks in this 2D histogram are then used as the initial centroids for K-means clustering
    If `n_colors=None`, K-means clustering is used to detect 2-`max_colors` clusters, and then
    `cluster_estimation` is used to determine the number of clusters (i.e. colors). Colors that are
    close to these centoids are then averaged to find the representative color for each cluster.

    Parameters
    ----------
    img : np.ndarray
        RGB image

    cspace : str
        Colorspace to use to detect and separate colors using `separate_colors`

    min_colorfulness : str
        Pixels with colorfulness/saturation less that this will be exluded.
        Calculated after binning colors.

    px_thresh: float
        Minimal frequency of a color

    n_bins : int
        Number of bins to use when binning colors (A, B in JAB image)

    max_colors : int
        Expected maximum number of colors in the image. Number of colors detected
        will be equal to, or less than, this value.

    n_colors : int
        Number of colors in the image. If `None`,  `n_colors`will be determined by clustering

    hue_only : bool
        If `True`, use `find_dominant_hues` for color detection. If, `False`, use `find_dominant_colors`

    cluster_estimation : str
        How to estimate the number of colors in the image. Options are "unimodal" or "elbow"
        "unimodal" tends to detect more colors than "elbow".

    Returns
    -------
    mean_rgb  np.ndarray
        The RGB colors that were detected

    color_mask : np.ndarray
        Mask indicating which pixels were used for clustering

    filtered_label_counts : np.ndarray
        Count of each color in the image

    """

    jab_img = rgb2jab(img, cspace=cspace)
    jch_img = colour.models.Jab_to_JCh(jab_img)
    if min_colorfulness == "auto":
        min_colorfulness = filters.threshold_otsu(jch_img[..., 1])

    fg_px = np.where(jch_img[..., 1] > min_colorfulness)
    if len(fg_px[0]) == 0:
        min_colorfulness = filters.threshold_otsu(jch_img[..., 1])
        fg_px = np.where(jch_img[..., 1] > min_colorfulness)

    ab_hist, a_bins, b_bins = np.histogram2d(jab_img[..., 1][fg_px], jab_img[..., 2][fg_px], bins=(n_bins, n_bins))
    non_zero_a_idx, non_zero_b_idx = np.where(ab_hist > px_thresh*ab_hist.size)
    non_zero_a = a_bins[non_zero_a_idx]
    non_zero_b = b_bins[non_zero_b_idx]
    weights = ab_hist[non_zero_a_idx, non_zero_b_idx]
    xy = np.dstack([non_zero_a, non_zero_b])[0]

    color_coordinates = peak_local_max(ab_hist, num_peaks=max_colors)
    initial_centroid_x = a_bins[color_coordinates[:, 0]]
    initial_centroid_y = b_bins[color_coordinates[:, 1]]
    intial_xy = np.dstack([initial_centroid_x, initial_centroid_y])[0]

    sq_D = spatial.distance.cdist(intial_xy, intial_xy)
    max_D = sq_D.max()
    most_dif_2Didx = np.where(sq_D == max_D)  # 2 most different colors
    xy_idx1 = most_dif_2Didx[0][0]
    xy_idx2 = most_dif_2Didx[1][0]
    intial_cluster_centers = "k-means++"

    if n_colors is None:

        k_intertia = []
        cluster_number = []
        cluster_centroids = []
        for i in range(1, max_colors+1):
            if i >= xy.shape[0]:
                continue
            if i == 1:
                k_centroids_xy = "k-means++"
            elif i == 2:
                k_centroids_xy = np.array([intial_xy[xy_idx1], intial_xy[xy_idx2]])
            else:
                possible_idx = list(range(sq_D.shape[0]))
                possible_idx.remove(xy_idx1)
                possible_idx.remove(xy_idx2)
                diff_idx = [xy_idx1, xy_idx2]
                for j in range(2, i):
                    max_d_idx = np.argmax([np.min(sq_D[i, diff_idx]) for i in possible_idx])
                    new_idx = possible_idx[max_d_idx]
                    diff_idx.append(new_idx)
                    possible_idx.remove(new_idx)

                k_centroids_xy = intial_xy[diff_idx]

            temp_xy_clusterer = cluster.KMeans(n_clusters=i, init=k_centroids_xy, random_state=0)
            k_intertia.append(temp_xy_clusterer.inertia_)
            cluster_number.append(i)
            cluster_centroids.append(temp_xy_clusterer.cluster_centers_)

        k_intertia = np.array(k_intertia)
        if cluster_estimation == "elbow":
            elbow_idx = find_elbow(np.array(cluster_number), k_intertia)
        else:
            elbow_idx = np.where(k_intertia > thresh_unimodal(k_intertia))[0][-1]

        n_colors = cluster_number[elbow_idx]
        intial_cluster_centers = cluster_centroids[elbow_idx]

    xy_clusterer = cluster.KMeans(n_clusters=n_colors, init=intial_cluster_centers, random_state=0)
    labels = xy_clusterer.fit_predict(xy, sample_weight=weights)
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    if hasattr(xy_clusterer, "cluster_centers_"):
        xy_centroids = xy_clusterer.cluster_centers_
    else:
        xy_centroids = np.vstack([np.mean(xy[labels==i], axis=0) for i in unique_labels])

    fg_jab = jab_img[fg_px]

    def _get_in_range(d_thresh):
        color_mask = np.full(img.shape[0:2], -1, dtype=int)
        color_label = 0
        filtered_label_counts = []
        mean_jab = []
        for i, cent_xy in enumerate(xy_centroids):
            dist_to_centroid = np.sqrt(np.sum((fg_jab[..., 1:3] - cent_xy)**2, axis=1))
            in_range = np.where(dist_to_centroid < d_thresh)[0]
            if len(in_range) > 0:
                centroid_jab = np.mean(fg_jab[in_range], axis=0)
                mean_jab.append(centroid_jab)

                color_mask[fg_px[0][in_range], fg_px[1][in_range]] = color_label
                color_label += 1
                filtered_label_counts.append(label_counts[i])

        return mean_jab, filtered_label_counts, color_label, filtered_label_counts, color_mask

    dist_thresh = np.sqrt((a_bins[1] - a_bins[0])**2 + (b_bins[1] - b_bins[0])**2)

    max_reps = 100
    dscaler = 1
    mean_jab = []
    while len(mean_jab) == 0:
        mean_jab, filtered_label_counts, color_label, filtered_label_counts, color_mask = _get_in_range(dscaler*dist_thresh)
        dscaler += 1

        if dscaler > max_reps:
            break

    if len(mean_jab) == 0:
        mean_j = np.repeat(jch_img[fg_px][..., 0].mean(), len(xy_centroids))
        mean_jab = np.hstack([mean_j[..., np.newaxis], xy_centroids])

    mean_jab = np.vstack(mean_jab)
    mean_rgb = 255*jab2rgb(mean_jab, cspace=cspace)

    # Sort so that most common colors are first
    filtered_label_counts = np.array(filtered_label_counts)
    if mean_rgb.shape[0] > 1:
        order_idx = np.argsort(filtered_label_counts)[::-1]
        mean_rgb = mean_rgb[order_idx]
        filtered_label_counts = filtered_label_counts[order_idx]

    return mean_rgb, color_mask, filtered_label_counts



def mean_color(rgb_vals, summary_fxn=np.mean):
    jab_vals = rgb2jab(rgb_vals)
    mean_jab = summary_fxn(jab_vals, axis=0)
    mean_rgb = jab2rgb(mean_jab)
    mean_rgb = (255*np.clip(mean_rgb, 0, 1))
    return mean_rgb


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
    Replace objects in mask with bounding boxes. If `merge_bbox`
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
        corners_xy = cnt.squeeze()
        on_top_border = corners_xy[:, 1].min() == 0
        on_btm_border = corners_xy[:, 1].max() == mask.shape[0] - 1
        on_left_border = corners_xy[:, 0].min() == 0
        on_right_border = corners_xy[:, 0].max() == mask.shape[1] - 1

        on_border = any([on_top_border, on_btm_border, on_left_border, on_right_border])
        if not on_border:
            cv2.drawContours(contour_mask, [cnt], 0, 255, -1)
        else:
            # Need to close contours on the border in order to fill holes
            contour_w_border = np.zeros_like(mask)
            cv2.drawContours(contour_w_border, [cnt], 0, 255, -1)
            if on_top_border:
                # Check top border
                idx_at_min_y = np.where(corners_xy[:, 1] == 0)[0]
                min_x = corners_xy[:, 0][idx_at_min_y.min()]
                max_x = corners_xy[:, 0][idx_at_min_y.max()]
                contour_w_border[0, min_x:max_x] = 255

            if on_left_border:
                # Check left border
                idx_at_min_x = np.where(corners_xy[:, 0] == 0)[0]
                min_y = corners_xy[:, 1][idx_at_min_x.min()]
                max_y = corners_xy[:, 1][idx_at_min_x.max()]
                contour_w_border[min_y:max_y, 0] = 255

            if on_right_border:
                # Check right border
                max_x = mask.shape[1] - 1
                idx_at_max_x = np.where(corners_xy[:, 0] == max_x)[0]
                min_y = corners_xy[:, 1][idx_at_max_x.min()]
                max_y = corners_xy[:, 1][idx_at_max_x.max()]
                contour_w_border[min_y:max_y, max_x] = 255

            if on_btm_border:
                # Check bottom border
                max_y = mask.shape[0] - 1
                idx_at_max_y = np.where(corners_xy[:, 1] == max_y)[0]
                min_x = corners_xy[:, 0][idx_at_max_y.min()]
                max_x = corners_xy[:, 0][idx_at_max_y.max()]
                contour_w_border[max_y, min_x:max_x] = 255

            on_border_contours, _ = cv2.findContours(contour_w_border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for b_cnt in on_border_contours:
                cv2.drawContours(contour_mask, [b_cnt], 0, 255, -1)

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


def collect_img_stats(img_list, norm_percentiles=[1, 5, 95, 99], mask_list=None):

    use_masks = mask_list is not None
    if use_masks:
        use_masks = mask_list[0] is not None

    if use_masks:
        img0 = img_list[0][mask_list[0] > 0]
    else:
        img0 = img_list[0].reshape(-1)

    all_histogram, _ = np.histogram(img0, bins=256)

    n = img0.size
    total_x = img0.sum()
    for i in range(1, len(img_list)):
        img = img_list[i]
        if mask_list is None:
            img_flat = img.reshape(-1)
        else:
            if mask_list[i] is None:
                img_flat = img.reshape(-1)
            else:
                img_flat = img[mask_list[i] > 0]

        img_hist, _ = np.histogram(img_flat, bins=256)
        all_histogram += img_hist
        n += img.size
        total_x += img.sum()

    mean_x = total_x/n
    ref_cdf = 100*np.cumsum(all_histogram)/np.sum(all_histogram)
    all_img_stats = np.array([len(np.where(ref_cdf <= q)[0]) for q in norm_percentiles])
    all_img_stats = np.hstack([all_img_stats, mean_x])
    all_img_stats = all_img_stats[np.argsort(all_img_stats)]

    return all_histogram, all_img_stats


def norm_img_stats(img, target_stats, mask=None):
    """Normalize an image

    Image will be normalized to have same stats as `target_stats`

    Based on method in
    "A nonlinear mapping approach to stain normalization in digital histopathology
    images using image-specific color deconvolution.", Khan et al. 2014

    Assumes that `img` values range between 0-255

    """

    if mask is not None:
        if isinstance(mask, pyvips.Image):
            np_mask = warp_tools.vips2numpy(mask)
        else:
            np_mask = mask
    else:
        np_mask = None

    _, src_stats_flat = collect_img_stats([img], mask_list=[np_mask])

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

