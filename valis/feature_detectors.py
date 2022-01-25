"""Functions and classes to detect and describe image features

Bundles OpenCV feature detectors and descriptors into the FeatureDD class

Also makes it easier to mix and match feature detectors and descriptors
from different pacakges (e.g. skimage and OpenCV). See CensureVggFD for
an example

"""

import cv2
from skimage import feature, exposure
import numpy as np
from . import valtils

DEFAULT_FEATURE_DETECTOR = cv2.BRISK_create()
"""The default OpenCV feature detector"""

MAX_FEATURES = 20000
"""Maximum number of image features that will be recorded. If the number
of features exceeds this value, the MAX_FEATURES features with the
highest response will be returned."""


def filter_features(kp, desc, n_keep=MAX_FEATURES):
    """Get keypoints with highest response

    Parameters
    ----------
    kp : list
        List of cv2.KeyPoint detected by an OpenCV feature detector.

    desc : ndarray
        2D numpy array of keypoint descriptors, where each row is a keypoint
        and each column a feature.

    n_keep : int
        Maximum number of features that are retained.

    Returns
    -------
    Keypoints and and corresponding descriptors that the the n_keep highest
    responses.

    """

    response = np.array([x.response for x in kp])
    keep_idx = np.argsort(response)[::-1][0:n_keep]
    return [kp[i] for i in keep_idx], desc[keep_idx, :]


class FeatureDD(object):
    """Abstract class for feature detection and description.

    User can create other feature detectors as subclasses, but each must
    return keypoint positions in xy coordinates along with the descriptors
    for each keypoint.

    Note that in some cases, such as KAZE, kp_detector can also detect
    features. However, in other cases, there may need to be a separate feature
    detector (like BRISK or ORB) and feature descriptor (like VGG).

    Attributes
    ----------
        kp_detector : object
            Keypoint detetor, by default from OpenCV

        kp_descriptor : object
            Keypoint descriptor, by default from OpenCV

        kp_detector_name : str
            Name of keypoint detector

        kp_descriptor : str
            Name of keypoint descriptor

    Methods
    -------
    detectAndCompute(image, mask=None)
        Detects and describes keypoints in image

    """

    def __init__(self, kp_detector=None, kp_descriptor=None):
        """
        Parameters
        ----------
            kp_detector : object
                Keypoint detetor, by default from OpenCV

            kp_descriptor : object
                Keypoint descriptor, by default from OpenCV

        """

        self.kp_detector = kp_detector
        self.kp_descriptor = kp_descriptor

        if kp_descriptor is not None and kp_detector is not None:
            # User provides both a detector and descriptor #
            self.kp_descriptor_name = kp_descriptor.__class__.__name__
            self.kp_detector_name = kp_detector.__class__.__name__

        if kp_descriptor is None and kp_detector is not None:
            # Will be using kp_descriptor for detectAndCompute #
            kp_descriptor = kp_detector
            kp_detector = None

        if kp_descriptor is not None and kp_detector is None:
            # User provides a descriptor, which must also be able to detect #
            self.kp_descriptor_name = kp_descriptor.__class__.__name__
            self.kp_detector_name = self.kp_descriptor_name

            try:
                _img = np.zeros((10, 10), dtype=np.uint8)
                kp_descriptor.detectAndCompute(_img, mask=None)

            except:
                msg = f"{self.kp_descriptor_name} unable to both detect and compute features. Setting to {DEFAULT_FEATURE_DETECTOR.__class__.__name__}"
                valtils.print_warning(msg)

                self.kp_detector = DEFAULT_FEATURE_DETECTOR

    def detect_and_compute(self, image, mask=None):
        """Detect the features in the image

        Detect the features in the image using the defined kp_detector, then
        describe the features using the kp_descriptor. The user can override
        this method so they don't have to use OpenCV's Keypoint class.

        Parameters
        ----------
        image : ndarray
            Image in which the features will be detected. Should be a 2D uint8
            image if using OpenCV

        mask : ndarray, optional
            Binary image with same shape as image, where foreground > 0,
            and background = 0. If provided, feature detection  will only be
            performed on the foreground.

        Returns
        -------
        kp : ndarry
            (N, 2) array positions of keypoints in xy corrdinates for N
            keypoints

        desc : ndarry
            (N, M) array containing M features for each of the N keypoints

        """

        image = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
        if self.kp_detector is not None:
            detected_kp = self.kp_detector.detect(image)
            kp, desc = self.kp_descriptor.compute(image, detected_kp)
            type(desc)

        else:
            kp, desc = self.kp_descriptor.detectAndCompute(image, mask=mask)

        if desc.shape[0] > MAX_FEATURES:

            kp, desc = filter_features(kp, desc)

        kp_pos_xy = np.array([k.pt for k in kp])

        return kp_pos_xy, desc

# Thin wrappers around OpenCV detectors and descriptors #


class OrbFD(FeatureDD):
    """Uses ORB for feature detection and description"""
    def __init__(self, kp_descriptor=cv2.ORB_create(MAX_FEATURES)):
        super().__init__(kp_descriptor=kp_descriptor)


class BriskFD(FeatureDD):
    """Uses BRISK for feature detection and description"""
    def __init__(self, kp_descriptor=cv2.BRISK_create()):
        super().__init__(kp_descriptor=kp_descriptor)


class KazeFD(FeatureDD):
    """Uses KAZE for feature detection and description"""
    def __init__(self, kp_descriptor=cv2.KAZE_create(extended=False)):
        super().__init__(kp_descriptor=kp_descriptor)


class AkazeFD(FeatureDD):
    """Uses AKAZE for feature detection and description"""
    def __init__(self, kp_descriptor=cv2.AKAZE_create()):
        super().__init__(kp_descriptor=kp_descriptor)


class DaisyFD(FeatureDD):
    """Uses BRISK for feature detection and DAISY for feature description"""
    def __init__(self, kp_detector=DEFAULT_FEATURE_DETECTOR,
                 kp_descriptor=cv2.xfeatures2d.DAISY_create()):
        super().__init__(kp_detector=kp_detector, kp_descriptor=kp_descriptor)


class LatchFD(FeatureDD):
    """Uses BRISK for feature detection and LATCH for feature description"""
    def __init__(self, kp_detector=DEFAULT_FEATURE_DETECTOR,
                 kp_descriptor=cv2.xfeatures2d.LATCH_create(rotationInvariance=True)):
        super().__init__(kp_detector=kp_detector, kp_descriptor=kp_descriptor)


class BoostFD(FeatureDD):
    """Uses BRISK for feature detection and Boost for feature description"""
    def __init__(self, kp_detector=DEFAULT_FEATURE_DETECTOR,
                 kp_descriptor=cv2.xfeatures2d.BoostDesc_create()):
        super().__init__(kp_detector=kp_detector, kp_descriptor=kp_descriptor)


class VggFD(FeatureDD):
    """Uses BRISK for feature detection and VGG for feature description"""
    def __init__(self,  kp_detector=DEFAULT_FEATURE_DETECTOR,
                 kp_descriptor=cv2.xfeatures2d.VGG_create(scale_factor=6.25)):
        super().__init__(kp_detector=kp_detector, kp_descriptor=kp_descriptor)


class OrbVggFD(FeatureDD):
    """Uses ORB for feature detection and VGG for feature description"""
    def __init__(self,  kp_detector=cv2.ORB_create(nfeatures=MAX_FEATURES, fastThreshold=0), kp_descriptor=cv2.xfeatures2d.VGG_create(scale_factor=6.25)):
        super().__init__(kp_detector=kp_detector, kp_descriptor=kp_descriptor)


# Example of a custom detector that uses the Censure feature detector
# from scikit-image along with the KAZE descriptor (OpenCV)
class FeatureDetector(object):
    """Abstract class that detects features in an image

    Features should be returned in a list of OpenCV cv2.KeyPoint objects.
    Useful if wanting to use a non-OpenCV feature detector

    Attributes
    ----------
    detector : object
        Object that can detect image features.

    Methods
    -------
    detect(image)

    Interface
    ---------
    Required methods are: detect

    """
    def __init__(self):
        self.detector = None

    def detect(self, image):
        """
        Use detector to detect features, and return keypoints as XY

        Returns
        ---------
        kp : KeyPoints
            List of OpenCV KeyPoint objects

        """
        pass


# Example of how to create a feature detector using OpenCV + skimage #
class SkCensureDetector(FeatureDetector):
    """A CENSURE feature detector from scikit image

    This scikit-image feature detecotr can be used with an
    OpenCV feature descriptor

    """
    def __init__(self, **kwargs):
        super().__init__()
        self.detector = feature.CENSURE(**kwargs)

    def detect(self, image):
        """
        Detect keypoints in image using CENSURE.
        See https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.CENSURE

        Uses keypoint info to create KeyPoint objects for OpenCV

        Paramters
        ---------
        image : ndarray
            image from keypoints will be detected


        Returns
        ---------
        kp : KeyPoints
            List of OpenCV KeyPoint objects

        """
        self.detector.detect(image)

        # Skimage returns keypoints as row, col, but need to be returned as xy
        kp_xy = self.detector.keypoints[:, ::-1].astype(float)
        # Now create a list of OpenCV KeyPoint objects with these coordinates
        kp = cv2.KeyPoint_convert(kp_xy.tolist())

        return kp


class CensureVggFD(FeatureDD):
    def __init__(self, kp_detector=SkCensureDetector(mode="Octagon",
                 max_scale=8, non_max_threshold=0.02),
                 kp_descriptor=cv2.xfeatures2d.VGG_create(scale_factor=6.25)):

        super().__init__(kp_detector=kp_detector, kp_descriptor=kp_descriptor)
        self.kp_descriptor_name = self.__class__.__name__
        self.kp_detector_name = self.__class__.__name__


# Example of a custom detector and descriptor using scikit-image #
class SkDaisy(FeatureDD):
    def __init__(self, dasiy_arg_dict=None):
        """
        Create FeatureDD that uses scikit-image's dense DASIY
        https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_daisy.html#sphx-glr-auto-examples-features-detection-plot-daisy-py

        """
        self.dasiy_arg_dict = {"step": 4,
                               "radius": 15,
                               "rings": 3,
                               "histograms": 8,
                               "orientations": 8,
                               "normalization": "l1",
                               "sigmas": None,
                               "ring_radii": None,
                               "visualize": False
                               }

        if dasiy_arg_dict is not None:
            self.dasiy_arg_dict.update(dasiy_arg_dict)

        self.kp_descriptor_name = self.__class__.__name__
        self.kp_detector_name = self.__class__.__name__

    def detect_and_compute(self, image, mask=None):
        descs = feature.daisy(image, **self.dasiy_arg_dict)

        # Keypoints in a regular grid, and each point has a feature array #
        # Below determines grid and then gets features
        rows = np.arange(0, descs.shape[0])
        cols = np.arange(0, descs.shape[1])
        all_rows, all_cols = np.meshgrid(rows, cols)

        all_rows = all_rows.reshape(-1)
        all_cols = all_cols.reshape(-1)
        n_samples = len(all_rows)

        flat_desc = [descs[all_rows[i]][all_cols[i]] for i in range(n_samples)]
        desc2d = np.vstack(flat_desc)

        step = self.dasiy_arg_dict["step"]
        radius = self.dasiy_arg_dict["radius"]
        feature_x = all_cols * step + radius
        feature_y = all_rows * step + radius
        kp_xy = np.dstack([feature_x, feature_y])[0]

        return kp_xy, desc2d
