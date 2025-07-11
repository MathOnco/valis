"""Functions and classes to match and filter image features
"""
import torch
import kornia
import numpy as np
import cv2
import torch
from copy import deepcopy
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_kernels
from skimage import transform
import traceback

from . import warp_tools, valtils, feature_detectors
from .superglue_models import matching, superglue, superpoint

AMBIGUOUS_METRICS = set(metrics.pairwise._VALID_METRICS).intersection(
    metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS.keys())
"""set:
    Metrics found in both the valid metrics ang kernel methods in
    sklearn.metrics.pairwise. Issue is that metrics are distances,
    while kernels are similarities. Metrics in this set are assumed
    to be distaces, unless the metric_type parameter in match_descriptors
    is set to "similarity". """


EPS = np.finfo(float).eps
"""float: epsilon error to avoid division by 0"""

GMS_NAME = "GMS"
"""str: If filter_method parameter in match_desc_and_kp is set to this,
    Grid-based Motion Statistics will be used to remove poor matches"""

RANSAC_NAME = "RANSAC"
USAC_MAGSAC_NAME = "USAC_MAGSAC"
RANSAC_DICT = {
    RANSAC_NAME: cv2.RANSAC,
    USAC_MAGSAC_NAME: cv2.USAC_MAGSAC
}
DEFAULT_RANSAC_NAME = USAC_MAGSAC_NAME
"""str: If filter_method parameter in match_desc_and_kp is set to this,
RANSAC will be used to remove poor matches
"""



SUPERGLUE_FILTER_NAME = "superglue"
"""str: If filter_method parameter in match_desc_and_kp is set to this,
only SuperGlue will be used to remove poor matches
"""

DEFAULT_MATCH_FILTER = USAC_MAGSAC_NAME
"""str: The defulat filter_method value, either RANSAC_NAME or GMS_NAME"""

DEFAULT_RANSAC = 7
"""int: Default RANSAC threshold"""

DEFAULT_FD = feature_detectors.VggFD
ROTATION_ESTIMATOR_FD = feature_detectors.VggFD

def convert_distance_to_similarity(d, n_features=64):
    """
    Convert distance to similarity
    Based on https://scikit-learn.org/stable/modules/metrics.html

    Parameters
    ----------
    d : float
        Value to convert

    n_features: int
        Number of features used to calcuate distance.
        Only needed when calc == 0
    Returns
    -------
    y : float
        Similarity
    """
    return np.exp(-d * (1 / n_features))


def convert_similarity_to_distance(s, n_features=64):
    """Convert similarity to distance

    Based on https://scikit-learn.org/stable/modules/metrics.html

    Parameters
    ----------
    s : float
        Similarity to convert

    n_features: int
        Number of features used to calcuate similarity.
        Only needed when calc == 0

    Returns
    -------
    y : float
        Distance

    """

    return -np.log(s + EPS) / (1 / n_features)


def filter_matches_ransac(kp1_xy, kp2_xy, ransac_val=DEFAULT_RANSAC, method=USAC_MAGSAC_NAME):
    f"""Remove poor matches using RANSAC

    Parameters
    ----------
    kp1_xy : ndarray
        (N, 2) array containing image 1s keypoint positions, in xy coordinates.

    kp2_xy : ndarray
        (N, 2) array containing image 2s keypoint positions, in xy coordinates.

    ransac_val: int
        RANSAC threshold, passed to cv2.findHomography as the
        ransacReprojThreshold parameter. Default value is {DEFAULT_RANSAC}

    method : int
        Method used to find inliers. Passed to `method` argument of `cv2.findHomography`.

    Returns
    -------
    filtered_src_points : (N, 2) array
        Inlier keypoints from kp1_xy

    filtered_dst_points : (N, 2) array
        Inlier keypoints from kp1_xy

    good_idx : (1, N) array
        Indices of inliers

    """
    method_num = RANSAC_DICT[method]

    if kp1_xy.shape[0] >= 4:
        _, mask = cv2.findHomography(kp1_xy, kp2_xy, method_num, ransac_val)
        good_idx = np.where(mask.reshape(-1) == 1)[0]
        filtered_src_points = kp1_xy[good_idx, :]
        filtered_dst_points = kp2_xy[good_idx, :]
    else:
        traceback_msg = traceback.format_exc()
        msg = f"Need at least 4 keypoints for RANSAC filtering, but only have {kp1_xy.shape[0]}"
        valtils.print_warning(msg, traceback_msg=traceback_msg)
        filtered_src_points = kp1_xy.copy()
        filtered_dst_points = kp2_xy.copy()
        good_idx = np.arange(0, kp1_xy.shape[0])

    return filtered_src_points, filtered_dst_points, good_idx


def filter_matches_gms(kp1_xy, kp2_xy, feature_d, img1_shape, img2_shape,
                       scaling, thresholdFactor=6.0):
    """Filter matches using GMS (Grid-based Motion Statistics) [1]

    This filtering method does best when there are a large number of features,
    so the ORB detector is recommended

    Note that this function assumes the keypoints and distances have been
    sorted such that each keypoint in kp1_xy has the same index as the
    matching keypoint in kp2_xy andd corresponding feautre distance in
    feature_d. For example, kp1_xy[0] should have the corresponding keypoint
    at kp2_xy[0] and the corresponding feature distance at feature_d[0].


    Parameters
    ----------
    kp1_xy : ndarray
        (N, 2) array with image 1s keypoint positions, in xy coordinates, for
        each of the N matched descriptors in desc1

    kp2_xy : narray
        (N, 2) array with image 2s keypoint positions, in xy coordinates, for
        each of the N matched descriptors in desc2

    feature_d: ndarray
        Feature distances between corresponding keypoints

    img1_shape: tuple
        Shape of image 1 (row, col)

    img2_shape: tuple
        Shape of image 2 (row, col)

    scaling: bool
        Whether or not image scaling should be considered

    thresholdFactor: float
        The higher, the fewer matches

    Returns
    -------
    filtered_src_points : (N, 2) array
        Inlier keypoints from kp1_xy

    filtered_dst_points : (N, 2) array
        Inlier keypoints from kp1_xy

    good_idx : (1, N) array
        Indices of inliers

    References
    ----------
    .. [1] JiaWang Bian, Wen-Yan Lin, Yasuyuki Matsushita, Sai-Kit Yeung,
    Tan Dat Nguyen, and Ming-Ming Cheng. Gms: Grid-based motion statistics for
    fast, ultra-robust feature correspondence. In IEEE Conference on Computer
    Vision and Pattern Recognition, 2017

    """

    kp1 = cv2.KeyPoint_convert(kp1_xy.tolist())
    kp2 = cv2.KeyPoint_convert(kp2_xy.tolist())
    matches = [cv2.DMatch(_queryIdx=i, _trainIdx=i, _imgIdx=0, _distance=feature_d[i]) for i in range(len(kp1_xy))]
    gms_matches = cv2.xfeatures2d.matchGMS(img1_shape, img2_shape, kp1, kp2, matches, withRotation=True,
                                           withScale=scaling, thresholdFactor=thresholdFactor)
    good_idx = np.array([d.queryIdx for d in gms_matches])

    if len(good_idx) == 0:
        filtered_src_points = []
        filtered_dst_points = []
    else:
        filtered_src_points = kp1_xy[good_idx, :]
        filtered_dst_points = kp2_xy[good_idx, :]

    return np.array(filtered_src_points), np.array(filtered_dst_points), np.array(good_idx)


def filter_matches_tukey(src_xy, dst_xy, tform=transform.SimilarityTransform()):
    """Detect and remove outliers using Tukey's method
    Adapted from https://towardsdatascience.com/detecting-and-treating-outliers-in-python-part-1-4ece5098b755

    Parameters
    ----------
    src_xy : ndarray
        (N, 2) array containing image 1s keypoint positions, in xy coordinates.

    dst_xy : ndarray
        (N, 2) array containing image 2s keypoint positions, in xy coordinates.

    Returns
    -------
    filtered_src_points : (N, 2) array
        Inlier keypoints from kp1_xy

    filtered_dst_points : (N, 2) array
        Inlier keypoints from kp1_xy

    good_idx : (1, N) array
        Indices of inliers

    """

    tform.estimate(src=dst_xy, dst=src_xy)
    M = tform.params

    warped_xy = warp_tools.warp_xy(src_xy, M)
    d = warp_tools.calc_d(warped_xy,  dst_xy)

    q1 = np.quantile(d, 0.25)
    q3 = np.quantile(d, 0.75)
    iqr = q3-q1
    inner_fence = 1.5*iqr
    outer_fence = 3*iqr

    # inner fence lower and upper end
    inner_fence_le = q1-inner_fence
    inner_fence_ue = q3+inner_fence

    # outer fence lower and upper end
    outer_fence_le = q1-outer_fence
    outer_fence_ue = q3+outer_fence

    outliers_prob = []
    outliers_poss = []
    inliers_prob = []
    inliers_poss = []
    for index, v in enumerate(d):
        if v <= outer_fence_le or v >= outer_fence_ue:
            outliers_prob.append(index)
        else:
            inliers_prob.append(index)
    for index, v in enumerate(d):
        if v <= inner_fence_le or v >= inner_fence_ue:
            outliers_poss.append(index)
        else:
            inliers_poss.append(index)

    src_xy_inlier = src_xy[inliers_prob, :]
    dst_xy_inlier = dst_xy[inliers_prob, :]

    return src_xy_inlier, dst_xy_inlier, inliers_prob


def filter_matches(kp1_xy, kp2_xy, method=DEFAULT_MATCH_FILTER,
                   filtering_kwargs={}):
    """Use RANSAC or GMS to remove poor matches

    Parameters
    ----------
    kp1_xy : ndarray
        (N, 2) array containing image 1s keypoint positions, in xy coordinates.

    kp2_xy : ndarray
        (N, 2) array containing image 2s keypoint positions, in xy coordinates.

    method: str
        `method` = "GMS" will use filter_matches_gms() to remove poor matches.
        This uses the Grid-based Motion Statistics.
        `method` = "RANSAC" will use RANSAC to remove poor matches

    filtering_kwargs: dict
        Extra arguments passed to filtering function

        If `method` == "GMS", these need to include: img1_shape, img2_shape,
        scaling, thresholdFactor. See filter_matches_gms for details

        If `method` == "RANSAC", this can be None, since the ransac value is
        a class attribute

    Returns
    -------
    filtered_src_points : ndarray
        (M, 2) ndarray of inlier keypoints from kp1_xy

    filtered_dst_points : (N, 2) array
        (M, 2) ndarray of inlier keypoints from kp2_xy

    good_idx : ndarray
        (M, 1) array containing ndices of inliers

    """

    all_matching_args = filtering_kwargs.copy()
    all_matching_args.update({"kp1_xy": kp1_xy, "kp2_xy": kp2_xy})
    if method.upper() == GMS_NAME:
        filter_fxn = filter_matches_gms
    elif method.upper() in RANSAC_DICT.keys():
        filter_fxn = filter_matches_ransac
        all_matching_args.update({"method": method})

    filtered_src_points, filtered_dst_points, good_idx = filter_fxn(**all_matching_args)

    # Do additional filtering to remove other outliers that may have been missed by RANSAC
    filtered_src_points, filtered_dst_points, good_idx = filter_matches_tukey(filtered_src_points, filtered_dst_points)
    return filtered_src_points, filtered_dst_points, good_idx


def match_descriptors(descriptors1, descriptors2, metric=None,
                      metric_type=None, p=2, max_distance=np.inf,
                      cross_check=True, max_ratio=1.0, metric_kwargs=None):
    """Brute-force matching of descriptors

    For each descriptor in the first set this matcher finds the closest
    descriptor in the second set (and vice-versa in the case of enabled
    cross-checking).


    Parameters
    ----------
    descriptors1 : ndarray
        (M, P) array of descriptors of size P about M keypoints in image 1.

    descriptors2 : ndarray
        (N, P) array of descriptors of size P about N keypoints in image 2.

    metric : string or callable
        Distance metrics used in spatial.distance.cdist() or sklearn.metrics.pairwise()
        Alterntively, can also use similarity metrics in sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS.
        By default the L2-norm is used for all descriptors of dtype float or
        double and the Hamming distance is used for binary descriptors automatically.

    p : int, optional
        The p-norm to apply for ``metric='minkowski'``.

    max_distance : float, optional
        Maximum allowed distance between descriptors of two keypoints
        in separate images to be regarded as a match.

    cross_check : bool, optional
        If True, the matched keypoints are returned after cross checking i.e. a
        matched pair (keypoint1, keypoint2) is returned if keypoint2 is the
        best match for keypoint1 in second image and keypoint1 is the best
        match for keypoint2 in first image.

    max_ratio : float, optional
        Maximum ratio of distances between first and second closest descriptor
        in the second set of descriptors. This threshold is useful to filter
        ambiguous matches between the two descriptor sets. The choice of this
        value depends on the statistics of the chosen descriptor, e.g.,
        for SIFT descriptors a value of 0.8 is usually chosen, see
        D.G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints",
        International Journal of Computer Vision, 2004.

    metric_kwargs : dict
        Optionl keyword arguments to be passed into pairwise_distances() or pairwise_kernels()
        from the sklearn.metrics.pairwise module

    Returns
    -------
    matches : (Q, 2) array
        Indices of corresponding matches in first and second set of
        descriptors, where ``matches[:, 0]`` denote the indices in the first
        and ``matches[:, 1]`` the indices in the second set of descriptors.

    distances : (Q, 1) array
        Distance values between each pair of matched descriptor

    metric_name : str or function
        Name metric used to calculate distances or similarity

    NOTE
    ----
    Modified from scikit-image to use scikit-learn's distance and kernal methods.
    """

    if descriptors1.shape[1] != descriptors2.shape[1]:
        raise ValueError("Descriptor length must equal.")

    if metric is None:
        if np.issubdtype(descriptors1.dtype, np.bool_):
            metric = 'hamming'
        else:
            metric = 'euclidean'

    if metric_kwargs is None:
        metric_kwargs = {}

    if metric == 'minkowski':
        metric_kwargs['p'] = p

    if metric in AMBIGUOUS_METRICS:
        print("metric", metric, "could be a distance in pairwise_distances() or similarity in pairwise_kernels().",
              "Please set metric_type. Otherwise, metric is assumed to be a distance")
    if callable(metric) or metric in metrics.pairwise._VALID_METRICS:

        distances = metrics.pairwise_distances(descriptors1, descriptors2, metric=metric, **metric_kwargs)
        if callable(metric) and metric_type is None:
            print(Warning("Metric passed as a function or class, but the metric type not provided",
                          "Assuming the metric function returns a distance. If a similarity is actually returned",
                          "set metric_type = 'similiarity'. If metric is a distance, set metric_type = 'distance'"
                          "to avoid this message"))

            metric_type = "distance"
        if metric_type == "similarity":
            distances = convert_similarity_to_distance(distances, n_features=descriptors1.shape[1])
    if metric in metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS:
        similarities = pairwise_kernels(descriptors1, descriptors2, metric=metric, **metric_kwargs)
        distances = convert_similarity_to_distance(similarities, n_features=descriptors1.shape[1])

    if callable(metric):
        metric_name = metric.__name__
    else:
        metric_name = metric

    indices1 = np.arange(descriptors1.shape[0])
    indices2 = np.argmin(distances, axis=1)

    if cross_check:
        matches1 = np.argmin(distances, axis=0)
        mask = indices1 == matches1[indices2]
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_distance < np.inf:
        mask = distances[indices1, indices2] < max_distance
        indices1 = indices1[mask]
        indices2 = indices2[mask]

    if max_ratio < 1.0:
        best_distances = distances[indices1, indices2]
        distances[indices1, indices2] = np.inf
        second_best_indices2 = np.argmin(distances[indices1], axis=1)
        second_best_distances = distances[indices1, second_best_indices2]
        second_best_distances[second_best_distances == 0] \
            = np.finfo(np.double).eps
        ratio = best_distances / second_best_distances
        mask = ratio < max_ratio
        indices1 = indices1[mask]
        indices2 = indices2[mask]

        return np.column_stack((indices1, indices2)), best_distances[indices1, indices2], metric, metric_type
    else:

        return np.column_stack((indices1, indices2)), distances[indices1, indices2], metric_name, metric_type


def match_desc_and_kp(desc1, kp1_xy, desc2, kp2_xy, metric=None, feature_detector_name=None,
                      metric_type=None, metric_kwargs=None, max_ratio=1.0,
                      filter_method=DEFAULT_MATCH_FILTER,
                      filtering_kwargs=None):
    """Match the descriptors of image 1 with those of image 2 and remove outliers.

    Metric can be a string to use a distance in scipy.distnce.cdist(),
    or a custom distance function

    Parameters
    ----------
        desc1 : ndarray
            (N, P) array of image 1's descriptions for N keypoints,
            which each keypoint having P features

        kp1_xy : ndarray
            (N, 2) array containing image 1's keypoint positions (xy)

        desc2 : ndarray
            (M, P) array of image 2's descriptions for M keypoints,
            which each keypoint having P features

        kp2_xy : (M, 2) array
            (M, 2) array containing image 2's keypoint positions (xy)

        feature_detector : str
            Name of feature descriptor use to match images

        metric: string, or callable
            Metric to calculate distance between each pair of features
            in desc1 and desc2. Can be a string to use as distance in
            spatial.distance.cdist, or a custom distance function

        metric_kwargs : dict
            Optionl keyword arguments to be passed into pairwise_distances()
            or pairwise_kernels() from the sklearn.metrics.pairwise module

        max_ratio : float, optional
            Maximum ratio of distances between first and second closest descriptor
            in the second set of descriptors. This threshold is useful to filter
            ambiguous matches between the two descriptor sets. The choice of this
            value depends on the statistics of the chosen descriptor, e.g.,
            for SIFT descriptors a value of 0.8 is usually chosen, see
            D.G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints",
            International Journal of Computer Vision, 2004.

        filter_method: str
            "GMS" will use uses the Grid-based Motion Statistics
            "RANSAC" will use RANSAC

        filtering_kwargs: dict
            Dictionary containing extra arguments for the filtering method.
            kp1_xy, kp2_xy, feature_d are calculated here, and don't need to
            be in filtering_kwargs. If filter_method == "GMS", then the
            required arguments are: img1_shape, img2_shape, scaling,
            thresholdFactor. See filter_matches_gms for details.

            If filter_method == "RANSAC", then the required
            arguments are: ransac_val. See filter_matches_ransac for details.

        Returns
        -------

        match_info12 : MatchInfo
                Contains information regarding the matches between image 1 and
                image 2. These results haven't undergone filtering, so
                contain many poor matches.

        filtered_match_info12 : MatchInfo
                Contains information regarding the matches between image 1 and
                image 2. These results have undergone filtering, and so
                contain good matches

        match_info21 : MatchInfo
                Contains information regarding the matches between image 2 and
                image 1. These results haven't undergone filtering, so contain
                many poor matches.

        filtered_match_info21 : MatchInfo
                Contains information regarding the matches between image 2 and
                image 1. These results have undergone filtering, and so contain
                good matches

    """

    if metric_kwargs is None:
        metric_kwargs = {}

    if filter_method.upper() == GMS_NAME:
        # GMS is supposed to perform best with a large number of features #
        cross_check = False
    elif filter_method.upper() in RANSAC_DICT.keys():
        cross_check = True

    matches, match_distances, metric_name, metric_type = \
        match_descriptors(desc1, desc2, metric=metric,
                          metric_type=metric_type,
                          metric_kwargs=metric_kwargs,
                          max_ratio=max_ratio,
                          cross_check=cross_check)

    desc1_match_idx = matches[:, 0]
    matched_kp1_xy = kp1_xy[desc1_match_idx, :]
    matched_desc1 = desc1[desc1_match_idx, :]

    desc2_match_idx = matches[:, 1]
    matched_kp2_xy = kp2_xy[desc2_match_idx, :]
    matched_desc2 = desc2[desc2_match_idx, :]

    mean_unfiltered_distance = np.mean(match_distances)
    mean_unfiltered_similarity = np.mean(convert_distance_to_similarity(match_distances, n_features=desc1.shape[1]))

    match_info12 = MatchInfo(matched_kp1_xy=matched_kp1_xy, matched_desc1=matched_desc1,
                             matches12=desc1_match_idx, matched_kp2_xy=matched_kp2_xy,
                             matched_desc2=matched_desc2, matches21=desc2_match_idx,
                             match_distances=match_distances, distance=mean_unfiltered_distance,
                             similarity=mean_unfiltered_similarity, metric_name=metric_name,
                             metric_type=metric_type, feature_detector_name=feature_detector_name)

    match_info21 = MatchInfo(matched_kp1_xy=matched_kp2_xy, matched_desc1=matched_desc2,
                             matches12=desc2_match_idx, matched_kp2_xy=matched_kp1_xy,
                             matched_desc2=matched_desc1, matches21=desc1_match_idx,
                             match_distances=match_distances, distance=mean_unfiltered_distance,
                             similarity=mean_unfiltered_similarity, metric_name=metric_name,
                             metric_type=metric_type, feature_detector_name=feature_detector_name)

    # Filter matches #
    all_filtering_kwargs = {"kp1_xy": matched_kp1_xy, "kp2_xy": matched_kp2_xy}
    if filtering_kwargs is None:
        if filter_method not in RANSAC_DICT.keys():
            print(Warning(f"filtering_kwargs not provided for {filter_method} match filtering. Will use {DEFAULT_RANSAC_NAME} instead"))
            filter_method = DEFAULT_RANSAC_NAME
            all_filtering_kwargs.update({"ransac_val": DEFAULT_RANSAC})
        else:
            all_filtering_kwargs.update({"ransac_val": DEFAULT_RANSAC})
    else:
        all_filtering_kwargs.update(filtering_kwargs)
        if filter_method == GMS_NAME:
            # At this point, filtering_kwargs needs to include:
            # img1_shape, img2_shape, scaling, and thresholdFactor.
            # Already added kp1_xy, kp2_xy. Now adding feature_d to
            # the argument dictionary

            all_filtering_kwargs.update({"feature_d": match_distances})

    filtered_matched_kp1_xy, filtered_matched_kp2_xy, good_matches_idx = \
        filter_matches(matched_kp1_xy, matched_kp2_xy, filter_method, all_filtering_kwargs)

    if len(good_matches_idx) > 0:
        filterd_match_distances = match_distances[good_matches_idx]
        filterd_matched_desc1 = matched_desc1[good_matches_idx, :]
        filterd_matched_desc2 = matched_desc2[good_matches_idx, :]

        good_matches12 = desc1_match_idx[good_matches_idx]
        good_matches21 = desc2_match_idx[good_matches_idx]

        mean_filtered_distance = np.mean(filterd_match_distances)
        mean_filtered_similarity = \
            np.mean(convert_distance_to_similarity(filterd_match_distances,
                                                   n_features=desc1.shape[1]))
    else:
        filterd_match_distances = []
        filterd_matched_desc1 = []
        filterd_matched_desc2 = []

        good_matches12 = []
        good_matches21 = []

        mean_filtered_distance = np.inf
        mean_filtered_similarity = 0

    # Record filtered matches
    filtered_match_info12 = MatchInfo(matched_kp1_xy=filtered_matched_kp1_xy, matched_desc1=filterd_matched_desc1,
                                      matches12=good_matches12, matched_kp2_xy=filtered_matched_kp2_xy,
                                      matched_desc2=filterd_matched_desc2, matches21=good_matches21,
                                      match_distances=filterd_match_distances, distance=mean_filtered_distance,
                                      similarity=mean_filtered_similarity, metric_name=metric_name,
                                      metric_type=metric_type, feature_detector_name=feature_detector_name)

    filtered_match_info21 = MatchInfo(matched_kp1_xy=filtered_matched_kp2_xy, matched_desc1=filterd_matched_desc2,
                                      matches12=good_matches21, matched_kp2_xy=filtered_matched_kp1_xy,
                                      matched_desc2=filterd_matched_desc1, matches21=good_matches12,
                                      match_distances=filterd_match_distances, distance=mean_filtered_distance,
                                      similarity=mean_filtered_similarity, metric_name=metric_name,
                                      metric_type=metric_type, feature_detector_name=feature_detector_name)

    return match_info12, filtered_match_info12, match_info21, filtered_match_info21


class MatchInfo(object):
    """Class that stores information related to matches. One per pair of images

    All attributes are all set as parameters during initialization
    """

    def __init__(self,
                 matched_kp1_xy, matched_desc1, matches12,
                 matched_kp2_xy, matched_desc2, matches21,
                 match_distances, distance, similarity,
                 metric_name, metric_type,
                 img1_name=None, img2_name=None,
                 feature_detector_name=None):

        """Stores information about matches and features

        Parameters
        ----------
        matched_kp1_xy : ndarray
            (Q, 2) array of image 1 keypoint xy coordinates after filtering

        matched_desc1 : ndarray
            (Q, P) array of matched descriptors for image 1, each of which has P features

        matches12 : ndarray
            (1, Q) array of indices of featiures in image 1 that matched those in image 2

        matched_kp2_xy : ndarray
            (Q, 2) array containing Q matched image 2 keypoint xy coordinates after filtering

        matched_desc2 : ndarray
            (Q, P) containing Q matched descriptors for image 2, each of which has P features

        matches21 : ndarray
            (1, Q) containing indices of featiures in image 2 that matched those in image 1

        match_distances : ndarray
            Distances between each of the Q pairs of matched descriptors

        n_matches : int
            Number of good matches (i.e. the number of inlier keypoints)

        distance : float
            Mean distance of features

        similarity : float
            Mean similarity of features

        metric_name : str
            Name of metric

        metric_type : str
            "distance" or "similarity"

        img1_name : str
            Name of the image that kp1 and desc1 belong to

        img2_name : str
            Name of the image that kp2 and desc2 belong to

        feature_detector_name : str
            Name of feature descriptor use to match images

        """

        self.matched_kp1_xy = matched_kp1_xy
        self.matched_desc1 = matched_desc1
        self.matches12 = matches12
        self.matched_kp2_xy = matched_kp2_xy
        self.matched_desc2 = matched_desc2
        self.matches21 = matches21
        self.match_distances = match_distances
        self.n_matches = len(match_distances)
        self.distance = distance
        self.similarity = similarity
        self.metric_name = metric_name
        self.metric_type = metric_type
        self.img1_name = img1_name
        self.img2_name = img2_name
        self.feature_detector_name = feature_detector_name

    def set_names(self, img1_name, img2_name):
        self.img1_name = img1_name
        self.img2_name = img2_name


class Matcher(object):
    """Class that matchs the descriptors of image 1 with those of image 2

    Outliers removed using RANSAC or GMS

    Attributes
    ----------
    metric: str, or callable
        Metric to calculate distance between each pair of features in
        desc1 and desc2. Can be a string to use as distance in
        spatial.distance.cdist, or a custom distance function

    metric_name: str
        Name metric used. Will be the same as metric if metric is string.
        If metric is function, this will be the name of the function.

    metric_type: str, or callable
        String describing what the custom metric function returns, e.g.
        'similarity' or 'distance'. If None, and metric is a function it
        is assumed to be a distance, but there will be a warning that this
        variable should be provided to either define that it is a
        similarity, or to avoid the warning by having
        metric_type='distance' In the case of similarity, the number of
        features will be used to convert distances

    ransac : int
        The residual threshold to determine if a match is an inlier.
        Only used if filter_method == {RANSAC_NAME}. Default is "RANSAC"

    gms_threshold : int
        Used when filter_method is "GMS".
        The higher, the fewer matches.

    scaling: bool
        Whether or not image scaling should be considered when
        filter_method is "GMS"

    metric_kwargs : dict
        Keyword arguments passed into the metric when calling
        spatial.distance.cdist

    match_filter_method: str
        "GMS" will use filter_matches_gms() to remove poor matches.
        This uses the Grid-based Motion Statistics (GMS) or RANSAC.

    """

    def __init__(self, feature_detector=DEFAULT_FD(), metric=None, metric_type=None, metric_kwargs=None,
                 match_filter_method=DEFAULT_MATCH_FILTER, ransac_thresh=DEFAULT_RANSAC,
                 gms_threshold=15, scaling=False):
        """
        Parameters
        ----------

        feature_detector : FeatureDD, optional
            FeatureDD object detects and computes image features.
            available feature_detectors are found in the `feature_detectors`
            module. If a desired feature detector is not available,
            one can be created by subclassing `feature_detectors.FeatureDD`.

        metric: str, or callable
            Metric to calculate distance between each pair of features in
            desc1 and desc2. Can be a string to use as distance in
            spatial.distance.cdist, or a custom distance function

        metric_type: str, or callable
            String describing what the custom metric function returns, e.g.
            'similarity' or 'distance'. If None, and metric is a function it
            is assumed to be a distance, but there will be a warning that this
            variable should be provided to either define that it is a
            similarity, or to avoid the warning by having
            metric_type='distance' In the case of similarity, the number of
            features will be used to convert distances

        metric_kwargs : dict
            Keyword arguments passed into the metric when calling
            spatial.distance.cdist

        filter_method: str
            "GMS" will use filter_matches_gms() to remove poor matches.
            This uses the Grid-based Motion Statistics (GMS) or RANSAC.

        ransac_val : int
            The residual threshold to determine if a match is an inlier.
            Only used if filter_method is "RANSAC".

        gms_threshold : int
            Used when filter_method is "GMS".
            The higher, the fewer matches.

        scaling: bool
            Whether or not image scaling should be considered when
            filter_method is "GMS".

        """

        self.feature_detector = feature_detector
        self.feature_name = feature_detector.__class__.__name__

        self.metric = metric
        if metric is not None:
            if isinstance(metric, str):
                self.metric_name = metric
            elif callable(metric):
                self.metric_name = metric.__name__
        else:
            self.metric_name = None

        self.metric_type = metric_type
        self.ransac = ransac_thresh
        self.gms_threshold = gms_threshold
        self.scaling = scaling
        self.metric_kwargs = metric_kwargs
        self.match_filter_method = match_filter_method
        self.rotation_invariant = True

    def match_images(self, img1=None, desc1=None, kp1_xy=None,
                     img2=None, desc2=None, kp2_xy=None, additional_filtering_kwargs=None, *args, **kwargs):

        """Match the descriptors of image 1 with those of image 2,
        Outliers removed using match_filter_method. Metric can be a string
        to use a distance in scipy.distnce.cdist(), or a custom distance
        function. Sets atttributes for Matcher object

        Parameters
        ----------
        img1 : (N, M) array
            Image to match to `img2`

        desc1 : (N, P) array
            Image 1s 2D array containinng N keypoints, each of which
            has P features

        kp1_xy : (N, 2) array
            Image 1s keypoint positions, in xy coordinates,  for each of the
            N descriptors in desc1

        img2 : (N, M) array
            Image to match to `img1`

        desc2 : (M, P) array
            Image 2s 2D array containinng M keypoints, each of which has
            P features

        kp2_xy : (M, 2) array
            Image 1s keypoint positions, in xy coordinates, for each of
            the M descriptors in desc2

        additional_filtering_kwargs: dict, optional
            Extra arguments passed to filtering function
            If self.match_filter_method == "GMS", these need to
            include: img1_shape, img2_shape. See filter_matches_gms for details
            If If self.match_filter_method == "RANSAC", this can be None,
            since the ransac value is class attribute

        Returns
        -------
        match_info12 : MatchInfo
                Contains information regarding the matches between image 1
                and image 2. These results haven't undergone filtering,
                so contain many poor matches.

        filtered_match_info12 : MatchInfo
                Contains information regarding the matches between image 1
                and image 2. These results have undergone
                filtering, and so contain good matches

        match_info21 : MatchInfo
                Contains information regarding the matches between image 2
                and image 1. These results haven't undergone filtering, so
                contain many poor matches.

        filtered_match_info21 : MatchInfo
                Contains information regarding the matches between image 2
                and image 1.

        """

        using_ransac = self.match_filter_method in RANSAC_DICT.keys()

        if self.match_filter_method == GMS_NAME:
            if additional_filtering_kwargs is not None:
                # At this point arguments need to include: img1_shape, img2_shape #
                filtering_kwargs = additional_filtering_kwargs.copy()
                filtering_kwargs.update({"scaling": self.scaling,
                                         "thresholdFactor": self.gms_threshold})
            else:
                print(Warning(f"Selected {self.match_filter_method},\
                              but did not provide argument\
                              additional_filtering_kwargs.\
                              Defaulting to RANSAC"))

                self.match_filter_method = DEFAULT_RANSAC_NAME
                filtering_kwargs = {"ransac_val": self.ransac}

        elif using_ransac:
            filtering_kwargs = {"ransac_val": self.ransac}

        else:
            print(Warning(f"Dont know {self.match_filter_method}.\
                Defaulting to RANSAC"))

            self.match_filter_method = DEFAULT_RANSAC_NAME
            filtering_kwargs = {"ransac_val": self.ransac}

        if desc1 is None and kp1_xy is None and img1 is not None:
            kp1_xy, desc1 = self.feature_detector.detect_and_compute(img1)

        if desc2 is None and kp2_xy is None and img2 is not None:
            kp2_xy, desc2 = self.feature_detector.detect_and_compute(img2)

        match_info12, filtered_match_info12, match_info21, filtered_match_info21 = \
            match_desc_and_kp(desc1=desc1, kp1_xy=kp1_xy, desc2=desc2, kp2_xy=kp2_xy,
                              metric=self.metric, metric_type=self.metric_type,
                              metric_kwargs=self.metric_kwargs,
                              filter_method=self.match_filter_method,
                              filtering_kwargs=filtering_kwargs,
                              feature_detector_name=self.feature_name)

        if self.metric_name is None:
            self.metric_name = match_info12.metric_name

        return match_info12, filtered_match_info12, match_info21, filtered_match_info21

    def estimate_rotation(self, moving_img=None, fixed_img=None, moving_kp_xy=None, fixed_kp_xy=None, angle_estimator=None, *args, **kwargs):
        """
        Use a rotation invariant feature descriptor to estimate angle to rotate moving_img to align with fixed_img.
        Match Info should contain the matches where kp1 refers to the fixed image, and kp2 refers to the moving image
        """
        if (moving_kp_xy is None or fixed_kp_xy is None) and (fixed_img is not None and moving_img is not None):
            if angle_estimator is None:
                angle_estimator = ROTATION_ESTIMATOR_FD()
            kp1_xy, desc1 = angle_estimator.detect_and_compute(fixed_img)
            kp2_xy, desc2 = angle_estimator.detect_and_compute(moving_img)

            _, match_info, _, _ = match_desc_and_kp(desc1, kp1_xy, desc2, kp2_xy)

            fixed_kp_xy = match_info.matched_kp1_xy
            moving_kp_xy = match_info.matched_kp2_xy

        angle_estimator = transform.SimilarityTransform()
        angle_estimator.estimate(fixed_kp_xy, moving_kp_xy) # Estimates inverse transform, and want 2 to align to 1
        rot_deg = np.rad2deg(angle_estimator.rotation)

        return rot_deg

class SuperGlueMatcher(Matcher):
    """Use SuperGlue to match images (`match_images`)

    Implementation adapted from https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/match_pairs.py

    References
    -----------
    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """

    def __init__(self, feature_detector=feature_detectors.SuperPointFD(), weights="indoor", keypoint_threshold=0.005, nms_radius=4,
                 sinkhorn_iterations=100, match_threshold=0.2, force_cpu=False,
                 metric=None, metric_type=None, metric_kwargs=None,
                 match_filter_method=DEFAULT_MATCH_FILTER, ransac_thresh=DEFAULT_RANSAC,
                 gms_threshold=15, scaling=False):

        """

        Use SuperGlue to match images (`match_images`)

        Adapted from https://github.com/magicleap/SuperGluePretrainedNetwork/blob/master/match_pairs.py

        Parameters
        ----------
        weights : str
            SuperGlue weights. Options= ["indoor", "outdoor"]

        keypoint_threshold : float
            SuperPoint keypoint detector confidence threshold

        nms_radius : int
            SuperPoint Non Maximum Suppression (NMS) radius (must be positive)

        sinkhorn_iterations : int
            Number of Sinkhorn iterations performed by SuperGlue

        match_threshold : float
            SuperGlue match threshold

        force_cpu : bool
            Force pytorch to run in CPU mode

        scaling: bool
            Whether or not image scaling should be considered when
            filter_method is "GMS".
        """

        super().__init__(metric=metric, metric_type=metric_type, metric_kwargs=metric_kwargs,
                 match_filter_method=match_filter_method, ransac_thresh=ransac_thresh,
                 gms_threshold=gms_threshold, scaling=scaling)

        self.feature_detector = feature_detector
        self.feature_name = feature_detector.__class__.__name__
        self.rotation_invariant = False
        self.weights = weights
        self.keypoint_threshold = keypoint_threshold
        self.nms_radius = nms_radius
        self.sinkhorn_iterations = sinkhorn_iterations
        self.match_threshold = match_threshold
        self.kp_descriptor_name = "SuperPoint"
        self.kp_detector_name = "SuperPoint"
        self.matcher = "SuperGlue"
        self.metric_name = "SuperGlue"
        self.metric_type = "distance"
        self.device = 'cuda' if torch.cuda.is_available() and not force_cpu else "cpu"

        self.config = {
            'superpoint': {
                'nms_radius': self.nms_radius,
                'keypoint_threshold': self.keypoint_threshold,
                'max_keypoints': feature_detectors.MAX_FEATURES,
                'device': self.device
            },
            'superglue': {
                'weights': self.weights,
                'sinkhorn_iterations': self.sinkhorn_iterations,
                'match_threshold': self.match_threshold,
            }
        }

        self.sg_matcher = superglue.SuperGlue(self.config["superglue"])

    def frame2tensor(self, img):
        tensor = torch.from_numpy(img/255.).float()[None, None].to(self.device)

        return tensor


    def calc_scores(self, tensor_img, kp_xy):
        sp = superpoint.SuperPoint(self.config["superpoint"])

        x = sp.relu(sp.conv1a(tensor_img))
        x = sp.relu(sp.conv1b(x))
        x = sp.pool(x)
        x = sp.relu(sp.conv2a(x))
        x = sp.relu(sp.conv2b(x))
        x = sp.pool(x)
        x = sp.relu(sp.conv3a(x))
        x = sp.relu(sp.conv3b(x))
        x = sp.pool(x)
        x = sp.relu(sp.conv4a(x))
        x = sp.relu(sp.conv4b(x))

        cPa = sp.relu(sp.convPa(x))
        scores = sp.convPb(cPa)
        scores = torch.nn.functional.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h*8, w*8)
        scores = superpoint.simple_nms(scores, sp.config['nms_radius'])
        kp = [torch.from_numpy(kp_xy[:, ::-1].astype(int))]
        scores = [s[tuple(k.t())] for s, k in zip(scores, kp)]
        scores = scores[0].unsqueeze(dim=0)

        return scores

    def prep_data(self, img, kp_xy, desc):
        """
        sp_kp = pred["keypoints"] # Tensor with shape [1, n_kp, 2],  float32
        sp_desc = pred["descriptors"] # Tensor with shape [1, n_features, n_kp], float32
        sp_scores = pred["scores"] # Tensor with shape [1, n_kp], float32
        """

        inp = self.frame2tensor(img)
        scores = self.calc_scores(tensor_img=inp, kp_xy=kp_xy)
        kp_xy_inp = torch.from_numpy(kp_xy[None, :].astype(np.float32))

        desc_inp = torch.from_numpy(desc.T[None, :].astype(np.float32))

        n_kp = kp_xy.shape[0]

        assert scores.dtype == kp_xy_inp.dtype == desc_inp.dtype == torch.float32
        assert scores.shape[1] == kp_xy_inp.shape[1] == desc_inp.shape[2] == n_kp

        return inp, kp_xy_inp, desc_inp, scores

    def match_images(self, img1=None, desc1=None, kp1_xy=None, img2=None, desc2=None, kp2_xy=None, additional_filtering_kwargs=None, rotation_deg=None):

        if img1 is not None and desc1 is None and kp1_xy is None:
            kp1_xy, desc1 = self.feature_detector.detect_and_compute(img1)

        inp1, kp1_xy_inp, desc1_inp, scores1 = self.prep_data(img=img1, kp_xy=kp1_xy, desc=desc1)

        if img2 is not None and desc2 is None and kp2_xy is None:
            if rotation_deg is None:
                rotation_deg = self.estimate_rotation(moving_img=img2, fixed_img=img1)

            rotated_img, rot_tform = warp_tools.rotate(img2, rotation_deg, resize=True)
            rotated_img = rotated_img.astype(np.uint8)
            r_kp2, r_desc2 = self.feature_detector.detect_and_compute(rotated_img)
        else:
            r_kp2 = kp2_xy
            r_desc2 = desc2
            rotated_img = img2
            rot_tform = transform.SimilarityTransform()

        inp2, kp2_xy_inp, desc2_inp, scores2 = self.prep_data(img=rotated_img, kp_xy=r_kp2, desc=r_desc2)

        data = {"image0": inp1,
                "descriptors0": desc1_inp,
                "keypoints0": kp1_xy_inp,
                "scores0": scores1,
                "image1": inp2,
                "descriptors1": desc2_inp,
                "keypoints1": kp2_xy_inp,
                "scores1": scores2
                }

        sg_pred = self.sg_matcher(data)

        sg_pred = {k: v[0].detach().numpy() for k, v in sg_pred.items()}
        sg_pred.update(data)

        # Keep the matching keypoints and descriptors
        matches, conf = sg_pred['matches0'], sg_pred['matching_scores0']
        valid = matches > -1

        desc1_match_idx = np.where(valid)[0]
        desc2_match_idx = matches[valid]

        matched_desc1 = desc1[desc1_match_idx, :]
        matched_kp1_xy = kp1_xy[desc1_match_idx, :]

        rot_matched_kp2 = r_kp2[desc2_match_idx, :]
        matched_kp2_xy = rot_tform(rot_matched_kp2)
        matched_desc2 = r_desc2[desc2_match_idx, :]


        match_distances = np.sqrt(np.sum((matched_desc1 - matched_desc2)**2, axis=1))
        match_distances = match_distances/match_distances.max()


        mean_unfiltered_distance = np.mean(match_distances)
        mean_unfiltered_similarity = np.mean(convert_distance_to_similarity(match_distances, n_features=desc1.shape[1]))

        match_info12 = MatchInfo(matched_kp1_xy=matched_kp1_xy, matched_desc1=matched_desc1,
                                matches12=desc1_match_idx, matched_kp2_xy=matched_kp2_xy,
                                matched_desc2=matched_desc2, matches21=desc2_match_idx,
                                match_distances=match_distances, distance=mean_unfiltered_distance,
                                similarity=mean_unfiltered_similarity, metric_name=self.metric_name,
                                metric_type=self.metric_type,
                                feature_detector_name=self.feature_name)

        match_info21 = MatchInfo(matched_kp1_xy=matched_kp2_xy, matched_desc1=matched_desc2,
                                matches12=desc2_match_idx, matched_kp2_xy=matched_kp1_xy,
                                matched_desc2=matched_desc1, matches21=desc1_match_idx,
                                match_distances=match_distances, distance=mean_unfiltered_distance,
                                similarity=mean_unfiltered_similarity, metric_name=self.metric_name,
                                metric_type=self.metric_type,
                                feature_detector_name=self.feature_name)

        # # Remove outliers
        filtered_matched_kp1_xy, filtered_matched_kp2_xy, good_matches_idx = filter_matches_ransac(matched_kp1_xy,
                                                                                                   matched_kp2_xy,
                                                                                                   method=self.match_filter_method)

        if len(good_matches_idx) > 0:
            filterd_match_distances = match_distances[good_matches_idx]
            filterd_matched_desc1 = matched_desc1[good_matches_idx, :]
            filterd_matched_desc2 = matched_desc2[good_matches_idx, :]

            good_matches12 = desc1_match_idx[good_matches_idx]
            good_matches21 = desc2_match_idx[good_matches_idx]

            mean_filtered_distance = np.mean(filterd_match_distances)
            mean_filtered_similarity = \
                np.mean(convert_distance_to_similarity(filterd_match_distances,
                                                    n_features=desc1.shape[1]))
        else:
            filterd_match_distances = []
            filterd_matched_desc1 = []
            filterd_matched_desc2 = []

            good_matches12 = []
            good_matches21 = []

            mean_filtered_distance = np.inf
            mean_filtered_similarity = 0

        # Record filtered matches
        filtered_match_info12 = MatchInfo(matched_kp1_xy=filtered_matched_kp1_xy, matched_desc1=filterd_matched_desc1,
                                        matches12=good_matches12, matched_kp2_xy=filtered_matched_kp2_xy,
                                        matched_desc2=filterd_matched_desc2, matches21=good_matches21,
                                        match_distances=filterd_match_distances, distance=mean_filtered_distance,
                                        similarity=mean_filtered_similarity, metric_name=self.metric_name,
                                        metric_type=self.metric_type,
                                        feature_detector_name=self.feature_name)

        filtered_match_info21 = MatchInfo(matched_kp1_xy=filtered_matched_kp2_xy, matched_desc1=filterd_matched_desc2,
                                        matches12=good_matches21, matched_kp2_xy=filtered_matched_kp1_xy,
                                        matched_desc2=filterd_matched_desc1, matches21=good_matches12,
                                        match_distances=filterd_match_distances, distance=mean_filtered_distance,
                                        similarity=mean_filtered_similarity, metric_name=self.metric_name,
                                        metric_type=self.metric_type,
                                        feature_detector_name=self.feature_name)

        return match_info12, filtered_match_info12, match_info21, filtered_match_info21

class LightGlueMatcher(Matcher):
    """
    LightGlue feautre matcher, implemented in Kornia
    Intended for to be paired with features detected using one of Kornia's feature detectors (e.g. DISK, DeDoDe, etc...)

    Citation
    ---------
    Philipp Lindenberger, Paul-Edouard Sarlin, and Marc Pollefeys.
    Lightglue: local feature matching at light speed.
    arXiv ePrint 2306.13643, 2023.

    """
    def __init__(self, feature_detector=None,
                 match_filter_method=DEFAULT_MATCH_FILTER, ransac_thresh=DEFAULT_RANSAC,
                 device=None, *args, **kwargs):
        """
        Parameters
        ----------
        """

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.match_filter_method = match_filter_method
        self.ransac_thresh = ransac_thresh
        self.metric_type = "distance"
        self.rotation_invariant = False

        self._feature_detector = None
        self.feature_name = None
        self.metric_name = None
        self.lg_matcher = None
        if feature_detector is not None:
            self.feature_detector = feature_detector

    def get_fd(self):
        return self._feature_detector

    def set_fd(self, feature_detector):
        if not issubclass(feature_detector.__class__, feature_detectors.KorniaFD):
            msg = (f"Using {feature_detector.__class__.__name__} with {self.__class__.__name__}. "
                   f"May get unexpected results, as {self.__class__.__name__} expects the descriptors to be generated by "
                   f"a subclass of {feature_detectors.KorniaFD.__name__}")

            valtils.print_warning(msg)

        self._feature_detector = feature_detector
        self.feature_name = feature_detector.__class__.__name__
        self.metric_name = feature_detector.light_glue_feature_name
        self.lg_matcher = kornia.feature.LightGlueMatcher(feature_detector.light_glue_feature_name).eval().to(self.device)

    feature_detector = property(fget=get_fd,
                                fset=set_fd,
                                doc="Get and set feature detector. Setting creates a new LightGlueMatcher with associated weights")


    def estimate_rotation_brute_force(self, desc1, kp1, lafs1, hw1, moving_img, n_angles=4):
        """
        Use a rotation invarianet feature descriptor to estimate angle to rotate img2 to align with img1
        """

        t_desc1 = torch.from_numpy(desc1)
        angle_step = 360//n_angles
        rotations = np.arange(0, 360, angle_step)

        best_kp1 = None
        best_kp2 = None
        best_tform = None
        max_matches = 0
        min_mean_distance = np.inf
        all_match_counts = [None] * len(rotations)
        all_mean_distances = [None] * len(rotations)
        for i, r in enumerate(rotations):
            rotated, tform = warp_tools.rotate(moving_img, r, resize=True)
            rotated = rotated.astype(np.uint8)
            r_kp2, r_desc2 = self.feature_detector.detect_and_compute(rotated)
            r_hw2 = rotated.shape[0:2]

            t_kp2 = torch.from_numpy(r_kp2).to(self.device)
            t_desc2 = torch.from_numpy(r_desc2).to(self.device)
            with torch.inference_mode():
                lafs2 = kornia.feature.laf_from_center_scale_ori(t_kp2[None], torch.ones(1, len(t_kp2), 1, 1, device=self.device))
                match_distances, idxs = self.lg_matcher(t_desc1, t_desc2, lafs1, lafs2, hw1=hw1, hw2=r_hw2)

            _kp1 = kp1[idxs[:, 0], :]
            _kp2 = r_kp2[idxs[:, 1], :]
            _kp1, _kp2, good_idx = filter_matches(_kp1, _kp2, self.match_filter_method, filtering_kwargs={})
            r_n_matches = len(good_idx)
            all_match_counts[i] = r_n_matches
            rot_min_mean_distances = match_distances.min().detach().item()
            all_mean_distances[i] = rot_min_mean_distances

            if rot_min_mean_distances < min_mean_distance:
                best_kp1 = _kp1
                best_kp2 = _kp2
                best_tform = tform
                max_matches = r_n_matches
                min_mean_distance = rot_min_mean_distances

        # Estimate rotation
        angle_estimator = transform.SimilarityTransform()
        kp2_in_og = warp_tools.warp_xy(best_kp2, best_tform.params)
        angle_estimator.estimate(kp2_in_og, best_kp1)
        rot_deg = np.rad2deg(angle_estimator.rotation)

        return rot_deg

    def match_images(self, img1, img2, desc1=None, kp1_xy=None, desc2=None, kp2_xy=None, rotation_deg=None, brute_force_angle=False, *args, **kwargs):
        """Match the descriptors of image 1 with those of image 2,
        Outliers removed using match_filter_method. Metric can be a string
        to use a distance in scipy.distnce.cdist(), or a custom distance
        function. Sets atttributes for Matcher object
        Parameters
        ----------
        img1 : (N, P) array
            Fixed image

        img2 : (N, 2) array
            Moving image

        Returns
        -------
        match_info12 : MatchInfo
                Contains information regarding the matches between image 1
                and image 2. These results haven't undergone filtering,
                so contain many poor matches.

        filtered_match_info12 : MatchInfo
                Contains information regarding the matches between image 1
                and image 2. These results have undergone
                filtering, and so contain good matches

        match_info21 : MatchInfo
                Contains information regarding the matches between image 2
                and image 1. These results haven't undergone filtering, so
                contain many poor matches.

        filtered_match_info21 : MatchInfo
                Contains information regarding the matches between image 2
                and image 1.

        """

        hw1 = warp_tools.get_shape(img1)[0:2]
        if kp1_xy is None and desc1 is None:
            kp1_xy, desc1 = self.feature_detector.detect_and_compute(img1)

        t_kp1 = torch.from_numpy(kp1_xy).to(self.device)
        t_desc1 = torch.from_numpy(desc1).to(self.device)
        with torch.inference_mode():
            lafs1 = kornia.feature.laf_from_center_scale_ori(t_kp1[None], torch.ones(1, len(t_kp1), 1, 1, device=self.device))

        if kp2_xy is None and desc2 is None:
            if rotation_deg is None:
                if brute_force_angle:
                    rotation_deg = self.estimate_rotation_brute_force(desc1=desc1,
                                                            hw1=hw1,
                                                            kp1=kp1_xy,
                                                            lafs1=lafs1,
                                                            moving_img=img2)
                else:
                    rotation_deg = self.estimate_rotation(moving_img=img2, fixed_img=img1)
            else:
                rotation_deg = 0

            rotated_img, rot_tform = warp_tools.rotate(img2, rotation_deg, resize=True)
            rotated_img = rotated_img.astype(np.uint8)

            r_kp2, r_desc2 = self.feature_detector.detect_and_compute(rotated_img)
            r_hw2 = rotated_img.shape[0:2]

        else:
            # Assume that kp2 and desc2 have been detected on rotated image
            r_kp2 = kp2_xy
            r_desc2 = desc2
            rotated_img = img2
            rot_tform = transform.SimilarityTransform()
            r_hw2 = warp_tools.get_shape(img2)[0:2]

        t_kp2 = torch.from_numpy(kp2_xy).to(self.device)
        t_desc2 = torch.from_numpy(desc2).to(self.device)

        with torch.inference_mode():
            lafs2 = kornia.feature.laf_from_center_scale_ori(t_kp2[None], torch.ones(1, len(t_kp2), 1, 1, device=self.device))

            match_distances, idxs = self.lg_matcher(t_desc1, t_desc2, lafs1, lafs2, hw1=hw1, hw2=r_hw2)
            match_distances = match_distances.detach().numpy()
            idxs = idxs.detach().numpy()

        desc1_match_idx = idxs[:, 0]
        matched_desc1 = desc1[desc1_match_idx, :]
        matched_kp1_xy = kp1_xy[desc1_match_idx, :]

        desc2_match_idx = idxs[:, 1]
        rot_matched_kp2 = r_kp2[desc2_match_idx, :]
        matched_kp2_xy = rot_tform(rot_matched_kp2)
        matched_desc2 = r_desc2[desc2_match_idx, :]

        mean_unfiltered_distance = np.mean(match_distances)
        mean_unfiltered_similarity = np.mean(convert_distance_to_similarity(match_distances, n_features=desc1.shape[1]))

        match_info12 = MatchInfo(matched_kp1_xy=matched_kp1_xy, matched_desc1=matched_desc1,
                                matches12=desc1_match_idx, matched_kp2_xy=matched_kp2_xy,
                                matched_desc2=matched_desc2, matches21=desc2_match_idx,
                                match_distances=match_distances, distance=mean_unfiltered_distance,
                                similarity=mean_unfiltered_similarity, metric_name=self.metric_name,
                                metric_type=self.metric_type,
                                feature_detector_name=self.feature_name)

        match_info21 = MatchInfo(matched_kp1_xy=matched_kp2_xy, matched_desc1=matched_desc2,
                                matches12=desc2_match_idx, matched_kp2_xy=matched_kp1_xy,
                                matched_desc2=matched_desc1, matches21=desc1_match_idx,
                                match_distances=match_distances, distance=mean_unfiltered_distance,
                                similarity=mean_unfiltered_similarity, metric_name=self.metric_name,
                                metric_type=self.metric_type,
                                feature_detector_name=self.feature_name)

        # # Remove outliers
        filtered_matched_kp1_xy, filtered_matched_kp2_xy, good_matches_idx = filter_matches_ransac(matched_kp1_xy,
                                                                                                   matched_kp2_xy,
                                                                                                   method=self.match_filter_method)

        if len(good_matches_idx) > 0:
            filterd_match_distances = match_distances[good_matches_idx]
            filterd_matched_desc1 = matched_desc1[good_matches_idx, :]
            filterd_matched_desc2 = matched_desc2[good_matches_idx, :]

            good_matches12 = desc1_match_idx[good_matches_idx]
            good_matches21 = desc2_match_idx[good_matches_idx]

            mean_filtered_distance = np.mean(filterd_match_distances)
            mean_filtered_similarity = \
                np.mean(convert_distance_to_similarity(filterd_match_distances,
                                                    n_features=desc1.shape[1]))
        else:
            filterd_match_distances = []
            filterd_matched_desc1 = []
            filterd_matched_desc2 = []

            good_matches12 = []
            good_matches21 = []

            mean_filtered_distance = np.inf
            mean_filtered_similarity = 0

        # Record filtered matches
        filtered_match_info12 = MatchInfo(matched_kp1_xy=filtered_matched_kp1_xy, matched_desc1=filterd_matched_desc1,
                                        matches12=good_matches12, matched_kp2_xy=filtered_matched_kp2_xy,
                                        matched_desc2=filterd_matched_desc2, matches21=good_matches21,
                                        match_distances=filterd_match_distances, distance=mean_filtered_distance,
                                        similarity=mean_filtered_similarity, metric_name=self.metric_name,
                                        metric_type=self.metric_type)

        filtered_match_info21 = MatchInfo(matched_kp1_xy=filtered_matched_kp2_xy, matched_desc1=filterd_matched_desc2,
                                        matches12=good_matches21, matched_kp2_xy=filtered_matched_kp1_xy,
                                        matched_desc2=filterd_matched_desc1, matches21=good_matches12,
                                        match_distances=filterd_match_distances, distance=mean_filtered_distance,
                                        similarity=mean_filtered_similarity, metric_name=self.metric_name,
                                        metric_type=self.metric_type)

        return match_info12, filtered_match_info12, match_info21, filtered_match_info21
