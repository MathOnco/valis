"""Functions and classes to match and filter image features
"""

import numpy as np
import cv2
import numba as nba
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_kernels

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
"""str: If filter_method parameter in match_desc_and_kp is set to this,
RANSAC will be used to remove poor matches
"""

DEFAULT_MATCH_FILTER = RANSAC_NAME
"""str: The defulat filter_method value, either RANSAC_NAME or GMS_NAME"""

DEFAULT_RANSAC = 7
"""int: Default RANSAC threshold"""


@nba.njit()
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


@nba.njit()
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


def filter_matches_ransac(kp1_xy, kp2_xy, ransac_val=DEFAULT_RANSAC):
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

    Returns
    -------
    filtered_src_points : (N, 2) array
        Inlier keypoints from kp1_xy

    filtered_dst_points : (N, 2) array
        Inlier keypoints from kp1_xy

    good_idx : (1, N) array
        Indices of inliers

    """

    _, mask = cv2.findHomography(kp1_xy, kp2_xy, cv2.RANSAC, ransac_val)
    good_idx = np.where(mask.reshape(-1) == 1)[0]
    filtered_src_points = kp1_xy[good_idx, :]
    filtered_dst_points = kp2_xy[good_idx, :]
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

    return filtered_src_points, filtered_dst_points, good_idx


def filter_matches(kp1_xy, kp2_xy, method=DEFAULT_MATCH_FILTER,
                   filtering_kwargs=None):
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
    else:
        filter_fxn = filter_matches_ransac

    filtered_src_points, filtered_dst_points, good_idx = filter_fxn(**all_matching_args)

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


def match_desc_and_kp(desc1, kp1_xy, desc2, kp2_xy, metric=None,
                      metric_type=None, metric_kwargs=None,
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

        metric: string, or callable
            Metric to calculate distance between each pair of features
            in desc1 and desc2. Can be a string to use as distance in
            spatial.distance.cdist, or a custom distance function

        metric_kwargs : dict
            Optionl keyword arguments to be passed into pairwise_distances()
            or pairwise_kernels() from the sklearn.metrics.pairwise module

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
    else:
        cross_check = True

    matches, match_distances, metric_name, metric_type = \
        match_descriptors(desc1, desc2, metric=metric,
                          metric_type=metric_type,
                          metric_kwargs=metric_kwargs,
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
                             metric_type=metric_type)

    match_info21 = MatchInfo(matched_kp1_xy=matched_kp2_xy, matched_desc1=matched_desc2,
                             matches12=desc2_match_idx, matched_kp2_xy=matched_kp1_xy,
                             matched_desc2=matched_desc1, matches21=desc1_match_idx,
                             match_distances=match_distances, distance=mean_unfiltered_distance,
                             similarity=mean_unfiltered_similarity, metric_name=metric_name,
                             metric_type=metric_type)

    # Filter matches #
    all_filtering_kwargs = {"kp1_xy": matched_kp1_xy, "kp2_xy": matched_kp2_xy}
    if filtering_kwargs is None:
        if filter_method != RANSAC_NAME:
            print(Warning(f"filtering_kwargs not provided for {filter_method} match filtering. Will use RANSAC instead"))
            filter_method = RANSAC_NAME
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
                                      metric_type=metric_type)

    filtered_match_info21 = MatchInfo(matched_kp1_xy=filtered_matched_kp2_xy, matched_desc1=filterd_matched_desc2,
                                      matches12=good_matches21, matched_kp2_xy=filtered_matched_kp1_xy,
                                      matched_desc2=filterd_matched_desc1, matches21=good_matches12,
                                      match_distances=filterd_match_distances, distance=mean_filtered_distance,
                                      similarity=mean_filtered_similarity, metric_name=metric_name,
                                      metric_type=metric_type)

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
                 img1_name=None, img2_name=None):

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
            "distsnce" or "similarity"

        img1_name : str
            Name of the image that kp1 and desc1 belong to

        img2_name : str
            Name of the image that kp2 and desc2 belong to

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

    def __init__(self, metric=None, metric_type=None, metric_kwargs=None,
                 match_filter_method=DEFAULT_MATCH_FILTER, ransac_thresh=DEFAULT_RANSAC,
                 gms_threshold=15, scaling=False):
        """
        Parameters
        ----------

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

    def match_images(self, desc1, kp1_xy, desc2, kp2_xy,
                     additional_filtering_kwargs=None):
        """Match the descriptors of image 1 with those of image 2,
        Outliers removed using match_filter_method. Metric can be a string
        to use a distance in scipy.distnce.cdist(), or a custom distance
        function. Sets atttributes for Matcher object

        Parameters
        ----------
        desc1 : (N, P) array
            Image 1s 2D array containinng N keypoints, each of which
            has P features

        kp1_xy : (N, 2) array
            Image 1s keypoint positions, in xy coordinates,  for each of the
            N descriptors in desc1

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

                self.match_filter_method = RANSAC_NAME
                filtering_kwargs = {"ransac_val": self.ransac}

        elif self.match_filter_method == RANSAC_NAME:
            filtering_kwargs = {"ransac_val": self.ransac}

        else:
            print(Warning(f"Dont know {self.match_filter_method}.\
                Defaulting to RANSAC"))

            self.match_filter_method = RANSAC_NAME
            filtering_kwargs = {"ransac_val": self.ransac}

        match_info12, filtered_match_info12, match_info21, filtered_match_info21 = \
            match_desc_and_kp(desc1, kp1_xy, desc2, kp2_xy,
                              metric=self.metric, metric_type=self.metric_type,
                              metric_kwargs=self.metric_kwargs,
                              filter_method=self.match_filter_method,
                              filtering_kwargs=filtering_kwargs)

        if self.metric_name is None:
            self.metric_name = match_info12.metric_name

        return match_info12, filtered_match_info12, match_info21, filtered_match_info21
