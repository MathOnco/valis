"""Classes and functions to perform serial rigid registration of a set of images

"""
import numpy as np
import os
import pickle
from fastcluster import linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import optimal_leaf_ordering, leaves_list
from skimage import transform, io
from skimage.transform import EuclideanTransform
import pandas as pd
import warnings
import imghdr
from tqdm import tqdm
import pathlib
import multiprocessing
from joblib import Parallel, delayed, parallel_backend
from time import time
from . import valtils
from . import warp_tools
from .feature_detectors import VggFD
from .feature_matcher import Matcher, convert_distance_to_similarity, GMS_NAME
# from .affine_optimizer import AffineOptimizerMattesMI


def get_image_files(img_dir, imgs_ordered=False):
    """Get images filenames in img_dir

    If imgs_ordered is True, then this ensures the returned list is sorted
    properly. Otherwise, the list is sorted lexicographicly.

    Parameters
    ----------
    img_dir : str
        Path to directory containing the images.

    imgs_ordered: bool, optinal
        Whether or not the order of images already known. If True, the file
        names should start with ascending numbers, with the first image file
        having the smallest number, and the last image file having the largest
        number. If False (the default), the order of images will be determined
        by ordering a distance matrix.

    Returns
    -------
        If `imgs_ordered` is True, then this ensures the returned list is sorted
        properly. Otherwise, the list is sorted lexicographicly.

    """

    img_list = [f for f in os.listdir(img_dir) if
                imghdr.what(os.path.join(img_dir, f)) is not None]

    if imgs_ordered:
        valtils.sort_nicely(img_list)
    else:
        img_list.sort()

    return img_list


def get_max_image_dimensions(img_list):
    """Find the maximum width and height of all images

    Parameters
    ----------
    img_list : list
        List of images

    Returns
    -------
    max_wh : tuple
        Maximum width and height of all images

    """

    shapes = [img.shape[0:2] for img in img_list]
    all_w, all_h = list(zip(*shapes))
    max_wh = (max(all_w), max(all_h))

    return max_wh


def order_Dmat(D):
    """ Cluster distance matrix and sort

    Leaf sorting is accomplished using optimal leaf ordering (Bar-Joseph 2001)

    Parmaters
    ---------
    D: ndarray
        (N, N) Symmetric distance matrix for N samples

    Returns
    -------
    sorted_D :ndarray
        (N, N) array Distance matrix sorted using optimal leaf ordering

    ordered_leaves : ndarray
        (1, N) array containing the leaves of dendrogram found during
        hierarchical clustering

    optimal_Z : ndarray
        ordered linkage matrix

    """

    D = D.copy()
    sq_D = squareform(D)
    Z = linkage(sq_D, 'single', preserve_input=True)

    optimal_Z = optimal_leaf_ordering(Z, sq_D)
    ordered_leaves = leaves_list(optimal_Z)

    sorted_D = D[ordered_leaves, :]
    sorted_D = sorted_D[:, ordered_leaves]

    return sorted_D, ordered_leaves, optimal_Z


class ZImage(object):
    """Class store info about an image, including the rigid registration parameters

    Attributes
    ----------
    image : ndarray
        Greyscale image that will be used for feature detection. This images
        should be greyscale and may need to have undergone preprocessing to
        make them look as similar as possible.

    full_img_f : str
        full path to the image

    img_id : int
        ID of the image, based on its ordering `processed_src_dir`

    name : str
        Name of the image. Usually `img_f` but with the extension removed.

    desc : ndarray
        (N, M) array of N desciptors for each keypoint, each of which has
        M features

    kp_pos_xy : ndarray
        (N, 2) array of position for each keypoint

    match_dict : dict
        Dictionary of image matches. Key= img_obj this ZImage is being
        compared to, value= MatchInfo containing information about the
        comparison, such as the position of matches, features for each match,
        number of matches, etc... The MatchInfo objects in this dictionary
        contain only the info for matches that were considered "good".

    unfiltered_match_dict : dict
        Dictionary of image matches. Key= img_obj this ZImage is being
        compared to, value= MatchInfo containing inoformation about the
        comparison, such as the position of matches, features for each match,
        number of matches, etc... The MatchInfo objects in this dictionary
        contain info for all matches that were cross-checked.

    stack_idx : int
        Position of image in sorted Z-stack

    T : ndarray
        Transformation matrix that translates the image such that it is in a
        padded image that has the same shape as all other images

    to_prev_A : ndarray
        Transformation matrix that warps image to align with the previous image

    optimal_M : ndarray
        Transformation matrix found by minimizing a cost function.
        Used as final optional step to refine alignment

    crop_T : ndarray
        Transformation matrix used to crop image after registration

    M : ndarray
        Final transformation matrix that aligns image in the Z-stack.

    M_inv : ndarray
        Inverse of final transformation matrix that aligns image in
        the Z-stack.

    registered_img : ndarray
        image after being warped

    padded_shape_rc : tuple
        Shape of padded image. All other images will have this shape

    registered_shape_rc = tuple:
        Shape of aligned image. All other aligned images will have this shape

    """

    def __init__(self, image, img_f, img_id, name):
        """Class that stores information about an image

        Parameters
        ----------
        image : ndarray
            Greyscale image that will be used for feature detection. This
            images should be single channel uint8 images, and may need to
            have undergone preprocessing and/or normalization to make them
            look as similar as possible.

        img_f : str
            full path to `image`

        img_id : int
            ID of the image, based on its ordering in the image source directory

        name : str
            Name of the image. Usually img_f but with the extension removed.

        """

        self.image = image
        self.full_img_f = img_f
        self.id = img_id
        self.name = name

        self.desc = None
        self.kp_pos_xy = None
        self.match_dict = {}
        self.unfiltered_match_dict = {}
        self.stack_idx = None

        self.padded_shape_rc = None
        self.T = np.identity(3)
        self.to_prev_A = np.identity(3)
        self.optimal_M = np.identity(3)
        self.crop_T = np.identity(3)
        self.M = np.identity(3)
        self.M_inv = np.identity(3)
        self.registered_img = None
        self.padded_shape_rc = None
        self.registered_shape_rc = None

    def reduce(self, prev_img_obj, next_img_obj):
        """Reduce amount of info stored, which can take up a lot of space.

        No longer need all descriptors. Only keep match info for neighgbors

        Parameters
        ----------
        prev_img_obj : Zimage
            Zimage below this Zimage

        next_img_obj :  Zimage
            Zimage above this Zimage

        """

        self.desc = None
        for img_obj in self.match_dict.keys():
            if prev_img_obj is not None and next_img_obj is not None:
                if prev_img_obj != img_obj and img_obj != next_img_obj:
                    # In middle of stack
                    self.match_dict[img_obj] = None

                elif prev_img_obj is None and img_obj != next_img_obj:
                    # First image doesn't have a previous neighbor
                    self.match_dict[img_obj] = None

                elif prev_img_obj != img_obj and next_img_obj is None:
                    # Last image doesn't have a next neighbor
                    self.match_dict[img_obj] = None


class SerialRigidRegistrar(object):
    """Class that performs serial rigid registration

    Registration is conducted by first detecting features in all images.
    Features are then matched between images, which are then used to construct
    a distance matrix, D. D is then sorted such that the most similar images
    are adjcent to one another. The rigid transformation matrics are then found to
    align each image with the previous image. Optionally, optimization can be
    performed to improve the alignments, although the "optimized" matrix will be
    discarded if it increases the distances between matched features.

    SerialRigidRegistrar creates a list and dictionary of ZImage objects,
    each of which contains information related to feature matching and
    the rigid registration matrices.

    Attributes
    ----------
    img_dir : str
        Path to directory containing the images that will be registered.
        The images in this folder should be single channel uint8 images.
        For the best registration results, they have undergone some sort
        of pre-processing and normalization. The preprocessing module
        contains methods for this, but the user may want/need to use other
        methods.

    aleady_sorted: bool, optional
        Whether or not the order of images already known. If True, the file
        names should start with ascending numbers, with the first image file
        having the smallest number, and the last image file having the largest
        number. If False (the default), the order of images will be determined
        by ordering a distance matrix.

    name : str
        Descriptive name of registrar, such as the sample's name

    img_file_list : list
        List of full paths to single channel uint8 images

    size : int
        Number of images to align

    distance_metric_name : str
        Name of distance metric used to determine the dis/similarity between
        each pair of images

    distance_metric_type : str
        Name of the type of metric used to determine the dis/similarity
        between each pair of images. Despite the name, it could be "similarity"
        if the Matcher object compares image feautres using a similarity
        metric. In that case, similarities are converted to distances.

    img_obj_list : list
        List of ZImage objects. Initially unordered, but
        eventually be sorted

    img_obj_dict : dict
        Dictionary of ZImage objects. Created to conveniently
        access ZIimages. Key = ZImage.name, value= ZImage

    optimal_Z :ndarray
        Ordered linkage matrix for `distance_mat`

    unsorted_distance_mat : ndarray
        Distance matrix with shape (N, N), where each element is the
        disimilariy betweewn each pair of the N images. The order of
        rows and columns reflects the order in which the images were read.
        This matrix is used to order the images the Z-stack.

    distance_mat : ndarray
        `unsorted_distance_mat` reorderd such that the most similar images
        are adjacent to one another

    unsorted_similarity_mat : ndarray
        Similar to `unsorted_distance_mat`, except the elements are
        image similarity

    similarity_mat : ndarray
        Similar to `distance_mat`, except the elements are image similarity

    features : str
        Name of feature detector and descriptor used

    transform_type : str
        Name of scikit-image transformer class that was used

    summary_df : Dataframe
        Pandas dataframe containin the registration error of the
        alignment between each image and the previous one in the stack.

    overlap_mask : ndarray
        Mask that covers intersection of all rigidly aligned images.

    overlap_mask_bbox_xywh : ndarray
        Bounding box of overlap_mask. Values are [min_x, min_y, width, height]

    """

    def __init__(self, img_dir, imgs_ordered=False, name=None):
        """Class that performs serial rigid registration

        Parameters
        ----------
        img_dir : str
            Path to directory containing the images that will be registered.
            The images in this folder should be single channel uint8 images.
            For the best registration results, they have undergone some sort
            of pre-processing and normalization. The preprocessing module
            contains methods for this, but the user may want/need to use other
            methods.

        imgs_ordered : bool
            Whether or not the order of images already known. If True, the file
            names should start with ascending numbers, with the first image
            file having the smallest number, and the last image file having
            the largest number. If False (the default), the order of images
            will be determined by sorting a distance matrix.

        name : str
            Descriptive name of registrar, such as the sample's name

        """

        self.img_dir = img_dir
        self.aleady_sorted = imgs_ordered
        self.name = name
        self.img_file_list = get_image_files(img_dir, imgs_ordered=imgs_ordered)
        self.size = len(self.img_file_list)
        self.distance_metric_name = None
        self.distance_metric_type = None
        self.img_obj_list = None
        self.img_obj_dict = {}
        self.optimal_z = None
        self.unsorted_distance_mat = None
        self.distance_mat = None
        self.unsorted_similarity_mat = None
        self.similarity_mat = None
        self.features = None
        self.transform_type = None
        self.summary = None
        self.overlap_mask = None
        self.overlap_mask_bbox_xywh = None

    def generate_img_obj_list(self, feature_detector, qt_emitter=None):
        """Create a list of ZImage objects

        Create a list of ZImage objects, each of which represents an image.
        This function also determines the maximum size of the images so that
        there is no cropping during warping. Finally, the features of each
        image are detected using the feature_detector

        Parameters
        ----------
        feature_detector : FeatureDD
            FeatureDD object that detects and computes image features.

        qt_emitter : PySide2.QtCore.Signal, optional
            Used to emit signals that update the GUI's progress bars

        """

        # NOTE tried parallelizing with joblib, but it's actually slower  #
        sorted_img_list = [io.imread(os.path.join(self.img_dir, f), True)
                           for f in self.img_file_list]
        # assert(sorted_img_list[0].dtype == np.uint8), valtils.print_warning("images must be uint8")

        out_w, out_h = get_max_image_dimensions(sorted_img_list)

        # Get dimensions if images were rotated 45 degrees or 90 degrees
        max_new_w = out_w*np.cos(45) + out_h*np.sin(45)
        max_new_h = out_w*np.sin(45) + out_h*np.cos(45)

        max_dist = np.ceil(np.max([out_w, out_h, max_new_h, max_new_w])).astype(np.int)
        out_shape = (max_dist, max_dist)
        img_obj_list = [None] * self.size

        for i in tqdm(range(self.size)):
            img_f = self.img_file_list[i]
            img = sorted_img_list[i]

            img_name = valtils.get_name(img_f)
            img_obj = ZImage(img, os.path.join(self.img_dir, img_f), i, name=img_name)
            img_obj.padded_shape_rc = out_shape
            img_obj.T = warp_tools.get_padding_matrix(img.shape, img_obj.padded_shape_rc)
            img_obj.kp_pos_xy, img_obj.desc = feature_detector.detect_and_compute(img)
            img_obj_list[i] = img_obj
            self.img_obj_dict[img_name] = img_obj
            if qt_emitter is not None:
                qt_emitter.emit(1)

        self.img_obj_list = img_obj_list
        self.features = feature_detector.__class__.__name__

    def match_sorted_imgs(self, matcher_obj, keep_unfiltered=False, qt_emitter=None):
        """Conduct feature matching between images that have already been sorted.

        Results will be stored in each ZImage's match_dict

        Parameters
        ----------
        matcher_obj : Matcher
            Object to match features between images.

        keep_unfiltered : bool
            Whether or not matcher_obj should store unfiltered matches

        qt_emitter : PySide2.QtCore.Signal, optional
            Used to emit signals that update the GUI's progress bars

        """

        def match_adj_img_obj(i):
            if i == 0:
                return None
            img_obj_1 = self.img_obj_list[i]
            img_obj_2 = self.img_obj_list[i-1]

            if matcher_obj.match_filter_method == GMS_NAME:
                filter_kwargs = {"img1_shape":img_obj_1.image.shape[0:2], "img2_shape": img_obj_2.image.shape[0:2]}
            else:
                filter_kwargs = None
            unfiltered_match_info12, filtered_match_info12, unfiltered_match_info21, filtered_match_info21 = \
                matcher_obj.match_images(img_obj_1.desc, img_obj_1.kp_pos_xy,
                                            img_obj_2.desc, img_obj_2.kp_pos_xy,
                                            filter_kwargs)
            if len(filtered_match_info12.matched_kp1_xy) == 0:
                warnings.warn(f"{len(filtered_match_info12.matched_kp1_xy)} between {img_obj_1.name} and {img_obj_2.name}")
            # Update match dictionaries #
            if keep_unfiltered:
                unfiltered_match_info12.set_names(img_obj_1.name, img_obj_2.name)
                img_obj_1.unfiltered_match_dict[img_obj_2] = unfiltered_match_info12

                unfiltered_match_info21.set_names(img_obj_2.name, img_obj_1.name)
                img_obj_2.unfiltered_match_dict[img_obj_1] = unfiltered_match_info21

            filtered_match_info12.set_names(img_obj_1.name, img_obj_2.name)
            img_obj_1.match_dict[img_obj_2] = filtered_match_info12

            filtered_match_info21.set_names(img_obj_2.name, img_obj_1.name)
            img_obj_2.match_dict[img_obj_1] = filtered_match_info21

            if qt_emitter is not None:
                    qt_emitter.emit(1)

        n_cpu = multiprocessing.cpu_count() - 1
        with parallel_backend("threading", n_jobs=n_cpu):
            Parallel()(delayed(match_adj_img_obj)(i) for i in range(self.size))

    def match_imgs(self, matcher_obj, keep_unfiltered=False, qt_emitter=None):
        """Conduct feature matching between all pairs of images.

        Results will be stored in each ZImage's match_dict

        Parameters
        ----------
        matcher_obj : Matcher
            Object to match features between images.

        keep_unfiltered : bool
            Whether or not matcher_obj should store unfiltered matches

        qt_emitter : PySide2.QtCore.Signal, optional
            Used to emit signals that update the GUI's progress bars

        """

        n_comparisions = int((self.size*(self.size-1))/2)
        pbar = tqdm(total=n_comparisions)

        def match_img_obj(i):
            img_obj_1 = self.img_obj_list[i]
            for j in np.arange(i+1, self.size):
                img_obj_2 = self.img_obj_list[j]
                if matcher_obj.match_filter_method == GMS_NAME:
                    filter_kwargs = {"img1_shape":img_obj_1.image.shape[0:2], "img2_shape": img_obj_2.image.shape[0:2]}
                else:
                    filter_kwargs = None
                unfiltered_match_info12, filtered_match_info12, unfiltered_match_info21, filtered_match_info21 = \
                    matcher_obj.match_images(img_obj_1.desc, img_obj_1.kp_pos_xy,
                                                img_obj_2.desc, img_obj_2.kp_pos_xy,
                                                filter_kwargs)
                if len(filtered_match_info12.matched_kp1_xy) == 0:
                    warnings.warn(f"{len(filtered_match_info12.matched_kp1_xy)} between {img_obj_1.name} and {img_obj_2.name}")
                # Update match dictionaries #
                if keep_unfiltered:
                    unfiltered_match_info12.set_names(img_obj_1.name, img_obj_2.name)
                    img_obj_1.unfiltered_match_dict[img_obj_2] = unfiltered_match_info12

                    unfiltered_match_info21.set_names(img_obj_2.name, img_obj_1.name)
                    img_obj_2.unfiltered_match_dict[img_obj_1] = unfiltered_match_info21

                filtered_match_info12.set_names(img_obj_1.name, img_obj_2.name)
                img_obj_1.match_dict[img_obj_2] = filtered_match_info12

                filtered_match_info21.set_names(img_obj_2.name, img_obj_1.name)
                img_obj_2.match_dict[img_obj_1] = filtered_match_info21

                pbar.update(1)
                if qt_emitter is not None:
                    qt_emitter.emit(1)

        n_cpu = multiprocessing.cpu_count() - 1
        with parallel_backend("threading", n_jobs=n_cpu):
            Parallel()(delayed(match_img_obj)(i) for i in range(self.size))

    def neighbor_match_filtering(self, img_obj, prev_img_obj, next_img_obj,
                                 tform, matcher_obj):
        """Remove poor matches by keeping only the matches found in neighbors

        Parameters
        ----------
        img_obj : ZImage
            current ZImage

        prev_img_obj : ZImage
            ZImage to below `img_obj`

        next_img_obj : ZImage
            ZImage to above `img_obj`

        tform : skimage.transform object
            The scikit-image transform object that estimates the
            parameter matrix

        matcher_obj : Matcher
            Object to match features between images.

        Returns
        -------

        improved: bool
            Whether or not neighbor filtering improved the alignment

        updated_prev_match_info12 : MatchInfo
            If improved is True, then `updated_prev_match_info12` includes only
            features, descriptors that were found in both neighbors. Otherwise,
            all of the original features will be maintained

        updated_next_match_info12 : MatchInfo
            If improved is True, then `updated_next_match_info12` includes only
            features, descriptors that were found in both neighbors. Otherwise,
            all of the original features will be maintained

        """

        def measure_d(src_xy, dst_xy, tform, M=None):
            """Measure distance between warped corresponding points
            """
            if M is None:
                tform.estimate(src=dst_xy, dst=src_xy)
                M = tform.params
            warped_xy = tform.inverse(src_xy)
            dxdy = dst_xy - warped_xy
            d = np.median(np.sqrt(dxdy[:, 0]**2 + dxdy[:, 1]**2))

            return d, M

        to_prev_match_info12 = img_obj.match_dict[prev_img_obj]
        to_next_match_info12 = img_obj.match_dict[next_img_obj]

        common_neighbor_matches = \
            np.intersect1d(to_prev_match_info12.matches12,
                           to_next_match_info12.matches12)
        improved = False
        if len(common_neighbor_matches) > 4:
            # Want at least 4 matches to estimate transformation

            to_prev_match_info12_l = list(to_prev_match_info12.matches12)
            matches_idx_in_prev = \
                [to_prev_match_info12.matches21[to_prev_match_info12_l.index(idx)]
                for idx in common_neighbor_matches]

            to_next_match_info12_l = list(to_next_match_info12.matches12)
            matches_idx_in_next = \
                [to_next_match_info12.matches21[to_next_match_info12_l.index(idx)]
                for idx in common_neighbor_matches]

            common_kp = img_obj.kp_pos_xy[common_neighbor_matches]
            common_desc = img_obj.desc[common_neighbor_matches]

            common_prev_kp = prev_img_obj.kp_pos_xy[matches_idx_in_prev]
            common_prev_desc = prev_img_obj.desc[matches_idx_in_prev]

            common_next_kp = next_img_obj.kp_pos_xy[matches_idx_in_next]
            common_next_desc = next_img_obj.desc[matches_idx_in_next]

            common_matches_d, common_matches_M = measure_d(common_kp,
                                                           common_prev_kp,
                                                           tform)

            original_to_prev_matches = img_obj.match_dict[prev_img_obj]
            original_d, _ = measure_d(original_to_prev_matches.matched_kp1_xy,
                                      original_to_prev_matches.matched_kp2_xy,
                                      tform)

            original_with_neighbor_filter_d, _ = measure_d(original_to_prev_matches.matched_kp1_xy,
                                                           original_to_prev_matches.matched_kp2_xy,
                                                           tform, M=common_matches_M)

            if common_matches_d < original_d and common_matches_d < original_with_neighbor_filter_d:
                # neighbor filtering improved alignment
                improved = True
                updated_prev_match_info12, _, updated_prev_match_info21, _ = \
                    matcher_obj.match_images(desc1=common_desc,
                                             kp1_xy=common_kp,
                                             desc2=common_prev_desc,
                                             kp2_xy=common_prev_kp)

                updated_next_match_info12, _, updated_next_match_info21, _ = \
                    matcher_obj.match_images(desc1=common_desc,
                                             kp1_xy=common_kp,
                                             desc2=common_next_desc,
                                             kp2_xy=common_next_kp)

        if improved:
            return improved, updated_prev_match_info12, updated_next_match_info12
        else:
            return improved, to_prev_match_info12.matches12, to_next_match_info12.matches12

    def update_match_dicts_with_neighbor_filter(self, tform, matcher_obj):
        """Remove poor matches by keeping only the matches found in neighbors

        Parameters
        ----------
        tform : skimage.transform object
            The scikit-image transform object that estimates the
            parameter matrix

        matcher_obj : Matcher
            Object to match features between images.

        """

        for i, img_obj in enumerate(self.img_obj_list):
            if i == 0 or i == self.size - 1:
                continue

            prev_idx = i - 1
            prev_img_obj = self.img_obj_list[prev_idx]

            next_idx = i + 1
            next_img_obj = self.img_obj_list[next_idx]
            improved, updated_prev_match_info12, updated_next_match_info12 = \
                self.neighbor_match_filtering(img_obj, prev_img_obj,
                                              next_img_obj, tform, matcher_obj)

            if improved:
                img_obj.match_dict[prev_img_obj] = updated_prev_match_info12
                img_obj.match_dict[next_img_obj] = updated_next_match_info12

    def build_metric_matrix(self, metric="n_matches"):
        """Create metric matrix based image similarity/distance

        Parameters
        ----------
        metric: str
            Name of metrric to use. If 'distance' that the distances and
            similiarities calculated during feature matching will be used.
            If 'n_matches', then the number of matches will be used for
            similariy, and 1/n_matches for distance.

        """

        distance_mat = np.zeros((self.size, self.size))
        similarity_mat = np.zeros_like(distance_mat)

        for i, obj1 in enumerate(self.img_obj_list):
            for j in np.arange(i, self.size):
                obj2 = self.img_obj_list[j]
                if i == j:
                    continue

                if metric == "n_matches":
                    s = obj1.match_dict[obj2].n_matches
                else:
                    s = obj1.match_dict[obj2].similarity
                    d = obj1.match_dict[obj2].distance
                    distance_mat[i, j] = d
                    distance_mat[j, i] = d

                similarity_mat[i, j] = s
                similarity_mat[j, i] = s

        min_s = similarity_mat.min()
        max_s = similarity_mat.max()
        min_d = distance_mat.min()
        max_d = distance_mat.max()

        # Make sure that image has highest similarity with itself
        similarity_mat[np.diag_indices_from(similarity_mat)] += max_s*0.01

        # Scale metrics between 0 and 1
        similarity_mat = (similarity_mat - min_s) / (max_s - min_s)
        similarity_mat[np.diag_indices_from(similarity_mat)] = 1
        if metric == "n_matches":
            distance_mat = 1 - similarity_mat
        else:
            distance_mat = (distance_mat - min_d) / (max_d - min_d)

        distance_mat[np.diag_indices_from(distance_mat)] = 0
        self.unsorted_similarity_mat = similarity_mat
        self.unsorted_distance_mat = distance_mat

    def sort(self):
        """Order images such that most similar images are adjacent

        Order the images in the stack by optimally ordering the leaves of
        dendrogram created by clustering a matrix of image feature distances.
        """

        sorted_D, sorted_idx, optimal_Z = order_Dmat(self.unsorted_distance_mat)
        self.optimal_z = optimal_Z
        self.distance_mat = sorted_D
        self.similarity_mat = self.unsorted_similarity_mat[sorted_idx, :]
        self.similarity_mat = self.similarity_mat[:, sorted_idx]
        self.img_file_list = [self.img_file_list[i] for i in sorted_idx]
        self.img_file_list = [self.img_file_list[i] for i in sorted_idx]
        self.img_obj_list = [self.img_obj_list[i] for i in sorted_idx]
        for z, img_obj in enumerate(self.img_obj_list):
            img_obj.stack_idx = z

    def align_to_prev(self, transformer, qt_emitter=None):
        """Use key points to align current image to previous image in the stack

       Parameters
        ---------
        transformer : skimage.transform object
            The scikit-image transform object that estimates the
            parameter matrix

        qt_emitter : PySide2.QtCore.Signal, optional
            Used to emit signals that update the GUI's progress bars

        """

        prev_img_obj = None
        prev_M = None
        for i in tqdm(range(self.size)):

            img_obj = self.img_obj_list[i]
            M = img_obj.T
            if i == 0:
                img_obj.to_prev_A = np.identity(3)
                prev_img_obj = img_obj
                prev_M = M.copy()
                if qt_emitter is not None:
                    qt_emitter.emit(1)
                continue

            to_prev_match_info = img_obj.match_dict[prev_img_obj]
            src_xy = warp_tools.warp_xy(to_prev_match_info.matched_kp1_xy, M)
            dst_xy = warp_tools.warp_xy(to_prev_match_info.matched_kp2_xy, prev_M)

            transformer.estimate(dst_xy, src_xy)
            img_obj.to_prev_A = transformer.params

            prev_M = M @ img_obj.to_prev_A
            prev_img_obj = img_obj

            if qt_emitter is not None:
                qt_emitter.emit(1)

    def optimize(self, affine_optimizer, qt_emitter=None):
        """Refine alignment by minimizing a metric

        Transformation will only be allowed if it both decreases the
        cost and median distance between keypoints.

        Parameters
        -----------
        affine_optimizer : AffineOptimzer
            Object that will minimize a cost function to find the optimal
            affine transformations

        qt_emitter : PySide2.QtCore.Signal, optional
            Used to emit signals that update the GUI's progress bars

        """

        prev_img = None
        prev_img_obj = None
        prev_M = None

        for i in tqdm(range(self.size)):
            img_obj = self.img_obj_list[i]
            M = img_obj.T @ img_obj.to_prev_A
            warped_img = warp_tools.warp_img(img_obj.image,
                                             M=M,
                                             out_shape_rc=img_obj.padded_shape_rc)

            if i == 0:
                img_obj.optimal_M = np.identity(3)
                prev_img = warped_img
                prev_img_obj = img_obj
                prev_M = M @ img_obj.optimal_M
                if qt_emitter is not None:
                    qt_emitter.emit(1)
                continue

            to_prev_match_info = img_obj.match_dict[prev_img_obj]
            before_src_xy = warp_tools.warp_xy(to_prev_match_info.matched_kp1_xy, M)
            before_dst_xy = warp_tools.warp_xy(to_prev_match_info.matched_kp2_xy, prev_M)
            before_tre, before_med_d = warp_tools.measure_error(before_src_xy,
                                                                before_dst_xy,
                                                                warped_img.shape)

            img_mask = np.ones(img_obj.image.shape[0:2], dtype=np.uint8)
            warped_img_mask = warp_tools.warp_img(img_mask,
                                                  M=M,
                                                  out_shape_rc=img_obj.padded_shape_rc)

            prev_img_mask = np.ones(prev_img_obj.image.shape[0:2], dtype=np.uint8)
            warped_prev_img_mask = warp_tools.warp_img(prev_img_mask,
                                                       M=prev_M,
                                                       out_shape_rc=prev_img_obj.padded_shape_rc)

            mask = np.zeros(warped_img_mask.shape, dtype=np.uint8)
            mask[(warped_img_mask != 0) & (warped_prev_img_mask != 0)] = 255

            if affine_optimizer.accepts_xy:
                moving_xy = before_src_xy
                fixed_xy = before_dst_xy
            else:
                moving_xy = None
                fixed_xy = None

            _, optimal_M, _ = affine_optimizer.align(warped_img, prev_img,
                                                     mask, initial_M=None,
                                                     moving_xy=moving_xy,
                                                     fixed_xy=fixed_xy)

            initial_cst = affine_optimizer.cost_fxn(warped_img, prev_img, mask)

            after_src_xy = warp_tools.warp_xy(to_prev_match_info.matched_kp1_xy, M @ optimal_M)
            after_dst_xy = warp_tools.warp_xy(to_prev_match_info.matched_kp2_xy, prev_M)

            optimal_reg_img = warp_tools.warp_img(warped_img,
                                                  M=optimal_M,
                                                  out_shape_rc=img_obj.padded_shape_rc)

            after_cst = affine_optimizer.cost_fxn(optimal_reg_img, prev_img, mask)

            after_tre, after_med_d = warp_tools.measure_error(after_src_xy,
                                                              after_dst_xy,
                                                              warped_img.shape)

            if after_cst is not None and initial_cst is not None:
                lower_cost = after_cst <= initial_cst
            else:
                lower_cost = True

            lower_d = after_med_d <= before_med_d
            if lower_cost and lower_d:
                prev_img = optimal_reg_img
                img_obj.optimal_M = optimal_M
            else:
                print(Warning(f"Somehow optimization made things worse.\
                              Cost was {initial_cst} but is now {after_cst}\
                              KP medD was {before_med_d}, but is now {after_med_d}"))

                img_obj.optimal_M = np.identity(3)
                prev_img = warped_img

            prev_M = M @ img_obj.optimal_M
            prev_img_obj = img_obj

            if qt_emitter is not None:
                qt_emitter.emit(1)

    def get_overlap_mask(self):
        """Create mask that covers intersection of all images
        """

        overlap_mask = np.zeros(self.img_obj_list[0].registered_img.shape[0:2], dtype=np.int)
        for img_obj in self.img_obj_list:
            img_mask = np.ones(img_obj.image.shape[0:2], dtype=np.int)
            warped_img_mask = warp_tools.warp_img(img_mask,
                                                  M=img_obj.M,
                                                  out_shape_rc=img_obj.registered_img.shape)

            overlap_mask += warped_img_mask

        overlap_mask[overlap_mask != self.size] = 0
        overlap_mask[overlap_mask != 0] = 255
        self.overlap_mask = overlap_mask.astype(np.uint8)
        self.overlap_mask_bbox_xywh = warp_tools.xy2bbox(warp_tools.mask2xy(overlap_mask))

    def calc_warped_img_size(self):
        """Determine the shape of the registered images
        """
        min_x = np.inf
        max_x = 0
        min_y = np.inf
        max_y = 0
        for i in range(self.size):
            img_obj = self.img_obj_list[i]
            M = img_obj.T @ img_obj.to_prev_A @ img_obj.optimal_M
            img_corners_rc = warp_tools.get_corners_of_image(img_obj.image.shape)
            warped_corners_xy = warp_tools.warp_xy(img_corners_rc[:, ::-1], M)

            min_x = np.min([np.min(warped_corners_xy[:, 0]), min_x])
            max_x = np.max([np.max(warped_corners_xy[:, 0]), max_x])
            min_y = np.min([np.min(warped_corners_xy[:, 1]), min_y])
            max_y = np.max([np.max(warped_corners_xy[:, 1]), max_y])

        w = int(np.ceil(max_x - min_x))
        h = int(np.ceil(max_y - min_y))

        return np.array([h, w])

    def finalize(self):
        """Combine transformation matrices and get final shape of registered images
        """

        min_x = np.inf
        max_x = 0
        min_y = np.inf
        max_y = 0
        M_list = [None] * self.size
        for i in tqdm(range(self.size)):

            img_obj = self.img_obj_list[i]

            M = img_obj.T @ img_obj.to_prev_A @ img_obj.optimal_M
            M_list[i] = M

            img_corners_rc = warp_tools.get_corners_of_image(img_obj.image.shape)
            warped_corners_xy = warp_tools.warp_xy(img_corners_rc[:, ::-1], M)

            min_x = np.min([np.min(warped_corners_xy[:, 0]), min_x])
            max_x = np.max([np.max(warped_corners_xy[:, 0]), max_x])
            min_y = np.min([np.min(warped_corners_xy[:, 1]), min_y])
            max_y = np.max([np.max(warped_corners_xy[:, 1]), max_y])

        w = int(np.ceil(max_x - min_x))
        h = int(np.ceil(max_y - min_y))
        crop_T = np.identity(3)
        crop_T[0, 2] = min_x
        crop_T[1, 2] = min_y

        for i, img_obj in enumerate(self.img_obj_list):
            img_obj.crop_T = crop_T
            img_obj.M = M_list[i] @ crop_T
            img_obj.M_inv = np.linalg.inv(img_obj.M)
            img_obj.registered_img = warp_tools.warp_img(img_obj.image,
                                                         M=img_obj.M,
                                                         out_shape_rc=(h, w))

            img_obj.registered_shape_rc = img_obj.registered_img.shape[0:2]

        self.get_overlap_mask()

    def clear_unused_matches(self):
        """Clear up space by removing unused matches between Zimages

        Will only keep matches between each ZImage and the previous
        Zimage in the stack

        """

        for i, img_obj in enumerate(self.img_obj_list):
            if i == 0:
                prev_img_obj = None
            else:
                prev_img_obj = self.img_obj_list[i-1]

            if i == self.size - 1:
                next_img_obj = None
            else:
                next_img_obj = self.img_obj_list[i+1]

            img_obj.reduce(prev_img_obj, next_img_obj)

    def summarize(self):
        """Summarize alignment error

        Returns
        -------
        summary_df: Dataframe
            Pandas dataframe containin the registration error of the
            alignment between each image and the previous one in the stack.

        """

        src_img_names = [None] * self.size
        dst_img_names = [None] * self.size
        og_med_d_list = [None] * self.size
        og_tre_list = [None] * self.size
        med_d_list = [None] * self.size

        weighted_med_d_list = [None] * self.size
        tre_list = [None] * self.size
        shape_list = [None] * self.size
        for i in range(0, self.size):
            img_obj = self.img_obj_list[i]
            src_img_names[i] = img_obj.name
            shape_list[i] = img_obj.registered_img.shape
            if i == 0:
                continue

            prev_img_obj = self.img_obj_list[i-1]
            dst_img_names[i] = prev_img_obj.name

            current_to_prev_matches = img_obj.match_dict[prev_img_obj]
            temp_current_pts = current_to_prev_matches.matched_kp1_xy
            temp_prev_pts = current_to_prev_matches.matched_kp2_xy

            og_tre_list[i], og_med_d_list[i] = \
                warp_tools.measure_error(temp_current_pts,
                                         temp_prev_pts,
                                         img_obj.image.shape)

            current_pts = warp_tools.warp_xy(temp_current_pts, img_obj.M)
            prev_pts = warp_tools.warp_xy(temp_prev_pts, prev_img_obj.M)

            tre_list[i], med_d_list[i] = \
                warp_tools.measure_error(current_pts,
                                         prev_pts,
                                         img_obj.image.shape)

            similarities = \
                convert_distance_to_similarity(current_to_prev_matches.match_distances,
                                               current_to_prev_matches.matched_desc1.shape[0])

            _, weighted_med_d_list[i] = \
                warp_tools.measure_error(current_pts, prev_pts,
                                         img_obj.image.shape, similarities)

        summary_df = pd.DataFrame({
            "from": src_img_names,
            "to": dst_img_names,
            "original_D": og_med_d_list,
            "D": med_d_list,
            "D_weighted": weighted_med_d_list,
            "original_TRE": og_tre_list,
            "TRE": tre_list,
            "shape": shape_list,
        })

        summary_df["series_d"] = warp_tools.calc_total_error(med_d_list[1:])
        summary_df["series_tre"] = warp_tools.calc_total_error(tre_list[1:])
        summary_df["series_weighted_d"] = warp_tools.calc_total_error(weighted_med_d_list[1:])
        summary_df["name"] = self.name

        return summary_df


def register_images(img_dir, dst_dir=None, name="registrar",
                    feature_detector=VggFD(),
                    matcher=Matcher(), transformer=EuclideanTransform(),
                    affine_optimizer=None,
                    imgs_ordered=False, similarity_metric="n_matches",
                    max_scaling=3.0, qt_emitter=None):
    """
    Rigidly align collection of images

    Parameters
    ----------
    img_dir : str
        Path to directory containing the images that the user would like
        to be registered. These images need to be single channel, uint8 images

    dst_dir : str, optional
        Top directory where aliged images should be save. SerialRigidRegistrar will
        be in this folder, and aligned images in the "registered_images"
        sub-directory. If None, the images will not be written to file

    name : str, optional
        Descriptive name of registrar, such as the sample's name

    feature_detector : FeatureDD
            FeatureDD object that detects and computes image features.

    matcher : Matcher
        Matcher object that will be used to match image features

    transformer : scikit-image Transform object
        Transformer used to find transformation matrix that will warp each
        image to the target image.

    affine_optimizer : AffineOptimzer object
            Object that will minimize a cost function to find the
            optimal affine transoformations

    imgs_ordered : bool
        Boolean defining whether or not the order of images in img_dir
        are already in the correct order. If True, then each filename should
        begin with the number that indicates its position in the z-stack. If
        False, then the images will be sorted by ordering a feature distance
        matix.

    similarity_metric : str
        Metric used to calculate similarity between images, which is in turn
        used to build the distance matrix used to sort the images.

    summary : Dataframe
        Pandas dataframe containing the median distance between matched features
        before and after registration.

    qt_emitter : PySide2.QtCore.Signal, optional
        Used to emit signals that update the GUI's progress bars


    Returns
    -------
    registrar : SerialRigidRegistrar
        SerialRigidRegistrar object contains general information about the alginments,
        but also a list of Z-images. Each ZImage contains the warp information
        for an image in the stack, including the transformation matrices
        calculated at each step, keypoint poisions, image descriptors, and
        matches with other images. See attributes from Zimage for more
        information.

    """

    tic = time()
    if affine_optimizer is not None:
        if transformer.__class__.__name__ != affine_optimizer.transformation:
            print(Warning("Transformer is of type ",
                          transformer.__class__.__name__,
                          "but affine_optimizer optimizes the",
                          affine_optimizer.transformation,
                          ". Setting", transformer.__class__.__name__,
                          "as the transform to be optimized"))

            affine_optimizer.transformation = transformer.__class__.__name__

    if transformer.__class__.__name__ == "EuclideanTransform":
        matcher.scaling = False
    else:
        matcher.scaling = True

    registrar = SerialRigidRegistrar(img_dir, imgs_ordered=imgs_ordered, name=name)
    print("\n======== Detecting features\n")
    registrar.generate_img_obj_list(feature_detector, qt_emitter=qt_emitter)

    print("\n======== Matching images\n")
    if registrar.aleady_sorted:
        registrar.match_sorted_imgs(matcher, keep_unfiltered=False,
                                    qt_emitter=qt_emitter)

        for i, img_obj in enumerate(registrar.img_obj_list):
            img_obj.stack_idx = i

    else:
        registrar.match_imgs(matcher, keep_unfiltered=False,
                             qt_emitter=qt_emitter)

        print("\n======== Sorting images\n")
        registrar.build_metric_matrix(metric=similarity_metric)
        registrar.sort()

    registrar.distance_metric_name = matcher.metric_name
    registrar.distance_metric_type = matcher.metric_type
    print("\n======== Aligning down stack\n")
    if registrar.size > 2:
        registrar.update_match_dicts_with_neighbor_filter(transformer, matcher)

    registrar.align_to_prev(transformer=transformer, qt_emitter=qt_emitter)


    # Check current output shape. If too large,  then  registration failed
    estimated_out_shape_rc = registrar.calc_warped_img_size()
    actual_max_scaling = np.max([estimated_out_shape_rc/img_obj.image.shape[0:2] for img_obj in registrar.img_obj_list])
    if actual_max_scaling > max_scaling:
        print(Warning(f"Max allowed scaling is {max_scaling},\
                      but was calculated as being {actual_max_scaling}.\
                      Registration failed"))
        return False

    if affine_optimizer is not None:
        print("\n======== Optimizing alignments\n")
        registrar.optimize(affine_optimizer, qt_emitter=qt_emitter)

    registrar.finalize()

    if dst_dir is not None:
        registered_img_dir = os.path.join(dst_dir, "registered_images")
        registered_data_dir = os.path.join(dst_dir, "data")
        for d in [registered_img_dir, registered_data_dir]:
            pathlib.Path(d).mkdir(exist_ok=True, parents=True)

        print("\n======== Summarizing alignments\n")
        summary_df = registrar.summarize()
        summary_file = os.path.join(registered_data_dir, name + "_results.csv")
        summary_df.to_csv(summary_file, index=False)

        registrar.summary = summary_df

        print("\n======== Saving results\n")
        pickle_file = os.path.join(registered_data_dir, name + "_registrar.pickle")
        pickle.dump(registrar, open(pickle_file, 'wb'))

        n_digits = len(str(registrar.size))
        for img_obj in registrar.img_obj_list:
            f_out = "".join([str.zfill(str(img_obj.stack_idx), n_digits),
                             "_", img_obj.name, ".png"])

            io.imsave(os.path.join(registered_img_dir, f_out),
                      img_obj.registered_img.astype(np.uint8))

    registrar.clear_unused_matches()
    toc = time()
    elapsed = toc - tic
    time_string, time_units = valtils.get_elapsed_time_string(elapsed)

    print(f"\n======== Rigid registration complete in {time_string} {time_units}\n")

    return registrar
