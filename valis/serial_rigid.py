"""Classes and functions to perform serial rigid registration of a set of images
"""

import torch
import kornia
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
from tqdm import tqdm
import pathlib
from time import time
from pqdm.threads import pqdm
import functools
from colorama import Fore, Style
from scipy import stats
from copy import deepcopy

from . import valtils
from . import preprocessing
from . import warp_tools
from . import slide_tools
from .feature_detectors import VggFD
from .feature_matcher import Matcher, convert_distance_to_similarity, GMS_NAME
from . import feature_matcher
from . import valtils


DENOISE_MSG = "Denoising images"
FEATURE_MSG = "Detecting features"
MATCHING_MSG = "Matching images"
REMATCHING_MSG = "Re-matching images"
TRANSFORM_MSG = "Finding transforms"
OPTIMIZING_MSG = "Optimizing transforms"
FINALIZING_MSG = "Finalizing"

msg_list = [DENOISE_MSG, FEATURE_MSG, MATCHING_MSG, TRANSFORM_MSG, FINALIZING_MSG, OPTIMIZING_MSG]
DENOISE_MSG, FEATURE_MSG, MATCHING_MSG, TRANSFORM_MSG, FINALIZING_MSG, OPTIMIZING_MSG = valtils.pad_strings(msg_list)

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
                slide_tools.get_img_type(os.path.join(img_dir, f)) is not None]

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

    fixed_obj : ZImage
        ZImage to which this ZImage was aligned, i.e. this is the "moving"
        image, and `fixed_obj` is the "fixed" image. This is set during
        the `align_to_prev` method of the SerialRigidRegistrar. The
        `fixed_obj` will either be immediately above or immediately
        below this ZImage in the image stack.

    reflection_M : ndarray
        Transformation to reflect the image in the x and/or y axis, before padding.
        Will be the first transformation performed

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
        self.fixed_obj = None

        self.padded_shape_rc = None
        self.reflection_M = np.identity(3)
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

    reference_img_f : str
        Filename of image that will be treated as the center of the stack.

    reference_img_idx : int
        Index of ZImage that corresponds to `reference_img_f`, after
        the `img_obj_list` has been sorted.

    align_to_reference : bool, optional
        Whether or not images should be aligne to a reference image
        specified by `reference_img_f`. Will be set to True if
        `reference_img_f` is provided.

    iter_order : list of tuples
        Each element of `iter_order` contains a tuple of stack
        indices. The first value is the index of the moving/current/from
        image, while the second value is the index of the moving/next/to
        image.

    summary_df : Dataframe
        Pandas dataframe containin the registration error of the
        alignment between each image and the previous one in the stack.

    """

    def __init__(self, img_dir, imgs_ordered=False, reference_img_f=None,
                 name=None, align_to_reference=False):
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

        reference_img_f : str, optional
            Filename of image that will be treated as the center of the stack.
            If None, the index of the middle image will be the reference.

        name : str, optional
            Descriptive name of registrar, such as the sample's name

        align_to_reference : bool, optional
            Whether or not images should be aligne to a reference image
            specified by `reference_img_f`. Will be set to True if
            `reference_img_f` is provided.

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

        self.reference_img_f = reference_img_f
        self.reference_img_idx = 0
        self.align_to_reference = align_to_reference
        self.iter_order = None

        self.summary = None

        if self.align_to_reference is False and reference_img_f is not None:
            og_ref_name = valtils.get_name(reference_img_f)
            msg = (f"The reference was specified as {og_ref_name} ",
                   f"but `align_to_reference` is `False`, and so images will be aligned serially *towards* the reference image. ",
                   f"If you would like all images to be *directly* aligned to {og_ref_name}, "
                   f"then set `align_to_reference` to `True`. Note that in both cases, {og_ref_name} will remain unwarped.")
            valtils.print_warning(msg)

    def generate_img_obj_list(self, feature_detector, valis_obj=None, qt_emitter=None):
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

        # NOTE tried parallelizing, but it's actually slower  #
        sorted_img_list = [io.imread(os.path.join(self.img_dir, f), True)
                           for f in self.img_file_list]

        out_w, out_h = get_max_image_dimensions(sorted_img_list)

        # Get dimensions if images were rotated 45 degrees
        rad_45 = np.deg2rad(45)
        max_new_w = out_w*np.cos(rad_45) + out_h*np.sin(rad_45)
        max_new_h = out_w*np.sin(rad_45) + out_h*np.cos(rad_45)

        max_dist = np.ceil(np.max([out_w, out_h, max_new_h, max_new_w])).astype(int)
        out_shape = (max_dist, max_dist)
        img_obj_list = [None] * self.size

        for i in tqdm(range(self.size), desc=FEATURE_MSG, unit="image", leave=None):
            img_f = self.img_file_list[i]
            img = sorted_img_list[i]

            img_name = valtils.get_name(img_f)
            img_obj = ZImage(img, os.path.join(self.img_dir, img_f), i, name=img_name)
            img_obj.padded_shape_rc = out_shape
            img_obj.T = warp_tools.get_padding_matrix(img.shape, img_obj.padded_shape_rc)

            if feature_detector is not None:
                detect_img = self.get_fd_detection_img(img_obj, feature_detector=feature_detector, valis_obj=valis_obj)
                img_obj.kp_pos_xy, img_obj.desc = feature_detector.detect_and_compute(detect_img)

            img_obj_list[i] = img_obj
            self.img_obj_dict[img_name] = img_obj
            if qt_emitter is not None:
                qt_emitter.emit(1)

        self.img_obj_list = img_obj_list
        self.features = feature_detector.__class__.__name__

    def match_sorted_imgs(self, matcher_obj, keep_unfiltered=False, valis_obj=None, use_images=False, parallel=True, qt_emitter=None):
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

        for i in range(self.size):
            img_obj = self.img_obj_list[i]
            if i > 0:
                prev_img_obj = self.img_obj_list[i-1]
                self.match_img_obj_pairs(img_obj, prev_img_obj, matcher_obj, keep_unfiltered=keep_unfiltered, valis_obj=valis_obj)

            if i < self.size - 1:
                next_img_obj = self.img_obj_list[i+1]
                self.match_img_obj_pairs(img_obj, next_img_obj, matcher_obj, keep_unfiltered=keep_unfiltered, valis_obj=valis_obj)

    def match_img_obj_pairs(self, img_obj_1, img_obj_2, matcher_obj, valis_obj=None, keep_unfiltered=False, qt_emitter=None):
        if matcher_obj.match_filter_method == GMS_NAME:
            filter_kwargs = {"img1_shape":img_obj_1.image.shape[0:2], "img2_shape": img_obj_2.image.shape[0:2]}
        else:
            filter_kwargs = {}

        img1 = self.get_fd_detection_img(img_obj_1, matcher_obj.feature_detector, valis_obj)
        img2 = self.get_fd_detection_img(img_obj_2, matcher_obj.feature_detector, valis_obj)

        unfiltered_match_info12, filtered_match_info12, unfiltered_match_info21, filtered_match_info21 = \
            matcher_obj.match_images(img1=img1, desc1=img_obj_1.desc, kp1_xy=img_obj_1.kp_pos_xy,
                                        img2=img2, desc2=img_obj_2.desc, kp2_xy=img_obj_2.kp_pos_xy,
                                        additional_filtering_kwargs=filter_kwargs,
                                        sorting_images=True)


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

    def match_imgs(self, matcher_obj, keep_unfiltered=False, valis_obj=None, qt_emitter=None):
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

        def match_img_obj(i, matcher_obj, keep_unfiltered, valis_obj):
            img_obj_1 = self.img_obj_list[i]
            for j in np.arange(i+1, self.size):
                img_obj_2 = self.img_obj_list[j]
                self.match_img_obj_pairs(img_obj_1, img_obj_2, matcher_obj, keep_unfiltered=keep_unfiltered, valis_obj=valis_obj)

        match_adj_fxn = functools.partial(match_img_obj, matcher_obj=matcher_obj, keep_unfiltered=keep_unfiltered, valis_obj=valis_obj)

        n_cpu = valtils.get_ncpus_available() - 1
        res = pqdm(range(self.size), match_adj_fxn, n_jobs=n_cpu, desc=MATCHING_MSG, unit="image", leave=None)

    def get_common_desc(self, current_img_obj, neighbor_obj, nf_kp_idx):
        """Get descriptors that correspond to filtered neighbor points
            Parameters
            ----------
            nf_kp_idx : ndarray
                Indicies of already matched keypoints that were found after
                neighbonr filtering
        """

        neighbor_match_info12 = current_img_obj.match_dict[neighbor_obj]
        nf_kp = neighbor_match_info12.matched_kp1_xy[nf_kp_idx]
        nf_desc = neighbor_match_info12.matched_desc1[nf_kp_idx]

        return  nf_desc, nf_kp

    def get_neighbor_matches_idx(self, img_obj, prev_img_obj, next_img_obj, window_size=100):
        """Get indices of features found in both neighbors

        Returns
        -------
        nf_prev_idx

        nf_next_idx


        """
        col_bins = np.arange(0, img_obj.image.shape[1], window_size)
        row_bins = np.arange(0, img_obj.image.shape[0], window_size)

        xy_to_prev = img_obj.match_dict[prev_img_obj].matched_kp1_xy
        xy_to_next = next_img_obj.match_dict[img_obj].matched_kp2_xy

        to_prev_density, _, _, xy_to_prev_idx = stats.binned_statistic_2d(xy_to_prev[:, 0], xy_to_prev[:, 1], None, "count", bins=[row_bins, col_bins])
        to_next_density, _, _, xy_to_next_idx = stats.binned_statistic_2d(xy_to_next[:, 0], xy_to_next[:, 1], None, "count", bins=[row_bins, col_bins])

        shared_pts, _, _  = np.intersect1d(xy_to_prev_idx, xy_to_next_idx, return_indices=True, assume_unique=False)
        nf_next_idx = np.where(np.isin(xy_to_next_idx, shared_pts))[0]
        nf_prev_idx = np.where(np.isin(xy_to_prev_idx, shared_pts))[0]


        return nf_prev_idx, nf_next_idx

    def neighbor_match_filtering(self, img_obj, prev_img_obj, next_img_obj, tform, window_size=50):
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
            warped_xy = warp_tools.warp_xy(src_xy, M)
            d = np.median(warp_tools.calc_d(warped_xy,  dst_xy))

            return d, M

        nf_prev_idx, nf_next_idx = self.get_neighbor_matches_idx(img_obj, prev_img_obj, next_img_obj, window_size=window_size)
        to_prev_match_info12 = img_obj.match_dict[prev_img_obj]
        to_prev_match_info21 = prev_img_obj.match_dict[img_obj]

        improved = False
        # Need at least 3 points for an affine transform
        if len(nf_prev_idx) >= 3:

            nf_moving_kp = to_prev_match_info12.matched_kp1_xy[nf_prev_idx, :]
            nf_fixed_kp = to_prev_match_info12.matched_kp2_xy[nf_prev_idx, :]

            original_with_neighbor_filter_d, _ = measure_d(nf_moving_kp,
                                                           nf_fixed_kp,
                                                           tform)

            original_d, _ = measure_d(to_prev_match_info12.matched_kp1_xy,
                                      to_prev_match_info12.matched_kp2_xy,
                                      tform)

            improved = original_with_neighbor_filter_d <= original_d
            if improved:
                msg = (f"Neighbor match filtering improved alignment, reducing error from {original_d:.4} to {original_with_neighbor_filter_d:.4}, "
                       f"but also descreased the number of matches from {to_prev_match_info12.n_matches} to {len(nf_prev_idx)}")
                valtils.print_warning(msg, None, Fore.GREEN)
                # neighbor filtering improved alignment

                # To update img_obj
                updated_prev_match_info12 = deepcopy(to_prev_match_info12)
                updated_prev_match_info12.matched_kp1_xy = nf_moving_kp
                updated_prev_match_info12.matched_kp2_xy = nf_fixed_kp
                updated_prev_match_info12.n_matches = nf_fixed_kp.shape[0]

                # To update prev_img_obj
                updated_prev_match_info21 = deepcopy(to_prev_match_info21)
                updated_prev_match_info21.matched_kp1_xy = nf_fixed_kp
                updated_prev_match_info21.matched_kp2_xy = nf_moving_kp
                updated_prev_match_info21.n_matches = nf_fixed_kp.shape[0]

        if improved:
            return improved, updated_prev_match_info12, updated_prev_match_info21
        else:
            return improved, to_prev_match_info12, to_prev_match_info21

    def update_match_dicts_with_neighbor_filter(self, tform, matcher_obj, window_size=50):
        """Remove poor matches by keeping only the matches found in neighbors

        Parameters
        ----------
        tform : skimage.transform object
            The scikit-image transform object that estimates the
            parameter matrix

        matcher_obj : Matcher
            Object to match features between images.

        """
        new_matches = {}
        for moving_idx, fixed_idx in self.iter_order:
            img_obj = self.img_obj_list[moving_idx]

            next_idx = 1*(moving_idx - fixed_idx) + moving_idx
            assert abs(next_idx - fixed_idx) == 2

            if next_idx < 0 or next_idx >= self.size:
                continue

            prev_img_obj = self.img_obj_list[fixed_idx]
            next_img_obj = self.img_obj_list[next_idx]

            improved, updated_prev_match_info12, updated_prev_match_info21 = \
                  self.neighbor_match_filtering(img_obj, prev_img_obj, next_img_obj, tform,
                  window_size=window_size)

            if improved:
                new_matches[img_obj.name] = [updated_prev_match_info12, updated_prev_match_info21]

        # Update matches
        for moving_idx, fixed_idx in self.iter_order:
            img_obj = self.img_obj_list[moving_idx]
            if not img_obj.name in new_matches:
                continue

            prev_img_obj = self.img_obj_list[fixed_idx]

            img_obj_new_matches = new_matches[img_obj.name]
            img_obj.match_dict[prev_img_obj] = img_obj_new_matches[0]
            prev_img_obj.match_dict[img_obj] = img_obj_new_matches[1]


    def get_fd_detection_img(self, img_obj, feature_detector, valis_obj=None):
        if feature_detector.rgb and valis_obj is not None:
            slide_obj = valis_obj.get_slide(img_obj.name)
            if slide_obj.is_rgb:
                detect_img, mask, original_shape_rc, uncropped_shape_rc, crop_bbox = valis_obj.get_roi_for_processing(slide_obj, processing_cls=None, mask=slide_obj.rigid_reg_mask)
            else:
                detect_img = img_obj.image
        else:
            detect_img = img_obj.image

        return detect_img

    def rematch(self, matcher_obj, valis_obj=None, keep_unfiltered=False):
        """
        Reclaculate features using feature detecror in `matcher_obj`, and then match with `matcher_obj`.
        Use existing features to estimate rotation. Apply rotation to moving image, detect features, and match.

        """

        ref_img_obj = self.img_obj_list[self.reference_img_idx]
        ref_img = self.get_fd_detection_img(ref_img_obj, feature_detector=matcher_obj.feature_detector, valis_obj=valis_obj)
        ref_kp, ref_desc = matcher_obj.feature_detector.detect_and_compute(ref_img)

        # Need existing matches to estimate rotation. Will dictionaries below to update matches once complete
        updated_matches12 = {img_obj.name: {} for img_obj in self.img_obj_list}
        updated_matches21 = {img_obj.name: {} for img_obj in self.img_obj_list}
        updated_filtered_matches12 = {img_obj.name: {} for img_obj in self.img_obj_list}
        updated_filtered_matches21 = {img_obj.name: {} for img_obj in self.img_obj_list}

        updated_kp = {ref_img_obj.name: ref_kp}
        updated_desc = {ref_img_obj.name: ref_desc}

        for moving_idx, fixed_idx in tqdm(self.iter_order, desc=REMATCHING_MSG, unit="image", leave=None):

            img_obj = self.img_obj_list[moving_idx]
            prev_img_obj = self.img_obj_list[fixed_idx]

            detect_img = self.get_fd_detection_img(img_obj, feature_detector=matcher_obj.feature_detector, valis_obj=valis_obj)

            if fixed_idx == self.reference_img_idx:
                prev_img = ref_img
                prev_kp = ref_kp
                prev_desc = ref_desc
                prev_M = np.eye(3)
                prev_detect_img = ref_img

            if matcher_obj.match_filter_method == GMS_NAME:
                filter_kwargs = {"img1_shape":prev_img_obj.image.shape[0:2], "img2_shape": img_obj.image.shape[0:2]}
            else:
                filter_kwargs = {}

            if hasattr(matcher_obj, "estimate_rotation"):
                og_match_info21 = prev_img_obj.match_dict[img_obj]
                moving_kp_xy = og_match_info21.matched_kp2_xy
                fixed_kp_xy = warp_tools.warp_xy(og_match_info21.matched_kp1_xy, prev_M)

                rotation_tform = transform.SimilarityTransform()
                rotation_tform.estimate(fixed_kp_xy, moving_kp_xy)

                img_corners_xy = warp_tools.get_corners_of_image(detect_img.shape[0:2])[:, ::-1]
                warped_corners_xy = warp_tools.warp_xy(img_corners_xy, M=rotation_tform.params)
                max_corners_rc = np.ceil(np.max(warped_corners_xy[:, ::-1], axis=0)).astype(int)
                min_corners_rc = np.floor(np.min(warped_corners_xy[:, ::-1], axis=0)).astype(int)
                M = rotation_tform.params.copy()
                warped_shape_rc = max_corners_rc.copy()
                if np.any(min_corners_rc < 0):
                    T = np.eye(3)
                    if min_corners_rc[0] < 0:
                        warped_shape_rc[0] -= min_corners_rc[0]
                        T[1, 2] = min_corners_rc[0]

                    if min_corners_rc[1] < 0:
                        warped_shape_rc[1] -= min_corners_rc[1]
                        T[0, 2] = min_corners_rc[1]

                    M = M @ T

                if matcher_obj.feature_detector.rgb and valis_obj is not None:
                    slide_obj = valis_obj.get_slide(img_obj.name)
                    bg_color = slide_obj.bg_color
                else:
                    bg_color = None

                rotated_img = warp_tools.warp_img(detect_img, M=M, out_shape_rc=warped_shape_rc, bg_color=bg_color)

            else:

                rotated_img = detect_img
                M = np.eye(3)

            rot_kp, rot_desc = matcher_obj.feature_detector.detect_and_compute(rotated_img)
            kp_in_og = warp_tools.warp_xy(rot_kp, np.linalg.inv(M))

            updated_kp[img_obj.name] = kp_in_og
            updated_desc[img_obj.name] = rot_desc


            match_info12, filtered_match_info12, match_info21, filtered_match_info21 = \
                matcher_obj.match_images(img1=prev_img, img2=rotated_img,
                                         desc1=prev_desc, kp1_xy=prev_kp,
                                         desc2=rot_desc, kp2_xy=rot_kp)

            # Update match info keypoints to their location in original, unrotated image
            prev_matched_kp_in_og = warp_tools.warp_xy(match_info12.matched_kp1_xy, M=np.linalg.inv(prev_M))
            matched_kp_in_og = warp_tools.warp_xy(match_info12.matched_kp2_xy, M=np.linalg.inv(M))
            match_info12.matched_kp1_xy = prev_matched_kp_in_og
            match_info12.matched_kp2_xy = matched_kp_in_og

            match_info21.matched_kp1_xy = matched_kp_in_og
            match_info21.matched_kp2_xy = prev_matched_kp_in_og

            filtered_matched_kp_in_og = warp_tools.warp_xy(filtered_match_info12.matched_kp2_xy, M=np.linalg.inv(M))
            prev_filtered_matched_kp_in_og = warp_tools.warp_xy(filtered_match_info12.matched_kp1_xy, M=np.linalg.inv(prev_M))
            filtered_match_info12.matched_kp1_xy = prev_filtered_matched_kp_in_og
            filtered_match_info12.matched_kp2_xy = filtered_matched_kp_in_og
            filtered_match_info21.matched_kp1_xy = filtered_matched_kp_in_og
            filtered_match_info21.matched_kp2_xy = prev_filtered_matched_kp_in_og

            updated_filtered_matches12[prev_img_obj.name][img_obj.name] = filtered_match_info12
            updated_filtered_matches21[img_obj.name][prev_img_obj.name] = filtered_match_info21

            if keep_unfiltered:
                updated_matches12[prev_img_obj.name][img_obj.name] = match_info12
                updated_matches21[img_obj.name][prev_img_obj.name] = match_info21


            # Update for next step. Use rotated info
            prev_img_obj = img_obj
            prev_img = rotated_img
            prev_detect_img = detect_img
            prev_kp = rot_kp
            prev_desc = rot_desc
            prev_M = M

        def _estimate_error(kp1, kp2):
            error_estimator = transform.SimilarityTransform()
            error_estimator.estimate(kp1, kp2)
            warped_1 = error_estimator(kp1)
            estimated_d = np.mean(warp_tools.calc_d(warped_1, kp2))

            return estimated_d

        new_matcher_str = f"{matcher_obj.__class__.__name__} using {matcher_obj.feature_detector.__class__.__name__} features"
        old_matcher_str = list(ref_img_obj.match_dict.values())[0].feature_detector_name

        ref_img_obj.kp_pos_xy = updated_kp[ref_img_obj.name]
        ref_img_obj.desc = updated_desc[ref_img_obj.name]
        for moving_idx, fixed_idx in tqdm(self.iter_order):
            img_obj = self.img_obj_list[moving_idx]
            prev_img_obj = self.img_obj_list[fixed_idx]
            new_kp = updated_kp[img_obj.name]
            new_desc = updated_desc[img_obj.name]

            img_obj.kp_pos_xy = new_kp
            img_obj.desc = new_desc

            old_matches = img_obj.match_dict[prev_img_obj] # Here, img_obj = 1, prev_img_obj = 2
            new_matches21 = updated_filtered_matches21[img_obj.name][prev_img_obj.name]

            n_old_matches = old_matches.n_matches
            n_new_matches = new_matches21.n_matches

            old_d = _estimate_error(old_matches.matched_kp1_xy, old_matches.matched_kp2_xy)
            new_d = _estimate_error(new_matches21.matched_kp1_xy, new_matches21.matched_kp2_xy)

            lower_d = new_d < old_d
            more_matches = n_new_matches > n_old_matches

            msg_clr = Fore.YELLOW
            comparision = "mixed"
            action = "combine old and new matches"
            if lower_d and more_matches:
                comparision = "better"
                action = "replace old matches"
                msg_clr = Fore.GREEN
            elif not lower_d and not more_matches:
                comparision = "worse"
                action = "ignore new matches"

            msg = (f"When matching {img_obj.name} to {prev_img_obj.name}, the results are {comparision} when using {new_matcher_str}. "
                   f"Number of matches: {new_matcher_str} = {n_new_matches}, {old_matcher_str} = {n_old_matches}. "
                   f"Mean distance between matches: {new_matcher_str} = {new_d:.4}, {old_matcher_str} = {old_d:.4}. "
                   f"Will {action} when estimating rigid transform.\n"
                  )

            valtils.print_warning(msg, None, rgb=msg_clr)

            if not lower_d and not more_matches:
                continue

            new_matches12 = updated_filtered_matches12[prev_img_obj.name][img_obj.name]

            if comparision == "mixed":
                match_feature_name = f"{old_matches.feature_detector_name} + {matcher_obj.feature_name}"
                _updated_kp1 = np.vstack([new_matches21.matched_kp1_xy, old_matches.matched_kp1_xy])
                _updated_kp2 = np.vstack([new_matches21.matched_kp2_xy, old_matches.matched_kp2_xy])
                updated_kp1, updated_kp2, good_idx = feature_matcher.filter_matches_ransac(_updated_kp1, _updated_kp2)
                combined_n_matches = len(good_idx)

                new_matches21.matched_kp1_xy = updated_kp1
                new_matches21.matched_kp2_xy = updated_kp2
                new_matches21.n_matches = combined_n_matches

                new_matches12.matched_kp1_xy = updated_kp2 #Here, img_obj is 2
                new_matches12.matched_kp2_xy = updated_kp1
                new_matches12.n_matches = combined_n_matches
            else:
                match_feature_name = matcher_obj.feature_name


            new_matches21.feature_detector_name = match_feature_name
            new_matches12.feature_detector_name = match_feature_name

            img_obj.match_dict[prev_img_obj] = new_matches21
            prev_img_obj.match_dict[img_obj] = new_matches12

            if keep_unfiltered:
                new_unfiltered_matches21 = updated_matches21[img_obj.name][prev_img_obj.name]
                new_unfiltered_matches12 = updated_matches12[prev_img_obj.name][img_obj.name]
                old_unfiltered_matches = img_obj.match_dict[prev_img_obj] # Here, img_obj = 1, prev_img_obj = 2

                updated_unfiltered_kp1 = np.vstack([new_unfiltered_matches21.matched_kp1_xy, old_unfiltered_matches.matched_kp1_xy])
                updated_unfiltered_kp2 = np.vstack([new_unfiltered_matches21.matched_kp2_xy, old_unfiltered_matches.matched_kp2_xy])

                combined_n_unfiltered_matches = updated_unfiltered_kp2.shape[0]

                new_unfiltered_matches21.matched_kp1_xy = updated_unfiltered_kp1
                new_unfiltered_matches21.matched_kp2_xy = updated_unfiltered_kp2
                new_unfiltered_matches21.n_matches = combined_n_unfiltered_matches

                new_unfiltered_matches12.matched_kp1_xy = updated_unfiltered_kp2 #Here, img_obj is 2
                new_unfiltered_matches12.matched_kp2_xy = updated_unfiltered_kp1
                new_unfiltered_matches12.n_matches = combined_n_unfiltered_matches

                new_unfiltered_matches21.feature_detector_name = match_feature_name
                new_unfiltered_matches12.feature_detector_name = match_feature_name

                img_obj.unfiltered_match_dict[prev_img_obj] = new_unfiltered_matches21
                prev_img_obj.unfiltered_match_dict[img_obj] = new_unfiltered_matches12

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
        self.img_obj_list = [self.img_obj_list[i] for i in sorted_idx]
        for z, img_obj in enumerate(self.img_obj_list):
            img_obj.stack_idx = z

    def get_iter_order(self):
        """Get order in which to align images

        Will treat the reference image as the center of the stack

        """
        if self.reference_img_f is not None:
            ref_img_name = valtils.get_name(self.reference_img_f)
        else:
            ref_img_name = None

        img_f_list = [img_obj.full_img_f for img_obj in self.img_obj_list]
        ref_img_idx = warp_tools.get_ref_img_idx(img_f_list, ref_img_name)
        self.reference_img_idx = ref_img_idx
        self.reference_img_f = self.img_obj_list[ref_img_idx].full_img_f
        self.iter_order = warp_tools.get_alignment_indices(self.size, ref_img_idx)
        for moving_idx, fixed_idx in self.iter_order:
            img_obj = self.img_obj_list[moving_idx]
            prev_img_obj = self.img_obj_list[fixed_idx]
            img_obj.fixed_obj = prev_img_obj

    def align_to_prev_check_reflections(self, transformer, matcher_obj, valis_obj=None, keep_unfiltered=False, qt_emitter=None):
        """Use key points to align current image to previous image in the stack, but checking if reflection improves alignment

        Parameters
        ---------
        transformer : skimage.transform object
            The scikit-image transform object that estimates the
            parameter matrix

        feature_detector : FeatureDD
            FeatureDD object that detects and computes image features.

        matcher_obj : Matcher
            Object to match features between images.

        keep_unfiltered : bool
            Whether or not matcher_obj should store unfiltered matches

        qt_emitter : PySide2.QtCore.Signal, optional
            Used to emit signals that update the GUI's progress bars

        """

        ref_img_obj = self.img_obj_list[self.reference_img_idx]
        for moving_idx, fixed_idx in tqdm(self.iter_order, desc=TRANSFORM_MSG, unit="image", leave=None):
            img_obj = self.img_obj_list[moving_idx]
            prev_img_obj = self.img_obj_list[fixed_idx]

            if fixed_idx == self.reference_img_idx:
                prev_M = ref_img_obj.T.copy()

            if matcher_obj.match_filter_method == GMS_NAME:
                filter_kwargs = {"img1_shape":img_obj.image.shape[0:2], "img2_shape": prev_img_obj.image.shape[0:2]}
            else:
                filter_kwargs = {}

            # Estimate current error without reflections. Don't need to re-detect and match features
            to_prev_match_info = img_obj.match_dict[prev_img_obj]
            transformer.estimate(to_prev_match_info.matched_kp2_xy, to_prev_match_info.matched_kp1_xy)
            unreflected_warped_src_xy = warp_tools.warp_xy(to_prev_match_info.matched_kp1_xy, transformer.params)
            _,  unreflected_d = warp_tools.measure_error(to_prev_match_info.matched_kp2_xy, unreflected_warped_src_xy, prev_img_obj.image.shape)

            reflected_d_vals = [unreflected_d]
            reflection_M = [np.eye(3)]
            transforms = [transformer.params]
            reflected_matches12 = [to_prev_match_info]
            reflected_matches21 = [prev_img_obj.match_dict[img_obj]]

            if keep_unfiltered and prev_img_obj in img_obj.unfiltered_match_dict:
                unfiltered_reflected_matches12 = [img_obj.unfiltered_match_dict[prev_img_obj]]
                unfiltered_reflected_matches21 = [prev_img_obj.unfiltered_match_dict[img_obj]]

            # Estimate error with reflections
            dst_xy = warp_tools.warp_xy(prev_img_obj.kp_pos_xy, prev_M)

            prev_detect_img = self.get_fd_detection_img(prev_img_obj, feature_detector=matcher_obj.feature_detector, valis_obj=valis_obj)
            prev_warped = warp_tools.warp_img(prev_detect_img, prev_M, out_shape_rc=prev_img_obj.padded_shape_rc)

            detect_img = self.get_fd_detection_img(img_obj, feature_detector=matcher_obj.feature_detector, valis_obj=valis_obj)
            for rx in [False, True]:
                for ry in [False, True]:
                    if not rx and not ry:
                        continue

                    rM = warp_tools.get_reflection_M(rx, ry, img_obj.image.shape)
                    reflected_img = warp_tools.warp_img(detect_img, rM @ img_obj.T, out_shape_rc=img_obj.padded_shape_rc)

                    reflected_src_xy, reflected_desc = matcher_obj.feature_detector.detect_and_compute(reflected_img)
                    unfiltered_match_info12, filtered_match_info12, unfiltered_match_info21, filtered_match_info21 = \
                        matcher_obj.match_images(img1=reflected_img, desc1=reflected_desc, kp1_xy=reflected_src_xy,
                                                 img2=prev_warped, desc2=prev_img_obj.desc, kp2_xy=dst_xy,
                                                 additional_filtering_kwargs=filter_kwargs,
                                                 **filter_kwargs)

                    # Record info #
                    _ = transformer.estimate(filtered_match_info12.matched_kp2_xy, filtered_match_info12.matched_kp1_xy)
                    reflected_warped_src_xy = warp_tools.warp_xy(filtered_match_info12.matched_kp1_xy, transformer.params)
                    _,  reflected_d = warp_tools.measure_error(filtered_match_info12.matched_kp2_xy, reflected_warped_src_xy, prev_img_obj.padded_shape_rc)
                    reflected_d_vals.append(reflected_d)
                    reflection_M.append(rM)
                    transforms.append(transformer.params)

                    # Move matched features to position in original images
                    img_inv_M = np.linalg.inv(rM @ img_obj.T)
                    prev_img_inv_M = np.linalg.inv(prev_M)

                    filtered_match_info12.matched_kp1_xy = warp_tools.warp_xy(filtered_match_info12.matched_kp1_xy, img_inv_M)
                    filtered_match_info12.matched_kp2_xy = warp_tools.warp_xy(filtered_match_info12.matched_kp2_xy, prev_img_inv_M)

                    filtered_match_info21.matched_kp1_xy = warp_tools.warp_xy(filtered_match_info21.matched_kp1_xy, prev_img_inv_M)
                    filtered_match_info21.matched_kp2_xy = warp_tools.warp_xy(filtered_match_info21.matched_kp2_xy, img_inv_M)

                    reflected_matches12.append(filtered_match_info12)
                    reflected_matches21.append(filtered_match_info21)

                    if keep_unfiltered:
                        unfiltered_match_info12.matched_kp1_xy = warp_tools.warp_xy(unfiltered_match_info12.matched_kp1_xy, img_inv_M)
                        unfiltered_match_info12.matched_kp2_xy = warp_tools.warp_xy(unfiltered_match_info12.matched_kp2_xy, prev_img_inv_M)

                        unfiltered_match_info21.matched_kp1_xy = warp_tools.warp_xy(unfiltered_match_info21.matched_kp1_xy, prev_img_inv_M)
                        unfiltered_match_info21.matched_kp2_xy = warp_tools.warp_xy(unfiltered_match_info21.matched_kp2_xy, img_inv_M)

                        unfiltered_reflected_matches12.append(unfiltered_match_info12)
                        unfiltered_reflected_matches21.append(unfiltered_match_info21)

            best_idx = np.argmin(reflected_d_vals)
            best_reflect_M = reflection_M[best_idx]
            best_M = transforms[best_idx]
            img_obj.to_prev_A = best_M
            img_obj.reflection_M = best_reflect_M
            prev_M = img_obj.reflection_M @ img_obj.T @ img_obj.to_prev_A

            ref_x, ref_y = best_reflect_M[[0, 1], [0, 1]] < 0
            if ref_x or ref_y:
                msg = f'detected relfections between {img_obj.name} and {prev_img_obj.name} along the'
                if ref_x and ref_y:
                    msg = f'{msg} x and y axes'
                elif ref_x:
                    msg = f'{msg} x axis'
                elif ref_y:
                    msg = f'{msg} y axis'
                msg = f'{msg}. Will include reflection for {img_obj.name}'
                valtils.print_warning(msg)

                # Update matches
                img_obj.match_dict[prev_img_obj] = reflected_matches12[best_idx]
                prev_img_obj.match_dict[img_obj] = reflected_matches21[best_idx]

                if keep_unfiltered:
                    img_obj.unfiltered_match_dict[prev_img_obj] = unfiltered_reflected_matches12[best_idx]
                    prev_img_obj.unfiltered_match_dict[img_obj] = unfiltered_reflected_matches21[best_idx]

            if qt_emitter is not None:
                qt_emitter.emit(1)

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
        ref_img_obj = self.img_obj_list[self.reference_img_idx]

        if qt_emitter is not None:
            qt_emitter.emit(1)

        for moving_idx, fixed_idx in tqdm(self.iter_order, desc=TRANSFORM_MSG, unit="image", leave=None):
            img_obj = self.img_obj_list[moving_idx]
            prev_img_obj = self.img_obj_list[fixed_idx]
            img_obj.fixed_obj = prev_img_obj

            if fixed_idx == self.reference_img_idx:
                prev_M = ref_img_obj.T.copy()

            to_prev_match_info = img_obj.match_dict[prev_img_obj]
            src_xy = warp_tools.warp_xy(to_prev_match_info.matched_kp1_xy, img_obj.T)
            dst_xy = warp_tools.warp_xy(to_prev_match_info.matched_kp2_xy, prev_M)

            transformer.estimate(dst_xy, src_xy)

            # Check quality. If poor, then set M to identity
            M = transformer.params

            M = self.check_M(src_xy, dst_xy, M)
            if np.all(M==np.eye(3)):
                msg = f"Rigid registration between {img_obj.name} and {prev_img_obj.name} appears to have failed. Will not rigidly warp {img_obj.name}"
                valtils.print_warning(msg)
                M = np.eye(3)

            img_obj.to_prev_A = M

            prev_M = img_obj.T @ img_obj.to_prev_A

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
        ref_img_obj = self.img_obj_list[self.reference_img_idx]
        ref_warped = warp_tools.warp_img(ref_img_obj.image, M=ref_img_obj.T,
                                         out_shape_rc=ref_img_obj.padded_shape_rc)
        if qt_emitter is not None:
            qt_emitter.emit(1)

        for moving_idx, fixed_idx in tqdm(self.iter_order, desc=OPTIMIZING_MSG, unit="image", leave=None):
            img_obj = self.img_obj_list[moving_idx]
            prev_img_obj = self.img_obj_list[fixed_idx]

            if prev_img_obj == ref_img_obj:
                prev_img = ref_warped
                prev_M = ref_img_obj.T

            M = img_obj.reflection_M @ img_obj.T @ img_obj.to_prev_A
            warped_img = warp_tools.warp_img(img_obj.image,
                                             M=M,
                                             out_shape_rc=img_obj.padded_shape_rc)

            to_prev_match_info = img_obj.match_dict[prev_img_obj]
            before_src_xy = warp_tools.warp_xy(to_prev_match_info.matched_kp1_xy, M)
            before_dst_xy = warp_tools.warp_xy(to_prev_match_info.matched_kp2_xy, prev_M)
            before_tre, before_med_d = warp_tools.measure_error(before_src_xy,
                                                                before_dst_xy,
                                                                warped_img.shape)

            # Get mask
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

            # Optimize area inside mask
            if affine_optimizer.accepts_xy:
                moving_xy = before_src_xy
                fixed_xy = before_dst_xy
            else:
                moving_xy = None
                fixed_xy = None

            with valtils.HiddenPrints():
                _, optimal_M, _ = affine_optimizer.align(moving=warped_img, fixed=prev_img,
                                                         mask=mask, initial_M=None,
                                                         moving_xy=moving_xy,
                                                         fixed_xy=fixed_xy)

            # Keep optimal M if it actually improved alignment
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
                msg = (f"Somehow optimization made things worse. "
                       f"Cost was {initial_cst} but is now {after_cst}"
                       f"KP medD was {before_med_d}, but is now {after_med_d}.")
                valtils.print_warning(msg)
                prev_img = warped_img

            prev_M = M @ img_obj.optimal_M

            if qt_emitter is not None:
                qt_emitter.emit(1)

    def calc_warped_img_size(self):
        """Determine the shape of the registered images
        """
        min_x = np.inf
        max_x = 0
        min_y = np.inf
        max_y = 0
        for i in range(self.size):
            img_obj = self.img_obj_list[i]
            M = img_obj.reflection_M @ img_obj.T @ img_obj.to_prev_A @ img_obj.optimal_M
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
        for i in tqdm(range(self.size), desc=FINALIZING_MSG, unit="image", leave=None):

            img_obj = self.img_obj_list[i]

            M = img_obj.reflection_M @ img_obj.T @ img_obj.to_prev_A @ img_obj.optimal_M
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
            img_obj.registered_img = warp_tools.warp_img(img=img_obj.image,
                                                         M=img_obj.M,
                                                         out_shape_rc=(h, w))

            img_obj.registered_shape_rc = img_obj.registered_img.shape[0:2]

    def check_M(self, src_xy, dst_xy, M):
        """
        Check if registration improved alignment

        Parameters
        ----------
        src_sxy : np.ndarray
            Source points matched to `dst_xy`

        dst_sxy : np.ndarray
            Dst points matched to `src_xy`

        M : np.ndarray
            Inverse transformation matrix to warp src_xy to dst_xy. Should be
            found by transform.estimate(dst_xy, src_xy)
        """
        warped_src_xy = warp_tools.warp_xy(src_xy, M)
        d_after_reg = np.mean(warp_tools.calc_d(dst_xy, warped_src_xy))
        d_before_reg = np.mean(warp_tools.calc_d(dst_xy, src_xy))
        if d_after_reg > d_before_reg:
            new_M = np.eye(3)
        else:
            new_M = M

        return new_M

    def wiggle_to_ref(self, transformer):
        """Compose rigid transforms to wiggle image to reference

        #. For each slide, get M that aligns it's rigidly warp points
        to it's fixed image's rigidly warped points. These will be `rolling_M`
        #. Then, for each slide, compose their `M` with each neighbor's `rolling M`
        until it gets to the reference slide

        """
        ref_obj = self.img_obj_list[self.reference_img_idx]
        # Find inverse transforms that will align rigid image to rigid neighbor
        rolling_M_list = [None] * self.size
        for img_obj in self.img_obj_list:
            if img_obj == ref_obj:
                continue

            matches = img_obj.match_dict[img_obj.fixed_obj]

            rigid_reg_moving_xy = warp_tools.warp_xy(matches.matched_kp1_xy, M=img_obj.M)
            rigid_reg_fixed_xy = warp_tools.warp_xy(matches.matched_kp2_xy, M=img_obj.fixed_obj.M)

            transformer.estimate(src=rigid_reg_fixed_xy, dst=rigid_reg_moving_xy)

            rolling_M = transformer.params
            rolling_M = self.check_M(rigid_reg_moving_xy, rigid_reg_fixed_xy, rolling_M)

            rolling_M_list[img_obj.stack_idx] = rolling_M

        # Compose rolling transforms
        wiggle_M_list = [None] * self.size
        for img_obj in self.img_obj_list:
            if img_obj == ref_obj:
                continue

            neighbor_slide = img_obj.fixed_obj
            wiggle_M = np.eye(3)
            while neighbor_slide != ref_obj:
                neighbor_rolling_M = rolling_M_list[neighbor_slide.stack_idx]
                wiggle_M = wiggle_M @ neighbor_rolling_M
                neighbor_slide = neighbor_slide.fixed_obj

            wiggle_M_list[img_obj.stack_idx] = wiggle_M

        # Update M
        for img_obj in self.img_obj_list:
            if img_obj == ref_obj:
                continue
            updated_M = img_obj.M @ wiggle_M_list[img_obj.stack_idx]
            img_obj.M = updated_M

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
            if i == self.reference_img_idx:
                continue

            prev_img_obj = img_obj.fixed_obj
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

        non_ref_idx = list(range(self.size))
        non_ref_idx.remove(self.reference_img_idx)
        summary_df["series_d"] = warp_tools.calc_total_error(summary_df.D.values[non_ref_idx])
        summary_df["series_tre"] = warp_tools.calc_total_error(summary_df.TRE.values[non_ref_idx])
        summary_df["series_weighted_d"] = warp_tools.calc_total_error(summary_df.D_weighted.values[non_ref_idx])
        summary_df["name"] = self.name

        return summary_df


def register_images(img_dir, dst_dir=None, name="registrar",
                    matcher=Matcher(),
                    matcher_for_sorting=Matcher(),
                    transformer=EuclideanTransform(),
                    affine_optimizer=None,
                    imgs_ordered=False, reference_img_f=None,
                    similarity_metric="n_matches",
                    check_for_reflections=False,
                    max_scaling=3.0, align_to_reference=False, qt_emitter=None, valis_obj=None, *args, **kwargs):
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

    reference_img_f : str, optional
        Filename of image that will be treated as the center of the stack.
        If None, the index of the middle image will be the reference.

    check_for_reflections : bool, optional
        Determine if alignments are improved by relfecting/mirroring/flipping
        images. Optional because it requires re-detecting features in each version
        of the images and then re-matching features, and so can be time consuming and
        not always necessary.

    similarity_metric : str
        Metric used to calculate similarity between images, which is in turn
        used to build the distance matrix used to sort the images.

    summary : Dataframe
        Pandas dataframe containing the median distance between matched features
        before and after registration.

    align_to_reference : bool, optional
        Whether or not images should be aligned to a reference image
        specified by `reference_img_f`.

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
        matcher_for_sorting.scaling = False
    else:
        matcher.scaling = True
        matcher_for_sorting.scaling = True

    registrar = SerialRigidRegistrar(img_dir,
                                     imgs_ordered=imgs_ordered,
                                     reference_img_f=reference_img_f,
                                     name=name,
                                     align_to_reference=align_to_reference)


    # Filter `registrar.img_file_list` to only include images in valis_obj.slide_dict
    # Can be useful if `img_dir` has images from previous runs, some of which aren't needed this time
    if valis_obj is not None:
        keep_imgs = [f for f in registrar.img_file_list if valtils.get_name(f) in valis_obj.slide_dict.keys()]
        registrar.img_file_list = keep_imgs
        registrar.size = len(keep_imgs)
        valis_obj.rigid_registrar = registrar

    # print("\n======== Detecting features\n")
    registrar.generate_img_obj_list(feature_detector=matcher_for_sorting.feature_detector, valis_obj=valis_obj, qt_emitter=qt_emitter)

    if valis_obj is not None:
        if valis_obj.create_masks:
            # Remove feature points outside of mask
            for img_obj in registrar.img_obj_dict.values():
                slide_obj = valis_obj.get_slide(img_obj.name)
                reg_mask = valis_obj.crop_rigid_reg_mask(slide_obj, mask=slide_obj.rigid_reg_mask)
                reg_mask = preprocessing.mask2bbox_mask(reg_mask)
                features_in_mask_idx = warp_tools.get_xy_inside_mask(xy=img_obj.kp_pos_xy, mask=reg_mask)
                if len(features_in_mask_idx) > 0:
                    img_obj.kp_pos_xy = img_obj.kp_pos_xy[features_in_mask_idx, :]
                    img_obj.desc = img_obj.desc[features_in_mask_idx, :]

    # print("\n======== Matching images\n")
    if registrar.aleady_sorted:
        registrar.match_sorted_imgs(matcher_for_sorting, keep_unfiltered=False,
                                    valis_obj=valis_obj,
                                    qt_emitter=qt_emitter)

        for i, img_obj in enumerate(registrar.img_obj_list):
            img_obj.stack_idx = i

    else:
        registrar.match_imgs(matcher_obj=matcher_for_sorting,
                             keep_unfiltered=False,
                             valis_obj=valis_obj,
                             qt_emitter=qt_emitter)

        # print("\n======== Sorting images\n")
        registrar.build_metric_matrix(metric=similarity_metric)
        registrar.sort()

    registrar.get_iter_order()
    registrar.distance_metric_name = matcher.metric_name
    registrar.distance_metric_type = matcher.metric_type
    # Recalculate features and match using feature detector built into the matcher
    do_rematch = (matcher_for_sorting.__class__.__name__ != matcher.__class__.__name__) \
              or (matcher_for_sorting.feature_detector.__class__.__name__ != matcher.feature_detector.__class__.__name__)

    if do_rematch:
        msg = (f"Images sorted using {matcher_for_sorting.feature_detector.__class__.__name__} features. "
               f"Will now use {matcher.__class__.__name__} to match images using {matcher.feature_detector.__class__.__name__} features")
        # Images sorted with feature_detector, but need to matched using matcher's feature_detector
        valtils.print_warning(msg, None)
        registrar.rematch(matcher_obj=matcher, valis_obj=valis_obj, keep_unfiltered=False)

    # print("\n======== Calculating transformations\n")
    if registrar.size > 2:
        registrar.update_match_dicts_with_neighbor_filter(transformer, matcher)

    if check_for_reflections:
        registrar.align_to_prev_check_reflections(transformer=transformer,
                                                  matcher_obj=matcher,
                                                  valis_obj=valis_obj,
                                                  keep_unfiltered=False,
                                                  qt_emitter=qt_emitter)
    else:
        registrar.align_to_prev(transformer=transformer, qt_emitter=qt_emitter)

    # Check current output shape. If too large, then  registration failed
    for img_obj in registrar.img_obj_list:
        s = transform.SimilarityTransform(img_obj.M).scale
        if s >= max_scaling or s <= 1/max_scaling:
            print(Warning(f"Max allowed scaling is {max_scaling},\
                          but was calculated as being {s}.\
                          Registration failed. Maybe try using the Euclidean transform."))
            return False

    if affine_optimizer is not None:
        # print("\n======== Optimizing alignments\n")
        registrar.optimize(affine_optimizer, qt_emitter=qt_emitter)

    registrar.finalize()

    if align_to_reference:
        registrar.wiggle_to_ref(transformer)

    registrar.transformer = transformer
    if dst_dir is not None:
        registered_img_dir = os.path.join(dst_dir, "registered_images")
        registered_data_dir = os.path.join(dst_dir, "data")
        for d in [registered_img_dir, registered_data_dir]:
            pathlib.Path(d).mkdir(exist_ok=True, parents=True)

        # print("\n======== Summarizing alignments\n")
        summary_df = registrar.summarize()
        summary_file = os.path.join(registered_data_dir, name + "_results.csv")
        summary_df.to_csv(summary_file, index=False)

        registrar.summary = summary_df

        # print("\n======== Saving results\n")
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
