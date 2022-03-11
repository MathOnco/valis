"""Classes and functions to perform serial non-rigid registration of a set of images

"""

import numpy as np
from skimage import transform, io, util
from tqdm import tqdm
import os
import re
import imghdr
from time import time
import pathlib
import pandas as pd
import pickle
from . import warp_tools
from . import non_rigid_registrars
from . import valtils
from . import serial_rigid
from . import viz


def get_ref_img_idx(img_f_list, ref_img_name=None):
    """Get index of reference image

    Parameters
    ----------
    img_f_list : list of str
        List of image file names

    ref_img_name : str, optional
        Name of image that will be treated as the center of the stack.
        If None, the index of the middle image will be returned.

    Returns
    -------
    ref_img_idx : int
        Index of reference image in img_f_list. Warnings are raised
        if `ref_img_name` matches either 0 or more than 1 images in `img_f_list`.

    """

    n_imgs = len(img_f_list)
    if ref_img_name is None:
        if n_imgs == 2:
            ref_img_idx = 0
        else:
            ref_img_idx = n_imgs // 2

    else:
        ref_img_name = os.path.split(ref_img_name)[1]

        name_matches = [re.search(ref_img_name, os.path.split(f)[1])
                        for f in img_f_list]

        ref_img_idx = [i for i in range(n_imgs) if name_matches[i] is not None]
        n_matches = len(ref_img_idx)

        if n_matches == 0:
            warning_msg = f"No files in img_f_list match {ref_img_name}"
            ref_img_idx = None
            valtils.print_warning(warning_msg)

        elif n_matches == 1:
            ref_img_idx = ref_img_idx[0]

        elif n_matches > 1:
            macthing_files = ", ".join(img_f_list[i] for i in ref_img_idx)
            warning_msg = f"More than 1 file in img_f_list matches {ref_img_name}. These files are: {macthing_files}"
            valtils.print_warning(warning_msg)

    return ref_img_idx


def get_alignment_indices(n_imgs, ref_img_idx=None):
    """Get indices to align in stack.

    Indices go from bottom to center, then top to center. In each case,
    the alignments go from closest to the center, to next closet, etc...
    The reference image is exclued from this list.
    For example, if `ref_img_idx` is 2, then the order is
    [(1, 2), (0, 1), (3, 2), ...,  (`n_imgs`-1, `n_imgs` - 2)].

    Parameters
    ----------
    n_imgs : int
        Number of images in the stack

    ref_img_idx : int, optional
        Position of reference image. If None, then this will set to
        the center of the stack

    Returns
    -------
    matching_indices : list of tuples
        Each element of `matching_indices` contains a tuple of stack
        indices. The first value is the index of the moving/current/from
        image, while the second value is the index of the moving/next/to
        image.

    """

    if ref_img_idx:
        ref_img_idx = n_imgs//2

    matching_indices = [None] * (n_imgs - 1)
    idx = 0
    for i in reversed(range(0, ref_img_idx)):
        current_idx = i
        next_idx = i + 1
        matching_indices[idx] = (current_idx, next_idx)
        idx += 1

    for i in range(ref_img_idx, n_imgs-1):
        current_idx = i + 1
        next_idx = i
        matching_indices[idx] = (current_idx, next_idx)
        idx += 1

    return matching_indices


def get_matching_xy_from_rigid_registrar(rigid_registrar, ref_img_name=None):
    """Get matching keypoints to use in serial non-rigid registration

    Parameters
    ----------
    rigid_registrar : SerialRigidRegistrar
        SerialRigidRegistrar that has aligned a series of images

    ref_img_name : str, optional
        Name of image that will be treated as the center of the stack.
        If None, the middle image will be used as the center

    Returns
    -------
    from_to_kp_dict : dict of list
        Key = image name, value = list of matched and aligned keypoints between
        each registered moving image and the registered fixed image.
        Each element in the list contains 2 arrays:

        #. Rigid registered xy in moving/current/from image
        #. Rigid registered xy in fixed/next/to image

    """

    img_f_list = [img_obj.full_img_f for img_obj in rigid_registrar.img_obj_list]
    ref_img_idx = get_ref_img_idx(img_f_list, ref_img_name)
    n_imgs = len(img_f_list)

    from_to_indices = get_alignment_indices(n_imgs, ref_img_idx)
    from_to_kp_dict = {}
    for idx in from_to_indices:

        moving_obj = rigid_registrar.img_obj_list[idx[0]]
        fixed_obj = rigid_registrar.img_obj_list[idx[1]]

        current_match_dict = moving_obj.match_dict[fixed_obj]
        moving_kp = current_match_dict.matched_kp1_xy
        fixed_kp = current_match_dict.matched_kp2_xy

        assert moving_kp.shape[0] == fixed_kp.shape[0]

        registered_moving = warp_tools.warp_xy(moving_kp, M=moving_obj.M)
        registered_fixed = warp_tools.warp_xy(fixed_kp, M=fixed_obj.M)

        from_to_kp_dict[moving_obj.name] = [registered_moving, registered_fixed]

    return from_to_kp_dict


def get_imgs_from_dir(src_dir):
    """Get images from source directory.

    Parameters
    ----------
    src_dir : str
        Location of images to be registered.

    Returns
    -------
    img_list : list of ndarray
        List of images to be registered

    img_f_list : list of str
        List of image file names

    img_names : list of str
        List of names for each image. Created by removing the extension

    """

    img_f_list = [f for f in os.listdir(src_dir) if
                  imghdr.what(os.path.join(src_dir, f)) is not None]

    valtils.sort_nicely(img_f_list)

    img_list = [io.imread(os.path.join(src_dir, f)) for f in img_f_list]
    # assert(img_list[0].dtype == np.uint8), valtils.print_warning("images must be uint8")

    img_names = [valtils.get_name(f) for f in img_f_list]

    return img_list, img_f_list, img_names


def get_imgs_rigid_reg(serial_rigid_reg):
    """Get images from SerialRigidRegistrar

    Parameters
    ----------
    serial_rigid_reg : SerialRigidRegistrar
        SerialRigidRegistrar that has rigidly aligned images

    Returns
    -------
    img_list : list of ndarray
        List of images to be registered

    img_f_list : list of str
        List of image file names

    img_names : list of str
        List of names for each image. Created by removing the extension

    """

    img_obj_list = serial_rigid_reg.img_obj_list
    img_list = [img_obj.registered_img for img_obj in img_obj_list]
    img_names = [img_obj.name for img_obj in img_obj_list]
    img_f_list = [img_obj.full_img_f for img_obj in img_obj_list]

    return img_list, img_f_list, img_names


class NonRigidZImage(object):
    """ Class that store info about an image, including both
    rigid and non-rigid registration parameters

    Attributes
    ----------

    image : ndarray
        Original, unwarped image with shape (P, Q)

    name : str
        Name of image.

    stack_idx : int
        Position of image in the stack

    moving_xy : ndarray, optional
        (V, 2) array containing points in the moving image that correspond
        to those in the fixed image. If these are provided, non_rigid_reg_class
        should be a subclass of non_rigid_registrars.NonRigidRegistrarXY

    fixed_xy : ndarray, optional
        (V, 2) array containing points in the fixed image that correspond
        to those in the moving image

    bk_dxdy : ndarray
        (2, N, M) numpy array of pixel displacements in
        the x and y directions from the reference image.
        dx = bk_dxdy[0], and dy=bk_dxdy[1].
        Used to warp images

    fwd_dxdy : ndarray
        Inversion of bk_dxdy. dx = fwd_dxdy[0], and dy=fwd_dxdy[1].
        Used to warp points

    warped_grid : ndarray
        Image showing deformation applied to a regular grid.

    """

    def __init__(self, image, name, stack_idx, moving_xy=None, fixed_xy=None):
        """
        Parameters
        ----------

        image : ndarray
            Original, unwarped image with shape (P, Q)

        name : str
            Name of image.

        stack_idx : int
            Position of image in the stack

        moving_xy : ndarray, optional
            (V, 2) array containing points in the moving image that correspond
            to those in the fixed image. If these are provided, non_rigid_reg_class
            should be a subclass of non_rigid_registrars.NonRigidRegistrarXY

        fixed_xy : ndarray, optional
            (V, 2) array containing points in the fixed image that correspond
            to those in the moving image

        """

        self.image = image
        self.name = name
        self.stack_idx = stack_idx
        self.moving_xy = moving_xy
        self.fixed_xy = fixed_xy
        self.registered_img = None
        self.warped_grid = None
        self.bk_dxdy = None
        self.fwd_dxdy = None

    def calc_deformation(self, registered_fixed_image, non_rigid_reg_class, bk_dxdy, params=None, mask=None):
        """
        Finds the non-rigid deformation fields that align this ("moving") image
        to the "fixed" image

        Parameters
        ----------
        registered_fixed_image : ndarray
            Adjacent, aligned image in the stack that this image is being
            aligned to. Has shape (P, Q)

        non_rigid_reg_class : NonRigidRegistrar
            Uninstantiated NonRigidRegistrar class that will be used to
            calculate the deformation fields between images

        params : dictionary, optional
            Keyword: value dictionary of parameters to be used in reigstration.
            Passed to the non_rigid_reg_class' calc() method.

            In the case where simple ITK will be used, params should be
            a SimpleITK.ParameterMap. Note that numeric values needd to be
            converted to strings.

        mask : ndarray, optional
            2D array with shape (P,Q) where non-zero pixel values are foreground,
            and 0 is background, which is ignnored during registration. If None,
            then all non-zero pixels in images will be used to create the mask.

        dxdy : ndarray, optional
            (2, P, Q) numpy array of pixel displacements in
            the x and y directions. dx = dxdy[0], and dy=dxdy[1].
            Used to warp the registered_img before finding deformation fields.

        """

        if bk_dxdy is not None:
            current_warp_map = warp_tools.get_warp_map(dxdy=bk_dxdy)
            moving_img = transform.warp(self.image, current_warp_map,
                                        preserve_range=True).astype(np.uint8)
        else:
            moving_img = self.image.copy()

        non_rigid_reg = non_rigid_reg_class(params=params)

        if self.moving_xy is not None and self.fixed_xy is not None and \
           issubclass(non_rigid_reg_class, non_rigid_registrars.NonRigidRegistrarXY):
            # Update positions #
            fwd_dxdy = warp_tools.get_inverse_field(bk_dxdy)
            fixed_xy = warp_tools.warp_xy(self.fixed_xy, M=None, fwd_dxdy=fwd_dxdy)
            moving_xy = warp_tools.warp_xy(self.moving_xy, M=None, fwd_dxdy=fwd_dxdy)
        else:
            fixed_xy = None
            moving_xy = None

        xy_args = {"moving_xy": moving_xy, "fixed_xy": fixed_xy}
        warped_moving, moving_grid_img, moving_bk_dxdy = \
            non_rigid_reg.register(moving_img=moving_img,
                                   fixed_img=registered_fixed_image,
                                   mask=mask,
                                   **xy_args)

        bk_dxdy_from_ref = [bk_dxdy[0] + moving_bk_dxdy[0],
                            bk_dxdy[1] + moving_bk_dxdy[1]]

        self.bk_dxdy = bk_dxdy_from_ref
        self.fwd_dxdy = warp_tools.get_inverse_field(bk_dxdy_from_ref)

        warp_map = warp_tools.get_warp_map(dxdy=bk_dxdy_from_ref)

        self.warped_grid = viz.color_displacement_grid(*bk_dxdy_from_ref)
        self.registered_img = transform.warp(self.image, warp_map, preserve_range=True).astype(np.uint8)


class SerialNonRigidRegistrar(object):
    """Class that performs serial non-rigid registration, based on results SerialRigidRegistrar

    A SerialNonRigidRegistrar finds the deformation fields that will non-rigidly align
    a series of images, using the rigid registration parameters found by a
    SerialRigidRegistrar object. There are two types of non-rigid registration
    methods:

    #. Images are aligned towards a reference image, which may or may not
    be at the center of the stack. In this case, the image directly "above" the
    reference image is aligned to the reference image, after which the image 2 steps
    above the reference image is aligned to the 1st (now aligned) image above
    the reference image, and so on. The process is similar when aligning images
    "below" the reference image.

    #. All images are aligned simultaneously, and so a reference image is not
    # required. An example is the SimpleElastix groupwise registration.

    Similar to SerialRigidRegistrar, SerialNonRigidRegistrar creates a list
    and dictionary of NonRigidZImage objects each of which contains information
    related to the non-rigid registration, including the original rigid
    transformation matrices, and the calculated deformation fields.

    Attributes
    ----------
    name : str, optional
        Optional name of this SerialNonRigidRegistrar

    from_dir : bool
        Whether or not the images are from a source directory. If False, then
        the images will be extracted from a SerialRigidRegistrar

    ref_image_name : str
        Name of mage that is being treated as the "center" of the stack.
        For example, this may be associated with an H+E image that is
        the 2nd image in a stack of 7 images.

    size : int
        Number of images to align

    shape : tuple of int
        Shape of each image to register. Must be the same for all images

    non_rigid_obj_dict : dict
        Dictionary, where each key is the name of a NonRigidZImage, and
        the value is the assocatiated NonRigidZImage

    non_rigid_reg_params: dictionary
        Dictionary containing parameters {name: value} to be used to initialize
        the NonRigidRegistrar.
        In the case where simple ITK is used by the, params should be
        a SimpleITK.ParameterMap. Note that numeric values nedd to be
        converted to strings.

    mask :  ndarray
        Mask used in non-rigid alignments, with shape (P, Q).

    mask_bbox_xywh : ndarray
        Bounding box of `mask` (top left x, top left y, width, height)

    summary : Dataframe
        Pandas dataframe containing the median distance between matched
        features before and after registration.

    """

    def __init__(self, src, ref_img_name=None, moving_to_fixed_xy=None, mask=None, name=None):
        """
        Parameters
        ----------
        src : SerialRigidRegistrar, str
            Either a SerialRigidRegistrar object that was used to optimally
            align a series of images, or a string indicating where the images
            to be aligned are located. If src is a string, the images should be
            named such that they are read in the correct order, i.e. each
            starting with a number.

        ref_img_name : str, optional
            Name of image that will be treated as the center of the stack.
            If None, the index of the middle image will be returned.

        moving_to_fixed_xy :  dict of list, or bool
            If `moving_to_fixed_xy` is a dict of list, then
            Key = image name, value = list of matched keypoints between
            each moving image and the fixed image.
            Each element in the list contains 2 arrays:

            #. Rigid registered xy in moving/current/from image
            #. Rigid registered xy in fixed/next/to image

            To deterime which pairs of images will be aligned, use
            `get_alignment_indices`. Can use `get_imgs_from_dir`
            to see the order inwhich the images will be read, which will correspond
            to the indices retuned by `get_alignment_indices`.

            If `src` is a SerialRigidRegistrar and `moving_to_fixed_xy` is
            True, then the matching features in the SerialRigidRegistrar will
            be used. If False, then matching features will not be used.

        mask :  ndarray, bool, optional
            Mask used in non-rigid alignments.

            If an ndarray, it must have the same size as the other images.

            If True, then the `overlap_mask` in the SerialRigidRegistrar
            will be used.

            If False or None, no mask will be used.

        name : optional
            Optional name for this SerialNonRigidRegistrar

        """

        self.src = src
        if isinstance(src, serial_rigid.SerialRigidRegistrar):
            self.from_dir = False
        elif isinstance(src, str):
            self.from_dir = True
        else:
            valtils.print_warning(f"src must be either a string or SerialRigidRegistrar")
            return None

        self.ref_img_name = None
        self.ref_img_idx = None
        self.name = name
        self.size = 0
        self.shape = None
        self.non_rigid_obj_dict = {}
        self.non_rigid_obj_list = None
        self.non_rigid_reg_params = None
        self.summary = None
        self.generate_non_rigid_obj_list(ref_img_name, moving_to_fixed_xy)
        self.set_mask(mask)

    def create_mask(self):
        temp_mask = np.zeros(self.shape, dtype=np.uint8)
        for nr_img_obj in self.non_rigid_obj_list:
            temp_mask[nr_img_obj.image > 0] = 255

        mask = warp_tools.bbox2mask(*warp_tools.xy2bbox(
                                    warp_tools.mask2xy(temp_mask)),
                                    temp_mask.shape)
        return mask

    def set_mask(self, mask):
        """Set mask and get its bounding box
        """

        if mask is not None:
            if isinstance(mask, bool) and not self.from_dir:
                mask = self.src.overlap_mask
            mask = np.clip(mask.astype(int)*255, 0, 255).astype(np.uint8)

        else:
            mask = self.create_mask()

        mask_bbox_xywh = warp_tools.xy2bbox(warp_tools.mask2xy(mask))
        self.mask = mask
        self.mask_bbox_xywh = mask_bbox_xywh

    def generate_non_rigid_obj_list(self, ref_img_name=None, moving_to_fixed_xy=None):
        """Create non_rigid_obj_list

        """

        if self.from_dir:
            img_list, img_f_list, img_names = get_imgs_from_dir(self.src)

        else:
            img_list, img_f_list, img_names = get_imgs_rigid_reg(self.src)

        self.size = len(img_list)

        ref_img_idx = get_ref_img_idx(img_f_list, ref_img_name)
        if ref_img_name is None:
            ref_img_name = img_names[ref_img_idx]
        else:
            ref_img_name = valtils.get_name(ref_img_name)

        self.ref_img_name = ref_img_name
        self.ref_img_idx = ref_img_idx

        if not self.from_dir and isinstance(moving_to_fixed_xy, bool):
            if moving_to_fixed_xy:
                moving_to_fixed_xy = \
                    get_matching_xy_from_rigid_registrar(self.src, ref_img_name)
            else:
                moving_to_fixed_xy = None

        self.non_rigid_obj_list = [None] * self.size
        series_shape = img_list[0].shape[0:2]
        self.shape = series_shape
        for i, img in enumerate(img_list):

            if isinstance(img[0,0], np.floating) and img.max() <= 1.0:
                img = util.img_as_ubyte(img)

            img = img.astype(np.uint8)

            assert img.shape[0:2] == series_shape, \
                valtils.print_warning("Images must all have the shape")

            img_name = img_names[i]
            moving_xy = None
            fixed_xy = None
            if moving_to_fixed_xy is not None and img_name != ref_img_name:
                if isinstance(moving_to_fixed_xy, dict):
                    xy_coords = moving_to_fixed_xy[img_name]
                    moving_xy = xy_coords[0]
                    fixed_xy = xy_coords[1]
                else:
                    msg = "moving_to_fixed_xy is not a dictionary. Will be ignored"
                    valtils.print_warning(msg)

            nr_obj = NonRigidZImage(img, img_name, stack_idx=i,
                                    moving_xy=moving_xy, fixed_xy=fixed_xy)

            if i == ref_img_idx:
                # Set reference image attributes #
                zero_displacement = np.zeros(img.shape[0:2])
                nr_obj.bk_dxdy = [zero_displacement, zero_displacement]
                nr_obj.fwd_dxdy = [zero_displacement, zero_displacement]
                nr_obj.warped_grid = viz.color_displacement_grid(*nr_obj.bk_dxdy)
                nr_obj.registered_img = img.copy()

            self.non_rigid_obj_list[i] = nr_obj

    def register_serial(self, non_rigid_reg_class, non_rigid_reg_params=None):
        """Non-rigidly align images in serial
        Parameters
        ----------
        non_rigid_reg_class : NonRigidRegistrar
            Uninstantiated NonRigidRegistrar class that will be used to
            calculate the deformation fields between images

        non_rigid_reg_params: dictionary, optional
            Dictionary containing parameters {name: value} to be used to initialize
            the NonRigidRegistrar.
            In the case where simple ITK is used by the, params should be
            a SimpleITK.ParameterMap. Note that numeric values nedd to be
            converted to strings.

        """

        self.non_rigid_reg_params = non_rigid_reg_params

        iter_order = get_alignment_indices(self.size, self.ref_img_idx)
        for moving_idx, fixed_idx in tqdm(iter_order):
            moving_obj = self.non_rigid_obj_list[moving_idx]
            fixed_obj = self.non_rigid_obj_list[fixed_idx]
            current_dxdy = fixed_obj.bk_dxdy

            moving_obj.calc_deformation(fixed_obj.registered_img,
                                        non_rigid_reg_class,
                                        current_dxdy,
                                        params=non_rigid_reg_params,
                                        mask=self.mask)

    def register_groupwise(self, non_rigid_reg_class, non_rigid_reg_params=None):
        """Non-rigidly align images as a group

        Parameters
        ----------
        non_rigid_reg_class : NonRigidRegistrarGroupwise
            Uninstantiated NonRigidRegistrar class that will be used to
            calculate the deformation fields between images

        non_rigid_reg_params: dictionary, optional
            Dictionary containing parameters {name: value} to be used to initialize
            the NonRigidRegistrar.
            In the case where simple ITK is used by the, params should be
            a SimpleITK.ParameterMap. Note that numeric values nedd to be
            converted to strings.

        """

        img_list = [nr_img_obj.image for nr_img_obj in self.non_rigid_obj_list]
        non_rigid_reg = non_rigid_reg_class(params=non_rigid_reg_params)

        print("\n======== Registering images (non-rigid)\n")
        warped_imgs, warped_grids, backward_deformations = non_rigid_reg.register(img_list, self.mask)
        for i, nr_img_obj in enumerate(self.non_rigid_obj_list):
            nr_img_obj.registered_img = warped_imgs[i]
            nr_img_obj.bk_dxdy = backward_deformations[i]
            nr_img_obj.warped_grid = viz.color_displacement_grid(*nr_img_obj.bk_dxdy)
            nr_img_obj.fwd_dxdy = warp_tools.get_inverse_field(nr_img_obj.bk_dxdy)

    def register(self, non_rigid_reg_class, non_rigid_reg_params):
        """Non-rigidly align images, either as a group or serially

        Images will be registered serially if `non_rigid_reg_class` is a
        subclass of NonRigidRegistrarGroupwise, then groupwise registration
        will be conductedd. If `non_rigid_reg_class` is a subclass of
        NonRigidRegistrar then images will be aligned serially.

        Parameters
        ----------
        non_rigid_reg_class : NonRigidRegistrar, NonRigidRegistrarGroupwise
            Uninstantiated NonRigidRegistrar or NonRigidRegistrarGroupwise class
            that will be used to calculate the deformation fields between images

        non_rigid_reg_params: dictionary, optional
            Dictionary containing parameters {name: value} to be used to initialize
            the NonRigidRegistrar.
            In the case where simple ITK is used by the, params should be
            a SimpleITK.ParameterMap. Note that numeric values nedd to be
            converted to strings.

        """

        if issubclass(non_rigid_reg_class, non_rigid_registrars.NonRigidRegistrarGroupwise):
            self.register_groupwise(non_rigid_reg_class, non_rigid_reg_params)
        else:
            self.register_serial(non_rigid_reg_class, non_rigid_reg_params)

        self.non_rigid_obj_dict = {img_obj.name: img_obj for img_obj
                                   in self.non_rigid_obj_list}

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
        shape_list = [None] * self.size

        og_med_d_list = [None] * self.size
        og_tre_list = [None] * self.size
        med_d_list = [None] * self.size
        tre_list = [None] * self.size

        src_img_names[self.ref_img_idx] = self.ref_img_name
        shape_list[self.ref_img_idx] = self.non_rigid_obj_list[self.ref_img_idx].image.shape

        iter_order = get_alignment_indices(self.size, self.ref_img_idx)
        print("\n======== Summarizing registration\n")
        for moving_idx, fixed_idx in tqdm(iter_order):
            moving_obj = self.non_rigid_obj_list[moving_idx]
            fixed_obj = self.non_rigid_obj_list[fixed_idx]
            src_img_names[moving_idx] = moving_obj.name
            dst_img_names[moving_idx] = fixed_obj.name
            shape_list[moving_idx] = moving_obj.image.shape

            og_tre_list[moving_idx], og_med_d_list[moving_idx] = \
                warp_tools.measure_error(moving_obj.moving_xy,
                                         moving_obj.fixed_xy,
                                         moving_obj.image.shape)

            warped_moving_xy = warp_tools.warp_xy(moving_obj.moving_xy,
                                                  M=None,
                                                  fwd_dxdy=moving_obj.fwd_dxdy)

            warped_fixed_xy = warp_tools.warp_xy(moving_obj.fixed_xy,
                                                 M=None,
                                                 fwd_dxdy=moving_obj.fwd_dxdy)

            tre_list[moving_idx], med_d_list[moving_idx] = \
                warp_tools.measure_error(warped_moving_xy,
                                         warped_fixed_xy,
                                         moving_obj.image.shape)

        summary_df = pd.DataFrame({
            "from": src_img_names,
            "to": dst_img_names,
            "original_D": og_med_d_list,
            "D": med_d_list,
            "original_TRE": og_tre_list,
            "TRE": tre_list,
            "shape": shape_list,
        })
        to_summarize_idx = [i for i in range(self.size) if i != self.ref_img_idx]
        summary_df["series_d"] = warp_tools.calc_total_error(np.array(med_d_list)[to_summarize_idx])
        summary_df["series_tre"] = warp_tools.calc_total_error(np.array(tre_list)[to_summarize_idx])
        summary_df["name"] = self.name

        self.summary_df = summary_df

        return summary_df


def register_images(src, non_rigid_reg_class=non_rigid_registrars.OpticalFlowWarper,
                    non_rigid_reg_params=None, dst_dir=None,
                    ref_img_name=None, moving_to_fixed_xy=None,
                    mask=None, name=None, qt_emitter=None):
    """
    Parameters
    ----------
    src : SerialRigidRegistrar, str
        Either a SerialRigidRegistrar object that was used to optimally
        align a series of images, or a string indicating where the images
        to be aligned are located. If src is a string, the images should be
        named such that they are read in the correct order, i.e. each
        starting with a number.

    non_rigid_reg_class : NonRigidRegistrar
        Uninstantiated NonRigidRegistrar class that will be used to
        calculate the deformation fields between images.
        By default this is an OpticalFlowWarper that uses the OpenCV
        implementation of DeepFlow.

    non_rigid_reg_params: dictionary, optional
        Dictionary containing parameters {name: value} to be used to initialize
        the NonRigidRegistrar.
        In the case where simple ITK is used by the, params should be
        a SimpleITK.ParameterMap. Note that numeric values nedd to be
        converted to strings.

    dst_dir : str, optional
        Top directory where aliged images should be save. SerialNonRigidRegistrar will
        be in this folder, and aligned images in the "registered_images"
        sub-directory. If None, the images will not be written to file

    ref_img_name : str, optional
        Name of image that will be treated as the center of the stack.
        If None, the index of the middle image will be returned.

    moving_to_fixed_xy :  dict of list, or bool
        If `moving_to_fixed_xy` is a dict of list, then
        Key = image name, value = list of matched keypoints between
        each moving image and the fixed image.
        Each element in the list contains 2 arrays:

        #. Rigid registered xy in moving/current/from image
        #. Rigid registered xy in fixed/next/to image

        To deterime which pairs of images will be aligned, use
        `get_alignment_indices`. Can use `get_imgs_from_dir`
        to see the order inwhich the images will be read, which will correspond
        to the indices retuned by `get_alignment_indices`.

        If `src` is a SerialRigidRegistrar and `moving_to_fixed_xy` is
        True, then the matching features in the SerialRigidRegistrar will
        be used. If False, then matching features will not be used.

    mask :  ndarray, bool, optional
        Mask used in non-rigid alignments.

        If an ndarray, it must have the same size as the other images.

        If True, then the `overlap_mask` in the SerialRigidRegistrar
        will be used.

        If False or None, no mask will be used.

    name : optional
        Optional name for this SerialNonRigidRegistrar

    qt_emitter : PySide2.QtCore.Signal, optional
        Used to emit signals that update the GUI's progress bars

    Returns
    -------
    nr_reg : SerialNonRigidRegistrar
        SerialNonRigidRegistrar that has registeredt the images in `src`
    """

    tic = time()
    nr_reg = SerialNonRigidRegistrar(src=src, ref_img_name=ref_img_name,
                                     moving_to_fixed_xy=moving_to_fixed_xy,
                                     mask=mask, name=name)

    nr_reg.register(non_rigid_reg_class, non_rigid_reg_params)

    if dst_dir is not None:
        registered_img_dir = os.path.join(dst_dir, "non_rigid_registered_images")
        registered_data_dir = os.path.join(dst_dir, "data")
        registered_grids_dir = os.path.join(dst_dir, "deformation_grids")
        for d in [registered_img_dir, registered_data_dir, registered_grids_dir]:
            pathlib.Path(d).mkdir(exist_ok=True, parents=True)

        print("\n======== Saving results\n")
        if moving_to_fixed_xy is not None:
            summary_df = nr_reg.summarize()
            summary_file = os.path.join(registered_data_dir, name + "_results.csv")
            summary_df.to_csv(summary_file, index=False)

        pickle_file = os.path.join(registered_data_dir, name + "_non_rigid_registrar.pickle")
        pickle.dump(nr_reg, open(pickle_file, 'wb'))

        for img_obj in nr_reg.non_rigid_obj_list:
            f_out = f"{img_obj.name}.png"

            io.imsave(os.path.join(registered_img_dir, f_out),
                      img_obj.registered_img.astype(np.uint8))

            colord_tri_grid = viz.color_displacement_tri_grid(img_obj.bk_dxdy[0],
                                                              img_obj.bk_dxdy[1])

            io.imsave(os.path.join(registered_grids_dir, f_out), colord_tri_grid)

    toc = time()
    elapsed = toc - tic
    time_string, time_units = valtils.get_elapsed_time_string(elapsed)
    print(f"\n======== Non-rigid registration complete in {time_string} {time_units}\n")

    return nr_reg
