"""Classes and functions to perform serial non-rigid registration of a set of images

"""

import numpy as np
from skimage import io
from tqdm import tqdm
import os
from time import time
import pathlib
import pandas as pd
import pickle
import pyvips
import inspect

from . import warp_tools
from . import non_rigid_registrars
from . import valtils
from . import serial_rigid
from . import viz
from . import preprocessing
from . import slide_tools

IMG_LIST_KEY = "img_list"
IMG_F_LIST_KEY = "img_f_list"
IMG_NAME_KEY = "name_list"
MASK_LIST_KEY = "mask_list"

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
    ref_img_idx = warp_tools.get_ref_img_idx(img_f_list, ref_img_name)
    n_imgs = len(img_f_list)

    from_to_indices = warp_tools.get_alignment_indices(n_imgs, ref_img_idx)
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

    mask_list : list of ndarray
        List of masks used for registration
    """

    img_f_list = [f for f in os.listdir(src_dir) if
                  slide_tools.get_img_type(os.path.join(src_dir, f)) is not None]

    valtils.sort_nicely(img_f_list)

    img_list = [io.imread(os.path.join(src_dir, f)) for f in img_f_list]

    img_names = [valtils.get_name(f) for f in img_f_list]

    mask_list = [None] * len(img_f_list)

    return img_list, img_f_list, img_names, mask_list


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

    mask_list : list of ndarray
        List of masks used for registration

    """
    img_list = [None] * serial_rigid_reg.size
    img_names = [None] * serial_rigid_reg.size
    img_f_list = [None] * serial_rigid_reg.size
    mask_list = [None] * serial_rigid_reg.size

    for i, img_obj in enumerate(serial_rigid_reg.img_obj_list):
        img_list[i] = img_obj.registered_img
        img_names[i] = img_obj.name
        img_f_list[i] = img_obj.full_img_f

        # Moving mask
        temp_mask = np.full_like(img_obj.image, 255)
        img_mask = warp_tools.warp_img(temp_mask, M=img_obj.M,
                                       out_shape_rc=img_obj.registered_img.shape,
                                       interp_method="nearest")
        mask_list[i] = img_mask

    return img_list, img_f_list, img_names, mask_list


def get_imgs_from_dict(img_dict):
    """Get images from source directory.

    Parameters
    ----------
    img_dict : dictionary
        Dictionary containing the following key : value pairs

        "img_list" : list of images to register
        "img_f_list" : list of filenames of each image
        "name_list" : list of image names. If not provided, will come from file names
        "mask_list" list of masks for each image

    All of the above are optional, except `img_list`.

    Returns
    -------
    img_list : list of ndarray
        List of images to be registered

    img_f_list : list of str
        List of image file names

    img_names : list of str
        List of names for each image. Created by removing the extension

    mask_list : list of ndarray
        List of masks used for registration

    """
    img_list = img_dict[IMG_LIST_KEY]

    names_provided = IMG_NAME_KEY in img_dict.keys()
    files_provided = IMG_F_LIST_KEY in img_dict.keys()
    masks_provided = MASK_LIST_KEY in img_dict.keys()

    n_imgs = len(img_list)
    if files_provided:
        img_f_list = img_dict[IMG_F_LIST_KEY]
    else:
        img_f_list = [None] * n_imgs

    if names_provided:
        img_names = img_dict[IMG_NAME_KEY]
    else:
        if files_provided:
            img_names = [valtils.get_name(f) for f in img_f_list]
        else:
            img_names = [None] * n_imgs

    if masks_provided:
        mask_list = img_dict[MASK_LIST_KEY]
    else:
        mask_list = [None] * n_imgs

    return img_list, img_f_list, img_names, mask_list


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

    def __init__(self, reg_obj, image, name, stack_idx, moving_xy=None, fixed_xy=None, mask=None):
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

        mask : ndarray, optional
            Mask covering area to be registered.

        """
        self.reg_obj = reg_obj
        self.image = image
        self.name = name
        self.stack_idx = stack_idx
        self.moving_xy = moving_xy
        self.fixed_xy = fixed_xy
        self.registered_img = None
        self.warped_grid = None
        self.bk_dxdy = None
        self.fwd_dxdy = None

        self.is_vips = isinstance(image, pyvips.Image)
        self.shape = self.get_shape(image)

        mask_shape = self.get_shape(mask)
        if self.is_vips and not self.check_if_vips(mask):
            mask = warp_tools.numpy2vips(mask)

        if np.all(mask_shape == self.shape):
            mask = warp_tools.resize_img(mask, self.shape)

        self.mask = mask

    def get_shape(self, img):
        if isinstance(img, pyvips.Image):
            shape = np.array([img.height, img.width])
        else:
            shape = img.shape[0:2]

        return shape

    def check_if_vips(self, img):
        return isinstance(img, pyvips.Image)

    def mask_img(self, img, mask):

        if isinstance(img, pyvips.Image):
            if isinstance(mask, np.ndarray):
                vips_mask = warp_tools.numpy2vips(mask)
            else:
                vips_mask = mask
            masked_img = (vips_mask == 0).ifthenelse(0, img)
        else:
            masked_img = img.copy()
            masked_img[mask == 0] = 0

        return masked_img

    def mask_dxdy(self, dxdy, mask):
        if isinstance(dxdy, pyvips.Image):
            masked_dxdy = self.mask_img(dxdy, mask)
        else:
            masked_dxdy = [self.mask_img(dxdy[0], mask), self.mask_img(dxdy[1], mask)]

        return masked_dxdy

    def split_params(self, params, non_rigid_reg_class):
        if params is not None:
            init_arg_list = inspect.getfullargspec(non_rigid_reg_class.__init__).args
            reg_arg_list = inspect.getfullargspec(non_rigid_reg_class.register).args

            init_kwargs = {k:v for k, v in params.items() if k in init_arg_list}
            reg_kwargs = {k:v for k, v in params.items() if k in reg_arg_list}

        else:
            init_kwargs = {}
            reg_kwargs = {}

        return init_kwargs, reg_kwargs

    def calc_deformation(self, registered_fixed_image, non_rigid_reg_class,
                         bk_dxdy=None, params=None, mask=None):
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

        bk_dxdy : ndarray, optional
            (2, P, Q) numpy array of pixel displacements in
            the x and y directions. dx = dxdy[0], and dy=dxdy[1].
            Used to warp the registered_img before finding deformation fields.

        params : dictionary, optional
            Keyword: value dictionary of parameters to be used in reigstration.
            Passed to the non_rigid_reg_class' init() method.

            In the case where simple ITK will be used, params should be
            a SimpleITK.ParameterMap. Note that numeric values needd to be
            converted to strings.

        mask : ndarray, optional
            2D array with shape (P,Q) where non-zero pixel values are foreground,
            and 0 is background, which is ignnored during registration. If None,
            then all non-zero pixels in images will be used to create the mask.

        """

        if self.reg_obj.from_rigid_reg:
            rigid_img_obj = self.reg_obj.src.img_obj_dict[self.name]
            M = rigid_img_obj.M
            unwarped_shape = rigid_img_obj.image.shape[0:2]
            og_reg_shape_rc = rigid_img_obj.registered_shape_rc

        if mask is not None:
            if isinstance(mask, pyvips.Image):
                reg_mask = warp_tools.vips2numpy(mask)
            else:
                reg_mask = mask.copy()
        else:
            reg_mask = None

        if bk_dxdy is not None:
            if isinstance(bk_dxdy, list):
                bk_dxdy = np.array(bk_dxdy)

            if reg_mask is not None:
                for_reg_dxdy = self.mask_dxdy(bk_dxdy, reg_mask)
            else:
                for_reg_dxdy = bk_dxdy

            if self.reg_obj.from_rigid_reg:
                for_reg_dxdy = warp_tools.remove_invasive_displacements(for_reg_dxdy,
                                                                        M=M,
                                                                        src_shape_rc=unwarped_shape,
                                                                        out_shape_rc=og_reg_shape_rc
                                                                        )

            moving_img = warp_tools.warp_img(self.image, bk_dxdy=for_reg_dxdy)
            if reg_mask is not None:
                # Update mask too
                reg_mask = warp_tools.warp_img(reg_mask, bk_dxdy=for_reg_dxdy)

        else:
            moving_img = self.image.copy()
            for_reg_dxdy = None
            if self.is_vips:
                bk_dxdy = pyvips.Image.black(self.shape[1], self.shape[0], bands=2)
            else:
                bk_dxdy = np.array([np.zeros(self.shape[0:2]), np.zeros(self.shape[0:2])])

        init_kwargs, reg_kwargs = self.split_params(params, non_rigid_reg_class)

        non_rigid_reg = non_rigid_reg_class(params=init_kwargs)

        if self.moving_xy is not None and self.fixed_xy is not None and \
           issubclass(non_rigid_reg_class, non_rigid_registrars.NonRigidRegistrarXY):
            if for_reg_dxdy is not None:
                # Update positions #
                fwd_dxdy = warp_tools.get_inverse_field(for_reg_dxdy)
                fixed_xy = warp_tools.warp_xy(self.fixed_xy, M=None, fwd_dxdy=fwd_dxdy)
                moving_xy = warp_tools.warp_xy(self.moving_xy, M=None, fwd_dxdy=fwd_dxdy)
            else:
                fixed_xy = self.fixed_xy
                moving_xy = self.moving_xy
        else:
            fixed_xy = None
            moving_xy = None

        xy_args = {"moving_xy": moving_xy, "fixed_xy": fixed_xy}
        reg_kwargs.update(xy_args)

        warped_moving, moving_grid_img, moving_bk_dxdy = \
            non_rigid_reg.register(moving_img=moving_img,
                                   fixed_img=registered_fixed_image,
                                   mask=reg_mask,
                                   **reg_kwargs)

        if self.reg_obj.from_rigid_reg:
            moving_bk_dxdy = warp_tools.remove_invasive_displacements(moving_bk_dxdy,
                                                                      M=M,
                                                                      src_shape_rc=unwarped_shape,
                                                                      out_shape_rc=og_reg_shape_rc
                                                                      )

        if not self.check_if_vips(moving_bk_dxdy):
            if reg_mask is not None:
                # Only add new transformations
                moving_bk_dxdy = self.mask_dxdy(moving_bk_dxdy, reg_mask)
            bk_dxdy_from_ref = np.array([bk_dxdy[0] + moving_bk_dxdy[0],
                                         bk_dxdy[1] + moving_bk_dxdy[1]])
        else:
            if reg_mask is not None:
                moving_bk_dxdy = self.mask_dxdy(moving_bk_dxdy, reg_mask)
                bk_dxdy_from_ref = bk_dxdy + moving_bk_dxdy

        img_bk_dxdy = bk_dxdy_from_ref.copy()
        if reg_mask is not None:
            img_bk_dxdy = self.mask_dxdy(img_bk_dxdy, reg_mask)

        if self.reg_obj.from_rigid_reg:
            img_bk_dxdy = warp_tools.remove_invasive_displacements(img_bk_dxdy,
                                                                   M=M,
                                                                   src_shape_rc=unwarped_shape,
                                                                   out_shape_rc=og_reg_shape_rc
                                                                   )
        self.bk_dxdy = img_bk_dxdy
        if hasattr(non_rigid_reg, "fwd_dxdy"):
            # Already calculated
            self.fwd_dxdy = non_rigid_reg.fwd_dxdy
        else:
            self.fwd_dxdy = warp_tools.get_inverse_field(self.bk_dxdy)

        if not self.is_vips:
            # If dxdy is a pyvips.Image, it's likely the displacement is too large to draw
            self.warped_grid = viz.color_displacement_grid(*self.bk_dxdy)

        self.registered_img = warp_tools.warp_img(self.image,
                                                  bk_dxdy=self.bk_dxdy,
                                                  out_shape_rc=self.shape)

        return bk_dxdy_from_ref


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

    from_rigid_reg : bool
        Whether or not the images are from a SerialRigidRegistrar

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

    def __init__(self, src, reference_img_f=None, moving_to_fixed_xy=None,
                 mask=None, name=None, align_to_reference=False, compose_transforms=True):
        """
        Parameters
        ----------
        src : SerialRigidRegistrar, str, dict

            A SerialRigidRegistrar object that was used to optimally
            align a series of images.

            If a string, it should indicating where the images
            to be aligned are located. If src is a string, the images should be
            named such that they are read in the correct order, i.e. each
            starting with a number.

            If a dictionary, it should contain the following key, value pairs:

            "img_list" : list of images to register
            "img_f_list" : list of filenames of each image
            "name_list" : list of image names. If not provided, will come from file names
            "mask_list" list of masks for each image


        reference_img_f : str, optional
            Filename of image that will be treated as the center of the stack.
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
            Mask used for all non-rigid alignments.

            If an ndarray, it must have the same size as the other images.

            If True, then the `overlap_mask` in the SerialRigidRegistrar
            will be used.

            If False or None, no mask will be used.

        name : optional
            Optional name for this SerialNonRigidRegistrar

        align_to_reference : bool, optional
            Whether or not images should be aligned to a reference image
            specified by `reference_img_f`.

        img_params : dict, optional
            Dictionary of parameters to be used for each particular image.
            Useful if images to be registered haven't been processed.
            Will be passed to `non_rigid_reg_class` init and register functions.
            key = file name, value= dictionary of keyword arguments and values

        """

        self.src = src
        if isinstance(src, serial_rigid.SerialRigidRegistrar):
            self.from_rigid_reg = True
        elif isinstance(src, str):
            self.from_rigid_reg = False
        elif isinstance(src, dict):
            self.from_rigid_reg = False
        else:
            valtils.print_warning(f"src must be either a SerialRigidRegistrar, string, or dictionary")
            return None

        self.name = name
        self.size = 0
        self.shape = None
        self.non_rigid_obj_dict = {}
        self.non_rigid_obj_list = None
        self.non_rigid_reg_params = None
        self.summary = None
        self.mask = mask

        self.reference_img_f = None
        self.ref_img_name = None
        self.ref_img_idx = None
        self.compose_transforms = compose_transforms

        self.align_to_reference = align_to_reference
        self.generate_non_rigid_obj_list(reference_img_f, moving_to_fixed_xy)

        if self.align_to_reference is False and reference_img_f is not None:
            og_ref_name = valtils.get_name(reference_img_f)
            msg = (f"The reference was specified as {og_ref_name} ",
                   f"but `align_to_reference` is `False`, and so images will be aligned serially. ",
                   f"If you would like all images to be directly aligned to {og_ref_name}, "
                   f"then set `align_to_reference` to `True`")
            valtils.print_warning(msg)


    def get_shape(self, img):
        if isinstance(img, pyvips.Image):
            shape = np.array([img.height, img.width])
        else:
            shape = img.shape[0:2]

        return shape

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
            if isinstance(mask, bool) and self.from_rigid_reg:
                mask = self.src.overlap_mask
            mask = np.clip(mask.astype(int)*255, 0, 255).astype(np.uint8)

        else:
            mask = self.create_mask()

        mask_bbox_xywh = warp_tools.xy2bbox(warp_tools.mask2xy(mask))
        self.mask = mask
        self.mask_bbox_xywh = mask_bbox_xywh

    def generate_non_rigid_obj_list(self, reference_img_f=None, moving_to_fixed_xy=None):
        """Create non_rigid_obj_list

        """

        if self.from_rigid_reg:
            img_list, img_f_list, img_names, mask_list = \
                get_imgs_rigid_reg(self.src)
        else:
            if isinstance(self.src, str):
                img_list, img_f_list, img_names, mask_list = \
                    get_imgs_from_dir(self.src)
                # overwrite `src` because all info now in NonRigidZImages
                self.src = "dictionary"

            elif isinstance(self.src, dict):
                img_list, img_f_list, img_names, mask_list = \
                    get_imgs_from_dict(self.src)

        self.size = len(img_list)
        self.shape = self.get_shape(img_list[0])

        if reference_img_f is not None:
            reference_name = valtils.get_name(reference_img_f)
        else:
            reference_name = None

        ref_img_idx = warp_tools.get_ref_img_idx(img_f_list, reference_name)

        if reference_img_f is None:
            reference_img_f = img_f_list[ref_img_idx]

        self.reference_img_f = reference_img_f
        self.ref_img_idx = ref_img_idx
        self.ref_img_name = reference_name

        if self.from_rigid_reg and isinstance(moving_to_fixed_xy, bool):
            if moving_to_fixed_xy:
                moving_to_fixed_xy = \
                    get_matching_xy_from_rigid_registrar(self.src, reference_name)
            else:
                moving_to_fixed_xy = None

        self.non_rigid_obj_list = [None] * self.size
        for i, img in enumerate(img_list):
            img_shape = self.get_shape(img)

            assert np.all(img_shape == self.shape), \
                valtils.print_warning("Images must all have the shape")

            img_name = img_names[i]
            mask = mask_list[i]

            moving_xy = None
            fixed_xy = None
            if moving_to_fixed_xy is not None and img_name != reference_img_f:
                if isinstance(moving_to_fixed_xy, dict):
                    xy_coords = moving_to_fixed_xy[img_name]
                    moving_xy = xy_coords[0]
                    fixed_xy = xy_coords[1]
                else:
                    msg = "moving_to_fixed_xy is not a dictionary. Will be ignored"
                    valtils.print_warning(msg)

            nr_obj = NonRigidZImage(self, img, img_name, stack_idx=i,
                                    moving_xy=moving_xy,
                                    fixed_xy=fixed_xy,
                                    mask=mask)

            if i == ref_img_idx:
                # Set reference image attributes #
                zero_displacement = np.zeros(self.shape)
                if not nr_obj.is_vips:
                    nr_obj.bk_dxdy = [zero_displacement, zero_displacement]
                    nr_obj.fwd_dxdy = [zero_displacement, zero_displacement]
                    nr_obj.warped_grid = viz.color_displacement_grid(*nr_obj.bk_dxdy)
                else:
                    nr_obj.bk_dxdy = pyvips.Image.black(nr_obj.shape[1], nr_obj.shape[0], bands=2)
                    nr_obj.fwd_dxdy = pyvips.Image.black(nr_obj.shape[1], nr_obj.shape[0], bands=2)

                nr_obj.registered_img = img.copy()

            self.non_rigid_obj_list[i] = nr_obj

    def update_img_params(self, non_rigid_reg_params=None, img_params=None, moving_name=None, fixed_name=None, is_tiler=False):
        """
        Update img params for non-rigid-registration
        """

        if img_params is not None and moving_name is not None:
            if len(img_params) == 0:
                indv_img_params = None
            else:
                indv_img_params = img_params[moving_name]

        else:
            indv_img_params = img_params

        if is_tiler:
            #Tiler needs processor arguments for moving and fixed images
            assert moving_name in img_params and fixed_name in img_params, "Tiled registration requires image processors for each image"

            moving_dict = img_params[moving_name]
            indv_img_params[non_rigid_registrars.NR_TILE_MOVING_P_KEY] = moving_dict[non_rigid_registrars.NR_PROCESSING_CLASS_KEY]
            indv_img_params[non_rigid_registrars.NR_TILE_MOVING_P_INIT_KW_KEY] = moving_dict[non_rigid_registrars.NR_PROCESSING_INIT_KW_KEY]
            indv_img_params[non_rigid_registrars.NR_TILE_MOVING_P_KW_KEY] = moving_dict[non_rigid_registrars.NR_PROCESSING_KW_KEY]

            fixed_dict = img_params[fixed_name]
            indv_img_params[non_rigid_registrars.NR_TILE_FIXED_P_KEY] = fixed_dict[non_rigid_registrars.NR_PROCESSING_CLASS_KEY]
            indv_img_params[non_rigid_registrars.NR_TILE_FIXED_P_INIT_KW_KEY] = fixed_dict[non_rigid_registrars.NR_PROCESSING_INIT_KW_KEY]
            indv_img_params[non_rigid_registrars.NR_TILE_FIXED_P_KW_KEY] = fixed_dict[non_rigid_registrars.NR_PROCESSING_KW_KEY]

        if non_rigid_reg_params is not None and indv_img_params is not None:

            updated_params = indv_img_params.copy()
            updated_params[non_rigid_registrars.NR_PARAMS_KEY] = non_rigid_reg_params

        elif non_rigid_reg_params is not None and indv_img_params is None:
            updated_params = non_rigid_reg_params

        elif non_rigid_reg_params is None and indv_img_params is not None:
            updated_params = indv_img_params

        else:
            updated_params = None

        return updated_params


    def register_serial(self, non_rigid_reg_class, non_rigid_reg_params=None, img_params=None):
        """Non-rigidly align images in serial
        Parameters
        ----------
        non_rigid_reg_class : NonRigidRegistrar
            Uninstantiated NonRigidRegistrar class that will be used to
            calculate the deformation fields between images

        non_rigid_reg_params: dictionary, optional
            Dictionary containing parameters {name: value} to be used to initialize
            `non_rigid_reg_class`.
            In the case where simple ITK is used by the, params should be
            a SimpleITK.ParameterMap. Note that numeric values nedd to be
            converted to strings.

        """
        current_dxdy = None
        self.non_rigid_reg_params = non_rigid_reg_params
        iter_order = warp_tools.get_alignment_indices(self.size, self.ref_img_idx)

        is_tiler = non_rigid_reg_class.__name__ == non_rigid_registrars.NonRigidTileRegistrar.__name__
        for moving_idx, fixed_idx in tqdm(iter_order, desc="Finding non-rigid transforms", unit="image"):
            moving_obj = self.non_rigid_obj_list[moving_idx]
            fixed_obj = self.non_rigid_obj_list[fixed_idx]

            if self.compose_transforms:
                if fixed_obj.stack_idx == self.ref_img_idx:
                    current_dxdy = None
                else:
                    current_dxdy = updated_dxdy

            if moving_obj.mask is not None:
                if self.mask is not None:
                    reg_mask = preprocessing.combine_masks(self.mask, moving_obj.mask, op="and")
                else:
                    reg_mask = moving_obj.mask

            elif self.mask is not None:
                reg_mask = self.mask
            else:
                reg_mask is None

            nr_reg_params = self.update_img_params(non_rigid_reg_params, img_params, moving_name=moving_obj.name, fixed_name=fixed_obj.name, is_tiler=is_tiler)
            updated_dxdy = moving_obj.calc_deformation(registered_fixed_image=fixed_obj.registered_img,
                                        non_rigid_reg_class=non_rigid_reg_class,
                                        bk_dxdy=current_dxdy,
                                        params=nr_reg_params,
                                        mask=reg_mask
                                        )


    def register_to_ref(self, non_rigid_reg_class, non_rigid_reg_params=None, img_params=None):
        """Non-rigidly align images to a reference image
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
        ref_nr_obj = self.non_rigid_obj_list[self.ref_img_idx]
        ref_img = ref_nr_obj.image
        is_tiler = non_rigid_reg_class.__name__ == non_rigid_registrars.NonRigidTileRegistrar.__name__
        for moving_idx in tqdm(range(self.size), desc="Finding non-rigid transforms", unit="image"):
            moving_obj = self.non_rigid_obj_list[moving_idx]
            if moving_obj.stack_idx == self.ref_img_idx:
                continue

            overlap_mask = None

            nr_reg_params = self.update_img_params(non_rigid_reg_params, img_params, moving_name=moving_obj.name, fixed_name=ref_nr_obj.name, is_tiler=is_tiler)

            moving_obj.calc_deformation(ref_img,
                                        non_rigid_reg_class,
                                        params=nr_reg_params,
                                        mask=overlap_mask)

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

        for i, nr_img_obj in tqdm(enumerate(self.non_rigid_obj_list), desc="Aligning images", unit="annotation"):
            nr_img_obj.registered_img = warped_imgs[i]
            nr_img_obj.bk_dxdy = backward_deformations[i]
            nr_img_obj.warped_grid = viz.color_displacement_grid(*nr_img_obj.bk_dxdy)
            nr_img_obj.fwd_dxdy = warp_tools.get_inverse_field(nr_img_obj.bk_dxdy)

    def register(self, non_rigid_reg_class, non_rigid_reg_params, img_params=None):
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
        img_params : dict, optional
            Dictionary of parameters to be used for each particular image.
            Useful if images to be registered haven't been processed.
            Will be passed to `non_rigid_reg_class` init and register functions.
            key = file name, value= dictionary of keyword arguments and values

        """

        if img_params is not None:
            named_img_params = {valtils.get_name(k):v for k, v in img_params.items()}
        else:
            named_img_params = None
        if issubclass(non_rigid_reg_class, non_rigid_registrars.NonRigidRegistrarGroupwise):
            self.register_groupwise(non_rigid_reg_class, non_rigid_reg_params)
        elif self.align_to_reference:
            self.register_to_ref(non_rigid_reg_class, non_rigid_reg_params, img_params=named_img_params)
        else:
            self.register_serial(non_rigid_reg_class, non_rigid_reg_params, img_params=named_img_params)

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

        iter_order = warp_tools.get_alignment_indices(self.size, self.ref_img_idx)
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
                    reference_img_f=None, moving_to_fixed_xy=None,
                    mask=None, name=None, align_to_reference=False,
                    img_params=None, compose_transforms=True, qt_emitter=None):
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

    reference_img_f : str, optional
        Filename of image that will be treated as the center of the stack.
        If None, the index of the middle image will be returned.

    moving_to_fixed_xy :  dict of list, or bool
        If `moving_to_fixed_xy` is a dict of list, then
        Key = image name, value = list of matched keypoints between
        each moving image and the fixed image.
        Each element in the list contains 2 arrays:

        #. Rigid registered xy in moving/current/from image
        #. Rigid registered xy in fixed/next/to image

        To deterime which pairs of images will be aligned, use
        `warp_tools.get_alignment_indices`. Can use `get_imgs_from_dir`
        to see the order inwhich the images will be read, which will correspond
        to the indices retuned by `warp_tools.get_alignment_indices`.

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

    align_to_reference : bool, optional
        Whether or not images should be aligne to a reference image
        specified by `reference_img_f`. Will be set to True if
        `reference_img_f` is provided.

    img_params : dict, optional
        Dictionary of parameters to be used for each particular image.
        Useful if images to be registered haven't been processed.
        Will be passed to `non_rigid_reg_class` init and register functions.
        key = file name, value= dictionary of keyword arguments and values

    qt_emitter : PySide2.QtCore.Signal, optional
        Used to emit signals that update the GUI's progress bars

    Returns
    -------
    nr_reg : SerialNonRigidRegistrar
        SerialNonRigidRegistrar that has registeredt the images in `src`
    """

    tic = time()
    nr_reg = SerialNonRigidRegistrar(src=src, reference_img_f=reference_img_f,
                                     moving_to_fixed_xy=moving_to_fixed_xy,
                                     mask=mask, name=name,
                                     align_to_reference=align_to_reference,
                                     compose_transforms=compose_transforms)

    nr_reg.register(non_rigid_reg_class, non_rigid_reg_params, img_params=img_params)

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
