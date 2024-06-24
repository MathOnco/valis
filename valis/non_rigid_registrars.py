"""Perform non-rigid registration
"""

import os
import pathlib
import cv2
import numpy as np
import SimpleITK as sitk
from skimage import transform, color, filters, exposure, util
from skimage import color as skcolor
import pyvips
from copy import deepcopy
import multiprocessing
from pqdm.threads import pqdm

from . import viz
from . import warp_tools
from . import preprocessing
from . import valtils

NR_CLS_KEY = "non_rigid_registrar_cls"
NR_PROCESSING_KW_KEY = "processer_kwargs"
NR_PROCESSING_INIT_KW_KEY = "init_processer_kwargs"
NR_PROCESSING_CLASS_KEY = "processer_cls"
NR_STATS_KEY = "target_stats"
NR_TILE_WH_KEY = "tile_wh"
NR_PARAMS_KEY = "params"

NR_MOVING = "moving"
NR_TILE_MOVING_P_KEY = f"{NR_MOVING}_{NR_PROCESSING_CLASS_KEY}"
NR_TILE_MOVING_P_INIT_KW_KEY = f"{NR_MOVING}_{NR_PROCESSING_INIT_KW_KEY}"
NR_TILE_MOVING_P_KW_KEY = f"{NR_MOVING}_{NR_PROCESSING_KW_KEY}"

NR_FIXED = "fixed"
NR_TILE_FIXED_P_KEY = f"{NR_FIXED}_{NR_PROCESSING_CLASS_KEY}"
NR_TILE_FIXED_P_INIT_KW_KEY = f"{NR_FIXED}_{NR_PROCESSING_INIT_KW_KEY}"
NR_TILE_FIXED_P_KW_KEY = f"{NR_FIXED}_{NR_PROCESSING_KW_KEY}"


# Abstract Classes #
class NonRigidRegistrar(object):
    """Abstract class for non-rigid registration using displacement fields

    Warps moving_img to align with fixed_img using backwards transformations
    VALIS offers 3 implementations: dense optical flow (OpenCV),
    SimpleElastix, and  groupwise SimpleElastix.
    Displacement fields can come from other packages, indluding
    SimpleITK, PIRT, DIPY, etc... Those other methods can be used by
    subclassing the NonRigidRegistrar classes in VALIS.

    Attributes
    ----------
    moving_img : ndarray
        Image with shape (N,M) thata is  warp to align with `fixed_img`.

    fixed_img : ndarray
        Image with shape (N,M) that `moving_img` is warped to align with.

    mask : ndarray
        2D array with shape (N,M) where non-zero pixel values are foreground,
        and 0 is background, which is ignnored during registration. If None,
        then all non-zero pixels in images will be used to create the mask.

    shape : tuple
        Number of rows and columns in each image. Will be (N,M).

    grid_spacing : int
        Number of pixels between deformation grid points.

    warped_image : ndarray
        Registered copy of `moving_img`.

    deformation_field_img : ndarray
        Image showing deformation applied to a regular grid.

    backward_dx : ndarray
        (N,M) array defining the displacements in the x-dimension.

    backward_dy : ndarray
        (N,M) array defining the displacements in the y-dimension.

    method : str
        Name of registration method.

    Note
    -----
    All NonRigidRegistrar subclasses need to have a calc() method,
    which must at least take the following arguments:
    moving_img, fixed_img, mask. calc() should return the displacement field
    as a (2, N, M) numpy array, with the first element being an array of
    displacements in the x-dimension, and the second element being an array of
    displacements in the y-dimension.

    Note that the NonRigidRegistrarXY subclass should be used if
    corresponding points in moving and fixed images can be used
    to aid the registration.

    """

    def __init__(self, params=None):
        """
        Parameters
        ----------
        params : dictionary
            Keyword: value dictionary of parameters to be used in reigstration.
            Will get used in the calc() method.

            In the case where simple ITK will be used, params should be
            a SimpleITK.ParameterMap. Note that numeric values needd to be
            converted to strings.

        """

        self.params = params
        self.moving_img = None
        self.fixed_img = None
        self.mask = None
        self.shape = None
        self.grid_spacing = None
        self.method = None
        self.warped_image = None
        self.deformation_field_img = None
        self.backward_dx = None
        self.backward_dy = None

        if isinstance(params, dict):
            if len(params) > 0:
                self._params_provided = True
            else:
                # Empty kwargs dictionary
                self._params_provided = False
        else:
            self._params_provided = False

    def apply_mask(self, mask):

        masked_moving = warp_tools.apply_mask(self.moving_img, mask)
        masked_fixed = warp_tools.apply_mask(self.fixed_img, mask)

        return masked_moving, masked_fixed

    def calc(self, moving_img, fixed_img, mask, *args, **kwargs):
        """Cacluate displacement fields

        Can record subclass specific atrributes here too

        Parameters
        ----------
        moving_img : ndarray
            Image to warp to align with `fixed_img`. Has shape (N, M).

        fixed_img : ndarray
            Image `moving_img` is warped to align with. Has shape (N, M).

        mask : ndarray
            2D array with shape (N,M) where non-zero pixel values are foreground,
            and 0 is background, which is ignnored during registration. If None,
            then all non-zero pixels in images will be used to create the mask.

        Returns
        -------
        bk_dxdy : ndarray
            (2, N, M) numpy array of pixel displacements in
            the x and y directions. dx = bk_dxdy[0], and dy=bk_dxdy[1].

        """

        bk_dxdy = None

        return bk_dxdy

    def create_mask(self):
        temp_mask = np.zeros(self.shape, dtype=np.uint8)
        img_list = [self.moving_img, self.fixed_img]
        for img in img_list:
            temp_mask[img > 0] = 255

        mask = warp_tools.bbox2mask(*warp_tools.xy2bbox(
                                    warp_tools.mask2xy(temp_mask)),
                                    temp_mask.shape)

        return mask

    def register(self, moving_img, fixed_img, mask=None, **kwargs):
        """
        Register images, warping moving_img to align with fixed_img

        Uses backwards transforms to register images (i.e. aligning
        fixed to moving), so the inverse transform needs to be used
        to warp points from moving_img. This is automatically done in
        warp_tools.warp_xy

        Parameters
        ----------
        moving_img : ndarray
            Image to warp to align with `fixed_img`.

        fixed_img : ndarray
            Image `moving_img` is warped to align with.

        mask : ndarray
            2D array with shape (N,M) where non-zero pixel values are foreground,
            and 0 is background, which is ignnored during registration. If None,
            then all non-zero pixels in images will be used to create the mask.

        **kwargs : dict, optional
            Additional keyword arguments passed to NonRigidRegistrar.calc

        Returns
        -------
        warped_img : ndarray
            Moving image registered to align with fixed image.

        warped_grid : ndarray
            Image showing deformation applied to a regular grid.

        bk_dxdy : ndarray
            (2, N, M) numpy array of pixel displacements in
            the x and y directions.

        """

        moving_shape = warp_tools.get_shape(moving_img)[0:2]
        fixed_shape = warp_tools.get_shape(fixed_img)[0:2]

        assert np.all(moving_shape == fixed_shape), \
            print("Images have different shapes")

        self.shape = moving_shape
        self.moving_img = moving_img
        self.fixed_img = fixed_img

        if mask is None:
            mask = np.full(self.shape, 255, dtype=np.uint8)

        self.mask = mask

        if self.mask is not None:
            # Only do registration inside mask #
            _, masked_fixed = self.apply_mask(self.mask)
            masked_moving = self.moving_img.copy()

            mask_bbox = warp_tools.xy2bbox(warp_tools.mask2xy(self.mask))
            min_c, min_r = mask_bbox[0:2]
            max_c, max_r = mask_bbox[0:2] + mask_bbox[2:]
            mask = self.mask[min_r:max_r, min_c:max_c]
            masked_moving = masked_moving[min_r:max_r, min_c:max_c]
            masked_fixed = masked_fixed[min_r:max_r, min_c:max_c]

        else:
            masked_moving = self.moving_img.copy()
            masked_fixed = self.fixed_img.copy()

        bk_dxdy = self.calc(moving_img=masked_moving,
                            fixed_img=masked_fixed,
                            mask=mask, **kwargs)

        if mask is not None:
            bk_dx = np.zeros(self.shape)
            bk_dx[min_r:max_r, min_c:max_c] = bk_dxdy[0]
            bk_dx[self.mask == 0] = 0

            bk_dy = np.zeros(self.shape)
            bk_dy[min_r:max_r, min_c:max_c] = bk_dxdy[1]
            bk_dy[self.mask == 0] = 0

            bk_dxdy = np.array([bk_dx, bk_dy])

        warped_img, warp_grid = self.get_warped_img_and_grid(bk_dxdy)

        self.backward_dx = bk_dxdy[..., 0]
        self.backward_dy = bk_dxdy[..., 1]
        self.deformation_field_img = warp_grid
        self.warped_image = warped_img

        return warped_img, warp_grid, bk_dxdy

    def get_grid_image(self, grid_spacing=None, thickness=1, grid_spacing_ratio=0.025):
        """Create an image of a regular grid.

        Usually used to visualize non-rigid deformations.

        Parameters
        ----------
        grid_spacing : int, optional
            Number of pixels between grid points.

        thickness : int, optional
            Thickness of lines in image.

        """

        if grid_spacing is None:
            if self.grid_spacing is not None:
                grid_spacing = int(self.grid_spacing)
            else:
                grid_spacing = np.max(np.array(self.shape)*grid_spacing_ratio).astype(int)

        grid_r, grid_c = viz.get_grid(self.shape[:2],
                                      grid_spacing=grid_spacing,
                                      thickness=thickness)

        grid_img = np.zeros(self.shape[:2])
        grid_img[grid_r, grid_c] = 255

        return grid_img

    def get_warped_img_and_grid(self, bk_dxdy):
        """Apply deformation to moving image and regular grid

        Parameters
        ----------
        bk_dxdy : ndarray
            (2, N, M) numpy array of pixel displacements in
            the x and y directions.

        Returns
        -------
        warped_img : ndarray
            Warped copy of moving image.

        warp_grid : ndarray
            Image showing deformation applied to regular grid.

        """

        warp_map = warp_tools.get_warp_map(dxdy=bk_dxdy)
        warped_img =transform.warp(self.moving_img, warp_map, preserve_range=True)
        self.warped_image = warped_img
        grid_img = self.get_grid_image(grid_spacing=16)
        warp_grid = transform.warp(grid_img, warp_map, preserve_range=True)

        return warped_img, warp_grid


class NonRigidRegistrarXY(NonRigidRegistrar):
    """Abstract class for non-rigid registration using displacement fields

    Subclass of NonRigidRegistrar that can (optionally) use corresponding
    points (xy coordinates) to aid in the registration

    Attributes
    ----------
    moving_img : ndarray
        Image with shape (N,M) thata is  warp to align with `fixed_img`.

    fixed_img : ndarray
        Image with shape (N,M) that `moving_img` is warped to align with.

    mask : ndarray
        2D array with shape (N,M) where non-zero pixel values are foreground,
        and 0 is background, which is ignnored during registration. If None,
        then all non-zero pixels in images will be used to create the mask.

    shape : tuple
        Number of rows and columns in each image. Will be (N,M).

    grid_spacing : int
        Number of pixels between deformation grid points/

    warped_image : ndarray
        Registered copy of `moving_img`.

    deformation_field_img : ndarray
        Image showing deformation applied to a regular grid.

    backward_dx : ndarray
        (N,M) array defining the displacements in the x-dimension.

    backward_dy : ndarray
        (N,M) array defining the displacements in the y-dimension.

    method : str
        Name of registration method.

    moving_xy : ndarray, optional
        (N, 2) array containing points in `moving_img` that correspond
        to those in the fixed image.

    fixed_xy : ndarray, optional
        (N, 2) array containing points in `fixed_img` that correspond
        to those in the moving image.

    Note
    ----
    All NonRigidRegistrarXY subclasses need to have a calc() method,
    which needs to at least take the following arguments:
    moving_img, fixed_img, mask, moving_xy, fixed_xy.
    calc() should return the warped moving image, warped regular grid,
    and the displacement field as an (2, N, M) numpy array.

    Note that NonRigidRegistrar should be used if corresponding points in
    moving and fixed images can not be used to aid the registration.

    """

    def __init__(self, params=None):
        super().__init__(params=params)
        """
        Parameters
        ----------
        params : dictionary
            Keyword: value dictionary of parameters to be used in reigstration.
            Will get used in the calc() method.

            In the case where simple ITK will be used, params should be
            a SimpleITK.ParameterMap. Note that numeric values needd to be
            converted to strings.

        moving_xy : ndarray, optional
            (N, 2) array containing points in the moving image that correspond
            to those in the fixed image.

        fixed_xy : ndarray, optional
            (N, 2) array containing points in the fixed image that correspond
            to those in the moving image.

        """

        self.moving_xy = None
        self.fixed_xy = None

    def register(self, moving_img, fixed_img, mask=None, moving_xy=None,
                 fixed_xy=None, **kwargs):
        """Register images, warping moving_img to align with fixed_img

        Uses backwards transforms to register images (i.e. aligning
        fixed to moving), so the inverse transform needs to be used
        to warp points from moving_img. This is automatically done in
        warp_tools.warp_xy

        Parameters
        ----------
        moving_img : ndarray
            Image to warp to align with `fixed_img`.

        fixed_img : ndarray
            Image `moving_img` is warped to align with.

        mask : ndarray
            2D array with shape (N,M) where non-zero pixel values are foreground,
            and 0 is background, which is ignnored during registration. If None,
            then all non-zero pixels in images will be used to create the mask.

        moving_xy : ndarray, optional
            (N, 2) array containing points in the `moving_img` that correspond
            to those in `fixed_img`.

        fixed_xy : ndarray, optional
            (N, 2) array containing points in the `fixed_img` that correspond
            to those in the `moving_img`.

        Returns
        -------
        warped_img : ndarray
            `moving_img` registered to align with `fixed_img`.

        warped_grid : ndarray
            Image showing deformation applied to a regular grid.

        bk_dxdy : ndarray
            (2, N, M) numpy array of pixel displacements in the
            x and y directions.

        """

        if moving_xy is not None:
            moving_xy, fixed_xy = self.filter_xy(moving_xy, fixed_xy,
                                                 moving_img.shape,
                                                 mask)

        self.moving_xy = moving_xy
        self.fixed_xy = fixed_xy
        warped_img, warp_grid, bk_dxdy = \
            NonRigidRegistrar.register(self, moving_img=moving_img,
                                       fixed_img=fixed_img,
                                       mask=mask,
                                       moving_xy=moving_xy,
                                       fixed_xy=fixed_xy,
                                       **kwargs)

        return warped_img, warp_grid, bk_dxdy

    def filter_xy(self, moving_xy, fixed_xy, img_shape_rc, mask=None):
        """Remove points outside image and/or mask

        """

        if mask is None:
            mask = np.full(img_shape_rc, 255, dtype=np.uint8)

        moving_inside_idx = warp_tools.get_inside_mask_idx(moving_xy, mask)
        fixed_inside_idx = warp_tools.get_inside_mask_idx(fixed_xy, mask)
        inside_idx = np.intersect1d(moving_inside_idx, fixed_inside_idx)

        return moving_xy[inside_idx, :], fixed_xy[inside_idx, :]


class NonRigidRegistrarGroupwise(NonRigidRegistrar):
    """Performs groupwise non-rigid registration

    This subclass can register a collection (>= 2) of images,
    and so is not limited to pairs of images.

    Attributes
    ----------
    img_list : list of ndarray
        List of images, each with shape (N,M) that are to be co-registered

    mask : ndarray
        2D array with shape (N,M) where non-zero pixel values are foreground,
        and 0 is background, which is ignnored during registration. If None,
        then all non-zero pixels in images will be used to create the mask.

    shape : tuple of int
        Number of rows and columns in each image. Will be (N,M).

    warped_image : ndarray
        Registered copy of `moving_img`.

    deformation_field_img : ndarray
        Image showing deformation applied to a regular grid.

    backward_dx : ndarray
        (N,M) array defining the displacements in the x-dimension.

    backward_dy : ndarray
        (N,M) array defining the displacements in the y-dimension.

    grid_spacing : int
        Number of pixels between deformation grid points

    method : str
        Name of registration method.

    size : int
        Number of images that are being registered as a group

    """
    def __init__(self, params=None):
        super().__init__(params=params)
        self.img_list = None
        self.size = 0

    def apply_mask(self, mask):
        """
        Apply mask to all images in img_list
        """
        for img in self.img_list:
            img[mask == 0] = 0

    def create_mask(self):
        temp_mask = np.zeros(self.shape, dtype=np.uint8)
        for img in self.img_list:
            temp_mask[img > 0] = 255

        mask = warp_tools.bbox2mask(*warp_tools.xy2bbox(
                                    warp_tools.mask2xy(temp_mask)),
                                    temp_mask.shape)
        return mask

    def register(self, img_list, mask=None):
        """Register images in img_list

        Uses backwards transforms to register images (i.e. aligning
        fixed to moving), so the inverse transform needs to be used
        to warp points from moving_img. This is automatically done in
        warp_tools.warp_xy

        Parameters
        ----------
        img_list : list of ndarray
            List of I images, each with shape (N,M) that are to
            be co-registered.

        Returns
        -------
        warped_img : list of ndarray
            List of moving images registered to align with the fixed image.

        warped_grid : list of ndarray
            Image showing deformation applied to a regular grid.

        bk_dxdy : list of ndarray
            List numpy array of pixel displacements in the x and y directions
            for each image. Has shape (I, N, M, 2).

        """

        self.shape = img_list[0].shape
        for img in img_list:
            assert img.shape == self.shape, print("Images have differernt shapes")

        self.img_list = img_list
        self.size = len(img_list)
        if mask is None:
            mask = np.full(self.img_list[0].shape[0:2], 255, dtype=np.uiint8)

        self.mask = mask
        if self.mask is not None:
            mask = mask.astype(np.uint8)
            self.mask = mask.copy()

            self.apply_mask(self.mask)
            mask_bbox = warp_tools.xy2bbox(warp_tools.mask2xy(mask))
            min_c, min_r = mask_bbox[0:2]
            max_c, max_r = mask_bbox[0:2] + mask_bbox[2:]
            mask = self.mask[min_r:max_r, min_c:max_c]
            temp_img_list = [None] * self.size
            for i, img in enumerate(self.img_list):

                temp_img_list[i] = img[min_r:max_r, min_c:max_c]
        else:
            temp_img_list = self.img_list

        backward_deformations = self.calc(temp_img_list, mask=mask)
        if self.mask is not None:
            temp_backward_deformations = [None] * self.size
            for i in range(self.size):
                bk_dx = np.zeros(self.shape)
                bk_dx[min_r:max_r, min_c:max_c] = backward_deformations[i][0]
                bk_dx[self.mask == 0] = 0

                bk_dy = np.zeros(self.shape)
                bk_dy[min_r:max_r, min_c:max_c] = backward_deformations[i][1]
                bk_dy[self.mask == 0] = 0

                temp_backward_deformations[i] = np.array([bk_dx, bk_dy])

            backward_deformations = np.array(temp_backward_deformations)

        self.backward_dx = backward_deformations[:, 0]
        self.backward_dy = backward_deformations[:, 1]

        n_imgs = len(self.img_list)
        warp_maps = [warp_tools.get_warp_map(dxdy=[self.backward_dx[i],
                                                   self.backward_dy[i]])
                     for i in range(n_imgs)]

        warped_imgs = [transform.warp(img_list[i], warp_maps[i], preserve_range=True)
                       for i in range(n_imgs)]

        grid_img = self.get_grid_image(grid_spacing=16)
        warped_grids = [transform.warp(grid_img, warp_maps[i], preserve_range=True)
                        for i in range(n_imgs)]

        self.warped_image = warped_imgs
        self.deformation_field_img = warped_grids

        return warped_imgs, warped_grids, backward_deformations


# Class members that perform non-rigid registrations #
class SimpleElastixWarper(NonRigidRegistrarXY):
    """Uses SimpleElastix to register images

    May optionally using corresponding points

    """
    def __init__(self, params=None, ammi_weight=0.33,
                 bending_penalty_weight=0.33, kp_weight=0.33):
        """
        Parameters
        ----------
        ammi_weight : float
            Weight given to the AdvancedMattesMutualInformation metric.

        bending_penalty_weight : float
            Weight given to the TransformBendingEnergyPenalty metric.

        kp_weight : float
            Weight given to the CorrespondingPointsEuclideanDistanceMetric
            metric. Only used if moving_xy and fixed_xy are provided as
            arguments to the `register()` method.

        """
        super().__init__(params=params)

        self.ammi_weight = ammi_weight
        self.bending_penalty_weight = bending_penalty_weight
        self.kp_weight = kp_weight


    @staticmethod
    def get_default_params(img_shape, grid_spacing_ratio=0.025):
        """
        Get default parameters for registration with sitk.ElastixImageFilter

        See https://simpleelastix.readthedocs.io/Introduction.html
        for advice on parameter selection
        """
        p = sitk.GetDefaultParameterMap("bspline")
        p["Metric"] = ['AdvancedMattesMutualInformation', 'TransformBendingEnergyPenalty']
        p["MaximumNumberOfIterations"] = ['1500']  # Can try up to 2000
        p['FixedImagePyramid'] = ["FixedRecursiveImagePyramid"]
        p['MovingImagePyramid'] = ["MovingRecursiveImagePyramid"]
        p['Interpolator'] = ["BSplineInterpolator"]
        p["ImageSampler"] = ["RandomCoordinate"]
        p["MetricSamplingStrategy"] = ["None"]  # Use all points
        p["UseRandomSampleRegion"] = ["true"]
        p["ErodeMask"] = ["true"]
        p["NumberOfHistogramBins"] = ["32"]
        p["NumberOfSpatialSamples"] = ["3000"]
        p["NewSamplesEveryIteration"] = ["true"]
        p["SampleRegionSize"] = [str(min([img_shape[1]//3, img_shape[0]//3]))]
        p["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
        p["ASGDParameterEstimationMethod"] = ["DisplacementDistribution"]
        p["HowToCombineTransforms"] = ["Compose"]
        grid_spacing_x = img_shape[1]*grid_spacing_ratio
        grid_spacing_y = img_shape[0]*grid_spacing_ratio
        grid_spacing = str(int(np.mean([grid_spacing_x, grid_spacing_y])))
        p["FinalGridSpacingInPhysicalUnits"] = [grid_spacing]
        p["WriteResultImage"] = ["false"]

        return p

    @staticmethod
    def elastix_invert_transform(registed_elastix_obj, sitk_fixed):
        """Invert transformation as described in elastix manual.

        See section 6.1.6: DisplacementMagnitudePenalty: inverting transformations

        Parameters
        ----------
        registed_elastix_obj: sitk.ElastixImageFilter
            sitk.ElastixImageFilter object that has completed
            image registration.

        sitk_fixed : SimpleITK.Image
            SimpleITK.Image created from the fixed image.

        Returns
        -------
        inverted_deformationField : ndarray
            (N,M,2) numpy array of pixel displacements in the
            x and y directions.

        NOTE
        ----
        sitk.IterativeInverseDisplacementField seems to do a better job,
        and is what is used in warp_tools.get_inverse_field. However, this
        method is maintained in case one would like to use it.

        """

        inverse_transformationFilter = sitk.TransformixImageFilter()
        transf_parameter_map = registed_elastix_obj.GetTransformParameterMap()
        transf_parameter_map[0]["Metric"] = ["DisplacementMagnitudePenalty"]
        transf_parameter_map[0]["HowToCombineTransforms"] = ["Compose"]
        inverse_transformationFilter.SetMovingImage(sitk_fixed)
        inverse_transformationFilter.SetTransformParameterMap(transf_parameter_map)
        inverse_transformationFilter.ComputeDeformationFieldOn()
        inverse_transformationFilter.Execute()
        inverted_deformationField = sitk.GetArrayFromImage(inverse_transformationFilter.GetDeformationField())

        return inverted_deformationField

    def write_elastix_kp(self, kp, fname):
        """Temporarily write fixed_xy and moving_xy to file

        Parameters
        ----------
        kp: ndarray
            (N, 2) numpy array of points (xy).

        fname: str
            Name of file in which to save the points.

        """

        argfile = open(fname, 'w')
        npts = kp.shape[0]
        argfile.writelines(f"index\n{npts}\n")
        for i in range(npts):
            xy = kp[i]
            argfile.writelines(f"{xy[0]} {xy[1]}\n")

    def run_elastix(self, moving_img, fixed_img, moving_xy=None, fixed_xy=None,
                    params=None, mask=None):

        """Run SimpleElastix to register images.

        Can using corresponding points to aid in registration by providing
        moving_xy and fixed_xy.

        Parameters
        ----------
        moving_img : ndarray
            Image to warp to align with `fixed_img`.

        fixed_img : ndarray
            Image `moving_img` is warped to align with.

        moving_xy : ndarray, optional
            (N, 2) array containing points in the moving image that correspond
            to those in the fixed image.

        fixed_xy : ndarray, optional
            (N, 2) array containing points in the fixed image that correspond
            to those in the moving image.

        mask : ndarray, optional
            2D array with shape (N,M) where non-zero pixel values are
            foreground, and 0 is background, which is ignnored during
            registration. If None, then all non-zero pixels in images
            will be used to create the mask.

        """

        elastix_image_filter_obj = sitk.ElastixImageFilter()

        if moving_xy is not None and fixed_xy is not None:

            rand_id = np.random.randint(0, 10000)
            fixed_kp_fname = os.path.join(pathlib.Path(__file__).parent,
                                          f".{rand_id}_fixedPointSet.pts")
            moving_kp_fname = os.path.join(pathlib.Path(__file__).parent,
                                           f".{rand_id}_.movingPointSet.pts")

            self.write_elastix_kp(fixed_xy, fixed_kp_fname)
            self.write_elastix_kp(moving_xy, moving_kp_fname)

            kp_dist_met = "CorrespondingPointsEuclideanDistanceMetric"
            current_metrics = list(params["Metric"])
            if not self._params_provided or kp_dist_met not in current_metrics:
                current_metrics.append(kp_dist_met)
                params["Metric"] = current_metrics
                weights = np.array([self.ammi_weight,
                                    self.bending_penalty_weight,
                                    self.kp_weight])

            elastix_image_filter_obj.SetParameterMap(params)
            elastix_image_filter_obj.SetFixedPointSetFileName(fixed_kp_fname)
            elastix_image_filter_obj.SetMovingPointSetFileName(moving_kp_fname)
        else:
            weights = np.array([self.ammi_weight, self.bending_penalty_weight])

        # Set metric weights #
        weights /= weights.sum()
        n_metrics = len(params["Metric"])
        n_res = eval(params["NumberOfResolutions"][0])
        for r in range(n_metrics):
            params[f'Metric{r}Weight'] = [str(weights[r])]*n_res

        elastix_image_filter_obj.SetParameterMap(params)

        # Perform registration #
        sitk_moving = sitk.GetImageFromArray(moving_img)
        sitk_fixed = sitk.GetImageFromArray(fixed_img)
        elastix_image_filter_obj.SetMovingImage(sitk_moving)
        elastix_image_filter_obj.SetFixedImage(sitk_fixed)

        if mask is not None:
            sitk_mask = sitk.Cast(sitk.GetImageFromArray(mask.astype(np.uint8)),
                                  sitk.sitkUInt8)

            elastix_image_filter_obj.SetFixedMask(sitk_mask)

        elastix_image_filter_obj.Execute()

        # Get deformation field #
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(elastix_image_filter_obj.GetTransformParameterMap())
        transformixImageFilter.ComputeDeformationFieldOn()
        transformixImageFilter.Execute()
        deformationField = sitk.GetArrayFromImage(transformixImageFilter.GetDeformationField())

        # Warp image #
        resultImage = elastix_image_filter_obj.GetResultImage()
        resultImage = sitk.GetArrayFromImage(resultImage)

        # Get deformation grid #
        grid_spacing = int(eval(params["FinalGridSpacingInPhysicalUnits"][0]))
        grid_img = self.get_grid_image(grid_spacing=grid_spacing)
        transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(grid_img))
        transformixImageFilter.Execute()
        warped_grid = sitk.GetArrayFromImage(transformixImageFilter.GetResultImage())

        if moving_xy is not None and fixed_xy is not None:
            if os.path.exists(fixed_kp_fname):
                os.remove(fixed_kp_fname)

            if os.path.exists(moving_kp_fname):
                os.remove(moving_kp_fname)

        tform_files = [f for f in os.listdir(".")
                    if f.startswith("TransformParameters.")
                    and f.endswith(".txt")]

        if len(tform_files) > 0:
            for f in tform_files:
                os.remove(f)

        return resultImage, warped_grid, deformationField, elastix_image_filter_obj, transformixImageFilter

    def calc(self, moving_img, fixed_img, mask=None,
             moving_xy=None, fixed_xy=None, *args, **kwargs):
        """Perform non-rigid registration using SimpleElastix.

        Can include corresponding points to help in registration by providing
        `moving_xy` and `fixed_xy`.

        """

        assert moving_img.shape == fixed_img.shape,\
            print("Images have different shapes")

        if not self._params_provided:
            self.params = self.get_default_params(self.moving_img.shape)

        warped_img, \
            warped_grid, \
            backward_deformation, \
            backward_elastix_image_filter_obj, \
            backward_transformixImageFilter = \
            self.run_elastix(moving_img, fixed_img,
                             moving_xy=moving_xy, fixed_xy=fixed_xy,
                             params=self.params, mask=mask)

        # Record other params #
        self.grid_spacing = int(eval(self.params["FinalGridSpacingInPhysicalUnits"][0]))
        self.elastix_params = self.params.asdict()
        self.params = None  # Can't pickle SimpleITK.ParameterMap
        self.method = backward_elastix_image_filter_obj.__class__.__name__
        dxdy = np.array([backward_deformation[..., 0], backward_deformation[..., 1]])

        return dxdy


class OpticalFlowWarper(NonRigidRegistrar):
    """Use dense optical flow to register images.

    Dense optical flow fields may not be diffeomorphic, and so
    this class provides options to smooth displacement fields.
    """
    def __init__(self, params=None, optical_flow_obj=None,
                 n_grid_pts=50, sigma_ratio=0.005,
                 paint_size=5000, fold_penalty=1e-6,
                 smoothing_method=None):
        """
        Parameters
        ----------
        params : dictionary
            Keyword: value dictionary of parameters to be used in reigstration.
            Will get used in the calc() method.

        optical_flow_obj : object
            Object that will perform dense optical flow.

        n_grid_pts : int
            Number of gridpoints used to detect folds. Also the number
            of gridpoints to use when regularizing he mesh when
            `method` = "regularize".

        paint_size : int
            Used to determine how much to resize the image to have
            efficient inpainting. Larger values = longer processing time.
            Only used if `smoothing_method` = "inpaint".

        fold_penalty : float
            How much to penalize folding/stretching. Larger values will make
            the deformation field more uniform, which may or may not be
            desired, as too much can remove all displacements.
            Only used if `smoothing_method` = "regularize"

        sigma_ratio : float
            Determines the amount of Gaussian smoothing, as
            sigma = max(shape) *sigma_ratio. Larger values do more
            smoothing. Only used if `smoothing_method` is "gauss".

        smoothing : str
            If "gauss", then a Gaussian blur will be applied to the
            deformation fields, using sigma defined by sigma_ratio.

            If "inpaint", folded regions will be detected and removed.
            Folded regions will then be removed using inpainting.

            If "regularize", folded regions will be detected and
            regularized using the method fescribed in
            "Foldover-free maps in 50 lines of code" Garanzha et al. 2021.

            If "None" then no smoothing will be applied.

        """

        super().__init__(params)

        self.smoothing_method = smoothing_method
        self.sigma_ratio = sigma_ratio
        self.paint_size = paint_size
        self.fold_penalty = fold_penalty
        self.n_grid_pts = n_grid_pts
        if optical_flow_obj is None:
            optical_flow_obj = cv2.optflow.createOptFlow_DeepFlow

        self.method = optical_flow_obj.__name__
        self.optical_flow_obj = optical_flow_obj()

    def calc(self, moving_img, fixed_img, *args, **kwargs):
        if self.method in ['createOptFlow_DenseRLOF', 'createOptFlow_SimpleFlow']:
            if moving_img.ndim == 2:
                moving_img = color.gray2rgb(moving_img)

            if fixed_img.ndim == 2:
                fixed_img = color.gray2rgb(fixed_img)

        backward_flow = self.optical_flow_obj.calc(fixed_img, moving_img,
                                                   np.zeros(moving_img.shape[0:2]))

        backward_flow = np.array([backward_flow[..., 0], backward_flow[..., 1]])
        if self.smoothing_method == "gauss":
            sigma = self.sigma_ratio*np.max(backward_flow[0].shape)
            smooth_dx = filters.gaussian(backward_flow[0], sigma=sigma)
            smooth_dy = filters.gaussian(backward_flow[1], sigma=sigma)
            backward_flow = np.array([smooth_dx, smooth_dy])

        elif self.smoothing_method == "inpaint":
            backward_flow = warp_tools.remove_folds_in_dxdy(backward_flow,
                                                            n_grid_pts=self.n_grid_pts,
                                                            paint_size=self.paint_size,
                                                            method=self.smoothing_method)
        elif self.smoothing_method == "regularize":
            backward_flow = warp_tools.untangle(backward_flow,
                                                n_grid_pts=self.n_grid_pts,
                                                penalty=self.fold_penalty,
                                                mask=self.mask)

        self.optical_flow_obj = None  # Can't pickle OpenCV objects

        return np.array(backward_flow)


class SimpleElastixGroupwiseWarper(NonRigidRegistrarGroupwise):
    """
    Performs groupwise non-rigid registration using SimpleElastix.

    SimpleElastixGroupwiseWarper can register a collection (>= 2) of images,
    and so is not limited to pairs of images.

    Attributes
    ----------
    img_list : list
        List of images, each with shape (N,M) that are to be co-registered.

    mask : ndarray
        2D array with shape (N,M) where non-zero pixel values are foreground,
        and 0 is background, which is ignnored during registration. If None,
        then all non-zero pixels in images will be used to create the mask.

    shape : tuple of int
        Number of rows and columns in each image.
        Will have shaape (N,M).

    warped_image : ndarray
        Registered copy of `moving_img`.

    deformation_field_img : ndarray
        Image showing deformation applied to a regular grid.

    backward_dx : ndarray
        (N,M) array defining the displacements in the x-dimension.

    backward_dy : ndarray
        (N,M) array defining the displacements in the y-dimension.

    grid_spacing : int
        Number of pixels between deformation grid points.

    method : str
        Name of registration method.

    """

    def __init__(self, params=None):
        super().__init__(params=params)

    @staticmethod
    def get_default_params(img_shape, grid_spacing_ratio=0.025):
        """
        See https://simpleelastix.readthedocs.io/Introduction.html for advice on parameter selection
        """
        p = sitk.GetDefaultParameterMap("groupwise")
        p["Metric"] = ['AdvancedMattesMutualInformation']
        p["MaximumNumberOfIterations"] = ['1500']  # Can try up to 2000
        p['FixedImagePyramid'] = ["FixedRecursiveImagePyramid"]
        p['MovingImagePyramid'] = ["MovingRecursiveImagePyramid"]
        p["ImageSampler"] = ["RandomCoordinate"]
        p["MetricSamplingStrategy"] = ["None"]  # Use all points
        p["UseRandomSampleRegion"] = ["true"]
        p["ErodeMask"] = ["true"]
        p["NumberOfSpatialSamples"] = ["3000"]
        p["NewSamplesEveryIteration"] = ["true"]
        p["Optimizer"] = ["AdaptiveStochasticGradientDescent"]
        p["ASGDParameterEstimationMethod"] = ["DisplacementDistribution"]
        p["HowToCombineTransforms"] = ["Compose"]
        grid_spacing_x = img_shape[1]*grid_spacing_ratio
        grid_spacing_y = img_shape[0]*grid_spacing_ratio
        grid_spacing = str(int(np.mean([grid_spacing_x, grid_spacing_y])))
        p["FinalGridSpacingInPhysicalUnits"] = [grid_spacing]
        p["WriteResultImage"] = ["false"]

        return p

    def calc(self, img_list, mask=None, *args, **kwargs):
        if self.params is None:
            self.params = SimpleElastixGroupwiseWarper.get_default_params(self.img_list[0].shape[:2])

        vectorOfImages = sitk.VectorOfImage()
        for img in img_list:
            vectorOfImages.push_back(sitk.GetImageFromArray(img))

        image = sitk.JoinSeries(vectorOfImages)
        elastix_image_filter_obj = sitk.ElastixImageFilter()
        elastix_image_filter_obj.SetFixedImage(image)
        elastix_image_filter_obj.SetMovingImage(image)
        elastix_image_filter_obj.SetParameterMap(self.params)

        if mask is not None:
            vectorOfMasks = sitk.VectorOfImage()
            for i in range(len(img_list)):
                vectorOfMasks.push_back(sitk.GetImageFromArray(mask))
            mask3d = sitk.JoinSeries(vectorOfMasks)
            elastix_image_filter_obj.SetFixedMask(mask3d)

        elastix_image_filter_obj.Execute()

        # Get warped images #
        resultImage = elastix_image_filter_obj.GetResultImage()
        resultImage = sitk.GetArrayFromImage(resultImage)

        # Get deformation fields #
        transformixImageFilter = sitk.TransformixImageFilter()
        transformixImageFilter.SetTransformParameterMap(elastix_image_filter_obj.GetTransformParameterMap())
        transformixImageFilter.SetMovingImage(image)
        transformixImageFilter.ComputeDeformationFieldOn()
        transformixImageFilter.Execute()
        deformationField = sitk.GetArrayFromImage(transformixImageFilter.GetDeformationField())[..., 0:2]

        # Get deformation grid #
        grid_spacing = int(eval(self.params["FinalGridSpacingInPhysicalUnits"][0]))
        self.elastix_params = self.params.asdict()
        self.params = None  # Can't pickle SimpleITK.ParameterMap
        grid_img = self.get_grid_image(grid_spacing=grid_spacing)
        self.method = elastix_image_filter_obj.__class__.__name__

        vectorOfGrids = sitk.VectorOfImage()
        for i in range(len(img_list)):
            vectorOfGrids.push_back(sitk.GetImageFromArray(grid_img))
        grid3d = sitk.JoinSeries(vectorOfGrids)

        transformixImageFilter.SetMovingImage(grid3d)
        transformixImageFilter.Execute()
        warped_grid = sitk.GetArrayFromImage(transformixImageFilter.GetResultImage())

        tform_files = [f for f in os.listdir(".")
                    if f.startswith("TransformParameters.")
                    and f.endswith(".txt")]

        if len(tform_files) > 0:
            for f in tform_files:
                os.remove(f)

        deformationField = np.array([[deformationField[i][...,  0],
                                      deformationField[i][...,  1]]
                                     for i in range(len(deformationField))])
        return deformationField


class NonRigidTileRegistrar(object):
    """Tile-wise non-rigid regisration

    Slices moving and fixed images into tiles and then registers each tile.
    Probably best for very large images.

    Attributes
    ----------
    moving_img : pyvips.Image
        Image with shape (N,M) thata is  warp to align with `fixed_img`.

    fixed_img : pyvips.Image
        Image with shape (N,M) that `moving_img` is warped to align with.

    mask : pyvips.Image
        2D array with shape (N,M) where non-zero pixel values are foreground,
        and 0 is background, which is ignnored during registration. If None,
        then all non-zero pixels in images will be used to create the mask.

    shape : tuple
        Number of rows and columns in each image. Will be (N,M).

    bk_dxdy_tiles : list
        List of bk_dxdy for each tile

    bk_dxdy : pyvips.Image
        Backwards isplacement field after stitching `bk_dxdy_tiles` together

    fwd_dxdy_tiles : list
        List of forward dxdy for each tile

    fwd_dxdy : pyvips.Image
        Displacement field after stitching `fwd_dxdy_tiles` together
    """

    def __init__(self, params=None, tile_wh=512, tile_buffer=100):
        """
        Parameters
        ----------
        params : dictionary
            Keyword: value dictionary of parameters to be used in reigstration.
            Will get used when initializing the `non_rigid_registrar_cls`

            In the case where simple ITK will be used, params should be
            a SimpleITK.ParameterMap. Note that numeric values needd to be
            converted to strings.

        tile_wh : int
            Width and height of tiles that will be used for registration

        tile_buffer : int
            The amount of overlap between each tile.

        """
        self.tile_wh = tile_wh
        self.tile_buffer = tile_buffer
        self.params = params

        self.moving_img = None
        self.fixed_img = None

        self.mask = None
        self.shape = None
        self.warped_image = None

        self.bk_dxdy_tiles = None
        self.bk_dxdy = None

        self.fwd_dxdy_tiles = None
        self.fwd_dxdy = None

    def norm_img(self, img, stats, mask=None):
        normed_img = exposure.rescale_intensity(img, out_range=(0, 255)).astype(np.uint8)
        normed_img = preprocessing.norm_img_stats(img=normed_img, target_stats=stats, mask=mask)
        normed_img = exposure.rescale_intensity(normed_img, out_range=(0, 255)).astype(np.uint8)

        return normed_img

    def norm_tiles(self, moving_img, fixed_img, tile_mask):
        try:
            _, target_processing_stats = preprocessing.collect_img_stats([fixed_img, moving_img])
            fixed_normed = self.norm_img(img=fixed_img, stats=target_processing_stats, mask=tile_mask)
            moving_normed = self.norm_img(moving_img, target_processing_stats, tile_mask)

        except ValueError:
            # Norm using full image's stats
            if self.target_stats is not None:
                try:
                    fixed_normed = self.norm_img(fixed_img, self.target_stats, tile_mask)
                    moving_normed = self.norm_img(moving_img, self.target_stats, tile_mask)
                except ValueError:
                    fixed_normed = fixed_img
                    moving_normed = moving_img
            else:
                fixed_normed = fixed_img
                moving_normed = moving_img

        return moving_normed, fixed_normed

    def process_tile(self, img, img_processer_cls, processer_init_kwargs={}, processer_kwargs={}):
        """Process tiles
        """

        processer_init_kwargs["image"] = img
        processer_init_kwargs['reader'] = deepcopy(processer_init_kwargs["reader"])
        processer_init_kwargs['level'] = 0
        processer = img_processer_cls(**processer_init_kwargs)
        try:
            processed_img = processer.process_image(**processer_kwargs)
        except TypeError:
            # processor.process_image doesn't take kwargs
            processed_img = processer.process_image()

        return processed_img

    def reg_tile(self, tile_idx, lock):
        with lock:
            # Use lock when accessing images
            tile_bbox_xywh = self.expanded_bboxes[tile_idx]
            moving_tile = self.moving_img.extract_area(*tile_bbox_xywh)
            fixed_tile = self.fixed_img.extract_area(*tile_bbox_xywh)

            np_fixed = warp_tools.vips2numpy(fixed_tile)
            np_moving = warp_tools.vips2numpy(moving_tile)

            if self.mask is not None:
                tile_mask = self.mask.extract_area(*tile_bbox_xywh)
                np_mask = warp_tools.vips2numpy(tile_mask)
            else:
                np_mask = None

            if moving_tile.interpretation == "srgb":
                # Limit registration to be inside image
                # Warped areas outside image have the same pixel values, usually 0
                edge_mask = 255*((np_moving.min(axis=2) != np_moving.max(axis=2)) & (np_fixed.min(axis=2) != np_fixed.max(axis=2))).astype(np.uint8)

                if np_mask is not None:
                    np_mask = 255*((edge_mask > 0) & (np_mask > 0)).astype(np.uint8)
                else:
                    np_mask = edge_mask

            # Check if either of the tiles are empty
            is_empty = fixed_tile.max() == fixed_tile.min() or moving_tile.max() == moving_tile.min()
            if np_mask is not None:
                is_empty = is_empty or np_mask.max() == 0

            if is_empty:
                # Nothing to register
                empty_dxdy = pyvips.Image.black(moving_tile.width, moving_tile.height, bands=2).cast("float")
                self.bk_dxdy_tiles[tile_idx] = empty_dxdy
                self.fwd_dxdy_tiles[tile_idx] = empty_dxdy

                return None

            # Process tiles
            if self.moving_processer_cls is not None:
                moving_processed = self.process_tile(img=np_moving,
                                                    img_processer_cls=self.moving_processer_cls,
                                                    processer_init_kwargs=self.moving_processer_init_kwargs,
                                                    processer_kwargs=self.moving_processer_kwargs)

            else:
                if np_moving.ndim > 2:
                    moving_g = np.abs(1 - skcolor.rgb2gray(np_moving))
                    moving_processed = util.img_as_ubyte(moving_g)
                else:
                    moving_processed = np_moving

            if self.fixed_processer_cls is not None:
                fixed_processed = self.process_tile(img=np_fixed,
                                            img_processer_cls=self.fixed_processer_cls,
                                            processer_init_kwargs=self.fixed_processer_init_kwargs,
                                            processer_kwargs=self.fixed_processer_kwargs)
            else:
                if np_fixed.ndim > 2:
                    fixed_g = np.abs(1 - skcolor.rgb2gray(np_fixed))
                    fixed_processed = util.img_as_ubyte(fixed_g)
                else:
                    fixed_processed = np_fixed

            moving_normed, fixed_normed = self.norm_tiles(moving_processed, fixed_processed, np_mask)

            tile_non_rigid_reg_obj = self.non_rigid_registrar_cls()

            _, _, bk_dxdy = tile_non_rigid_reg_obj.register(moving_normed, fixed_normed)
            fwd_dxdy = warp_tools.get_inverse_field(bk_dxdy)

            vips_tile_bk_dxdy = warp_tools.numpy2vips(np.dstack(bk_dxdy).astype(np.float32))
            vips_tile_fwd_dxdy = warp_tools.numpy2vips(np.dstack(fwd_dxdy).astype(np.float32))

            self.bk_dxdy_tiles[tile_idx] = vips_tile_bk_dxdy
            self.fwd_dxdy_tiles[tile_idx] = vips_tile_fwd_dxdy

    def calc(self, *args, **kwargs):
        """Cacluate displacement fields
        Each tile is registered and then stitched together
        """

        print("======== Registering tiles\n")

        n_cpu = valtils.get_ncpus_available() - 1

        lock = multiprocessing.Lock()
        args = [{"tile_idx":i, "lock":lock} for i in range(self.n_tiles)]
        res = pqdm(args, self.reg_tile, n_jobs=n_cpu, unit="image", leave=None, argument_type='kwargs')

        bk_dxdy = warp_tools.stitch_tiles(self.bk_dxdy_tiles, self.expanded_bboxes, self.n_rows, self.n_cols, self.tile_buffer)
        fwd_dxdy = warp_tools.stitch_tiles(self.fwd_dxdy_tiles, self.expanded_bboxes, self.n_rows, self.n_cols, self.tile_buffer)

        return bk_dxdy, fwd_dxdy

    def register(self, moving_img, fixed_img, mask=None, non_rigid_registrar_cls=OpticalFlowWarper,
                 moving_processer_cls=None, moving_init_processer_kwargs={}, moving_processer_kwargs=None,
                 fixed_processer_cls=None, fixed_init_processer_kwargs={}, fixed_processer_kwargs=None,
                 target_stats=None, **kwargs):
        """
        Register images, warping moving_img to align with fixed_img

        Uses backwards transforms to register images (i.e. aligning
        fixed to moving), so the inverse transform needs to be used
        to warp points from moving_img. This is automatically done in
        warp_tools.warp_xy

        Parameters
        ----------
        moving_img : ndarray, pyvips.Image
            Image to warp to align with `fixed_img`.

        fixed_img : ndarray, pyvips.Image
            Image `moving_img` is warped to align with.

        mask : ndarray, pyvips.Image
            2D array with shape (N,M) where non-zero pixel values are foreground,
            and 0 is background, which is ignnored during registration. If None,
            then all non-zero pixels in images will be used to create the mask.

        non_rigid_registrar_cls : NonRigidRegistrar, optional
            Uninstantiated NonRigidRegistrar class that will be used
            to calculate the deformation fields between images.

        processing_cls : preprocessing.ImageProcesser, optional
            preprocessing.ImageProcesser used to process the images

        processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `processing_cls`

        target_stats : ndarray
            Target stats used to normalize each tile after being processed.

        **kwargs : dict, optional
            Additional keyword arguments passed to NonRigidRegistrar.calc

        Returns
        -------
        warped_img : pyvips.Image
            Moving image registered to align with fixed image.

        fwd_dxdy :
            (2, N, M)  pyvips.Image with pixel displacements in
            the x and y directions. Found by registering `moving_img`
            to `fixed_img`. Used for point warping

        bk_dxdy : pyvips.Image
            (2, N, M)  pyvips.Image with pixel displacements in
            the x and y directions. Found by registering `fixed_img` to
            `moving_img`. Used for image warping

        """

        self.is_array = False
        if not isinstance(moving_img, pyvips.Image):
            self.is_array = True

        if self.is_array:
            shape_rc = np.array(moving_img.shape)
        else:
            shape_rc = np.array([moving_img.height, moving_img.width])

        self.shape = shape_rc

        self.non_rigid_registrar_cls = non_rigid_registrar_cls
        self.target_stats = target_stats

        self.moving_processer_cls = moving_processer_cls
        self.moving_processer_kwargs = moving_processer_kwargs
        self.moving_processer_init_kwargs = moving_init_processer_kwargs

        self.fixed_processer_cls = fixed_processer_cls
        self.fixed_processer_kwargs = fixed_processer_kwargs
        self.fixed_processer_init_kwargs = fixed_init_processer_kwargs

        if self.is_array:
            moving_img = warp_tools.numpy2vips(moving_img)

        if not isinstance(fixed_img, pyvips.Image):
            fixed_img = warp_tools.numpy2vips(fixed_img)

        if mask is not None:
            if not isinstance(mask, pyvips.Image):
                mask = warp_tools.numpy2vips(mask)

        self.moving_img = moving_img
        self.fixed_img = fixed_img
        self.mask = mask

        temp_tile_bboxes = warp_tools.get_grid_bboxes(self.shape, self.tile_wh, self.tile_wh, inclusive=True)
        self.expanded_bboxes = np.array([warp_tools.expand_bbox(bbox_xywh, self.tile_buffer, self.shape) for bbox_xywh in temp_tile_bboxes])

        self.n_tiles = len(temp_tile_bboxes)
        self.bk_dxdy_tiles = [None] * self.n_tiles
        self.fwd_dxdy_tiles = [None] * self.n_tiles
        self.n_cols = len(np.unique(temp_tile_bboxes[:, 0]))
        self.n_rows = len(np.unique(temp_tile_bboxes[:, 1]))

        bk_dxdy, fwd_dxdy = self.calc()

        warped_img = warp_tools.warp_img(moving_img, bk_dxdy=bk_dxdy)
        if self.is_array:
            bk_dxdy = warp_tools.vips2numpy(bk_dxdy)
            bk_dxdy = [bk_dxdy[..., 0], bk_dxdy[..., 1]]

            warped_img = warp_tools.vips2numpy(warped_img)

        self.bk_dxdy = bk_dxdy
        self.fwd_dxdy = fwd_dxdy

        return warped_img, fwd_dxdy, bk_dxdy


