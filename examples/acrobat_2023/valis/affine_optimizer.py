"""Optimize rigid alignment

Contains functions related to optimization, as well as the AffineOptimizer
class that performs the optimzation. This class can be subclassed to implement
custom optimization methods.

There are several subclasses, but AffineOptimizerMattesMI is the
the fastest and most accurate, and so is default affine optimizer in VALIS.
It's not recommended that the other subclasses be used, but they are kept
to provide examples on how to subclass AffineOptimizer.
"""

from scipy import ndimage, optimize
import numba as nba
import numpy as np
from skimage import transform, util
import cv2
import os
import SimpleITK as sitk
from scipy import interpolate
import pathlib
from . warp_tools import get_affine_transformation_params, \
    get_corners_of_image, warp_xy

# Cost functions #
EPS = np.finfo("float").eps


def mse(arr1, arr2, mask=None):
    """Compute the mean squared error between two arrays."""

    if mask is None:
        return np.mean((arr1 - arr2)**2)
    else:
        return np.mean((arr1[mask != 0] - arr2[mask != 0]) ** 2)


def displacement(moving_image, target_image, mask=None):
    """Minimize average displacement between moving_image and target_image
    """

    opt_flow = cv2.optflow.createOptFlow_DeepFlow()
    flow = opt_flow.calc(util.img_as_ubyte(target_image),
                         util.img_as_ubyte(moving_image), None)
    if mask is not None:
        dx = flow[..., 0][mask != 0]
        dy = flow[..., 1][mask != 0]
    else:
        dx = flow[..., 0].reshape(-1)
        dy = flow[..., 1].reshape(-1)

    mean_displacement = np.mean(np.sqrt(dx**2 + dy**2))
    return mean_displacement


def cost_mse(param, reference_image, target_image, mask=None):
    transformation = make_transform(param)
    transformed = transform.warp(target_image, transformation, order=3)
    return mse(reference_image, transformed, mask)


def downsample2x(image):
    """Down sample image.
    """

    offsets = [((s + 1) % 2) / 2 for s in image.shape]
    slices = [slice(offset, end, 2)
              for offset, end in zip(offsets, image.shape)]
    coords = np.mgrid[slices]
    return ndimage.map_coordinates(image, coords, order=1)


def gaussian_pyramid(image, levels=6):
    """Make a Gaussian image pyramid.

    Parameters
    ----------
    image : array of float
        The input image.
    max_layer : int, optional
        The number of levels in the pyramid.

    Returns
    -------
    pyramid : iterator of array of float
        An iterator of Gaussian pyramid levels, starting with the top
        (lowest resolution) level.
    """
    pyramid = [image]

    for level in range(levels - 1):
        image = downsample2x(image)
        pyramid.append(image)

    return pyramid


def make_transform(param):
    if len(param) == 3:
        r, tc, tr = param
        s = None
    else:
        r, tc, tr, s = param

    return transform.SimilarityTransform(rotation=r,
                                         translation=(tc, tr),
                                         scale=s)


@nba.njit()
def bin_image(img, p):
    x_min = np.min(img)
    x_max_ = np.max(img)
    x_range = x_max_ - x_min + EPS
    binned_img = np.zeros_like(img)
    _bins = p * (1 - EPS)  # Keeps right bin closed
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            binned_img[i, j] = int(_bins * ((img[i, j] - x_min) / (x_range)))

    return binned_img


@nba.njit()
def solve_abc(verts):
    """
    Find coefficients A,B,C that will allow estimation of intesnity of point
    inside triangle with vertices v0, v1, v2. Each vertex is in the format of
    [x,y,z] were z=intensity of pixel at point x,y

    Parameters
    ----------
    verts : 3x3 array
        Each row has coordinates x,y and z, where z in the image intensiy at
        point xy (i.e. image[y, r])

    Returns
    -------
    abc : [A,B,C]
        Coefficients to estimate intensity in triangle, as well as the
        intersection of isointensity lines

    """
    a = np.array([[verts[0, 0], verts[0, 1], 1],
                 [verts[1, 0], verts[1, 1], 1],
                 [verts[2, 0], verts[2, 1], 1]])
    b = verts[:, 2]

    try:
        abc = np.linalg.inv(a) @ b
    except np.linalg.LinAlgError:
        sln = np.linalg.lstsq(a, b)
        abc = sln[0]

    return abc


@nba.njit()
def area(x1, y1, x2, y2, x3, y3):
    # From https://www.geeksforgeeks.org/check-whether-a-given-point-lies-inside-a-triangle-or-not/
    a = np.abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)
    return a


@nba.njit()
def isInside(x1, y1, x2, y2, x3, y3, x, y):
    # Calculate area of triangle ABC
    A = area(x1, y1, x2, y2, x3, y3)
    # A = np.abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)
    # Calculate area of triangle PBC
    A1 = area(x, y, x2, y2, x3, y3)

    # Calculate area of triangle PAC
    A2 = area(x1, y1, x, y, x3, y3)

    # Calculate area of triangle PAB
    A3 = area(x1, y1, x2, y2, x, y)

    # print(A, A1, A2, A3)
    # print(A == (A1 + A2 + A3))

    # Check if sum of A1, A2 and A3
    # is same as A
    if (A == (A1 + A2 + A3)):
        return 1
    else:
        return 0


@nba.njit()
def get_intersection(alpha1, alpha2, abc1, abc2):
    """

   Parameters
    ----------
    alpha1 : float
        Intensity of point in image 1

    alpha2 : float
        Intensity of point in image 2

    abc1: [A,B,C]
        Coefficients to interpolate value for triangle in image1

    abc2: [A,B,C]
        Coefficients to interpolate value for corresponding triangle in image2

    """
    # Find interestion of isointensity lines ###
    intensities = np.array([alpha1 - abc1[2], alpha2 - abc2[2]])
    coef = np.array([[abc1[0], abc1[1]],
                     [abc2[0], abc2[1]]
                     ])
    try:
        xy = np.linalg.inv(coef) @ intensities
    except np.linalg.LinAlgError:
        sln = np.linalg.lstsq(coef, intensities)
        xy = sln[0]
    return xy


@nba.njit()
def get_verts(img, x, y, pos=0):
    """
    Get veritices of triangle and intenisty at each vertex
    """
    if pos == 0:
        # Lower left
        verts = np.array([[x, y, img[y, x]],  # BL
                          [x + 1, y, img[y, x + 1]],  # BR
                          [x, y + 1, img[y + 1, x]]  # TL
                          ])
    if pos == 1:
        # Upper right
        verts = np.array([[x, y+1, img[y+1, x]],  # BL
                          [x + 1, y, img[y, x + 1]],  # BR
                          [x+1, y + 1, img[y + 1, x + 1]]  # TL
                          ])

    return verts


@nba.njit()
def hist2d(x, y, n_bins):
    """
    Build 2D histogram by determining the bin each x and y value falls in
    https://stats.stackexchange.com/questions/236205/programmatically-calculate-which-bin-a-value-will-fall-into-for-a-histogram
    """

    x_min = np.min(x)
    x_max_ = np.max(x)
    x_range = x_max_ - x_min + EPS

    y_min = np.min(y)
    y_max = np.max(y)
    y_range = y_max - y_min + EPS

    _bins = n_bins * (1 - EPS)  # Keeps right bin closed
    x_margins = np.zeros(n_bins)
    y_margins = np.zeros(n_bins)
    results = np.zeros((n_bins, n_bins))
    for i in range(len(x)):
        x_bin = int(_bins*((x[i]-x_min)/(x_range)))
        y_bin = int(_bins*((y[i] - y_min) / (y_range)))

        x_margins[x_bin] += 1
        y_margins[y_bin] += 1
        results[x_bin, y_bin] += 1

    return results, x_margins, y_margins


@nba.njit()
def update_joint_H(binned_moving, binned_fixed, H, M, sample_pts, pos=0,
                   precalcd_abc=None):

    q = H.shape[0]
    for i, sxy in enumerate(sample_pts):
        # Get vertices and intensities in each image.
        # Note that indices are as rc, but vertices need to be xy
        img1_v = get_verts(binned_moving, sxy[0], sxy[1], pos)
        abc1 = solve_abc(img1_v)

        if precalcd_abc is None:
            img2_v = get_verts(binned_fixed, sxy[0], sxy[1], pos)
            abc2 = solve_abc(img2_v)
        else:
            # ABC for fixed image's trianges are precomputed
            abc2 = precalcd_abc[i]

        x_lims = np.array([np.min(img1_v[:, 0]), np.max(img1_v[:, 0])])
        y_lims = np.array([np.min(img1_v[:, 1]), np.max(img1_v[:, 1])])
        for alpha1 in range(0, q):
            for alpha2 in range(0, q):
                xy = get_intersection(alpha1, alpha2, abc1, abc2)
                if xy[0] <= x_lims[0] or xy[0] >= x_lims[1] or \
                   xy[1] <= y_lims[0] or xy[1] >= y_lims[1]:
                    continue

                    #  Determine if intersection inside triangle ###
                vote = isInside(img1_v[0, 0], img1_v[0, 1],
                                img1_v[1, 0], img1_v[1, 1],
                                img1_v[2, 0], img1_v[2, 1],
                                xy[0], xy[1])

                H[alpha1, alpha2] += vote

    return H


@nba.jit()
def get_neighborhood(im, i, j, r):
    """
    Get values in a neighborhood
    """

    return im[i - r:i + r + 1, j - r:j + r + 1].flatten()


@nba.jit()
def build_P(A, B, r, mask):
    hood_size = (2 * r + 1) ** 2
    d = 2 * hood_size
    N = (A.shape[0] - 2*r)*(A.shape[1] - 2*r)
    P = np.zeros((d, N))

    idx = 0
    for i in range(r, A.shape[0]):
        # Skip borders
        if i < r or i > A.shape[0] - r - 1:
            continue
        for j in range(r, A.shape[1]):
            pmask = get_neighborhood(mask, i, j, r)
            if j < r or j > A.shape[1] - r - 1 or np.min(pmask) == 0:
                continue

            pa = get_neighborhood(A, i, j, r)
            pb = get_neighborhood(B, i, j, r)

            P[:hood_size, idx] = pa
            P[hood_size:, idx] = pb

            idx += 1

    return P[:, :idx]


@nba.njit()
def entropy(x):
    """
    Caclulate Shannon's entropy for array x

    Parameters
    ----------
    x : array
        Array from which to calculate entropy

    Returns
    -------
    h : float
        Shannon's entropy
    """
    # x += EPS ## Avoid -Inf if there is log(0)
    px = x/np.sum(x)
    px = px[px > 0]
    h = -np.sum(px * np.log(px))
    return h


@nba.njit()
def entropy_from_c(cov_mat, d):
    e = np.log(((2*np.pi*np.e) ** (d/2)) *
               (np.linalg.det(cov_mat) ** 0.5) + EPS)
    return e


@nba.njit()
def region_mi(A, B, mask, r=4):
    P = build_P(A, B, r, mask)  # d x N matrix: N points with d dimensions

    # Center points so each dimensions is around 0
    C = np.cov(P, rowvar=True, bias=True)
    hood_size = (2 * r + 1) ** 2
    d = hood_size*2
    HA = entropy_from_c(C[0:hood_size, 0:hood_size], d)
    HB = entropy_from_c(C[hood_size:, hood_size:], d)
    HC = entropy_from_c(C, d)

    RMI = HA + HB - HC
    if RMI < 0:
        RMI = 0

    return RMI


@nba.njit()
def normalized_mutual_information(A, B, mask, n_bins=256):
    """
    Build 2D histogram by determining the bin each x and y value falls in
    https://stats.stackexchange.com/questions/236205/programmatically-calculate-which-bin-a-value-will-fall-into-for-a-histogram
    """

    x_min = np.min(A)
    x_max_ = np.max(A)
    x_range = x_max_ - x_min + EPS

    y_min = np.min(B)
    y_max = np.max(B)
    y_range = y_max - y_min + EPS

    _bins = n_bins * (1 - EPS)  # Keeps right bin closed
    x_margins = np.zeros(n_bins)
    y_margins = np.zeros(n_bins)
    results = np.zeros((n_bins, n_bins))

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if mask[i, j] == 0:
                continue

            x = A[i, j]
            y = B[i, j]

            x_bin = int(_bins * ((x - x_min) / x_range))
            y_bin = int(_bins * ((y - y_min) / y_range))

            x_margins[x_bin] += 1
            y_margins[y_bin] += 1
            results[x_bin, y_bin] += 1

    n = np.sum(results)
    results /= n
    x_margins /= n
    y_margins /= n

    H_A = entropy(x_margins)
    H_B = entropy(y_margins)
    H_AB = entropy(results.flatten())
    MI = (H_A + H_B) / H_AB
    if MI < 0:
        MI = 0

        return MI


def sample_img(img, spacing=10):
    sr, sc = np.meshgrid(np.arange(0, img.shape[0], spacing), np.arange(0, img.shape[1], spacing))
    sample_r = sr.reshape(-1) + np.random.uniform(0, spacing/2, sr.size)
    sample_c = sc.reshape(-1) + np.random.uniform(0, spacing/2, sc.size)
    interp = interpolate.RectBivariateSpline(np.arange(0, img.shape[0]), np.arange(0, img.shape[1]), img)
    z = np.array([interp(sample_r[i], sample_c[i])[0][0] for i in range(len(sample_c))])
    return z[(0 <= z) & (z <= img.max())]


def MI(fixed, moving, nb, spacing):
    fixed_sampled = sample_img(fixed, spacing)
    moving_sampled = sample_img(moving, spacing)
    results, x_margins, y_margins = hist2d(moving_sampled, fixed_sampled, nb)

    n = np.sum(results)
    results /= n
    x_margins /= n
    y_margins /= n

    H_A = entropy(x_margins)
    H_B = entropy(y_margins)
    H_AB = entropy(results.flatten())
    MI = (H_A + H_B) / H_AB
    if MI < 0:
        MI = 0

    return MI


class AffineOptimizer(object):
    """Class that optimizes ridid registration

    Attributes
    ----------
    nlevels : int
        Number of levels in the Gaussian pyramid

    nbins : int
        Number of bins to have in histograms used to estimate mutual information

    optimization : str
        Optimization method. Can be any method from scipy.optimize
        "FuzzyPSO" for Fuzzy Self-Tuning PSO in the fst-pso package (https://pypi.org/project/fst-pso/)
        "gp_minimize", "forest_minimize", "gbrt_minimize" from scikit-opt

    transformation : str
        Type of transformation, "EuclideanTransform" or "SimilarityTransform"

    current_level : int
        Current level of the Guassian pyramid that is being registered

    accepts_xy : bool
        Bool declaring whether or not the optimizer will use corresponding points to optimize the registration

    Methods
    -------
    setup(moving, fixed, mask, initial_M=None)
        Gets images ready for alignment

    cost_fxn(fixed_image, transformed, mask)
        Calculates metric that is to be minimized

    align(moving, fixed, mask, initial_M=None, moving_xy=None, fixed_xy=None)
        Align images by minimizing cost_fxn


    Notes
    -----
    All AffineOptimizer subclasses need to have the method align(moving, fixed, mask, initial_M, moving_xy, fixed_xy)
    that returns the aligned image, optimal_M, cost_list

    AffineOptimizer subclasses must also have a cost_fxn(fixed_image, transformed, mask) method that
    returns the registration metric value

    If one wants to use the same optimization methods, but a different cost function, then the subclass only needs
    to have a new cost_fxn method. See AffineOptimizerDisplacement for an example implementing a new cost function

    Major overhauls are possible too. See AffineOptimizerMattesMI for an example on using SimpleITK's
    optimization methods inside of an AffineOptimizer subclass

    If the optimizer uses corressponding points, then the class attribute
    accepts_xy needs to be set to True. The default is False.

    """
    accepts_xy = False

    def __init__(self, nlevels=1, nbins=256, optimization="Powell", transformation="EuclideanTransform"):
        """AffineOptimizer registers moving and fixed images by minimizing a cost function

        Parameters
        ----------
        nlevels : int
            Number of levels in the Gaussian pyramid

        nbins : int
            Number of bins to have in histograms used to estimate mutual information

        optimization : str
            Optimization method. Can be any method from scipy.optimize

        transformation : str
            Type of transformation, "EuclideanTransform" or "SimilarityTransform"
        """

        self.nlevels = nlevels
        self.nbins = nbins
        self.optimization = optimization
        self.transformation = transformation
        self.current_level = nlevels - 1
        self.accepts_xy = AffineOptimizer.accepts_xy

    def setup(self, moving, fixed, mask, initial_M=None):
        """Get images ready for alignment

        Parameters
        ----------

        moving : ndarray
            Image to warp to align with fixed

        fixed : ndarray
            Image moving is warped to align to

        mask : ndarray
            2D array having non-zero pixel values, where values of 0 are ignnored during registration

        initial_M : (3x3) array
            Initial transformation matrix

        """
        self.moving = moving
        self.fixed = fixed

        if mask is None:
            self.mask = np.zeros(fixed.shape[0:2], dtype=np.uint8)
            self.mask[fixed != 0] = 1
        else:
            self.mask = mask

        self.pyramid_fixed = list(gaussian_pyramid(fixed, levels=self.nlevels))
        self.pyramid_moving = list(gaussian_pyramid(moving, levels=self.nlevels))
        self.pyramid_mask = list(gaussian_pyramid(self.mask, levels=self.nlevels))
        if self.transformation == "EuclideanTransform":
            self.p = np.zeros(3)
        else:
            self.p = np.zeros(4)
            self.p[3] = 1

        if initial_M is not None:
            (tx, ty), rotation, (scale_x, scale_y), shear = \
                get_affine_transformation_params(initial_M)

            self.p[0] = rotation
            self.p[1] = tx
            self.p[2] = ty
            if transform == "SimilarityTransform":
                self.p[3] = scale_x

    def cost_fxn(self, fixed_image, transformed, mask):
        return -normalized_mutual_information(fixed_image, transformed, mask, n_bins=self.nbins)

    def calc_cost(self, p):
        """Static cost function passed into scipy.optimize
        """
        transformation = make_transform(p)
        transformed = transform.warp(self.pyramid_moving[self.current_level], transformation.params, order=3)
        if np.all(transformed == 0):
            return np.inf

        return self.cost_fxn(self.pyramid_fixed[self.current_level], transformed, self.pyramid_mask[self.current_level])

    def align(self, moving, fixed, mask, initial_M=None, moving_xy=None, fixed_xy=None):
        """Align images by minimizing self.cost_fxn. Aligns each level of the Gaussian pyramid, and uses previous transform
        as the initial guess in the next round of optimization. Also uses other "good" estimates to define the
        parameter boundaries.

        Parameters
        ----------
        moving : ndarray
            Image to warp to align with fixed

        fixed : ndarray
            Image moving is warped to align with

        mask : ndarray
            2D array having non-zero pixel values, where values of 0 are ignnored during registration

        initial_M : (3x3) array
            Initial transformation matrix

        moving_xy : ndarray, optional
            (N, 2) array containing points in the moving image that correspond to those in the fixed image

        fixed_xy : ndarray, optional
            (N, 2) array containing points in the fixed image that correspond to those in the moving image

        Returns
        -------
        aligned : (N,M) array
            Moving image warped to align with the fixed image
        M : (3,3) array
            Optimal transformation matrix

        cost_list : list
            list containing the minimized cost for each level in the pyramid

        """

        self.setup(moving, fixed, mask, initial_M)
        method = self.optimization
        levels = range(self.nlevels-1, -1, -1)  # Iterate from top to bottom of pyramid
        cost_list = [None] * self.nlevels
        other_params = None
        for n in levels:
            self.current_level = n
            self.p[1:3] *= 2

            if other_params is None:
                max_tc = self.pyramid_moving[self.current_level].shape[1]
                max_tr = self.pyramid_moving[self.current_level].shape[0]
                param_bounds = [[0, np.deg2rad(360)],
                                [-max_tc, max_tc],
                                [-max_tr, max_tr]]

                if self.transformation == "SimilarityTransform":
                    param_bounds.append([self.p[3] * 0.5, self.p[3] * 2])
            # Update bounds based on best fits in previous level
            else:
                param_mins = np.min(other_params, axis=0)
                param_maxes = np.max(other_params, axis=0)
                param_bounds = [[param_mins[0], param_maxes[0]],
                                [2*param_mins[1], 2*param_maxes[1]],
                                [2*param_mins[2], 2*param_maxes[2]]]

                if self.transformation == "SimilarityTransform":
                    param_bounds.append([param_mins[3], param_maxes[3]])

            # Optimize #
            if method.upper() == 'BH':
                res = optimize.basinhopping(self.calc_cost, self.p)
                new_p = res.x
                cst = res.fun
                if n <= self.nlevels//2:  # avoid basin-hopping in lower levels
                    method = 'Powell'

            elif method == 'Nelder-Mead':
                res = optimize.minimize(self.calc_cost, self.p, method=method, bounds=param_bounds)
                new_p = res.x
                cst = np.float(res.fun)

            else:
                # Default is Powell, which doesn't accept bounds
                res = optimize.minimize(self.calc_cost, self.p, method=method, options={"return_all": True})
                new_p = res.x
                cst = np.float(res.fun)
                if hasattr(res, "allvecs"):
                    other_params = np.vstack(res.allvecs)

            if n <= self.nlevels // 2:  # avoid basin-hopping in lower levels
                method = 'Powell'

            # Update #
            self.p = new_p

            cost_list[self.current_level] = cst
            tf = make_transform(self.p)
            optimal_M = tf.params
            w = transform.warp(self.pyramid_moving[n], optimal_M, order=3)
            if np.all(w == 0):
                print(Warning("Image warped out of bounds. Registration failed"))
                return False, np.ones_like(optimal_M), cost_list

        tf = make_transform(self.p)
        M = tf.params
        aligned = transform.warp(self.moving, M, order=3)
        return aligned, M, cost_list


class AffineOptimizerMattesMI(AffineOptimizer):
    """ Optimize rigid registration using Simple ITK

    AffineOptimizerMattesMI is an AffineOptimizer subclass that uses simple ITK's AdvancedMattesMutualInformation.
    If moving_xy and fixed_xy are also provided, then Mattes mutual information will be maximized, while the distance
    between moving_xy and fixed_xy will be minimized (the CorrespondingPointsEuclideanDistanceMetric in Simple ITK).

    Attributes
    ----------
    nlevels : int
        Number of levels in the Gaussian pyramid

    nbins : int
        Number of bins to have in histograms used to estimate mutual information

    transformation : str
        Type of transformation, "EuclideanTransform" or "SimilarityTransform"

    Reg : sitk.ElastixImageFilter
        sitk.ElastixImageFilter object that will perform the optimization

    fixed_kp_fname : str
        Name of file where to fixed_xy will be temporarily be written. Eventually deleted

    moving_kp_fname : str
        Name of file where to moving_xy will be temporarily be written. Eventually deleted


    Methods
    -------
    setup(moving, fixed, mask, initial_M=None, moving_xy=None, fixed_xy=None)
        Create parameter map and initialize Reg

    calc_cost(p)
        Inherited but not used, returns None

    write_elastix_kp(kp, fname)
        Temporarily write fixed_xy and moving_xy to file

    align(moving, fixed, mask, initial_M=None, moving_xy=None, fixed_xy=None)
        Align images by minimizing cost_fxn

    """

    accepts_xy = True

    def __init__(self, nlevels=4.0, nbins=32,
                 optimization="AdaptiveStochasticGradientDescent", transform="EuclideanTransform"):
        super().__init__(nlevels, nbins, optimization, transform)

        self.Reg = None
        self.accepts_xy = AffineOptimizerMattesMI.accepts_xy
        self.fixed_kp_fname = os.path.join(pathlib.Path(__file__).parent, ".fixedPointSet.pts")
        self.moving_kp_fname = os.path.join(pathlib.Path(__file__).parent, ".movingPointSet.pts")

    def cost_fxn(self, fixed_image, transformed, mask):
        return None

    def write_elastix_kp(self, kp, fname):
        """
        Temporarily write fixed_xy and moving_xy to file

        Parameters
        ----------
        kp: ndarray
            (N, 2) numpy array of points (xy)

        fname: str
            Name of file in which to save the points
        """

        argfile = open(fname, 'w')
        npts = kp.shape[0]
        argfile.writelines(f"index\n{npts}\n")
        for i in range(npts):
            xy = kp[i]
            argfile.writelines(f"{xy[0]} {xy[1]}\n")

    def setup(self, moving, fixed, mask, initial_M=None, moving_xy=None, fixed_xy=None):
        """
        Create parameter map and initialize Reg

        Parameters
        ----------

        moving : ndarray
            Image to warp to align with fixed

        fixed : ndarray
            Image moving is warped to align to

        mask : ndarray
            2D array having non-zero pixel values, where values of 0 are ignnored during registration

        initial_M : (3x3) array
            Initial transformation matrix

        moving_xy : ndarray, optional
            (N, 2) array containing points in the moving image that correspond to those in the fixed image

        fixed_xy : ndarray, optional
            (N, 2) array containing points in the fixed image that correspond to those in the moving image
        """

        if initial_M is None:
            initial_M = np.eye(3)

        self.moving = moving
        self.fixed = fixed

        self.Reg = sitk.ElastixImageFilter()
        rigid_map = sitk.GetDefaultParameterMap('affine')

        rigid_map['NumberOfResolutions'] = [str(int(self.nlevels))]
        if self.transformation == "EuclideanTransform":
            rigid_map["Transform"] = ["EulerTransform"]
        else:
            rigid_map["Transform"] = ["SimilarityTransform"]

        rigid_map["Registration"] = ["MultiMetricMultiResolutionRegistration"]
        if moving_xy is not None and fixed_xy is not None:
            self.write_elastix_kp(fixed_xy, self.fixed_kp_fname)
            self.write_elastix_kp(moving_xy, self.moving_kp_fname)
            current_metrics = rigid_map["Metric"]
            current_metrics = list(current_metrics)
            current_metrics.append("CorrespondingPointsEuclideanDistanceMetric")
            rigid_map["Metric"] = current_metrics
            self.Reg.SetFixedPointSetFileName(self.fixed_kp_fname)
            self.Reg.SetMovingPointSetFileName(self.moving_kp_fname)

        rigid_map["Optimizer"] = [self.optimization]
        rigid_map["NumberOfHistogramBins"] = [str(self.nbins)]
        self.Reg.SetParameterMap(rigid_map)

        if mask is not None:
            self.Reg.SetFixedMask(sitk.GetImageFromArray(mask))

        sitk_moving = sitk.GetImageFromArray(moving)
        sitk_fixed = sitk.GetImageFromArray(fixed)
        self.Reg.SetMovingImage(sitk_moving)  # image to warp
        self.Reg.SetFixedImage(sitk_fixed)  # image to align with

    def calc_cost(self, p):
        return None

    def align(self, moving, fixed, mask, initial_M=None,
              moving_xy=None, fixed_xy=None):
        """
        Optimize rigid registration

        Parameters
        ----------
        moving : ndarray
            Image to warp to align with fixed

        fixed : ndarray
            Image moving is warped to align with

        mask : ndarray
            2D array having non-zero pixel values, where values of 0 are ignnored during registration

        initial_M : (3x3) array
            Initial transformation matrix

        moving_xy : ndarray, optional
            (N, 2) array containing points in the moving image that correspond to those in the fixed image

        fixed_xy : ndarray, optional
            (N, 2) array containing points in the fixed image that correspond to those in the moving image


        Returns
        -------
        aligned : (N,M) array
            Moving image warped to align with the fixed image

        M : (3,3) array
            Optimal transformation matrix

        cost_list : None
            None is returned because costs are not recorded

        """

        self.setup(moving, fixed, mask, initial_M, moving_xy, fixed_xy)
        self.Reg.Execute()

        # See section 2.6 in manual. This is the inverse transform.
        # Rotation is in radians
        tform_params = self.Reg.GetTransformParameterMap()[0]["TransformParameters"]
        if self.transformation == "EuclideanTransform":
            rotation, tx, ty = [eval(v) for v in tform_params]
            scale = 1.0
        else:
            scale, rotation, tx, ty = [eval(v) for v in tform_params]

        M = transform.SimilarityTransform(scale=scale, rotation=rotation,
                                          translation=(tx, ty)).params

        aligned = transform.warp(self.moving, M, order=3)

        # Clean up #
        if moving_xy is not None and fixed_xy is not None:
            if os.path.exists(self.fixed_kp_fname):
                os.remove(self.fixed_kp_fname)

            if os.path.exists(self.moving_kp_fname):
                os.remove(self.moving_kp_fname)

            tform_files = [f for f in os.listdir(".") if
                           f.startswith("TransformParameters.") and
                           f.endswith(".txt")]

            if len(tform_files) > 0:
                for f in tform_files:
                    os.remove(f)

        return aligned, M, None


class AffineOptimizerRMI(AffineOptimizer):
    def __init__(self,  r=6, nlevels=1, nbins=256, optimization="Powell", transform="euclidean"):
        super().__init__(nlevels, nbins, optimization, transform)
        self.r = r

    def cost_fxn(self, fixed_image, transformed, mask):
        r_ratio = self.r/np.min(self.pyramid_fixed[0].shape)
        level_rad = int(r_ratio*np.min(fixed_image.shape))
        if level_rad == 0:
            level_rad = 1

        return -region_mi(fixed_image, transformed, mask, r=level_rad)


class AffineOptimizerDisplacement(AffineOptimizer):
    def __init__(self, nlevels=1, nbins=256, optimization="Powell", transform="euclidean"):
        super().__init__(nlevels, nbins, optimization, transform)

    def cost_fxn(self, fixed_image, transformed, mask):

        return displacement(fixed_image, transformed, mask)


class AffineOptimizerKNN(AffineOptimizer):
    def __init__(self, nlevels=1, nbins=256, optimization="Powell", transform="euclidean"):
        super().__init__(nlevels, nbins, optimization, transform)
        self.HA_list = [None]*nlevels

    def shannon_entropy(self, X, k=1):
        """
        Adapted from https://pybilt.readthedocs.io/en/latest/_modules/pybilt/common/knn_entropy.html
        to use sklearn's KNN, which is much faster
        """

        from sklearn import neighbors
        from scipy.special import gamma, psi
        # Get distance to kth nearest neighbor
        knn = neighbors.NearestNeighbors(n_neighbors=k)
        knn.fit(X.reshape(-1, 1))
        r_k, idx = knn.kneighbors()
        lr_k = np.log(r_k[r_k > 0])
        d = 1
        if len(X.shape) == 2:
            d = X.shape[1]
        # volume of unit ball in d^n
        v_unit_ball = np.pi ** (0.5 * d) / gamma(0.5 * d + 1.0)
        n = len(X)
        H = psi(n) - psi(k) + np.log(v_unit_ball) + (np.float(d) / np.float(n)) * (lr_k.sum())

        return H

    def mutual_information(self, A, B):

        if self.HA_list[self.current_level] is None:
            # Only need to caluclate once per level, becuase the fixed
            # image doesn't change

            self.HA_list[self.current_level] = self.shannon_entropy(A)

        HA = self.HA_list[self.current_level]
        HB = self.shannon_entropy(B)

        joint = np.hstack([A, B])

        Hjoint = self.shannon_entropy(joint, k=2)

        MI = HA + HB - Hjoint
        if MI < 0:
            MI = 0
        return MI

    def cost_fxn(self, fixed_image, transformed, mask):
        if mask is not None:
            fixed_flat = fixed_image[mask != 0]
            transformed_flat = transformed[mask != 0]
        else:
            fixed_flat = fixed_image.reshape(-1)
            transformed_flat = transformed.reshape(-1)

        return -self.mutual_information(fixed_flat, transformed_flat)


class AffineOptimizerOffGrid(AffineOptimizer):
    def __init__(self, nlevels, nbins=256, optimization="Powell", transform="euclidean", spacing=5):
        super().__init__(nlevels, nbins, optimization, transform)
        self.spacing = spacing

    def setup(self, moving, fixed, mask, initial_M=None):
        AffineOptimizer.setup(self, moving, fixed, mask, initial_M)

        self.moving_interps = [self.get_interp(img)
                               for img in self.pyramid_moving]
        self.fixed_interps = [self.get_interp(img)
                              for img in self.pyramid_fixed]

        self.z_range = (min(np.min(self.moving[self.nlevels - 1]),
                        np.min(self.fixed[self.nlevels - 1])),
                        max(np.max(self.moving[self.nlevels - 1]),
                        np.max(self.fixed[self.nlevels - 1])))

        self.grid_spacings = [self.get_scpaing_for_levels(self.pyramid_fixed[i], self.spacing) for i in range(self.nlevels)]
        self.grid_flat = [self.get_regular_grid_flat(i)
                          for i in range(self.nlevels)]

    def get_scpaing_for_levels(self, img_shape, max_level_spacing):
        max_shape = self.pyramid_fixed[self.nlevels - 1].shape
        shape_ratio = np.mean([img_shape[0]/max_shape[0],
                               img_shape[0]/max_shape[0]])

        level_spacing = int(max_level_spacing*shape_ratio)
        if level_spacing == 0:
            level_spacing = 1

        return level_spacing

    def get_regular_grid_flat(self, level):
        sr, sc = np.meshgrid(np.arange(0, self.pyramid_fixed[level].shape[0],
                                       self.grid_spacings[level]),
                             np.arange(0, self.pyramid_fixed[level].shape[1],
                                       self.grid_spacings[level]))

        sr = sr.reshape(-1)
        sc = sc.reshape(-1)
        filtered_sr = sr[self.pyramid_mask[level][sr, sc] > 0]
        filtered_sc = sc[self.pyramid_mask[level][sr, sc] > 0]
        return (filtered_sr, filtered_sc)

    def get_interp(self, img):
        return interpolate.RectBivariateSpline(np.arange(0, img.shape[0], dtype=np.float), np.arange(0, img.shape[1], dtype=np.float), img)

    def interp_point(self, zr, zc, interp, z_range):
        z = np.array([interp(zr[i], zc[i])[0][0] for i in range(zr.size)])
        z[z < z_range[0]] = z_range[0]
        z[z > z_range[1]] = z_range[1]
        return z

    def calc_cost(self, p):

        transformation = make_transform(p)
        corners_rc = get_corners_of_image(self.pyramid_fixed[self.current_level].shape)
        warped_corners = warp_xy(corners_rc, transformation.params)
        if np.any(warped_corners < 0) or \
           np.any(warped_corners[:, 0] > self.pyramid_fixed[self.current_level].shape[0]) or \
           np.any(warped_corners[:, 1] > self.pyramid_fixed[self.current_level].shape[1]):
            return np.inf

        sr, sc = self.grid_flat[self.current_level]
        sample_r = sr + np.random.uniform(0, self.grid_spacings[self.current_level] / 2, sr.size)
        sample_c = sc + np.random.uniform(0, self.grid_spacings[self.current_level] / 2, sc.size)
        # Only sample points in mask
        warped_xy = warp_xy(np.dstack([sample_c, sample_r])[0], transformation.params)
        fixed_intensities = self.interp_point(warped_xy[:, 1], warped_xy[:, 0], self.fixed_interps[self.current_level], self.z_range)
        moving_intensities = self.interp_point(sample_r, sample_c, self.moving_interps[self.current_level], self.z_range)

        return self.cost_fxn(fixed_intensities, moving_intensities, self.pyramid_mask[self.current_level])

    def cost_fxn(self, fixed_intensities, transformed_intensities, mask):
        """
        """
        results, _, _ = np.histogram2d(fixed_intensities, transformed_intensities, bins=self.nbins)
        n = np.sum(results)

        results /= n
        x_margins = np.sum(results, axis=0)
        y_margins = np.sum(results, axis=1)

        H_A = entropy(x_margins)
        H_B = entropy(y_margins)
        H_AB = entropy(results.flatten())

        MI = (H_A + H_B) / H_AB
        if MI < 0:
            MI = 0

        return -MI
