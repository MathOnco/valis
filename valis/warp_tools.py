import multiprocessing
from scipy.optimize import fmin_l_bfgs_b
from scipy import ndimage, spatial
import shapely
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from skimage import draw, restoration, transform, filters, morphology
import tqdm
import cv2
from PIL import Image, ImageDraw
import numpy as np
import weightedstats
import warnings
import pyvips
from interpolation.splines import UCGrid, filter_cubic, eval_cubic
import SimpleITK as sitk
from colorama import Fore
import os
import re
from copy import deepcopy
from . import valtils

pyvips.cache_set_max(0)


def is_pyvips_22():
    pvips_ver = pyvips.__version__.split(".")
    pyvips_22 = eval(pvips_ver[0]) >= 2 and eval(pvips_ver[1]) >= 2
    return pyvips_22


def get_ref_img_idx(img_f_list, ref_img_name=None):
    """Get index of reference image

    Parameters
    ----------
    img_f_list : list of str
        List of image file names

    ref_img_name : str, optional
        Filename of image that will be treated as the center of the stack.
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
        ref_img_name = valtils.get_name(os.path.split(ref_img_name)[1])
        img_names = [valtils.get_name(f).lower() for f in img_f_list]
        name_matches = [re.search(ref_img_name.lower(), n) for n in img_names]
        ref_img_idx = [i for i in range(n_imgs) if name_matches[i] is not None]
        n_matches = len(ref_img_idx)

        if n_matches == 0:
            ref_img_idx = n_imgs // 2
            warning_msg = (f"No files in img_f_list match {ref_img_name}"
                           f"Returning middle image, which is {img_f_list[ref_img_idx]}")

            valtils.print_warning(warning_msg)

        elif n_matches == 1:
            ref_img_idx = ref_img_idx[0]

        elif n_matches > 1:
            macthing_files = ", ".join(img_f_list[i] for i in ref_img_idx)
            ref_img_idx = ref_img_idx[0]
            warning_msg = (f"More than 1 file in img_f_list matches {ref_img_name}. "
                           f"These files are: {macthing_files}. "
                           f"Returning first match, which is {img_f_list[ref_img_idx]}")

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

    if ref_img_idx is None:
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


def calc_memory_size_gb(shape, nchannels, np_dtype):
    """Estimate amount of space an image will take up, in Gb
    """

    bitdepth = "".join(re.findall(r'\d+', np_dtype))
    if len(bitdepth) > 0:
        bitdepth = eval(bitdepth)
    else:
        bitdepth = 1

    n_px = nchannels*np.multiply(*shape)
    gb = ((n_px*8)/bitdepth)/(2**30)

    return gb


def remove_invasive_displacements(bk_dxdy, M, src_shape_rc, out_shape_rc, inpaint_holes=False):
    """Remove displacements that would distort the image edges
    Finds areas where areas outside of the image get brought inside. Can
    happen if displacements are combined.

    Parameters
    ----------
    bk_dxdy : list
        Displacement fields [x, y]

    M : ndarray
        3x3 transformation matrix

    src_shape_rc : tuple
        Shape (row, col) of the image before affine transform

    Returns
    -------
    new_dxdy : list
        `bk_dxdy` but with invasive displacements set to 0

    """

    new_dx = bk_dxdy[0].copy()
    new_dy = bk_dxdy[1].copy()
    if M is not None:
        affine_mask = warp_img(np.full(src_shape_rc, 255, dtype=np.uint8), M, out_shape_rc=out_shape_rc, interp_method="nearest")
        if not np.all(out_shape_rc == bk_dxdy[0].shape):
            affine_mask = resize_img(affine_mask, bk_dxdy[0].shape, interp_method="nearest")
            new_dx[affine_mask == 0] = 0
            new_dy[affine_mask == 0] = 0

    else:
        affine_mask = np.full(out_shape_rc, 255, dtype=np.uint8)

    inv_mask = 255*(affine_mask == 0).astype(np.uint8)
    inv_nr = warp_img(inv_mask, bk_dxdy=bk_dxdy)
    out_to_in = ((inv_nr > 0) & (affine_mask > 0))

    selem = morphology.disk(3)
    out_to_in  = morphology.binary_dilation(out_to_in, selem)

    new_dy = bk_dxdy[1].copy()
    new_dx = bk_dxdy[0].copy()

    new_dx[out_to_in] = 0
    new_dy[out_to_in] = 0

    nr_img = np.round(warp_img(affine_mask, bk_dxdy=[new_dx, new_dy])).astype(np.uint8)

    holes_mask = ((nr_img == 0) & (affine_mask > 0))
    holes_mask = 255*(morphology.binary_dilation(holes_mask, selem)).astype(np.uint8)

    if inpaint_holes and holes_mask.max() > 0:
        new_dx = cv2.inpaint(new_dx.astype(np.float32), holes_mask, 3, cv2.INPAINT_TELEA)
        new_dy = cv2.inpaint(new_dy.astype(np.float32), holes_mask, 3, cv2.INPAINT_TELEA)
    else:
        new_dx[holes_mask > 0] = 0
        new_dy[holes_mask > 0] = 0

    new_dxdy = np.array([new_dx, new_dy])

    return new_dxdy


def rescale_img(img, scaling):
    is_array = False
    if not isinstance(img, pyvips.Image):
        is_array = True
        img = numpy2vips(img)

    resized = img.resize(scaling)
    if is_array:
        resized = vips2numpy(resized)

    return resized


def resize_img(img, out_shape_rc, interp_method="bicubic"):

    is_array = False
    if not isinstance(img, pyvips.Image):
        is_array = True
        img = numpy2vips(img)

    out_h, out_w = out_shape_rc

    src_shape_rc = np.array([img.height, img.width])
    sy, sx = (np.array(out_shape_rc)/src_shape_rc)
    S = [sx, 0, 0, sy]

    interpolator = pyvips.Interpolate.new(interp_method)
    resized = img.affine(S,
                         oarea=[0, 0, out_w, out_h],
                         interpolate=interpolator,
                         premultiplied=True
                         )

    if is_array:
        resized = vips2numpy(resized)

    return resized


def scale_dxdy(dxdy, out_shape_rc):
    if isinstance(dxdy, np.ndarray):
        vips_dxdy = numpy2vips(np.dstack(dxdy))
    else:
        vips_dxdy = dxdy

    sxy = (np.array(out_shape_rc)/np.array([vips_dxdy.height, vips_dxdy.width]))[::-1]
    scaled_dx = float(sxy[0])*vips_dxdy[0]
    scaled_dy = float(sxy[1])*vips_dxdy[1]
    scaled_dxdy = scaled_dx.bandjoin(scaled_dy)
    scaled_dxdy = resize_img(scaled_dxdy, out_shape_rc)

    return scaled_dxdy


def get_src_img_shape_and_M(M, transformation_src_shape_rc, transformation_dst_shape_rc, dst_shape_rc):
    """Determine the size of an image that, when warped, will have the same relative position
    as in the original transformation dst image.

    For exmample, used to determine how large a source image needs to be in order to
    be warped to land in the an image with shape dst_shape_rc.

    Parameters
    ----------

    M : ndarray
         3x3 affine transformation matrix to perform rigid warp on image with
         shape `transformation_src_shape_rc`. The shape of this warped image
         would be  `transformation_dst_shape_rc`.

    transformation_dst_shape_rc : (int, int)
        Shape of the image with shape `transformation_src_shape_rc` after warping.
        This could be the shape of the original image after applying `M`.

    src_shape_rc : (int, int)
        Shape of the image from which the points originated. For example,
        this could be a larger/smaller version of the image that was
        used for feature detection.

    dst_shape_rc : (int, int)
        Shape of image (with shape `src_shape_rc`) after warping

    Returns
    -------
    src_shape_rc : (int, int)
        Shape of scaled image that can warped to an image with shape `dst_shape_rc`

    scaled_M : ndarray
        A scaled version of `M` that will warp an image with shape `src_shape_rc`

    """

    img_corners_xy = get_corners_of_image(transformation_src_shape_rc)[:, ::-1]
    warped_corners = warp_xy(img_corners_xy, M=M,
                             transformation_src_shape_rc=transformation_src_shape_rc,
                             transformation_dst_shape_rc=transformation_dst_shape_rc
                             )

    dst_sxy = (np.array(dst_shape_rc)/np.array(transformation_dst_shape_rc))[::-1]
    scaled_warped_corners = dst_sxy*warped_corners
    scaled_M = scale_M(M, *dst_sxy)

    scaled_unwarped_corners = warp_xy(scaled_warped_corners, M=np.linalg.inv(scaled_M))
    src_slide_bbox = xy2bbox(scaled_unwarped_corners)
    src_shape_rc = np.round(src_slide_bbox[2:] + src_slide_bbox[:2]).astype(int)

    return src_shape_rc, scaled_M


def save_img(dst_f, img, thumbnail_size=None):
    """Save an image using pyvips

    Parameters
    ----------
    dst_f : str
        Filename for saved image

    img : ndarray, pyvips.Image
        Image to be saved. Numpy arrays will be converted to pvips.Image

    thumbnail_size : optional, int
        If not None, the image will be resized to fit within this size

    """
    if not isinstance(img, pyvips.Image):
        vips_img = numpy2vips(img)
    else:
        vips_img = img

    if thumbnail_size is not None:
        vips_wh = np.array([vips_img.width, vips_img.height])
        s = np.min(thumbnail_size/vips_wh)
        if s < 1:
            out_img = vips_img.resize(s)
        else:
            out_img = vips_img
    else:
        out_img = vips_img

    out_img.write_to_file(dst_f)


def get_pts_in_bbox(xy, xywh):
    x0, y0 = xywh[0:2]
    x1, y1 = xywh[0:2] + xywh[2:]
    in_bbox_idx = np.where((xy[:, 0] >= x0)  & (xy[:, 0] < x1) & (xy[:, 1] >= y0)  & (xy[:, 1] < y1)==True)[0]
    xy_in_bbox = xy[in_bbox_idx]
    return xy_in_bbox, in_bbox_idx


def get_img_dimensions(img_f):
    """
    Get image dimensions (width, height) without opening file

    Parameters
    ----------
    img_f: str
        Path to image

    Returns
    -------
    img_dims : [(w, h)]
        Image dimensions (width, height)

    """
    img = Image.open(img_f)
    return img.size[0:2]


def get_shape(img):
    """ Get shape of image (row, col, nchannels)

    Parameters
    ----------

    img : numpy.array, pyvips.Image
        Image to get shape of

    Returns
    -------
    shape_rc : numpy.array
        Number of rows and columns and channels in the image

    """

    if isinstance(img, pyvips.Image):
        shape_rc = np.array([img.height, img.width])
        ndim = img.bands
    else:
        shape_rc = np.array(img.shape[0:2])

        if img.ndim > 2:
            ndim = img.shape[2]
        else:
            ndim = 1

    shape = np.array([*shape_rc, ndim])

    return shape


def apply_mask(img, mask):
    """Mask an image

    """
    mask_is_vips = isinstance(mask, pyvips.Image)
    if not mask_is_vips:
        vips_mask = numpy2vips(mask)
    else:
        vips_mask = mask

    img_is_vips = isinstance(img, pyvips.Image)
    if not img_is_vips:
        vips_img = numpy2vips(img)
    else:
        vips_img = img.copy()

    masked_img = (vips_mask == 0).ifthenelse(0, vips_img)

    if not img_is_vips:
        masked_img = vips2numpy(masked_img)

    return masked_img


def get_grid_bboxes(shape_rc, bbox_w, bbox_h, inclusive=False):
    """
    Get list of bbox xywh for an image with shape shape_rc. Returned array ordered such that the bounding boxes go
    left to right, top to bottom, starting at the top left of the image (r=0, c=0)

    Parameters
    ----------
    shape_rc: (n_row, n_col)
        Shape of the image

    bbox_w: int
        Width of each bounding box

    bbox_h: int
        Height of each bounding box

    inclusive: bool
        If True, bbox_list will inclide boxes that go to edege of image, even if their width/height is smaller than
        bbox_w or bbox_h. Default is False.

    Returns
    -------
    bbox_list : [N, 4] array
        Array containing the top left xy coordinates, width, height of each bounding box. Bounding boxes go from
        left to right, top to bottom.

    Example
    --------
    img_shape = (100, 200)
    bbox_w = 20
    bbox_h = 20

    bbox_list = get_grid_bboxes(img_shape, bbox_w, bbox_h)

    """

    temp_x = np.arange(0, shape_rc[1], bbox_w)
    temp_y = np.arange(0, shape_rc[0], bbox_h)

    if inclusive:
        if shape_rc[1] not in temp_x:
            temp_x = np.hstack([temp_x, shape_rc[1]])

        if shape_rc[0] not in temp_y:
            temp_y = np.hstack([temp_y, shape_rc[0]])

    tl_y, tl_x = np.meshgrid(temp_y, temp_x, indexing="ij")
    bbox_list = [[tl_x[i, j],
                  tl_y[i, j],
                  tl_x[i+1, j+1] - tl_x[i, j],
                  tl_y[i+1, j+1] - tl_y[i, j]]
                 for i in range(len(temp_y)-1)
                 for j in range(len(temp_x)-1)]

    return np.array(bbox_list)


def expand_bbox(bbox_xywh, expand, shape_rc=None):
    new_xy = bbox_xywh[0:2] - expand
    new_xy[new_xy < 0] = 0
    new_x, new_y = new_xy

    new_w, new_h = bbox_xywh[2:] + 2*expand

    if shape_rc is not None:
        h, w = shape_rc
        if new_x + new_w >= w:
            new_w = w - new_x

        if new_y + new_h >= h:
            new_h = h - new_y

    return np.array([*new_xy, new_w, new_h])


def stitch_tiles(tile_list, tile_bboxes, nrow, ncol, overlap):
    """
    #. Blend across row, added tiles to the right edge
    #. Blend each row to bottom of the one above
    """

    is_array = False
    if not isinstance(tile_list[0], pyvips.Image):
        is_array = True
        tile_list = [numpy2vips(tile) for tile in tile_list]

    row_mosaics = [None] * nrow
    col_range = range(0, ncol)
    for i in range(nrow):
        col_tiles = [tile_list[index2d_to_1d(i, j, ncol=ncol)] for j in col_range]
        row_mosaic = col_tiles[0]

        for j in range(1, ncol):
            tile_idx = index2d_to_1d(i, j, ncol)

            # Get offset of where to merge right tile
            right_bbbox = tile_bboxes[tile_idx]
            left_idx = tile_idx - 1
            left_bbox = tile_bboxes[left_idx]
            left_tile_br = left_bbox[2:] + left_bbox[:2]
            x_offset, _ = left_tile_br - right_bbbox[:2]
            offset = x_offset - row_mosaic.width
            right_tile = col_tiles[j]

            row_mosaic = row_mosaic.merge(right_tile, "horizontal", offset, 0, mblend=overlap)
        row_mosaics[i] = row_mosaic

    stitched = row_mosaics[0]
    for i in range(1, nrow):
        bottom_idx = index2d_to_1d(i, 0, ncol)
        bottom_bbbox = tile_bboxes[bottom_idx]

        top_bbox = tile_bboxes[bottom_idx - ncol]
        top_br = top_bbox[2:] + top_bbox[:2]
        _, y_offset = top_br - bottom_bbbox[:2]
        v_offset = y_offset - stitched.height

        bottom = row_mosaics[i]
        stitched = stitched.merge(bottom, "vertical", 0, v_offset, mblend=overlap)

    if is_array:
        stitched = vips2numpy(stitched)

    return stitched


def index2d_to_1d(row, col, ncol):
    idx = (ncol*row) + col

    return idx


def index1d_to_2d(idx, ncol):
    row = idx // ncol
    col = idx % ncol

    return row, col


def get_triangular_mesh(x_pos, y_pos):
    """Get a triangular mesh

    Parameters
    ----------
    x_pos : ndarray
        X-positions of each vertex

    y_pos : int
        Y-positions of each vertex

    Returns
    -------
    tri_verts : ndarray
        X-Y coordinates of vertices

    tri_faces : ndarray
        Indices of the vertices of each mesh face

    """

    tl_y, tl_x = np.meshgrid(y_pos, x_pos, indexing="ij")
    grid_boxes_wh = [[tl_x[i, j],
                    tl_y[i, j],
                    tl_x[i+1, j+1] - tl_x[i, j],
                    tl_y[i+1, j+1] - tl_y[i, j]]
                    for i in range(len(y_pos)-1)
                    for j in range(len(x_pos)-1)]

    grid_boxes_xy = [bbox2xy(wh) for wh in grid_boxes_wh]
    vert_dict = {}
    tri_faces = []
    current_max_vert_id = 0
    for bbox_xy in grid_boxes_xy:
        bbox = xy2bbox(bbox_xy)
        bbox_center_xy = tuple(bbox[0:2] + bbox[2:]/2)
        bbox_tuples = [tuple(xy) for xy in bbox_xy]
        for vert in bbox_tuples:
            if not vert in vert_dict:
                vert_dict[vert] = current_max_vert_id
                current_max_vert_id += 1

        vert_dict[bbox_center_xy] = current_max_vert_id
        current_max_vert_id += 1

        # 4 triangles in bbox. Bbbox : 0=TL, 1=TR, 2=BR, 3=BL #
        # Each sorted clockwise, with A= being most top left
        left_face = [vert_dict[bbox_tuples[0]],
                    vert_dict[bbox_center_xy],
                    vert_dict[bbox_tuples[3]]]

        top_face = [vert_dict[bbox_tuples[0]],
                        vert_dict[bbox_tuples[1]],
                        vert_dict[bbox_center_xy]]

        right_face = [vert_dict[bbox_center_xy],
                        vert_dict[bbox_tuples[1]],
                        vert_dict[bbox_tuples[2]]]

        btm_face = [vert_dict[bbox_center_xy],
                    vert_dict[bbox_tuples[2]],
                    vert_dict[bbox_tuples[3]]]

        tri_faces.extend([left_face, top_face, right_face, btm_face])


    temp_tri_verts = list(vert_dict.keys())
    tri_verts = np.array([temp_tri_verts[i] for i in vert_dict.values()])
    tri_faces = np.array(tri_faces)

    return tri_verts, tri_faces


def mattes_mi(img1, img2, nbins=50,  mask=None):
    """Measure Mattes mutual information between 2 images.

    Parameters
    ----------
    img1 : ndarray
        First image with shape (N, M)

    img1 : ndarray
        Second image with shape (N, M)

    nbins : int
        Number of histogram bins

    mask : ndarray, None
        Mask with shape (N, M) that indiates where the metric
        should be calulated. If None, the metric will be calculated
        for all NxM pixels.

    Returns
    -------
    mmi : float
        Mattes mutation inormation

    """

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricSamplingStrategy(reg.NONE)
    reg.SetInitialTransform(sitk.Transform(2, sitk.sitkIdentity))
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=nbins)
    if mask is not None:
        sitk_mask = sitk.GetImageFromArray(mask)
        reg.SetMetricFixedMask(sitk_mask)
        reg.SetMetricMovingMask(sitk_mask)

    if not np.issubdtype(img1.dtype, np.floating):
        img1 = img1.astype(float)

    if not np.issubdtype(img2.dtype, np.floating):
        img2 = img2.astype(float)

    mmi = reg.MetricEvaluate(sitk.GetImageFromArray(img1), sitk.GetImageFromArray(img2))

    return -1*mmi


def calc_rotated_shape(w, h, degree):
    ### https://stackoverflow.com/questions/3231176/how-to-get-size-of-a-rotated-rectangle

    rad = np.deg2rad(degree)
    new_w = np.abs(w * np.cos(rad)) + np.abs(h * np.sin(rad))
    new_h = np.abs(w * np.sin(rad)) + np.abs(h * np.cos(rad))


    return new_w, new_h


def order_points(pts_xy):
    """
    Order points in clockwise order (TL, TR, BR, BL)
    https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/

    Parameters
    ----------
    pts_xy : [N, 2] array
        Points to order clockwise, in xy coordinates

    Returns
    -------
    cw_pts_xy : [N, 2] array
    Points ordered clockwise, in xy coordinates

    """

    ### https://math.stackexchange.com/questions/978642/how-to-sort-vertices-of-a-polygon-in-counter-clockwise-order

    # warnings.warn("Outpout is now clockwise. May need update functions that call this")
    # sort the points based on their x-coordinates
    xSorted = pts_xy[np.argsort(pts_xy[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = spatial.distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    cw_pts_xy = np.array([tl, tr, br, bl], dtype="float32")

    return cw_pts_xy


def get_resize_M(in_shape_rc, out_shape_rc):

    in_corners = get_corners_of_image(in_shape_rc)
    out_corners = get_corners_of_image(out_shape_rc)
    sy, sx = out_corners[2]/in_corners[2]

    resize_M = np.identity(3)
    resize_M[0, 0] = sx
    resize_M[1, 1] = sy

    return resize_M


def get_corners_of_image(shape_rc):
    """
    Get corners of image in clockwise order (TL, TR, BR, BL)

    Parameters
    ----------
    shape_rc : (int, int)
        Chape of image that corners come from

    Returns
    -------
    corners_rc : 4 x 2 array
        Array with positions of each corner, sorted clockwise, and in row-col coordinates

    """

    max_x = shape_rc[1]
    max_y = shape_rc[0]
    bl = [0, 0]
    br = [max_x, 0]
    tl = [0, max_y]
    tr = [max_x, max_y]

    corners = np.array([bl, br, tr, tl])
    corners_rc = corners[:, ::-1]

    return corners_rc


def _numpy2vips_pre_22(a):
    """
    From https://stackoverflow.com/questions/61138272/efficiently-saving-tiles-to-a-bigtiff-image

    """
    dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex',
    }

    if a.ndim > 2:
        height, width, bands = a.shape
    else:
        height, width = a.shape
        bands = 1

    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      dtype_to_format[str(a.dtype)])
    return vi


def _vips2numpy_pre_22(vi):
    """
    https://github.com/libvips/pyvips/blob/master/examples/pil-numpy-pyvips.py

    """
    format_to_dtype = {
        'uchar': np.uint8,
        'char': np.int8,
        'ushort': np.uint16,
        'short': np.int16,
        'uint': np.uint32,
        'int': np.int32,
        'float': np.float32,
        'double': np.float64,
        'complex': np.complex64,
        'dpcomplex': np.complex128,
    }

    img = np.ndarray(buffer=vi.write_to_memory(),
                      dtype=format_to_dtype[vi.format],
                      shape=[vi.height, vi.width, vi.bands])
    if vi.bands == 1:
        img = img[..., 0]

    return img


def _vips2numpy_22(vi):
    img = vi.numpy()

    return img


def _numpy2vips_22(a):
    vi = pyvips.Image.new_from_array(a)

    return vi


def numpy2vips(a):

    if is_pyvips_22():
        vi = _numpy2vips_22(a)
    else:
        vi = _numpy2vips_pre_22(a)

    return vi

def vips2numpy(vi):
    if is_pyvips_22():
        a = _vips2numpy_22(vi)
    else:
        a = _vips2numpy_pre_22(vi)

    return a


def pad_img(img, padded_shape):
    padding_T = get_padding_matrix(img.shape[0:2], padded_shape)
    padded_img = warp_img(img, padding_T, out_shape_rc=padded_shape)

    return padded_img, padding_T

def warp_img(img, M=None, bk_dxdy=None, out_shape_rc=None,
             transformation_src_shape_rc=None,
             transformation_dst_shape_rc=None,
             bbox_xywh=None,
             bg_color=None,
             interp_method="bicubic"):
    """Warp an image using rigid and/or non-rigid transformations

    Warp an image using the trasformations defined by `M` and the optional
    displacement field, `bk_dxdy`. Transformations will be scaled so that
    they can be applied to the image.

    Parameters
    ----------
    img : ndarray, optional
        Image to be warped

    M : ndarray, optional
        3x3 Affine transformation matrix to perform rigid warp

    bk_dxdy : ndarray, optional
        A list containing the backward x-axis (column) displacement,
        and y-axis (row) displacement applied after the rigid transformation.

    out_shape_rc : tuple of int
        Shape of the `img` after warping.

    transformation_src_shape_rc : tuple of int
        Shape of image that was used to find the transformations M and/or `bk_dxdy`.
        For example, this could be the original image in which features
        were detected

    transformation_dst_shape_rc : tuple of int
        Shape of image with shape transformation_src_shape_rc after
        being warped. Should be specified if `img` is a rescaled
        version of the image for which the `M` and `bk_dxdy` were found.

    bbox_xywh : tuple
        Bounding box to crop warped image. Should be in reference to the image
        with shape = `out_shape_rc`, which may or not be the same as
        `transformation_dst_shape_rc`. For example, to crop a region
        from a large warped slide, `bbox_xywh` should refer to an area
        in that warped slide, not an area in the image used to find the
        transformation.

    bg_color : optional, list
        Background color, if `None`, then the background color will be black

    interp_method : str, optional

    Returns
    -------
    warped : ndarray, pyvips.Image
        Warped version of `img`

    """

    is_array = False
    if not isinstance(img, pyvips.Image):
        is_array = True
        img = numpy2vips(img)

    src_shape_rc = np.array([img.height, img.width])
    if transformation_src_shape_rc is None:
        transformation_src_shape_rc = src_shape_rc

    # Determine shape of unscaled output. If not provided, find shape big enough to avoid cropping
    if transformation_dst_shape_rc is None:
        if bk_dxdy is not None:
            if isinstance(bk_dxdy, pyvips.Image):
                transformation_dst_shape_rc = np.array([bk_dxdy.height, bk_dxdy.width])
            else:
                transformation_dst_shape_rc = bk_dxdy[0].shape
        elif out_shape_rc is not None:
            transformation_dst_shape_rc = out_shape_rc
        else:
            transformation_src_corners_rc = get_corners_of_image(transformation_src_shape_rc)
            warped_transformation_src_corners_xy = warp_xy(transformation_src_corners_rc[:, ::-1], M)
            transformation_dst_shape_rc = np.ceil(np.max(warped_transformation_src_corners_xy[:, ::-1], axis=0)).astype(int)

    # Determine shape of scaled output
    if out_shape_rc is None:
        out_shape_rc = transformation_dst_shape_rc

    src_shape_rc = np.array(src_shape_rc)
    transformation_src_shape_rc = np.array(transformation_src_shape_rc)
    out_shape_rc = np.array(out_shape_rc)
    transformation_dst_shape_rc = np.array(transformation_dst_shape_rc)

    src_sxy, dst_sxy, displacement_sxy, displacement_shape_rc = get_warp_scaling_factors(
                                                                     transformation_src_shape_rc=transformation_src_shape_rc,
                                                                     transformation_dst_shape_rc=transformation_dst_shape_rc,
                                                                     src_shape_rc=src_shape_rc, dst_shape_rc=out_shape_rc,
                                                                     bk_dxdy=bk_dxdy)
    if bbox_xywh is not None:
        do_crop = True
    else:
        do_crop = False

    # Determine if any transformations need to be done
    if M is not None:
        do_rigid = True
    else:
        do_rigid = False

    if bk_dxdy is not None:
        do_non_rigid = True
    else:
        do_non_rigid = False

    if not any([do_rigid, do_non_rigid, do_crop]):
        if is_array:
            img = vips2numpy(img)
        return img

    # Do transformations
    if bg_color is None:
        bg_color = [0] * img.bands
        bg_extender = pyvips.enums.Extend.BLACK
    else:
        bg_extender = pyvips.enums.Extend.BACKGROUND
        bg_color = list(bg_color)

    interpolator = pyvips.Interpolate.new(interp_method)
    if do_rigid:
        if not np.all(src_sxy == 1):

            img_corners_xy = get_corners_of_image(src_shape_rc)[:, ::-1]
            warped_corners = warp_xy(img_corners_xy, M=M,
                                     transformation_src_shape_rc=transformation_src_shape_rc,
                                     transformation_dst_shape_rc=transformation_dst_shape_rc,
                                     src_shape_rc=src_shape_rc,
                                     dst_shape_rc=out_shape_rc)
            M_tform = transform.ProjectiveTransform()
            M_tform.estimate(warped_corners, img_corners_xy)
            warp_M = M_tform.params

        else:
            warp_M = M

        tx, ty = warp_M[:2, 2]
        warp_M = np.linalg.inv(warp_M)
        vips_M = warp_M[:2, :2].reshape(-1).tolist()
        affine_warped = img.affine(vips_M,
            oarea=[0, 0, out_shape_rc[1], out_shape_rc[0]],
            interpolate=interpolator,
            idx=-tx,
            idy=-ty,
            premultiplied=True,
            background=bg_color,
            extend=bg_extender
            )
    else:
        affine_warped = img

    if do_non_rigid:
        # Scale dxdy map
        if not isinstance(bk_dxdy, pyvips.Image):
            temp_dxdy = numpy2vips(np.dstack(bk_dxdy))
        else:
            temp_dxdy = bk_dxdy

        if dst_sxy is not None:
            scaled_dx = float(dst_sxy[0]) * temp_dxdy[0]
            scaled_dy = float(dst_sxy[1]) * temp_dxdy[1]
            vips_dxdy = scaled_dx.bandjoin(scaled_dy)
        else:
            vips_dxdy = temp_dxdy

        if dst_sxy is not None:
            S = [dst_sxy[0], 0, 0, dst_sxy[1]]
        else:
            S = [1.0, 0.0, 0.0, 1.0]


        warp_dxdy = vips_dxdy.affine(S,
                        oarea=[0, 0, out_shape_rc[1], out_shape_rc[0]],
                        interpolate=interpolator,
                        premultiplied=True)

        index = pyvips.Image.xyz(affine_warped.width, affine_warped.height)
        warp_index = (index[0] + warp_dxdy[0]).bandjoin(index[1] + warp_dxdy[1])

        try:
            #Option to set backround color in mapim added in libvips 8.13
            warped = affine_warped.mapim(warp_index,
                premultiplied=True,
                background=bg_color,
                extend=bg_extender,
                interpolate=interpolator)

        except pyvips.error.Error:
            warped = affine_warped.mapim(warp_index, interpolate=interpolator)
            if bg_color is not None:
                warped = (warped == 0).ifthenelse(bg_color, warped)

    else:
        warped = affine_warped

    if bbox_xywh is not None:
            warped = warped.extract_area(*bbox_xywh)

    if is_array:
        warped = vips2numpy(warped)

    return warped


def warp_img_inv(img, M=None, fwd_dxdy=None, transformation_src_shape_rc=None, transformation_dst_shape_rc=None, src_shape_rc=None, bk_dxdy=None, bg_color=None, interp_method="bicubic"):
    """Unwarp an image using rigid and/or non-rigid transformations

    Unwarp an image using the trasformations defined by `M` and the optional
    displacement field, `bk_dxdy`. This is accomplished by inverting `M` and
    using the "foward" displacements in `fwd_dxdy`. If `fwd_dxdy` is not provided,
    `bk_dxdy` will be inverted. Transformations will be scaled so that they can be applied to the images
    with different sizes.

    Parameters
    ----------
    img : ndarray, optional
        Image to be warped

    M : ndarray, optional
        3x3 Affine transformation matrix to perform rigid warp

    fwd_dxdy : ndarray, optional
        A list containing the forward x-axis (column) displacement,
        and y-axis (row) displacements.

    transformation_src_shape_rc : tuple of int
        Shape of image that was used to find the transformations M and/or `bk_dxdy`.
        For example, this could be the original image in which features
        were detected

    transformation_dst_shape_rc : tuple of int
        Shape of image with shape transformation_src_shape_rc after
        being warped. Should be specified if `img` is a rescaled
        version of the image for which the `M` and `bk_dxdy` were found.

    src_shape_rc : tuple of int
        Shape of the `img` before warping.

    bg_color : optional, list
        Background color, if `None`, then the background color will be black

    interp_method : str, optional

    Returns
    -------
    warped : ndarray, pyvips.Image
        Warped version of `img`

    """

    do_non_rigid = bk_dxdy is not None or fwd_dxdy is not None
    do_rigid = M is not None

    if not do_rigid and not do_non_rigid:
        return img

    is_array = False
    if not isinstance(img, pyvips.Image):
        is_array = True
        img = numpy2vips(img)

    warped_src_shape_rc = np.array([img.height, img.width])
    if transformation_dst_shape_rc is None:
        transformation_dst_shape_rc = warped_src_shape_rc

    src_sxy, dst_sxy, displacement_sxy, displacement_shape_rc = get_warp_scaling_factors(transformation_src_shape_rc=transformation_src_shape_rc,
                                                                     transformation_dst_shape_rc=transformation_dst_shape_rc,
                                                                     src_shape_rc=src_shape_rc, dst_shape_rc=warped_src_shape_rc,
                                                                     bk_dxdy=bk_dxdy, fwd_dxdy=fwd_dxdy)

    # Do transformations
    if bg_color is None:
        bg_color = [0] * img.bands
        bg_extender = pyvips.enums.Extend.BLACK
    else:
        bg_extender = pyvips.enums.Extend.BACKGROUND
        bg_color = list(bg_color)

    interpolator = pyvips.Interpolate.new(interp_method)
    # Undo non-rigid transformation #
    if do_non_rigid:
        if bk_dxdy is not None and fwd_dxdy is None:
            fwd_dxdy = get_inverse_field(bk_dxdy)

        if not isinstance(fwd_dxdy, pyvips.Image):
            temp_dxdy = numpy2vips(np.dstack(fwd_dxdy))
        else:
            temp_dxdy = fwd_dxdy

        if dst_sxy is not None:
            scaled_dx = float(dst_sxy[0]) * temp_dxdy[0]
            scaled_dy = float(dst_sxy[1]) * temp_dxdy[1]
            vips_dxdy = scaled_dx.bandjoin(scaled_dy)
        else:
            vips_dxdy = temp_dxdy

        if dst_sxy is not None:
            S = [dst_sxy[0], 0, 0, dst_sxy[1]]
        else:
            S = [1.0, 0.0, 0.0, 1.0]

        warp_dxdy = vips_dxdy.affine(S,
                        oarea=[0, 0, img.width, img.height],
                        interpolate=interpolator,
                        premultiplied=True)

        index = pyvips.Image.xyz(img.width, img.height)
        warp_index = (index[0] + warp_dxdy[0]).bandjoin(index[1] + warp_dxdy[1])

        try:
            #Option to set backround color in mapim added in libvips 8.13
            nr_warped = img.mapim(warp_index,
                premultiplied=True,
                background=bg_color,
                extend=bg_extender,
                interpolate=interpolator)

        except pyvips.error.Error:
            nr_warped = img.mapim(warp_index, interpolate=interpolator)
            if bg_color is not None:
                nr_warped = (nr_warped == 0).ifthenelse(bg_color, nr_warped)

    else:
        nr_warped = img

    if do_rigid:

        img_corners_xy = get_corners_of_image(src_shape_rc)[:, ::-1]
        warped_corners = warp_xy(img_corners_xy, M=M,
                                    transformation_src_shape_rc=transformation_src_shape_rc,
                                    transformation_dst_shape_rc=transformation_dst_shape_rc,
                                    src_shape_rc=src_shape_rc,
                                    dst_shape_rc=warped_src_shape_rc)
        M_tform = transform.ProjectiveTransform()
        M_tform.estimate(img_corners_xy, warped_corners)
        warp_M = M_tform.params

        tx, ty = warp_M[:2, 2]
        warp_M = np.linalg.inv(warp_M)
        vips_M = warp_M[:2, :2].reshape(-1).tolist()
        warped = nr_warped.affine(vips_M,
                    oarea=[0, 0, src_shape_rc[1], src_shape_rc[0]],
                    interpolate=interpolator,
                    idx=-tx,
                    idy=-ty,
                    premultiplied=True,
                    background=bg_color,
                    extend=bg_extender
                    )

    else:
        warped = nr_warped


    if is_array:
        warped = vips2numpy(warped)

    return warped


def warp_img_from_to(img, from_M=None, from_transformation_src_shape_rc=None,
                   from_transformation_dst_shape_rc=None,
                   from_dst_shape_rc=None, from_bk_dxdy=None,
                   to_M=None, to_transformation_src_shape_rc=None,
                   to_transformation_dst_shape_rc=None, to_src_shape_rc=None,
                   to_bk_dxdy=None, to_fwd_dxdy=None, bg_color=None, interp_method="bicubic"):
    """Warp image onto another

    Warps `img` to registered coordinates using the "from" parameters, and then uses
    the inverse "to" parameters to warp that image to the "to" image's coordinate system.
    Can be useful for transfering annotations from one image to another.

    Note: If `img` is a labeled image, it is recommended to set `interp_method` to "nearest"

    Parameters
    ----------
    xy : ndarray
        [P, 2] array of xy coordinates for P points

    from_M : ndarray, optional
         3x3 affine transformation matrix to perform rigid warp in the "from" image

    from_transformation_src_shape_rc : (int, int)
        Shape of image that was used to find the transformation in the
        "from" image. For example, this could be the original image in
        which features were detected in the "from" image.

    from_transformation_dst_shape_rc : (int, int)
        Shape (row, col) of registered image. As the "from"  and "to" images have been registered,
        this shape should be the same for both images.

    from_src_shape_rc : optional, (int, int)
        Shape of the unwarped image from which the points originated. For example,
        this could be a larger/smaller version of the "from" image that was
        used for feature detection.

    from_dst_shape_rc : optional, (int, int)
        Shape of from image (with shape `src_shape_rc`) after warping

    from_bk_dxdy : ndarray
        (2, N, M) numpy array of pixel displacements in the x and y in the "from" image.
        dx = bk_dxdy[0], and dy=bk_dxdy[1].

    from_fwd_dxdy : ndarray
        Inverse of `from_bk_dxdy`

    to_M : ndarray, optional
        3x3 affine transformation matrix to perform rigid warp in the "to" image

    to_transformation_src_shape_rc :  optional, (int, int)
        Shape of "to" image that was used to find the transformations.
        For example, this could be the original image in which features were detected

    to_src_shape_rc : optional, (int, int)
        Shape of the unwarped "to" image to which the points will be warped. For example,
        this could be a larger/smaller version of the "to" image that was
        used for feature detection.

    to_bk_dxdy : ndarray
        (2, N, M) numpy array of pixel displacements in the x and y in the "to" image.
        dx = bk_dxdy[0], and dy=bk_dxdy[1].

    to_fwd_dxdy : ndarray
        Inverse of `to_bk_dxdy`

    bg_color : optional, list
        Background color, if `None`, then the background color will be black

    interp_method : str, optional

    Returns
    -------
    in_target_space : ndarray, pvips.Image
        `img` warped onto the "to" image

    """


    in_reg_space = warp_img(img,
                            M=from_M,
                            bk_dxdy=from_bk_dxdy,
                            out_shape_rc=from_dst_shape_rc,
                            transformation_src_shape_rc=from_transformation_src_shape_rc,
                            transformation_dst_shape_rc=from_transformation_dst_shape_rc,
                            bg_color=bg_color,
                            interp_method=interp_method
                            )

    in_target_space = warp_img_inv(img=in_reg_space,
                                   M=to_M,
                                   fwd_dxdy=to_fwd_dxdy,
                                   transformation_src_shape_rc=to_transformation_src_shape_rc,
                                   transformation_dst_shape_rc=to_transformation_dst_shape_rc,
                                   src_shape_rc=to_src_shape_rc,
                                   bk_dxdy=to_bk_dxdy,
                                   bg_color=bg_color,
                                   interp_method=interp_method
                                   )

    return in_target_space


def crop_img(img, xywh):
    is_array = False
    if not isinstance(img, pyvips.Image):
        is_array = True
        img = numpy2vips(img)

    wh = np.round(xywh[2:]).astype(int)
    cropped = img.extract_area(*xywh[:2], *wh)
    if is_array:
        cropped = vips2numpy(cropped)

    return cropped


def get_warp_map(M=None, dxdy=None, transformation_dst_shape_rc=None,
                 dst_shape_rc=None, transformation_src_shape_rc=None,
                 src_shape_rc=None, return_xy=False):
    """Get map to warp an image
    Get a coordinate map that will perform the warp defined by M and the optional displacement field, dxdy
    Map can be scaled so that it can be applied to an image with shape unwarped_out_shape_rc
    Result is returned as a pyvips.Image, but it can be converted to a numpy array.

    Parameters
    ----------
    M : ndarray, optional
        3x3 Affine transformation matrix to perform rigid warp

    dxdy : ndarray, optional
        A list containing the x-axis (column) displacement, and y-axis (row)
        displacement. Will be applied after `M` (if available)

    transformation_dst_shape_rc : tuple of int
        Shape of the image with shape transformation_src_shape_rc after warping.
        This could be the shape of the original image after being warped

    dst_shape_rc : tuple of int, optional
        Shape of image (with shape out_shape_rc) after warping

    transformation_src_shape_rc : tuple of int
        Shape of image that was used to find the transformation.
        For example, this could be the original image in which features were detected

    src_shape_rc : tuple of int, optional
        Shape of the image to which the transform will be applied. For example, this could be a larger/smaller
        version of the image that was used for feature detection.


    Returns
    -------
    coord_map : ndarry
        A 2band numpy array that has location of each pixel in
        `src_shape_rc` the warped image (with shape `dst_shape_rc`)

    """


    if M is None and dxdy is None:
        warnings.warn("Please provide `M` and/or `dxdy`")
        return None

    if dxdy is None and transformation_dst_shape_rc is None:
        warnings.warn("Please provide `transformation_dst_shape_rc`")
        return None

    if dxdy is not None and transformation_dst_shape_rc is None:
        transformation_dst_shape_rc = dxdy[0].shape

    if dst_shape_rc is None:
        dst_shape_rc = transformation_dst_shape_rc

    if src_shape_rc is None:
        src_shape_rc = transformation_src_shape_rc


    if np.all(transformation_dst_shape_rc == dst_shape_rc):
        grid_r, grid_c = np.indices(transformation_dst_shape_rc)

    else:
        scaled_y = np.linspace(0, dst_shape_rc[0], num=transformation_dst_shape_rc[0])
        scaled_x = np.linspace(0, dst_shape_rc[1], num=transformation_dst_shape_rc[1])
        grid_y, grid_x = np.meshgrid(scaled_y, scaled_x, indexing="ij")
        scaled_xy = np.dstack([grid_x.reshape(-1), grid_y.reshape(-1)])[0]
        sy, sx = np.array(dst_shape_rc)/np.array(transformation_dst_shape_rc)
        S = transform.SimilarityTransform(scale=(sx, sy))
        src_xy_pos = S.inverse(scaled_xy)
        grid_r, grid_c = src_xy_pos[:, 1].reshape(transformation_dst_shape_rc), src_xy_pos[:, 0].reshape(transformation_dst_shape_rc)

    if dxdy is None:
        r_in_src = grid_r
        c_in_src = grid_c
    else:
        r_in_src = grid_r + dxdy[1]
        c_in_src = grid_c + dxdy[0]

    if M is not None:
        tformer = transform.ProjectiveTransform(matrix=M)
        xy_pos_in_src = tformer(np.dstack([c_in_src.reshape(-1), r_in_src.reshape(-1)])[0])
        xy_pos_in_src = [xy_pos_in_src[:, 0].reshape(transformation_dst_shape_rc), xy_pos_in_src[:, 1].reshape(transformation_dst_shape_rc)]

    else:
        xy_pos_in_src = [c_in_src, r_in_src]

    if np.any(transformation_src_shape_rc != src_shape_rc):
        in_scale_y, in_scale_x = np.array(src_shape_rc)/np.array(transformation_src_shape_rc)
        in_S = transform.SimilarityTransform(scale=(in_scale_x, in_scale_y))
        xy_pos_in_src = in_S(np.dstack([xy_pos_in_src[0].reshape(-1), xy_pos_in_src[1].reshape(-1)])[0])
        xy_pos_in_src = [xy_pos_in_src[:, 0].reshape(transformation_dst_shape_rc), xy_pos_in_src[:, 1].reshape(transformation_dst_shape_rc)]

    if return_xy:
        c1, c2 = 0, 1
    else:
        c1, c2 = 1, 0

    coord_map = np.array([xy_pos_in_src[c1], xy_pos_in_src[c2]])

    return coord_map


def get_padding_matrix(img_shape_rc, out_shape_rc):
    img_h, img_w = img_shape_rc
    out_h, out_w = out_shape_rc

    d_h = (out_h - img_h)
    d_w = (out_w - img_w)

    h_pad = d_h/2
    w_pad = d_w/2
    T = np.identity(3).astype(np.float64)
    T[0, 2] = -w_pad
    T[1, 2] = -h_pad

    return T


def get_reflection_M(reflect_x, reflect_y, shape_rc):
    """Get transformation matrix to reflect an image

    Parameters
    ----------

    reflect_x : bool
        Whether or not to reflect the x-axis (columns)

    reflecct y : bool
        Whether or not to reflect the y-axis (rows)

    shape_rc : tuple of int
        Shape of the image being reflected

    Returns
    -------
    reflection_M : ndarray
        Transformation matrix that will reflect an image along the
        specified axes.

    """

    reflection_M = np.eye(3)
    if reflect_x:
        reflection_M[0, 0] *= -1
        reflection_M[0, 2] += shape_rc[1] - 1

    if reflect_y:
        reflection_M[1, 1] *= -1
        reflection_M[1, 2] += shape_rc[0] - 1

    return reflection_M


def get_img_area(img_shape_rc, M=None):

    prev_img_corners = get_corners_of_image(img_shape_rc)[:, ::-1]

    if M is not None:
        prev_img_corners = warp_xy(prev_img_corners, M)

    prev_img_corners = order_points(prev_img_corners)
    prev_area = 0.5*np.abs(np.dot(prev_img_corners[:, 0],np.roll(prev_img_corners[:, 1],1))-np.dot(prev_img_corners[:, 1],np.roll(prev_img_corners[:, 0],1)))
    return prev_area


def get_overlap_mask(img1, img2):
    mask = np.zeros_like(img1)
    mask[img1 > 0] += 1
    mask[img2 > 0] += 1
    mask[mask != 2] = 0

    return mask


def center_and_get_translation_matrix(img_shape_rc, x, y, w, h):
    '''
    x, y, w, h attributes or
    :param img_shape_rc:
    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    '''

    # Center smaller image inside larger image #
    img_center_w = int(img_shape_rc[1] / 2)
    img_center_h = int(img_shape_rc[0] / 2)


    out_center_w = int(w / 2) + x
    out_center_h = int(h / 2) + y

    x_center_shift = img_center_w - out_center_w
    y_center_shift = img_center_h - out_center_h

    T = np.array([[1, 0, -x_center_shift], [0, 1, -y_center_shift]]).astype(np.float64)

    return T


def get_affine_transformation_params(M):
    """
    Get individula components affine transformation.
    Based on properties in skimage._geometric.AffineTransform

    Parameters
    ----------
    M : (3,3) array
        Transformation matrix found one of scikit-image's transformation objects

    Returns
    -------
    (tx, ty) : (float, float)
        Translation in X and Y direction

    rotation : float
        Counter clockwise rotation, in radians

    (scale_x, scale_y) : (float, float)
        Scale in the X and Y dimensions

    shear : float
        Shear angle in counter-clockwise direction as radians.

    """

    scale_x = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)
    scale_y = np.sqrt(M[0, 1] ** 2 + M[1, 1] ** 2)
    rotation = np.arctan2(M[1, 0], M[0, 0])
    tx, ty = M[0:2, 2]
    shear = np.arctan2(-M[0, 1], M[1, 1]) - rotation

    return (tx, ty), rotation, (scale_x, scale_y), shear


def decompose_affine_transformation(M):
    """
    Get individula components affine transformation.
    Based on properties in skimage._geometric.AffineTransform

    Parameters
    ----------
    M : (3,3) array
        Transformation matrix found one of scikit-image's transformation objects

    Returns
    -------
    T : (3,3) array
        Translation matrix

    R : (3,3) array
        counter-clockwise rotation matrix

    S : (3,3) array
        Scaling matrix

    H : (3,3) array
        Shear matrix

    """

    txy, rotation, sxy, shear = get_affine_transformation_params(M)

    T = transform.AffineTransform(translation=txy).params
    R = transform.AffineTransform(rotation=rotation).params
    S = transform.AffineTransform(scale=sxy).params
    H = transform.AffineTransform(shear=shear).params

    return T, R, S, H


def get_rotate_around_center_M(img_shape, rotation_rad):
    #Based on skimage warp.rotate, but can have scaling at end
    rows, cols = img_shape[0:2]

    # rotation around center
    center = np.array((cols, rows)) / 2. - 0.5
    tform1 = transform.SimilarityTransform(translation=center)
    tform2 = transform.SimilarityTransform(rotation=rotation_rad)
    tform3 = transform.SimilarityTransform(translation=-center)
    tform = tform3 + tform2 + tform1
    return tform.params


def calc_d(pt1, pt2):
    """
    Calculate euclidean disrances between each pair coresponding points in pt1 and pt2

    Parameters
    ----------
    pt1 : (2, N) array
        Array of N 2D points

    pt2 : (2, N) array
        Array of N 2D points

    Returns
    -------
    d : [N]
        distnace between correspoing points in pt1 and pt2
    """

    d = np.sqrt(np.sum((pt1 - pt2)**2, axis=1))
    return d


def get_mesh(shape, grid_spacing, bbox_rc_wh=None, inclusive=False):
    """Get meshgrid for given shape and spacing.

    Can provide bbox positions to limit gridsize in image

    Parameters
    ----------
    shape : tuple
        Number of rows and columns in image

    grid_spacing : int
        Number of pixels between gridpoints

    bbox_rc_wh : tuple
        (row, column, width, height) of bounding box

    inclusive : bool
        Whether or not to include image edges

    """

    if bbox_rc_wh is not None:
        min_r = bbox_rc_wh[0]
        max_r = bbox_rc_wh[0] + bbox_rc_wh[3]
        min_c = bbox_rc_wh[1]
        max_c = bbox_rc_wh[1] + bbox_rc_wh[2]
    else:

        min_r = 0
        min_c = 0
        max_r = shape[0]
        max_c = shape[1]

    r_grid_pts = np.arange(min_r, max_r, grid_spacing)
    c_grid_pts = np.arange(min_c, max_c, grid_spacing)

    if inclusive:
        if max(r_grid_pts) != shape[0]-1:
            r_grid_pts = np.hstack([r_grid_pts, shape[0]-1])

        if max(c_grid_pts) != shape[1]-1:
            c_grid_pts = np.hstack([c_grid_pts, shape[1]-1])

    return np.meshgrid(r_grid_pts, c_grid_pts, indexing="ij")


def smooth_dxdy(dxdy, grid_spacing_ratio=0.015, sigma_ratio=0.005,
                method="gauss"):
    """Smooth displacement fields

    Use cubic interpolation to smooth displacement field

    Parameters
    ----------
    dxdy : ndarray
        (2, N, M) numpy array of pixel displacements in the
        x and y directions

    grid_spacing_ratio : float
        Fraction of image shape that should be used for spacing
        between points in grid used to smooth displacement fields.
        Larger values will do more smoothing. Only used if method
        is "cubic"

    sigma_ratio : float
        Determines the amount of Gaussian smoothing, as
        sigma = max(shape) *sigma_ratio. Larger values do more
        smoothing. Only used if method is "gauss"

    method : str
        If "gauss", then a Gaussian blur will be applied to the
        deformation fields, using sigma defined by sigma_ratio.
        If "cubic", then cubic interpolation is used to smooth
        the fields, using grid_spacing_ratio to determine
        the sampling points.

    Returns
    -------
    smooth_dxdy : ndarray
        Smoothed copy of dxdy

    """

    dx, dy = dxdy
    if method.lower().startswith("c"):
        grid_spacing_x = dx.shape[1]*grid_spacing_ratio
        grid_spacing_y = dx.shape[0]*grid_spacing_ratio
        grid_spacing = int(np.mean([grid_spacing_x, grid_spacing_y]))

        subgrid_r, subgrid_c = get_mesh(dx.shape, grid_spacing, inclusive=True)

        grid = UCGrid((0.0, float(dx.shape[1]), int(subgrid_r.shape[1])),
                      (0.0, float(dx.shape[0]), int(subgrid_r.shape[0])))

        grid_y, grid_x = np.indices(dx.shape)
        grid_xy = np.dstack([grid_x.reshape(-1), grid_y.reshape(-1)]).astype(float)[0]

        dx_cubic_coeffs = filter_cubic(grid, dx[subgrid_r, subgrid_c]).T
        dy_cubic_coeffs = filter_cubic(grid, dy[subgrid_r, subgrid_c]).T
        smooth_dx = eval_cubic(grid, dx_cubic_coeffs, grid_xy).reshape(dx.shape)
        smooth_dy = eval_cubic(grid, dy_cubic_coeffs, grid_xy).reshape(dx.shape)

    elif method.lower().startswith("g"):
        sigma = sigma_ratio*np.max(dx.shape)
        smooth_dx = filters.gaussian(dx, sigma=sigma)
        smooth_dy = filters.gaussian(dy, sigma=sigma)

    return np.dstack([smooth_dx, smooth_dy])


def get_inverse_field(backwards_xy_deltas, n_inter=10):
    """
    Invert transform
    """

    sitk_bk_dxdy = sitk.GetImageFromArray(np.dstack(backwards_xy_deltas),  isVector=True)
    sitk_fw_dxdy = sitk.IterativeInverseDisplacementField(sitk_bk_dxdy, numberOfIterations=n_inter)
    fwd_dxdy = sitk.GetArrayFromImage(sitk_fw_dxdy)
    fwd_dxdy = [fwd_dxdy[..., 0], fwd_dxdy[..., 1]]

    return fwd_dxdy


def warp_xy_rigid(xy, inv_matrix):
    """ Warp points

    Warp xy given an inverse transformation matrix found using one of scikit-image's transform objects
    Inverse matrix should have been found using tform(dst, src)
    Adpated from skimage._geometric.ProjectiveTransform._apply_mat
    Changed so that inverse matrix (found using dst -> src) automatically inverted to warp points forward (src -> dst)
    """
    xy = np.array(xy, copy=False, ndmin=2)

    x, y = np.transpose(xy)
    src_pts = np.vstack((x, y, np.ones_like(x)))
    try:
        dst_pts = src_pts.T @ np.linalg.inv(inv_matrix).T
    except np.linalg.LinAlgError :
        print("Singular matrix")
        dst_pts = src_pts.T @ np.linalg.pinv(inv_matrix).T

    # below, we will divide by the last dimension of the homogeneous
    # coordinate matrix. In order to avoid division by zero,
    # we replace exact zeros in this column with a very small number.
    dst_pts[dst_pts[:, 2] == 0, 2] = np.finfo(float).eps
    # rescale to homogeneous coordinates
    dst_pts[:, :2] /= dst_pts[:, 2:3]
    return dst_pts[:, :2]


def get_warp_scaling_factors(transformation_src_shape_rc=None, transformation_dst_shape_rc=None, src_shape_rc=None, dst_shape_rc=None, bk_dxdy=None, fwd_dxdy=None):
    """Get scaling factors needed to warp points

    If a returned value is None, it means there is no need to scale the image
    Returns
    -------
    src_sxy : ndarray
        Scaling to go from transformation_src_shape_rc -> src_shape_rc (i.e. transformation_src_shape_rc/src_shape_rc)

    dst_sxy : ndarray
        When `bk_dxdy` or `fwd_dxdy` is None, this is the scaling to go from
        transformation_dst_shape_rc -> dst_shape_rc (i.e. dst_shape_rc/transformation_dst_shape_rc).

        When `bk_dxdy` or `fwd_dxdy` are provided, this is the scaling that goes from the
        displacement -> `dst_shape_rc`

    displacement_sxy :
        Scaling for dxdy for when non-rigid transformations found using an
        image with a size different than transformation_dst_shape_rc.

        For example, if displacement was found on an image 2x the one with
        `transformation_dst_shape_rc`, this would be 2. Used to warp points
        from position in image with shape transformation_dst_shape_rc to position
        in `bk_dxdy` or `fwd_dxdy`.

    displacement_shape_rc : (int, int)
        Shape of displacement field used for non-rigid transforms

    """
    do_non_rigid = bk_dxdy is not None or fwd_dxdy is not None

    # convert shapes to arrays
    if src_shape_rc is not None:
        src_shape_rc = np.array(src_shape_rc)

    if transformation_src_shape_rc is not None:
        transformation_src_shape_rc = np.array(transformation_src_shape_rc)

    if dst_shape_rc is not None:
        dst_shape_rc = np.array(dst_shape_rc)

    if transformation_dst_shape_rc is not None:
        transformation_dst_shape_rc = np.array(transformation_dst_shape_rc)

    # Get input scaling
    if transformation_src_shape_rc is not None and src_shape_rc is not None:
        # Scale points to where they would be in image with transformation_src_shape_rc
        if np.all(transformation_src_shape_rc == src_shape_rc):
            src_sxy = None
        else:
            src_sxy = (src_shape_rc/transformation_src_shape_rc)[::-1]
    else:
        src_sxy = None

    # Get output shapes
    non_rigid_is_array = False
    if bk_dxdy is not None or fwd_dxdy is not None:
        if bk_dxdy is not None:
            if not isinstance(bk_dxdy, pyvips.Image):
                non_rigid_is_array = True
        if fwd_dxdy is not None:
            if not isinstance(fwd_dxdy, pyvips.Image):
                non_rigid_is_array = True

    if do_non_rigid:
        if bk_dxdy is not None:
            if non_rigid_is_array:
                displacement_shape_rc = np.array(bk_dxdy[0].shape)
            else:
                displacement_shape_rc = np.array([bk_dxdy.height, bk_dxdy.width])
        elif fwd_dxdy is not None:
            if non_rigid_is_array:
                displacement_shape_rc = np.array(fwd_dxdy[0].shape)
            else:
                displacement_shape_rc = np.array([fwd_dxdy.height, fwd_dxdy.width])

    if transformation_dst_shape_rc is None and do_non_rigid:
            transformation_dst_shape_rc = displacement_shape_rc

    if dst_shape_rc is None and transformation_dst_shape_rc is not None:
        dst_shape_rc = transformation_dst_shape_rc

    # Get output scalings
    if do_non_rigid:
        if not np.all(transformation_dst_shape_rc == displacement_shape_rc):
            # non-rigid found on scaled image
            displacement_sxy = (displacement_shape_rc/transformation_dst_shape_rc)[::-1]
            dst_sxy = (dst_shape_rc/displacement_shape_rc)[::-1]
        else:
            displacement_sxy = None
            dst_sxy = (dst_shape_rc/transformation_dst_shape_rc)[::-1]

        if np.all(dst_sxy == 1):
            dst_sxy = None
    else:
        # Determine how to scale to images for position in image with shape = dst_shape_rc
        dst_sxy = None
        displacement_shape_rc = None
        displacement_sxy = None
        if transformation_dst_shape_rc is not None and dst_shape_rc is not None:
            if not np.all(dst_shape_rc == transformation_dst_shape_rc):
                dst_sxy = (dst_shape_rc/transformation_dst_shape_rc)[::-1]

    return src_sxy, dst_sxy, displacement_sxy, displacement_shape_rc



def _warp_pt_vips(xy, M=None, vips_bk_dxdy=None, vips_fwd_dxdy=None, src_sxy=None, dst_sxy=None, displacement_sxy=None, displacement_shape_rc=None, pt_buffer=100):
    """Warp single point when the displacement fields are pyvips.Image objects

    """
    do_non_rigid = vips_bk_dxdy is not None or vips_fwd_dxdy is not None

    if src_sxy is not None:
        in_src_xy = xy/src_sxy

    else:
        in_src_xy = xy

    if M is not None:
        rigid_xy = warp_xy_rigid(in_src_xy, M).astype(float)[0]
        if not do_non_rigid:
            if dst_sxy is not None:
                return rigid_xy*dst_sxy
            else:
                return rigid_xy
    else:
        rigid_xy = in_src_xy

    if displacement_sxy is not None:
        # displacement was found on scaled version of the rigidly registered image.
        # So move points into new displacement field
        rigid_xy *= displacement_sxy


    bbox_xy_tl  = (rigid_xy - pt_buffer//2).astype(int)
    bbox_xy_br  = np.ceil(rigid_xy + pt_buffer//2).astype(int)
    bbox_x01 = np.clip(np.array([bbox_xy_tl[0], bbox_xy_br[0]]), 0, displacement_shape_rc[1])
    bbox_y01 = np.clip(np.array([bbox_xy_tl[1], bbox_xy_br[1]]), 0, displacement_shape_rc[0])

    bbox_w = -int(np.subtract(*bbox_x01))
    bbox_h = -int(np.subtract(*bbox_y01))
    region_bbox_xywh = np.array([bbox_x01[0], bbox_y01[0], bbox_w, bbox_h])

    # Move point to position in tile
    rigid_xy_in_tile = rigid_xy - region_bbox_xywh[:2]

    # Get region dxdy
    if vips_bk_dxdy is None and vips_fwd_dxdy is not None:
        vips_region_dxdy = vips_fwd_dxdy.extract_area(*region_bbox_xywh)
        region_dxdy = vips2numpy(vips_region_dxdy)
    elif vips_bk_dxdy is not None and vips_fwd_dxdy is None:
        vips_region_bk_dxdy = vips_bk_dxdy.extract_area(*region_bbox_xywh)
        region_bk_dxdy = vips2numpy(vips_region_bk_dxdy)
        region_dxdy = np.dstack(get_inverse_field(region_bk_dxdy[..., 0], region_bk_dxdy[..., 1]))

    grid = UCGrid((0.0, float(bbox_w-1), int(bbox_w)),
                  (0.0, float(bbox_h-1), int(bbox_h)))

    dx_cubic_coeffs = filter_cubic(grid, region_dxdy[..., 0]).T
    dy_cubic_coeffs = filter_cubic(grid, region_dxdy[..., 1]).T

    new_x = region_bbox_xywh[0] + rigid_xy_in_tile[0] + eval_cubic(grid, dx_cubic_coeffs, rigid_xy_in_tile)
    new_y = region_bbox_xywh[1] + rigid_xy_in_tile[1] + eval_cubic(grid, dy_cubic_coeffs, rigid_xy_in_tile)

    nonrigid_xy = np.array([new_x, new_y])
    if dst_sxy is not None:
        nonrigid_xy *= dst_sxy

    return nonrigid_xy


def _warp_xy_vips(xy, M=None, transformation_src_shape_rc=None, transformation_dst_shape_rc=None,
                 src_shape_rc=None, dst_shape_rc=None, vips_bk_dxdy=None, vips_fwd_dxdy=None, pt_buffer=100):
    """
    Warp xy points using M and/or bk_dxdy/fwd_dxdy.
    Used when `vips_bk_dxdy` or `vips_fwd_dxdy` is a pyvips.Image

    Parameters
    ----------
    xy : ndarray
        [P, 2] array of xy coordinates for P points

    M : ndarray, optional
         3x3 affine transformation matrix to perform rigid warp

    transformation_src_shape_rc : (int, int)
        Shape of image that was used to find the transformation.
        For example, this could be the original image in which features were detected

    transformation_dst_shape_rc : (int, int), optional
        Shape of the image with shape `transformation_src_shape_rc` after warping.
        This could be the shape of the original image after applying `M`.

    src_shape_rc : optional, (int, int)
        Shape of the image from which the points originated. For example,
        this could be a larger/smaller version of the image that was
        used for feature detection.

    dst_shape_rc : optional, (int, int)
        Shape of image (with shape `src_shape_rc`) after warping

    vips_bk_dxdy : pyvips.Image
        (2, N, M) numpy array of pixel displacements in the x and y
        directions from the reference image. dx = bk_dxdy[0],
        and dy=bk_dxdy[1]. If `bk_dxdy` is not None, but
        `fwd_dxdy` is None, then `bk_dxdy` will be inverted to warp `xy`.

    vips_fwd_dxdy : pyvips.Image
        Inverse of bk_dxdy. dx = fwd_dxdy[0], and dy=fwd_dxdy[1].
        This is what is actually used to warp the points.

    pt_buffer : int
        This method slices the region surrounding the point from the displacement fields.
        The `pt_buffer` determines the size of the window around the point.

    Returns
    -------
    warped_xy : [P, 2] array
        Array of warped xy coordinates for P points

    """
    src_sxy, dst_sxy, displacement_sxy, displacement_shape_rc = get_warp_scaling_factors(transformation_src_shape_rc=transformation_src_shape_rc,
                                                                        transformation_dst_shape_rc=transformation_dst_shape_rc,
                                                                        src_shape_rc=src_shape_rc, dst_shape_rc=dst_shape_rc,
                                                                        bk_dxdy=vips_bk_dxdy, fwd_dxdy=vips_fwd_dxdy)


    warped_xy = np.vstack([_warp_pt_vips(pt, M, vips_bk_dxdy=vips_bk_dxdy, vips_fwd_dxdy=vips_fwd_dxdy, src_sxy=src_sxy, dst_sxy=dst_sxy, displacement_sxy=displacement_sxy, displacement_shape_rc=displacement_shape_rc, pt_buffer=pt_buffer) for pt in xy])

    return warped_xy


def _warp_xy_numpy(xy, M=None, transformation_src_shape_rc=None, transformation_dst_shape_rc=None,
            src_shape_rc=None, dst_shape_rc=None,
            bk_dxdy=None, fwd_dxdy=None):
    """
    Warp xy points using M and/or bk_dxdy/fwd_dxdy. If bk_dxdy is provided, it will be inverted to  create fwd_dxdy

    Parameters
    ----------
    xy : ndarray
        [P, 2] array of xy coordinates for P points

    M : ndarray, optional
         3x3 affine transformation matrix to perform rigid warp

    transformation_src_shape_rc : (int, int)
        Shape of image that was used to find the transformation.
        For example, this could be the original image in which features were detected

    transformation_dst_shape_rc : (int, int), optional
        Shape of the image with shape `transformation_src_shape_rc` after warping.
        This could be the shape of the original image after applying `M`.

    src_shape_rc : optional, (int, int)
        Shape of the image from which the points originated. For example,
        this could be a larger/smaller version of the image that was
        used for feature detection.

    dst_shape_rc : optional, (int, int)
        Shape of image (with shape `src_shape_rc`) after warping

    bk_dxdy : ndarray
        (2, N, M) numpy array of pixel displacements in the x and y
        directions from the reference image. dx = bk_dxdy[0],
        and dy=bk_dxdy[1]. If `bk_dxdy` is not None, but
        `fwd_dxdy` is None, then `bk_dxdy` will be inverted to warp `xy`.

    fwd_dxdy : ndarray
        Inverse of bk_dxdy. dx = fwd_dxdy[0], and dy=fwd_dxdy[1].
        This is what is actually used to warp the points.

    Returns
    -------
    warped_xy : [P, 2] array
        Array of warped xy coordinates for P points

    """

    do_non_rigid = bk_dxdy is not None or fwd_dxdy is not None

    if M is None and not do_non_rigid:
        return xy

    src_sxy, dst_sxy, displacement_sxy, displacement_shape_rc = get_warp_scaling_factors(transformation_src_shape_rc=transformation_src_shape_rc,
                                                                     transformation_dst_shape_rc=transformation_dst_shape_rc,
                                                                     src_shape_rc=src_shape_rc, dst_shape_rc=dst_shape_rc,
                                                                     bk_dxdy=bk_dxdy, fwd_dxdy=fwd_dxdy)
    if src_sxy is not None:
        in_src_xy = xy/src_sxy
    else:
        in_src_xy = xy

    if M is not None:
        rigid_xy = warp_xy_rigid(in_src_xy, M).astype(float)
        if not do_non_rigid:
            if dst_sxy is not None:
                return rigid_xy*dst_sxy
            else:
                return rigid_xy
    else:
        rigid_xy = in_src_xy

    if displacement_sxy is not None:
        # displacement was found on scaled version of the rigidly registered image.
        # So move points into new displacement field
        rigid_xy *= displacement_sxy

    if bk_dxdy is not None and fwd_dxdy is None:
        fwd_dxdy = get_inverse_field(bk_dxdy)

    grid = UCGrid((0.0, float(displacement_shape_rc[1]-1), int(displacement_shape_rc[1])),
                  (0.0, float(displacement_shape_rc[0]-1), int(displacement_shape_rc[0])))

    dx_cubic_coeffs = filter_cubic(grid, fwd_dxdy[0]).T
    dy_cubic_coeffs = filter_cubic(grid, fwd_dxdy[1]).T

    new_x = rigid_xy[:, 0] + eval_cubic(grid, dx_cubic_coeffs, rigid_xy)
    new_y = rigid_xy[:, 1] + eval_cubic(grid, dy_cubic_coeffs, rigid_xy)

    nonrigid_xy = np.dstack([new_x, new_y])[0]
    if dst_sxy is not None:
        nonrigid_xy *= dst_sxy

    return nonrigid_xy


def warp_xy(xy, M=None, transformation_src_shape_rc=None, transformation_dst_shape_rc=None,
            src_shape_rc=None, dst_shape_rc=None,
            bk_dxdy=None, fwd_dxdy=None, pt_buffer=100):
    """
    Warp xy points using M and/or bk_dxdy/fwd_dxdy. If bk_dxdy is provided, it will be inverted to  create fwd_dxdy

    Parameters
    ----------
    xy : ndarray
        [P, 2] array of xy coordinates for P points

    M : ndarray, optional
         3x3 affine transformation matrix to perform rigid warp

    transformation_src_shape_rc : (int, int)
        Shape of image that was used to find the transformation.
        For example, this could be the original image in which features were detected

    transformation_dst_shape_rc : (int, int), optional
        Shape of the image with shape `transformation_src_shape_rc` after warping.
        This could be the shape of the original image after applying `M`.

    src_shape_rc : optional, (int, int)
        Shape of the image from which the points originated. For example,
        this could be a larger/smaller version of the image that was
        used for feature detection.

    dst_shape_rc : optional, (int, int)
        Shape of image (with shape `src_shape_rc`) after warping

    bk_dxdy : ndarray, pyvips.Image
        (2, N, M) numpy array of pixel displacements in the x and y
        directions from the reference image. dx = bk_dxdy[0],
        and dy=bk_dxdy[1]. If `bk_dxdy` is not None, but
        `fwd_dxdy` is None, then `bk_dxdy` will be inverted to warp `xy`.

    fwd_dxdy : ndarray, pyvips.Image
        Inverse of bk_dxdy. dx = fwd_dxdy[0], and dy=fwd_dxdy[1].
        This is what is actually used to warp the points.

    pt_buffer : int
        If `bk_dxdy` or `fwd_dxdy` are pyvips.Image object, then
        pt_buffer` determines the size of the window around the point used to
        get the local displacements.


    Returns
    -------
    warped_xy : [P, 2] array
        Array of warped xy coordinates for P points

    """

    do_non_rigid = bk_dxdy is not None or fwd_dxdy is not None

    if M is None and not do_non_rigid:
        return xy

    if isinstance(bk_dxdy, pyvips.Image) or isinstance(fwd_dxdy, pyvips.Image):
        warped_xy = _warp_xy_vips(xy, M, transformation_src_shape_rc=transformation_src_shape_rc,
                                  transformation_dst_shape_rc=transformation_dst_shape_rc,
                                  src_shape_rc=src_shape_rc, dst_shape_rc=dst_shape_rc,
                                  vips_bk_dxdy=bk_dxdy, vips_fwd_dxdy=fwd_dxdy, pt_buffer=pt_buffer)
    else:
        warped_xy = _warp_xy_numpy(xy, M, transformation_src_shape_rc=transformation_src_shape_rc,
                                   transformation_dst_shape_rc=transformation_dst_shape_rc,
                                   src_shape_rc=src_shape_rc, dst_shape_rc=dst_shape_rc,
                                   bk_dxdy=bk_dxdy, fwd_dxdy=fwd_dxdy)
    return warped_xy


def warp_xy_inv(xy, M=None, transformation_src_shape_rc=None, transformation_dst_shape_rc=None, src_shape_rc=None, dst_shape_rc=None, bk_dxdy=None, fwd_dxdy=None):
    """Warp points from registered coordinates to original coordinates

    Parameters
    ----------
    xy : ndarray
        [P, 2] array of xy coordinates for P points

    M : ndarray, optional
         3x3 affine transformation matrix to perform rigid warp

    transformation_src_shape_rc : (int, int)
        Shape of image that was used to find the transformation.
        For example, this could be the original image in which features were detected

    transformation_dst_shape_rc : (int, int), optional
        Shape of the image with shape `transformation_src_shape_rc` after warping.
        This could be the shape of the original image after applying `M`.

    src_shape_rc : optional, (int, int)
        Shape of the image from which the points originated. For example,
        this could be a larger/smaller version of the image that was
        used for feature detection.

    dst_shape_rc : optional, (int, int)
        Shape of image (with shape `src_shape_rc`) after warping

    bk_dxdy : ndarray
        (2, N, M) numpy array of pixel displacements in the x and y
        directions from the reference image. dx = bk_dxdy[0],
        and dy=bk_dxdy[1].  This is what is actually used to warp the points.

    fwd_dxdy : ndarray
        Inverse of bk_dxdy. dx = fwd_dxdy[0], and dy=fwd_dxdy[1].
        If `fwd_dxdy` is not None, but
        `bk_dxdy` is None, then `fwd_dxdy` will be inverted to warp `xy`.

    """
    do_non_rigid = bk_dxdy is not None or fwd_dxdy is not None

    if M is None and not do_non_rigid:
        return xy

    src_sxy, dst_sxy, displacement_sxy, displacement_shape_rc = get_warp_scaling_factors(transformation_src_shape_rc=transformation_src_shape_rc,
                                                                     transformation_dst_shape_rc=transformation_dst_shape_rc,
                                                                     src_shape_rc=src_shape_rc, dst_shape_rc=dst_shape_rc,
                                                                     bk_dxdy=bk_dxdy, fwd_dxdy=fwd_dxdy)

    if dst_sxy is not None:
        xy_in_reg_img = xy/dst_sxy
    else:
        xy_in_reg_img = xy

    # Get points into position in the rigid image #
    if do_non_rigid:
        if fwd_dxdy is not None and bk_dxdy is None:
            bk_dxdy = get_inverse_field(fwd_dxdy)

        xy_in_rigid = warp_xy(xy_in_reg_img, fwd_dxdy=bk_dxdy)
        if displacement_sxy is not None:
            xy_in_rigid /= displacement_sxy
    else:
        xy_in_rigid = xy_in_reg_img

    if M is not None:
         xy_inv = warp_xy(xy_in_rigid, M=np.linalg.inv(M))
    else:
        xy_inv = xy_in_rigid

    if src_sxy is not None:
        xy_inv *= src_sxy

    return xy_inv


def warp_xy_from_to(xy, from_M=None, from_transformation_src_shape_rc=None,
                   from_transformation_dst_shape_rc=None, from_src_shape_rc=None,
                   from_dst_shape_rc=None,from_bk_dxdy=None, from_fwd_dxdy=None,
                   to_M=None, to_transformation_src_shape_rc=None,
                   to_transformation_dst_shape_rc=None, to_src_shape_rc=None,
                   to_dst_shape_rc=None, to_bk_dxdy=None, to_fwd_dxdy=None):
    """Warp points in one image to their position in another unregistered image

    Takes a set of points found in the unwarped "from" image, and warps them to their
    position in the unwarped "to" image.

    Parameters
    ----------
    xy : ndarray
        [P, 2] array of xy coordinates for P points

    from_M : ndarray, optional
         3x3 affine transformation matrix to perform rigid warp in the "from" image

    from_transformation_src_shape_rc : (int, int)
        Shape of image that was used to find the transformation in the
        "from" image. For example, this could be the original image in
        which features were detected in the "from" image.

    from_transformation_dst_shape_rc : (int, int)
        Shape (row, col) of registered image. As the "from"  and "to" images have been registered,
        this shape should be the same for both images.

    from_src_shape_rc : optional, (int, int)
        Shape of the unwarped image from which the points originated. For example,
        this could be a larger/smaller version of the "from" image that was
        used for feature detection.

    from_dst_shape_rc : optional, (int, int)
        Shape of from image (with shape `src_shape_rc`) after warping

    from_bk_dxdy : ndarray
        (2, N, M) numpy array of pixel displacements in the x and y in the "from" image.
        dx = bk_dxdy[0], and dy=bk_dxdy[1].

    from_fwd_dxdy : ndarray
        Inverse of `from_bk_dxdy`

    to_M : ndarray, optional
        3x3 affine transformation matrix to perform rigid warp in the "to" image

    to_transformation_src_shape_rc :  optional, (int, int)
        Shape of "to" image that was used to find the transformations.
        For example, this could be the original image in which features were detected

    to_src_shape_rc : optional, (int, int)
        Shape of the unwarped "to" image to which the points will be warped. For example,
        this could be a larger/smaller version of the "to" image that was
        used for feature detection.

    to_dst_shape_rc : optional, (int, int)
        Shape of to image (with shape `src_shape_rc`) after warping

    to_bk_dxdy : ndarray
        (2, N, M) numpy array of pixel displacements in the x and y in the "to" image.
        dx = bk_dxdy[0], and dy=bk_dxdy[1].

    to_fwd_dxdy : ndarray
        Inverse of `to_bk_dxdy`

    Returns
    -------
    xy_in_to : ndarray
        position of `xy` in the unwarped "to" image

    """

    xy_in_reg_space = warp_xy(xy, M=from_M,
                              transformation_src_shape_rc=from_transformation_src_shape_rc,
                              transformation_dst_shape_rc=from_transformation_dst_shape_rc,
                              src_shape_rc=from_src_shape_rc,
                              dst_shape_rc=from_dst_shape_rc,
                              bk_dxdy=from_bk_dxdy,
                              fwd_dxdy=from_fwd_dxdy
                              )

    xy_in_to_space = warp_xy_inv(xy_in_reg_space, M=to_M,
                                 transformation_src_shape_rc=to_transformation_src_shape_rc,
                                 transformation_dst_shape_rc=to_transformation_dst_shape_rc,
                                 src_shape_rc=to_src_shape_rc,
                                 dst_shape_rc=to_dst_shape_rc,
                                 bk_dxdy=to_bk_dxdy,
                                 fwd_dxdy=to_fwd_dxdy
                                )
    return xy_in_to_space


def clip_xy(xy, shape_rc):
    """Clip xy coordintaes to be within image

    """
    clipped_x =  np.clip(xy[:, 0], 0, shape_rc[1])
    clipped_y =  np.clip(xy[:, 1], 0, shape_rc[0])

    clipped_xy = np.dstack([clipped_x, clipped_y])[0]
    return clipped_xy


def _warp_shapely(geom, warp_fxn, warp_kwargs, shift_xy=None):
    """Warp a shapely geometry
    Based on shapely.ops.trasform

    """
    if "dst_shape_rc" in warp_kwargs:
        dst_shape_rc = warp_kwargs["dst_shape_rc"]
    elif "to_dst_shape_rc" in warp_kwargs:
        dst_shape_rc = warp_kwargs["to_dst_shape_rc"]
    else:
        dst_shape_rc  = None

    if geom.is_empty:
        return type(geom)([])
    if geom.geom_type in ("Point", "LineString", "LinearRing", "Polygon"):
        if geom.geom_type in ("Point", "LineString", "LinearRing"):
            warped_xy = warp_fxn(np.vstack(geom.coords), **warp_kwargs)
            if shift_xy is not None:
                warped_xy -= shift_xy
            if dst_shape_rc is not None:
                warped_xy = clip_xy(warped_xy, dst_shape_rc)

            return type(geom)(warped_xy.tolist())

        elif geom.geom_type == "Polygon":
            shell_xy = warp_fxn(np.vstack(geom.exterior.coords), **warp_kwargs)
            if shift_xy is not None:
                shell_xy -= shift_xy

            if dst_shape_rc is not None:
                shell_xy = clip_xy(shell_xy, dst_shape_rc)

            shell = type(geom.exterior)(shell_xy.tolist())
            holes = []
            for ring in geom.interiors:
                holes_xy = warp_fxn(np.vstack(ring.coords), **warp_kwargs)
                if shift_xy is not None:
                    holes_xy -= shift_xy
                if dst_shape_rc is not None:
                    holes_xy = clip_xy(holes_xy, dst_shape_rc)

                holes.append(type(ring)(holes_xy))

            return type(geom)(shell, holes)

    elif geom.geom_type.startswith("Multi") or geom.geom_type == "GeometryCollection":
        return type(geom)([_warp_shapely(part, warp_fxn, warp_kwargs) for part in geom.geoms])
    else:
        raise shapely.errors.GeometryTypeError(f"Type {geom.geom_type!r} not recognized")


def warp_shapely_geom(geom, M=None, transformation_src_shape_rc=None, transformation_dst_shape_rc=None,
            src_shape_rc=None, dst_shape_rc=None,
            bk_dxdy=None, fwd_dxdy=None, pt_buffer=100, shift_xy=None):
    """
    Warp xy points using M and/or bk_dxdy/fwd_dxdy. If bk_dxdy is provided, it will be inverted to  create fwd_dxdy

    Parameters
    ----------
    geom : shapely.geometery
        Shapely geom to warp

    M : ndarray, optional
         3x3 affine transformation matrix to perform rigid warp

    transformation_src_shape_rc : (int, int)
        Shape of image that was used to find the transformation.
        For example, this could be the original image in which features were detected

    transformation_dst_shape_rc : (int, int), optional
        Shape of the image with shape `transformation_src_shape_rc` after warping.
        This could be the shape of the original image after applying `M`.

    src_shape_rc : optional, (int, int)
        Shape of the image from which the points originated. For example,
        this could be a larger/smaller version of the image that was
        used for feature detection.

    dst_shape_rc : optional, (int, int)
        Shape of image (with shape `src_shape_rc`) after warping

    bk_dxdy : ndarray, pyvips.Image
        (2, N, M) numpy array of pixel displacements in the x and y
        directions from the reference image. dx = bk_dxdy[0],
        and dy=bk_dxdy[1]. If `bk_dxdy` is not None, but
        `fwd_dxdy` is None, then `bk_dxdy` will be inverted to warp `xy`.

    fwd_dxdy : ndarray, pyvips.Image
        Inverse of bk_dxdy. dx = fwd_dxdy[0], and dy=fwd_dxdy[1].
        This is what is actually used to warp the points.

    pt_buffer : int
        If `bk_dxdy` or `fwd_dxdy` are pyvips.Image object, then
        pt_buffer` determines the size of the window around the point used to
        get the local displacements.

    shift_xy : tuple of int, optional
        How much to shift the geom after being warped

    Returns
    -------
    warped_geom : shapely.geom
       Warped `geom`

    """

    warp_kwargs = {"M":M,
                   "transformation_src_shape_rc": transformation_src_shape_rc,
                   "transformation_dst_shape_rc": transformation_dst_shape_rc,
                   "src_shape_rc": src_shape_rc,
                   "dst_shape_rc": dst_shape_rc,
                   'bk_dxdy': bk_dxdy,
                   "fwd_dxdy": fwd_dxdy,
                   "pt_buffer": pt_buffer}

    if shift_xy is not None:
        shift_xy = np.array(shift_xy)

    warped_geom = _warp_shapely(geom, warp_xy, warp_kwargs, shift_xy)

    return warped_geom



def warp_shapely_geom_from_to(geom, from_M=None, from_transformation_src_shape_rc=None,
                   from_transformation_dst_shape_rc=None, from_src_shape_rc=None,
                   from_dst_shape_rc=None,from_bk_dxdy=None, from_fwd_dxdy=None,
                   to_M=None, to_transformation_src_shape_rc=None,
                   to_transformation_dst_shape_rc=None, to_src_shape_rc=None,
                   to_dst_shape_rc=None, to_bk_dxdy=None, to_fwd_dxdy=None):
    """
    Warp xy points using M and/or bk_dxdy/fwd_dxdy. If bk_dxdy is provided, it will be inverted to  create fwd_dxdy

    Parameters
    ----------
    geom : shapely.geometery
        Shapely geom to warp

    M : ndarray, optional
         3x3 affine transformation matrix to perform rigid warp

    transformation_src_shape_rc : (int, int)
        Shape of image that was used to find the transformation.
        For example, this could be the original image in which features were detected

    transformation_dst_shape_rc : (int, int), optional
        Shape of the image with shape `transformation_src_shape_rc` after warping.
        This could be the shape of the original image after applying `M`.

    src_shape_rc : optional, (int, int)
        Shape of the image from which the points originated. For example,
        this could be a larger/smaller version of the image that was
        used for feature detection.

    dst_shape_rc : optional, (int, int)
        Shape of image (with shape `src_shape_rc`) after warping

    bk_dxdy : ndarray, pyvips.Image
        (2, N, M) numpy array of pixel displacements in the x and y
        directions from the reference image. dx = bk_dxdy[0],
        and dy=bk_dxdy[1]. If `bk_dxdy` is not None, but
        `fwd_dxdy` is None, then `bk_dxdy` will be inverted to warp `xy`.

    fwd_dxdy : ndarray, pyvips.Image
        Inverse of bk_dxdy. dx = fwd_dxdy[0], and dy=fwd_dxdy[1].
        This is what is actually used to warp the points.

    pt_buffer : int
        If `bk_dxdy` or `fwd_dxdy` are pyvips.Image object, then
        pt_buffer` determines the size of the window around the point used to
        get the local displacements.


    Returns
    -------
    warped_geom : shapely.geom
       Warped `geom`

    """

    warp_kwargs = {"from_M": from_M,
                   "from_transformation_src_shape_rc": from_transformation_src_shape_rc,
                   "from_transformation_dst_shape_rc": from_transformation_dst_shape_rc,
                   "from_src_shape_rc": from_src_shape_rc,
                   "from_dst_shape_rc":from_dst_shape_rc,
                   "from_bk_dxdy":from_bk_dxdy,
                   "from_fwd_dxdy":from_fwd_dxdy,
                   "to_M":to_M,
                   "to_transformation_src_shape_rc": to_transformation_src_shape_rc,
                   "to_transformation_dst_shape_rc": to_transformation_dst_shape_rc,
                   "to_src_shape_rc": to_src_shape_rc,
                   "to_dst_shape_rc": to_dst_shape_rc, "to_bk_dxdy": to_bk_dxdy,
                   "to_fwd_dxdy":to_fwd_dxdy}

    warped_geom = _warp_shapely(geom, warp_xy_from_to, warp_kwargs)

    return warped_geom


def get_inside_mask_idx(xy, mask):
    """Remove points outside of mask

    Remove points that are outside of the mask

    Parameters
    ----------
    xy : ndarray
        (P, 2) array containing P points (xy coordinates)

    mask : ndarray
        (N, M) unit8 array  where 255 indicates the region of interest
        0 indicates background

    Returns
    -------
    inside_mask_idx : ndarray
        (Q) array containing the indices of points inside the mask.

    """
    mask_cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    inside_mask = np.array([cv2.pointPolygonTest(mask_cnt[0],
                                                    tuple(xy[i]),
                                                    False)
                            for i in range(xy.shape[0])])

    inside_mask_idx = np.where(inside_mask == 1.0)[0]

    return inside_mask_idx


def mask2xy(mask):
    if mask.ndim > 2:
        mask_y, mask_x = np.where(np.all(mask > 0, axis=2))
    else:
        mask_y, mask_x = np.where(mask > 0)
    min_x = np.min(mask_x)
    max_x = np.max(mask_x)
    min_y = np.min(mask_y)
    max_y = np.max(mask_y)

    bbox = np.array([
        [min_x, min_y],
        [max_x+1, min_y],
        [max_x+1, max_y+1],
        [min_x, max_y+1]
    ])

    return bbox


def bbox2mask(x, y, w, h, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y:y+h+1, x:x+w+1] = 255

    return mask


def xy2bbox(xy):
    min_x = np.min(xy[:, 0])
    max_x = np.max(xy[:, 0])
    min_y = np.min(xy[:, 1])
    max_y = np.max(xy[:, 1])
    w = abs(max_x - min_x)
    h = abs(max_y - min_y)

    return(np.array([min_x, min_y, w, h]))


def bbox2xy(xywh):
    """
    Get xy coordinates of bounding box, clockwise from top-left, i.e. TL, TR, BR, BL

    Parameters
    -----------
    xywh: [4, ] array
        (top left x-coordinate, top left y coordiante, width, height) of a bounding box

    Returns
    -------
     bbox_xy: [4, 2] array
        XY coordinates of bounding box, clockwise from top-left, i.e. TL, TR, BR, BL


    Example
    -------
    xywh = [10, 12, 5, 5]
    bbox_corners = bbox2xy(xywh)
    """
    x, y, w, h = xywh
    tl = [x, y]
    tr = [x + w, y]
    br = [x + w, y + h]
    bl = [x, y + h]
    bbox_xy = np.array([tl, tr, br, bl])

    return bbox_xy


def get_xy_inside_mask(xy, mask):
    """Get indices of `xy` that are inside `mask`

    Parameters
    ----------
    xy : ndarray
        [N, 2] array of coordinates to check

    mask : ndarray
        Binary image where 255 is considered foreground


    Returns
    -------
    keep_idx : ndarray
        Indices of `xy` that are inside of `mask`
    """

    mask_cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_polys = [shapely.geometry.Polygon(np.squeeze(cnt)) for cnt in mask_cnt if len(cnt) > 2]
    in_mask = np.zeros(xy.shape[0])
    for i, pt_xy in enumerate(xy):
        pt = shapely.geometry.Point(pt_xy)
        for poly in mask_polys:
            if poly.within(pt) or poly.contains(pt):
                in_mask[i] = 1
                break

    keep_idx = np.where(in_mask > 0)[0]


    # draw_img = np.dstack([mask]*3)
    # from skimage import draw
    # for i in range(xy.shape[0]):
    #     circ_pos = draw.disk(xy[i][::-1], radius=3)
    #     if i in keep_idx:
    #         clr = [0, 255, 0]
    #     else:
    #         clr = [255, 0, 0]

    #     draw_img[circ_pos] = clr

    # io.imsave(os.path.join(registrar.dst_dir, f"{slide_obj.name}_pt.png"), draw_img)


    return keep_idx


def calc_total_error(error):
    """
    Calculate error for alignments. Average error, weighted by proximiity to center of stack.
    Errors towards the center of the stack should carry greater weight, because they throw off a larger number of slices
    than errors in slides closer to the ends.

    """
    n = len(error)
    mid_pt = n / 2
    # Errors in middle carry larger weight, since it throws off other half
    dist_from_center = n - (np.abs(np.arange(0, n) - mid_pt))
    error_weights = dist_from_center / dist_from_center.sum()
    weighted_error = np.average(error, weights=error_weights)
    return weighted_error


def measure_error(src_xy, dst_xy, shape, feature_similarity=None):
    """
    Calculates the relative Target Registration Error (rTRE) and median Euclidean distance between a set of corresponding
    points (https://anhir.grand-challenge.org/Performance_Metrics/). If feature_similarity is not None, then
    distances are weighted by feature similarity. More similar features should ideally be closer together.

    Parameters
    ----------
    src_xy : [N, 2] array
        XY coordinates of features in src image. Each element should correspond to a matching feature coordinate in dst_xy

    dst_xy : [N, 2] array
            XY coordinates of features dst image. Each element should correspond to a matching feature coordinate in src_xy

    shape: (int, int)
        number of rows and columns in the image. Should be same for src and dst images

    feature_similarity: optional, [N]
        similarity of corresponding features in src image and dst image. Used to weight the median distance

    Returns
    -------
    med_tre : float
        Median relative Target Registration Error (rTRE) between images

    med_d : float
        Median Euclidean distance between src_xy and dst_xy, optinally weighted by feature similarity

    """
    d = np.sqrt((src_xy[:, 0]-dst_xy[:, 0])**2 + (src_xy[:, 1]-dst_xy[:, 1])**2)
    rtre = d/np.sqrt(np.sum(np.power(shape, 2)))
    med_tre = np.median(rtre)

    if feature_similarity is not None:
        med_d = weightedstats.weighted_median(d.tolist(), feature_similarity.tolist())
    else:
        med_d = np.median(d)


    return med_tre, med_d


def scale_M(M, scale_x, scale_y):
    """Scale transformation matrix

    http://answers.opencv.org/question/26173/the-relationship-between-homography-matrix-and-scaling-images/

    Parameters
    ----------
    M : ndarray
        3x3 transformation matrix

    scale_x : float
        How much to scale the transformation along the x-axis

    scale_y : float
        How much to scale the transformation along the y-axis

    Returns
    -------
    scaled_M : ndarray
        3x3 transformation matrix for use in an image with a
        different shape

    """
    S = np.identity(3)
    S[0, 0] = scale_x
    S[1, 1] = scale_y
    scaled_M = S @ M @ np.linalg.inv(S)
    return scaled_M


def get_overlapping_poly(mesh_poly_coords):
    """Clips mesh faces that overlap

    mesh_poly_coords : list of ndarray
        List of poylgon vertices for each mesh faces.

    """
    buffer_v = 0.01
    poly_l = [Polygon(verts).buffer(-buffer_v) for verts in np.round(mesh_poly_coords, 2)]
    s = STRtree(poly_l)
    n_poly = len(poly_l)
    overlapping_poly_list = []
    poly_diffs = []

    def clip_poly(i):
        poly = poly_l[i]
        if not poly.is_valid:
            overlapping_poly_list.append(poly.buffer(buffer_v))
            return None

        others = unary_union([p for p in s.query(poly) if p != poly and p.is_valid])
        intersection = poly.intersection(others)

        if intersection.area != 0:

            overlapping_poly_list.append(poly)
            diff = others.difference(poly)
            if isinstance(diff, MultiPolygon):
                for g in diff.geoms:
                    poly_diffs.append(g.buffer(buffer_v))
            else:
                poly_diffs.append(diff.buffer(buffer_v))

    n_cpu = multiprocessing.cpu_count() - 1
    with parallel_backend("threading", n_jobs=n_cpu):
        Parallel()(delayed(clip_poly)(i) for i in tqdm.tqdm(range(n_poly)))

    return overlapping_poly_list, poly_diffs


def untangle(dxdy, n_grid_pts=50, penalty=10e-6, mask=None):
    """Remove tangles caused by 2D displacement
    Based on method described in
    "Foldover-free maps in 50 lines of code" Garanzha et al. 2021.

    Parameters
    ----------
    dxdy : ndarray
        2xMxN array of displacement fields

    n_grid_pts : int, optional
        Number of grid points to sample, in each dimension

    penalty : float
        How much to penalize tangles

    mask : ndarray
        Mask indicating which areas should be untangled

    Returns
    -------
    untangled_dxdy : ndarray
        Copy of `dxdy`, but with displacements adjusted so that they
        won't introduce tangles.

    """

    qut = QuadUntangler(dxdy, n_grid_pts=n_grid_pts, fold_penalty=penalty)
    mesh = qut.mesh
    if mask is not None:
        frozen_mask = mask.copy()
        if np.any(mask.shape != mesh.padded_shape):
            padding_T = get_padding_matrix(mask.shape, mesh.padded_shape)
            frozen_mask = transform.warp(frozen_mask, padding_T, output_shape=mesh.padded_shape, preserve_range=True)
        # Freeze regions that aren't folded
        frozen_mask[0:frozen_mask.shape[0]-1, 0:mesh.c_offset] = 0 # left
        frozen_mask[0:frozen_mask.shape[0]-1, frozen_mask.shape[1]-mesh.c_offset : frozen_mask.shape[1]-1] = 0 # right
        frozen_mask[0:mesh.r_offset, 0:frozen_mask.shape[1]-1] = 0 # top
        frozen_mask[frozen_mask.shape[0]-mesh.r_offset : frozen_mask.shape[0] - 1, 0:frozen_mask.shape[1]-1] = 0 # bottom
        frozen_point = frozen_mask[mesh.sample_pos_xy[:, 1].astype(int),
                                   mesh.sample_pos_xy[:, 0].astype(int)].reshape(-1) == 0

        qut.mesh.boundary = frozen_point


    untangled_mesh = qut.untangle()
    qut.mesh.x = untangled_mesh
    untangled_coords = np.dstack([untangled_mesh[:mesh.nverts], untangled_mesh[mesh.nverts:]])[0]
    untangled_coords *= mesh.scaling
    untangled_dx = (mesh.sample_pos_xy[:, 0] - untangled_coords[:, 0]).reshape((mesh.nr, mesh.nc))
    untangled_dy = (mesh.sample_pos_xy[:, 1] - untangled_coords[:, 1]).reshape((mesh.nr, mesh.nc))

    padded_shape = mesh.padded_shape
    grid = UCGrid((0.0, float(padded_shape[1]), int(mesh.nc)),
                  (0.0, float(padded_shape[0]), int(mesh.nr)))

    dx_cubic_coeffs = filter_cubic(grid, untangled_dx).T
    dy_cubic_coeffs = filter_cubic(grid, untangled_dy).T

    img_y, img_x = np.indices(padded_shape)
    img_xy = np.dstack([img_x.reshape(-1), img_y.reshape(-1)]).astype(float)[0]
    untangled_dx = eval_cubic(grid, dx_cubic_coeffs, img_xy).reshape(padded_shape)
    untangled_dy = eval_cubic(grid, dy_cubic_coeffs, img_xy).reshape(padded_shape)

    inv_T = np.linalg.inv(mesh.padding_T)
    untangled_dx = transform.warp(untangled_dx, inv_T, output_shape=mesh.shape_rc, preserve_range=True)
    untangled_dy = transform.warp(untangled_dy, inv_T, output_shape=mesh.shape_rc, preserve_range=True)
    untangled_dxdy = np.array([untangled_dx, untangled_dy])

    return untangled_dxdy


def remove_folds_in_dxdy(dxdy, n_grid_pts=50, method="inpaint", paint_size=5000, fold_penalty=1e-6):
    """Remove folds in displacement fields

    Find and remove folds in displacement fields

    Parameters
    ---------
    method : str, optional
        "inpaint" will use inpainting to fill in areas idenetified
        as containing folds. "regularize" will unfold those regions
        using the mehod described in "Foldover-free maps in 50 lines of code"
        Garanzha et al. 2021.

    n_grid_pts : int
        Number of gridpoints used to detect folds. Also the number
        of gridpoints to use when regularizing he mesh when
        `method` = "regularize".

    paint_size : int
        Used to determine how much to resize the image to have efficient inpainting.
        Larger values = longer processing time. Only used if `method` = "inpaint".

    fold_penalty : float
        How much to penalize folding/stretching. Larger values will make
        the deformation field more uniform. Only used if `method` = "regularize"

    Returns
    -------
    no_folds_dxdy : ndarray
        An array containing the x-axis (column) displacement, and y-axis (row)
        displacement after removing folds.

    """

    # Use triangular mesh to find regions with folds
    # TriangleMesh will warp triangle points using dxdy to determine location vertices in warped image
    # It is assumed dxdy is a backwards transform found by registering images.
    # Because TriMesh is warping points, the inverse of dxdy is used.
    # Any image create from these points can be warped to their original position using dxdy

    valtils.print_warning("Looking for folds", None, rgb=Fore.YELLOW)
    tri_mesh = TriangleMesh(dxdy, n_grid_pts)
    padded_shape = tri_mesh.padded_shape

    tri_verts_xy = np.dstack([tri_mesh.x[:tri_mesh.nverts], tri_mesh.x[tri_mesh.nverts:]])[0]*tri_mesh.scaling
    tri_xy = np.array([tri_verts_xy[t, :] for t in tri_mesh.tri])

    overlapping_poly_list, poly_diff_list = get_overlapping_poly(tri_xy)
    poly_overlap_mask = np.zeros(padded_shape, dtype=np.uint8)
    for poly in overlapping_poly_list:
        poly_r, poly_c = draw.polygon(*poly.exterior.xy[::-1], shape=padded_shape)
        poly_overlap_mask[poly_r, poly_c] = 255

    # Warp mask back to original image. Should isolaate regions that will cause folding
    warp_map = get_warp_map(dxdy=tri_mesh.padded_dxdy)
    src_folds_mask = transform.warp(poly_overlap_mask, warp_map, preserve_range=True)
    src_folds_mask[src_folds_mask != 0] = 255
    src_folds_mask = ndimage.binary_fill_holes(src_folds_mask).astype(np.uint8)*255

    folded_area = len(np.where(src_folds_mask > 0)[0])
    if folded_area == 0:
        return dxdy

    if method == 'regularize':
        valtils.print_warning("Removing folds using regularizaation", None, rgb=Fore.YELLOW)
        # Untanlge folded regions using regularization
        qut = QuadUntangler(dxdy, n_grid_pts=n_grid_pts, fold_penalty=fold_penalty)
        mesh = qut.mesh
        frozen_mask = src_folds_mask.copy()
        # Freeze regions that aren't folded
        frozen_mask[0:frozen_mask.shape[0]-1, 0:mesh.c_offset] = 0 # left
        frozen_mask[0:frozen_mask.shape[0]-1, frozen_mask.shape[1]-mesh.c_offset : frozen_mask.shape[1]-1] = 0 # right
        frozen_mask[0:mesh.r_offset, 0:frozen_mask.shape[1]-1] = 0 # top
        frozen_mask[frozen_mask.shape[0]-mesh.r_offset : frozen_mask.shape[0] - 1, 0:frozen_mask.shape[1]-1] = 0 # bottom
        frozen_point = frozen_mask[mesh.sample_pos_xy[:, 1].astype(int),
                                   mesh.sample_pos_xy[:, 0].astype(int)].reshape(-1) == 0

        qut.mesh.boundary = frozen_point

        # Untangle and interpolate
        untangled_mesh = qut.untangle()
        qut.mesh.x = untangled_mesh
        untangled_coords = np.dstack([untangled_mesh[:mesh.nverts], untangled_mesh[mesh.nverts:]])[0]
        untangled_coords *= mesh.scaling
        untangled_dx = (mesh.sample_pos_xy[:, 0] - untangled_coords[:, 0]).reshape((mesh.nr, mesh.nc))
        untangled_dy = (mesh.sample_pos_xy[:, 1] - untangled_coords[:, 1]).reshape((mesh.nr, mesh.nc))

        grid = UCGrid((0.0, float(padded_shape[1]), int(mesh.nc)),
                      (0.0, float(padded_shape[0]), int(mesh.nr)))

        dx_cubic_coeffs = filter_cubic(grid, untangled_dx).T
        dy_cubic_coeffs = filter_cubic(grid, untangled_dy).T

        img_y, img_x = np.indices(padded_shape)
        img_xy = np.dstack([img_x.reshape(-1), img_y.reshape(-1)]).astype(float)[0]
        no_folds_dx = eval_cubic(grid, dx_cubic_coeffs, img_xy).reshape(padded_shape)
        no_folds_dy = eval_cubic(grid, dy_cubic_coeffs, img_xy).reshape(padded_shape)

    else:

        s = np.sqrt(paint_size)/np.sqrt(folded_area)
        if s > 1:
            s = 1

        inpaint_mask = transform.rescale(src_folds_mask, s, preserve_range=True)

        to_paint_dx = transform.rescale(tri_mesh.padded_dxdy[0], s, preserve_range=True)
        painted_dx = restoration.inpaint_biharmonic(to_paint_dx, inpaint_mask)
        smooth_dx = transform.resize(painted_dx, tri_mesh.padded_shape, preserve_range=True)

        to_paint_dy = transform.rescale(tri_mesh.padded_dxdy[1], s, preserve_range=True)
        painted_dy = restoration.inpaint_biharmonic(to_paint_dy, inpaint_mask)
        smooth_dy = transform.resize(painted_dy, tri_mesh.padded_shape, preserve_range=True)

        blending_mask = filters.gaussian(src_folds_mask, 1)
        no_folds_dx = blending_mask*smooth_dx + (1-blending_mask)*tri_mesh.padded_dxdy[0]
        no_folds_dy = blending_mask*smooth_dy + (1-blending_mask)*tri_mesh.padded_dxdy[1]

    # Crop to original shape #
    no_folds_dx = transform.warp(no_folds_dx, inv_T, output_shape=tri_mesh.shape_rc, preserve_range=True)
    no_folds_dy = transform.warp(no_folds_dy, inv_T, output_shape=tri_mesh.shape_rc, preserve_range=True)
    no_folds_dxdy = np.array([no_folds_dx, no_folds_dy])

    return no_folds_dxdy


class QuadMesh(object):

    def __init__(self, dxdy, n_grid_pts=50):
        shape = np.array(dxdy[0].shape)
        self.shape_rc = shape
        grid_spacing =  int(np.min(np.round(shape/n_grid_pts)))

        new_r = shape[0] - shape[0] % grid_spacing + grid_spacing
        self.r_padding = new_r - shape[0]
        sample_y = np.floor(np.arange(0, new_r + grid_spacing, grid_spacing))

        new_c = shape[1] - shape[1] % grid_spacing + grid_spacing
        sample_x = np.arange(0, new_c + grid_spacing, grid_spacing)
        self.c_padding = new_c - shape[1]

        nr = len(sample_y)
        nc = len(sample_x)
        padded_shape = np.array([new_r+1, new_c+1])
        self.padded_shape = padded_shape
        y_center, x_center = padded_shape/2
        self.nverts = nr*nc
        self.nr = nr
        self.nc = nc

        self.r_offset, self.c_offset = (padded_shape - shape)//2

        # Pad displacement #
        self.padding_T = get_padding_matrix(shape, padded_shape)

        padded_dx = transform.warp(dxdy[0], self.padding_T, output_shape=padded_shape, preserve_range=True)
        padded_dy = transform.warp(dxdy[1], self.padding_T, output_shape=padded_shape, preserve_range=True)

        self.padded_dxdy = np.array([padded_dx, padded_dy])
        # Flattend indices for each pixel in a quadrat
        quads = [[r*nc + c, r*nc + c + 1, (r+1)*nc + c + 1, (r+1)*nc + c] for r in range(nr-1) for c in range(nc-1)]
        self.quads = quads
        self.boundary = [None] * self.nverts

        for i in range(self.nverts):
            r_idx = i // nc
            c_idx = i % nc
            r = sample_y[r_idx]
            c = sample_x[c_idx]
            if r <= y_center or r >= new_r - y_center or c <= x_center or c >= new_c - x_center:
                self.boundary[i] = True

            else:
                self.boundary[i] = False

        sample_pos_y, sample_pos_x = np.meshgrid(sample_y, sample_x, indexing="ij")
        unwarped_xy = np.dstack([sample_pos_x.reshape(-1), sample_pos_y.reshape(-1)])[0].astype(float)
        self.sample_pos_xy = unwarped_xy
        sample_xy = warp_xy(unwarped_xy, M=None, bk_dxdy=[padded_dx, padded_dy])
        self.warped_xy = sample_xy
        scaled_coords = self.scale_coords(sample_xy)
        self.x = np.hstack([scaled_coords[:, 0], scaled_coords[:, 1]])

    def scale_coords(self, xy):
        max_side = np.max(self.padded_shape)
        scaled_coords = xy/max_side
        self.scaling = max_side

        return scaled_coords


    def __str__(self):
        ret = ""
        for v in range(self.nverts):
            ret = ret + ("v %f %f 0\n" % (self.x[v], self.x[v+self.nverts]))
        for f in self.quads:
            ret = ret + ("f %d %d %d %d\n" % (f[0]+1, f[1]+1, f[2]+1, f[3]+1))
        return ret

    def show(self):
        res = 1000
        off = 100
        image = Image.new(mode='L', size=(res, res), color=255)
        draw = ImageDraw.Draw(image)

        for quad in self.quads:
            for e in range(4):
                i = quad[e]
                j = quad[(e+1)%4]

                line = ((off+self.x[i]*res/2, off+self.x[i+self.nverts]*res/2), (off+self.x[j]*res/2, off+self.x[j+self.nverts]*res/2))
                draw.line(line, fill=128)
        del draw
        image.show()


class TriangleMesh(object):
    def __init__(self, dxdy, n_grid_pts=50):
        shape = np.array(dxdy[0].shape)
        self.shape_rc = shape
        grid_spacing =  int(np.min(np.round(shape/n_grid_pts)))

        new_r = shape[0] - shape[0] % grid_spacing + grid_spacing
        self.r_padding = new_r - shape[0]
        sample_y = np.floor(np.arange(0, new_r + grid_spacing, grid_spacing))

        new_c = shape[1] - shape[1] % grid_spacing + grid_spacing
        sample_x = np.arange(0, new_c + grid_spacing, grid_spacing)
        self.c_padding = new_c - shape[1]

        nr = len(sample_y)
        nc = len(sample_x)
        padded_shape = np.array([new_r+1, new_c+1])
        self.padded_shape = padded_shape
        self.r_offset, self.c_offset = (padded_shape - shape)//2

        self.nverts = nr*nc
        self.nr = nr
        self.nc = nc
        y_center, x_center = padded_shape/2

        self.padding_T = get_padding_matrix(shape, padded_shape)

        padded_dx = transform.warp(dxdy[0], self.padding_T, output_shape=padded_shape, preserve_range=True)
        padded_dy = transform.warp(dxdy[1], self.padding_T, output_shape=padded_shape, preserve_range=True)

        self.padded_dxdy = np.array([padded_dx, padded_dy])

        # Get triangle vertices
        sample_x = np.arange(0, new_c + grid_spacing, grid_spacing)
        sample_y = np.arange(0, new_r + grid_spacing, grid_spacing)

        tri_verts, tri_faces = get_triangular_mesh(sample_x, sample_y)
        self.nverts = tri_verts.shape[0]
        self.tri_verts = tri_verts
        self.boundary = [None] * self.nverts
        for i in range(self.nverts):
            c, r = tri_verts[i]

            if r <= y_center or r >= new_r - y_center or c <= x_center or c >= new_c - x_center:
                self.boundary[i] = True
            else:
                self.boundary[i] = False

        sample_xy = warp_xy(tri_verts, M=None, bk_dxdy=[padded_dx, padded_dy])
        self.warped_xy = sample_xy

        self.tri = tri_faces
        self.nfacets = len(self.tri)
        self.vert = self.scale_coords(sample_xy)
        self.x = np.hstack([self.vert[:, 0], self.vert[:, 1]])

    def scale_coords(self, xy):

        max_side = np.max(self.padded_shape)
        scaled_coords = xy/max_side
        self.scaling = max_side

        return scaled_coords


class QuadUntangler(object):
    def __init__(self, dxdy, fold_penalty=1e-6, n_grid_pts=50):
        self.shape = np.array(dxdy[0].shape)
        self.mesh = QuadMesh(dxdy, n_grid_pts)
        self.mesh_type = self.mesh.__class__.__name__
        self.n_grid_pts = n_grid_pts
        self.n = self.mesh.nverts
        self.fold_penalty = fold_penalty

    def untangle(self):
        n = self.n
        mesh = self.mesh
        Q = [np.matrix('-1,-1;1,0;0,0;0,1'), np.matrix('-1,0;1,-1;0,1;0,0'),  # quadratures for
             np.matrix('0,0;0,-1;1,1;-1,0'), np.matrix('0,-1;0,0;1,0;-1,1') ] # every quad corner

        def jacobian(U, qc, quad):
            return np.matrix([[U[quad[0]  ], U[quad[1]  ], U[quad[2]  ], U[quad[3]  ]],
                              [U[quad[0]+n], U[quad[1]+n], U[quad[2]+n], U[quad[3]+n]]]) * Q[qc]

        mindet = min([np.linalg.det( jacobian(mesh.x, qc, quad) ) for quad in mesh.quads for qc in range(4)])
        eps = np.sqrt(1e-6**2 + min(mindet, 0)**2) # the regularization parameter e
        eps *= 1/self.fold_penalty

        def energy(U): # compute the energy and its gradient for the map u
            F,G = 0, np.zeros(2*n)
            for quad in mesh.quads: # sum over all quads
                for qc in range(4): # evaluate the Jacobian matrix for every quad corner
                    J = jacobian(U, qc, quad)
                    det = np.linalg.det(J)
                    chi  = det/2 + np.sqrt(eps**2 + det**2)/2    # the penalty function
                    chip = .5 + det/(2*np.sqrt(eps**2 + det**2)) # its derivative

                    f = np.trace(np.transpose(J)*J)/chi # quad corner shape quality
                    F += f
                    dfdj = (2*J - np.matrix([[J[1,1],-J[1,0]],[-J[0,1],J[0,0]]])*f*chip)/chi
                    dfdu = Q[qc] * np.transpose(dfdj) # chain rule for the actual variables
                    for i,v in enumerate(quad):
                        if (mesh.boundary[v]): continue # the boundary verts are locked
                        G[v  ] += dfdu[i,0]
                        G[v+n] += dfdu[i,1]
            return F,G

        # factr are: 1e12 for low accuracy; 1e7 for moderate accuracy; 10.0 for extremely high accuracy.
        factr = 1e7
        untangled = fmin_l_bfgs_b(energy, mesh.x, factr=factr)[0] # inner L-BFGS loop

        return untangled


class _TriUntangler(object):
    def __init__(self, dxdy, n_grid_pts=50):
        self.shape = np.array(dxdy[0].shape)

        # self.mesh = QuadMesh(dxdy, n_grid_pts)
        self.mesh = TriangleMesh(dxdy, n_grid_pts)
        self.mesh_type = self.mesh.__class__.__name__
        self.n_grid_pts = n_grid_pts
        self.n = self.mesh.nverts
        self.n_tri = len(self.mesh.tri)

    def triangle_area2d(self, a, b, c):
        x = 0
        y = 1
        tri_area = .5*((b[y]-a[y])*(b[x]+a[x]) + (c[y]-b[y])*(c[x]+b[x]) + (a[y]-c[y])*(a[x]+c[x]))
        return tri_area

    def triangle_aspect_ratio_2d(self, a, b, c):

        l1 = np.linalg.norm(b-a)
        l2 = np.linalg.norm(c-b)
        l3 = np.linalg.norm(a-c)
        lmax = max([l1, l2, l3])

        return lmax*(l1+l2+l3)/(4.*np.sqrt(3.)*self.triangle_area2d(a, b, c))


    def setup(self):
        area = [None] * self.n_tri
        ref_tri = [None] * self.n_tri
        for t, faces in enumerate(self.mesh.tri):


            ax, bx, cx = self.mesh.x[self.mesh.tri[t]]
            ay, by, cy = self.mesh.x[self.mesh.tri[t]+ self.mesh.nverts]

            A = np.array([ax, ay])
            B = np.array([bx, by])
            C = np.array([cx, cy])


            area[t] = self.triangle_area2d(A, B, C)

            ar = self.triangle_aspect_ratio_2d(A, B, C)
            if ar > 10:
                #if the aspect ratio is bad, assign an equilateral reference triangle
                l1 = np.linalg.norm(B-A)
                l2 = np.linalg.norm(C-B)
                l3 = np.linalg.norm(A-C)
                a = (l1 + l2 + l3)/3 # edge length is the average of the original triangle
                area[t] = np.sqrt(3.)/4.*a*a
                A = np.array([0., 0.])
                B = np.array([a, 0.])
                C = np.array([a/2., np.sqrt(3.)/2.*a])

            ST = np.matrix([B-A, C-A])
            ST_invert_transpose = np.linalg.inv(ST).T
            ref_tri[t] = np.array([[-1, -1], [1, 0], [0, 1] ]) @ ST_invert_transpose

        self.area = area
        self.ref_tri = ref_tri


    def untangle(self):
        self.setup()
        def evaluate_jacobian(X, t):
            J = np.matrix(np.zeros((2, 2)))
            for i in range(3):
                for d in range(2):
                    J[d] += self.ref_tri[t][i, d] + X[self.mesh.tri[t][i] + self.n*d]

            K = np.array([[J[1, 1], -J[1, 0]],
                         [-J[0, 1], J[0, 0]]
            ])

            det = np.linalg.det(J)

            return J, K, det

        mindet = np.inf
        for t in range(self.n_tri):
            _, _, det = evaluate_jacobian(self.mesh.x, t)
            mindet = np.min([mindet, det])

        eps = np.sqrt(1e-6**2 + min(mindet, 0)**2) # the regularization parameter e
        theta = 1./128

        def chi(eps, det):
            if det < 0:
                return (det + np.sqrt(eps*eps + det*det) + 10**-6)*.5
            else:
                return .5*eps*eps / (np.sqrt(eps*eps + det*det) - det + 10**-6)

        def chi_deriv(eps, det):
            return .5+det/(2.*np.sqrt(eps*eps + det*det + 10**-6))


        def energy(U):


            F,G = 0, np.zeros(2*self.n)

            for t in range(self.n_tri):
                J, K, det = evaluate_jacobian(U, t)

                c1 = chi(eps, det)
                c2 = chi_deriv(eps, det)

                f = np.trace(np.transpose(J)*J)/c1 # corner shape quality
                g = (1+det*det)/c1

                F += ((1-theta)*f + theta*g)*self.area[t]

                for dim in range(2):

                    a = J[dim] # tangent basis
                    b = K[dim] # dual basis
                    dfda = (a*2. - b*f*c2)/c1
                    dgda = b*(2*det-g*c2)/c1
                    for i in range(3):
                        v = self.mesh.tri[t][i]
                        if self.mesh.boundary[v]: continue # the boundary verts are locked
                        # og_pos = G[v+ dim*self.n]
                        G[v+ dim*self.n] += (self.ref_tri[t][i] @ np.transpose(dfda*(1.-theta) + dgda*theta))*self.area[t]
                        # new_pos = G[v+ dim*self.n]
                        # print(new_pos - og_pos)
            return F, G

        n_iter = 3

        for i in range(n_iter):

            self.mesh.x = fmin_l_bfgs_b(energy, self.mesh.x, factr=1e12)[0] # inner L-BFGS loop
            # updated_xy = self.mesh.x.reshape((self.n, 2))
            updated_xy = np.dstack([self.mesh.x[self.n:], self.mesh.x[:self.n]])[0]
            # plt.triplot(updated_xy[:, 0], -updated_xy[:, 1], self.mesh.tri, linewidth=0.5)
            plt.triplot(updated_xy[:, 1], -updated_xy[:, 0], self.mesh.tri, linewidth=0.5)
            plt.axis("equal")
            plt.savefig(f"{i}_smooth_mesh.png")
            plt.close()

