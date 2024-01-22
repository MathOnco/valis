"""
Testing warp_img_from_to, which uses both forward and backwared warps

*. warp 2 images. These are the "unregistered versions"
*. Inverse of above transfroms are the registration parameters
*. use warp_img_from_to on the "unregistered" image1.
*. Resulting image should line up with the other "unregistered one"

*. Do the same using corners that were detected. So can test point warping

"""

from skimage import io, transform, draw
import sys
import os
import numpy as np
import pathlib
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from valis import viz, warp_tools
# valis_src_dir = '/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis/'
# sys.path.append(valis_src_dir)


def get_parent_dir():
    cwd = os.getcwd()
    dir_split = cwd.split(os.sep)
    split_idx = [i for i in range(len(dir_split)) if dir_split[i] == "valis_project"][0]
    parent_dir = os.sep.join(dir_split[:split_idx+1])
    return parent_dir


parent_dir = get_parent_dir()
in_container = sys.platform == "linux" and os.getcwd() == '/usr/local/src'
if in_container:
    results_dst_dir = os.path.join(parent_dir, f"valis/tests/docker")
else:
    results_dst_dir = os.path.join(parent_dir, f"valis/tests/{sys.version_info.major}{sys.version_info.minor}")

def gen_M(img_shape_rc, txy=(0, 0), sxy=(1,1), rot_deg=0):


    tform = transform.SimilarityTransform(scale=sxy, rotation=np.deg2rad(rot_deg), translation=txy)
    img_corners_xy = warp_tools.get_corners_of_image(img_shape_rc[0:2])[:, ::-1]
    warped_corners = warp_tools.warp_xy(img_corners_xy, M=tform.params)
    bbox_xywh = warp_tools.xy2bbox(warped_corners)

    T = np.eye(3)
    T[0:2, 2] = bbox_xywh[0:2]
    M = tform.params @ T

    shape_rc = np.ceil(bbox_xywh[2:][::-1]).astype(int)

    return M, shape_rc


def gen_wave(n, signal_fxn, amp=10, period=0.5, phase_shift=0, v_shift=0):

    # https://www.mathsisfun.com/algebra/amplitude-period-frequency-phase-shift.html

    b = 2*np.pi/period
    x = np.linspace(-1 , 1, n)
    y = amp*signal_fxn(b*(x+phase_shift)) + v_shift
    return y


def gen_dxdy(img_shape, wave_fxn, amp_range=(0, 10), phase_range=(0, 1), period_range = (0, 1), shift_range=None):

    amp = np.random.uniform(*amp_range)
    phase = np.random.uniform(*phase_range)
    period = np.random.uniform(*period_range)
    if shift_range is None:
        v_shift_range = (0, img_shape[1])
        h_shift_range = (0, img_shape[0])
    else:
        v_shift_range = shift_range
        h_shift_range = shift_range


    if wave_fxn == np.sin:
        wave_fxn2 = np.cos

    elif wave_fxn == np.cos:
        wave_fxn2 = np.sin

    v_shift = np.random.uniform(*v_shift_range)
    h_shift = np.random.uniform(*h_shift_range)

    v_dx_img = np.vstack([gen_wave(img_shape[0], wave_fxn, amp=amp, phase_shift=phase, period=period, v_shift=v_shift) for i in range(img_shape[1])]).T
    v_dy_img = np.vstack([gen_wave(img_shape[0], wave_fxn2, amp=amp, phase_shift=phase, period=period, v_shift=v_shift) for i in range(img_shape[1])]).T

    h_dx_img = np.vstack([gen_wave(img_shape[1], wave_fxn, amp=amp, phase_shift=phase, period=period, v_shift=h_shift) for i in range(img_shape[0])])
    h_dy_img = np.vstack([gen_wave(img_shape[1], wave_fxn2, amp=amp, phase_shift=phase, period=period, v_shift=h_shift) for i in range(img_shape[0])])

    # grid = viz.color_displacement_grid(h_dx_img, h_dy_img, thickness=2, grid_spacing_ratio=0.01)

    dx = v_dx_img + h_dx_img
    dy = v_dy_img + h_dy_img

    return [dx, dy]


def get_points(img):
    """Get corners of image to test warping points
    """
    coords = corner_peaks(corner_harris(img), min_distance=5, threshold_rel=0.02)
    coords_subpix_rc = corner_subpix(img, coords, window_size=13)

    return coords_subpix_rc[:, ::-1]


def draw_pts(img, pt_xy):
    if img.ndim == 2:
        viz_img = np.dstack([img]*3)
    else:
        viz_img = img.copy()

    for pt in pt_xy:
        pt_rc = draw.disk(pt[::-1], 10, shape=img.shape)
        viz_img[pt_rc[0], pt_rc[1]] = [0, 255, 0]

    return viz_img


def test_img_warp(max_mi=0.5, max_px_d=0.1):

    img_f = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/write_up/logo/tri_logo/valis_logo.png"
    # dst_dir = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/tests/warp_from_to"
    # dst_dir = f"/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis/tests/tmp{sys.version_info.major}{sys.version_info.minor}/warp_functions"
    dst_dir = os.path.join(results_dst_dir, "warp_functions")
    pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)

    img = io.imread(img_f, True)
    img = (255*(1-img)).astype(np.uint8)
    img_shape = img.shape[0:2]

    pt_corners = get_points(img)
    img_w_pts = draw_pts(img, pt_corners)
    io.imsave(os.path.join(dst_dir, "img_og.png"), img_w_pts)

    M1, unreg_shape1 = gen_M(img.shape, (50, 100), (1.1, 1.1), 45)
    dxdy1 = gen_dxdy(img_shape=img_shape, wave_fxn=np.sin, amp_range=(10, 10), phase_range=(0.2, 0.2), period_range=(0.3, 0.3), shift_range=(0, 0))
    _unreg1 = warp_tools.warp_img(img, bk_dxdy=dxdy1)
    unreg1 = warp_tools.warp_img(_unreg1, M=M1, out_shape_rc=unreg_shape1)

    _unreg_pt1 = warp_tools.warp_xy(pt_corners, bk_dxdy=dxdy1)
    unreg_pt1 = warp_tools.warp_xy(_unreg_pt1, M=M1)

    viz1 = draw_pts(unreg1, unreg_pt1)
    io.imsave(os.path.join(dst_dir, "unreg1.png"), viz1)


    M2, unreg_shape2 = gen_M(img.shape, (25, 50), (0.9, 0.9), 225)
    dxdy2 = gen_dxdy(img_shape=img_shape, wave_fxn=np.cos, amp_range=(12, 12), phase_range=(0.2, 0.2), period_range=(0.4, 0.4), shift_range=(0, 0))
    _unreg2 = warp_tools.warp_img(img, bk_dxdy=dxdy2)
    unreg2 = warp_tools.warp_img(_unreg2, M=M2, out_shape_rc=unreg_shape2)

    _unreg_pt2 = warp_tools.warp_xy(pt_corners, bk_dxdy=dxdy2)
    unreg_pt2 = warp_tools.warp_xy(_unreg_pt2, M=M2)

    viz2 = draw_pts(unreg2, unreg_pt2)
    io.imsave(os.path.join(dst_dir, "unreg2.png"), viz2)

    # Inverse of above transforms would register the images (putting back to unwarped state)
    M1_reg = np.linalg.inv(M1)
    M1_bk_dxdy = warp_tools.get_inverse_field(dxdy1)

    M2_reg = np.linalg.inv(M2)
    M2_bk_dxdy = warp_tools.get_inverse_field(dxdy2)


    img_2_on_1_rigid = warp_tools.warp_img_from_to(unreg2,
                    from_M=M2_reg,
                    from_transformation_src_shape_rc=unreg_shape2,
                    from_transformation_dst_shape_rc=img_shape,
                    from_dst_shape_rc=None,
                    from_bk_dxdy=None,
                    to_M=M1_reg,
                    to_transformation_src_shape_rc=unreg_shape1,
                    to_transformation_dst_shape_rc=img_shape,
                    to_src_shape_rc=unreg_shape1,
                    to_bk_dxdy=None,
                    to_fwd_dxdy=None)

    pt2_on_1_rigid = warp_tools.warp_xy_from_to(unreg_pt2,
                                        from_M=M2_reg,
                                            from_transformation_src_shape_rc=unreg_shape2,
                                            from_transformation_dst_shape_rc=img_shape,
                                            from_dst_shape_rc=None,
                                            from_bk_dxdy=None,
                                            to_M=M1_reg,
                                            to_transformation_src_shape_rc=unreg_shape1,
                                            to_transformation_dst_shape_rc=img_shape,
                                            to_src_shape_rc=unreg_shape1,
                                            to_bk_dxdy=None,
                                            to_fwd_dxdy=None
                                            )

    viz_img_2_on_1_rigid = draw_pts(img_2_on_1_rigid, pt2_on_1_rigid)
    io.imsave(os.path.join(dst_dir, "2_on_1_rigid.png"), viz_img_2_on_1_rigid)

    img_2_on_1_non_rigid = warp_tools.warp_img_from_to(unreg2,
                    from_M=M2_reg,
                    from_transformation_src_shape_rc=unreg_shape2,
                    from_transformation_dst_shape_rc=img_shape,
                    from_dst_shape_rc=None,
                    from_bk_dxdy=M2_bk_dxdy,
                    to_M=M1_reg,
                    to_transformation_src_shape_rc=unreg_shape1,
                    to_transformation_dst_shape_rc=img_shape,
                    to_src_shape_rc=unreg_shape1,
                    to_bk_dxdy=M1_bk_dxdy
                    )

    pt2_on_1_non_rigid = warp_tools.warp_xy_from_to(unreg_pt2,
                                                    from_M=M2_reg,
                                                    from_transformation_src_shape_rc=unreg_shape2,
                                                    from_transformation_dst_shape_rc=img_shape,
                                                    from_dst_shape_rc=None,
                                                    from_bk_dxdy=M2_bk_dxdy,
                                                    to_M=M1_reg,
                                                    to_transformation_src_shape_rc=unreg_shape1,
                                                    to_transformation_dst_shape_rc=img_shape,
                                                    to_src_shape_rc=unreg_shape1,
                                                    to_bk_dxdy=M1_bk_dxdy)


    mi = warp_tools.mattes_mi(unreg1, img_2_on_1_non_rigid)
    pt_d = np.median(warp_tools.calc_d(pt2_on_1_non_rigid, unreg_pt1))
    # assert pt_d < max_px_d and mi < max_mi

    viz_img_2_on_1_non_rigid = draw_pts(img_2_on_1_non_rigid, pt2_on_1_non_rigid)
    io.imsave(os.path.join(dst_dir, "2_on_1_non_rigid.png"), viz_img_2_on_1_non_rigid)

    passed = pt_d < max_px_d and mi < max_mi

    assert passed


if __name__ == "__main__":
    test_img_warp()