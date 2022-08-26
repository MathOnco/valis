"""
Code used for the ACROBAT Grand Challenge

Shows how to create a custom image processor (AcrobatProcessor),
perform micro registration with a mask, warp points, and run
using command line arguments.

"""

import pathlib
import os
import shutil
from time import time
import argparse
import imghdr
import numpy as np
import pandas as pd
import re
from skimage import color as skcolor, draw, exposure, filters, morphology
import colour
import cv2

from valis import registration, valtils, warp_tools, viz, preprocessing
from valis.preprocessing import ColorfulStandardizer, DEFAULT_COLOR_STD_C

DRAW_IMG_SIZE = 500

def get_lines_img(img, v_ksize, h_ksize):

    v_krnl = np.ones((1, v_ksize))
    edges_no_v_lines = morphology.opening(img, v_krnl)

    h_krnl = np.ones((h_ksize, 1))
    edges_no_h_lines = morphology.opening(img, h_krnl)

    no_lines_img = np.dstack([edges_no_v_lines, edges_no_h_lines]).min(axis=2)
    lines_img = np.dstack([edges_no_v_lines, edges_no_h_lines]).max(axis=2)
    lines_diff = lines_img - no_lines_img

    return lines_diff


class AcrobatProcessor(ColorfulStandardizer):
    """Preprocess images for registration

    Standardizes colorfulness, removes black borders, and subtracts the background

    """

    def __init__(self, image, src_f, level, series, *args, **kwargs):
        super().__init__(image=image, src_f=src_f, level=level,
                         series=series, *args, **kwargs)


    def create_mask(self, od_clip=0.25, lines_diff_t=0.025):
        """
        Use optical density to create mask.

        Ended up working better without masks though
        """
        od_img = preprocessing.rgb2od(self.image)
        od_summary = od_img.min(axis=2)

        line_img = get_lines_img(od_summary, 15, 15)

        lines_mask = 255*(line_img > lines_diff_t).astype(np.uint8)
        very_dense = 255*(od_summary >= 0.1).astype(np.uint8)

        edge_artifacts = np.zeros_like(lines_mask)
        edge_artifacts[lines_mask > 0] = 255
        edge_artifacts[very_dense > 0] = 255
        _, edge_artifacts = preprocessing.create_edges_mask(edge_artifacts)

        mask = 255*(edge_artifacts == 0).astype(np.uint8)

        squashed_od = np.clip(od_summary, 0, od_clip)

        bg_od, _ = filters.threshold_multiotsu(squashed_od[mask > 0])

        bg_od = np.quantile(squashed_od[mask > 0], 0.5)
        fg = 255*(squashed_od > bg_od).astype(np.uint8)
        fg[mask == 0] = 0
        fg = preprocessing.mask2contours(fg, 0)
        fg_lines = get_lines_img(fg, 25, 25)
        fg[fg_lines > 0] = 0

        fg = preprocessing.remove_small_obj_and_lines_by_dist(fg)

        bbox_mask = preprocessing.mask2bbox_mask(fg)

        return bbox_mask

    def process_image(self, blk_thresh=0.75, c=DEFAULT_COLOR_STD_C, invert=True, *args, **kwargs):

        # Process image using default method
        std_rgb = preprocessing.standardize_colorfulness(self.image, c)
        std_g = skcolor.rgb2gray(std_rgb)

        if invert:
            std_g = 255 - std_g
        processed_img = exposure.rescale_intensity(std_g, in_range="image", out_range=(0, 255)).astype(np.uint8)

        # Detect black borders commonly found in this dataset
        cam_d, cam = preprocessing.calc_background_color_dist(self.image)
        cam_black = colour.convert(np.repeat(0, 3), 'sRGB', 'CAM16UCS')
        black_dist = np.sqrt(np.sum((cam - cam_black)**2, axis=2))
        dark_regions = 255*(black_dist < blk_thresh).astype(np.uint8)
        dark_contours, _ = cv2.findContours(dark_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        edge_artifact_mask = np.zeros_like(dark_regions)

        for cnt in dark_contours:
            cnt_xy = np.squeeze(cnt, 1)
            on_border_idx = np.where((cnt_xy[:, 0] == 0) |
                                     (cnt_xy[:, 0] == dark_regions.shape[1]-1) |
                                     (cnt_xy[:, 1] == 0) |
                                     (cnt_xy[:, 1] == dark_regions.shape[0]-1)
                                     )[0]

            if len(on_border_idx) > 0 :
                cv2.drawContours(edge_artifact_mask, [cnt], 0, 255, -1)


        eps = np.finfo("float").eps
        with colour.utilities.suppress_warnings(colour_usage_warnings=True):
            if 1 < self.image.max() <= 255 and np.issubdtype(self.image.dtype, np.integer):
                cam = colour.convert(self.image/255 + eps, 'sRGB', 'CAM16UCS')
            else:
                cam = colour.convert(self.image + eps, 'sRGB', 'CAM16UCS')

        # Subtract background
        brightest_thresh = np.quantile(cam[..., 0][edge_artifact_mask==0], 0.9)
        brightest_idx = np.where(cam[..., 0] >= brightest_thresh)
        brightest_pixels = processed_img[brightest_idx]
        brightest_rgb = np.median(brightest_pixels, axis=0)
        no_bg = processed_img - brightest_rgb
        no_bg = np.clip(no_bg, 0, 255)
        no_bg[edge_artifact_mask != 0] = 0

        # Adjust range and perform adaptive histogram equalization
        no_bg = (255*exposure.equalize_adapthist(no_bg/no_bg.max())).astype(np.uint8)

        return no_bg


def create_he_mask(he_img, j_range=(0.0, 0.9), c_range=(0.05, 1), h_range=(150, 275), h_rotation=270):
    """
    Segment H&E stain in the polar CAM16-UCS colorspace
    """
    jch = preprocessing.rgb2jch(he_img, h_rotation=h_rotation)

    he_mask = 255*( (jch[..., 0] >= j_range[0]) &
                (jch[..., 0] < j_range[1])  &
                (jch[..., 1] >= c_range[0]) &
                (jch[..., 1] < c_range[1])  &
                (jch[..., 2] >= h_range[0]) &
                (jch[..., 2] < h_range[1])
                ).astype(np.uint8)

    he_mask = preprocessing.mask2contours(he_mask)
    he_mask = preprocessing.remove_small_obj_and_lines_by_dist(he_mask)

    return he_mask


def create_reg_mask(reg, j_range=(0.05, 0.9), c_range=(0.05, 1), h_range=(150, 275), h_rotation=270):
    """
    Mask is the bounding bbox around the H&E+ tissue
    """

    img_names = list(reg.slide_dict.keys())
    he_img_name = [n for n in img_names if re.search("_HE_", n) is not None][0]

    he_slide = reg.get_slide(he_img_name)
    he_mask = create_he_mask(he_slide.image, j_range=j_range, c_range=c_range, h_range=h_range, h_rotation=h_rotation)
    nr_he_mask = he_slide.warp_img(he_mask, interp_method="nearest", crop=False)

    mask_bbox = warp_tools.xy2bbox(warp_tools.mask2xy(nr_he_mask))
    c0, r0 = mask_bbox[:2]
    c1, r1 = mask_bbox[:2] + mask_bbox[2:]
    reg_mask = np.zeros_like(nr_he_mask)
    reg_mask[r0:r1, c0:c1] = 255

    return reg_mask



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Register images for the ACROBAT Grand Challenge')
    parser.add_argument('-src_dir', type=str, help='source image to warp')
    parser.add_argument('-dst_dir', type=str, help='where to save results')
    parser.add_argument('-name', type=str, help='what to call the registrar')
    parser.add_argument('-micro_reg_fraction', type=str, help='Size of images used for 2nd registation')
    parser.add_argument('-landmarks_f', type=str, help='location of landmarks')
    args = vars(parser.parse_args())

    src_dir = args["src_dir"]
    dst_dir = args["dst_dir"]
    name = args["name"]
    micro_reg_fraction = args["micro_reg_fraction"]
    landmarks_f = args["landmarks_f"]

    micro_reg_fraction = eval(micro_reg_fraction)
    print(name)

    # Do initial registration, setting the reference image to be the H&E image
    img_list = [f for f in os.listdir(src_dir) if imghdr.what(os.path.join(src_dir, f)) is not None]
    img_list = [os.path.join(src_dir, f) for f in img_list]
    he_img_f = [f for f in img_list if re.search("_HE_", f) is not None][0]
    start = time()


    with valtils.HiddenPrints():
        registrar = registration.Valis(src_dir, dst_dir,
                                       crop="reference",
                                       reference_img_f=he_img_f,
                                       create_masks=False,
                                       name=name)

        rigid_registrar, non_rigid_registrar, error_df = registrar.register(brightfield_processing_cls=AcrobatProcessor)



    # Determine how large of an image to use for micro registration
    img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
    min_max_size = np.min([np.max(d) for d in img_dims])
    img_areas = [np.multiply(*d) for d in img_dims]
    max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])

    if micro_reg_fraction == 1.0:
        # Full size image
        micro_reg_size = None
    if isinstance(micro_reg_fraction, float):
        micro_reg_size = np.floor(min_max_size*micro_reg_fraction).astype(int)
    else:
        micro_reg_size = micro_reg_fraction
        micro_reg_fraction = micro_reg_fraction/min_max_size

    # Determine if image will need to be divided into tiles
    max_microreg_size  = (micro_reg_size, micro_reg_size)
    displacement_gb = registrar.size*warp_tools.calc_memory_size_gb(max_microreg_size, 2, "float32")
    processed_img_gb = registrar.size*warp_tools.calc_memory_size_gb(max_microreg_size, 1, "uint8")
    img_gb = registrar.size*warp_tools.calc_memory_size_gb(max_microreg_size, 3, "uint8")
    estimated_gb = img_gb + displacement_gb + processed_img_gb
    if estimated_gb > registration.TILER_THRESH_GB:
        # Tiles may not have edge artifacts the AcrobatProcessor handles
        img_processor = ColorfulStandardizer
    else:
        # Image will be higher resolution of the one used before, so use the same processor
        img_processor = AcrobatProcessor

    # Perform microregistration within the bounding box of the H&E+ tissue
    micro_reg_mask = create_reg_mask(registrar)

    micro_reg, micro_error = registrar.register_micro(max_non_rigid_registartion_dim_px=micro_reg_size,
                                reference_img_f=he_img_f,
                                align_to_reference=True,
                                mask=micro_reg_mask,
                                brightfield_processing_cls=img_processor
                                )


    stop = time()
    elapsed = stop - start
    elapsed_min = np.round(elapsed/60, 6)

    # Create benchmarking results
    pt_dir = os.path.join(dst_dir, "acrobat", "landmarks") # Warp points in the IHC image. Save in a separate directory
    plot_dir = os.path.join(dst_dir, "acrobat", "plots") # Will also draw the warped points on both images
    for d in [pt_dir, plot_dir]:
        pathlib.Path(d).mkdir(exist_ok=True, parents=True)

    draw_rad = 2
    pt_cmap = (255*viz.jzazbz_cmap()).astype(np.uint8)

    # Read in points
    pt_df = pd.read_csv(landmarks_f)
    sample_df = pt_df.loc[pt_df["anon_id"] == eval(name)]
    sample_df["name"] = [valtils.get_name(x) for x in sample_df["anon_filename_ihc"]]

    # Get some info used to draw the landmarks
    from_slides = [o.name for o in registrar.slide_dict.values()]
    ref_slide = registrar.get_ref_slide()
    ref_wh = ref_slide.slide_dimensions_wh[0]
    from_slides.remove(ref_slide.name)

    ref_reg_img = ref_slide.processed_img
    draw_ref_s = np.min(DRAW_IMG_SIZE/np.array(ref_reg_img.shape[0:2]))
    draw_ref_img = warp_tools.rescale_img(ref_reg_img, draw_ref_s)
    draw_ref_img = skcolor.gray2rgb(draw_ref_img)
    ref_slide_to_draw_sxy = np.array(draw_ref_img.shape[0:2][::-1])/ref_slide.slide_dimensions_wh[0]

    updated_df_list = [None] * len(from_slides)

    # Warp and plot the landmarks.
    for i, sname in enumerate(from_slides):

        # Convert from um to pixel in IHC image
        src_slide = registrar.slide_dict[sname]
        pair_df = sample_df.loc[sample_df["name"] == src_slide.name]
        src_mpp = pair_df[["mpp_ihc_10X"]].values[0][0]
        src_xy_um = pair_df[["ihc_x", "ihc_y"]].values
        src_xy_px = src_xy_um/src_mpp
        warped_xy_px = src_slide.warp_xy_from_to(src_xy_px, ref_slide)

        # Convert from pixel to um in H&E
        dst_mpp = pair_df[["mpp_he_10X"]].values[0][0]
        warped_xy_um = warped_xy_px*dst_mpp
        pair_df[["he_x", "he_y"]] = warped_xy_um
        updated_df_list[i] = pair_df

        # Estimate error using same metrics as acrobat, but using features
        moving_feature_xy_warped = src_slide.warp_xy_from_to(src_slide.xy_matched_to_prev,
                                                             ref_slide,
                                                             src_pt_level=src_slide.processed_img_shape_rc
                                                             )

        moving_feature_xy_warped[:, 0] = np.clip(moving_feature_xy_warped[:, 0], 0, ref_wh[0])
        moving_feature_xy_warped[:, 1] = np.clip(moving_feature_xy_warped[:, 1], 0, ref_wh[1])

        ref_sxy = np.array(ref_slide.slide_dimensions_wh[0])/np.array(ref_slide.processed_img_shape_rc[::-1])
        ref_in_slide_xy = src_slide.xy_in_prev*ref_sxy

        d = warp_tools.calc_d(ref_in_slide_xy*ref_slide.resolution, moving_feature_xy_warped*src_slide.resolution)
        feature_p90 = np.percentile(d, q=90)
        feature_mean_d = np.mean(d)
        pair_df["p90"] = feature_p90
        pair_df["mean"] = feature_mean_d
        updated_df_list[i] = pair_df

        # Draw landmarks on both. Source (IHC) on left, target (H&E) on right
        src_reg_img = src_slide.warp_img(src_slide.processed_img)
        draw_src_s = np.min(DRAW_IMG_SIZE/np.array(src_reg_img.shape[0:2]))
        draw_src_img = warp_tools.rescale_img(src_reg_img, draw_src_s)
        draw_src_img = skcolor.gray2rgb(draw_src_img)

        combo_img = np.hstack([draw_src_img, draw_ref_img])
        c_shift = draw_src_img.shape[1]

        src_slide_to_draw_sxy = (np.array(draw_src_img.shape[0:2][::-1])/np.array(ref_slide.slide_dimensions_wh[0]))
        warped_in_src_xy = src_slide.warp_xy(src_xy_px)

        unwarped_src_sxy = np.array(src_slide.processed_img.shape[0:2][::-1])/src_slide.slide_dimensions_wh[0]
        unwarped_draw_rc = (src_xy_px*unwarped_src_sxy)[:, ::-1]
        unwarped_draw_img = skcolor.gray2rgb(src_slide.processed_img)

        src_draw_rc = (warped_in_src_xy*src_slide_to_draw_sxy)[:, ::-1]
        ref_draw_rc = (warped_xy_px*ref_slide_to_draw_sxy)[:, ::-1]
        for pt_idx, src_pt in enumerate(src_draw_rc):
            ref_pt = ref_draw_rc[pt_idx]
            ref_pt[1] += c_shift

            clr = pt_cmap[np.random.choice(pt_cmap.shape[0], 1)[0]]

            src_circ = draw.disk(src_pt, draw_rad, shape=combo_img.shape)
            target_circ = draw.disk(ref_pt, draw_rad, shape=combo_img.shape)
            pt_line = list(draw.line(*src_pt.astype(int), *ref_pt.astype(int)))
            pt_line[0] = np.clip(pt_line[0], 0, combo_img.shape[0]-1)
            pt_line[1] = np.clip(pt_line[1], 0, combo_img.shape[1]-1)
            pt_line = tuple(pt_line)

            combo_img[pt_line] = clr
            combo_img[src_circ] = clr
            combo_img[target_circ] = clr

            unwarped_circ = draw.disk(unwarped_draw_rc[pt_idx], draw_rad, shape=unwarped_draw_img.shape)
            unwarped_draw_img[unwarped_circ] = clr

        pt_img_f_out = os.path.join(plot_dir, f"{registrar.name}_{src_slide.name}_to_{ref_slide.name}.png")
        warp_tools.save_img(pt_img_f_out, combo_img)

    # Save the results
    updated_df = pd.concat(updated_df_list)
    updated_df = updated_df.drop(["name"], axis=1)
    pt_f = os.path.join(pt_dir, f"{registrar.name}_landmarks.csv")
    updated_df.to_csv(pt_f, index=False)

    # Delete registrar to save space on computer.
    try:
        reg_f = os.path.join(registrar.data_dir, f"{registrar.name}_registrar.pickle")
        os.remove(reg_f)
        if os.path.exists(registrar.displacements_dir):
            shutil.rmtree(registrar.displacements_dir)

    except OSError as e:
        print("Error: %s : %s" % (registrar.data_dir, e.strerror))

    registration.kill_jvm()