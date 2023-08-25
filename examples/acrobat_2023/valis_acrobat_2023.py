"""
Code used for the ACROBAT 2023 Grand Challenge
"""

import pathlib
import os
import shutil
from time import time
import argparse
import numpy as np
import pandas as pd
import pyvips

from valis import registration, warp_tools, preprocessing, slide_io
from valis.micro_rigid_registrar import MicroRigidRegistrar

ANON_COL = "anon_id"
PT_COL = "point_id"
SRC_COL = "wsi_source"
TARGET_COL = "wsi_target"
LANDMARK_COL = "landmarks_csv"
DST_DIR_COL = "output_dir_name"

SRC_MMP = "mpp_source"
SRC_X_COL = "x_source"
SRC_Y_COL = "y_source"

TARGET_X_COL = "x_target"
TARGET_Y_COL = "y_target"
TARGET_MMP = "mpp_target"

TARGET_IMG_NAME = "registered_source_image.ome.tiff"
TARGET_LANDMARK_NAME = "registered_landmarks.csv"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Register images for the 2023 ACROBAT Grand Challenge')
    parser.add_argument('-PATH_TABLE_CSV', type=str, help='csv containing image pairs and landmarks')
    parser.add_argument('-PATH_IMAGES_DIR', type=str, help='csv containing image pairs and landmarks')
    parser.add_argument('-PATH_ANNOT_DIR', type=str, help='csv containing image pairs and landmarks')
    parser.add_argument('-PATH_OUTPUT_DIR', type=str, help='csv containing image pairs and landmarks')
    parser.add_argument('-row_idx', type=int, help='name/row of sample')
    parser.add_argument('-micro_reg_fraction', type=str, help='Size of images used for 2nd registation')
    parser.add_argument('-use_masks', type=str, help='Whether or not to use masks for registration')

    args = vars(parser.parse_args())

    micro_reg_fraction = args["micro_reg_fraction"]
    row_idx = args["row_idx"]
    img_src_dir = args["PATH_IMAGES_DIR"]
    landmark_src_dir = args["PATH_ANNOT_DIR"]
    parent_dst_dir = args["PATH_OUTPUT_DIR"]
    input_table = args["PATH_TABLE_CSV"]
    use_masks = eval(args["use_masks"])

    micro_reg_fraction = eval(micro_reg_fraction)
    input_df = pd.read_csv(input_table)
    sample_row = input_df.iloc[row_idx]

    src_img_f = sample_row[SRC_COL]
    target_img_f = sample_row[TARGET_COL]
    landmarks_f = os.path.join(landmark_src_dir, sample_row[LANDMARK_COL])
    sample_id = str(sample_row[DST_DIR_COL])
    dst_dir = os.path.join(parent_dst_dir, sample_id)

    print(sample_id)
    # Do initial registration, setting the reference image to be the H&E image
    img_list = [os.path.join(img_src_dir, d) for d in [target_img_f, src_img_f]]

    start = time()
    registrar = registration.Valis("./",  os.path.split(dst_dir)[0],
                                   crop="reference",
                                   reference_img_f=target_img_f,
                                   align_to_reference=True,
                                   img_list=img_list,
                                   create_masks=use_masks,
                                   max_processed_image_dim_px=500,
                                   max_non_rigid_registration_dim_px=2000,
                                   micro_rigid_registrar_cls=MicroRigidRegistrar,
                                   name=sample_id)

    rigid_registrar, non_rigid_registrar, error_df = registrar.register(brightfield_processing_cls=preprocessing.StainFlattener,
                                                                        brightfield_processing_kwargs={"adaptive_eq":True})

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

    micro_reg, micro_error = registrar.register_micro(max_non_rigid_registration_dim_px=micro_reg_size,
                                                      reference_img_f=target_img_f,
                                                      align_to_reference=True,
                                                      brightfield_processing_cls=preprocessing.StainFlattener)
    stop = time()
    elapsed = stop - start
    elapsed_min = np.round(elapsed/60, 6)

    # Create benchmarking results
    pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)

    non_rigid_error = np.nanmean(micro_error["non_rigid_D"].values)
    rigid_error = np.nanmean(micro_error["rigid_D"].values)
    apply_non_rigid = non_rigid_error < rigid_error
    if not apply_non_rigid:
        print(f"Non-rigid registration didn't improve alignments for {registrar.name}")

    # Read in points
    pt_df = pd.read_csv(landmarks_f)
    src_mpp = pt_df[SRC_MMP].values[0]
    src_xy_um = pt_df[[SRC_X_COL, SRC_Y_COL]].values
    src_xy_px = src_xy_um/src_mpp

    # Warp points
    src_slide = registrar.get_slide(src_img_f)
    ref_slide = registrar.get_slide(target_img_f)
    warped_xy_px = src_slide.warp_xy_from_to(src_xy_px, ref_slide, non_rigid=apply_non_rigid)

    # Clip any points that fell outside of image
    ref_slide_shape_wh = ref_slide.slide_dimensions_wh[0]
    warped_xy_px[:, 0] = np.clip(warped_xy_px[:, 0], 0, ref_slide_shape_wh[0])
    warped_xy_px[:, 1] = np.clip(warped_xy_px[:, 1], 0, ref_slide_shape_wh[1])

    dst_mpp = pt_df[TARGET_MMP].values[0]
    warped_xy_um = warped_xy_px*dst_mpp

    registered_df = pd.DataFrame({ANON_COL: pt_df[ANON_COL].values,
                                  PT_COL: pt_df[PT_COL].values,
                                  TARGET_X_COL: warped_xy_um[:, 0],
                                  TARGET_Y_COL: warped_xy_um[:, 1]})

    pt_f_out = os.path.join(dst_dir, TARGET_LANDMARK_NAME)
    registered_df.to_csv(pt_f_out, index=False)

    # Save registered source image
    s = 1/8
    source_slide = registrar.get_slide(src_img_f)
    warped_slide = source_slide.warp_slide(level=0, non_rigid=apply_non_rigid)
    warped_thumb = warp_tools.rescale_img(warped_slide, s)
    thumb_xyzct = slide_io.get_shape_xyzct((warped_thumb.width, warped_thumb.height), warped_thumb.bands)
    bf_dtype = slide_io.vips2bf_dtype(warped_thumb.format)
    thumb_px_phys_size = (dst_mpp/s, dst_mpp/s, 'µm')

    ome_xml_obj = slide_io.update_xml_for_new_img(source_slide.reader.metadata.original_xml,
                                                  new_xyzct=thumb_xyzct,
                                                  bf_dtype=bf_dtype,
                                                  is_rgb=source_slide.reader.metadata.is_rgb,
                                                  series=source_slide.reader.series,
                                                  pixel_physical_size_xyu=thumb_px_phys_size,
                                                  channel_names=source_slide.reader.metadata.channel_names
                                                  )

    ome_xml_obj.creator = f"pyvips version {pyvips.__version__}"
    ome_xml_str = ome_xml_obj.to_xml()

    dst_f = os.path.join(dst_dir, TARGET_IMG_NAME)
    slide_io.save_ome_tiff(warped_thumb, dst_f, ome_xml_str)

    # Delete output not needed by acrobat
    for f in os.listdir(dst_dir):
        if not f in [TARGET_LANDMARK_NAME, TARGET_IMG_NAME]:
            full_f = os.path.join(dst_dir, f)
            if os.path.isdir(full_f):
                # remove directories
                shutil.rmtree(full_f)
            else:
                # remove files
                os.remove(full_f)

    registration.kill_jvm()