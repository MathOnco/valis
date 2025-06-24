""" Registration of whole slide images (WSI)

This example shows how to register, warp, and save a collection
of whole slide images (WSI) using the default parameters.

The results directory contains several folders:

1. *data* contains 2 files:
    * a summary spreadsheet of the alignment results, such
    as the registration error between each pair of slides, their
    dimensions, physical units, etc...

    * a pickled version of the registrar. This can be reloaded
    (unpickled) and used later. For example, one could perform
    the registration locally, but then use the pickled object
    to warp and save the slides on an HPC. Or, one could perform
    the registration and use the registrar later to warp
    points in the slide.

2. *overlaps* contains thumbnails showing the how the images
    would look if stacked without being registered, how they
    look after rigid registration, and how they would look
    after non-rigid registration.

3. *rigid_registration* shows thumbnails of how each image
    looks after performing rigid registration.

4. *non_rigid_registration* shows thumbnails of how each
    image looks after non-rigid registration.

5. *deformation_fields* contains images showing what the
    non-rigid deformation would do to a triangular mesh.
    These can be used to get a better sense of how the
    images were altered by non-rigid warping

6. *processed* shows thumnails of the processed images.
    This are thumbnails of the images that are actually
    used to perform the registration. The pre-processing
    and normalization methods should try to make these
    images look as similar as possible.


After registraation is complete, one should view the
results to determine if they aare acceptable. If they
are, then one can warp and save all of the slides.
"""



import torch
import kornia

import time
import os
import numpy as np
from itertools import chain
import pathlib
import ome_types
from skimage import transform
import shutil
import sys
import os
import pandas as pd

from valis import registration, valtils, slide_io, micro_rigid_registrar
from valis.feature_matcher import *

def check_for_no_transforms_in_ref(ref_slide, reference_img_fname):
    assert ref_slide.name == valtils.get_name(reference_img_fname), "Reference image is not the same as specified image"
    og_crop = ref_slide.crop
    ref_slide.crop = registration.CROP_REF
    # M may have translations for cropping to overlap, but they are ignored when cropping to reference (verified below)
    try:
        tformer = transform.AffineTransform(ref_slide.M)
        assert np.all(tformer.scale == [1.0, 1.0]), "Reference image has unexpected scaling"
        assert tformer.rotation == 0, "Reference image has unexpected rotation"
        assert tformer.shear == 0, "Reference image has unexpected shearing"

        # Check that translations only crop padded image to the reference image's origin
        out_shape_rc = ref_slide.slide_dimensions_wh[0][::-1]
        sxy = (out_shape_rc/ref_slide.processed_img_shape_rc)[::-1]
        scaled_txy = sxy*ref_slide.M[:2, 2]

        crop_bbox, _ = ref_slide.get_crop_xywh(ref_slide.crop, out_shape_rc=out_shape_rc)
        cropping_to_origin = np.all(np.abs(crop_bbox[0:2] + scaled_txy) < 1)
        assert cropping_to_origin, "translations don't move reference to the origin"

        # Check for non-rigid transforms
        assert np.dstack(ref_slide.bk_dxdy).min() == 0 and np.dstack(ref_slide.bk_dxdy).max() == 0, "Found unexpected non-rigid transforms"

        # Check that points are warped correctly too
        xy = np.array([[0, 0]])
        warped_origin = np.round(ref_slide.warp_xy(xy))
        assert np.all(warped_origin == 0), "reference image not warping points correctly"

        # Compare raw values
        unwarped_ref_img = ref_slide.slide2vips(level=0)
        warped_ref_img = ref_slide.warp_slide(level=0)
        eq_img = (unwarped_ref_img == warped_ref_img)
        min_eq = eq_img.min()
        assert min_eq == 255, "warped and original images do not have the same values"

    except AssertionError:
        print("test failed")

    ref_slide.crop = og_crop



def get_dirs():
    dst_parent_folder = f"{sys.platform}_{sys.version_info.major}{sys.version_info.minor}"
    try:
        current_file_path = os.path.abspath(__file__)
        current_directory = os.path.dirname(current_file_path)
        parent_dir = os.path.split(current_directory)[0]
        results_dst_dir = os.path.join(current_directory, dst_parent_folder)
        print("worked")

    except:
        cwd = os.getcwd()
        dir_split = cwd.split(os.sep)
        split_idx = [i for i in range(len(dir_split)) if dir_split[i] == "valis_project"][0]
        parent_dir = os.path.join(os.sep.join(dir_split[:split_idx+1]), "valis")
        results_dst_dir = os.path.join(parent_dir, f"tests/{sys.version_info.major}{sys.version_info.minor}")

    return parent_dir, results_dst_dir


def cnames_from_filename(src_f):
    """Get channel names from file name
    Note that the DAPI channel is not part of the filename
    but is always the first channel.

    """
    f = valtils.get_name(src_f)
    return ["DAPI"] + f.split(" ")


parent_dir, results_dst_dir = get_dirs()
results_dst_dir = os.path.join(results_dst_dir, "examples")
datasets_src_dir = os.path.join(parent_dir, "examples/example_datasets/")


def register_hi_rez(src_dir):
    high_rez_dst_dir = os.path.join(results_dst_dir, "high_rez")
    micro_reg_fraction = 0.25 # Fraction full resolution used for non-rigid registration

    # Perform high resolution rigid registration using the MicroRigidRegistrar
    start = time.time()
    registrar = registration.Valis(src_dir, high_rez_dst_dir, micro_rigid_registrar_cls=micro_rigid_registrar.MicroRigidRegistrar)
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    # Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration on an image that is 25% full resolution.
    img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
    min_max_size = np.min([np.max(d) for d in img_dims])
    img_areas = [np.multiply(*d) for d in img_dims]
    max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])
    micro_reg_size = np.floor(min_max_size*micro_reg_fraction).astype(int)

    # Perform high resolution non-rigid registration
    micro_reg, micro_error = registrar.register_micro(max_non_rigid_registration_dim_px=micro_reg_size)

    ref_slide = registrar.get_ref_slide()
    ref_slide_src_f = ref_slide.src_f
    check_for_no_transforms_in_ref(ref_slide, ref_slide_src_f)

    stop = time.time()
    elapsed = stop - start
    print(f"regisration time is {elapsed/60} minutes")

    # We can also plot the high resolution matches using `Valis.draw_matches`:
    matches_dst_dir = os.path.join(registrar.dst_dir, "hi_rez_matches")
    registrar.draw_matches(matches_dst_dir)


def test_register_ihc(max_error=70):
    """Tests registration and lossy jpeg compression

    """
    ihc_src_dir = os.path.join(datasets_src_dir, "ihc")
    # ihc_src_dir = "/Users/gatenbcd/Dropbox/Documents/Andriy/Bina_alignments/slides/NSG_from_Marusyk/NSG 48 hours_374"
    try:

        registrar = registration.Valis(ihc_src_dir, results_dst_dir)
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()

        registrar.draw_matches(registrar.dst_dir)
        avg_error = np.max(error_df["mean_non_rigid_D"])

        if avg_error > max_error:
            # shutil.rmtree(ihc_dst_dir, ignore_errors=True)
            assert False, f"error was {avg_error} but should be below {max_error}"

        registered_slide_dst_dir = os.path.join(registrar.dst_dir, "registered_slides", registrar.name)
        registrar.warp_and_save_slides(dst_dir=registered_slide_dst_dir, Q=90, compression="jpeg")

        ref_slide = registrar.get_ref_slide()
        ref_slide_src_f = ref_slide.src_f

        check_for_no_transforms_in_ref(ref_slide, ref_slide_src_f)
        assert True

    except Exception as e:
        assert False, e


def test_register_cycif(max_error=3):
    """
    Goals:
        * Aligment and merging of staining rounds
        * Make sure error is below threshold
        * Checks channel names of merged image are in the correct order (https://github.com/MathOnco/valis/issues/56#issuecomment-1821050877)
        * Check jpeg2000 compression
        * Check that reference slide is unwarped


    """

    drop_duplicates = True
    cycif_src_dir = os.path.join(datasets_src_dir, "cycif")
    try:
        img_list = list(pathlib.Path(cycif_src_dir).rglob("*.ome.tiff"))
        img_list = np.roll(img_list, 1)
        ref_img_f = str(img_list[0])

        registrar = registration.Valis(cycif_src_dir, results_dst_dir, img_list=img_list, imgs_ordered=True, reference_img_f=ref_img_f)
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()
        avg_error = np.max(error_df["mean_non_rigid_D"])

        if avg_error > max_error:
            assert False, f"error was {avg_error} but should be below {max_error}"

        channel_name_dict = {str(f): cnames_from_filename(f) for f in img_list}

        dst_f = os.path.join(registrar.dst_dir, "registered_slides", f"{registrar.name}.ome.tiff")
        merged_img, channel_names, ome_xml = registrar.warp_and_merge_slides(dst_f,
                                                channel_name_dict=channel_name_dict,
                                                drop_duplicates=drop_duplicates,
                                                Q=90, compression="jp2k")

        # Check that specified reference image is not warped
        ref_slide = registrar.get_ref_slide()
        ref_slide_src_f = ref_img_f
        check_for_no_transforms_in_ref(ref_slide, ref_slide_src_f)

        # Check merged image has channel names in expected order

        assert [slide_obj.stack_idx for slide_obj in registrar.slide_dict.values()] == list(range(registrar.size)), "Slides got sorted when `imgs_ordered=True`"

        sorted_img_list = registrar.get_sorted_img_f_list() # Get images in the same order used for registration (may be different than order in directory)
        expected_channel_order = list(chain.from_iterable([channel_name_dict[str(f)] for f in sorted_img_list]))
        if drop_duplicates:
            cnames_df = pd.DataFrame(expected_channel_order, columns=['cname'])
            expected_channel_order = list(cnames_df.drop_duplicates(keep="first")['cname'])

        saved_ome_xml = ome_types.from_xml(ome_xml)
        saved_channel_names = [x.name for x in saved_ome_xml.images[0].pixels.channels]

        assert np.all(expected_channel_order == saved_channel_names), (f"Channels not saved in correct order.\n"
                                                                       f"Expected: {expected_channel_order}.\n"
                                                                       f"Got:      {saved_channel_names}"
                                                                       f"Img list: {[os.path.split(x)[1] for x in img_list]}")

        assert True

    except Exception as e:
        assert False, e


def test_register_hi_rez_ihc():
    ihc_src_dir = os.path.join(datasets_src_dir, "ihc")
    register_hi_rez(src_dir=ihc_src_dir)


def test_register_hi_rez_cycif():
    cycif_src_dir = os.path.join(datasets_src_dir, "cycif")
    register_hi_rez(src_dir=cycif_src_dir)


if __name__ == "__main__":
    test_register_ihc()
    test_register_cycif()
    test_register_hi_rez_ihc()
    test_register_hi_rez_cycif()
