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

4. *non_rigid_registration* shows thumbnaials of how each
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

import time
import os
import numpy as np

import shutil
import sys
from valis import registration, valtils


def get_parent_dir():
    cwd = os.getcwd()
    dir_split = cwd.split(os.sep)
    split_idx = [i for i in range(len(dir_split)) if dir_split[i] == "valis_project"][0]
    parent_dir = os.sep.join(dir_split[:split_idx+1])
    return parent_dir


def cnames_from_filename(src_f):
    """Get channel names from file name
    Note that the DAPI channel is not part of the filename
    but is always the first channel.

    """
    f = valtils.get_name(src_f)
    return ["DAPI"] + f.split(" ")


parent_dir = get_parent_dir()
datasets_src_dir = os.path.join(parent_dir, "valis/examples/example_datasets/")
results_dst_dir = os.path.join(parent_dir, f"valis/tests/tmp{sys.version_info.major}{sys.version_info.minor}")

def register_ihc(max_error=45):
    ihc_src_dir = os.path.join(datasets_src_dir, "ihc")
    ihc_dst_dir = os.path.join(results_dst_dir, "ihc")
    try:
        registrar = registration.Valis(ihc_src_dir, results_dst_dir)
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()
        micro_non_rigid_registrar, micro_error_df = registrar.register_micro()
        avg_error = np.max(micro_error_df["mean_non_rigid_D"])

        if avg_error > max_error:
            # shutil.rmtree(ihc_dst_dir, ignore_errors=True)
            assert False, f"error was {avg_error} but should be below {max_error}"

        registered_slide_dst_dir = os.path.join(registrar.dst_dir, "registered_slides", registrar.name)
        registrar.warp_and_save_slides(registered_slide_dst_dir)
        # registration.kill_jvm()

        # shutil.rmtree(ihc_dst_dir, ignore_errors=True)

        assert True

    except Exception as e:
        # shutil.rmtree(ihc_dst_dir, ignore_errors=True)
        assert False, e

def test_register_cycif(max_error=3):
    cycif_src_dir = os.path.join(datasets_src_dir, "cycif")
    cycif_dst_dir = os.path.join(results_dst_dir, "cycif")
    try:
        registrar = registration.Valis(cycif_src_dir, results_dst_dir)
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()
        micro_non_rigid_registrar, micro_error_df = registrar.register_micro()
        avg_error = np.max(micro_error_df["mean_non_rigid_D"])

        if avg_error > max_error:
            # shutil.rmtree(cycif_dst_dir, ignore_errors=True)
            assert False, f"error was {avg_error} but should be below {max_error}"

        channel_name_dict = {f: cnames_from_filename(f) for
                             f in registrar.original_img_list}

        dst_f = os.path.join(registrar.dst_dir, "registered_slides", f"{registrar.name}.ome.tiff")
        merged_img, channel_names, ome_xml = registrar.warp_and_merge_slides(dst_f,
                                            channel_name_dict=channel_name_dict,
                                            drop_duplicates=True)

        # registration.kill_jvm()

        # shutil.rmtree(cycif_dst_dir, ignore_errors=True)

        assert True

    except Exception as e:
        # shutil.rmtree(cycif_dst_dir, ignore_errors=True)
        assert False, e