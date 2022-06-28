"""Merging of whole slide images (WSI) CyCIF images

This example shows how to register and merge a set
of slides. In this case, there are 3 CyCIF images.

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
results to determine if they aare acceptable.

Since the slides are being merged, one may want to provide
channel names. This can be accomplished by passing a
channel names dictionary to the Valis.warp_and_merge_slides
method.

"""

import time
import os
from valis import registration, valtils

slide_src_dir = "./example_datasets/cycif"
results_dst_dir = "./expected_results/registration"

# Create a Valis object and use it to register the slides in slide_src_dir
start = time.time()
registrar = registration.Valis(slide_src_dir, results_dst_dir)
rigid_registrar, non_rigid_registrar, error_df = registrar.register()
stop = time.time()
elapsed = stop - start

print(f"regisration time is {elapsed/60} minutes")


# Merge registered channels
def cnames_from_filename(src_f):
    """Get channel names from file name
    Note that the DAPI channel is not part of the filename
    but is always the first channel.

    """
    f = valtils.get_name(src_f)
    return ["DAPI"] + f.split(" ")


channel_name_dict = {f: cnames_from_filename(f) for
                     f in registrar.original_img_list}

dst_f = os.path.join("./expected_results/registered_slides", registrar.name, registrar.name + ".ome.tiff")
start = time.time()
merged_img, channel_names, ome_xml = registrar.warp_and_merge_slides(dst_f,
                                      channel_name_dict=channel_name_dict,
                                      perceputally_uniform_channel_colors=True,
                                      drop_duplicates=True)
stop = time.time()
elapsed = stop - start

print(f"Time to warp, merge, and save slides is {elapsed/60} minutes")

registration.kill_jvm()
