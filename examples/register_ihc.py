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
from valis import registration


slide_src_dir = "./example_datasets/ihc"
results_dst_dir = "./expected_results/registration"

# Create a Valis object and use it to register the slides in slide_src_dir
start = time.time()
registrar = registration.Valis(slide_src_dir, results_dst_dir)
rigid_registrar, non_rigid_registrar, error_df = registrar.register()
stop = time.time()
elapsed = stop - start
print(f"regisration time is {elapsed/60} minutes")


# Check results in registered_slide_dst_dir. If they look good, export the registered slides
registered_slide_dst_dir = os.path.join("./expected_results/registered_slides", registrar.name)
start = time.time()
registrar.warp_and_save_slides(registered_slide_dst_dir)
stop = time.time()
elapsed = stop - start
print(f"saving {registrar.size} slides took {elapsed/60} minutes")


# Shutdown the JVM
registration.kill_jvm()
