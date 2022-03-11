We have provided two test datasets, located in the `examples/example_datasets` folders. The images in each dataset are much smaller versions of the original WSI, as the originals are too large to transfer easily. However, VALIS can be used with much larger images.

## IHC registration
The `register_ihc.py` example registers the slides in `examples/example_datasets/ihc`, with results being saved in `examples/expected_results/registration/ihc`. Inside this folder are several subfolders that show the results of the alignment:

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

After the registration has completed, the script also saves the registered slides to `examples/expected_results/registered_slides/ihc`. On a 2018 MacBook pro with a 2.6 GHz Intel Core i7 processor and 32Gb RAM, registration took 1.16 minutes and saving all of the slides took 12.5 seconds.

## CyCIF registration
The `register_and_merge_cycif.py` example registers the slides in `examples/example_datasets/cycif`, with results being saved in `examples/expected_results/registration/cycif`. After registration is complete, the slides are registered and merged to create a 7 channel immunofluorescence image, which is saved in `examples/expected_results/registered_slides/cycif`. On a 2018 MacBook pro with a 2.6 GHz Intel Core i7 processor and 32Gb RAM, registration took 1.33 minutes and saving all of the slides took 4.97 seconds.
