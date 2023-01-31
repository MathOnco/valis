Examples
********

.. important::
    Always be sure to always kill the JVM at the end of your script. Not doing so can prevent the software from closing. This can be accomplished by calling  either :code:`registration.kill_jvm()` or :code:`slide_io.kill_jvm()`

Slide registration
==================

.. image::  https://github.com/MathOnco/valis/raw/main/docs/_images/challenging_dataset_adincar33.png

.. important::
    One of the most imporant parameters used to initialize a Valis object is :code:`max_processed_image_dim_px`. The default value is 850, but if registration fails or is poor, try adjusting that value. Generally speaking, values between 500-2000 work well. In cases where there is little empty space, around the tissue, smaller values may be better. However, if there is a large amount of empty space/slide (as in the images above), larger values may be needed so that the tissue is at a high enough resolution. Finally, larger values can potentially generate more accurate registrations, but will be slower, require more memory, and won't always produce better results.


.. important::
    If the order of slices is known and needs to be preserved, such as building a 3D image, set :code:`imgs_ordered=True` when intialzing the VALIS object. Otherwise, VALIS will sort the images based on similarity, which may or may not correspond on the sliced order. If using this option, ensure that the names of the files allow them to be sorted properly, e.g. 01.tiff, 02.tiff ... 10.tiff, etc...


In this example, the slides that need to be registered are located in :code:`/path/to/slides`. This process involves creating a Valis object, which is what conducts the registration. In this example no reference image is specfied, and so all images will be aligned towards the center of the image stack. In this case, the resulting images will be cropped to the region where all of the images overlap. However, one can specify the reference image when intialzing the :code:`Valis` object, by setting :code:`reference_img_f` to the filename of the image the others should be aligned towards. When the reference image is specifed, the images will be cropped such that only the regions which overlap with the reference image will be saved. While this is the default behavior, one can also specify the cropping method by setting the :code:`crop` parameter value when initialzing the :code:`Valis` object. The cropping method can also be changed when saving the registered images (see below).

.. code-block:: python

    from valis import registration
    slide_src_dir = "/path/to/slides"
    results_dst_dir = "./slide_registration_example"
    registered_slide_dst_dir = "./slide_registration_example/registered_slides"

    # Create a Valis object and use it to register the slides in slide_src_dir
    registrar = registration.Valis(slide_src_dir, results_dst_dir)
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

The next example shows how align each image to a reference image, followed up by micro-registration. The reference image the others should be aligned towards is set with the :code:`reference_img_f` argument when initialzing the :code:`Valis` object. This initial registration is followed up by micro-registration in order to better align features that were not present in the smaller images used for the first registration (The size of the images used for micro-registration can is set with the :code:`max_non_rigid_registartion_dim_px` argument in :code:`Valis.register_micro`). Setting :code:`align_to_reference` to `True` will align each image directly *to* the reference image, as opposed to *towards* it.


.. code-block:: python

    from valis import registration
    slide_src_dir = "/path/to/slides"
    results_dst_dir = "./slide_registration_example"
    registered_slide_dst_dir = "./slide_registration_example/registered_slides"
    reference_slide = "HE.tiff"

    # Create a Valis object and use it to register the slides in slide_src_dir, aligning towards the reference slide.
    registrar = registration.Valis(slide_src_dir, results_dst_dir, reference_img_f=reference_slide)
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    # Perform micro-registration on higher resolution images, aligning directly to the reference image
    registrar.register_micro(max_non_rigid_registartion_dim_px=2000, align_to_reference=True)


After registration is complete, one can view the results to determine if they are acceptable. In this example, the results are located in  :code:`./slide_registration_example`. Inside this folder will be 6 subfolders:


#. **data** contains 2 files:

   * a summary spreadsheet of the alignment results, such as the registration error between each pair of slides, their dimensions, physical units, etc...

   * a pickled version of the registrar. This can be reloaded (unpickled) and used later. For example, one could perfom the registration locally, but then use the pickled object to warp and save the slides on an HPC. Or, one could perform the registration and use the registrar later to warp points found in the (un-registered) slide.


#. **overlaps** contains thumbnails showing the how the images would look if stacked without being registered, how they look after rigid registration, and how they look after non-rigid registration. The rightmost images in the figure above provide examples of these overlap images.


#. **rigid_registration** shows thumbnails of how each image looks after performing rigid registration. These would be similar to the bottom row in the figure above.


#. **non_rigid_registration** shows thumbnaials of how each image looks after non-rigid registration. These would be similar to the bottom row in the figure above.


#. **deformation_fields** contains images showing what the non-rigid deformation would do to a triangular mesh. These can be used to get a sense of how the images were altered by non-rigid warping. In these images, the color indicates the direction of the displacement, while brightness indicates it's magnitude. These would be similar to those in the middle row in the figure above.


#. **processed** shows thumnails of the processed images. These are thumbnails of the images that were actually used to perform the registration. The pre-processing and normalization methods should try to make these images look as similar as possible.


#. **masks** show the images with outlines of their rigid registration mask drawn around them. If non-rigid registration is being performed, there will also be an image of the reference image with the non-rigid registration mask drawn around it.


If the results look good, then one can warp and save all of the slides as ome.tiffs. When saving the images, there are three cropping options:

#. :code:`crop="overlap"` will crop the images to the region where all of the images overlap.
#. :code:`crop="reference"` will crop the images to the region where they overlap with the reference image.
#. :code:`crop="all"` will not perform any cropping. While this keep the all of the image, the dimensions of the registered image can be substantially larger than one that was cropped, as it will need to be large enough accomodate all of the other images.

While the cropping setting can also be set when initializing the :code:`Valis` object, any of the above cropping methods can be used when saving the images.

.. code-block:: python

    # Save all registered slides as ome.tiff
    registrar.warp_and_save_slides(registered_slide_dst_dir, crop="overlap")

    # Kill the JVM
    registration.kill_jvm()

The ome.tiff images can subsequently be used for downstream analysis, such as `QuPath <https://qupath.github.io/>`_

.. image::  https://github.com/MathOnco/valis/raw/main/docs/_images/ome_tiff_zoom.png


One can also choose to save individual slides. This is accomplished by accessing the Slide object associated with a particular file, :code:`slide_f` and then "telling" it to save the slide as :code:`out_f.ome.tiff`.

.. code-block:: python

    slide_obj = registrar.get_slide(slide_f)
    slide_obj.warp_and_save_slide("out_f.ome.tiff")

Finally, if the non-rigid registration is deemed to have distored the image too much, one can apply only the rigid transformation by setting :code:`non_rigid=False` in :code:`slide_obj.warp_and_save_slide` or :code:`registrar.warp_and_save_slides`.

Create multiplex image from immunofluorescence images
======================================================
Following registration, VALIS can merge the slides to create a single composite image. However, this should only be done for non-RGB images, such as multi/single-channel immunofluorescence images. An example would be slides of multiple CyCIF rounds. The user also has the option to provide channel names, but if not provided the channel names will become the "channel (filename)" given the channel name in the metadata. For example, if the file name is round1.ndpis then the DAPI channel name will be "DAPI (round1)"). In this example, the channel names are taken from the filename, which have the form "Tris CD20 FOXP3 CD3.ndpis", "Tris CD4 CD68 CD3 1in25 ON.ndpis", etc... The channel names need to be in a dictionary, where key=filename, value = list of channel names.

.. important::
    By default, if a channel occurs in more than 1 image, only the 1st instance will be merged. For example, if DAPI is in all images, then only the DAPI channel of the 1st image will be in the resulting slide. This can be disabled by setting :code:`drop_duplicates=False` in :code:`warp_and_merge_slides`

First, create a VALIS object and use it to register slides located in :code:`slide_src_dir`

.. code-block:: python

    from valis import registration
    slide_src_dir = "/path/to/slides"
    results_dst_dir = "./slide_merging_example"  # Registration results saved here
    merged_slide_dst_f = "./slide_merging_example/merged_slides.ome.tiff"  # Where to save merged slide

    registrar = registration.Valis(slide_src_dir, results_dst_dir)
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

Check the results in :code:`results_dst_dir`, and if the look good merge and save the slide. Once complete, be sure to kill the JVM.

.. code-block:: python

    # Create function to extract channel names from the image.
    def cnames_from_filename(src_f):
        """Get channel names from file name
        Note that the DAPI channel is not part of the filename
        but is always the first channel.
        """

        f = valtils.get_name(src_f)
        return ["DAPI"] + f.split(" ")[1:4]

    channel_name_dict = {f:cnames_from_filename(f) for f in registrar.original_img_list}
    merged_img, channel_names, ome_xml = \
        registrar.warp_and_merge_slides(merged_slide_dst_f,
                                        channel_name_dict=channel_name_dict,
                                        drop_duplicates=True)

    registration.kill_jvm() # Kill the JVM

.. image::  https://github.com/MathOnco/valis/raw/main/docs/_images/merge_ome_tiff.png



Warping points
===============
Once the registration parameters have been found, VALIS can be used to warp point data, such as cell coordinates, mask polygon vertices, etc... In this example, slides will be registered, and the registration parameters will then be used warp cell positions located in a separate .csv. This accomplished by accessing the :code:`Slide` object associated with each registered slide. This is done by passing the slide's filename (with or without the extension) to :code:`registrar.get_slide`. This :code:`Slide` object can the be used to warp the individual slide and/or points associated with the un-registered slide. This can be useful in cases where one has already performed an analysis on the un-registered slides, as one can just warp the point data, as opposed to warping each slide and re-conducting the analysis.

.. important::
    It is essential that the image from which the coordinates are derived has the same aspect ratio as the image used for registration. That is, the images used for registration must be scaled up/down versions of the image from which the coordinates are taken. For example, registration may be performed on lower resolution images (an upper image pyramid level), and applied to cell coordinates found by performing cell segmenation on the full resolution (pyramid level 0) image. The default is to assume that the points came from the highest resolution image, but this can be changed by setting :code:`pt_level` to either the pyramid level of the image the points originated, or its dimensions (width, height, in pixels). Also, the coordinates need to be in pixel units, not physical units. Finally, be sure that the coordinates are X,Y (column, row), with the origin being the top left corner of the image.

In this first example, cell segmentation and phenotyping has already been performed on the unregistered images. We can now use the :code:`Valis` object that performed the registration to warp the cell positions to their location in the registered images.

.. code-block:: python

    import os
    import numpy as np
    import pandas as pd
    import pathlib
    import pickle
    from valis import registration

    slide_src_dir = "path/to/slides"
    point_data_dir = "path/to/cell_positions"
    results_dst_dir = "./point_warping_example"

    # Load a Valis object that has already registered the images.
    registrar_f = "path/to/results/data/registrar.pickle"
    registrar = registration.load_registrar(registrar_f)

    # Get .csv files containing cell coordinates
    point_data_list = list(pathlib.Path(point_data_dir).rglob("*.csv"))

    # Go through each file and warp the cell positions
    for f in point_data_list:
        # Get Slide object associated with the slide from which the point data originated
        # Point data and image have similar file names
        fname = os.path.split(f)[1]
        corresponding_img = fname.split(".tif")[0]
        slide_obj = registrar.get_slide(corresponding_img)

        # Read data and calculate cell centroids (x, y)
        points_df = pd.read_csv(f)
        x = np.mean(points_df[["XMin", "XMax"]], axis=1).values
        y = np.mean(points_df[["YMin", "YMax"]], axis=1).values
        xy = np.dstack([x, y])[0]

        # Use Slide to warp the coordinates
        warped_xy = slide_obj.warp_xy(xy)

        # Update dataframe with registered cell centroids
        points_df[["registered_x", "registered_y"]] = warped_xy

        # Save updated dataframe
        pt_f_out = os.path.split(f)[1].replace(".csv", "_registered.csv")
        full_pt_f_out = os.path.join(results_dst_dir, pt_f_out)
        points_df.to_csv(full_pt_f_out, index=False)

    registration.kill_jvm() # Kill the JVM

Here is a comparison of before and after applying registration to cell positions found in the original un-aligned images:

.. image::  https://github.com/MathOnco/valis/raw/main/docs/_images/point_warping.png

In this second example, a region of interest (ROI) was marked in one of the unregistered images, in this case "ihc_2.ome.tiff" . Using the :code:`Slide` object associated with "ihc_2.ome.tiff", we can warp those ROI coordinates to their position in the registered images, and then use those to slice the registered ROI from each slide. Because VALIS uses pyvips to read and warp the slides, this process does not require the whole image to be loaded into memory and warped. As such, this is fast and does not require much memory. It's also worth noting that because the points are being warped to the registred coordinate system, the slide that is the source of the ROI coordinates does not have to be the same slide that was treated as the reference image during registration.

.. code-block:: python

    import os
    import pickle
    import numpy as np
    import matplotlib.pyplot as plt
    import pathlib
    from valis import registration, warp_tools

    # Load a registrar that has already registered the images.
    registrar_f = "./expected_results/registration/ihc/data/ihc_registrar.pickle"
    registrar = registration.load_registrar(registrar_f)
    # Set the pyramid level from which the ROI coordinates originated. Usually 0 when working with slides.
    COORD_LEVEL = 0

    # ROI coordinates, in microns. These came from the unregistered slide, "ihc_2.ome.tiff"
    bbox_xywh_um = [14314, 13601, 3000, 3000]
    bbox_xy_um = warp_tools.bbox2xy(bbox_xywh_um)

    # Get slide from which the ROI coordinates originated
    pt_source_img_f = "ihc_2.ome.tiff"
    pt_source_slide = registrar.get_slide(pt_source_img_f)

    # Convert coordinates to pixel units
    um_per_px = pt_source_slide.reader.scale_physical_size(COORD_LEVEL)[0:2]
    bbox_xy_px = bbox_xy_um/np.array(um_per_px)

    # Warp coordinates to position in registered slides
    bbox_xy_in_registered_img = pt_source_slide.warp_xy(bbox_xy_px,
                                                        slide_level=COORD_LEVEL,
                                                        pt_level=COORD_LEVEL)

    bbox_xywh_in_registered_img = warp_tools.xy2bbox(bbox_xy_in_registered_img)
    bbox_xywh_in_registered_img = np.round(bbox_xywh_in_registered_img).astype(int)

    # Create directory where images will be saved
    dst_dir = "./expected_results/roi"
    pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)

    # Warp each slide and slice the ROI from it using each pyips.Image's "extract_area" method.
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    ax = axes.ravel()
    for i, slide in enumerate(registrar.slide_dict.values()):
        warped_slide = slide.warp_slide(level=COORD_LEVEL)
        roi_vips = warped_slide.extract_area(*bbox_xywh_in_registered_img)
        roi_img = warp_tools.vips2numpy(roi_vips)
        ax[i].imshow(roi_img)
        ax[i].set_title(slide.name)
        ax[i].set_axis_off()

    fig.delaxes(ax[5]) # Only 5 images, so remove 6th subplot
    out_f = os.path.join(dst_dir, f"{registrar.name}_roi.png")
    plt.tight_layout()
    plt.savefig(out_f)
    plt.close()

    # Opening the slide initialized the JVM, so it needs to be killed
    registration.kill_jvm()

The extracted and registered ROI are shown below:

.. image::  https://github.com/MathOnco/valis/raw/main/examples/expected_results/roi/ihc_roi.png


Transferring annotations
========================
In this example, VALIS uses the registration parameters to transfer annotations found from one image to another. In this case, the annotation were performed in QuPath and exported as a geojson file. Given the geojson file, VALIS can then warp each shape in the file from the reference slide to its position on the un-registered target slide. The registered annotations can then be saved and loaded into QuPath along with the target image. Below, :code:`annotation_img_f` refers to the filename associated with the image on which the original annoation was performed, :code:`target_img_f` is the filename of the image associated with the image the annotations will be transferred to, :code:`annotation_geojson_f` is the name of the file with the annoation shapes, and :code:`warped_geojson_annotation_f` is the name of geojson file the registered annotations will be saved to.


.. code-block:: python

    import json
    from valis import registration

    # Perform registration
    registrar = registration.Valis(slide_src_dir, results_dst_dir)
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    # Transfer annotation from image associated with annotation_img_f and image associated with target_img_f
    annotation_source_slide = registrar.get_slide(annotation_img_f)
    target_slide = registrar.get_slide(target_img_f)


    warped_geojson_from_to = annotation_source_slide.warp_geojson_from_to(annotation_geojson_f, target_slide)
    warped_geojson = annotation_source_slide.warp_geojson(annotation_geojson_f)

    # Save annotation as warped_geojson_annotation_f, which can be dragged and dropped into QuPath
    with open(warped_geojson_annotation_f, 'w') as f:
        json.dump(warped_geojson, f)



.. image:: https://github.com/MathOnco/valis/raw/main/docs/_images/annotation_transfer.png


Converting slides to ome.tiff
=============================
In addition to registering slide, VALIS can convert slides to ome.tiff, maintaining the original metadata. If the original is image is not RGB, the option :code:`colormap` can be used to give each channel a specific color using a dictionary, where the key is the channel name, and the value is the RGB tuple (0-255). If :code:`colormap` is not provided, the original channel colors will be used.


.. code-block:: python

    from valis import slide_io
    slide_src_f = "path/to/slide
    converted_slide_f = "converted.ome.tiff"
    slide_io.convert_to_ome_tiff(slide_src_f,
                                 converted_slide_f,
                                 level=0)
    slide_io.kill_jvm()

.. image::  https://github.com/MathOnco/valis/raw/main/docs/_images/pu_color_mplex.png


Reading slides
===============
VALIS also provides functions to read images/slides using libvips, Bio-Formats, or Openslide. These reader objects also contain some of the slide's metatadata. The :code:`slide2image` method will return a numpy array of the slide, while :code:`slide2vips` will return a :code:`pyvips.Image`, which is ideal when working with very large images. The user can specify the pyramid level, series, and bounding box, but the default is level 0, series 0, and the whole image. See :code:`slide_io.SlideReader` and :code:`slide_io.MetaData` for more details.


.. code-block:: python

    from valis import slide_io
    slide_src_f = "path/to/slide.svs
    series = 0

    # Get reader for slide format
    reader_cls = slide_io.get_slide_reader(slide_src_f, series=series) #Get appropriate slide reader class
    reader = reader_cls(slide_src_f, series=series) # Instantiate reader

    #Get size of images in each pyramid level (width, height)
    pyramid_level_sizes_wh = reader.metadata.slide_dimensions

    # Get physical units per pixel
    pixel_physical_size_xyu = reader.metadata.pixel_physical_size_xyu

    # Get channel names (None if image is RGB)
    channel_names = reader.metadata.channel_names

    # Get original xml metadata
    original_xml = reader.metadata.original_xml

    # Get smaller pyramid level 3 as a numpy array
    img = reader.slide2image(level=3)

    # Get full resolution image as a pyvips.Image
    full_rez_vips = reader.slide2vips(level=0)

    # Slice region of interest from level 0 and return as numpy array
    roi_img = reader.slide2image(level=0, xywh=(100, 100, 500, 500))

    slide_io.kill_jvm()


Warping slides with custom transforms
======================================
VALIS provides the functions to apply transformations to slides and then save the registered slide, meaning the user can provide their own transformation parameters. In this example, `src_f` is the path to the file associated with the slide, `M` is the inverse rigid registration matrix, and `bk_dxdy` is a list of the backwards non-rigid displacement fields (i.e. [dx, dy]), each found by aligning the fixed/target image to the moving/source image.

.. important::
    The transformations will need to be inverted if they were found the other way around, i.e. aligning the moving/source image to the fixed/target image. Transformation matrices can be inverted using :code:`np.linalg.inv`, while displacement fields can be inverted using :code:`warp_tools.get_inverse_field`.


One may also need to provide the shape of the image (row, col) used to find the rigid transformation (if applicable), which is the `transformation_src_shape_rc` argument. In this case, it is the shape of the processed image that was used during feature detection. Similarly, `transformation_dst_shape_rc` is the shape of the registered image, in this case the shape of the processed image after being warped. Finally, `aligned_slide_shape_rc` is the shape of the warped slide. Please see :code:`slide_io.warp_and_save_slide` for more information and options, like defining background color, crop area, etc..

.. code-block:: python

    from valis import slide_io

    # Read and warp the slide #
    slide_src_f = "path/to/slide
    dst_f = "path/to/write/slide.ome.tiff"
    series = 0
    pyramid_level=0

    slide_io.warp_and_save_slide(src_f=slide_src_f,
                                 dst_f=dst_f,
                                 transformation_src_shape_rc=processed_img_shape_rc,
                                 transformation_dst_shape_rc=small_registered_img_shape_rc,
                                 aligned_slide_shape_rc=aligned_slide_shape_rc,
                                 level=pyramid_level,
                                 series=series,
                                 M=M,
                                 dxdy=dxdy)


    slide_io.kill_jvm()

Using non-defaults
===================
The defaults used by VALIS work well, but one may wish to try some other values/class, and/or create their own affine optimizer, feature detector, non-rigid registrar, etc... This examples shows how to conduct registration using non-default values

.. note::
    This example assumes that `SimpleElastix <https://simpleelastix.readthedocs.io/GettingStarted.html>`__ has been installed.

.. code-block:: python

    from valis import registration, feature_detectors, non_rigid_registrars, affine_optimizer
    slide_src_dir = "path/to/slides"
    results_dst_dir = "./slide_registration_example_non_defaults"
    registered_slide_dst_dir = "./slide_registration_example/registered_slides"


    # Select feature detector, affine optimizer, and non-rigid registration method.
    # Will use KAZE for feature detection and description
    # SimpleElastix will be used for non-rigid warping and affine optimization
    feature_detector_cls = feature_detectors.KazeFD
    non_rigid_registrar_cls = non_rigid_registrars.SimpleElastixWarper
    affine_optimizer_cls = affine_optimizer.AffineOptimizerMattesMI

    # Create a Valis object and use it to register the slides in slide_src_dir
    registrar = registration.Valis(slide_src_dir, results_dst_dir,
                                   feature_detector_cls=feature_detector_cls,
                                   affine_optimizer_cls=affine_optimizer_cls,
                                   non_rigid_registrar_cls=non_rigid_registrar_cls)


    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    registration.kill_jvm() # Kill the JVM
