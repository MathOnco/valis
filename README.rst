
|docs| |CI| |pypi|

.. .. |Upload Python Package| image:: https://github.com/MathOnco/valis/actions/workflows/python-publish.yml/badge.svg
    :target: https://github.com/MathOnco/valis/actions/workflows/python-publish.yml

.. .. |build-status| image:: https://circleci.com/gh/readthedocs/readthedocs.org.svg?style=svg
..     :alt: build status
..     :target: https://circleci.com/gh/readthedocs/readthedocs.org

.. |docs| image:: https://readthedocs.org/projects/valis/badge/?version=latest
    :target: https://valis.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. |CI| image:: https://github.com/MathOnco/valis/workflows/CI/badge.svg?branch=main
    :target: https://github.com/MathOnco/valis/actions?workflow=CI
    :alt: CI Status

.. .. |conda| image:: https://img.shields.io/conda/vn/conda-forge/valis_wsi
    :alt: Conda (channel only)

.. |pypi| image:: https://badge.fury.io/py/valis-wsi.svg
    :target: https://badge.fury.io/py/valis-wsi

.. .. |coverage| image:: https://codecov.io/gh/readthedocs/readthedocs.org/branch/master/graph/badge.svg
..     :alt: Test coverage
..     :scale: 100%
..     :target: https://codecov.io/gh/readthedocs/readthedocs.org


.. image::  https://github.com/MathOnco/valis/raw/main/docs/_images/banner.gif


VALIS, which stands for Virtual Alignment of pathoLogy Image Series, is a fully automaated pipeline to register whole slide images (WSI) using rigid and/or non-rigid transformtions. A full description of the method is descriped in the paper by `Gatenbee et al. 2021 <https://www.biorxiv.org/content/10.1101/2021.11.09.467917v1>`_. VALIS uses `Bio-Formats <https://www.openmicroscopy.org/bio-formats/>`_, `OpenSlide <https://openslide.org/>`__, `libvips <https://www.libvips.org/>`_, and `scikit-image <https://scikit-image.org/>`_ to read images and slides, and so is able to work with over 300 image formats. Registered images can be saved as `ome.tiff <https://docs.openmicroscopy.org/ome-model/5.6.3/ome-tiff/>`_ slides that can be used in downstream analyses. ome.tiff format is opensource and widely supported, being readable in several different programming languages (Python, Java, Matlab, etc...) and software, such as `QuPath <https://qupath.github.io/>`_, `HALO by Idica Labs <https://indicalab.com/halo/>`_, etc...

The registration pipeline is fully automated and goes as follows:

    .. image::  https://github.com/MathOnco/valis/raw/main/docs/_images/pipeline.png

   #. Images/slides are converted to numpy arrays. As WSI are often too large to fit into memory, these images are usually lower resolution images from different pyramid levels.

   #. Images are processed to single channel images. They are then normalized to make them look as similar as possible.

   #. Image features are detected and then matched between all pairs of image.

   #. If the order of images is unknown, they will be optimally ordered based on their feature similarity

   #. Images will be aligned *towards* (not to) a reference image. If the reference image was not specified, it will automatically be set to the image at the center of the stack.

   #. Rigid registration is performed serially, with each image being rigidly aligned towards the reference image. That is, if the reference image is the 5th in the stack, image 4 will be aligned to 5 (the reference), and then 3 will be aligned to the now registered version of 4, and so on. VALIS uses feature detection to match and align images, but one can optionally perform a final step that maximizes the mutual information betweeen each pair of images.

   #. Non-rigid registration is then performed either by:

        * aliging each image towards the reference image following the same sequence used during rigid registation.
        * using groupwise registration that non-rigidly aligns the images to a common frame of reference. Currently this is only possible if `SimpleElastix <https://simpleelastix.github.io>`__ is installed.

   #. Error is measured by calculating the distance between registered matched features in the full resolution images.

The transformations found by VALIS can then be used to warp the full resolution slides. It is also possible to merge non-RGB registered slides to create a highly multiplexed image. These aligned and/or merged slides can then be saved as ome.tiff images.

In addition to warping images and slides, VALIS can also warp point data, such as cell centoids or ROI coordinates.

Full documentation can be found at `ReadTheDocs <https://valis.readthedocs.io/en/latest/>`_.

.. contents:: Table of Contents
   :local:
   :depth: 1

Installation
============

.. note::
    VALIS requires Python >=3.7

conda (recommened for non-Windows users)
----------------------------------------
VALIS will soon be available in the conda-forge channel of conda. However, unfortunately `libvips <https://www.libvips.org/>`_, a  core dependency, is not yet available for Windows users on conda-forge.

.. Before proceeding, make sure the conda-forge is on the conda channel list:

.. .. code-block:: bash

..    $ conda config --append channels conda-forge

.. Next, create and activate a virtual environment. This example use "valis_conda_env" for the virtual environment name, but it could be anything you'd like.

.. .. code-block:: bash

..    $ conda update conda
..    $ conda create -n valis_conda_env python
..    $ conda activate valis_conda_env

.. Finally, install using conda

.. .. code-block:: bash

..     $ conda install -c conda-forge valis_wsi

pip
---
VALIS can be downloaded from PyPI as the `valis-wsi <https://pypi.org/project/valis-wsi/#description>`_ package using the pip command. However, VALIS requires several system level packages, which will need to be installed first.

Prerequisites
~~~~~~~~~~~~~

VALIS uses Bioforamts to read many slide formats. Bioformats is written in Java, and VALIS uses the Python package jpype to access the Bioformats jar. Therefore, the user will need to have installed a Java Development Kit (JDK) containing the Java Runtime Environment (JRE):

#. Download appropriate JDK from `java downloads <http://www.oracle.com/technetwork/java/javase/downloads/index.html>`_


#.  Edit your system and environment variables to update the Java home

    .. code-block:: bash

        $ export JAVA_HOME=/usr/libexec/java_home


#. Verify the path has been added:

   .. code-block:: bash

       $ echo $JAVA_HOME

   should print something like :code:`usr/libexec/java_home`


#. (optional) If you will be working with files that have extensions: '.vmu', '.mrxs' '.svslide', you will also need to install `OpenSlide <https://openslide.org>`_. Note that this is not the same as openslide-python, which contains Python wrappers for OpenSlide.

   .. important::

       OpenSlide requires `pixman <http://www.pixman.org>`_, which must be version 0.40.0. If pixman is a different version, then the slides may be distorted when reading from any pyramid level other than 0.

#. VALIS uses `pyvips <https://github.com/libvips/pyvips>`_ to warp and save the whole slide images (WSI) as ome.tiffs. Pyvips requires `libvips <https://www.libvips.org/>`_ (not a Python package) to be on your library search path, and so libvips must be installed separately. See the `pyvips installation notes <https://github.com/libvips/pyvips/blob/master/README.rst#non-conda-install>`_ for instructions on how to do this for your operating system. If you already have libvips installed, please make sure it's version is >= 8.11.

Install
~~~~~~~

Once the above prerequisites have been satistifed, valis can be installed using pip, idealy within a virtual environment

.. code-block:: bash

    $ python3 -m venv venv_valis
    $ source ./venv_valis/bin/activate
    $ python3 -m pip install --upgrade pip
    $ python3 pip install valis-wsi

SimpleElastix (optional)
------------------------

The defaults used by VALIS work well, but VALIS also provides optional classes that require `SimpleElastix <https://simpleelastix.github.io>`_. In particular, these classes are:

#. affine_optimizer.AffineOptimizerMattesMI, which uses sitk.ElastixImageFilter to simultaneously maximize Mattes Mutual Information and minimize the spatial distance between matched features.


#. non_rigid_registrars.SimpleElastixWarper, which uses sitk.ElastixImageFilter to find non-rigid transformations between pairs of images.


#. non_rigid_registrars.SimpleElastixGroupwiseWarper, which uses sitk.ElastixImageFilter to find non-rigid transformations using groupwise registration.

To install SimpleElastix, you should probably uninstall the current version of SimpleITK in your environment, and then install SimpleElastix as described in the `SimpleElastix docs <https://simpleelastix.readthedocs.io/GettingStarted.html>`_.

Examples
========

.. important::
    Always be sure to always kill the JVM at the end of your script. Not doing so can prevent the software from closing. This can be accomplished by calling  either :code:`registration.kill_jvm()` or :code:`slide_io.kill_jvm()`

Slide registration
------------------

.. image::  https://github.com/MathOnco/valis/raw/main/docs/_images/challenging_dataset_adincar33.png

.. important::
    One of the most imporant parameters used to initialize a Valis object is :code:`max_processed_image_dim_px`. The default value is 850, but if registration fails or is poor, try adjusting that value. Generally speaking, values between 500-2000 work well. In cases where there is little empty space, around the tissue, smaller values may be better. However, if there is a large amount of empty space/slide (as in the images above), larger values will be needed so that the tissue is at a high enough resolution. Finally, larger values can potentially generate more accurate registrations, but will be slower, require more memory, and won't always produce better results.


.. important::
    If the order of slices is known and needs to be preserved, such as building a 3D image, set :code:`imgs_ordered=True` when intialzing the VALIS object. Otherwise, VALIS will sort the images based on similarity, which may or may not correspond on the sliced order. If using this option, be sure that the names of the files allow them to be sorted properly, e.g. 01.tiff, 02.tiff ... 10.tiff, etc...


In this example, the slides that need to be registered are located in :code:`/path/to/slides`. This process simply involves the creation of a Valis object, which is what conducts the registration. In this example no reference image is specfied, and so all images will be aligned towards the center. In this case, the resulting images will be cropped to the region where all of the images overlap. However, one can specify the reference image when intialzing the :code:`Valis` object, by setting :code:`reference_img_f` to the filename of the image the others should be aligned towards. When the reference image is specifed, the images will be cropped such that only the regions which overlap with the reference image will be saved. While this is the default behavior, one can also specify the cropping method by setting the :code:`crop` parameeter value when initialzing the :code:`Valis` object. The cropping method can also be changed when saving the registered images (see below).

.. code-block:: python

    from valis import registration
    slide_src_dir = "/path/to/slides"
    results_dst_dir = "./slide_registration_example"
    registered_slide_dst_dir = "./slide_registration_example/registered_slides"

    # Create a Valis object and use it to register the slides in slide_src_dir
    registrar = registration.Valis(slide_src_dir, results_dst_dir)
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

After registration is complete, one can view the results to determine if they are acceptable. In this example, the results are located in  :code:`./slide_registration_example`. Inside this folder will be 6 subfolders:


#. **data** contains 2 files:

   * a summary spreadsheet of the alignment results, such as the registration error between each pair of slides, their dimensions, physical units, etc...

   * a pickled version of the registrar. This can be reloaded (unpickled) and used later. For example, one could perfom the registration locally, but then use the pickled object to warp and save the slides on an HPC. Or, one could perform the registration and use the registrar later to warp points found in the (un-registered) slide.


#. **overlaps** contains thumbnails showing the how the images would look if stacked without being registered, how they look after rigid registration, and how they look after non-rigid registration. The rightmost images in the figure above provide examples of these overlap images.


#. **rigid_registration** shows thumbnails of how each image looks after performing rigid registration. These would be similar to the bottom row in the figure above.


#. **non_rigid_registration** shows thumbnaials of how each image looks after non-rigid registration. These would be similar to the bottom row in the figure above.


#. **deformation_fields** contains images showing what the non-rigid deformation would do to a triangular mesh. These can be used to get a sense of how the images were altered by non-rigid warping. In these images, the color indicates the direction of the displacement, while brightness indicates it's magnitude. These would be similar to those in the middle row in the figure above.


#. **processed** shows thumnails of the processed images. These are thumbnails of the images that were actually used to perform the registration. The pre-processing and normalization methods should try to make these images look as similar as possible.


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


One can also choose to save individual slides. This is accomplished by accessing the Slide object associated with a particular file, :code:`slide_f` and then "telling" it to save the slide aas :code:`out_f.ome.tiff`.

.. code-block:: python

    slide_obj = registrar.get_slide(slide_f)
    slide_obj.warp_and_save_slide(out_f.ome.tiff)

Finally, if the non-rigid registration is deemed to have distored the image too much, one can apply only the rigid transformation by setting :code:`non_rigid=False` in :code:`slide_obj.warp_and_save_slide` or :code:`registrar.warp_and_save_slides`.

Slide registration and merging
------------------------------
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
--------------
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
    registrar = pickle.load(open(registrar_f, 'rb'))

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
    registrar = pickle.load(open(registrar_f, 'rb'))

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
.. .. image::  ../examples/expected_results/roi/ihc_roi.png


Converting slides to ome.tiff
-----------------------------
In addition to registering slide, VALIS can convert slides to ome.tiff, maintaining the original metadata. If the original is image is not RGB, the option :code:`perceputally_uniform_channel_colors=True` can be used to give each channel a perceptually uniform color, derived from the `JzAzBz <https://www.osapublishing.org/DirectPDFAccess/5166548C-BD18-487D-8601630F3A343883_368272/oe-25-13-15131.pdf?da=1&id=368272&seq=0&mobile=no>`_ colorspace. An advantage of using perceptually uniform colors is that markers should appear brighter only if there is higher expression, not because the color (such as yellow) is perceived to be brighter.

.. code-block:: python

    from valis import slide_io
    slide_src_f = "path/to/slide
    converted_slide_f = "converted.ome.tiff"
    slide_io.convert_to_ome_tiff(slide_src_f,
                                converted_slide_f,
                                level=0,
                                perceputally_uniform_channel_colors=True)
    slide_io.kill_jvm()

.. image::  https://github.com/MathOnco/valis/raw/main/docs/_images/pu_color_mplex.png


Using non-defaults
------------------
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

Change Log
==========

Version 1.0.0rc6 (in progress)
--------------------------------
#. More accurate color mixing with fewer artifacts. Affects overlap images and pseudo-colored multi-channel images.
#. Initializing  'is_flattended_pyramid' with False. Pull request #6
#. Reformatting flattened pyramids to have same datatype as that in metadata.
#. Save all images using pyvips. Should be faster.
#. Use Bio-Formats to read non-RGB ome-tiff. Addresses an issue where converting non-RGB ome-tiff read using pyvips was very slow.

Version 1.0.0rc5 (April 5, 2022)
--------------------------------
#. Can provide a reference image that the others will be aligned towards. To do this, when initializinig the Valis object, set the :code:`reference_img_f` argument to be the file name of the reference image. If not set by the user, the reference image will be set as the one at the center of the ordered image stack
#. Both non-rigid and rigid now align *towards* a reference image, meaning that reference image will have neither rigid nor non-rigid transformations applied to it.
#. Two cropping methods. First option is to crop seach registered slides to contain only the areas where all registered images overlap. The second option is to crop the registered slide to contain only the area that intersects with the reference image. It is also possible to not crop an image/slide.
#. Images are now cropped during the warp, not after, and so is now faster and requires less memory. For example, on a 2018 MacBook Pro with a 2.6 GHz Intel Core i7 processor, it takes 2-3 minutes to warp and save a 41399 x 43479 RGB image.
#. Warping of images and slides done using the same function, built around pyvips. Faster, more consistent, and should prevent excessive memory usage.
#. Fixed bug that caused a crash when warping large ome.tiff images.
#. Read slides and images using pyvips whenever possible.
#. Background color now automatically set to be same as the brightest (IHC) or darkest (IF) pixel in the image. Because of this, the "bg_color" argument in the slide warping functions was removed.
#. Reduced accumulation of unwanted non-rigid deformations
#. Displacement fields drawn on top of non-rigid registered image to help determine where the deformations occured.
#. If a slide has multiple series, and a series is not specficed, the slide reader will read the series containing the largest image.

License
-------

`MIT`_ © 2021-2022 Chandler Gatenbee

.. _MIT: LICENSE.txt
