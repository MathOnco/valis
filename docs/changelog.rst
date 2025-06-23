Change Log
**********

Version 1.2.0 (June 15, 2025)
-------------------------------------
.. image::  https://github.com/MathOnco/valis/raw/main/docs/_images/new_fd_2025_error_v_time.png
.. image::  https://github.com/MathOnco/valis/raw/main/docs/_images/new_fd_2025_error_correlation.png

Major improvements to rigid registration by using more modern feature detectors and matchers. More specifically, Disk and DeDoDe feature detectors outperform BRISK and VGG (previous defaults). Different combinations of feature detectors, image sizes, and colorspaces (RGB or grayscale) were benchmarked using the ANHIR dataset. These results indicate that using the DISK feature detector on grayscale images both increases accuracy (i.e. lower TRE) and increase the correlation between VALIS' estimated error and the true error. However, using RGB does have somewhat higher accuracy and slightly faster runtimes. If you'd like to use RGB images, set the :code:`rgb=True` when initializing Disk or DeDoDe feature detectors, pass that to the LightGlueMatcher, and then have that be the feature matcher used by Valis. That is:

.. code-block:: python

    from valis import registration, feature_detectors, feature_matcher
    slide_src_dir = "/path/to/slides"
    results_dst_dir = "./slide_registration_example"
    registered_slide_dst_dir = "./slide_registration_example/registered_slides"

    fd = feature_detectors.DiskFD(rgb=True, num_features=2000)
    matcher = feature_matcher.LightGlueMatcher(fd)
    registrar = registration.Valis(slide_src_dir, results_dst_dir, matcher=matcher)
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()


This improved rigid registration should also lead to fewer unwanted non-rigid deformations, and thus better results over all. Below are more details on these and other changes, as well as describing several bug fixes:

#. Set default feature detector and matcher to be DISK and LightGlueMatcher, used on grayscale images with a maximum dimension of 512pixels.
#. Added DISK and DeDoDe feature detectors (see :code:`feature_detectors.DiskFD` and :code:`feature_detectors.DeDoDeFD`), as implemented in `Kornia <https://kornia.github.io/>`_. These feature detectors can also work with RGB images instead of the processed grayscale images. This can be accomplished by setting :code:`rgb=True` when initializing the feature detector. Do note these feature detectors are not rotation invariant, but the rotation angle can be estimated using features matched using a rotation invariant feature detector, such as BRISK. This is automatically done when using :code:`feature_matcher.LightGlueMatcher`(see below).
#. Added LightGlueMatcher feature matcher (see :code:`feature_matcher.LightGlueMatcher`), which is intended to be used with the DiskFD and DeDoDeFD feature detectors.
#. Feature matchers now instantiated with feature detectors.
#. Instantiated feature detectors and matchers can be passed into :code:`Valis.register`. This makes it possible to set the number of features to detect, whether RGB features should be used, etc... For example, :code:`Valis(matcher=feature_matcher.LightGlueMatcher(feature_detectors.DiskFD(rgb=True, num_features=2000)))` will use the LightGlueMatcher to match 2000 RGB Disk features for each image pair.
#. Using USAC_MAGSAC to remove matched feature outliers, replacing RANSAC.
#. Default image size for non-rigid registraiton has been doubled from 1024 to 2048.
#. Added RAFT non-rigid registration method (see :code:`non_rigid_registrars.RAFTWarper`), as implemented in `torchvision <https://pytorch.org/vision/0.12/auto_examples/plot_optical_flow.html>`_. RAFT can also be used with RGB images by setting :code:`rgb=True` when initializing :code:`non_rigid_registrars.RAFTWarper`. Can use RAFT by passing :code:`non_rigid_registrars.RAFTWarper()` to the :code:`non_rigid_registrar_cls` when initialzing the :code:`Valis` object, e.g. :code:`Valis(non_rigid_registrar_cls=non_rigid_registrars.RAFTWarper())`.
#. Can now use different feature detectors for image sorting and image matching. This is useful when the preferred feautre detector is not roatation invariant, as the features matched during sorting can be used to estimate an initial rotation.
#. Fix to avoid trying to access a negative level of an image pyramid (GitHub issue `141 <https://github.com/MathOnco/valis/issues/141>`_)
#. Now possible to re-register a subset of images in :code:`src_dir`. Previously, an error would occur if an attempt at registration was made, followed subsequent attempts using only a subset of images in :code:`src_dir`.
#. Now using the :code:`autocrop` option when reading images with OpenSlide (via libvips). Only reads in the specimen area, as opposed to the entire slide, facilitating higher resolution registration.
#. Avoid bug when searching for a file that has special characters in the name (GitHub issue `170 <https://github.com/MathOnco/valis/issues/170>`_)
#. Test examples in Docker container by mounting a Docker volume that contains example data and scripts. Makes Docker testing more robust, and should avoid issues like those reported in GitHub issues `139 <https://github.com/MathOnco/valis/issues/139>`_ and `176 <https://github.com/MathOnco/valis/issues/176>`_
#. Now requires pillow ≥ 10.3 (GitHub issue `180 <https://github.com/MathOnco/valis/issues/180>`_)
#. Reads in formats readable from BioFormats to avoid unnecessarily launching the JVM when calling :code:`slide_io.get_slide_reader`.
#. Switched from Poetry to UV for Python package management


Version 1.1.0 (June 24, 2024)
-------------------------------------
#. Rigid registration now performed after detecting the tissue and slicing it from the image. This allows higher resolution images to be used for rigid registration, improving accuracy. Implementing this enhancement involved making several changes to VALIS' defaults (see below). Please note that this behavior can be disabled by setting :code:`crop_for_rigid_reg=False` when initializing the :code:`Valis` object.
#. A new masking method was developed to better detect the tissue and pickup less noise (:code:`preprocessing.create_tissue_mask_with_jc_dist`)
#. Added several new pre-processing methods, including :code:`preprocessing.OD`, :code:`preprocessing.JCDist`, :code:`preprocessing.ColorDeconvolver`, and :code:`preprocessing.ColorDeconvolver`.
#. :code:`preprocessing.OD` has replaced :code:`preprocessing.ColorfulStandardizer` as the default RGB pre-processing method. Testing indicates that this method tends to (but doesn't always) perform better across a range of challenging images, as measured by median number of feature matches and mean TRE. To get results similar to previous versions of VALIS, one can set :code:`brightfield_processing_cls=preprocessing.ColorfulStandardizer` when calling :code:`Valis.register`.
#. The default image size for rigid registration has been increased from 850 to 1024, (i.e. :code:`Valis(max_processed_image_dim_px=1024)`).
#. Image denoising for rigid registration is no longer the default behavior. This can be turned back on by setting :code:`Valis(denoise_rigid=False)`.
#. Reduced memory usage to help enable aligning a large number of images (Github issue `105 <https://github.com/MathOnco/valis/issues/105>`_). VALIS has now be tested aligning 556 serially sliced images.
#. Passing radians to :code:`np.cos` and :code:`np.sin` when determining how much to pad images, as noted in `Github issue 103 <https://github.com/MathOnco/valis/issues/103>`_. These images get cropped, so this should not affect registration accuracy.
#. Can now read ome.tiff that do not have a SUBIFD, reported in Github issue `101 <https://github.com/MathOnco/valis/issues/101>`_
#. Improved saving of non-pyarmid images, related to Github issue `101 <https://github.com/MathOnco/valis/issues/101>`_
#. Better compatibility with non-uint8 RGB images.
#. Updated code to ensure merged images have channels in the same order as when sorted or specified.
#. Dockerfile now compatible with Python 3.12

Version 1.0.4 (February 2, 2024)
-------------------------------------
#. Added checks and unit tests to verify reference image is not being warped. Can confirm no transformations were being applied to the reference image, but values may have differed slightly due to interpolation effects, as the reference image was being padded and cropped. To avoid these interpolation effects, the original reference image is returned when "warping" the reference slide with :code:`crop="reference"`, which occurs regardless of the :code:`align_to_reference` setting. This avoids unnecessary computation and potential interpolation errors. There are still checks to make sure that the warped image would have been the same as the unwarped image.
#. Merge channels based on the position of each slide in the stack. This will be the same as the original order when :code:`imgs_ordered=True`.
#. Can provide colormaps for each slide when calling :code:`Valis.warp_and_save_slides`
#. Added :code:`pyramid` argument to :code:`Valis.warp_and_save_slides`
#. Ignore setting series=0 message when there is only 1 series
#. Updated openCV version in project.toml, as suggested in Github issue `76 <https://github.com/MathOnco/valis/issues/76#issuecomment-1916501989>`_
#. Added :code:`slide_io.check_xml_img_match` to determine if there are mismatches between the xml metadata and the image that was read. If there are mismatches, the metadata will be updated based on the image (instead of the xml) and warning messages will be printed to let the user know about the mismatch.
#. If a single channel image does not have channel names in the metadata, the channel name will be set to the image's name.
#. Added :code:`denoise_rigid` as an argument to initialize the :code:`Valis` object. Determines whether or not to denoise the processed images prior to rigid registration. Had been fixed as True in previous versions, but this makes it optional (default remains True).
#. Fixed issue where merged images were being interpreted as RGB and saved with extra channels (reported in Github issue `76 <https://github.com/MathOnco/valis/issues/76#issuecomment-1916501989>`_)

Version 1.0.3 (January 25, 2024)
-------------------------------------
#. Can specify which slide readers to use for each image by passing a dictionary to the :code:`reader_dict` argument in :code:`Valis.register`. The keys, value pairs are image filename and instantiated :code:`SlideReader` to use to read that file. Valis will try to find an appropriate reader for any omitted files. Can be especially useful in cases where one needs different series for different images, as the :code:`series` argument is set when the :code:`SlideReader` is created.
#. Each :code:`Slide` is assigned a :code:`SlideReader`, ensuring that the same series will always be read.
#. Added traceback messages to critical try/except blocks to help with debugging.
#. Micro-registration can now use multiple image processors, and so should be able to perform multi-modal registration
#. Now possible to save the images as non-pyarmid WSI by setting :code:`pyramid=False` when calling the various slide saving methods (requested in `github issue 56 <https://github.com/MathOnco/valis/issues/56>`_).
#. Tested the :code:`slide_io.CziJpgxrReader` with more jpegxr compressed czi images, including  3 RGB (2 mosaic, 1 not mosaic),  1 multichannel non-RGB (mosaic), 1 single channel (mosaic). Related to `github issue 76 <https://github.com/MathOnco/valis/issues/76>`_.
#. Added checks to make sure all channel names are in the colormap, addressing `github issues 78 <https://github.com/MathOnco/valis/issues/78>`_ and `86 <https://github.com/MathOnco/valis/issues/86>`_ .
#. Setting :code:`colormap=None` to the various save functions will not add any color channels, and so the slide viewer's default colormaps will be used.
#. Updated :code:`slide_io.get_slide_reader` to give preference to reading images with libvips/openslide. Should be faster since image will not need to be constructed from tiles.
#. JVM will only be initialized if bioformats is needed to read the image.
#. Updated :code:`slide_io.VipsSlideReader` to use the ome-types pacakge to extract metadata, instead of Bio-formats. Should avoid having to launch JVM unless Bio-formats is really needed.
#. Added checks to ensure that channels in merged image are in the correct order when :code:`imgs_ordered=True`, addressing the comment `github issue 56 <https://github.com/MathOnco/valis/issues/56#issuecomment-1821050877>`_ .
#. Added tests for images with minimal ome-xml (i.e. no channel names, units, etc...)
#. Removed usage of :code:`imghdr`, which is being deprecated
#. Replaced joblib with pqdm. May resolve issue posted on `image.sc <https://forum.image.sc/t/valis-image-registration-unable-to-generate-expected-results/89466>`_
#. Removed interpolation and numba packages as dependencies
#. Updated ome-types' parser to "lxml"
#. Merged `github pull request 95 <https://github.com/MathOnco/valis/pull/95>`_.


Version 1.0.2 (October 11, 2023)
-------------------------------------
#. Fix issue with pip installation, where the pyproject.toml tried to get aicspylibczi from Github, not PyPi

Version 1.0.1 (October 6, 2023)
-------------------------------------
#. Bug fixes to functions related to saving slides as ome.tiff
#. Address numba deprecation warnings

Version 1.0.0 (October 4, 2023)
-------------------------------------
#. Added option for high resolution rigid registration using the :code:`micro_rigid_registrar.MicroRigidRegistrar` class. To use this option, pass an uninstantiated :code:`micro_rigid_registrar.MicroRigidRegistrar` to :code:`micro_rigid_registrar_cls` when initializing the :code:`Valis` object. This class refines the rigid registration by detecting and matching features in higher resolution images warped using the initial rigid transforms. This should result in more accurate rigid registration, error estimates, and hopefully fewer unwanted non-rigid deformations.
#. Added support for SuperPoint and SuperGlue feature detection and matching
#. Masks not applied to images for rigid registration. Instead, the masks are used to filter the matches (i.e. keep only matches inside masks).
#. Added support to extract different Z and T planes using the :code:`slide_io.BioFormatsSlideReader`.
#. Non-rigid masks found by combining the intensities of the rigidly aligned images, as opposed to combining the rigid masks. Testing indicates these masks fit more tightly around the tissue, which will translate to having higher resolution images being used for non-rigid registration.
#. Added the :code:`preprocessing.StainFlattener` class, which can be used with brightfield images.
#. Added option for lossy compression by setting the :code:`Q` parameter using functions that save slides. Confirmed that RGB images saved using lossy JPEG and JPEG2000 compression open as expected in QuPath. Do note that float images saved using these compression methods will get cast to uint8 images. Addresses request made in `github issue 60 <https://github.com/MathOnco/valis/issues/60>`_.
#. Added :code:`Valis.draw_matches` method to visualize feature matches between each pair of images.
#. Fixed issue converting big-endian WSI to :code:`pyvips.Image` (reported on `image.sc <https://forum.image.sc/t/problems-registering-fluorescence-ome-tiffs-using-valis/82685>_`)
#. Added citation information
#. Updated docker container to include pytorch

Version 1.0.0rc15 (May 10, 2023)
-------------------------------------
#. Added import for :code:`aicspylibczi.CziFile` in :code:`slide_io` (found in github issue 44). Also added :code:`aicspylibczi` to the poetry lock file.
#. Added :code:`src_f` argument in :code:`Slide.warp_and_save_slide`. Previously would end up using the :code:`Slide.src_f`, and preventing one from being able to warp and save other images using the same transformations (github issue 49).
#. Various bug fixes to allow the initial non-rigid registration to work with larger images (which may be :code:`pyvips.Image` objects).
#. Fixed bug where errors that occurred while reading images would prevent Python from shutting down.
#. Updated documentation for :code:`valis.preprocessing`
#. Added more tests
#. Fixed many typos in the documentation.

Version 1.0.0rc14 (April 24, 2023)
-------------------------------------
#. Added :code:`max_ratio` as an argument for :code:`feature_matcher.match_desc_and_kp` (github issue 36).
#. Added :code:`CziJpgxrReader` to read CZI images that have JPGXR compression but can't be opened with Bioformats. It's very limited and experimental (only tested with single scence RGB), but may be an issue specific to Apple silcon?
#. Supports scenario where images might be assigned the same name (e.g. same file names, but different directories).
#. Support tiling for initial non-rigid registration, making it possible to perform non-rigid on much larger images
#. Skips empty images (github issue 44).
#. Can now specify an :code:`ImageProcesser` for each image by passing a dicitonary to the :code:`processor_dict` argrument of :code:`Valis.register`. Keys should be the filename of the image, and values a list, where the first element is the :code:`ImageProcesser` to use, and the second element is a dictionary of keyword argruments passed to :code:`ImageProcesser.process_image`. This should make it easier to register different image modalities.
#. Added an H&E color deconvolution :code:`ImageProcesser` using the method described in M. Macenko et al., ISBI 2009. Generously provided by Github user aelskens (Arthur Elskenson) (PR 42).
#. Small improvements in :code:`valtils` functions, provided by Github user ajinkya-kulkarni (Ajinkya Kulkarni) (PR 46).
#. Docker Images bundled with bioformats jar file, so does not require internet connection or Maven. Also now checks for bioformats jar in valis folder
#. Fixed bug that threw error when trying to warp an empty Shapely polygon
#. Fixed bug in micro-registration, related to trying to combine numpy and pyvips images (github issues 40 and 47)
#. Fixed typo in "max_non_rigid_registration_dim_px", which was "max_non_rigid_registartion_dim_px" (github issue 39)
#. Fixed error that caused excessive memory consumption when trying to mask numpy array with pyvips image in :code:`preprocessing.norm_img_stats`


Version 1.0.0rc13 (January 31, 2023)
-------------------------------------
#. Now available as a Docker image
#. Added methods to transfer geojson annotations, such as those generated by QuPath, from one slide to another (:code:`Slide.warp_geojson_from_to` and :code:`Slide.warp_geojson`). Also provide examples in documentation. Addresses `github issue 13 <https://github.com/MathOnco/valis/issues/13>`_
#. Fixed bug reported in `github issue 33 <https://github.com/MathOnco/valis/issues/33>`_
#. Default is to not compose non-rigid transformations, reducing accumulation of unwanted distortions, especially in 3D.
#. The :code:`scale_factor` parameter for :code:`feature_detectors.VggFD` is now set to 5.0, as per the OpenCV documentation
#. Installlation now uses `poetry <https://python-poetry.org/>`_ via the pyproject.toml file. Includes a poetry.lock file, but it can be deleted before installation if so desired.
#. Removed bioformats_jar as a dependency
#. Added a datasets page
#. Moved examples to separate page


Version 1.0.0rc12 (November 7, 2022)
------------------------------------
#. Fixed bug where would get out of bounds errors when cropping with user provided transformations (github issue 14 https://github.com/MathOnco/valis/issues/14)
#. Fixed bug where feature matches not drawn in correct location in :code:`src_img` in :code:`viz.draw_matches`.
#. Can now check if refelcting/mirroring/flipping images improves alignment by setting :code:`check_for_reflections=True` when initializing the :code:`Valis` object. Addresses githib issue 22 (https://github.com/MathOnco/valis/issues/22)
#. Channel colors now transfered to registered image (github issue 23 https://github.com/MathOnco/valis/issues/23). Also option to provide a colormap when saving the slides. This replaces the :code:`perceputally_uniform_channel_colors` argument


Version 1.0.0rc11 (August 26, 2022)
-----------------------------------
#. Fixed bug when providing rigid transformations (Issue 14, https://github.com/MathOnco/valis/issues/14).
#. Can now warp one image onto another, making it possible to transfer annotations using labeled images (Issue 13 https://github.com/MathOnco/valis/issues/13). This can be done using a Slide object's :code:`warp_img_from_to` method. See example in examples/warp_annotation_image.py
#. :code:`ImageProcesser` objects now have a  :code:`create_mask` function that is used to build the masks for rigid registration. These are then used to create the mask used for non-rigid registration, where they are combined such that the final mask is where they overlap and/or touch.
#. Non-rigid registration performed on higher resolution version of the image. The area inside the non-rigid mask is sliced out such that it encompasses the area inside the mask but has a maximum dimension of  :code:`Valis.max_non_rigid_registartion_dim_px`. This can improve accuracy when the tissue is only a small part of the image. If masks aren't created, this region will be where all of the slides overalp.
#. Version used to submit results to the ACROBAT Grand Challenge. Code used to perform registration can be found in examples/acrobat_grand_challenge.py. This example also shows how to use and create a custom :code:`ImageProcesser` and perform micro-registration with a mask.


Version 1.0.0rc10 (August 11, 2022)
-----------------------------------
#. Fixed compatibility with updated interpolation package (Issue 12).

Version 1.0.0rc9 (August 4, 2022)
---------------------------------
#. Reduced memory usage for micro-registration and warping. No longer copying memory before warping, and large displacement fields saved as .tiff images instead of .vips images.
#. Reduced unwanted accumulation of displacements
#. :code:`viz.draw_matches` now returns an image instead of a matplotlib pyplot
#. Pull request 9-11 bug fixes (many thanks to crobbins327 and zindy): Not converting uint16 to uint8 when reading using Bio-Formats or pyvips; fixed rare error when filtering neighbor matches; :code:`viz.get_grid` consistent on Linux and Windows; typos.


Version 1.0.0rc8 (July 1, 2022)
-------------------------------
#. Now compatible with single channel images. These images are treated as immunofluorescent images, and so custom pre-processing classes and arguments should be passed to :code:`if_processing_cls` and :code:`if_processing_kwargs` of the :code:`Valis.register` method. The current method will perform adaptive histogram equalization and scales the image to 0-255 (see :code:`preprocessing.ChannelGetter`). Also, since it isn't possible to determine if the single channel image is a greyscale RGB (light background) or single channel immunofluorescence (or similar with dark background), the background color will not be estimated, meaning that in the registered image the area outside of the warped image will be black (as opposed to the estimated background color). Tissue masks will still be created, but if it seems they are not covering enough area then try setting :code:`create_masks` to `False` when initializing the :code:`Valis` object.


Version 1.0.0rc7 (June 27, 2022)
--------------------------------
#. Can set size of image to be used for non-rigid registration, which may help improve aligment of micro-architectural structures. However this will increase the amount of time it takes to perform non-rigid registration, and will increase amount of memory used during registration, and the size of the pickled :code: `Valis` object. To change this value, set the :code:`max_non_rigid_registartion_dim_px` parameter when initializing the :code:`Valis` object.
#. Can now do a second non-rigid registartion on higher resolution images, including the full resolution one. This can be done with the :code:`Valis.register_micro`. If the images are large, they will be sliced into tiles, and then each tile registered with one another. The deformation fields will be saved separately as .vips images within the data folder.
#. Added :code:`registration.load_registrar` function to open a :code:`Valis` object. This should be used instead of `pickle.load`.
#. Creating and applying tissue masks before registration. This improves image normalization, reduces the number of poor feature matches, and helps remove unwanted non-rigid deformations (especially around the image edges), all of which improve alignment accuracy. This step can be skipped by setting :code:`create_masks` to `False` when initializing the :code:`Valis` object.
#. Now possible to directly non-rigidly align to the reference image specified by :code:`reference_img_f`. This can be done by setting :code:`align_to_reference` to `True` when initializing the :code:`Valis` object. The default is `False`, which means images will be aligned serially towards the reference image.  This option is also available with :code:`Valis.register_micro`, meaning that one could do a second alignment, but aligning all directly to a reference image.
#. RANSAC filtered matches found for rigid registration undergo second round of filtering, this time using Tukey's method to remove matches whose distance after  being warped would be considered outliers.
#. Now have option off whether or not to compose non-rigid transformations. This can be set specifying the :code:`compose_non_rigid` argument when initialzing the `Valis` object.
#. Can provide rigid transformation matrices by passing in a dictionary to the :code:`do_rigid` parameter when initializing the :code:`Valis` object. Setting :code:`do_rigid` to `False` will completely skip the rigid registration step. See the documentation for initializing the `Valis` object for more details.
#. Added examples of how to read slides and use custom transforms
#. Benchmarked using ANHIR Grand Challenge dataset and posted results on leaderboard.
#. bioformats_jar has been deprecated, so added support for its replacement, scyjava. However, the default behavior will be to use the bioformats_jar JAR file if it's already been installed. One can also now specify the JAR file when calling :code:`init_jvm`.

Version 1.0.0rc6 (April 18, 2022)
---------------------------------
#. More accurate color mixing with fewer artifacts. Affects overlap images and pseudo-colored multi-channel images.
#. Initializing  'is_flattended_pyramid' with False. Pull request #6
#. Reformatting flattened pyramids to have same datatype as that in metadata.
#. Saving all images using pyvips. Should be faster.
#. Using Bio-Formats to read non-RGB ome-tiff. Addresses an issue where converting non-RGB ome-tiff to numpy was very slow.

Version 1.0.0rc5 (April 5, 2022)
---------------------------------
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
