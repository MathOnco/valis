
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

.. image:: https://zenodo.org/badge/444523406.svg
   :target: https://zenodo.org/badge/latestdoi/444523406


.. .. |coverage| image:: https://codecov.io/gh/readthedocs/readthedocs.org/branch/master/graph/badge.svg
..     :alt: Test coverage
..     :scale: 100%
..     :target: https://codecov.io/gh/readthedocs/readthedocs.org

|
|

.. image::  https://github.com/MathOnco/valis/raw/main/docs/_images/banner.gif

|
|


VALIS, which stands for Virtual Alignment of pathoLogy Image Series, is a fully automated pipeline to register whole slide images (WSI) using rigid and/or non-rigid transformtions. A full description of the method is described in the paper by `Gatenbee et al. 2023 <https://www.nature.com/articles/s41467-023-40218-9>`_. VALIS uses `Bio-Formats <https://www.openmicroscopy.org/bio-formats/>`_, `OpenSlide <https://openslide.org/>`__, `libvips <https://www.libvips.org/>`_, and `scikit-image <https://scikit-image.org/>`_ to read images and slides, and so is able to work with over 300 image formats. Registered images can be saved as `ome.tiff <https://docs.openmicroscopy.org/ome-model/5.6.3/ome-tiff/>`_ slides that can be used in downstream analyses. ome.tiff format is opensource and widely supported, being readable in several different programming languages (Python, Java, Matlab, etc...) and software, such as `QuPath <https://qupath.github.io/>`_, `HALO by Idica Labs <https://indicalab.com/halo/>`_, etc...

The registration pipeline is fully automated and goes as follows:

    .. image::  https://github.com/MathOnco/valis/raw/main/docs/_images/pipeline.png

   #. Images/slides are converted to numpy arrays. As WSI are often too large to fit into memory, these images are usually lower resolution images from different pyramid levels.

   #. Images are processed to single channel images. They are then normalized to make them look as similar as possible. Masks are then created to focus registration on the tissue.

   #. Image features are detected and then matched between all pairs of image.

   #. If the order of images is unknown, they will be optimally ordered based on their feature similarity. This increases the chances of successful registration because each image will be aligned to one that looks very similar.

   #. Images will be aligned *towards* (not to) a reference image. If the reference image is not specified, it will automatically be set to the image at the center of the stack.

   #. Rigid registration is performed serially, with each image being rigidly aligned towards the reference image. That is, if the reference image is the 5th in the stack, image 4 will be aligned to 5 (the reference), and then 3 will be aligned to the now registered version of 4, and so on. Only features found in both neighboring slides are used to align the image to the next one in the stack. VALIS uses feature detection to match and align images, but one can optionally perform a final step that maximizes the mutual information between each pair of images. This rigid registration can optionally be updated by matching features in higher resolution versions of the images (see :code:`micro_rigid_registrar.MicroRigidRegistrar`).

   #. The registered rigid masks are combined to create a non-rigid registration mask. The bounding box of this mask is then used to extract higher resolution versions of the tissue from each slide. These higher resolution images are then processed as above and used for non-rigid registration, which is performed either by:

        * aligning each image towards the reference image following the same sequence used during rigid registration.
        * using groupwise registration that non-rigidly aligns the images to a common frame of reference. Currently this is only possible if `SimpleElastix <https://simpleelastix.github.io>`__ is installed.

   #. One can optionally perform a second non-rigid registration using an even higher resolution versions of each image. This is intended to better align micro-features not visible in the original images, and so is referred to as micro-registration. A mask can also be used to indicate where registration should take place.

   #. Error is estimated by calculating the distance between registered matched features in the full resolution images.

The transformations found by VALIS can then be used to warp the full resolution slides. It is also possible to merge non-RGB registered slides to create a highly multiplexed image. These aligned and/or merged slides can then be saved as ome.tiff images. The transformations can also be use to warp point data, such as cell centroids, polygon vertices, etc...

In addition to registering images, VALIS provides tools to read slides using Bio-Formats and OpenSlide, which can be read at multiple resolutions and converted to numpy arrays or pyvips.Image objects. One can also slice regions of interest from these slides and warp annotated images. VALIS also provides functions to convert slides to the ome.tiff format, preserving the original metadata. Please see examples and documentation for more details.


Full documentation with installation instructions and examples can be found at `ReadTheDocs <https://valis.readthedocs.io/en/latest/>`_.


License
-------

`MIT`_ © 2021-2023 Chandler Gatenbee

.. _MIT: LICENSE.txt