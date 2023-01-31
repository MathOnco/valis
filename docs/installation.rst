Installation
************

.. note::
    Currently, VALIS requires Python 3.9 or 3.10

DockerHub
=========
VALIS is available as a Docker image and can be downloaded from `DockerHub <https://hub.docker.com/r/cdgatenbee/valis-wsi>`_. Starting a container will launch an Ubuntu shell, and so Python needs to be called when executing the script. In this example, the user has a file called "register.py" that takes :code:`src_dir` and :code:`dst_dir` arguments, which registers all of the images in :code:`src_dir` and saves the results in :code:`dst_dir`. This example bind mounts the home directory, and thus the full paths need to be specified.

.. code-block:: bash
  
    $ docker run --memory=20g  -v "$HOME:$HOME" valis-wsi python3 full/path/to/register.py -src_dir full/path/to/images_to_align -dst_dir full/path/to/where_to_save_results


.. important::
    To avoid the container from shutting down prematurely, be sure to set appropriately high memory limits (including in Docker Desktop).

PyPi
=====
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

#. Install `Maven <https://maven.apache.org/index.html>_`, which is also required to use Bioformats

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
~~~~~~~~~~~~~~~~~~~~~~~~

The defaults used by VALIS work well, but VALIS also provides optional classes that require `SimpleElastix <https://simpleelastix.github.io>`_. In particular, these classes are:

#. affine_optimizer.AffineOptimizerMattesMI, which uses sitk.ElastixImageFilter to simultaneously maximize Mattes Mutual Information and minimize the spatial distance between matched features.


#. non_rigid_registrars.SimpleElastixWarper, which uses sitk.ElastixImageFilter to find non-rigid transformations between pairs of images.


#. non_rigid_registrars.SimpleElastixGroupwiseWarper, which uses sitk.ElastixImageFilter to find non-rigid transformations using groupwise registration.

To install SimpleElastix, you should probably uninstall the current version of SimpleITK in your environment, and then install SimpleElastix as described in the `SimpleElastix docs <https://simpleelastix.readthedocs.io/GettingStarted.html>`_.

From source
============
One will need to install and use `Poetry <https://python-poetry.org/>`_ to install VALIS from the source code. As Poetry only installs the Python dependencies, one will also need to follow the steps above to install the JDK, Maven, libvips, and openslide. Note that the poetry lock file is included in the repository, which can be deleted before installation if so desired.