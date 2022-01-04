Welcome to VALIS' documentation!
===================================

**VALIS** stands for Virtual Alignment of pathoLogy Image Series, and is a Python library that 
registers (align) whole slide images (WSI). It uses Bioformats and OpenSlide to read slides, 
and so is compatible with many formats. VALIS will find the rigid and non-rigid registration
parmaters, which can be used to warp the WSI. The method used to find the registration parameters
is described in https://www.biorxiv.org/content/10.1101/2021.11.09.467917v1.abstract. 
Pyvips is used to warp the slides, and so VALIS is able to register very large images. 
Pyvips is then used to save the registered slides in the ome.tiff format for downstream analysis. 
In addition to warping images, VALIS can also use the registration parameters to warp points (e.g. cell positions), and is also
able to convert slides to the ome.tiff format.

VALIS has its documentation hosted on Read the Docs.
.. note::
   Coming soon...
   This project is under active development.

Contents
--------

.. toctree::

   usage
   api
