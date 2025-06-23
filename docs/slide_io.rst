Slide I/O
**********

Functions
=========

.. automodule:: valis.slide_io
    :members: init_jvm, kill_jvm, get_slide_reader, create_ome_xml, update_xml_for_new_img, save_ome_tiff, convert_to_ome_tiff

Classes
=======

MetaData
--------
.. autoclass:: valis.slide_io::MetaData
    :members: __init__

SlideReader
-----------
.. autoclass:: valis.slide_io::SlideReader
    :members: __init__, slide2vips, slide2image, create_metadata

BioFormatsSlideReader
---------------------
.. autoclass:: valis.slide_io::BioFormatsSlideReader
    :members: __init__, slide2vips, slide2image, create_metadata
    :inherited-members: SlideReader
    :show-inheritance:

VipsSlideReader
---------------
.. autoclass:: valis.slide_io::VipsSlideReader
    :show-inheritance:
    :members: __init__, slide2vips, slide2image, create_metadata
    :inherited-members: SlideReader

FlattenedPyramidReader
----------------------
.. autoclass:: valis.slide_io::FlattenedPyramidReader
    :show-inheritance:
    :members: __init__, slide2vips, slide2image, create_metadata
    :inherited-members: VipsSlideReader

ImageReader
-----------
.. autoclass:: valis.slide_io::ImageReader
    :show-inheritance:
    :members: __init__, slide2vips, slide2image, create_metadata
    :inherited-members: SlideReader