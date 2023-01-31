
Registration
*************

Functions
=========
.. automodule:: valis.registration
    :members: load_registrar, init_jvm, kill_jvm

Classes
=======

Valis
-----
.. autoclass:: valis.registration::Valis
    :members: __init__, register, get_slide, warp_and_save_slides, warp_and_merge_slides

Slide
-----
.. autoclass:: valis.registration::Slide
    :members: __init__, slide2image, slide2vips, warp_img, warp_slide, warp_and_save_slide, warp_xy, warp_xy_from_to
