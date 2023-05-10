Image pre-processing
********************

Functions
=========
.. automodule:: valis.preprocessing
    :members: standardize_colorfulness, get_luminosity, match_histograms, get_channel_stats, norm_img_stats, rgb2jab, jab2rgb, rgb2jch, rgb2od, stainmat2decon, deconvolve_img, create_tissue_mask_from_rgb, create_tissue_mask_from_multichannel, norm_img_stats, match_histograms


Classes
=======

Base ImageProcesser
-------------------
.. autoclass:: valis.preprocessing::ImageProcesser
    :members: __init__, process_image, create_mask

ChannelGetter
-------------
.. autoclass:: valis.preprocessing::ChannelGetter
    :members:  __init__, process_image, create_mask
    :show-inheritance:

ColorfulStandardizer
--------------------
.. autoclass:: valis.preprocessing::ColorfulStandardizer
    :members:  __init__, process_image, create_mask
    :show-inheritance:

BgColorDistance
--------------------
.. autoclass:: valis.preprocessing::BgColorDistance
    :members:  __init__, process_image, create_mask
    :show-inheritance:

Luminosity
--------------------
.. autoclass:: valis.preprocessing::Luminosity
    :members:  __init__, process_image, create_mask
    :show-inheritance:

H&E deconvolution
--------------------
.. autoclass:: valis.preprocessing::HEDeconvolution
    :members:  __init__, process_image, create_mask
    :show-inheritance: