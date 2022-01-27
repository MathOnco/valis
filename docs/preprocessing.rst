Image pre-processing
********************

Functions
=========
.. automodule:: valis.preprocessing
    :members: standardize_colorfulness, get_luminosity, match_histograms, get_channel_stats, norm_img_stats

Classes
=======

Base ImageProcesser
-------------------
.. autoclass:: valis.preprocessing::ImageProcesser
    :members: __init__, process_image

ChannelGetter
-------------
.. autoclass:: valis.preprocessing::ChannelGetter
    :members:  __init__, process_image
    :show-inheritance:

ColorfulStandardizer
--------------------
.. autoclass:: valis.preprocessing::ColorfulStandardizer
    :members:  __init__, process_image
    :show-inheritance:

Luminosity
--------------------
.. autoclass:: valis.preprocessing::Luminosity
    :members:  __init__, process_image
    :show-inheritance:
