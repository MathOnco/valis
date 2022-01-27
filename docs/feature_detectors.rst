Feature detectors and descriptors
*********************************
Functions
=========

.. automodule:: valis.feature_detectors
    :members: filter_features

Classes
=========

Base feature detector
---------------------
.. autoclass:: valis.feature_detectors::FeatureDD
    :members: __init__, detect_and_compute

BRISK
-----
.. autoclass:: valis.feature_detectors::BriskFD
    :show-inheritance:

KAZE
----
.. autoclass:: valis.feature_detectors::KazeFD
    :show-inheritance:

AKAZE
-----
.. autoclass:: valis.feature_detectors::AkazeFD
    :show-inheritance:

DAISY
-----
.. autoclass:: valis.feature_detectors::DaisyFD
    :show-inheritance:

LATCH
-----
.. autoclass:: valis.feature_detectors::LatchFD
    :show-inheritance:

BOOST
-----
.. autoclass:: valis.feature_detectors::BoostFD
    :show-inheritance:

VGG
-----
.. autoclass:: valis.feature_detectors::VggFD
    :show-inheritance:

Orb + Vgg
---------
.. autoclass:: valis.feature_detectors::OrbVggFD
    :show-inheritance: