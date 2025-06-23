Feature matching
****************

Functions
=========

.. automodule:: valis.feature_matcher
    :members: match_desc_and_kp, filter_matches

Classes
=========

MatchInfo
---------
.. autoclass:: valis.feature_matcher::MatchInfo
    :members: __init__

Matcher
-------
.. autoclass:: valis.feature_matcher::Matcher
    :members: __init__, match_images, estimate_rotation

LightGlueMatcher
-----------------
.. autoclass:: valis.feature_matcher::LightGlueMatcher
    :members: __init__, match_images

SuperPointAndGlue
------------------
.. autoclass:: valis.feature_matcher::SuperPointAndGlue
    :members: __init__, match_images

SuperGlueMatcher
------------------
.. autoclass:: valis.feature_matcher::SuperGlueMatcher
    :members: __init__, match_images
