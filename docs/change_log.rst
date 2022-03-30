Change Log
**********

Version 1.0.0rc5 (March 24, 2022)
=================================
 #. Can provide a reference image that the others will be aligned towards. To do this, when initializinig the Valis object, set the `reference_img_f` argument to be the file name of the reference image. If not set by the user, the reference image will be set as the one at the center of the ordered image stack
 #. Both non-rigid and rigid now align *towards* a reference image, meaning that reference image will have neither rigid nor non-rigid transformations applied to initializinig
 #.