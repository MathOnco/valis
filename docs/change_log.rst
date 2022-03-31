Change Log
**********

Version 1.0.0rc5 (March 24, 2022)
=================================
 #. Can provide a reference image that the others will be aligned towards. To do this, when initializinig the Valis object, set the `reference_img_f` argument to be the file name of the reference image. If not set by the user, the reference image will be set as the one at the center of the ordered image stack
 #. Both non-rigid and rigid now align *towards* a reference image, meaning that reference image will have neither rigid nor non-rigid transformations applied to initializinig
 #. Two cropping methods. First option is to crop seach registered slides to contain only the areas where all registered images overlap. The second option is to crop the registered slide to contain only the area that intersects with the reference image.
 #. Warping of images and slides done using the same function, built around pyvips.
 #. Read slides using pyvips whenever possible.
 #. Background color now automatically set to be same as the brightest (IHC) or darkes (IF) pixel in the image. Because of this, the "bg_color" argument was removed.
 #. Reduced unwanted non-rigid deformations by masking displacement fields before warp
 #. Displacement fields drawn ontop of non-rigid registered image to help determine where the deformations are happening.
 #. If the slide has multiple series, and a series is not specficed, the slide reader will read the series containing the largest image.


