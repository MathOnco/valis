Version 1.0.0rc5 (April 5, 2022)
--------------------------------
#. Can provide a reference image that the others will be aligned towards. To do this, when initializinig the Valis object, set the :code:`reference_img_f` argument to be the file name of the reference image. If not set by the user, the reference image will be set as the one at the center of the ordered image stack
#. Both non-rigid and rigid now align *towards* a reference image, meaning that reference image will have neither rigid nor non-rigid transformations applied to it.
#. Two cropping methods. First option is to crop seach registered slides to contain only the areas where all registered images overlap. The second option is to crop the registered slide to contain only the area that intersects with the reference image. It is also possible to not crop an image/slide.
#. Images are now cropped during the warp, not after, and so is now faster and requires less memory. For example, on a 2018 MacBook Pro with a 2.6 GHz Intel Core i7 processor, it takes 2-3 minutes to warp and save a 41399 x 43479 RGB image.
#. Warping of images and slides done using the same function, built around pyvips. Faster, more consistent, and should prevent excessive memory usage.
#. Fixed bug that caused a crash when warping large ome.tiff images.
#. Read slides and images using pyvips whenever possible.
#. Background color now automatically set to be same as the brightest (IHC) or darkest (IF) pixel in the image. Because of this, the "bg_color" argument in the slide warping functions was removed.
#. Reduced accumulation of unwanted non-rigid deformations
#. Displacement fields drawn on top of non-rigid registered image to help determine where the deformations occured.
#. If a slide has multiple series, and a series is not specficed, the slide reader will read the series containing the largest image.
