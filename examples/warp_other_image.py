"""
Example showing how to using the registration parameters to warp another image.

Example image will be larger version of the tissue mask
"""

import sys
from valis import slide_io, warp_tools, valtils, registration, warp_tools
import matplotlib.pyplot as plt

# os.getcwd()
import time
import os
from valis import registration, warp_tools
from skimage import filters, measure, morphology, segmentation, color
import numpy as np


slide_src_dir = "./example_datasets/ihc"
slide_src_dir = "./valis/examples/example_datasets/ihc"

results_dst_dir = "./valis/examples/warp_associated_img/"

# Create a Valis object and use it to register the slides in slide_src_dir
start = time.time()
registrar = registration.Valis(slide_src_dir, results_dst_dir)
rigid_registrar, non_rigid_registrar, error_df = registrar.register()
stop = time.time()
elapsed = stop - start

print(f"regisration time is {elapsed/60} minutes")


# Create examples of associated images to warp.

slide_obj = registrar.get_slide("ihc_2")
processed_img = slide_obj.pad_cropped_processed_img()
s = 2
new_shape = np.array(processed_img.shape[0:2])*s
img = warp_tools.resize_img(slide_obj.image, new_shape)

# In this example, we'll create a mask, based on a larger version of the image used during registration

gray_img = 1 - color.rgb2gray(img)
mask = gray_img > filters.threshold_li(gray_img)

# Use the associated Slide to warp the mask. Need to set `interp_method="nearest"` since this is a binary image
warped_mask = slide_obj.warp_img(mask, interp_method="nearest")

# Use the associated Slide to warp larger version of the image
warped_scaled_img =  slide_obj.warp_img(img)

# Visualise mask overlaid on the warped image
mask_boundaries = segmentation.find_boundaries(warped_mask)
warped_scaled_img[mask_boundaries] = [0, 255, 0]

plt.imshow(warped_scaled_img)
plt.show()


# Shutdown the JVM
registration.kill_jvm()
