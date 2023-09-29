""" Registration of whole slide images (WSI) using higher resolution images

This example shows how to register the slides using higher resolution images.
An initial rigid transform is found using low resolition images, but the
`MicroRigidRegistrar` can be used to update that transform using feature matches
found in higher resoltion images. This can be followed up by the high resolution
non-rigid registration (i.e. micro-registration).

"""

import sys
sys.path.append("/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis")


andry_dir = "/Users/gatenbcd/Dropbox/Documents/Andriy/Bina_alignments/slides/NSG_from_Marusyk/NSG 48 hours_374"
mrxs_dir = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ihc_mrxs"
andor_dir = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ihc_bf"
cycif_dir = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/cycif"
ad_dir = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/adenoma"
tme_dir = "/Users/gatenbcd/Dropbox/Documents/Lab_Chung/Collaboration_with_Sandy/lab_chung_hnscc/images/TME/06S17070207"
slide_src_dir = andor_dir

import time
import os
import numpy as np
from valis import registration
from valis.micro_rigid_registrar import MicroRigidRegistrar # For high resolution rigid registration


slide_src_dir = "./example_datasets/ihc"
results_dst_dir = "./expected_results/registration_hi_rez"
micro_reg_fraction = 0.25 # Fraction full resolution used for non-rigid registration

# Perform high resolution rigid registration using the MicroRigidRegistrar
start = time.time()
registrar = registration.Valis(slide_src_dir, results_dst_dir, micro_rigid_registrar_cls=MicroRigidRegistrar)
rigid_registrar, non_rigid_registrar, error_df = registrar.register()

# Calculate what `max_non_rigid_registration_dim_px` needs to be to do non-rigid registration on an image that is 25% full resolution.
img_dims = np.array([slide_obj.slide_dimensions_wh[0] for slide_obj in registrar.slide_dict.values()])
min_max_size = np.min([np.max(d) for d in img_dims])
img_areas = [np.multiply(*d) for d in img_dims]
max_img_w, max_img_h = tuple(img_dims[np.argmax(img_areas)])
micro_reg_size = np.floor(min_max_size*micro_reg_fraction).astype(int)

# Perform high resolution non-rigid registration
micro_reg, micro_error = registrar.register_micro(max_non_rigid_registration_dim_px=micro_reg_size)


stop = time.time()
elapsed = stop - start
print(f"regisration time is {elapsed/60} minutes")

# We can also plot the high resolution matches using `Valis.draw_matches`:
matches_dst_dir = os.path.join(registrar.dst_dir, "hi_rez_matches")
registrar.draw_matches(matches_dst_dir)