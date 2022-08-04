"""
This example shows how warp a set of ROI coordinates and use them to slice
the ROI from the registered images.

The steps are as follows:

1. Load a pickled Valis object that has already registered some
slides.

2. Access the Slide object associated with the slide
from which the ROI coordinates originated.

3. Warping of points assumes the coordinates are in pixel units.
However, here the original coordinaates are in microns, and so need to be
converted pixel units.

4. The Slide object can now be used to warp the coordinates.
This yields the ROI coordinates in all of the registered slides.

5. Warp and slice the ROI from each slide using the Slide's pyvips.Image
extract_area method.

Note that because that the Slides are pyvips.Image objects, and so do
not need to be loaded into memory to do the warping. Therefore, warping
and ROI extraction is fast.

It is also worth noting that it's important to know the pyramid level,
or image shape, from which the coordinates originated. If working with
slides, the pyramid level is probably 0, and here is corresponds to the
`COORD_LEVEL` variable. It would also be possible to input the source
image's dimensions.

"""


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pathlib

from valis import registration, warp_tools

# Load a registrar that has been saved
registrar_f = "./expected_results/registration/ihc/data/ihc_registrar.pickle"
registrar = registration.load_registrar(registrar_f)
COORD_LEVEL = 0  # pyramid level from which the ROI coordinates originated. Usually 0.

# ROI coordinates, in microns. These came from the unregistered slide "ihc_2.ome.tiff"
bbox_xywh_um = [14314, 13601, 3000, 3000]
bbox_xy_um = warp_tools.bbox2xy(bbox_xywh_um)

# Get slide from which the ROI coordinates originated
pt_source_img_f = "ihc_2.ome.tiff"
pt_source_slide = registrar.get_slide(pt_source_img_f)

# Convert coordinates to pixel units
um_per_px = pt_source_slide.reader.scale_physical_size(COORD_LEVEL)[0:2]
bbox_xy_px = bbox_xy_um/np.array(um_per_px)

# Warp coordinates to position in registered slides
bbox_xy_in_registered_img = pt_source_slide.warp_xy(bbox_xy_px,
                                                    slide_level=COORD_LEVEL,
                                                    pt_level=COORD_LEVEL)

bbox_xywh_in_registered_img = warp_tools.xy2bbox(bbox_xy_in_registered_img)
bbox_xywh_in_registered_img = np.round(bbox_xywh_in_registered_img).astype(int)

# Create directory where images will be saved
dst_dir = "./expected_results/roi"
pathlib.Path(dst_dir).mkdir(exist_ok=True, parents=True)

# Warp each slide and slice the ROI from it using each pyips.Image's "extract_area" method.
fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
ax = axes.ravel()
for i, slide in enumerate(registrar.slide_dict.values()):
    warped_slide = slide.warp_slide(level=COORD_LEVEL)
    roi_vips = warped_slide.extract_area(*bbox_xywh_in_registered_img)
    roi_img = warp_tools.vips2numpy(roi_vips)
    ax[i].imshow(roi_img)
    ax[i].set_title(slide.name)
    ax[i].set_axis_off()

fig.delaxes(ax[5]) # Only 5 images, so remove 6th subplot
out_f = os.path.join(dst_dir, f"{registrar.name}_roi.png")
plt.tight_layout()
plt.savefig(out_f)
plt.close()

# Opening the slide initialized the JVM, so it needs to be killed
registration.kill_jvm()
