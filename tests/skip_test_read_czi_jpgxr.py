"""
Test reading czi images with JPGXR compression
"""

import glob
import os
import numpy as np
import sys
sys.path.append("/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis")
from valis import slide_io



def get_parent_dir():
    cwd = os.getcwd()
    dir_split = cwd.split(os.sep)
    split_idx = [i for i in range(len(dir_split)) if dir_split[i] == "valis_project"][0]
    parent_dir = os.sep.join(dir_split[:split_idx+1])
    return parent_dir


# def read_mosaic(self, level=0, xywh=None, *args, **kwargs):
#     # level=1
#     scale_factor = self._get_zoom_levels()[level]
#     czi_reader = CziFile(self.src_f)
#     mosaic_data = czi_reader.read_mosaic(C=0, scale_factor=scale_factor)
#     is_rgb = self._check_rgb()
#     is_bgr = czi_reader.pixel_type.startswith("bgr")
#     if is_rgb and self.is_bgr:


# def run_example():

#     src_f = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/aics_czi/mosaic_test.czi"
#     # src_f = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/aics_czi/RGB-8bit.czi"
#     reader_cls = slide_io.CziJpgxrReader
#     reader = reader_cls(src_f)
#     self = reader
#     czi_reader = CziFile(src_f)

#     # s = np.mean(reader.metadata.slide_dimensions[-1]/reader.metadata.slide_dimensions[0])
#     self = czi_reader
#     plane_constraints = self._get_coords_from_kwargs({"C":0})
#     background_color = self.czilib.RgbFloat()
#     background_color.r = 0.0
#     background_color.g = 0.0
#     background_color.b = 0.0

#     region = self.czilib.BBox()
#     region.w = -1
#     region.h = -1

#     mosaic_data = czi_reader.read_mosaic(C=0, scale_factor=0.1)

#     self.reader.read_mosaic(plane_constraints, 1.0, region, background_color)


def test_convert_czi_jpegxr():

    slide_io.init_jvm()

    parent_dir = get_parent_dir()
    img_dir = "resources/slides/czi_jpegxr"
    slide_src_dir = os.path.join(parent_dir, img_dir)

    img_list = glob.glob(slide_src_dir + '/*' + ".czi")
    img_f = img_list[0]
    try:
        reader_cls = slide_io.get_slide_reader(img_f)
        reader = reader_cls(img_f)

    except Exception as e:
        print("FAILED", e)

    print("read")

    test_img_size = 500
    level = np.where(np.max(reader.metadata.slide_dimensions) <= test_img_size)[0]
    if len(level) == 0:
        level = len(reader.metadata.slide_dimensions) - 1

    img = reader.slide2vips(level=level)


    # print("converted")
    # slide_io.kill_jvm()

    assert True

    # print("shutdown")
    # quit()



