"""
Make sure VALIS still works when images have minimal metadata
"""

import os
import pathlib

import sys
# sys.path.append("/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis")
from valis import registration


from valis.slide_tools import *
from valis.slide_io import *

def get_parent_dir():
    cwd = os.getcwd()
    dir_split = cwd.split(os.sep)
    split_idx = [i for i in range(len(dir_split)) if dir_split[i] == "valis_project"][0]
    parent_dir = os.sep.join(dir_split[:split_idx+1])
    return parent_dir

parent_dir = get_parent_dir()

in_container = sys.platform == "linux" and os.getcwd() == '/usr/local/src'
if in_container:
    results_dst_dir = os.path.join(parent_dir, f"valis/tests/docker")
else:
    results_dst_dir = os.path.join(parent_dir, f"valis/tests/{sys.version_info.major}{sys.version_info.minor}")


# from valis.registration import *
def test_align_min_metadata():
    src_dir = os.path.join(parent_dir, "resources/slides/cycif_no_meta")
    dst_dir = os.path.join(results_dst_dir, "align_min_metadata")
    registrar = registration.Valis(src_dir, dst_dir)
    registrar.register()

    src_f = registrar.original_img_list[0]
    self = VipsSlideReader(src_f)

    registrar.warp_and_save_slides(dst_dir)
    # self = registrar
    registrar.warp_and_merge_slides(dst_f=os.path.join(dst_dir, "merged.ome.tiff"))

    from valis.slide_io import *
    self = slide_obj.reader
    src_f =

    assert True


if __name__ == "__main__":
    test_align_min_metadata()
