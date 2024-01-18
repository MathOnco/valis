
import os
import pathlib

import sys
sys.path.append("/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis")
from valis import registration


from valis.slide_tools import *

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


def test_align_min_metadata():
    src_dir = os.path.join(parent_dir, "resources/slides/cycif_no_meta")
    dst_dir = os.path.join(results_dst_dir, "align_min_metadata")
    registrar = registration.Valis(src_dir, dst_dir)
    registrar.register()

    registrar.warp_and_save_slides(dst_dir)
    registrar.warp_and_merge_slides(dst_f=os.path.join(dst_dir, "merged.ome.tiff"))

    # self = registrar

if __name__ == "__main__":
    test_align_min_metadata()
