import os
import pathlib
import numpy as np
import os

import sys
# sys.path.append("/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis")
from valis import slide_io, warp_tools, valtils, registration


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

def test_warp_other_images():
    # src_dir = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/cycif_sep_channels"
    dst_dir = os.path.join(results_dst_dir, "warp_other_images")

    src_dir = os.path.join(get_parent_dir(), "resources/slides/cycif_sep_channels")

    slide_dst_dir = os.path.join(dst_dir, "warped_slides")
    pathlib.Path(slide_dst_dir).mkdir(exist_ok=True, parents=True)

    try:
        img_list = list(pathlib.Path(src_dir).rglob("*DAPI*"))

        registrar = registration.Valis(src_dir, dst_dir, img_list=img_list)
        rigid_registrar, non_rigid_registrar, error_df = registrar.register()

        for dapi_f in img_list:
            round_dir = os.path.split(dapi_f)[0].split(os.path.sep)[-1].replace(" ", "_")
            src_round_dir = os.path.join(src_dir, round_dir)
            other_img_list = [os.path.join(src_round_dir, f) for f in os.listdir(src_round_dir) if f != os.path.split(dapi_f)[1]]
            slide_obj = registrar.get_slide(dapi_f)
            dapi_thumb_vips = slide_obj.warp_slide(0)
            dapi_thumb = dapi_thumb_vips.numpy()
            for img_f in other_img_list:
                dst_f = os.path.join(slide_dst_dir, f"{valtils.get_name(img_f)}_warped.ome.tiff")
                slide_obj.warp_and_save_slide(src_f=img_f, dst_f=dst_f, compression="jp2k")

                thumb_reader = slide_io.get_slide_reader(dst_f)
                thumb_img = thumb_reader(dst_f).slide2image(0)
                assert not np.all(dapi_thumb==thumb_img), "slide warped version of itself"

                os.remove(dst_f)
                # thumb_img = (255*(thumb_img/thumb_img.max())).astype(np.uint8)
                # thumb_f = dst_f.replace(".ome.tiff", ".png")
                # warp_tools.save_img(thumb_f, thumb_img)

                # import ome_types
                # ome_types.from_tiff(dst_f)
        assert True
    except Exception as e:
        print(e)
        assert False