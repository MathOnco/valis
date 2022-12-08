
import sys
import pathlib
import os
import numpy as np

from valis import slide_io, valtils, warp_tools



# docker run -it --rm --name openslide -v "$HOME:$HOME" valis /Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis/docker_openslide_test.py -src_f "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ihc_mrxs/DCIS_6_ELASTASE_CK.mrxs"

# run -d option detaches so can run in background
# docker volume create valis_vol
# Create volume, run script that saves results in volume (deleting container with --rm), then copy results to host
#  docker run -it --rm --name openslide -v valis-vol:"$HOME" valis docker_openslide_test.py -src_f "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ihc_mrxs/DCIS_6_ELASTASE_CK.mrxs"

#  docker run -it --rm --name openslide --mount source=valis-vol,target="$HOME" valis docker_openslide_test.py -src_f "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ihc_mrxs/DCIS_6_ELASTASE_CK.mrxs"


# docker run -it --rm --name openslide --mount source=valis-vol,target="$HOME" valis /Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis/docker_openslide_test.py -src_f "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ihc_mrxs/DCIS_6_ELASTASE_CK.mrxs"

# docker volume create valis_vol

# docker run -it --rm --name openslide -v valis_vol:"$HOME" valis /Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis/docker_openslide_test.py -src_f "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ihc_mrxs/DCIS_6_ELASTASE_CK.mrxs"


# docker run -it --rm --name openslide -v "$HOME:$HOME" valis /Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis/docker_openslide_test.py -src_f "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ihc_mrxs/DCIS_6_ELASTASE_CK.mrxs"

# docker run -it --name openslide --mount type=volume,source="HOME",target="HOME" valis /Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis/docker_openslide_test.py -src_f "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ihc_mrxs/DCIS_6_ELASTASE_CK.mrxs"

# docker run -d --name openslide -v "$HOME:$HOME" valis /Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis/docker_openslide_test.py -src_f "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ihc_mrxs/DCIS_6_ELASTASE_CK.mrxs"

# src_f = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ihc_mrxs/DCIS_6_ELASTASE_CK.mrxs"
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process images for alignment')
    parser.add_argument('-src_f', type=str, help='source image to read')
    args = vars(parser.parse_args())

    src_f = args["src_f"]

    MAX_SIZE = 2000

    reader_cls = slide_io.get_slide_reader(src_f)
    reader = reader_cls(src_f)
    level = np.where(np.max(reader.metadata.slide_dimensions, axis=1)  < MAX_SIZE)[0][0]
    # reader.metadata.slide_dimensions[level]
    img = reader.slide2image(level)
    # out_f = os.path.join(os.getcwd(), valtils.get_name(src_f) + "_docker_mount.png")
    dst_dir = os.path.split(src_f)[0]
    out_f = os.path.join(dst_dir, valtils.get_name(src_f) + "_docker_mount.png")
    print("saving", out_f)
    warp_tools.save_img(out_f, img)






