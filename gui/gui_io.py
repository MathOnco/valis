
import pathlib
import os
import pandas as pd

import sys
sys.path.append("/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis")
from valis import slide_io, slide_tools, valtils

sys.path.append("/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis/gui")
import gui_options
import inspect

# import ctypes
import os
# import stat


# gui_options.PROCESSOR_KEY = "image processors"
# IF_PROCESSOR_KEY = "if processor"
# BF_PROCESSOR_KEY = "bf processor"

SAMPLE_NAME_KEY = "Sample"
"""str: Name of sample"""

FILE_NAME_KEY = "File"
"""str or Pathlike: Path to file in directory associated with `Sample`"""

USE_KEY = "Align"
"""Bool: Whether or not the file is an image that will be aligned"""

USE_KEY = "Align"

# class SampleInfo(object):
#     def __init__(self, image_list):
#         """
#         """
#         self.image_list = image_list

# class ImageInfo(object):
#     def __init__(self, src_f):
#         """
#         """
#         self.src_f = src_f

# filepath = path_list[0]
def get_image_list(src_dir, ignore_csv=True):
    """
    Assumes that the user wants to align all images found in a common directory.

    This function will use the contents of `src_dir` to determine which files belong to the same sample.
    There are several possible scenarios:
    * `src_dir` does not contain subdirectories, and so is assumed to contain files associated with a single sample
    * `src_dir` contains mix of images and subdirectories
        * Possibly something like .mrxs, where there are index files and subdirectories with image tiles
        * Could also a parent directory containing images and folders (where each folder is a sample)
    * `src_dir` contains only folders
        * Each folder is a sample (use logic above)
        * Each folder contains a staining round (e.g. .ndpi images with an .npdis)
    """


    input_df_dict = {}

    # dir_list = list(os.walk(src_dir))
    # root, d_names, f_names = dir_list[20] # csv
    # root, d_names, f_names = dir_list[0]
    for root, d_names, f_names in os.walk(src_dir):
        # print(root, d_names, f_names)

        is_round, master_slide = slide_tools.determine_if_staining_round(root)
        root_root = root.split(os.sep)[-2]
        if is_round:
            sample_name = root_root
        else:
            sample_name = os.path.split(root)[1]

        in_list = sample_name in input_df_dict or root_root in input_df_dict
        if in_list:
            continue

        images_in_dir = []
        subdir_is_round = False
        # `src_dir` contains subdirectories. Check if these are staining
        if len(d_names) > 0:
            for d in d_names:
                full_d = os.path.join(root, d)
                subdir_is_round, master_slide = slide_tools.determine_if_staining_round(full_d)
                if subdir_is_round and sample_name not in input_df_dict:
                    images_in_dir.append(master_slide)

        if len(images_in_dir) == 0:
            images_in_dir = [os.path.join(root, f) for f in f_names if slide_tools.get_img_type(os.path.join(root, f)) is not None]
            if ignore_csv:
                images_in_dir = [f for f in images_in_dir if not f.endswith(".csv")]

        contains_images = len(images_in_dir) > 0
        contains_dirs = len(d_names) > 0

        full_f_names = [os.path.join(root, f) for f in f_names]
        if subdir_is_round:
            round_dirs = [os.path.join(root, d) for d in d_names]
            # all_files = [os.path.join(d, f) for f in os.listdir(d) for d in round_dirs]
            _all_files = [list(pathlib.Path(d).iterdir()) for d in round_dirs]
            all_files = [str(item) for sublist in _all_files for item in sublist if item.is_file()]

            sample_df = pd.DataFrame({SAMPLE_NAME_KEY:sample_name,
                                      FILE_NAME_KEY: all_files,
                                      USE_KEY:[True if f in images_in_dir else False for f in all_files]
                                      })
            input_df_dict[sample_name] = sample_df

        # Check if folders and files have the same names
        elif contains_images and contains_dirs and not in_list:
            img_names = [valtils.get_name(f) for f in images_in_dir]
            img_has_dir = [x in d_names for x in img_names]
            if all(img_has_dir):
                # contents of root are for single sample, but something like .mrxs, where each image file has a directory of subimages
                sample_df = pd.DataFrame({SAMPLE_NAME_KEY:sample_name,
                                          FILE_NAME_KEY:full_f_names,
                                          USE_KEY:[True if f in images_in_dir else False for f in full_f_names]})

                input_df_dict[sample_name] = sample_df

        elif contains_images and not contains_dirs and not in_list:
            # contents of root are for single sample
            sample_df = pd.DataFrame({SAMPLE_NAME_KEY:sample_name,
                                      FILE_NAME_KEY:full_f_names,
                                      USE_KEY:[True if f in images_in_dir else False for f in full_f_names]})

            input_df_dict[sample_name] = sample_df


    input_df  = pd.concat(input_df_dict)
    input_df.to_csv("example_project_structure.csv", index=False)
    #     for f in f_names:
    #         # Check if each file is an image
    #         full_f = os.path.join(root, f)
    #         if slide_tools.get_img_type(full_f) is not None:
    #             if sample_name not in sample_list:
    #                 sample_list.append(sample_name)


    ## Assume maximum depth is 2
    # path_obj = pathlib.Path(src_dir)
    # if path_obj.is_file():
    #     return None, None

    # subs = list(path_obj.iterdir())
    # # dir_list = [x for x in subs if x.is_dir()]
    # # img_list = [slide_tools.get_img_type(sub_obj) is not None]

    # # contains_subdirs = False
    # img_list = []
    # dir_list = []
    # for sub_obj in subs:
    #     if sub_obj.is_file():
    #         if slide_tools.get_img_type(sub_obj) is not None:
    #             print(sub_obj)
    #             img_list.append(sub_obj)
    #     elif sub_obj.is_dir():
    #         dir_list.append(sub_obj)

    # if len(img_list) == 0 and len(dir_list) > 0:
    #     # is_round, master_slide = slide_tools.determine_if_staining_round(src_dir)
    #     for d in dir_list:
    #         is_round, master_slide = slide_tools.determine_if_staining_round(d)
    #         if is_round:
    #             img_list.append(master_slide)
    #         else:
    #             print(f"recursing for {src_dir}")
    #             sample_name, img_list = get_image_list(d)

    # sample_name = os.path.split(src_dir)[1]

    # return sample_dict





            # self.original_img_list.append(f)
            # img_names.append(valtils.get_name(f))



if __name__ == "__main__":
    sample_list = []

    # src_dir = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides"
    # src_dir = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ihc_bf"
    # src_dir = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/cycif"
    # src_dir = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ihc_mrxs"

    path_obj = pathlib.Path(src_dir)
    path_list = [x for x in path_obj.iterdir() if not os.path.split(x)[1].startswith(".")]

    all_subirs = all([x.is_dir() for x in path_list])
    has_rounds = all([slide_tools.determine_if_staining_round(d)[0] for d in path_obj.iterdir()])


    if not all_subirs or has_rounds:
        sample_name, img_list = get_image_list(src_dir)
        sample_list.append(sample_name)
    else:
        for d in os.listdir(src_dir):
            full_d = os.path.join(src_dir, d)
            sample_name, img_list = get_image_list(full_d)
            if sample_name is None:
                continue

            sample_list.append(sample_name)


