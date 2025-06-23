"""
Compare performance of feature detectors
"""
import os
import sys
sys.path.append("/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis")
import time
import torch
import numpy as np
from valis import registration, valtils, feature_detectors, feature_matcher, preprocessing, viz, warp_tools

import matplotlib.pyplot as plt

def get_dirs():
    cwd = os.getcwd()
    in_container = sys.platform == "linux" and os.getcwd() == cwd
    if not in_container:
        dir_split = cwd.split(os.sep)
        split_idx = [i for i in range(len(dir_split)) if dir_split[i] == "valis_project"][0]
        parent_dir = os.sep.join(dir_split[:split_idx+1])

        results_dst_dir = os.path.join(parent_dir, f"valis/tests/{sys.version_info.major}{sys.version_info.minor}")
    else:
        parent_dir = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project"
        results_dst_dir = os.path.join(parent_dir, f"valis/tests/docker")

    return parent_dir, results_dst_dir, in_container


def cnames_from_filename(src_f):
    """Get channel names from file name
    Note that the DAPI channel is not part of the filename
    but is always the first channel.

    """
    f = valtils.get_name(src_f)
    return ["DAPI"] + f.split(" ")


# from valis import slide_io
# slide_io.init_jvm("path/to/bioformats.jar")
# from valis import registration

# registrar = registration.Valis(ihc_src_dir, dst_dir)
# _, _, error_df = registrar.register(reader_cls=slide_io.VipsSlideReader)


parent_dir, results_dst_dir, in_container = get_dirs()
results_dst_dir = os.path.join(results_dst_dir, "examples/feature_detectors")
datasets_src_dir = os.path.join(parent_dir, "valis/examples/example_datasets/")
ihc_src_dir = os.path.join(datasets_src_dir, "ihc")

# Use DeDoDe RGB features
brightfield_processing_cls=preprocessing.OD
brightfield_processing_kwargs={"adaptive_eq":False}

dedode_matcher_obj = feature_matcher.LightGlueMatcher(feature_detectors.DeDoDeFD(rgb=False))
disk_matcher_obj = feature_matcher.LightGlueMatcher(feature_detectors.DiskFD(rgb=False))
default_matcher_obj = feature_matcher.Matcher()

matcher_list = [dedode_matcher_obj, disk_matcher_obj, default_matcher_obj]
n_matchers = len(matcher_list)
elapsed_times = n_matchers*[None]
avg_errors = n_matchers*[None]
avg_matches = n_matchers*[None]
reg_list = n_matchers*[None]

for i, matcher in enumerate(matcher_list):
    dst_dir = os.path.join(results_dst_dir, matcher.feature_detector.__class__.__name__)
    start = time.time()
    registrar = registration.Valis(ihc_src_dir, dst_dir, matcher=matcher)
    _, _, error_df = registrar.register(brightfield_processing_cls=brightfield_processing_cls, brightfield_processing_kwargs=brightfield_processing_kwargs)
    stop = time.time()
    elapsed = stop - start

    registrar.draw_matches(registrar.dst_dir)
    ref_slide_obj = registrar.get_ref_slide()
    n_matches = [slide_obj.xy_in_prev.shape[0] for slide_obj in registrar.slide_dict.values() if slide_obj != ref_slide_obj]

    avg_errors[i] = np.max(error_df["mean_non_rigid_D"])
    reg_list[i] = registrar
    elapsed_times[i] = elapsed
    avg_matches[i] = np.mean(n_matches)

default_reg_idx = [i for i in range(n_matchers) if not isinstance(matcher_list[i], feature_matcher.LightGlueMatcher)][0]
default_reg = reg_list[default_reg_idx]
ref_slide_obj = default_reg.get_ref_slide()
slide_list =[slide_obj for slide_obj in registrar.slide_dict.values() if slide_obj != ref_slide_obj]
# n_matches = [slide_obj.xy_in_prev.shape[0] for slide_obj in slide_list]
# min_matches_idx = np.argmin(n_matches)
# moving_slide_name = slide_list[min_matches_idx].name
# fixed_slide_name = slide_list[min_matches_idx].fixed_slide.name


for slide_obj in slide_list:
    moving_slide_name = slide_obj.name
    fixed_slide_name = slide_obj.fixed_slide.name
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))
    ax = axes.ravel()
    for i, reg in enumerate(reg_list):
        moving_slide = reg.get_slide(moving_slide_name)
        fixed_slide = reg.get_slide(fixed_slide_name)

        if moving_slide.image.ndim == 3 and moving_slide.is_rgb:
            moving_draw_img = warp_tools.resize_img(moving_slide.image, moving_slide.processed_img_shape_rc)
        else:
            moving_draw_img = moving_slide.pad_cropped_processed_img()

        if fixed_slide.image.ndim == 3 and fixed_slide.is_rgb:
            fixed_draw_img = warp_tools.resize_img(fixed_slide.image, fixed_slide.processed_img_shape_rc)
        else:
            fixed_draw_img = fixed_slide.pad_cropped_processed_img()

        all_matches_img = viz.draw_matches(src_img=moving_draw_img, kp1_xy=moving_slide.xy_matched_to_prev,
                                            dst_img=fixed_draw_img,  kp2_xy=moving_slide.xy_in_prev,
                                            rad=3, alignment='vertical')

        fd_name = matcher_list[i].feature_detector.__class__.__name__
        n_matches = moving_slide.xy_matched_to_prev.shape[0]
        ax[i].imshow(all_matches_img)
        ax[i].set_title(f"{fd_name}= {n_matches} matches")
        ax[i].tick_params(left = False, right = False , labelleft = False ,
        labelbottom = False, bottom = False)

    fig.tight_layout()
    # plt.tick_params(left = False, right = False , labelleft = False ,
    #         labelbottom = False, bottom = False)
    plt.savefig(os.path.join(results_dst_dir, f"{moving_slide_name}_to_{fixed_slide_name}.png"))
    plt.close()


