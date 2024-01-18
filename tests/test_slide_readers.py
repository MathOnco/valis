"""
Make sure that correct slide reader is being used and can read specified slide

# TODO
* what happens when trying to read mulitseries ome.tiff with libvips?
* is reading IF ome.tiff faster with libvips or bio-formats?
"""

import os
from time import time
from aicspylibczi import CziFile

import sys
sys.path.append("/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis")
from valis import slide_io, slide_tools

def test_vips_reader():
    """

    """

    svs = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ihc_bf/6069 idh2 cd34.svs"
    openslide_only = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ihc_mrxs/DCIS_6_ELASTASE_CK.mrxs"
    one_series_ome_rgb = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/valis_ome_tiff/ihc_1.ome.tiff"
    one_series_ome_if = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/valis_ome_tiff/cycif.ome.tiff"
    large_jpeg_rgb = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/debugging/github_issue_84/images/1.jpg"
    large_jpeg_if = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/debugging/github_issue_84/images/2.jpg"
    large_tif = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/debugging/github_issue_81/skin HMGB1 round 3 with neg ctrl_TileScan 1_Merged_ch00/4i 1st round skin lamin B1 tile _TileScan 1_Merged_ch00.tif"

    img_f_list = [
          svs,
          openslide_only,
          one_series_ome_rgb,
          one_series_ome_if,
          large_jpeg_rgb,
          large_jpeg_if,
          large_tif#,
        #   nifti
        ]

    expected_reader_cls = slide_io.VipsSlideReader
    for src_f in img_f_list:
        slide_reader_cls = slide_io.get_slide_reader(src_f)
        assert slide_tools.get_img_type(src_f) is not None

        assert slide_reader_cls == expected_reader_cls, print(f"expected to get {expected_reader_cls.__name__} but got {slide_reader_cls.__name__}")

        slide_reader = slide_reader_cls(src_f)
        vips_img = slide_reader.slide2vips(0)

        # slide_reader_cls = expected_reader_cls
        # slide_reader = slide_reader_cls(src_f)
        # src_f = img_f


def test_bf_reader():


    n_series_ome_if = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ome_tiff_IF/LuCa-7color_Scan1.ome (1).tiff"
    n_series_ome_rgb = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ome_tiff_rgb/Leica-1.ome.tiff"
    bf_only_ndpis = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/cycif/Round 1/Tris CD20 FOXP3 CD3.ndpis"
    bf_only_vsi = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/debugging/tony_yeung/images/additional/Slide160_cycle3_01.vsi"
    rgb_czi = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/aics_czi/RGB-8bit.czi"
    mosiac_czi = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/aics_czi/mosaic_test.czi"

    # nifti = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/benchmarking/historeg/HistoReg/CBICA_Toolkit/data/nifti1/1.nii.gz"
    # bmp = "/Users/gatenbcd/Dropbox/Documents/image_processing/alignment_paper/comparison_to_Wang2015/comparison_to_Wang2015_pycode/valis_alignment/full_size/0_source_raw_52.bmp"


    img_f_list = [
        n_series_ome_if,
        n_series_ome_rgb,
        bf_only_ndpis,
        bf_only_vsi,
        rgb_czi,
        mosiac_czi
        # nifti
        # bmp
    ]

    # import importlib
    # importlib.reload(slide_io)
    # from valis.slide_io import *
    # src_f = img_f
    # src_f = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/debugging/github_issue_81/skin HMGB1 round 3 with neg ctrl_TileScan 1_Merged_ch00/4i 1st round skin lamin B1 tile _TileScan 1_Merged_ch00.tif"
    # src_f = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/ihc_bf/6069 idh2 cd34.svs"
    expected_reader_cls = slide_io.BioFormatsSlideReader
    for src_f in img_f_list:
        slide_reader_cls = slide_io.get_slide_reader(src_f)
        assert slide_reader_cls == expected_reader_cls, print(f"expected to get {expected_reader_cls.__name__} but got {slide_reader_cls.__name__}")

        slide_reader = slide_reader_cls(src_f)
        n_levels = len(slide_reader.metadata.slide_dimensions)


        vips_img = slide_reader.slide2vips(n_levels-1)

        assert slide_tools.get_img_type(src_f) is not None



def test_flattened_pyramid_reader():

    import os

    import sys
    sys.path.append("/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis")
    from valis import slide_io

    qptiff = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/debugging/qptiff/37_C1D15_[46831,12690]_component_data.qptif"
    halo_stitched_tif = "/Users/gatenbcd/Dropbox/Documents/LungU01/PhaseII/PhaseII_VectraData/WTS Beg Anderson Data/Beg_P1_76_Slides_WTS stitched Images/Beg_P1_48_C1D15_06S17081023.tif"
    # halo_stitched_tif = "/Users/gatenbcd/Dropbox/Documents/SimonL/eoghan_adincar/images/batch_1/2358/Hu_Stromal_2385-17_K_FUSED.tif"

    img_f_list = [
        qptiff,
        halo_stitched_tif
    ]


    # import importlib
    # importlib.reload(slide_io)
    # from valis.slide_io import *

    expected_reader_cls = slide_io.FlattenedPyramidReader
    # src_f = qptiff
    src_f = halo_stitched_tif
    for src_f in img_f_list:
        assert os.path.exists(src_f)
        slide_reader_cls = slide_io.get_slide_reader(src_f)
        assert slide_reader_cls == expected_reader_cls, print(f"expected to get {expected_reader_cls.__name__} but got {slide_reader_cls.__name__}")

        slide_reader = slide_reader_cls(src_f)

        n_levels = len(slide_reader.metadata.slide_dimensions)
        vips_img = slide_reader.slide2vips(n_levels-1)

        # import ome_types
        # ome_types.from_xml(slide_reader.metadata.original_xml)

        # slide_reader.metadata.channel_names

        assert slide_tools.get_img_type(src_f) is not None


def test_sk_img_reader():
    bmp = "/Users/gatenbcd/Dropbox/Documents/image_processing/alignment_paper/comparison_to_Wang2015/comparison_to_Wang2015_pycode/valis_alignment/full_size/0_source_raw_52.bmp"


def test_czi_jpegxr():
    rgb_czi_jpegxr = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/resources/slides/czi_jpegxr/2CS003-4_06-Stitching.czi"
    single_c_czi_jpegxr = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/debugging/github_issue_76/src/JK023-15.czi"
    czi_f = "/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/debugging/github_issue_60/czi_image/32161012-014.czi"

    img_f_list = [
        rgb_czi_jpegxr,
        single_c_czi_jpegxr,
        czi_f
    ]


    import sys
    sys.path.append("/Users/gatenbcd/Dropbox/Documents/image_processing/valis_project/valis")
    from valis import slide_io

    # import importlib
    # importlib.reload(slide_io)
    # from valis.slide_io import *
    expected_reader_cls = slide_io.CziJpgxrReader
    src_f = czi_f
    for src_f in img_f_list:
        czi = CziFile(src_f)
        comp_tree = czi.meta.findall(".//OriginalCompressionMethod")
        if len(comp_tree) > 0:
            # is_czi_jpgxr = comp_tree[0].text.lower() == "jpgxr"
            # print(src_f, is_czi_jpgxr)
            slide_reader_cls = slide_io.get_slide_reader(src_f)
            assert slide_reader_cls == expected_reader_cls, print(f"expected to get {expected_reader_cls.__name__} but got {slide_reader_cls.__name__}")
            slide_reader = slide_reader_cls(src_f)

            n_levels = len(slide_reader.metadata.slide_dimensions)
            vips_img = slide_reader.slide2vips(n_levels-1)

        assert slide_tools.get_img_type(src_f) is not None



def test_compare_if_ome_readtime():
        """
        Expect that libvips should read one-series ome-tiffs more quickly
        """
        large_series_ome_if = "/Users/gatenbcd/Dropbox/Documents/BCI-EvoCa2/chandler/CycIF_example/registered_slides/K3.ome.tiff"


        v_start = time()
        vips_reader = slide_io.VipsSlideReader(large_series_ome_if)
        vips_img = vips_reader.slide2vips(level=0)
        v_stop = time()
        v_elapsed = v_stop - v_start

        b_start = time()
        bf_reader = slide_io.BioFormatsSlideReader(large_series_ome_if)
        bf_img = bf_reader.slide2vips(level=0)
        b_stop = time()
        b_elapsed = b_stop - b_start

        assert v_elapsed < b_elapsed

