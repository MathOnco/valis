"""Smoketest: align two images using the same code path as examples/align_two_images.py.

Downloads the two smallest example images from the upstream repo on first run
and caches them in tests/example_datasets/ so subsequent runs are offline.

Visual artifacts (overlap thumbnails, deformation fields, etc.) are written to
tests/test_output/ so you can inspect alignment quality after the run.
"""

import os
import urllib.request
import pytest

TESTS_DIR = os.path.dirname(__file__)
DATASETS_DIR = os.path.join(TESTS_DIR, "example_datasets", "cycif")
OUTPUT_DIR = os.path.join(TESTS_DIR, "test_output")

# Two smallest cycif files from upstream
IMAGES = {
    "CD4 CD68 CD3.ome.tiff": "https://raw.githubusercontent.com/MathOnco/valis/main/examples/example_datasets/cycif/CD4%20CD68%20CD3.ome.tiff",
    "CD20 FOXP3 CD3.ome.tiff": "https://raw.githubusercontent.com/MathOnco/valis/main/examples/example_datasets/cycif/CD20%20FOXP3%20CD3.ome.tiff",
}


def _ensure_datasets():
    os.makedirs(DATASETS_DIR, exist_ok=True)
    for name, url in IMAGES.items():
        path = os.path.join(DATASETS_DIR, name)
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
    return [os.path.join(DATASETS_DIR, name) for name in IMAGES]


def test_align_two_images():
    """Register two images and verify alignment completes with low error."""
    from valis import registration

    img_list = _ensure_datasets()
    reference = img_list[
        1
    ]  # CD20 FOXP3 CD3 as reference (matches example script pattern)

    registrar = registration.Valis(
        src_dir=DATASETS_DIR,
        dst_dir=OUTPUT_DIR,
        name="smoketest",
        img_list=img_list,
        reference_img_f=reference,
        align_to_reference=True,
        check_for_reflections=False,
    )
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    assert error_df is not None, "register() returned no error_df"
    assert len(error_df) > 0, "error_df is empty"

    max_error = error_df["mean_non_rigid_D"].max()
    assert (
        max_error < 50
    ), f"Alignment error too high: {max_error:.1f}px (threshold: 50px)"

    # Verify the reference slide has no warping applied
    ref_slide = registrar.get_ref_slide()
    import numpy as np

    dxdy = np.dstack(ref_slide.bk_dxdy)
    assert (
        dxdy.min() == 0 and dxdy.max() == 0
    ), "Reference slide should have zero displacement field"
