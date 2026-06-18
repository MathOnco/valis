"""Unit tests for valis utilities.

These tests do not require external datasets and run purely on synthetic data.
"""

import importlib
import pathlib
import pkgutil
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import smoke test — guards against missing-import / NameError-at-runtime bugs
# in module-level code by walking every submodule under ``valis``.
# ---------------------------------------------------------------------------


def _iter_valis_modules():
    import valis

    for info in pkgutil.walk_packages(valis.__path__, prefix="valis."):
        # superglue_models pulls in heavy DL deps that are an optional install.
        if info.name.startswith("valis.superglue_models"):
            continue
        yield info.name


@pytest.mark.parametrize("module_name", list(_iter_valis_modules()))
def test_module_imports(module_name):
    importlib.import_module(module_name)


# ---------------------------------------------------------------------------
# Coordinate-convention utilities (warp_tools.rc_to_wh / wh_to_rc)
# ---------------------------------------------------------------------------


class TestCoordConversions:
    def test_rc_to_wh_basic(self):
        from valis.warp_tools import rc_to_wh

        assert rc_to_wh((100, 200)) == (200, 100)

    def test_wh_to_rc_basic(self):
        from valis.warp_tools import wh_to_rc

        assert wh_to_rc((200, 100)) == (100, 200)

    def test_roundtrip_rc_wh(self):
        from valis.warp_tools import rc_to_wh, wh_to_rc

        shape = (480, 640)
        assert wh_to_rc(rc_to_wh(shape)) == shape

    def test_roundtrip_wh_rc(self):
        from valis.warp_tools import rc_to_wh, wh_to_rc

        size = (1920, 1080)
        assert rc_to_wh(wh_to_rc(size)) == size

    def test_square_image(self):
        from valis.warp_tools import rc_to_wh, wh_to_rc

        assert rc_to_wh((256, 256)) == (256, 256)
        assert wh_to_rc((256, 256)) == (256, 256)


# ---------------------------------------------------------------------------
# CropMode enum
# ---------------------------------------------------------------------------


class TestCropMode:
    def test_values(self):
        from valis.registration import CropMode

        assert CropMode.OVERLAP == "overlap"
        assert CropMode.REFERENCE == "reference"
        assert CropMode.NONE == "all"

    def test_string_equality(self):
        from valis.registration import CropMode

        # StrEnum: enum members compare equal to their string value
        assert CropMode.OVERLAP == "overlap"
        assert "overlap" == CropMode.OVERLAP

    def test_backward_compat_constants(self):
        from valis.registration import CROP_OVERLAP, CROP_REF, CROP_NONE, CropMode

        assert CROP_OVERLAP == CropMode.OVERLAP
        assert CROP_REF == CropMode.REFERENCE
        assert CROP_NONE == CropMode.NONE

    def test_all_members(self):
        from valis.registration import CropMode

        members = {m.value for m in CropMode}
        assert members == {"overlap", "reference", "all"}


# ---------------------------------------------------------------------------
# Valis construction validation (Issue 13)
# ---------------------------------------------------------------------------


class TestValisConstructionValidation:
    def test_nonexistent_src_dir(self):
        from valis.registration import Valis

        with pytest.raises(FileNotFoundError):
            Valis("/this/path/does/not/exist", "/tmp/dst")

    def test_src_dir_is_file(self, tmp_path):
        from valis.registration import Valis

        f = tmp_path / "not_a_dir.txt"
        f.write_text("hello")
        with pytest.raises(NotADirectoryError):
            Valis(str(f), str(tmp_path / "dst"))

    def test_empty_src_dir(self, tmp_path):
        from valis.registration import Valis

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No supported images"):
            Valis(str(empty_dir), str(tmp_path / "dst"))

    def test_single_image_src_dir(self, tmp_path):
        """Need at least 2 images for registration."""
        import tifffile
        from valis.registration import Valis

        src = tmp_path / "src"
        src.mkdir()
        # Write a minimal single-channel TIFF
        img = np.zeros((32, 32), dtype=np.uint8)
        tifffile.imwrite(str(src / "img1.tif"), img)

        with pytest.raises(ValueError, match="At least 2 images"):
            Valis(str(src), str(tmp_path / "dst"))


# ---------------------------------------------------------------------------
# warp_tools.get_alignment_indices — edge cases
# ---------------------------------------------------------------------------


class TestGetAlignmentIndices:
    def test_two_images(self):
        from valis.warp_tools import get_alignment_indices

        indices = get_alignment_indices(2, ref_img_idx=0)
        assert len(indices) == 1

    def test_five_images_length(self):
        from valis.warp_tools import get_alignment_indices

        indices = get_alignment_indices(5, ref_img_idx=2)
        assert len(indices) == 4

    def test_reference_not_in_indices(self):
        from valis.warp_tools import get_alignment_indices

        ref = 2
        indices = get_alignment_indices(5, ref_img_idx=ref)
        for from_idx, to_idx in indices:
            assert from_idx != ref

    def test_all_images_covered(self):
        from valis.warp_tools import get_alignment_indices

        n = 5
        indices = get_alignment_indices(n, ref_img_idx=2)
        from_idxs = {i for i, _ in indices}
        # Every non-reference image must appear exactly once as "from"
        assert from_idxs == {0, 1, 3, 4}


# ---------------------------------------------------------------------------
# RegistrationConfig dataclass (Issue 3)
# ---------------------------------------------------------------------------


class TestRegistrationConfig:
    def test_defaults_match_valis_defaults(self):
        from valis.registration import RegistrationConfig, DEFAULT_SIMILARITY_METRIC
        from valis.registration import (
            DEFAULT_MAX_IMG_DIM,
            DEFAULT_MAX_PROCESSED_IMG_SIZE,
        )

        cfg = RegistrationConfig()
        assert cfg.similarity_metric == DEFAULT_SIMILARITY_METRIC
        assert cfg.max_image_dim_px == DEFAULT_MAX_IMG_DIM
        assert cfg.max_processed_image_dim_px == DEFAULT_MAX_PROCESSED_IMG_SIZE

    def test_for_ihc_preset(self):
        from valis.registration import RegistrationConfig, CropMode

        cfg = RegistrationConfig.for_ihc()
        assert cfg.non_rigid_registrar_cls is None
        assert cfg.crop == CropMode.REFERENCE
        assert cfg.max_processed_image_dim_px == 1024

    def test_for_cycif_preset(self):
        from valis.registration import RegistrationConfig, CropMode

        cfg = RegistrationConfig.for_cycif()
        assert cfg.non_rigid_registrar_cls is not None
        assert cfg.crop == CropMode.OVERLAP

    def test_config_applied_to_valis(self, tmp_path):
        """RegistrationConfig values should appear on the Valis object."""
        import tifffile
        from valis.registration import Valis, RegistrationConfig, CropMode

        src = tmp_path / "src"
        src.mkdir()
        for i in range(2):
            tifffile.imwrite(
                str(src / f"img{i}.tif"), np.zeros((32, 32), dtype=np.uint8)
            )

        cfg = RegistrationConfig(
            max_processed_image_dim_px=256,
            crop=CropMode.NONE,
        )
        registrar = Valis(str(src), str(tmp_path / "dst"), config=cfg)
        assert registrar.max_processed_image_dim_px == 256
        assert registrar.crop == CropMode.NONE

    def test_explicit_kwarg_overrides_config(self, tmp_path):
        """Explicit keyword argument must take precedence over config value."""
        import tifffile
        from valis.registration import Valis, RegistrationConfig, CropMode

        src = tmp_path / "src"
        src.mkdir()
        for i in range(2):
            tifffile.imwrite(
                str(src / f"img{i}.tif"), np.zeros((32, 32), dtype=np.uint8)
            )

        cfg = RegistrationConfig(crop=CropMode.OVERLAP)
        registrar = Valis(str(src), str(tmp_path / "dst"), config=cfg, crop="reference")
        assert registrar.crop == "reference"


# ---------------------------------------------------------------------------
# DisplacementField (Issue 6)
# ---------------------------------------------------------------------------


class TestDisplacementField:
    def test_empty_by_default(self):
        from valis.registration import DisplacementField

        df = DisplacementField()
        assert df.is_empty
        assert not df.is_on_disk

    def test_set_and_retrieve_array(self):
        from valis.registration import DisplacementField

        arr = np.zeros((2, 4, 4), dtype=np.float32)
        df = DisplacementField(array=arr)
        assert not df.is_empty
        result = df.as_numpy()
        assert np.array_equal(result, arr)

    def test_is_on_disk_after_set_path(self, tmp_path):
        from valis.registration import DisplacementField

        p = tmp_path / "dxdy.tiff"
        df = DisplacementField()
        df.set_path(p)
        assert df.is_on_disk

    def test_set_array_rejects_pyvips(self):
        import pyvips
        from valis.registration import DisplacementField

        vips_img = pyvips.Image.black(4, 4)
        df = DisplacementField()
        with pytest.raises(TypeError):
            df.set_array(vips_img)
