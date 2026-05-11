"""Serialization helpers and shared types for the registration package.

Contains ``load_registrar``, ``DisplacementField``, ``CropMode``, and
``RegistrationConfig``.  Import these from ``valis.registration`` directly;
the sub-module layout is an implementation detail.
"""

import logging
import os
import pathlib
import pickle
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import pyvips

from ._constants import (
    CropMode,
    DEFAULT_MATCHER,
    DEFAULT_MATCHER_FOR_SORTING,
    DEFAULT_TRANSFORM_CLASS,
    DEFAULT_AFFINE_OPTIMIZER_CLASS,
    DEFAULT_SIMILARITY_METRIC,
    DEFAULT_NON_RIGID_CLASS,
    DEFAULT_NON_RIGID_KWARGS,
    DEFAULT_MAX_IMG_DIM,
    DEFAULT_MAX_PROCESSED_IMG_SIZE,
    DEFAULT_MAX_NON_RIGID_REG_SIZE,
    DEFAULT_THUMBNAIL_SIZE,
    DEFAULT_NORM_METHOD,
)
from .. import warp_tools

logger = logging.getLogger(__name__)


def load_registrar(src_f: Union[str, pathlib.Path]) -> "Valis":  # noqa: F821
    """Load a Valis object

    Parameters
    ----------
    src_f : str or Path
        Path to pickled Valis object

    Returns
    -------
    registrar : Valis

        Valis object used for registration

    """
    registrar = pickle.load(open(src_f, "rb"))

    data_dir = registrar.data_dir
    read_data_dir = os.path.split(src_f)[0]

    # If registrar has moved, will need to update paths to results
    # and displacement fields
    if data_dir != read_data_dir:
        new_dst_dir = os.path.split(read_data_dir)[0]
        registrar.dst_dir = new_dst_dir
        registrar.set_dst_paths()

        for slide_obj in registrar.slide_dict.values():
            slide_obj.update_results_img_paths()

    return registrar


class DisplacementField:
    """Owns a single displacement field, transparently backed by memory or disk.

    Encapsulates the three-way storage state that ``Slide`` previously managed
    with ``stored_dxdy``, ``_bk_dxdy_np``, and ``_bk_dxdy_f`` attributes.

    Parameters
    ----------
    array : ndarray, optional
        In-memory numpy displacement array.
    path : str or Path, optional
        Path to a TIFF on disk (used when ``stored_dxdy=True``).
    """

    def __init__(
        self,
        array: Optional[np.ndarray] = None,
        path: Optional[Union[str, pathlib.Path]] = None,
    ):
        self._array: Optional[np.ndarray] = array
        self._path: Optional[pathlib.Path] = pathlib.Path(path) if path else None

    # ------------------------------------------------------------------
    # Properties

    @property
    def is_empty(self) -> bool:
        return self._array is None and self._path is None

    @property
    def is_on_disk(self) -> bool:
        return self._path is not None

    # ------------------------------------------------------------------
    # Data access

    def as_numpy(self) -> Optional[np.ndarray]:
        """Return displacement as a numpy array (lazy-loads from disk if needed)."""
        if self._array is not None:
            return self._array
        if self._path is not None and self._path.exists():
            vips_img = pyvips.Image.new_from_file(str(self._path))
            return warp_tools.vips2numpy(vips_img)
        return None

    def as_vips(self) -> Optional[pyvips.Image]:
        """Return displacement as a pyvips Image (lazy-loads from disk if needed)."""
        if self._path is not None and self._path.exists():
            return pyvips.Image.new_from_file(str(self._path))
        if self._array is not None:
            return warp_tools.numpy2vips(self._array)
        return None

    # ------------------------------------------------------------------
    # Persistence

    def set_array(self, array: np.ndarray) -> None:
        """Store displacement as an in-memory numpy array."""
        if isinstance(array, pyvips.Image):
            raise TypeError(
                "DisplacementField.set_array() expects a numpy array; "
                "use set_path() for pyvips-backed storage."
            )
        self._array = array

    def set_path(self, path: Union[str, pathlib.Path]) -> None:
        """Record the on-disk path (does not read from or write to it)."""
        self._path = pathlib.Path(path)

    def save(self, path: Union[str, pathlib.Path]) -> None:
        """Save the in-memory array to *path* as a TIFF."""
        if self._array is None:
            raise ValueError("No in-memory array to save.")
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        vips_img = warp_tools.numpy2vips(self._array)
        vips_img.tiffsave(str(path))
        self._path = path

    def load(self, path: Union[str, pathlib.Path]) -> None:
        """Load the displacement from *path* into memory."""
        self._path = pathlib.Path(path)
        self._array = None  # force lazy reload on next as_numpy() call


@dataclass
class RegistrationConfig:
    """Groups the registration-flow parameters for ``Valis``.

    Passing a ``RegistrationConfig`` to ``Valis.__init__`` via the ``config``
    keyword is the preferred way to configure the pipeline for new code.
    All individual keyword arguments on ``Valis.__init__`` continue to work
    unchanged for backward compatibility; if both are supplied the explicit
    keyword argument takes precedence over the value in ``config``.

    Parameters
    ----------
    feature_detector_cls :
        FeatureDD class for feature detection.  ``None`` means use whatever
        is set on ``matcher``.
    matcher :
        Matcher used after image ordering.
    matcher_for_sorting :
        Matcher used to build the image-dissimilarity matrix.
    transformer_cls :
        Scikit-image rigid transform class.
    affine_optimizer_cls :
        AffineOptimizer class, or ``None`` to skip optimisation.
    similarity_metric :
        Metric for building the dissimilarity matrix (``"n_matches"`` or a
        scipy distance string).
    non_rigid_registrar_cls :
        NonRigidRegistrar instance, or ``None`` to skip non-rigid registration.
    non_rigid_reg_params :
        Keyword arguments forwarded to ``non_rigid_registrar_cls``.
    compose_non_rigid :
        Whether to compose non-rigid deformation fields serially.
    micro_rigid_registrar_cls :
        MicroRigidRegistrar class for high-resolution refinement, or ``None``.
    micro_rigid_registrar_params :
        Keyword arguments forwarded to ``micro_rigid_registrar_cls``.
    imgs_ordered :
        ``True`` if images are already in the correct order.
    do_rigid :
        ``True`` to perform rigid registration.
    denoise_rigid :
        ``True`` to denoise processed images before rigid registration.
    crop_for_rigid_reg :
        ``True`` to zoom into the tissue mask before rigid registration.
    check_for_reflections :
        ``True`` to test whether flipped/mirrored images align better.
    max_image_dim_px :
        Maximum width/height of images loaded from pyramid levels.
    max_processed_image_dim_px :
        Maximum width/height of processed images used for feature detection.
    max_non_rigid_registration_dim_px :
        Maximum width/height used for non-rigid registration.
    crop :
        Default crop mode for warped output.  Accepts a ``CropMode`` value or
        the equivalent string (``"overlap"``, ``"reference"``, ``"all"``).
    create_masks :
        ``True`` to create and apply tissue masks during registration.
    thumbnail_size :
        Maximum width/height of the result thumbnails written to ``dst_dir``.
    norm_method :
        Normalisation strategy: ``"img_stats"``, ``"histo_match"``, or ``None``.

    Examples
    --------
    Use the defaults (equivalent to calling ``Valis`` with no configuration
    kwargs):

    >>> from valis.registration import Valis, RegistrationConfig
    >>> registrar = Valis(src_dir, dst_dir, config=RegistrationConfig())

    IHC-optimised preset — larger processing window, no non-rigid step:

    >>> cfg = RegistrationConfig.for_ihc()
    >>> registrar = Valis(src_dir, dst_dir, config=cfg)
    """

    feature_detector_cls: object = None
    matcher: object = field(default_factory=lambda: DEFAULT_MATCHER)
    matcher_for_sorting: object = field(
        default_factory=lambda: DEFAULT_MATCHER_FOR_SORTING
    )
    transformer_cls: type = DEFAULT_TRANSFORM_CLASS
    affine_optimizer_cls: Optional[type] = DEFAULT_AFFINE_OPTIMIZER_CLASS
    similarity_metric: str = DEFAULT_SIMILARITY_METRIC
    non_rigid_registrar_cls: object = field(
        default_factory=lambda: DEFAULT_NON_RIGID_CLASS
    )
    non_rigid_reg_params: dict = field(default_factory=dict)
    compose_non_rigid: bool = False
    micro_rigid_registrar_cls: Optional[type] = None
    micro_rigid_registrar_params: dict = field(default_factory=dict)
    imgs_ordered: bool = False
    do_rigid: bool = True
    denoise_rigid: bool = False
    crop_for_rigid_reg: bool = True
    check_for_reflections: bool = False
    max_image_dim_px: int = DEFAULT_MAX_IMG_DIM
    max_processed_image_dim_px: int = DEFAULT_MAX_PROCESSED_IMG_SIZE
    max_non_rigid_registration_dim_px: int = DEFAULT_MAX_NON_RIGID_REG_SIZE
    crop: Optional[Union[str, "CropMode"]] = None
    create_masks: bool = True
    thumbnail_size: int = DEFAULT_THUMBNAIL_SIZE
    norm_method: Optional[str] = DEFAULT_NORM_METHOD

    @classmethod
    def for_ihc(cls) -> "RegistrationConfig":
        """Preset for brightfield IHC/H&E image series.

        Uses a larger processing window for better feature coverage and skips
        non-rigid registration, which is often unnecessary for well-prepared
        brightfield sections.
        """
        return cls(
            max_processed_image_dim_px=1024,
            non_rigid_registrar_cls=None,
            crop=CropMode.REFERENCE,
        )

    @classmethod
    def for_cycif(cls) -> "RegistrationConfig":
        """Preset for cyclic immunofluorescence (CyCIF) image series.

        Enables non-rigid registration and uses overlap cropping to handle
        the field-of-view shifts common in multi-round imaging.
        """
        return cls(
            non_rigid_registrar_cls=DEFAULT_NON_RIGID_CLASS,
            crop=CropMode.OVERLAP,
        )
