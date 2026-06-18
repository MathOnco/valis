"""
Cheap pre-alignment check: which of the 8 D4 transforms (4 rotations x
optional mirror) of a moving image best matches a reference image.

Run this before full registration so the expensive alignment only has to
handle small residual translation/rotation, not a 90 degree rotation or a
flipped scan.
"""

import logging
from dataclasses import dataclass

import numpy as np
import cv2

logger = logging.getLogger(__name__)


D4_TRANSFORMS = (
    ("identity", 0, False),
    ("rot90", 1, False),
    ("rot180", 2, False),
    ("rot270", 3, False),
    ("flip", 0, True),
    ("flip_rot90", 1, True),
    ("flip_rot180", 2, True),
    ("flip_rot270", 3, True),
)


@dataclass
class OrientationMatch:
    name: str
    k: int  # number of CCW 90-degree rotations
    mirror: bool  # horizontal flip applied before rotation
    score: float  # normalized cross-correlation in [-1, 1]
    scores: dict  # name -> score for all 8 transforms


def apply_d4(img: np.ndarray, k: int, mirror: bool) -> np.ndarray:
    if mirror:
        img = np.fliplr(img)
    if k:
        img = np.rot90(img, k=k)
    return img


def apply_d4_pyvips(img, k: int, mirror: bool):
    """Apply the same D4 transform as :func:`apply_d4` to a pyvips Image.

    numpy's ``rot90(k=1)`` is counter-clockwise; pyvips' ``rot90`` is
    clockwise. The mapping below matches the numpy convention so a
    transform discovered on thumbnails applies identically to the
    full-resolution pyvips image.
    """
    if mirror:
        img = img.fliphor()
    k = k % 4
    if k == 1:
        img = img.rot270()  # CCW 90
    elif k == 2:
        img = img.rot180()
    elif k == 3:
        img = img.rot90()  # CCW 270 == CW 90
    return img


def describe(match: "OrientationMatch") -> str:
    """Human-readable description of what would be applied."""
    if match.k == 0 and not match.mirror:
        return "identity (no rotation or flip needed)"
    parts = []
    if match.mirror:
        parts.append("horizontal flip")
    if match.k:
        parts.append(f"{match.k * 90}deg CCW rotation")
    return " + ".join(parts)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def _thumbnail(img: np.ndarray, size: int, use_gradient: bool) -> np.ndarray:
    """Resize to a (size, size) float32 thumbnail.

    A square thumbnail is required so every D4 transform of the moving image
    has the same shape as the reference, even when the originals have
    different aspect ratios. The aspect distortion is identical for both
    images, so it cancels out in the correlation.
    """
    g = _to_gray(img)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    g = g.astype(np.float32)
    if use_gradient:
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        g = cv2.magnitude(gx, gy)
    g -= g.mean()
    n = np.linalg.norm(g)
    if n > 0:
        g /= n
    return g


def find_best_orientation(
    reference: np.ndarray,
    moving: np.ndarray,
    downsample_size: int = 128,
    use_gradient: bool = True,
) -> OrientationMatch:
    """Pick the D4 transform of ``moving`` that best matches ``reference``.

    Parameters
    ----------
    reference, moving : np.ndarray
        2D grayscale or HxWx3 RGB arrays. Dimensions need not match; both
        are resized to a common square thumbnail.
    downsample_size : int
        Side length of the square thumbnail used for scoring. 64-256 is the
        useful range; larger is more discriminating but slower (cost is
        O(downsample_size^2) per transform, x8 transforms).
    use_gradient : bool
        Score on Sobel gradient magnitude instead of raw intensity. More
        robust to brightness/stain differences between modalities; turn off
        if the two images are known to be intensity-comparable.
    """
    if downsample_size < 8:
        raise ValueError("downsample_size must be >= 8")

    # Don't upsample: cap at the smaller side of the smaller input. Going
    # beyond that fabricates pixels and can't add real signal.
    max_useful = min(
        reference.shape[0], reference.shape[1], moving.shape[0], moving.shape[1]
    )
    effective = min(downsample_size, max_useful)
    if effective != downsample_size:
        logger.info(
            "orientation check: clamping downsample_size %d -> %d "
            "(max useful given ref %dx%d and moving %dx%d)",
            downsample_size,
            effective,
            reference.shape[1],
            reference.shape[0],
            moving.shape[1],
            moving.shape[0],
        )
    downsample_size = effective

    ref_thumb = _thumbnail(reference, downsample_size, use_gradient)

    # Build the moving thumbnail once, then apply D4 to the thumbnail. Since
    # D4 commutes with the (square) resize, this matches "transform then
    # downsample" while doing the expensive resize a single time.
    mov_thumb = _thumbnail(moving, downsample_size, use_gradient)

    scores: dict[str, float] = {}
    best = None
    for name, k, mirror in D4_TRANSFORMS:
        candidate = apply_d4(mov_thumb, k, mirror)
        # Both arrays are zero-mean and unit-norm, so dot product == NCC.
        score = float(np.tensordot(ref_thumb, candidate, axes=2))
        scores[name] = score
        if best is None or score > best[3]:
            best = (name, k, mirror, score)

    name, k, mirror, score = best
    logger.info(
        "orientation check: best=%s score=%.4f (size=%d, gradient=%s)",
        name,
        score,
        downsample_size,
        use_gradient,
    )
    return OrientationMatch(name=name, k=k, mirror=mirror, score=score, scores=scores)
