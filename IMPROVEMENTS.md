# VALIS Code Improvement Proposals

This document captures code quality, usability, and architectural issues found in the codebase, along with concrete suggestions for addressing them. Issues are grouped by theme and roughly ordered by impact.

---

## 1. God Class: `registration.py`

`registration.py` is 6,278 lines and contains two deeply coupled classes (`Valis` and `Slide`) that each do far too much. `Valis` alone has 52 methods and ~80+ instance attributes.

**Problems:**
- Impossible to understand the registration pipeline at a glance
- Tightly coupled I/O, preprocessing, rigid registration, non-rigid registration, cropping, and saving
- Cannot test any sub-step without running the whole pipeline
- Any change risks breaking unrelated behavior

**Suggestion:** Split into focused components along the natural pipeline stages:

```
registration/
├── pipeline.py          # Valis orchestrator (thin, delegates to others)
├── slide.py             # Slide metadata + transformation state
├── crop.py              # All cropping logic extracted here
├── warp.py              # Warp/save orchestration
└── state.py             # Serialization / resume-from-disk logic
```

Each component can be tested and understood independently. The `Valis` class becomes a thin orchestrator that composes them.

---

## 2. Missing Type Hints

600+ functions have no type annotations, including the entire public API. This makes the package hard to use without reading source code.

**Before:**
```python
def warp_img(self, img, transformation_src_shape_rc=None,
             transformation_dst_shape_rc=None, interp_method="bicubic",
             bbox_xywh=None, out_shape_rc=None, ...):
```

**After:**
```python
def warp_img(
    self,
    img: np.ndarray,
    transformation_src_shape_rc: tuple[int, int] | None = None,
    transformation_dst_shape_rc: tuple[int, int] | None = None,
    interp_method: str = "bicubic",
    bbox_xywh: tuple[int, int, int, int] | None = None,
    out_shape_rc: tuple[int, int] | None = None,
) -> np.ndarray:
```

At minimum, annotate all public API methods. Adding `py.typed` to the package and running `mypy` in CI would catch regressions. Start with `registration.py` since that's what users touch directly.

---

## 3. 27-Parameter Constructor

`Valis.__init__` accepts 27 parameters, many of which are interdependent. There's no validation that incompatible options weren't combined.

**Suggestion:** Use a configuration dataclass to group related options:

```python
@dataclass
class RegistrationConfig:
    feature_detector: FeatureDetectorBase = field(default_factory=OrbFD)
    feature_matcher: MatcherBase = field(default_factory=VggMatcher)
    non_rigid_registrar_cls: type[NonRigidRegistrar] = OpticalFlowWarper
    micro_rigid: bool = False
    max_image_dim_px: int = 850
    max_processed_image_dim_px: int = 2000
    crop: Literal["overlap", "reference", "all"] = "overlap"
    compose_non_rigid: bool = False

registrar = Valis(src_dir, dst_dir, config=RegistrationConfig())
```

This groups related settings, enables documentation per-config, and makes it easy to create pre-built configs for common cases (e.g., `RegistrationConfig.for_ihc()`, `RegistrationConfig.for_cycif()`).

---

## 4. String Constants — Use Enums

Crop mode and other string constants are used throughout the codebase with no validation. A typo silently does the wrong thing.

**Before:**
```python
registrar.warp_and_save_slides(crop="overlap")  # or "reference", or "all"?
```

**After:**
```python
from enum import StrEnum

class CropMode(StrEnum):
    OVERLAP = "overlap"
    REFERENCE = "reference"
    NONE = "all"

registrar.warp_and_save_slides(crop=CropMode.OVERLAP)
```

`StrEnum` (Python 3.11+) or `str, Enum` base class maintains backward compatibility with code passing string literals.

---

## 5. Inconsistent Array/Coordinate Conventions

The codebase mixes three coordinate conventions without a clear boundary:

- `rc` — (row, col), i.e. (height, width), numpy convention
- `wh` — (width, height), image dimension convention
- `xy` — (x, y), display/point convention

The conversion `shape_rc[::-1]` appears 20+ times. Off-by-one errors between conventions are the most common source of subtle bugs.

**Suggestion:** Define thin wrapper types or at minimum a module-level conversion function, and document which convention each function uses in its signature name or docstring.

```python
# Explicit conversion utilities, not ad-hoc slicing
def rc_to_wh(shape_rc: tuple[int, int]) -> tuple[int, int]:
    return shape_rc[1], shape_rc[0]

def wh_to_rc(wh: tuple[int, int]) -> tuple[int, int]:
    return wh[1], wh[0]
```

Longer term, consider `NamedTuple` or dataclass types (`ShapeRC`, `SizeWH`) to make convention violations a type error.

---

## 6. Displacement Field State Management

`Slide` uses a convoluted three-way state for displacement fields: in-memory numpy array, pyvips image, or lazy-loaded from disk — controlled by a `stored_dxdy` flag and private attributes `_bk_dxdy_f`, `_bk_dxdy_np`. The property getter conditionally loads from any of these sources.

This pattern:
- Is easy to corrupt (can be in inconsistent state)
- Makes memory usage unpredictable
- Requires readers to understand all three paths

**Suggestion:** Introduce a `DisplacementField` class that owns the backing storage decision:

```python
class DisplacementField:
    """Owns a displacement field, transparently backed by memory or disk."""
    def __init__(self, array: np.ndarray | None = None, path: Path | None = None): ...

    def as_numpy(self) -> np.ndarray: ...
    def as_vips(self) -> pyvips.Image: ...
    def save(self, path: Path) -> None: ...
    def load(self, path: Path) -> None: ...
```

`Slide` then holds a single `DisplacementField` instance and delegates all storage decisions to it.

---

## 7. Silent Exception Handling

Throughout the code (especially `feature_detectors.py`, `feature_matcher.py`), broad `except Exception` blocks log a warning and continue. This hides failures completely.

**Before:**
```python
try:
    keypoints, descriptors = detector.detect_and_compute(img)
except Exception as e:
    logger.warning(e)
    keypoints, descriptors = [], None
```

**After:**
```python
try:
    keypoints, descriptors = detector.detect_and_compute(img)
except DetectionError as e:
    logger.warning("Feature detection failed for %s: %s", self.name, e)
    raise  # or return a typed failure result
```

At minimum, narrow the exception types. For features that can legitimately produce no matches, use a typed result rather than None-or-empty:

```python
@dataclass
class DetectionResult:
    keypoints: list
    descriptors: np.ndarray | None
    failed: bool = False
    reason: str = ""
```

---

## 8. No Unit Tests

There are exactly 4 test functions, all of which are integration tests requiring external datasets that are not in the repository. There are no unit tests, no mocks, no fixtures, and no edge case coverage.

**Highest-value areas to add tests:**

| Function/Class | What to test |
|---|---|
| `warp_tools` coordinate transforms | Roundtrip accuracy, edge cases |
| Crop logic (`get_crop_xywh`, etc.) | Off-by-one, overlap calculation |
| `slide_io` format detection | Common extensions, unknown formats |
| Feature matcher filtering | RANSAC with degenerate inputs |
| `Valis.register()` with 1 image | Should error cleanly |
| Array convention conversions | rc/wh/xy roundtrips |

A small synthetic test image (black with known features) can drive most unit tests without requiring real WSI data. Add `pytest-cov` to CI to track coverage.

---

## 9. Segfault on Import Order

The README contains this warning:

> "Python will segfault if this project is not imported first before any other pytorch-related import"

This is a critical usability issue. It should be addressed at the package level rather than documented as a warning. Options:

1. **Lazy-import torch** inside the functions/classes that need it, rather than at module level
2. **Use `importlib.import_module`** with explicit ordering if early binding is required
3. **Isolate SuperGlue/SuperPoint** into a subpackage with its own import guard and document that subpackage as optional
4. **Add a guard in `__init__.py`** that detects conflicting imports and raises a clear error with a fix message

Leaving a segfault as a known issue makes the package unsuitable for use inside larger applications.

---

## 10. Duplicate / Overlapping Crop Methods

There are 5+ crop-related methods split across `Slide` and `Valis` with overlapping responsibilities:

- `Slide.get_crop_xywh()`
- `Slide.get_overlap_crop_xywh()`
- `Slide.get_aligned_to_ref_slide_crop_xywh()`
- `Valis.get_crop_xywh()`
- `Valis.get_overlap_indices()`

Each handles edge cases slightly differently. It's not obvious which to call for a given task.

**Suggestion:** Consolidate into a single `CropCalculator` or make `Valis` the single entry point for crop logic and deprecate the `Slide`-level methods. Document in the class docstring which cropping scenario each method covers.

---

## 11. Warp Method Proliferation

There are 7+ warp methods with subtly different semantics:

- `warp_img()`
- `warp_img_from_to()`
- `warp_xy()`
- `warp_points()`
- `warp_geojson()`
- `warp_annotations()`
- `warp_and_save_slides()`

For new users, it's not clear which to use. For existing users, each has slightly different cropping and interpolation behavior.

**Suggestion:** Document a decision tree in the class-level docstring:

```
Use warp_img()         — to warp a numpy array
Use warp_img_from_to() — to warp between two specific slides (not just to reference)
Use warp_xy()          — to transform point coordinates
Use warp_geojson()     — to transform GeoJSON annotations
Use warp_and_save_slides() — to warp all registered slides and write to disk
```

Also consider a unified `warp(target)` method that dispatches based on type.

---

## 12. Example Coverage

There is exactly one example (`align_two_images.py`), which is a 378-line specialized CLI tool. It doesn't serve as an introduction to the library.

**Missing examples:**

- **Minimal working example** — register a directory of images, save as OME-TIFF, 20 lines
- **Multi-round CyCIF** — register across fluorescence rounds
- **Non-rigid registration** — when to use it, how to configure it
- **Resume from saved state** — how to re-use a registration without recomputing
- **Extract transformation matrices** — for users who want to apply transforms in another tool
- **Error handling** — what to do when `register()` returns failures

---

## 13. No Validation on Construction

`Valis.__init__` accepts `src_dir` and `dst_dir` but doesn't validate them until `register()` is called. If `src_dir` doesn't exist or contains no supported images, the error surfaces far later than it should.

**Suggestion:** Validate at construction time and fail fast:

```python
def __init__(self, src_dir: str | Path, dst_dir: str | Path, ...):
    src_dir = Path(src_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"src_dir does not exist: {src_dir}")
    images = self._find_images(src_dir)
    if len(images) == 0:
        raise ValueError(f"No supported images found in {src_dir}")
    if len(images) < 2:
        raise ValueError(f"At least 2 images required for registration, found {len(images)}")
```

---

## 14. Package Size and Optional Dependencies

The package requires torch 2.7+ (~2GB) as a hard dependency, even though torch is only needed for SuperGlue/SuperPoint-based feature detection. Users who want to use ORB or SIFT don't need torch at all.

**Suggestion:** Make deep learning dependencies optional:

```toml
[project.optional-dependencies]
dl = ["torch>=2.7.1", "torchvision", "kornia", "einops"]
full = ["valis[dl]", ...]
```

```python
# In feature_detectors.py
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

class SuperPointFD(FeatureDetectorBase):
    def __init__(self):
        if not _TORCH_AVAILABLE:
            raise ImportError("SuperPoint requires torch. Install with: pip install valis[dl]")
```

---

## Summary Table

| # | Issue | Severity | Effort | Status |
|---|-------|----------|--------|--------|
| 1 | God class `registration.py` | Critical | High | **done** — split into package |
| 2 | No type hints | High | Medium | **partial** — public API annotated |
| 3 | 27-parameter constructor | High | Medium | **done** — `RegistrationConfig` dataclass |
| 4 | String constants, no enums | Medium | Low | **done** |
| 5 | Mixed coordinate conventions | High | Medium | **partial** — `rc_to_wh`/`wh_to_rc` added |
| 6 | Displacement field state | High | Medium | **done** — `DisplacementField` class added |
| 7 | Silent exception handling | Medium | Low | **done** — bare `except:` → `except Exception:` |
| 8 | No unit tests | High | High | **partial** — 26 unit tests |
| 9 | Import-order segfault | High | Medium | **done** |
| 10 | Duplicate crop methods | Medium | Low | **partial** — decision tree in Valis docstring |
| 11 | Warp method proliferation | Medium | Low | **done** — decision tree in Valis docstring |
| 12 | Example coverage | Medium | Low | **done** — 4 example scripts added |
| 13 | No construction validation | Medium | Low | **done** |
| 14 | Optional deep learning deps | Medium | Medium | **done** — torch/kornia moved to `[dl]` extras |

## Completed work

### Issue 4 — CropMode enum (`registration.py`)
Added `CropMode(StrEnum)` with members `OVERLAP`, `REFERENCE`, and `NONE`. The old
`CROP_OVERLAP`/`CROP_REF`/`CROP_NONE` module constants now point to enum members, so
all existing code is backward compatible. String literals (e.g. `"overlap"`) continue
to work wherever a `CropMode` is accepted.

### Issue 13 — Construction validation (`registration.py`)
`Valis.__init__` now validates `src_dir` eagerly (when `img_list` is `None`) and raises:
- `FileNotFoundError` if the directory does not exist
- `NotADirectoryError` if the path is a file
- `ValueError` if no supported images are found
- `ValueError` if fewer than 2 images are found

### Issue 9 — Import-order segfault guard (`__init__.py`)
`valis/__init__.py` inspects `sys.modules` on import. If any of `torch`, `torchvision`,
`kornia`, `einops`, or `timm` are already loaded, it raises a descriptive `ImportError`
with a clear fix message instead of silently segfaulting.

### Issues 2 & 5 — Type hints and coordinate utilities
- Added `from __future__ import annotations` and `Optional`/`Union` imports to
  `registration.py`.
- Annotated the primary public API: `load_registrar()`, `Slide.warp_img()`,
  `Slide.warp_xy()`, `Slide.warp_geojson()`, `Valis.__init__()`, `Valis.register()`,
  `Valis.warp_and_save_slides()`.
- Added `rc_to_wh(shape_rc)` and `wh_to_rc(wh)` conversion utilities to
  `warp_tools.py` to replace ad-hoc `[::-1]` slicing.

### Issue 8 — Unit tests (`tests/test_unit.py`)
Now 26 unit tests (no external datasets required) covering:
- `rc_to_wh` / `wh_to_rc` roundtrips and edge cases
- `CropMode` enum values and backward-compat constants
- `Valis` construction validation (bad path, empty dir, single image)
- `get_alignment_indices` edge cases
- `RegistrationConfig` defaults, presets, application to `Valis`, kwarg override
- `DisplacementField` empty state, array storage, disk path, pyvips rejection

### Issues 7, 10, 11 — Exception handling and docstrings
- `feature_detectors.py`: bare `except:` narrowed to `except Exception:`
- Added a "Choosing a warp method" and "Choosing a crop mode" decision table
  to the `Valis` class docstring.

### Issue 3 — RegistrationConfig dataclass (`registration.py`)
A `RegistrationConfig` dataclass groups all registration-flow parameters with
typed, documented fields and factory defaults. Pass it via `Valis(..., config=cfg)`.
Explicit keyword arguments always win over config values (backward compatible).
Includes two convenience presets:
- `RegistrationConfig.for_ihc()` — larger window, rigid-only, reference crop
- `RegistrationConfig.for_cycif()` — non-rigid enabled, overlap crop

### Issue 6 — DisplacementField class (`registration.py`)
`DisplacementField` encapsulates the three-way storage state (in-memory numpy,
pyvips-backed, or lazy disk load) that `Slide` previously managed with
`stored_dxdy`, `_bk_dxdy_np`, and `_bk_dxdy_f`.  Public interface:
`as_numpy()`, `as_vips()`, `set_array()`, `set_path()`, `save()`, `load()`.

### Issue 12 — Example scripts (`examples/`)
Four new runnable examples (no external datasets required to read):
- `basic_registration.py` — minimal 20-line pipeline
- `non_rigid_registration.py` — when and how to use non-rigid registration
- `resume_from_saved_state.py` — reload a pickled registrar; warp slides or points
- `extract_transforms.py` — dump transformation matrices to JSON + numpy files

### Issue 14 — Optional deep-learning dependencies (`pyproject.toml`)
`torch`, `torchvision`, `kornia`, and `einops` moved from `[project.dependencies]`
to `[project.optional-dependencies].dl`.  Users who only need OpenCV-based
detectors can install with `pip install valis-wsi` (~2 GB lighter).
`pip install 'valis-wsi[dl]'` restores the full feature set.

Classes that require torch (`SuperPointFD`, `KorniaFD`, `DiskFD`, `DeDoDeFD`,
`SuperGlueMatcher`, `LightGlueMatcher`) now raise a descriptive `ImportError`
with install instructions when torch is not available.  `DEFAULT_MATCHER` in
`registration.py` falls back to `VggFD`-based matching when torch is absent.

---

### Issue 1 — God-class split (`registration/` package)
`registration.py` (6,649 lines, two deeply coupled classes) converted to a
proper Python package at `src/valis/registration/`:

```
registration/
├── __init__.py      Re-exports every public name — zero breaking changes
├── _constants.py    Module-level constants, defaults, shared imports (~180 lines)
├── state.py         CropMode, DisplacementField, RegistrationConfig, load_registrar
├── slide.py         Slide class (~1,500 lines)
└── pipeline.py      Valis class (~4,800 lines)
```

- `from valis.registration import Valis, Slide, load_registrar` continues to work.
- `slide.py` uses `TYPE_CHECKING` to reference `Valis` without a circular import.
- All 26 unit tests pass unchanged.

A reasonable starting point that would have the most immediate usability impact without requiring a full rewrite:

1. ~~Add type hints to all public API methods (issues 2, 5)~~ **done (partial)**
2. ~~Replace string constants with enums (issue 4)~~ **done**
3. ~~Add construction validation (issue 13)~~ **done**
4. ~~Fix or guard the import-order segfault (issue 9)~~ **done**
5. ~~Add unit tests for coordinate transforms and crop logic (issue 8)~~ **done (partial)**
