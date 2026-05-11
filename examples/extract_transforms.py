"""Extract transformation matrices from a completed registration.

After registration, each ``Slide`` object holds:
  - ``slide.M``          — the 3×3 inverse rigid transformation matrix
  - ``slide.bk_dxdy``    — backward non-rigid displacement field (numpy array)
  - ``slide.fwd_dxdy``   — forward non-rigid displacement field (numpy array)

This example shows how to dump those values so they can be applied in another
tool (e.g. ImageJ/FIJI, QuPath, or a custom pipeline).

Usage
-----
    # Run after basic_registration.py has been executed:
    python examples/extract_transforms.py --data /path/to/results/<name>/data/
"""

import argparse
import glob
import json
import pathlib

import numpy as np

from valis import registration


def find_pickle(data_dir: str) -> str:
    matches = glob.glob(str(pathlib.Path(data_dir) / "*.pickle"))
    if not matches:
        raise FileNotFoundError(f"No pickled registrar found in {data_dir}")
    return matches[0]


def main(data_dir: str, out_dir: str) -> None:
    pickle_f = find_pickle(data_dir)
    registrar = registration.load_registrar(pickle_f)

    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    transforms = {}
    for slide_name, slide in registrar.slide_dict.items():
        entry = {
            "name": slide_name,
            "src_f": slide.src_f,
            "M": slide.M.tolist() if slide.M is not None else None,
            "transformation_src_shape_rc": list(slide.processed_img_shape_rc),
            "transformation_dst_shape_rc": list(slide.reg_img_shape_rc),
        }
        transforms[slide_name] = entry

        # Save displacement fields as numpy .npy files if available
        bk = slide.bk_dxdy
        fwd = slide.fwd_dxdy
        if isinstance(bk, np.ndarray):
            np.save(str(out_path / f"{slide_name}_bk_dxdy.npy"), bk)
        if isinstance(fwd, np.ndarray):
            np.save(str(out_path / f"{slide_name}_fwd_dxdy.npy"), fwd)

    json_f = out_path / "transforms.json"
    with open(json_f, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"Transformation matrices written to: {json_f}")
    print(f"Displacement fields (if any) written to: {out_path}")

    # Print a quick summary table
    print(f"\n{'Slide':<40} {'Has M':>6} {'Has dxdy':>9}")
    print("-" * 58)
    for name, entry in transforms.items():
        has_m = "yes" if entry["M"] is not None else "no"
        bk_f = out_path / f"{name}_bk_dxdy.npy"
        has_dxdy = "yes" if bk_f.exists() else "no"
        print(f"{name:<40} {has_m:>6} {has_dxdy:>9}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export VALIS transformation matrices")
    parser.add_argument(
        "--data", required=True, help="Path to the registrar's data/ directory"
    )
    parser.add_argument(
        "--out", default=None, help="Output directory (default: data/../transforms)"
    )
    args = parser.parse_args()

    out = args.out or str(pathlib.Path(args.data).parent / "transforms")
    main(args.data, out)
