"""Resume registration from a saved Valis object.

Once ``registrar.register()`` has been run once, VALIS pickles the registrar to
``<dst_dir>/<name>/data/``.  This example shows how to reload it and warp slides
(or point data) without rerunning the full registration.

Usage
-----
    # First run (creates the pickle):
    python examples/basic_registration.py --src /path/to/slides --dst /path/to/results

    # Resume (warp slides only):
    python examples/resume_from_saved_state.py --data /path/to/results/<name>/data/

    # Or warp a set of (x, y) points from a CSV:
    python examples/resume_from_saved_state.py --data /path/to/results/<name>/data/ \\
        --points slide1.tiff points.csv
"""

import argparse
import glob
import pathlib

import numpy as np
import pandas as pd

from valis import registration


def find_pickle(data_dir: str) -> str:
    matches = glob.glob(str(pathlib.Path(data_dir) / "*.pickle"))
    if not matches:
        raise FileNotFoundError(f"No pickled registrar found in {data_dir}")
    return matches[0]


def main(data_dir: str, registered_dir: str, points_args: list) -> None:
    pickle_f = find_pickle(data_dir)
    print(f"Loading registrar from: {pickle_f}")
    registrar = registration.load_registrar(pickle_f)

    # --- Warp and save slides -------------------------------------------
    pathlib.Path(registered_dir).mkdir(parents=True, exist_ok=True)
    registrar.warp_and_save_slides(registered_dir)
    print(f"Warped slides saved to: {registered_dir}")

    # --- Warp point coordinates (optional) --------------------------------
    if points_args:
        slide_name, csv_path = points_args
        slide_obj = registrar.get_slide(slide_name)
        if slide_obj is None:
            print(
                f"Could not find slide '{slide_name}' in registrar. Available slides:"
            )
            for name in registrar.slide_dict:
                print(f"  {name}")
            return

        pts_df = pd.read_csv(csv_path)
        xy = pts_df[["x", "y"]].values.astype(float)
        warped_xy = slide_obj.warp_xy(xy)
        out_f = str(pathlib.Path(csv_path).with_suffix("")) + "_warped.csv"
        pd.DataFrame(warped_xy, columns=["x_warped", "y_warped"]).to_csv(
            out_f, index=False
        )
        print(f"Warped points saved to: {out_f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume VALIS from saved state")
    parser.add_argument(
        "--data", required=True, help="Path to the registrar's data/ directory"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Where to save warped slides (default: data/../registered)",
    )
    parser.add_argument(
        "--points",
        nargs=2,
        metavar=("SLIDE_NAME", "CSV"),
        help="Warp (x,y) points from CSV for the given slide name",
    )
    args = parser.parse_args()

    data_dir = args.data
    out_dir = args.out or str(pathlib.Path(data_dir).parent / "registered")
    main(data_dir, out_dir, args.points)
