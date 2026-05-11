"""Basic VALIS registration example.

Register a directory of images, then warp and save them as ome.tiff files.
This is the minimal working example — roughly 20 lines of "real" code.

Usage
-----
    python examples/basic_registration.py --src /path/to/slides --dst /path/to/results

The script will:
  1. Register every supported image in ``src_dir``
  2. Print a summary table of registration error
  3. Save each registered slide as an ome.tiff in ``dst_dir/registered/``
"""

import argparse
import pathlib
import sys

from valis import registration


def main(src_dir: str, dst_dir: str) -> None:
    registrar = registration.Valis(src_dir, dst_dir)

    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    print("\nRegistration summary:")
    print(
        error_df[
            ["from", "to", "mean_original_D", "mean_rigid_D", "mean_non_rigid_D"]
        ].to_string(index=False)
    )

    registered_dir = str(pathlib.Path(dst_dir) / "registered")
    registrar.warp_and_save_slides(registered_dir)
    print(f"\nWarped slides saved to: {registered_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic VALIS registration")
    parser.add_argument(
        "--src", required=True, help="Directory containing source slides"
    )
    parser.add_argument(
        "--dst", required=True, help="Directory for registration results"
    )
    args = parser.parse_args()

    main(args.src, args.dst)
