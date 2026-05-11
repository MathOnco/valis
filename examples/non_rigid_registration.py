"""Non-rigid registration example.

Demonstrates how to enable and configure non-rigid (deformable) registration,
when to use it, and how to tune the key parameters.

Non-rigid registration is useful when:
  - Images were acquired at different times (tissue shrinkage / swelling)
  - Different stains cause structural deformation
  - There is non-linear distortion from the imaging system

For most brightfield H&E / IHC series, rigid registration is sufficient.
Non-rigid registration is most valuable for cyclic immunofluorescence (CyCIF),
CODEX, or any multi-round assay where the tissue can deform between rounds.

Usage
-----
    python examples/non_rigid_registration.py --src /path/to/slides --dst /path/to/results

Optional flags
--------------
    --no-non-rigid    Run rigid-only (useful for comparison)
    --compose         Compose non-rigid fields serially (useful for large deformations)
"""

import argparse
import pathlib

from valis import registration, non_rigid_registrars
from valis.registration import RegistrationConfig, CropMode


def main(src_dir: str, dst_dir: str, do_non_rigid: bool, compose: bool) -> None:
    if do_non_rigid:
        # OpticalFlowWarper is the default and works well for most cases.
        # For large deformations, try SimpleElastix (requires SimpleITK).
        nr_cls = non_rigid_registrars.OpticalFlowWarper()
        print("Non-rigid registration: OpticalFlowWarper")
    else:
        nr_cls = None
        print("Non-rigid registration: disabled (rigid only)")

    config = RegistrationConfig(
        non_rigid_registrar_cls=nr_cls,
        compose_non_rigid=compose,
        # Larger window improves non-rigid accuracy at the cost of speed
        max_non_rigid_registration_dim_px=2048,
        crop=CropMode.OVERLAP,
    )

    registrar = registration.Valis(src_dir, dst_dir, config=config)
    rigid_registrar, non_rigid_registrar, error_df = registrar.register()

    cols = ["from", "to", "mean_rigid_D"]
    if do_non_rigid:
        cols.append("mean_non_rigid_D")
    print("\nRegistration error:")
    print(error_df[cols].to_string(index=False))

    registered_dir = str(pathlib.Path(dst_dir) / "registered")
    registrar.warp_and_save_slides(registered_dir, non_rigid=do_non_rigid)
    print(f"\nWarped slides saved to: {registered_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Non-rigid VALIS registration")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument(
        "--no-non-rigid",
        action="store_true",
        help="Disable non-rigid registration (rigid only)",
    )
    parser.add_argument(
        "--compose",
        action="store_true",
        help="Compose non-rigid deformation fields serially",
    )
    args = parser.parse_args()

    main(args.src, args.dst, not args.no_non_rigid, args.compose)
