#!/usr/bin/env python
"""Warp a new moving image through an existing Valis registration result.

Given a saved Valis registrar (the ``*_registrar.pickle`` produced by
``Valis.register()``) and a new image, this re-uses a registered slide's
learned warp (rigid matrix ``M`` + non-rigid displacement field) to warp the
new image into the registered coordinate space. The result is written as an
OME-TIFF.

A note on Valis's model: there is no single "fixed" / "moving" pair. Valis
aligns a *stack* of slides onto one **reference** slide (the "fixed" one) and
stores a separate warp for every other ("moving") slide. So to warp a new
image you must say which slide's warp to apply:

* If the registration has exactly one non-reference (moving) slide, that warp
  is used automatically -- this covers the common two-image case.
* Otherwise (multi-slide alignment) the choice is ambiguous and you must name
  the slide with ``--slide``. Use ``--list`` to see the available slides and
  which one is the reference.

The new image must be an alternate copy of the slide that produced the chosen
warp -- e.g. a different stain channel, a mask, or a re-scanned version of the
same physical slide. Its pyramid geometry should match the slide the transform
was computed on; otherwise the transform is rescaled and alignment may be
approximate (Valis will log a warning).

Examples
--------
List the slides in a registrar (and which is the reference)::

    python scripts/warp_new_image.py --registrar path/to/foo_registrar.pickle --list

Warp a new image (single moving slide -- warp inferred automatically)::

    python scripts/warp_new_image.py \
        --registrar path/to/foo_registrar.pickle \
        --moving-image path/to/new_image.ome.tiff \
        --output path/to/new_image_registered.ome.tiff

Multi-slide registration -- name the slide whose warp to apply::

    python scripts/warp_new_image.py \
        --registrar path/to/foo_registrar.pickle \
        --slide HE \
        --moving-image path/to/new_image.ome.tiff \
        --output path/to/new_image_registered.ome.tiff
"""

import argparse
import os
import sys

# NOTE: import valis before any pytorch-related import or Python may segfault.
from valis import registration


def parse_args():
    parser = argparse.ArgumentParser(
        description="Warp a new moving image through an existing Valis warp function.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--registrar",
        required=True,
        help="Path to the saved Valis registrar pickle (*_registrar.pickle).",
    )
    parser.add_argument(
        "--moving-image",
        help="Path to the new moving image to warp.",
    )
    parser.add_argument(
        "--output",
        help="Path for the warped OME-TIFF output (forced to .ome.tiff).",
    )
    parser.add_argument(
        "--slide",
        default=None,
        help=(
            "Name (or source path) of the slide whose warp to apply. Valis "
            "registers a stack of slides into a shared reference frame and "
            "stores one warp per slide -- there is no single 'moving' image to "
            "infer. When the registration has exactly one non-reference "
            "(i.e. moving) slide, its warp is used automatically. For "
            "multi-slide alignments this argument is required to disambiguate."
        ),
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="Pyramid level to warp (0 = full resolution).",
    )
    parser.add_argument(
        "--rigid-only",
        action="store_true",
        help="Apply only the rigid transform (skip the non-rigid displacement).",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Do not crop the warped image to the registered overlap region.",
    )
    parser.add_argument(
        "--interp-method",
        default="bicubic",
        help="Interpolation method for warping.",
    )
    parser.add_argument(
        "--compression",
        default=None,
        help="TIFF compression (default: Valis default, e.g. lzw).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the slides available in the registrar and exit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.registrar):
        sys.exit(f"Registrar pickle not found: {args.registrar}")

    print(f"Loading registrar: {args.registrar}")
    registrar = registration.load_registrar(args.registrar)

    slide_names = sorted(registrar.slide_dict.keys())
    if args.list:
        ref_slide = registrar.get_ref_slide()
        ref_name = ref_slide.name if ref_slide is not None else "?"
        print(f"\nSlides in registrar (reference = {ref_name!r}):")
        for name in slide_names:
            print(f"  - {name}")
        return

    if not args.moving_image:
        sys.exit("--moving-image is required (or use --list).")
    if not args.output:
        sys.exit("--output is required (or use --list).")
    if not os.path.isfile(args.moving_image):
        sys.exit(f"Moving image not found: {args.moving_image}")

    # Resolve which slide's warp to use.
    ref_slide = registrar.get_ref_slide()
    ref_name = ref_slide.name if ref_slide is not None else None

    if args.slide is not None:
        # Explicit override.
        try:
            slide_obj = registrar.get_slide(args.slide)
        except KeyError:
            sys.exit(
                f"Slide {args.slide!r} not found. Available slides: {slide_names}"
            )
        print(f"Using slide warp: {slide_obj.name!r}")
    else:
        # Default: a Valis registration aligns a stack onto one reference
        # ("fixed") slide; every other slide is a "moving" slide with its own
        # warp. If there is exactly one non-reference slide, it is unambiguously
        # "the moving image" and we use its warp. Otherwise the choice is
        # ambiguous and the user must pass --slide.
        moving_names = [n for n in slide_names if n != ref_name]
        if len(moving_names) == 1:
            slide_obj = registrar.slide_dict[moving_names[0]]
            print(
                f"Using the single moving slide's warp: {slide_obj.name!r} "
                f"(reference/fixed slide is {ref_name!r})"
            )
        else:
            sys.exit(
                f"This registration has {len(moving_names)} moving slides "
                f"(reference/fixed slide is {ref_name!r}), so the warp to apply "
                f"is ambiguous.\n"
                f"Re-run with --slide set to one of: {moving_names}"
            )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    non_rigid = not args.rigid_only
    crop = not args.no_crop

    print(
        f"Warping {args.moving_image}\n"
        f"  level={args.level} non_rigid={non_rigid} crop={crop} "
        f"interp={args.interp_method}"
    )

    warp_kwargs = dict(
        dst_f=args.output,
        level=args.level,
        non_rigid=non_rigid,
        crop=crop,
        src_f=args.moving_image,
        interp_method=args.interp_method,
    )
    if args.compression is not None:
        warp_kwargs["compression"] = args.compression

    slide_obj.warp_and_save_slide(**warp_kwargs)

    print(f"\nDone. Wrote warped image to: {args.output}")


if __name__ == "__main__":
    main()
