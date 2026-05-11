#!/usr/bin/env python
"""Crop one TMA core out of a whole-slide image given a bbox JSON entry."""

import argparse
import json
import pyvips


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--boxes", required=True)
    p.add_argument("--tma-id", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.boxes) as f:
        boxes = json.load(f)
    b = boxes[str(args.tma_id)]
    img = pyvips.Image.new_from_file(args.image)
    crop = img.crop(b["x"], b["y"], b["w"], b["h"])
    crop.tiffsave(args.out, compression="none", bigtiff=True, strip=True)
    print(f"Wrote {args.out}  ({b['w']}x{b['h']})")


if __name__ == "__main__":
    main()
