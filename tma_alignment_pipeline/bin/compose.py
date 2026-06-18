#!/usr/bin/env python
"""Compose aligned TMA crops back onto a blank morphology-sized canvas."""

import argparse
import glob
import json
import os
import re
import pyvips


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--morphology",
        required=True,
        help="reference morphology OME-TIFF; canvas size taken from its header",
    )
    p.add_argument("--morphology-boxes", required=True)
    p.add_argument(
        "--aligned-dir",
        required=True,
        help="dir containing aligned_tma_<id>.ome.tif files",
    )
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.morphology_boxes) as f:
        boxes = json.load(f)

    morph = pyvips.Image.new_from_file(args.morphology)
    img_w, img_h = morph.width, morph.height
    print(f"Canvas: {img_w}x{img_h}")

    canvas = pyvips.Image.black(img_w, img_h, bands=3).cast("uchar")

    pattern = re.compile(r"aligned_tma_(\d+)\.ome\.tif$")
    aligned = {}
    for fp in glob.glob(os.path.join(args.aligned_dir, "aligned_tma_*.ome.tif")):
        m = pattern.search(os.path.basename(fp))
        if m:
            aligned[m.group(1)] = fp

    for tma_id in sorted(boxes.keys(), key=int):
        bbox = boxes[tma_id]
        fp = aligned.get(tma_id)
        if not fp or not os.path.exists(fp):
            print(f"  TMA {tma_id}: aligned file missing, skipping")
            continue
        warped = pyvips.Image.new_from_file(fp, page=0)
        if warped.bands == 1:
            warped = warped.bandjoin([warped, warped])
        elif warped.bands > 3:
            warped = warped.extract_band(0, n=3)
        x, y = bbox["x"], bbox["y"]
        w = min(warped.width, img_w - x)
        h = min(warped.height, img_h - y)
        if w != warped.width or h != warped.height:
            warped = warped.crop(0, 0, w, h)
        canvas = canvas.insert(warped, x, y)
        print(f"  TMA {tma_id}: inserted at ({x},{y})")

    canvas.tiffsave(
        args.out,
        bigtiff=True,
        tile=True,
        pyramid=True,
        tile_width=256,
        tile_height=256,
        compression="lzw",
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
