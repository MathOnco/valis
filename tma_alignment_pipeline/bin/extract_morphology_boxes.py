#!/usr/bin/env python
"""Extract morphology TMA bounding boxes from the geoparquet file."""

import argparse
import json
import geopandas as gpd
import pyvips


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument(
        "--morphology",
        required=True,
        help="morphology OME-TIFF; width/height read from its header",
    )
    p.add_argument("--um-per-px", type=float, default=0.2125)
    p.add_argument("--pad", type=int, default=200)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    morph = pyvips.Image.new_from_file(args.morphology)
    img_w, img_h = morph.width, morph.height
    print(f"Morphology image: {img_w}x{img_h}")

    gdf = gpd.read_parquet(args.parquet)
    boxes = {}
    for _, row in gdf.iterrows():
        tma_id = int(row["tma_id"])
        minx, miny, maxx, maxy = row.geometry.bounds
        x_orig = int(minx / args.um_per_px) - args.pad
        y_orig = int(miny / args.um_per_px) - args.pad
        x1 = int(maxx / args.um_per_px) + args.pad
        y1 = int(maxy / args.um_per_px) + args.pad
        x = max(0, x_orig)
        y = max(0, y_orig)
        x1 = min(img_w, x1)
        y1 = min(img_h, y1)
        boxes[tma_id] = {
            "x": x,
            "y": y,
            "w": x1 - x,
            "h": y1 - y,
            "x_orig": x_orig,
            "y_orig": y_orig,
        }

    with open(args.out, "w") as f:
        json.dump(boxes, f, indent=2)
    print(f"Wrote {len(boxes)} bboxes to {args.out}")


if __name__ == "__main__":
    main()
