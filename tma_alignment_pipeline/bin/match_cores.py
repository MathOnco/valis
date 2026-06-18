#!/usr/bin/env python
"""
Detect tissue core centroids in the H&E whole-slide image and match them
by grid rank to the morphology TMA centroids from the geoparquet, then
emit an hne_tma_boxes.json sized to each core's detected circle radius.
"""

import argparse
import json
import os

import cv2
import geopandas as gpd
import numpy as np
import pyvips


def gap_grid_sort(pts, n_rows):
    sorted_items = sorted(pts.items(), key=lambda kv: kv[1][1])
    ids = [item[0] for item in sorted_items]
    y_vals = np.array([item[1][1] for item in sorted_items])
    gaps = np.diff(y_vals)
    split_positions = sorted(np.argsort(gaps)[::-1][: n_rows - 1] + 1)
    rows, prev = [], 0
    for pos in split_positions:
        rows.append(ids[prev:pos])
        prev = pos
    rows.append(ids[prev:])
    order = []
    for row in rows:
        order.extend(sorted(row, key=lambda i: pts[i][0]))
    return order


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--he-image", required=True)
    p.add_argument("--morphology-boxes", required=True)
    p.add_argument("--um-per-px", type=float, default=0.2125)
    p.add_argument(
        "--n-rows",
        type=int,
        default=10,
        help="number of TMA rows expected on the slide",
    )
    p.add_argument("--target-thumb-height", type=int, default=3000)
    p.add_argument("--pad", type=int, default=150)
    p.add_argument("--min-area", type=int, default=1500)
    p.add_argument("--out", required=True)
    p.add_argument("--mask-debug", default=None)
    args = p.parse_args()

    gdf = gpd.read_parquet(args.parquet)
    morph_pts = {}
    for _, row in gdf.iterrows():
        tid = int(row["tma_id"])
        morph_pts[tid] = (
            row.geometry.centroid.x / args.um_per_px,
            row.geometry.centroid.y / args.um_per_px,
        )
    print(f"Morphology: {len(morph_pts)} centroids")

    he = pyvips.Image.new_from_file(args.he_image)
    he_w, he_h = he.width, he.height
    scale = args.target_thumb_height / he_h
    thumb = he.resize(scale)
    arr = thumb.numpy()
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = arr

    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if args.mask_debug:
        cv2.imwrite(args.mask_debug, mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    es_spots = []
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < args.min_area:
            continue
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx_full = (M["m10"] / M["m00"]) / scale
        cy_full = (M["m01"] / M["m00"]) / scale
        _, radius_thumb = cv2.minEnclosingCircle(cnt)
        es_spots.append((cx_full, cy_full, radius_thumb / scale))
    print(f"H&E: {len(es_spots)} spots (need {len(morph_pts)})")

    if len(es_spots) != len(morph_pts):
        raise SystemExit(
            f"core count mismatch — adjust --min-area or --target-thumb-height "
            f"(detected {len(es_spots)}, expected {len(morph_pts)})"
        )

    es_pts = {i: (s[0], s[1]) for i, s in enumerate(es_spots)}
    es_radii = {i: s[2] for i, s in enumerate(es_spots)}

    morph_order = gap_grid_sort(morph_pts, n_rows=args.n_rows)
    es_order = gap_grid_sort(es_pts, n_rows=args.n_rows)
    tma_to_es = {morph_order[i]: es_order[i] for i in range(len(morph_order))}

    with open(args.morphology_boxes) as f:
        morph_boxes = json.load(f)

    hne_boxes = {}
    for tma_id_str in morph_boxes:
        tid = int(tma_id_str)
        es_idx = tma_to_es[tid]
        ex, ey = es_pts[es_idx]
        r = es_radii[es_idx] + args.pad
        x = max(0, int(ex - r))
        y = max(0, int(ey - r))
        x1 = min(he_w, int(ex + r))
        y1 = min(he_h, int(ey + r))
        hne_boxes[tma_id_str] = {"x": x, "y": y, "w": x1 - x, "h": y1 - y}

    with open(args.out, "w") as f:
        json.dump(hne_boxes, f, indent=2)
    print(f"Wrote {len(hne_boxes)} boxes to {args.out}")


if __name__ == "__main__":
    main()
