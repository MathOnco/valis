#!/usr/bin/env python
"""Side-by-side thumbnail of one TMA's morphology crop vs. its H&E crop."""

import argparse
import json
import os

import cv2
import numpy as np
import pyvips


def crop_thumb(img, box, thumb_h):
    crop = img.crop(box["x"], box["y"], box["w"], box["h"])
    scale = thumb_h / box["h"]
    arr = crop.resize(scale).numpy()
    if arr.ndim == 2 or arr.shape[2] == 1:
        arr = arr.squeeze()
        lo = np.percentile(arr[arr > 0], 1) if arr.max() > 0 else 0
        hi = np.percentile(arr[arr > 0], 99.9) if arr.max() > 0 else 255
        arr = np.clip(
            (arr.astype(np.float32) - lo) / max(hi - lo, 1) * 255, 0, 255
        ).astype(np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--morphology", required=True)
    p.add_argument("--he", required=True)
    p.add_argument("--morphology-boxes", required=True)
    p.add_argument("--he-boxes", required=True)
    p.add_argument("--tma-id", required=True)
    p.add_argument("--thumb-height", type=int, default=400)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.morphology_boxes) as f:
        mb = json.load(f)[str(args.tma_id)]
    with open(args.he_boxes) as f:
        eb = json.load(f)[str(args.tma_id)]

    morph_img = pyvips.Image.new_from_file(args.morphology)
    he_img = pyvips.Image.new_from_file(args.he)

    m = crop_thumb(morph_img, mb, args.thumb_height)
    e = crop_thumb(he_img, eb, args.thumb_height)

    max_h = max(m.shape[0], e.shape[0])
    if m.shape[0] < max_h:
        m = np.pad(m, ((0, max_h - m.shape[0]), (0, 0), (0, 0)))
    if e.shape[0] < max_h:
        e = np.pad(e, ((0, max_h - e.shape[0]), (0, 0), (0, 0)))

    cv2.putText(
        m,
        f"TMA {args.tma_id} morphology",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        e,
        f"TMA {args.tma_id} H&E",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    div = np.full((max_h, 4, 3), 255, dtype=np.uint8)
    panel = np.hstack([m, div, e])
    cv2.imwrite(args.out, panel)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
