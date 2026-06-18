# TMA alignment pipeline

A small [Nextflow](https://www.nextflow.io/) pipeline that aligns an H&E
whole-slide image to a reference morphology (e.g. Xenium / IF) whole-slide
image, **one TMA core at a time**, then composes the per-core warped H&E
results back onto a single morphology-sized canvas.

The per-core alignment itself is delegated to
[Valis](https://github.com/MathOnco/valis) via the `align_two_images.py`
helper script.

## How it works

The core problem this pipeline solves: **whole-slide registration between
an H&E scan and a multiplex/IF "morphology" image of a TMA fails when run
on the full slides**. The two modalities are at different physical scales,
have different rotations, contain large blank regions between cores, and
each core is small relative to the slide — feature-based registration
gets distracted by global structure and misaligns individual cores.

The trick is to **divide the slide into individual TMA cores up front**,
register each one independently, then paste the per-core results back onto
a single canvas in the morphology's coordinate system. Each per-core
registration is a tractable problem (one piece of tissue, roughly
centered, similar in scale), and the pieces are embarrassingly parallel.

This requires solving two sub-problems before alignment can run:

1. **Where is each TMA core on each slide?** For the morphology image,
   we trust the polygons in the TMA-boundaries GeoParquet — they were
   drawn against this exact coordinate system. The polygon bounds (in
   microns) divide by `um_per_px` to get morphology pixel bboxes directly.
2. **Which H&E core corresponds to which morphology TMA?** The H&E image
   has no a priori bbox annotations, and a naive global affine fit
   between morphology and H&E centroids accumulates error toward the
   slide's bottom rows (the original `step2b` failure mode). Instead, we
   detect H&E core centroids with Otsu thresholding + contour analysis on
   a thumbnail, then sort **both** sets of centroids into the same grid
   order using a gap-based row-detection heuristic: sort by `y`, find the
   `n_rows - 1` largest gaps in the sorted `y` values, use those gaps as
   row boundaries, then sort each row by `x`. Matching is then by rank
   within that grid — TMA `i` in the morphology grid maps to detected
   core `i` in the H&E grid. This sidesteps the bad-affine problem
   entirely: we never compute a global transform, just two independent
   topological sorts.

Once both sets of bboxes exist, the rest is straightforward:

- Crop both modalities to each core's bbox (one pair per TMA).
- Hand each pair to Valis `align_two_images.py`, which does the actual
  per-core feature-based registration and warping (morphology = reference,
  H&E = moving).
- Allocate a blank canvas the size of the morphology image and `insert`
  each warped H&E crop at its morphology bbox position.

A side-effect of this design is that **everything from step 3 onward is
per-TMA and parallelizable** — Nextflow fans out `VERIFY_CROP_MATCHING`
and `CROP_AND_ALIGN` to one task per core, and `-resume` lets you re-run
only the cores that failed without re-doing the ones that already
succeeded.

### Why a separate verify step

The grid-rank matching can fail silently if the H&E thumbnail mask misses
a core or merges two adjacent ones — the counts will mismatch (we hard-fail
there) but more subtly, an off-by-one in any row will scramble every
subsequent assignment. `VERIFY_CROP_MATCHING` writes a side-by-side
thumbnail of `morphology_crop` vs. `hne_crop` for each TMA, labeled with
the TMA ID. Flipping through `results/verify_crops/*.png` is the cheap
way to confirm the matching is correct before letting Valis spend hours
on 30+ alignments.

## Pipeline steps

1. **`EXTRACT_MORPHOLOGY_BOXES`** — reads the TMA boundaries geoparquet,
   converts the polygon bounds from microns to morphology pixel coordinates,
   pads them, and writes `morphology_boxes.json`.
2. **`MATCH_CORES`** — thumbnails the H&E image, Otsu-thresholds it to a
   tissue mask, extracts contour centroids + radii for every core, and
   matches them by grid-rank (largest y-gaps define rows) to the morphology
   centroids from the parquet. Writes `hne_tma_boxes.json`.
3. **`VERIFY_CROP_MATCHING`** — runs once per TMA in parallel. Produces a
   side-by-side thumbnail of that core's morphology crop vs. its H&E crop
   so you can eyeball the matching before / during the heavy alignment step.
4. **`CROP_AND_ALIGN`** — runs **once per TMA in parallel**. Crops both
   modalities to that core's bbox and runs Valis pairwise alignment
   (morphology = reference, H&E = moving). Emits `aligned_tma_<id>.ome.tif`.
5. **`COMPOSE`** — pastes every `aligned_tma_<id>.ome.tif` onto a blank
   canvas the size of the morphology slide at its `morphology_boxes.json`
   position. Output: `aligned_to_morphology.ome.tif` (tiled, pyramidal).

## Inputs

| Param            | Description                                                                                                                                  |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `--parquet`      | TMA-boundaries **GeoParquet** (`tma_boundaries_with_metadata.geo.parquet`). One row per TMA, polygon geometry in **microns**, column `tma_id`. |
| `--morphology`   | Reference morphology **OME-TIFF** (standard fluorescence focus image: bright signal on dark background; inverted-fluorescence is also supported via `--reference_stain inverted-fluorescence`). Coordinate system used for the final composed output. Width/height are read from the file header. |
| `--he`           | Moving **H&E OME-TIFF** to be aligned to the morphology image.                                                                               |
| `--outdir`       | Output directory (default: `results`).                                                                                                       |

### Optional tuning parameters

| Param                  | Default                            | Description                                                              |
| ---------------------- | ---------------------------------- | ------------------------------------------------------------------------ |
| `--um_per_px`          | `0.2125`                           | Microns per pixel for the morphology image (used to convert parquet µm → px). |
| `--pad`                | `200`                              | Padding (px) around each morphology bbox.                                |
| `--he_pad`             | `150`                              | Padding (px) added to the detected H&E core circle radius.               |
| `--n_rows`             | `10`                               | Expected number of TMA rows on the slide (used for grid-rank matching).  |
| `--thumb_height`       | `3000`                             | Thumbnail height (px) used for H&E core detection.                       |
| `--min_area`           | `1500`                             | Minimum thumbnail-pixel contour area to count as a core.                 |
| `--verify_thumb_height`| `400`                              | Per-panel height (px) of the side-by-side verify thumbnails.             |
| `--valis_python`       | conda env path on `tanseyw`        | Python interpreter that has Valis installed. The alignment driver (`bin/align_two_images.py`) is bundled in this directory. |
| `--max_processed_dim`  | `1024`                             | `--max-processed-image-dim-px` passed to Valis.                          |
| `--reference_stain`    | `fluorescence`                     | `--reference-stain` passed to Valis. Use `inverted-fluorescence` for white-background DAPI-style inputs. |
| `--image_stain`        | `he-hematoxylin-sparse`            | `--image-stain` passed to Valis.                                         |

## Outputs

Published into `--outdir`:

```
results/
├── morphology_boxes.json          # morphology-coord bboxes, keyed by tma_id
├── hne_tma_boxes.json              # H&E-coord bboxes, keyed by tma_id
├── he_mask.png                    # debug: Otsu tissue mask of the H&E thumbnail
├── verify_crops/
│   └── tma_<id>.png               # side-by-side morphology vs. H&E crop, per core
├── aligned/
│   └── aligned_tma_<id>.ome.tif   # per-TMA warped H&E (one per core)
├── logs/
│   └── align_tma_<id>.log         # Valis stdout/stderr per core
├── aligned_to_morphology.ome.tif  # final composed image, in morphology coords
├── nextflow_report.html
└── nextflow_trace.txt
```

## Running it

The `bin/` scripts assume `pyvips`, `geopandas`, `opencv-python`, and `numpy`
are importable. The simplest setup is to activate the Valis conda env first,
since it already has everything:

```bash
conda activate /data1/tanseyw/quinnj2/conda_environments/valis
```

Then:

```bash
nextflow run main.nf \
    --parquet     tma_boundaries_with_metadata.geo.parquet \
    --morphology  morphology_focus_0000_8bit.tif \
    --he          ES-3990_R1-S1_cropped_rgb.tif \
    --outdir      results
```

To run on LSF instead of the local machine:

```bash
nextflow run main.nf -profile lsf [...]
```

To resume after a failed/partial run (Nextflow's killer feature — already
finished cores will not be re-aligned):

```bash
nextflow run main.nf -resume [...]
```

## Notes / caveats

- **Core count must match.** `MATCH_CORES` fails if the number of detected
  H&E cores differs from the number of morphology TMAs. If that happens,
  inspect `results/he_mask.png` and tune `--min_area` or `--thumb_height`.
- **Grid-rank matching** assumes the TMAs are arranged in a regular grid
  and that both modalities have the same number of rows. The original
  ES-3990 case had 10 rows; set `--n_rows` accordingly.
- **Stain flags** are forwarded verbatim to Valis. The defaults are
  tuned for standard fluorescence reference (bright nuclei on dark bg)
  vs. sparse H&E hematoxylin moving — change them if you align other
  modality pairs. For an inverted (white-background) DAPI-style reference,
  pass `--reference_stain inverted-fluorescence`.
