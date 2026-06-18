#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

/*
 * TMA alignment pipeline:
 *   1. extract morphology bboxes from a TMA-boundaries geoparquet
 *   2. detect H&E core centroids and match them by grid rank to morphology TMAs
 *   3. per-TMA: crop both modalities and run Valis pairwise alignment
 *   4. compose all aligned crops back onto a morphology-sized canvas
 */

params.parquet         = null
params.morphology      = null   // morphology OME-TIFF (reference)
params.he              = null   // H&E OME-TIFF (moving)
params.outdir          = "results"

params.um_per_px       = 0.2125
params.pad             = 200
params.he_pad          = 150
params.n_rows          = 10
params.thumb_height    = 3000
params.min_area        = 1500
params.verify_thumb_height = 400

params.valis_python    = "/data1/tanseyw/quinnj2/conda_environments/valis/bin/python"
params.max_processed_dim = 1024
params.reference_stain = "fluorescence"
params.image_stain     = "he-hematoxylin"


process EXTRACT_MORPHOLOGY_BOXES {
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path parquet
    path morphology

    output:
    path "morphology_boxes.json", emit: boxes

    script:
    """
    extract_morphology_boxes.py \\
        --parquet ${parquet} \\
        --morphology ${morphology} \\
        --um-per-px ${params.um_per_px} \\
        --pad ${params.pad} \\
        --out morphology_boxes.json
    """
}


process MATCH_CORES {
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path parquet
    path he
    path morphology_boxes

    output:
    path "hne_tma_boxes.json", emit: boxes
    path "he_mask.png",       optional: true

    script:
    """
    match_cores.py \\
        --parquet ${parquet} \\
        --he-image ${he} \\
        --morphology-boxes ${morphology_boxes} \\
        --um-per-px ${params.um_per_px} \\
        --n-rows ${params.n_rows} \\
        --target-thumb-height ${params.thumb_height} \\
        --pad ${params.he_pad} \\
        --min-area ${params.min_area} \\
        --out hne_tma_boxes.json \\
        --mask-debug he_mask.png
    """
}


process VERIFY_CROP_MATCHING {
    tag "tma_${tma_id}"
    publishDir "${params.outdir}/verify_crops", mode: 'copy'

    input:
    tuple val(tma_id),
          path(morphology),
          path(he),
          path(morphology_boxes),
          path(hne_tma_boxes)

    output:
    path "tma_${tma_id}.png"

    script:
    """
    verify_crops.py \\
        --morphology ${morphology} \\
        --he ${he} \\
        --morphology-boxes ${morphology_boxes} \\
        --he-boxes ${hne_tma_boxes} \\
        --tma-id ${tma_id} \\
        --thumb-height ${params.verify_thumb_height} \\
        --out tma_${tma_id}.png
    """
}


process CROP_AND_ALIGN {
    tag "tma_${tma_id}"
    publishDir "${params.outdir}/aligned", mode: 'copy', pattern: "aligned_tma_*.ome.tif"
    publishDir "${params.outdir}/logs",    mode: 'copy', pattern: "align_tma_*.log"

    input:
    tuple val(tma_id),
          path(morphology),
          path(he),
          path(morphology_boxes),
          path(hne_tma_boxes)

    output:
    path "aligned_tma_${tma_id}.ome.tif", emit: aligned, optional: true
    path "align_tma_${tma_id}.log",       emit: log

    script:
    """
    set -o pipefail

    crop_tma.py \\
        --image ${morphology} \\
        --boxes ${morphology_boxes} \\
        --tma-id ${tma_id} \\
        --out morph_tma_${tma_id}.tif

    crop_tma.py \\
        --image ${he} \\
        --boxes ${hne_tma_boxes} \\
        --tma-id ${tma_id} \\
        --out he_tma_${tma_id}.tif

    mkdir -p aligned_${tma_id}
    ${params.valis_python} \$(which align_two_images.py) \\
        --reference morph_tma_${tma_id}.tif \\
        --image     he_tma_${tma_id}.tif \\
        --output-dir aligned_${tma_id} \\
        --max-processed-image-dim-px ${params.max_processed_dim} \\
        --reference-stain ${params.reference_stain} \\
        --image-stain ${params.image_stain} \\
        --no-script-orientation \\
        > align_tma_${tma_id}.log 2>&1

    if [ -f aligned_${tma_id}/aligned.ome.tif ]; then
        mv aligned_${tma_id}/aligned.ome.tif aligned_tma_${tma_id}.ome.tif
    fi
    """
}


process COMPOSE {
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path morphology
    path morphology_boxes
    path aligned_tifs

    output:
    path "aligned_to_morphology.ome.tif"

    script:
    """
    mkdir -p aligned_in
    for f in ${aligned_tifs}; do ln -sf \$PWD/\$f aligned_in/; done

    compose.py \\
        --morphology ${morphology} \\
        --morphology-boxes ${morphology_boxes} \\
        --aligned-dir aligned_in \\
        --out aligned_to_morphology.ome.tif
    """
}


workflow {
    if (!params.parquet || !params.morphology || !params.he) {
        error "Required: --parquet, --morphology, --he"
    }

    parquet_ch    = Channel.fromPath(params.parquet, checkIfExists: true)
    morphology_ch = Channel.fromPath(params.morphology, checkIfExists: true)
    he_ch         = Channel.fromPath(params.he, checkIfExists: true)

    morph_boxes = EXTRACT_MORPHOLOGY_BOXES(parquet_ch, morphology_ch).boxes
    hne_boxes   = MATCH_CORES(parquet_ch, he_ch, morph_boxes).boxes

    tma_ids = morph_boxes
        .map { f -> new groovy.json.JsonSlurper().parse(f).keySet().collect { it as String } }
        .flatten()

    per_tma = tma_ids.combine(morphology_ch).combine(he_ch)
                     .combine(morph_boxes).combine(hne_boxes)

    VERIFY_CROP_MATCHING(per_tma)

    aligned = CROP_AND_ALIGN(per_tma).aligned

    COMPOSE(morphology_ch, morph_boxes, aligned.collect())
}
