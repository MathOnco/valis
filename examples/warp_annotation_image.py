"""
Example showing how to warp an annotation image to go onto another image.

`slide_src_dir` is where the slides to be registered are located
`results_dst_dir` is where to save the results
`slide_src_dir` is where the slides to be registered are located
`reference_img_f` is the filename of the image on which the annotations are based
`labeled_img_f` is the filename of actual annotation image
"""

import os
import pathlib
from valis import registration, slide_io, viz, warp_tools



# Perform registration. Can optinally set the reference image to be the same as the one the annotations are based on (i.e. `reference_img_f`)
registrar = registration.Valis(slide_src_dir, results_dst_dir, reference_img_f=reference_img_f)
rigid_registrar, non_rigid_registrar, error_df = registrar.register()

# Load labeled image saved as `labeled_img_f`
labeled_img_reader_cls = slide_io.get_slide_reader(labeled_img_f)
labeled_img_reader = labeled_img_reader_cls(labeled_img_f)
labeled_img = labeled_img_reader.slide2vips(0)

# Have reference slide warp the labeled image onto the others and save the results
reference_slide = registrar.get_slide(reference_img_f)

annotations_dst_dir = os.path.join(registrar.dst_dir, "annotations")
pathlib.Path(annotations_dst_dir).mkdir(exist_ok=True, parents=True)

# Create and save an annotation image for each slide
for slide_obj in registrar.slide_dict.values():
    if slide_obj == reference_slide:
        continue

    # Warp the labeled image from the annotation image to this different slide #
    transferred_annotation_img = reference_slide.warp_img_from_to(labeled_img,
                                    to_slide_obj=slide_obj,
                                    interp_method="nearest")

    # Create image metadata. Note that you could add channel names if your labeled image has a different classification in each channel
    bf_dtype = slide_io.vips2bf_dtype(transferred_annotation_img.format)
    xyzct = slide_io.get_shape_xyzct((transferred_annotation_img.width, transferred_annotation_img.height), transferred_annotation_img.bands)
    px_phys_size = reference_slide.reader.metadata.pixel_physical_size_xyu
    new_ome = slide_io.create_ome_xml(xyzct, bf_dtype, is_rgb=False, pixel_physical_size_xyu=px_phys_size)
    ome_xml = new_ome.to_xml()

    # Save the labeled image as an ome.tiff #
    dst_f = os.path.join(annotations_dst_dir, f"{slide_obj.name}_annotations.ome.tiff")
    slide_io.save_ome_tiff(transferred_annotation_img, dst_f=dst_f, ome_xml=ome_xml)

    # Save a thumbnail with the annotation on top of the image #
    small_annotation_img = warp_tools.resize_img(transferred_annotation_img, slide_obj.image.shape[0:2])
    small_annotation_img_np = warp_tools.vips2numpy(small_annotation_img)
    small_img_with_annotation = viz.draw_outline(slide_obj.image, small_annotation_img_np)
    thumbnail_dst_f = os.path.join(annotations_dst_dir, f"{slide_obj.name}_annotated.png")
    warp_tools.save_img(thumbnail_dst_f, small_img_with_annotation)