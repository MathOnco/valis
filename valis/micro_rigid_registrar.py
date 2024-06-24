import numpy as np
from skimage import exposure, transform
import multiprocessing
from colorama import Fore
from contextlib import suppress

from . import feature_matcher
from . import feature_detectors
from . import preprocessing
from . import warp_tools
from . import valtils

from pqdm.threads import pqdm

ROI_MASK = "mask"
ROI_MATCHES = "matches"

DEFAULT_ROI = ROI_MASK

DEFAULT_FD = feature_detectors.VggFD
DEFAULT_MATCHER = feature_matcher.Matcher

DEFAULT_BF_PROCESSOR = preprocessing.OD
DEFAULT_BF_PROCESSOR_KWARGS = {}

DEFAULT_FLOURESCENCE_CLASS = preprocessing.ChannelGetter
DEFAULT_FLOURESCENCE_PROCESSING_ARGS = {"channel": "dapi", "adaptive_eq": True}

class MicroRigidRegistrar(object):
    """Refine rigid registration using higher resolution images

    Rigid transforms found during lower resolution images are applied to the
    WSI and then downsampled. The higher resolution registered images are then
    divided into tiles, which are processed and normalized. Next, features are
    detected and matched for each tile, the results of which are combined into
    a common keypoint list. These higher resolution keypoints are then used to
    estimate a new rigid transform. Replaces thumbnails in the
    rigid registration folder.

    Attributes
    ----------
    val_obj : Valis
        The "parent" object that registers all of the slides.

    feature_detector_cls : FeatureDD, optional
        Uninstantiated FeatureDD object that detects and computes
        image features. Default is SuperPointFD. The
        available feature_detectors are found in the `feature_detectors`
        module. If a desired feature detector is not available,
        one can be created by subclassing `feature_detectors.FeatureDD`.

    matcher : Matcher
        Matcher object that will be used to match image features

    scale : float
        Degree of downsampling to use for the reigistration, based on the
        registered WSI shape (i.e. Slide.aligned_slide_shape_rc)

    tile_wh : int
        Width and height of tiles extracted from registered WSI

    roi : string
        Determines how the region of interest is defined. `roi="mask"` will
        use the bounding box of non-rigid registration mask to define the search area.
        `roi=matches` will use the bounding box of the previously matched features to
        define the search area.

    iter_order : list of tuples
        Determines the order in which images are aligned. Goes from reference image to
        the edges of the stack.

    """

    def __init__(self, val_obj, feature_detector_cls=DEFAULT_FD,
                 matcher=DEFAULT_MATCHER, processor_dict=None,
                 scale=0.5**3, tile_wh=2**9, roi=DEFAULT_ROI):
        """

        Parameters
        ----------
        val_obj : Valis
            The "parent" object that registers all of the slides.

        feature_detector_cls : FeatureDD, optional
            Uninstantiated FeatureDD object that detects and computes
            image features. Default is SuperPointFD. The
            available feature_detectors are found in the `feature_detectors`
            module. If a desired feature detector is not available,
            one can be created by subclassing `feature_detectors.FeatureDD`.

        matcher : Matcher
            Matcher object that will be used to match image features

        processor_dict : dict, optional
            Each key should be the filename of the image, and the value either a subclassed
            preprocessing.ImageProcessor, or a list, where the 1st element is the processor,
            and the second element a dictionary of keyword arguments passed to the processor.
            If `None`, a default processor will be assigned to each image based on its modality.

        scale : float
            Degree of downsampling to use for the reigistration, based on the
            registered WSI shape (i.e. Slide.aligned_slide_shape_rc)

        tile_wh : int
            Width and height of tiles extracted from registered WSI

        roi : string
            Determines how the region of interest is defined. `roi="mask"` will
            use the bounding box of non-rigid registration mask to define the search area.
            `roi=matches` will use the bounding box around the matching features, which may
            be smaller than the registration mask.

        """

        self.val_obj = val_obj
        self.feature_detector_cls = feature_detector_cls
        self.matcher = matcher
        self.processor_dict = processor_dict
        self.scale = scale
        self.tile_wh = tile_wh
        self.roi = roi
        self.iter_order = warp_tools.get_alignment_indices(val_obj.size, val_obj.reference_img_idx)

    def create_mask(self, moving_slide, fixed_slide):
        """Create mask used to define bounding box of search area

        """

        pair_slide_list = [moving_slide, fixed_slide]
        if self.val_obj.create_masks:
            temp_mask = self.val_obj._create_mask_from_processed(slide_list=pair_slide_list)
            if temp_mask.max() == 0:
                temp_mask = self.val_obj._create_non_rigid_reg_mask_from_bbox(slide_list=pair_slide_list)
        else:
            temp_mask = self.val_obj._create_non_rigid_reg_mask_from_bbox(slide_list=pair_slide_list)

        fixed_bbox = np.full(fixed_slide.processed_img_shape_rc, 255, dtype=np.uint8)
        fixed_mask = fixed_slide.warp_img(fixed_bbox, non_rigid=False, crop=False, interp_method="nearest")

        mask = preprocessing.combine_masks(temp_mask, fixed_mask, op="and")

        return mask

    def register(self,
                 brightfield_processing_cls=DEFAULT_BF_PROCESSOR, brightfield_processing_kwargs=DEFAULT_BF_PROCESSOR_KWARGS,
                 if_processing_cls=DEFAULT_FLOURESCENCE_CLASS, if_processing_kwargs=DEFAULT_FLOURESCENCE_PROCESSING_ARGS):
        """

        Parameters
        ----------
        brightfield_processing_cls : ImageProcesser
            ImageProcesser to pre-process brightfield images to make them look as similar as possible.
            Should return a single channel uint8 image.

        brightfield_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `brightfield_processing_cls`

        if_processing_cls : ImageProcesser
            ImageProcesser to pre-process immunofluorescent images to make them look as similar as possible.
            Should return a single channel uint8 image.

        if_processing_kwargs : dict
            Dictionary of keyward arguments to be passed to `if_processing_cls`

        """

        processor_dict = self.val_obj.create_img_processor_dict(brightfield_processing_cls=brightfield_processing_cls,
                                brightfield_processing_kwargs=brightfield_processing_kwargs,
                                if_processing_cls=if_processing_cls,
                                if_processing_kwargs=if_processing_kwargs,
                                processor_dict=self.processor_dict)

        # Get slides in correct order
        slide_idx, slide_names = list(zip(*[[slide_obj.stack_idx, slide_obj.name] for slide_obj in self.val_obj.slide_dict.values()]))
        slide_order = np.argsort(slide_idx) # sorts ascending
        slide_list = [self.val_obj.slide_dict[slide_names[i]] for i in slide_order]

        for moving_idx, fixed_idx in self.iter_order:
            moving_slide = slide_list[moving_idx]
            fixed_slide = slide_list[fixed_idx]

            assert moving_slide.fixed_slide == fixed_slide

            mask = self.create_mask(moving_slide, fixed_slide)



            self.align_slides(moving_slide, fixed_slide, processor_dict=processor_dict, mask=mask)

    def align_slides(self, moving_slide, fixed_slide, processor_dict, mask=None):
        moving_img = moving_slide.warp_slide(level=0, non_rigid=False, crop=False)
        moving_img = warp_tools.rescale_img(moving_img, self.scale)

        moving_shape_rc = warp_tools.get_shape(moving_img)[0:2]
        moving_sxy = (moving_shape_rc/moving_slide.reg_img_shape_rc)[::-1]

        fixed_img = fixed_slide.warp_slide(0, non_rigid=False, crop=False)
        fixed_img = warp_tools.rescale_img(fixed_img, self.scale)

        fixed_shape_rc = warp_tools.get_shape(fixed_img)[0:2]
        fixed_sxy = (fixed_shape_rc/fixed_slide.reg_img_shape_rc)[::-1]

        # Perform Rigid registration where masks overlap
        aligned_slide_shape_rc = warp_tools.get_shape(moving_img)[0:2]
        if self.roi == ROI_MASK:
            small_reg_bbox = warp_tools.mask2xy(mask)
        elif self.roi == ROI_MATCHES:
            reg_moving_xy = warp_tools.warp_xy(moving_slide.xy_matched_to_prev, moving_slide.M)
            reg_fixed_xy = warp_tools.warp_xy(moving_slide.xy_in_prev, fixed_slide.M)
            small_reg_bbox = np.vstack([reg_moving_xy, reg_fixed_xy])

        reg_s = (aligned_slide_shape_rc/np.array(mask.shape))[::-1]
        reg_bbox = warp_tools.xy2bbox(small_reg_bbox*reg_s)
        slide_mask = warp_tools.resize_img(warp_tools.numpy2vips(mask), warp_tools.get_shape(fixed_img)[0:2], interp_method="nearest")

        # Collect high rez matches
        bbox_tiles = self.get_tiles(reg_bbox, self.tile_wh)
        n_tiles = len(bbox_tiles)
        high_rez_moving_match_xy_list = [None]*n_tiles
        high_rez_fixed_match_xy_list = [None]*n_tiles

        moving_processing_cls, moving_processing_kwargs = processor_dict[moving_slide.name]
        fixed_processing_cls, fixed_processing_kwargs = processor_dict[fixed_slide.name]

        def _match_tile(bbox_id):
            bbox_xy = bbox_tiles[bbox_id]

            matcher = self.matcher()
            fd = self.feature_detector_cls()

            region_xywh = warp_tools.xy2bbox(bbox_xy)
            region_mask = slide_mask.extract_area(*region_xywh)
            if region_mask.max() == 0:

                return None

            moving_region, moving_processed, moving_bbox_xywh = self.process_roi(img=moving_img,
                                                                            slide_obj=moving_slide,
                                                                            xy=bbox_xy,
                                                                            processor_cls=moving_processing_cls,
                                                                            processor_kwargs=moving_processing_kwargs,
                                                                            apply_mask=False,
                                                                            scale=1.0
                                                                            )

            fixed_region, fixed_processed, fixed_bbox_xywh = self.process_roi(img=fixed_img,
                                                                            slide_obj=fixed_slide,
                                                                            xy=bbox_xy,
                                                                            processor_cls=fixed_processing_cls,
                                                                            processor_kwargs=fixed_processing_kwargs,
                                                                            apply_mask=False,
                                                                            scale=1.0
                                                                            )

            moving_normed, fixed_normed = self.norm_imgs(img_list=[moving_processed, fixed_processed])

            try:
                if hasattr(matcher, "kp_detector_name"):
                    # Matcher ( e.g. SuperPointAndGlue) can both detect and describe keypoints
                    _, filtered_match_info12, _, _ = matcher.match_images(img1=moving_normed, img2=fixed_normed)

                else:

                    moving_kp, moving_desc = fd.detect_and_compute(moving_normed)
                    fixed_kp, fixed_desc = fd.detect_and_compute(fixed_normed)

                    _, filtered_match_info12, _, _ = matcher.match_images(img1=moving_normed, desc1=moving_desc, kp1_xy=moving_kp,
                                                                          img2=fixed_normed,  desc2=fixed_desc,  kp2_xy=fixed_kp)

                filtered_matched_moving_xy = filtered_match_info12.matched_kp1_xy
                filtered_matched_fixed_xy = filtered_match_info12.matched_kp2_xy
                matched_moving_desc = filtered_match_info12.matched_desc1
                matched_fixed_desc = filtered_match_info12.matched_desc2

                if filtered_matched_moving_xy.shape[0] < 3:
                    return None

                filtered_matched_moving_xy, filtered_matched_fixed_xy, tukey_idx = feature_matcher.filter_matches_tukey(filtered_matched_moving_xy, filtered_matched_fixed_xy, tform=transform.EuclideanTransform())
                matched_moving_desc = matched_moving_desc[tukey_idx, :]
                matched_fixed_desc = matched_fixed_desc[tukey_idx, :]
                if filtered_matched_moving_xy.shape[0] < 3:
                    return None

            except Exception as e:
                return None

            matched_moving_xy = filtered_matched_moving_xy.copy()
            matched_fixed_xy = filtered_matched_fixed_xy.copy()

            # Add ROI offset to matched points
            matched_moving_xy += moving_bbox_xywh[0:2]
            matched_fixed_xy += fixed_bbox_xywh[0:2]

            high_rez_moving_match_xy_list[bbox_id] = matched_moving_xy
            high_rez_fixed_match_xy_list[bbox_id] = matched_fixed_xy

        print(f"Aligning {moving_slide.name} to {fixed_slide.name}. ROI width, height is {reg_bbox[2:]} pixels")
        n_cpu = valtils.get_ncpus_available() - 1

        with suppress(UserWarning):
            # Avoid printing warnings that not enough matches were found, which can happen frequently with this
            res = pqdm(range(n_tiles), _match_tile, n_jobs=n_cpu)

        # Remove tiles that didn't have any matches
        high_rez_moving_match_xy_list = [xy for xy in high_rez_moving_match_xy_list if xy is not None]
        high_rez_fixed_match_xy_list = [xy for xy in high_rez_fixed_match_xy_list if xy is not None]

        high_rez_moving_match_xy = np.vstack(high_rez_moving_match_xy_list)
        high_rez_fixed_match_xy = np.vstack(high_rez_fixed_match_xy_list)

        temp_high_rez_moving_matched_kp_xy, temp_high_rez_fixed_matched_kp_xy, ransac_idx = feature_matcher.filter_matches_ransac(high_rez_moving_match_xy, high_rez_fixed_match_xy, 20)
        high_rez_moving_matched_kp_xy, high_rez_fixed_matched_kp_xy, tukey_idx = feature_matcher.filter_matches_tukey(temp_high_rez_moving_matched_kp_xy, temp_high_rez_fixed_matched_kp_xy, tform=transform.EuclideanTransform())

        scaled_moving_kp = high_rez_moving_matched_kp_xy*(1/moving_sxy)
        scaled_fixed_kp = high_rez_fixed_matched_kp_xy*(1/fixed_sxy)

        if self.val_obj.create_masks:
            moving_kp_in_og = warp_tools.warp_xy(scaled_moving_kp, M=np.linalg.inv(moving_slide.M))
            moving_features_in_mask_idx = warp_tools.get_xy_inside_mask(xy=moving_kp_in_og, mask=moving_slide.rigid_reg_mask)

            fixed_kp_in_og = warp_tools.warp_xy(scaled_fixed_kp, M=np.linalg.inv(fixed_slide.M))
            fixed_features_in_mask_idx = warp_tools.get_xy_inside_mask(xy=fixed_kp_in_og, mask=fixed_slide.rigid_reg_mask)

            if len(moving_features_in_mask_idx) > 0 and len(fixed_features_in_mask_idx) > 0:
                matches_in_masks = np.intersect1d(moving_features_in_mask_idx, fixed_features_in_mask_idx)
                if len(matches_in_masks) > 0:
                    scaled_moving_kp = scaled_moving_kp[matches_in_masks, :]
                    scaled_fixed_kp = scaled_fixed_kp[matches_in_masks, :]

                    high_rez_moving_matched_kp_xy = high_rez_moving_matched_kp_xy[matches_in_masks, :]
                    high_rez_fixed_matched_kp_xy = high_rez_fixed_matched_kp_xy[matches_in_masks, :]

        # Estimate M using position in larger image
        transformer = transform.SimilarityTransform()
        transformer.estimate(high_rez_fixed_matched_kp_xy, high_rez_moving_matched_kp_xy)
        M = transformer.params

        # Scale for use on original processed image
        slide_corners_xy = warp_tools.get_corners_of_image(moving_shape_rc)[::-1]
        warped_slide_corners = warp_tools.warp_xy(slide_corners_xy, M=M,
                                    transformation_src_shape_rc=moving_shape_rc,
                                    transformation_dst_shape_rc=fixed_shape_rc,
                                    src_shape_rc=moving_slide.reg_img_shape_rc,
                                    dst_shape_rc=fixed_slide.reg_img_shape_rc)

        M_tform = transform.ProjectiveTransform()
        M_tform.estimate(warped_slide_corners, slide_corners_xy)
        scaled_M = M_tform.params

        new_M = moving_slide.M @ scaled_M

        matched_moving_in_og = warp_tools.warp_xy(scaled_moving_kp, M=np.linalg.inv(moving_slide.M))
        matched_fixed_in_og = warp_tools.warp_xy(scaled_fixed_kp, M=np.linalg.inv(fixed_slide.M))

        og_d = np.mean(warp_tools.calc_d(warp_tools.warp_xy(moving_slide.xy_matched_to_prev, M=moving_slide.M), warp_tools.warp_xy(moving_slide.xy_in_prev, fixed_slide.M)))
        new_d = np.mean(warp_tools.calc_d(warp_tools.warp_xy(matched_moving_in_og, M=new_M), warp_tools.warp_xy(matched_fixed_in_og, fixed_slide.M)))

        n_old_matches = moving_slide.xy_matched_to_prev.shape[0]
        n_new_matches = high_rez_fixed_matched_kp_xy.shape[0]

        improved = (n_new_matches >= n_old_matches)
        if improved:
            res_msg = "micro rigid registration improved alignments."
            msg_clr = Fore.GREEN
        else:
            res_msg = "micro rigid registration did not improve alignments. Keeping low rez registration parameters."
            msg_clr = Fore.YELLOW

        full_res_msg = f"{res_msg} N low rez matches= {n_old_matches}, N high rez matches = {n_new_matches}. Low rez D= {og_d}, high rez D={new_d}"
        valtils.print_warning(full_res_msg, rgb=msg_clr)
        if improved:

            moving_slide.M = new_M
            moving_slide.xy_matched_to_prev = matched_moving_in_og
            moving_slide.xy_in_prev = matched_fixed_in_og

            moving_slide.xy_matched_to_prev_in_bbox = matched_moving_in_og
            moving_slide.xy_in_prev_in_bbox = matched_fixed_in_og


    def get_tiles(self, bbox_xywh, wh):

        x_step = np.min([wh, np.floor(bbox_xywh[2]).astype(int)])
        y_step = np.min([wh, np.floor(bbox_xywh[3]).astype(int)])

        x_pos = np.arange(bbox_xywh[0], bbox_xywh[0]+bbox_xywh[2], x_step)
        max_x, max_y = np.round(bbox_xywh[0:2] + bbox_xywh[2:]).astype(int)
        if x_pos[-1] < max_x - 1:
            x_pos = np.array([*x_pos, max_x])

        y_pos = np.arange(bbox_xywh[1], bbox_xywh[1]+bbox_xywh[3], y_step)
        if y_pos[-1] < max_y - 1:
            y_pos = np.array([*y_pos, max_y])

        tile_bbox_list = [np.array([[x_pos[i], y_pos[j]], [x_pos[i+1], y_pos[j+1]]]) for j in range(len(y_pos) - 1) for i in range(len(x_pos) - 1)]

        return tile_bbox_list

    def norm_imgs(self, img_list):
        _, target_processing_stats = preprocessing.collect_img_stats(img_list)

        normed_list = [None] * len(img_list)
        for i, img in enumerate(img_list):
            try:
                processed = preprocessing.norm_img_stats(img, target_processing_stats)
            except ValueError:
                processed = img

            normed_list[i] = exposure.rescale_intensity(processed, out_range=(0, 255)).astype(np.uint8)

        return normed_list

    def process_roi(self, img, slide_obj, xy, processor_cls, processor_kwargs, apply_mask=True, scale=0.5):
        is_array = isinstance(img, np.ndarray)
        if is_array:
            vips_img = warp_tools.numpy2vips(img)
        else:
            vips_img = img

        bbox = warp_tools.xy2bbox(xy)
        bbox_wh = np.ceil(bbox[2:]).astype(int)
        region = vips_img.extract_area(*bbox[0:2], *bbox_wh)

        if scale != 1.0:
            region = warp_tools.rescale_img(region, scale)

        region_np = warp_tools.vips2numpy(region)

        processor = processor_cls(region_np, src_f=slide_obj.src_f, level=0, series=slide_obj.series, reader=slide_obj.reader)
        processed_img = processor.process_image(**processor_kwargs)

        if apply_mask:
            mask = processor.create_mask()
            processed_img[mask == 0] = 0

        return region_np, processed_img, bbox
