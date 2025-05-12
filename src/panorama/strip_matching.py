"""
Strip matching algorithms for panorama creation.
"""

import itertools
import numpy as np
import cv2
from src.utils.image_utils import get_bounding_box, resize_to_match_height


def generate_parameter_combinations(match_widths, center_region_percents, y_offset_ranges):
    """
    Generate all combinations of parameters to try for strip matching.
    
    Args:
        match_widths: List of strip widths to try
        center_region_percents: List of center region percentages to try
        y_offset_ranges: List of vertical offset ranges to try
    
    Returns:
        List of parameter combinations as tuples
    """
    return list(itertools.product(match_widths, center_region_percents, y_offset_ranges))


def compute_ssd(strip1, strip2):
    """
    Compute the Sum of Squared Differences between two image strips.
    Only compares non-black pixels (object regions).
    
    Args:
        strip1: First strip
        strip2: Second strip
    
    Returns:
        Normalized SSD (lower is better)
    """
    # ensure the strips have the same shape
    min_height = min(strip1.shape[0], strip2.shape[0])
    min_width = min(strip1.shape[1], strip2.shape[1])
    
    strip1_cropped = strip1[:min_height, :min_width]
    strip2_cropped = strip2[:min_height, :min_width]
    
    # create masks for non-black pixels
    mask1 = (strip1_cropped.sum(axis=2) > 10).astype(np.float32)
    mask2 = (strip2_cropped.sum(axis=2) > 10).astype(np.float32)
    
    # combined mask (where both strips have non-black pixels)
    combined_mask = mask1 * mask2
    
    # count valid pixels
    valid_pixels = np.sum(combined_mask)
    
    if valid_pixels < 10:  # too few pixels to compare
        return float('inf')
    
    # convert to float for calculation
    s1 = strip1_cropped.astype(np.float32)
    s2 = strip2_cropped.astype(np.float32)
    
    # compute SSD only for non-black pixels
    diff = s1 - s2
    ssd = np.sum(diff * diff * combined_mask[:, :, np.newaxis])
    
    # normalize by number of valid pixels
    ssd /= (valid_pixels * 3)  # divide by valid pixels × channels
    
    return ssd


def find_best_match_with_offset(left_strip, next_image, x_pos, match_width, y_offset=0):
    """
    Find the best match with a vertical offset.
    
    Args:
        left_strip: Left edge strip from current panorama
        next_image: Next image to search in
        x_pos: X position to check
        match_width: Width of strip to match
        y_offset: Vertical offset to apply
        
    Returns:
        SSD value for this match
    """
    # check if position is valid
    if x_pos < 0 or x_pos + match_width > next_image.shape[1]:
        return float('inf')
    
    # extract strip from next image with vertical offset
    h, w = next_image.shape[:2]
    
    # create a shifted version of the next_image
    if y_offset == 0:
        shifted_image = next_image
    else:
        shifted_image = np.zeros_like(next_image)
        if y_offset > 0:
            # Shift down
            if h - y_offset > 0: shifted_image[y_offset:, :] = next_image[:h-y_offset, :]
            else: return float('inf') # Shift too large, no overlap
        else:
            # Shift up (y_offset is negative)
            if h + y_offset > 0: shifted_image[:h+y_offset, :] = next_image[-y_offset:, :]
            else: return float('inf') # Shift too large, no overlap
    
    next_strip = shifted_image[:, x_pos:x_pos+match_width]
    
    # skip if strip is too narrow (should be caught by x_pos + match_width check earlier)
    if next_strip.shape[1] < match_width:
        return float('inf')
    
    # compute SSD
    return compute_ssd(left_strip, next_strip)


def find_best_match_in_central_region(prev_strip, next_image, match_width, 
                                    center_region_percent=0.6, y_offset_range=(-5, 5)):
    """
    Find the best matching position for left_strip within the central region of next_image.
    Supports vertical offsets for better matching.
    MODIFIED to use cv2.matchTemplate.
    
    Args:
        prev_strip: Left (the most recent) edge strip from current panorama (this is the TEMPLATE)
        next_image: Next image to search in
        match_width: Width of strip to match (this should be the width of prev_strip)
        center_region_percent: Width of central region to search as fraction of object width
        y_offset_range: Range of vertical offsets to try (min_offset_inclusive, max_offset_inclusive)
        
    Returns:
        best_pos: Best matching position (top-left X in `next_image` after shifting)
        min_ssd: SSD value at best match (from TM_SQDIFF_NORMED, [0,1])
        best_y_offset: Best vertical offset found
        bbox: Object bounding box in original (unshifted) next_image
    """
    # get bounding box of object in next image
    bbox = get_bounding_box(next_image) # This is for the *original* next_image
    obj_x_min_next, _, obj_x_max_next, _ = bbox
    
    # --- OPTIMIZATION: Use cv2.matchTemplate ---
    # Original search loops:
    # # compute center_x, half_width
    # object_width = obj_x_max_next - obj_x_min_next
    # center_x = (obj_x_min_next + obj_x_max_next) // 2
    # half_width = int(object_width * center_region_percent / 2)

    # # search region around center_x    
    # search_start = max(0, center_x - half_width)
    # search_end = min(next_image.shape[1] - match_width, center_x + half_width)
    
    # best_pos = search_start # Default if no better found
    # min_ssd = float('inf')
    # best_y_offset = 0
    
    # if prev_strip.shape[1] != match_width:
    #     if prev_strip.shape[1] > match_width:
    #         prev_strip = prev_strip[:, :match_width]
    #     # else: # prev_strip is narrower, this is problematic for fixed match_width
    #         # print(f"Warning: prev_strip width {prev_strip.shape[1]} < match_width {match_width}")
    #         # return best_pos, min_ssd, best_y_offset, bbox # Or error

    # # try different vertical offsets
    # y_offsets = range(y_offset_range[0], y_offset_range[1] + 1)
    
    # # search for best match in central region with different y-offsets
    # for x in range(search_start, search_end + 1, 2): # note step=2 for faster processing
    #     for y_offset_val_loop in y_offsets: # Renamed to avoid conflict
    #         ssd = find_best_match_with_offset(prev_strip, next_image, x, match_width, y_offset_val_loop)
            
    #         if ssd < min_ssd:
    #             min_ssd = ssd
    #             best_pos = x
    #             best_y_offset = y_offset_val_loop
    
    # # fine-tune search around best position
    # fine_start = max(search_start, best_pos - 2)
    # fine_end = min(search_end, best_pos + 2)
    
    # for x in range(fine_start, fine_end + 1):
    #     for y_offset_val_loop in y_offsets: # Renamed to avoid conflict
    #         ssd = find_best_match_with_offset(prev_strip, next_image, x, match_width, y_offset_val_loop)
            
    #         if ssd < min_ssd:
    #             min_ssd = ssd
    #             best_pos = x
    #             best_y_offset = y_offset_val_loop
    # return best_pos, min_ssd, best_y_offset, bbox

    # OPTIMIZED version:
    h_next, w_next = next_image.shape[:2]
    
    # Template is prev_strip. Ensure its width is `match_width`.
    # The `match_width` argument should ideally be derived from `prev_strip.shape[1]`.
    # For this optimization, we assume `prev_strip` is the template and its width is what we use.
    template_actual_width = prev_strip.shape[1]
    if template_actual_width == 0 or prev_strip.shape[0] == 0 : # Empty template
        return 0, float('inf'), 0, bbox 
    
    template_cv = prev_strip.astype(np.float32) # For cv2.matchTemplate

    best_overall_ssd = float('inf')
    best_overall_pos_x = 0 
    best_overall_y_offset = 0

    # Define search ROI in the original next_image based on center_region_percent
    object_width_next = obj_x_max_next - obj_x_min_next
    search_center_x_next = (obj_x_min_next + obj_x_max_next) // 2
    
    # ROI half-width for searching. This should accommodate the template.
    roi_half_search_width = int(object_width_next * center_region_percent / 2) + template_actual_width // 2
    
    # Define the search window for the *top-left corner* of the template
    search_x_min_in_next = max(0, search_center_x_next - roi_half_search_width)
    search_x_max_in_next = min(w_next - template_actual_width, search_center_x_next + roi_half_search_width)

    if search_x_min_in_next > search_x_max_in_next : # No valid search window
        # This can happen if object is very small or center_region_percent is too restrictive.
        # Fallback: search a wider area or full image width.
        search_x_min_in_next = 0
        search_x_max_in_next = w_next - template_actual_width
        if search_x_min_in_next > search_x_max_in_next: # Still no valid window (image narrower than template)
            return 0, float('inf'), 0, bbox


    y_offsets_to_try = range(y_offset_range[0], y_offset_range[1] + 1)

    for y_offset_val_loop in y_offsets_to_try:
        shifted_image_cv = np.zeros_like(next_image)
        if y_offset_val_loop == 0:
            shifted_image_cv = next_image.copy()
        else:
            if y_offset_val_loop > 0:
                if h_next - y_offset_val_loop > 0: shifted_image_cv[y_offset_val_loop:, :] = next_image[:h_next - y_offset_val_loop, :]
                else: continue # Shift too large
            else: # y_offset_val_loop < 0
                if h_next + y_offset_val_loop > 0: shifted_image_cv[:h_next + y_offset_val_loop, :] = next_image[-y_offset_val_loop:, :]
                else: continue # Shift too large
        
        img_to_search_cv = shifted_image_cv.astype(np.float32)

        # Extract the ROI from img_to_search_cv
        # The ROI for matchTemplate should be at least as large as the template.
        # search_x_max_in_next is already adjusted for template_actual_width.
        roi_to_search_in = img_to_search_cv[:, search_x_min_in_next : search_x_max_in_next + template_actual_width]
        
        if template_cv.shape[0] > roi_to_search_in.shape[0] or \
           template_cv.shape[1] > roi_to_search_in.shape[1]:
            # If template is larger than ROI (e.g., due to y_offset shrinking effective height, or ROI too narrow)
            # Fallback to searching the whole shifted image for this y_offset
            roi_to_search_in = img_to_search_cv
            current_search_x_offset = 0 # Match location will be relative to full image
            if template_cv.shape[0] > roi_to_search_in.shape[0] or \
               template_cv.shape[1] > roi_to_search_in.shape[1]:
                continue # Template still too large for the entire shifted image

        else:
            current_search_x_offset = search_x_min_in_next # Match location will be relative to ROI, add this back

        try:
            # Use TM_SQDIFF_NORMED: output is [0,1], lower is better.
            res_map = cv2.matchTemplate(roi_to_search_in, template_cv, cv2.TM_SQDIFF_NORMED)
            min_val_match, _, min_loc_match, _ = cv2.minMaxLoc(res_map)
            
            # min_loc_match[0] is the x-position within `roi_to_search_in`
            # Convert to x-position in the original `next_image` coordinates (before ROI extraction)
            current_match_pos_x = current_search_x_offset + min_loc_match[0]
            current_match_ssd = min_val_match # This is the normalized SSD

            if current_match_ssd < best_overall_ssd:
                best_overall_ssd = current_match_ssd
                best_overall_pos_x = current_match_pos_x
                best_overall_y_offset = y_offset_val_loop
        except cv2.error:
            # print(f"cv2.matchTemplate failed for y_offset {y_offset_val_loop}")
            continue # Error with this y_offset, try next
            
    return best_overall_pos_x, best_overall_ssd, best_overall_y_offset, bbox


def find_best_match_with_parameters(current_pano, next_img, param_combinations):
    """
    Try multiple parameter combinations to find the best match.
    
    Args:
        current_pano: Current panorama
        next_img: Next image to stitch
        param_combinations: List of parameter combinations to try
            Each combination is a tuple (match_width, center_region_percent, y_offset_range_tuple)
            
    Returns:
        best_params: Best parameters (match_width, center_region_percent, y_offset_range_tuple)
        best_results: Best results (best_pos, min_ssd, best_y_offset, bbox_of_original_next_img)
    """
    best_ssd = float('inf')
    best_params = None
    best_results = None

    if next_img is None or next_img.shape[0] == 0 or next_img.shape[1] == 0:
        return (param_combinations[0] if param_combinations else (0,0,(0,0))), \
               (0, float('inf'), 0, [0,0,0,0] if next_img is None else get_bounding_box(next_img))
    
    for match_width_param, center_region_percent_param, y_offset_range_param_tuple in param_combinations:
        # Extract LEFT edge strip from current panorama for this match_width_param
        # This `prev_strip` is the template.
        if current_pano is None or current_pano.shape[1] < match_width_param or match_width_param <= 0:
            continue # Invalid pano or match_width for this combination

        prev_strip = current_pano[:, :match_width_param]
        
        # Find best match with these parameters
        # `match_width_param` here defines the width of `prev_strip` to be used as template
        best_pos, ssd, best_y_offset, bbox_original_next = find_best_match_in_central_region(
            prev_strip, # This is the template
            next_img,   # Image to search in
            match_width_param, # This should match prev_strip.shape[1] implicitly by the optimized func
            center_region_percent_param,
            y_offset_range_param_tuple # This is already a tuple like (-5,5)
        )
        
        # Update if better match found
        if ssd < best_ssd:
            best_ssd = ssd
            best_params = (match_width_param, center_region_percent_param, y_offset_range_param_tuple)
            best_results = (best_pos, ssd, best_y_offset, bbox_original_next) # bbox is for original next_img
    
    if best_results is None: # No combination yielded a valid result
        fallback_bbox = get_bounding_box(next_img)
        return (param_combinations[0] if param_combinations else (0,0,(0,0))), \
               (0, float('inf'), 0, fallback_bbox)

    return best_params, best_results


def evaluate_pair_quality(images, match_width=30, center_region_percent=0.15, y_offset_range=(-5, 5)):
    """
    Evaluate the quality of each image pair by computing the best SSD match.
    Used to identify problematic images that might need to be skipped.
    
    Args:
        images: List of input images
        match_width: Width of strip to match (template width)
        center_region_percent: Width of central region to search
        y_offset_range: Range of vertical offsets to try (min_offset_inclusive, max_offset_inclusive)
    
    Returns:
        List of dictionaries with quality metrics for each pair
    """
    if len(images) < 2:
        return []
    
    # ensure all images have the same height for consistent strip extraction if needed by evaluation
    common_height = min(img.shape[0] for img in images)
    resized_images = [resize_to_match_height(img, common_height) for img in images]
    
    match_qualities = []
    
    # evaluate each consecutive image pair
    for i in range(len(resized_images) - 1):
        img1 = resized_images[i]   # Source of template (right_strip)
        img2 = resized_images[i+1] # Image to search in

        # Extract right edge strip from first image (img1) to use as template
        # Ensure the strip width is `match_width`
        if img1.shape[1] < match_width or match_width <= 0: # img1 too narrow for template
            # print(f"Warning: img1 too narrow for template in evaluate_pair_quality for pair {i},{i+1}")
            match_qualities.append({'pair': (i, i+1), 'ssd': float('inf'), 'y_offset': 0, 'best_pos': 0})
            continue

        # Take strip from the absolute right edge of img1
        right_strip_template = img1[:, -match_width:]
        
        # Find best match of `right_strip_template` in `img2`
        # The `match_width` passed to find_best_match_in_central_region should be `right_strip_template.shape[1]`
        best_pos, ssd, best_y_offset, _ = find_best_match_in_central_region(
            right_strip_template, # Template
            img2,                 # Image to search
            right_strip_template.shape[1], # Actual width of the template being used
            center_region_percent,
            y_offset_range
        )
        
        match_qualities.append({
            'pair': (i, i+1),
            'ssd': ssd, # This will be the TM_SQDIFF_NORMED value
            'y_offset': best_y_offset,
            'best_pos': best_pos
        })
    
    return match_qualities