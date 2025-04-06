"""
Strip matching algorithms for panorama creation.
"""

import itertools
import numpy as np
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
            shifted_image[y_offset:, :] = next_image[:h-y_offset, :]
        else:
            # Shift up (y_offset is negative)
            shifted_image[:h+y_offset, :] = next_image[-y_offset:, :]
    
    next_strip = shifted_image[:, x_pos:x_pos+match_width]
    
    # skip if strip is too narrow
    if next_strip.shape[1] < match_width:
        return float('inf')
    
    # compute SSD
    return compute_ssd(left_strip, next_strip)


def find_best_match_in_central_region(prev_strip, next_image, match_width, 
                                    center_region_percent=0.6, y_offset_range=(-5, 5)):
    """
    Find the best matching position for left_strip within the central region of next_image.
    Supports vertical offsets for better matching.
    
    Args:
        prev_strip: Left (the most recent) edge strip from current panorama
        next_image: Next image to search in
        match_width: Width of strip to match
        center_region_percent: Width of central region to search as fraction of object width
        y_offset_range: Range of vertical offsets to try (min, max)
        
    Returns:
        best_pos: Best matching position
        min_ssd: SSD value at best match
        best_y_offset: Best vertical offset
        bbox: Object bounding box in next image
    """
    # get bounding box of object in next image
    bbox = get_bounding_box(next_image)
    x_min, _, x_max, _ = bbox
    
    # compute center_x, half_width
    object_width = x_max - x_min
    center_x = (x_min + x_max) // 2
    half_width = int(object_width * center_region_percent / 2)

    # search region around center_x    
    search_start = max(0, center_x - half_width)
    search_end = min(next_image.shape[1] - match_width, center_x + half_width)
    
    best_pos = search_start
    min_ssd = float('inf')
    best_y_offset = 0
    
    if prev_strip.shape[1] != match_width:
        # If it's larger, crop it / if it's smaller, this might not work well...
        if prev_strip.shape[1] > match_width:
            prev_strip = prev_strip[:, :match_width]
    
    # try different vertical offsets
    y_offsets = range(y_offset_range[0], y_offset_range[1] + 1)
    
    # search for best match in central region with different y-offsets
    for x in range(search_start, search_end + 1, 2): # note step=2 for faster processing
        for y_offset in y_offsets:
            ssd = find_best_match_with_offset(prev_strip, next_image, x, match_width, y_offset)
            
            if ssd < min_ssd:
                min_ssd = ssd
                best_pos = x
                best_y_offset = y_offset
    
    # fine-tune search around best position
    fine_start = max(search_start, best_pos - 2)
    fine_end = min(search_end, best_pos + 2)
    
    for x in range(fine_start, fine_end + 1):
        for y_offset in y_offsets:
            ssd = find_best_match_with_offset(prev_strip, next_image, x, match_width, y_offset)
            
            if ssd < min_ssd:
                min_ssd = ssd
                best_pos = x
                best_y_offset = y_offset
    
    return best_pos, min_ssd, best_y_offset, bbox


def find_best_match_with_parameters(current_pano, next_img, param_combinations):
    """
    Try multiple parameter combinations to find the best match.
    
    Args:
        current_pano: Current panorama
        next_img: Next image to stitch
        param_combinations: List of parameter combinations to try
            Each combination is a tuple (match_width, center_region_percent, y_offset_range)
            
    Returns:
        best_params: Best parameters (match_width, center_region_percent, y_offset_range)
        best_results: Best results (best_pos, min_ssd, best_y_offset, bbox)
    """
    best_ssd = float('inf')
    best_params = None
    best_results = None
    
    for match_width, center_region_percent, y_offset_range in param_combinations:
        # Extract LEFT edge strip from current panorama for this match_width
        prev_strip = current_pano[:, :match_width]
        
        # Find best match with these parameters
        best_pos, ssd, best_y_offset, bbox = find_best_match_in_central_region(
            prev_strip,
            next_img,
            match_width,
            center_region_percent,
            y_offset_range
        )
        
        # Update if better match found
        if ssd < best_ssd:
            best_ssd = ssd
            best_params = (match_width, center_region_percent, y_offset_range)
            best_results = (best_pos, ssd, best_y_offset, bbox)
    
    return best_params, best_results


def evaluate_pair_quality(images, match_width=30, center_region_percent=0.15, y_offset_range=(-5, 5)):
    """
    Evaluate the quality of each image pair by computing the best SSD match.
    Used to identify problematic images that might need to be skipped.
    
    Args:
        images: List of input images
        match_width: Width of strip to match
        center_region_percent: Width of central region to search
        y_offset_range: Range of vertical offsets to try (min, max)
    
    Returns:
        List of dictionaries with quality metrics for each pair
    """
    if len(images) < 2:
        return []
    
    # ensure all images have the same height
    common_height = min(img.shape[0] for img in images)
    resized_images = [resize_to_match_height(img, common_height) for img in images]
    
    match_qualities = []
    
    # evaluate each consecutive image pair
    for i in range(len(resized_images) - 1):
        img1 = resized_images[i]
        img2 = resized_images[i + 1]
        
        # Extract right edge strip from first image
        bbox = get_bounding_box(img1)
        x_min, _, x_max, _ = bbox
        
        strip_pos = max(0, x_max - match_width)
        right_strip = img1[:, strip_pos:strip_pos+match_width]
        
        # Find best match with second image
        best_pos, ssd, best_y_offset, _ = find_best_match_in_central_region(
            right_strip, img2, match_width, center_region_percent, y_offset_range
        )
        
        # Store quality metrics
        match_qualities.append({
            'pair': (i, i+1),
            'ssd': ssd,
            'y_offset': best_y_offset,
            'best_pos': best_pos
        })
    
    return match_qualities
