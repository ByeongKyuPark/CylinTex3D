"""
Panorama stitching module for creating complete panoramas from segmented images.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

from src.utils.image_utils import load_images_from_folder, get_bounding_box, resize_to_match_height
from src.utils.visualization import visualize_matching, save_panorama_result
from src.panorama.strip_matching import find_best_match_with_parameters, evaluate_pair_quality

# internal function to create a panorama , which is called by the wrapper function.
def create_panorama_internal(images, param_combinations, adjacent_strip_width=30, blend_width=5):
    """
    create a panorama from a list of images using FLEXISBLE parameter matching.
    
    Args:
        images: List of input images
        param_combinations: List of parameter combinations to try
        adjacent_strip_width: Width of strips to add to panorama
        blend_width: Width of blending region
        
    Returns:
        Tuple of (panorama image, list of parameters used for each image)
    """
    if len(images) < 2:
        return images[0] if images else None, []
    
    print("Creating panorama...")
    
    # resize to match the smallest height
    common_height = min(img.shape[0] for img in images)
    resized_images = [resize_to_match_height(img, common_height) for img in images]
    
    # get the center strip from the first image
    bbox = get_bounding_box(resized_images[0])
    x_min, y_min, x_max, y_max = bbox
    
    center_x = (x_min + x_max) // 2
    
    start_x = max(0, center_x - adjacent_strip_width // 2)
    center_strip = resized_images[0][:, start_x:start_x+adjacent_strip_width]
    
    # panorama with the center strip
    panorama = center_strip.copy()
    
    # this will be useful for visualization
    parameters_used = []
    
    for i in tqdm(range(1, len(resized_images)), desc="Building panorama"):
        print(f"\nProcessing image {i}/{len(resized_images)-1}")
        
        # curr pano and next image
        current_pano = panorama.copy()
        next_img = resized_images[i]
        
        try:
            best_params, best_results = find_best_match_with_parameters(
                current_pano,
                next_img,
                param_combinations
            )
            
            match_width, center_region_percent, y_offset_range = best_params
            best_x_pos, ssd, best_y_offset, next_bbox = best_results
            
            print(f"Image {i}: Best match with parameters:")
            print(f"  Match width: {match_width}, Center region: {center_region_percent}, Y-offset: {best_y_offset}")
            print(f"  Position: {best_x_pos}, SSD: {ssd:.6f}")
            
            # append parameters just used
            parameters_used.append({
                'index': i,
                'match_width': match_width,
                'center_percent': center_region_percent,
                'y_offset': best_y_offset,
                'position': best_x_pos,
                'ssd': ssd
            })
            
            adjacent_pos = max(0, best_x_pos - adjacent_strip_width)
            
            original_width = adjacent_strip_width
            if adjacent_pos + adjacent_strip_width > best_x_pos:
                adjusted_width = best_x_pos - adjacent_pos
                
                if adjusted_width <= 0:
                    print(f"  Warning: No space for adjacent strip in image {i}, skipping")
                    continue
                adjacent_strip_width = adjusted_width
            
            if best_y_offset == 0:
                shifted_img = next_img
            else:
                h, w = next_img.shape[:2]
                shifted_img = np.zeros_like(next_img)
                
                # Shift down if best_y_offset > 0, shift up if best_y_offset < 0
                if best_y_offset > 0:
                    # Shift down
                    shifted_img[best_y_offset:, :] = next_img[:h-best_y_offset, :]
                else:
                    # Shift up (best_y_offset is negative)
                    shifted_img[:h+best_y_offset, :] = next_img[-best_y_offset:, :]
            
            # note we extract the strip after applying the vertical offset
            adjacent_strip = shifted_img[:, adjacent_pos:adjacent_pos+adjacent_strip_width]
            
            # append this strip to the LEFT of the panorama            
            new_width = current_pano.shape[1] + adjacent_strip_width
            
            try:
                # wider panorama
                new_panorama = np.zeros((common_height, new_width, 3), dtype=np.uint8)
                new_panorama[:, :adjacent_strip_width] = adjacent_strip
                new_panorama[:, adjacent_strip_width:] = current_pano
                
            except Exception as e:
                print(f"Error in creating/copying panorama: {e}")
                continue
            
            # blending
            if blend_width > 0:
                for x in range(blend_width):
                    alpha = x / blend_width
                    
                    strip_x = adjacent_strip_width - blend_width + x
                    
                    if strip_x >= 0 and strip_x < adjacent_strip.shape[1]:
                        # create masks for non-black pixels                        
                        strip_mask = (adjacent_strip[:, strip_x].sum(axis=1) > 10)
                        pano_mask = (current_pano[:, 0].sum(axis=1) > 10)
                        
                        # blend only where both have non-black pixels
                        blend_mask = strip_mask & pano_mask
                        
                        # compute blended values
                        if np.any(blend_mask):
                            strip_vals = adjacent_strip[:, strip_x].copy()
                            pano_vals = current_pano[:, 0].copy()
                            
                            # get indices of rows to blend
                            blend_indices = np.where(blend_mask)[0]
                            
                            # apply blend to the new panorama
                            blend_pos = adjacent_strip_width - blend_width + x
                            if 0 <= blend_pos < new_width:
                                for idx in blend_indices:
                                    try:
                                        new_panorama[idx, blend_pos] = (1 - alpha) * strip_vals[idx] + alpha * pano_vals[idx]
                                    except Exception as e:
                                        pass  # Skip problematic pixels
            
            panorama = new_panorama
            adjacent_strip_width = original_width  # reset for next iter
            
        except Exception as e:
            print(f"Error processing image {i}: {e}")
            import traceback
            traceback.print_exc()
    
    # [Cropping]
    # due to padding, alignment shifts (or resizing during blending)
    # we may have empty rows/columns in the panorama, which need to be cropped.
    try:
        gray = cv2.cvtColor(panorama, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # non-empty rows and columns
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        
        if np.any(rows) and np.any(cols):
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            # add small margin
            margin = 10
            y_min = max(0, y_min - margin)
            y_max = min(panorama.shape[0] - 1, y_max + margin)
            x_min = max(0, x_min - margin)
            x_max = min(panorama.shape[1] - 1, x_max + margin)
            
            # crop
            panorama = panorama[y_min:y_max+1, x_min:x_max+1]
        else:
            print("No content found for cropping!")
    except Exception as e:
        print(f"Error in cropping: {e}")
    
    print("Panorama creation complete!")
    return panorama, parameters_used

# wrapper function.
def create_panorama(segmented_dir, output_dir, param_combinations, adjacent_strip_width=45, blend_width=5):
    """
    Create a panorama from segmented images in a directory.
    
    Args:
        segmented_dir: Directory containing segmented images
        output_dir: Directory to save results
        param_combinations: List of parameter combinations to try
        adjacent_strip_width: Width of strips to add to panorama
        blend_width: Width of blending region
        
    Returns:
        True if successful, False otherwise
    """
    # load segmented images
    print(f"Loading segmented images from {segmented_dir}")
    images, filenames = load_images_from_folder(segmented_dir)
    
    if not images:
        print("No images found!")
        return False
    
    print(f"Found {len(images)} segmented images")
    
    # evaluate image pair quality
    print("\nEvaluating image pair quality...")
    match_qualities = evaluate_pair_quality(images)
    
    # sort by quality and print results
    sorted_qualities = sorted(match_qualities, key=lambda x: x['ssd'])
    print("\nImage pair quality (sorted by SSD, lower is better):")
    for q in sorted_qualities:
        print(f"Pair {q['pair']}: SSD={q['ssd']:.6f}, Y-offset={q['y_offset']}")
    
    # Create the panorama
    panorama, parameters_used = create_panorama_internal(
        images,
        param_combinations,
        adjacent_strip_width,
        blend_width
    )
    
    if panorama is None:
        print("Failed to create panorama!")
        return False
    
    # save the panorama
    save_panorama_result(panorama, output_dir)
    
    # create visualizations
    visualize_matching(
        images,
        parameters_used,
        adjacent_strip_width,
        output_dir
    )
    
    # Log the parameters used
    log_file = os.path.join(output_dir, 'parameters_log.txt')
    with open(log_file, 'w') as f:
        f.write("Parameters used for each image:\n")
        for params in parameters_used:
            f.write(f"Image {params['index']}: match_width={params['match_width']}, " +
                   f"center_percent={params['center_percent']}, y_offset={params['y_offset']}, " +
                   f"SSD={params['ssd']:.6f}\n")
    
    print(f"Panorama creation successful! Results saved to {output_dir}")
    return True
