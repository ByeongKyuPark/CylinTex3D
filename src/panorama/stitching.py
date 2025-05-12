"""
Panorama stitching module for creating complete panoramas from segmented images.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

from src.utils.image_utils import load_images_from_folder, get_bounding_box, resize_to_match_height, create_directory # Added create_directory
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
    if not images: return None, [] # Handle empty images list
    if len(images) < 2:
        return images[0].copy() if images else None, [] # Return a copy
    
    print("Creating panorama...")
    
    # resize to match the smallest height
    common_height = min(img.shape[0] for img in images)
    resized_images = [resize_to_match_height(img, common_height) for img in images]
    
    # get the center strip from the first image
    # This initializes the panorama by taking a strip of `adjacent_strip_width` from the center of the first image.
    bbox_first = get_bounding_box(resized_images[0])
    center_x_first = (bbox_first[0] + bbox_first[2]) // 2
    
    # Ensure initial strip is valid
    initial_pano_strip_width = min(adjacent_strip_width, resized_images[0].shape[1])
    start_x_pano_init = max(0, center_x_first - initial_pano_strip_width // 2)
    end_x_pano_init = min(start_x_pano_init + initial_pano_strip_width, resized_images[0].shape[1])
    actual_initial_pano_width = end_x_pano_init - start_x_pano_init

    if actual_initial_pano_width <= 0:
        print("Error: Cannot initialize panorama, first image strip is empty.")
        return None, []
        
    panorama = resized_images[0][:, start_x_pano_init:end_x_pano_init].copy()
    
    parameters_used = []
    
    # The loop prepends strips to the LEFT of the `panorama`
    for i in tqdm(range(1, len(resized_images)), desc="Building panorama"):
        # print(f"\nProcessing image {i+1}/{len(resized_images)}") # tqdm provides progress
        
        # `current_pano` is the panorama built so far. Template comes from its LEFT edge.
        # `next_img` is the new image to find a match in.
        current_pano_for_match = panorama.copy() # Make a copy for matching
        next_img_to_stitch = resized_images[i]
        
        try:
            # `find_best_match_with_parameters` expects (template_source, image_to_search_in, ...)
            # The template source is `current_pano_for_match`. The image to search in is `next_img_to_stitch`.
            best_params, best_results = find_best_match_with_parameters(
                current_pano_for_match, # This is the "current_pano" argument in find_best_match_with_parameters
                next_img_to_stitch,     # This is the "next_img" argument
                param_combinations
            )
            
            if best_params is None or best_results is None: # No match found by find_best_match_with_parameters
                print(f"Warning: No match found for image index {i}. Skipping this image.")
                continue

            # `match_width` from `best_params` is the width of the template strip taken from `current_pano_for_match`
            # `best_x_pos` is the x-coord in `next_img_to_stitch` (after potential y-shift) where match began
            match_width_used, center_region_perc_used, y_offset_range_used = best_params
            best_x_pos_in_next, ssd_val, best_y_offset_found, _ = best_results # bbox is for original next_img
            
            # print(f"Image {i}: Best match params: mw={match_width_used}, crp={center_region_perc_used}, y_range={y_offset_range_used}")
            # print(f"  Results: pos_in_next={best_x_pos_in_next}, ssd={ssd_val:.4f}, y_offset={best_y_offset_found}")
            
            parameters_used.append({
                'index': i, 'match_width': match_width_used,
                'center_percent': center_region_perc_used, 'y_offset': best_y_offset_found,
                'position': best_x_pos_in_next, 'ssd': ssd_val
            })
            
            # Shift `next_img_to_stitch` vertically based on `best_y_offset_found`
            h_next_img, w_next_img = next_img_to_stitch.shape[:2]
            shifted_next_img = np.zeros_like(next_img_to_stitch)
            if best_y_offset_found == 0:
                shifted_next_img = next_img_to_stitch.copy()
            elif best_y_offset_found > 0:
                if h_next_img - best_y_offset_found > 0:
                    shifted_next_img[best_y_offset_found:, :] = next_img_to_stitch[:h_next_img - best_y_offset_found, :]
                else: continue # Shift too large
            else: # best_y_offset_found < 0
                if h_next_img + best_y_offset_found > 0:
                    shifted_next_img[:h_next_img + best_y_offset_found, :] = next_img_to_stitch[-best_y_offset_found:, :]
                else: continue # Shift too large

            # Extract the strip to PREPEND to the panorama.
            # This strip is from `shifted_next_img`, to the LEFT of `best_x_pos_in_next`.
            # Its width is `adjacent_strip_width` (from function args).
            end_col_for_strip_to_add = best_x_pos_in_next
            start_col_for_strip_to_add = max(0, end_col_for_strip_to_add - adjacent_strip_width)
            actual_width_of_prepended_strip = end_col_for_strip_to_add - start_col_for_strip_to_add
            
            if actual_width_of_prepended_strip <= 0:
                print(f"  Warning: No space for adjacent strip in image {i} (width was {actual_width_of_prepended_strip}), skipping.")
                continue
            
            strip_to_prepend = shifted_next_img[:, start_col_for_strip_to_add:end_col_for_strip_to_add]
            
            # Create new panorama: [strip_to_prepend | panorama (which is current_pano_for_match here essentially)]
            # `panorama` variable is updated at the end of the loop.
            temp_panorama_with_prepend = np.concatenate((strip_to_prepend, panorama), axis=1)
            
            # Blending
            # Original blending loop:
            # if blend_width > 0:
            #     # The original loop's `adjacent_strip_width` for indexing was the width of `strip_to_prepend`
            #     # `current_pano` was `panorama` (the one being appended to)
            #     # `new_panorama` was `temp_panorama_with_prepend`
            #     # `adjacent_strip` was `strip_to_prepend`
            #     #
            #     # The logic was:
            #     # new_panorama[idx, blend_pos] = (1-alpha)*strip_vals[idx] + alpha*pano_vals[idx]
            #     # where strip_vals came from the right edge of `strip_to_prepend`
            #     # and pano_vals came from the *first column* of `panorama` (before prepending)
            #     # blend_pos was relative to the start of `temp_panorama_with_prepend`
            #     #   and covered the rightmost `blend_width` cols of `strip_to_prepend` part.

            #     # Width of the `strip_to_prepend`
            #     width_of_prepended_part = strip_to_prepend.shape[1] 

            #     for x_blend_loop in range(blend_width): # x_blend_loop from 0 to blend_width-1
            #         alpha = x_blend_loop / float(blend_width) # Ensure float division
                    
            #         # strip_x is the column index within strip_to_prepend (from its right edge)
            #         # It goes from (width_of_prepended_part - blend_width) up to (width_of_prepended_part - 1)
            #         strip_x_col_in_prepended = width_of_prepended_part - blend_width + x_blend_loop
                    
            #         if strip_x_col_in_prepended >= 0 and strip_x_col_in_prepended < width_of_prepended_part:
            #             # Masks for non-black pixels
            #             # `strip_to_prepend` is `adjacent_strip` in original
            #             # `panorama` (before prepend) is `current_pano` in original
            #             strip_mask_blend = (strip_to_prepend[:, strip_x_col_in_prepended].sum(axis=1) > 10)
            #             pano_mask_blend = (panorama[:, 0].sum(axis=1) > 10) # Always first col of old panorama
                        
            #             blend_apply_mask = strip_mask_blend & pano_mask_blend
                        
            #             if np.any(blend_apply_mask):
            #                 strip_vals_blend = strip_to_prepend[:, strip_x_col_in_prepended].copy()
            #                 pano_vals_blend = panorama[:, 0].copy() # Always first col
                            
            #                 blend_indices_to_apply = np.where(blend_apply_mask)[0]
                            
            #                 # blend_pos_in_new_pano is same as strip_x_col_in_prepended because strip_to_prepend is at the start
            #                 blend_pos_in_new_pano = strip_x_col_in_prepended 

            #                 if 0 <= blend_pos_in_new_pano < temp_panorama_with_prepend.shape[1]:
            #                     for idx_blend in blend_indices_to_apply:
            #                         try:
            #                             temp_panorama_with_prepend[idx_blend, blend_pos_in_new_pano] = \
            #                                 (1 - alpha) * strip_vals_blend[idx_blend] + alpha * pano_vals_blend[idx_blend]
            #                         except Exception: # Skip problematic pixels
            #                             pass 
            
            # OPTIMIZED Blending (vectorized version of the original logic):
            if blend_width > 0 and strip_to_prepend.shape[1] >= blend_width:
                width_of_prepended_part = strip_to_prepend.shape[1]
                
                # Region in `temp_panorama_with_prepend` to modify:
                # These are the rightmost `blend_width` columns of the `strip_to_prepend` part.
                blend_target_cols_start = width_of_prepended_part - blend_width
                blend_target_cols_end = width_of_prepended_part

                # Source pixels from `strip_to_prepend` for blending (its rightmost `blend_width` cols)
                strip_source_for_blend = strip_to_prepend[:, -blend_width:].astype(np.float32)
                
                # Source pixels from `panorama` (the one before prepending this strip) - ALWAYS its first column
                pano_col0_for_blend = panorama[:, 0:1].astype(np.float32) # Shape (H, 1, 3)

                # Alpha ramp for blending: 0 to (blend_width-1)/blend_width
                alphas_blend = np.linspace(0, 1.0, blend_width, endpoint=False)[np.newaxis, :, np.newaxis] # (1, blend_width, 1)
                
                # Calculate blended values: (1-alpha)*strip_source + alpha*pano_col0_broadcasted
                # Broadcast pano_col0_for_blend from (H,1,3) to (H,blend_width,3)
                pano_col0_broadcasted_for_blend = np.repeat(pano_col0_for_blend, blend_width, axis=1)
                
                blended_patch_values = (1.0 - alphas_blend) * strip_source_for_blend + alphas_blend * pano_col0_broadcasted_for_blend
                
                # Masks for active regions (only blend if both sources have content)
                strip_active_mask_blend = np.any(strip_source_for_blend > 10, axis=2) # (H, blend_width)
                pano_col0_active_mask_blend = np.any(pano_col0_for_blend > 10, axis=2)[:,0] # (H,)
                
                final_apply_blend_mask = strip_active_mask_blend & pano_col0_active_mask_blend[:, np.newaxis] # (H, blend_width)
                final_apply_blend_mask_3ch = np.repeat(final_apply_blend_mask[:, :, np.newaxis], 3, axis=2)

                # Get the target region from temp_panorama_with_prepend (which is float32 for modification)
                target_region_to_blend_float = temp_panorama_with_prepend[:, blend_target_cols_start:blend_target_cols_end].astype(np.float32)
                
                # Apply blended_patch_values where final_apply_blend_mask_3ch is True
                # np.copyto handles the conditional assignment based on 'where'
                np.copyto(target_region_to_blend_float, blended_patch_values, where=final_apply_blend_mask_3ch)
                
                # Put the (partially) blended region back into temp_panorama_with_prepend
                temp_panorama_with_prepend[:, blend_target_cols_start:blend_target_cols_end] = \
                    np.clip(target_region_to_blend_float, 0, 255).astype(np.uint8)
            # End of OPTIMIZED Blending

            panorama = temp_panorama_with_prepend # Update the main panorama for the next iteration
            # adjacent_strip_width = original_width # This was in original, but adjacent_strip_width is a func arg, not looped var
            
        except Exception as e:
            print(f"Error processing image index {i} for panorama stitching: {e}")
            import traceback
            traceback.print_exc()
            # If error, `panorama` remains unchanged from previous iteration.
    
    # Cropping - Original logic
    try:
        if panorama is not None and panorama.size > 0 : # Check if panorama is valid
            gray_pano = cv2.cvtColor(panorama, cv2.COLOR_RGB2GRAY)
            _, binary_pano = cv2.threshold(gray_pano, 10, 255, cv2.THRESH_BINARY)
            
            rows_pano = np.any(binary_pano, axis=1)
            cols_pano = np.any(binary_pano, axis=0)
            
            if np.any(rows_pano) and np.any(cols_pano):
                y_min_crop, y_max_crop = np.where(rows_pano)[0][[0, -1]]
                x_min_crop, x_max_crop = np.where(cols_pano)[0][[0, -1]]
                
                # Original margin logic
                margin_crop = 10 
                y_min_crop = max(0, y_min_crop - margin_crop)
                y_max_crop = min(panorama.shape[0] - 1, y_max_crop + margin_crop)
                x_min_crop = max(0, x_min_crop - margin_crop)
                x_max_crop = min(panorama.shape[1] - 1, x_max_crop + margin_crop)
                
                # Ensure crop indices are valid
                if y_min_crop < y_max_crop and x_min_crop < x_max_crop:
                    panorama = panorama[y_min_crop:y_max_crop+1, x_min_crop:x_max_crop+1]
                # else: print("No valid crop region found after margin.")
            # else: print("No content found for cropping in panorama.")
        # else: print("Panorama is None or empty before cropping.")
    except Exception as e_crop:
        print(f"Error in panorama cropping: {e_crop}")
    
    print("Panorama creation complete!")
    return panorama, parameters_used

# wrapper function.
def create_panorama(segmented_dir, output_dir, param_combinations, adjacent_strip_width=45, blend_width=5):
    """
    Create a panorama from segmented images in a directory.
    
    Args:
        segmented_dir: Directory containing segmented images (e.g., results/segmented/segmented)
        output_dir: Directory to save results (e.g., results/)
        param_combinations: List of parameter combinations to try
        adjacent_strip_width: Width of strips to add to panorama
        blend_width: Width of blending region
        
    Returns:
        True if successful, False otherwise
    """
    # Ensure output_dir (e.g. "results/") exists for saving logs and visualizations
    create_directory(output_dir)

    # load segmented images
    print(f"Loading segmented images from {segmented_dir}")
    # `segmented_dir` should point to the folder with the actual segmented RGB images.
    # Your `process_images` saves them into `output_folder/segmented/segmented/`.
    # So, `main.py` should pass `results/segmented/segmented` as `segmented_dir` here.
    images, filenames = load_images_from_folder(segmented_dir)
    
    if not images:
        print(f"No images found in {segmented_dir}!")
        return False
    
    print(f"Found {len(images)} segmented images.")
    
    # evaluate image pair quality
    print("\nEvaluating image pair quality...")
    # The evaluate_pair_quality function uses the optimized strip matching internally.
    # It might need adjustment for the `param_combinations` it uses or take them as an arg.
    # For now, assuming it uses its default params or the optimized find_best_match_in_central_region.
    # If `evaluate_pair_quality` takes default args for matching different from `param_combinations`, that's fine.
    match_qualities = evaluate_pair_quality(images) # Uses its own default matching params, or could take some
    
    # sort by quality and print results
    if match_qualities: # Check if not empty
        sorted_qualities = sorted(match_qualities, key=lambda x: x['ssd'] if x['ssd'] is not None else float('inf'))
        print("\nImage pair quality (sorted by SSD, lower is better, top 5):")
        for q_idx, q in enumerate(sorted_qualities[:5]):
            print(f"  Pair {q['pair']}: SSD={q['ssd']:.6f}, Y-offset={q['y_offset']}")
    else:
        print("No pair qualities evaluated (not enough images or error).")

    # Create the panorama
    panorama_img, parameters_used_log = create_panorama_internal(
        images,
        param_combinations, # These are for the main stitching process
        adjacent_strip_width,
        blend_width
    )
    
    if panorama_img is None or panorama_img.size == 0: # Check for empty panorama
        print("Failed to create panorama or panorama is empty.")
        return False
    
    # save the panorama
    # `save_panorama_result` expects `output_dir` to be `results/`
    save_panorama_result(panorama_img, output_dir) # Saves to `results/panorama.png`
    
    # create visualizations
    # `visualize_matching` also expects `output_dir` to be `results/` for its output images
    # visualize_matching(
    #     images, # Pass the resized_images if common_height was used, or original
    #     parameters_used_log,
    #     adjacent_strip_width, # The target width
    #     output_dir # e.g. "results/"
    # )
    print("Skipping detailed matching visualization in create_panorama for brevity. Can be re-enabled.")
    
    # Log the parameters used
    log_file = os.path.join(output_dir, 'panorama_stitching_parameters_log.txt') # More descriptive name
    with open(log_file, 'w') as f:
        f.write("Parameters used for stitching each image strip:\n")
        for params_entry in parameters_used_log:
            f.write(f"Image Index {params_entry['index']}: "
                   f"match_width_param={params_entry['match_width']}, "
                   f"center_perc_param={params_entry['center_percent']}, " 
                   f"y_offset_found={params_entry['y_offset']}, " 
                   f"matched_x_pos_in_next_img={params_entry['position']}, "
                   f"SSD(TM_NORMED)={params_entry['ssd']:.6f}\n")
    
    print(f"Panorama creation successful! Main panorama saved in {output_dir}, log in {log_file}")
    return True