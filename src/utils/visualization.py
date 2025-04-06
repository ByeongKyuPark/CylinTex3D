"""
Visualization functions for displaying and saving results.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from src.utils.image_utils import resize_to_match_height
from src.utils.image_utils import get_bounding_box
    
def visualize_segmentation(original, segmented, mask, show=True, save_path=None):
    """
    Visualize segmentation results with original image, segmented object, and mask.
    
    Args:
        original: Original input image
        segmented: Segmented object
        mask: Binary mask
        show: Whether to display the visualization
        save_path: Path to save the visualization image
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(segmented)
    plt.title('Segmented Object')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_matching(images, parameters_used, adjacent_strip_width=30, output_dir="results"):
    """
    Create visualizations of the strip matching process for each image pair.
    
    Args:
        images: List of input images
        parameters_used: List of parameters used for each match
        adjacent_strip_width: Width of adjacent strips
        output_dir: Directory to save visualization images
    """
    if len(images) < 2:
        return
    
    print("\nCreating visualizations...")
    
    # ensure all images have the same height
    common_height = min(img.shape[0] for img in images)
    resized_images = [resize_to_match_height(img, common_height) for img in images]
    
    # get center strip from first image
    bbox = get_bounding_box(resized_images[0])
    x_min, _, x_max, _ = bbox
    center_x = (x_min + x_max) // 2
    
    # extract center strip from first image
    start_x = max(0, center_x - adjacent_strip_width // 2)
    center_strip = resized_images[0][:, start_x:start_x+adjacent_strip_width]
    
    # initialize panorama with the center strip
    panorama = center_strip.copy()
    
    for param_data in parameters_used:
        i = param_data['index']
        match_width = param_data['match_width']
        best_pos = param_data['position']
        best_y_offset = param_data['y_offset']
        
        print(f"Creating visualization for images {i-1} and {i}")
        
        # get current panorama and next image
        current_pano = panorama.copy()
        next_img = resized_images[i]
        
        # apply vertical offset to the next image
        if best_y_offset == 0:
            shifted_img = next_img
        else:
            h, w = next_img.shape[:2]
            shifted_img = np.zeros_like(next_img)
            
            if best_y_offset > 0:
                # shift down
                shifted_img[best_y_offset:, :] = next_img[:h-best_y_offset, :]
            else:
                # shift up (best_y_offset is negative)
                shifted_img[:h+best_y_offset, :] = next_img[-best_y_offset:, :]
        
        # extract LEFT edge strip from current panorama
        left_strip = current_pano[:, :match_width]
        
        try:
            # calculate position for the adjacent strip (to the LEFT of the match)
            adjacent_pos = max(0, best_pos - adjacent_strip_width)
            
            # adjust width if needed
            actual_width = min(adjacent_strip_width, best_pos - adjacent_pos)
            
            # extract adjacent strip if possible
            if actual_width > 0:
                adjacent_strip = shifted_img[:, adjacent_pos:adjacent_pos+actual_width]
                
                # create a new panorama with the added strip to the LEFT
                new_width = current_pano.shape[1] + actual_width
                new_panorama = np.zeros((common_height, new_width, 3), dtype=np.uint8)
                
                # copy the adjacent strip to the left
                new_panorama[:, :actual_width] = adjacent_strip
                
                # copy current panorama to the right
                new_panorama[:, actual_width:] = current_pano
                
            else:
                adjacent_strip = np.zeros((common_height, 1, 3), dtype=np.uint8)
                new_panorama = current_pano.copy()
            
            # extract matching strip for VISUALIZATION
            matching_strip = shifted_img[:, best_pos:best_pos+match_width] if best_pos + match_width <= shifted_img.shape[1] else np.zeros((common_height, 1, 3), dtype=np.uint8)
            
            try:
                fig = plt.figure(figsize=(15, 10))
                
                # set up a 2x3 grid for our new layout
                # row 1: Current panorama and left edge strip
                # row 2: Original/shifted comparison, matching strip, and adjacent strip
                
                gs = plt.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[3, 1, 1])
                
                # first row - Current panorama (spans 2 columns)
                ax_pano = plt.subplot(gs[0, 0:2])
                ax_pano.imshow(current_pano)
                ax_pano.axvline(x=min(match_width, current_pano.shape[1]-1), color='r', linewidth=2)
                ax_pano.set_title(f"Current panorama (after {i} images)")
                ax_pano.axis('off')
                
                # first row - Left edge strip
                ax_left = plt.subplot(gs[0, 2])
                ax_left.imshow(left_strip)
                ax_left.set_title("LEFT edge strip from panorama")
                ax_left.axis('off')
                
                #sSecond row - Original/Shifted comparison
                ax_compare = plt.subplot(gs[1, 0])
                
                # create side-by-side comparison if there's a vertical offset
                if best_y_offset != 0:
                    h, w = next_img.shape[:2]
                    composite = np.zeros((h, w*2, 3), dtype=np.uint8)
                    composite[:, :w] = next_img
                    composite[:, w:] = shifted_img
                    
                    ax_compare.imshow(composite)
                    ax_compare.axvline(x=w, color='y', linewidth=2, linestyle='--')
                    # mark the match region and adjacent strip on the shifted part (right side)
                    ax_compare.axvline(x=w+best_pos, color='r', linewidth=2)
                    ax_compare.axvline(x=min(w+best_pos+match_width, composite.shape[1]-1), color='r', linewidth=2)
                    
                    if adjacent_pos >= 0 and actual_width > 0:
                        ax_compare.axvline(x=w+adjacent_pos, color='g', linewidth=2)
                        ax_compare.axvline(x=min(w+adjacent_pos+actual_width, composite.shape[1]-1), color='g', linewidth=2)
                    
                    ax_compare.set_title(f"Original | Shifted (Y: {best_y_offset})")
                else:
                    ax_compare.imshow(next_img)
                    ax_compare.axvline(x=best_pos, color='r', linewidth=2)
                    ax_compare.axvline(x=min(best_pos+match_width, next_img.shape[1]-1), color='r', linewidth=2)
                    
                    if adjacent_pos >= 0 and actual_width > 0:
                        ax_compare.axvline(x=adjacent_pos, color='g', linewidth=2)
                        ax_compare.axvline(x=min(adjacent_pos+actual_width, next_img.shape[1]-1), color='g', linewidth=2)
                    
                    ax_compare.set_title(f"Image {i+1} with match (red) and strip (green)")
                
                ax_compare.axis('off')
                
                # second row - Matching strip
                ax_match = plt.subplot(gs[1, 1])
                ax_match.imshow(matching_strip)
                ax_match.set_title(f"Matching strip in image {i+1}")
                ax_match.axis('off')
                
                # second row - Adjacent strip
                ax_adjacent = plt.subplot(gs[1, 2])
                ax_adjacent.imshow(adjacent_strip)
                ax_adjacent.set_title(f"LEFT adjacent strip from image {i+1} (added)")
                ax_adjacent.axis('off')
                
                # overall title with match parameters
                plt.suptitle(f"Match: Width={match_width}, Y-Offset={best_y_offset}, SSD={param_data['ssd']:.6f}", fontsize=16)
                
                plt.tight_layout()
                
                output_file = os.path.join(output_dir, f"matching_{i-1}_{i}.png")
                plt.savefig(output_file)
                plt.close()
                
            except Exception as e:
                print(f"Error creating visualization: {e}")
                import traceback
                traceback.print_exc()
            
            # update panorama for next iteration
            panorama = new_panorama
            
        except Exception as e:
            print(f"Error processing image {i} for visualization: {e}")
            import traceback
            traceback.print_exc()
            # continue with current panorama
            panorama = current_pano
    
    print("Visualizations complete")
    

def save_panorama_result(panorama, output_dir):
    """
    Save panorama result with and without white background.
    
    Args:
        panorama: The created panorama image
        output_dir: Directory to save results
    """
    if panorama is None:
        print("No panorama to save!")
        return
        
    # Save original panorama
    output_path = os.path.join(output_dir, 'panorama.png')
    cv2.imwrite(output_path, cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR))
    print(f"Saved to {output_path}")
    
    # Save panorama with white background
    panorama_white = panorama.copy()
    mask = (panorama.sum(axis=2) < 10)
    panorama_white[mask] = [255, 255, 255]
    
    output_path_white = os.path.join(output_dir, 'panorama_white.png')
    cv2.imwrite(output_path_white, cv2.cvtColor(panorama_white, cv2.COLOR_RGB2BGR))
    print(f"White background version saved to {output_path_white}")
