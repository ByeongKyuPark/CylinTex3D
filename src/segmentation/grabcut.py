"""
Object segmentation using "GrabCut" algorithm.
This module handles the extraction of objects from background.
"""

import os
import cv2
import numpy as np
from src.utils.image_utils import load_images_from_folder, create_directory
from src.utils.visualization import visualize_segmentation

def segment_object_grabcut(image):
    """
    Segment the object using GrabCut algorithm.
    Assumes object is roughly in the center of the image.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (segmented image, mask)
    """
    # mask with same (W,H) as image, but 'grayscale'
    mask = np.zeros(image.shape[:2], np.uint8)
    
    height, width = image.shape[:2]
    margin_x = width // 4
    margin_y = height // 4

    # assuming object is roughly at the center  
    rect = (margin_x, margin_y, width - 2*margin_x, height - 2*margin_y)
    
    # 13 is fixed by OpenCV according to the documentation
    NUM_PARAMS = 13
    num_iters = 5
    bgd_model = np.zeros((1, num_iters * NUM_PARAMS), np.float64)
    fgd_model = np.zeros((1, num_iters * NUM_PARAMS), np.float64)
    
    # doing GrabCut
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, num_iters, cv2.GC_INIT_WITH_RECT)
    
    # background: 0, 2 -> 0
    # foreground: 1, 3 -> 1
    mask_returned = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    segmented = image * mask_returned[:, :, np.newaxis]
    
    return segmented, mask_returned

def refine_mask_with_morphology(mask, kernel_size=5):
    """
    Refine the mask using morphological operations.
    
    Args:
        mask: Binary mask
        kernel_size: Size of morphological kernel
        
    Returns:
        Refined mask
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # fill holes
    refined_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    # remove noise
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
    return refined_mask

def ensure_content_visible(segmented_image, min_brightness=30):
    """
    Brighten the image if it's too dark.
    
    Args:
        segmented_image: Segmented image
        min_brightness: Minimum brightness threshold
        
    Returns:
        Enhanced image
    """
    # RGB -> LAB
    lab = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # 10 is a threshold for black background
    object_mask = np.any(segmented_image > 10, axis=2)
    
    # check if there something to brighten (non-black background)
    if np.any(object_mask):
        l_values = l_channel[object_mask]
        avg_lightness = np.mean(l_values)
        
        # if too dark, brighten 'L' channel
        if avg_lightness < min_brightness:
            brightness_factor = min_brightness / max(1, avg_lightness)

            # match the avg lightness to the min_brightness
            l_channel[object_mask] = np.clip(l_channel[object_mask] * brightness_factor, 0, 255)

            enhanced = cv2.merge([l_channel, a, b]) # L`+A+B
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            # even though we only modified the L channel, 
            # converting color spaces might have changed the RGB values
            
            # therefore, we start from a clean RGB image
            # and only apply the enhanced 'object' pixels
            result = np.zeros_like(segmented_image)
            result[object_mask] = enhanced[object_mask]
            return result
    
    return segmented_image

def process_images(folder_path, output_folder):
    """
    Process all images in a folder and save the segmented objects.
    
    Args:
        folder_path: Path to the folder containing original images
        output_folder: Path to save results
    """
    # Create output directories
    create_directory(output_folder)
    masks_dir = os.path.join(output_folder, 'masks')
    segmented_dir = os.path.join(output_folder, 'segmented')
    create_directory(masks_dir)
    create_directory(segmented_dir)
    
    # Load images
    images, filenames = load_images_from_folder(folder_path)    
    
    print(f"Processing {len(images)} images for segmentation...")
    
    for i, (image, filename) in enumerate(zip(images, filenames)):
        print(f"Processing image {i+1}/{len(images)}: {filename}")
        
        # Segment the object
        segmented, mask = segment_object_grabcut(image)

        # Refine the mask
        refined_mask = refine_mask_with_morphology(mask)
        
        # Apply refined mask
        refined_segmented = image.copy()
        refined_segmented[refined_mask == 0] = 0
        
        # Enhance the contrast to ensure visibility
        enhanced_segmented = ensure_content_visible(refined_segmented)
        
        # Save results
        mask_filename = os.path.join(masks_dir, f"mask_{filename}")
        segmented_filename = os.path.join(segmented_dir, f"segmented_{filename}")

        cv2.imwrite(mask_filename, refined_mask * 255)
        cv2.imwrite(segmented_filename, cv2.cvtColor(enhanced_segmented, cv2.COLOR_RGB2BGR))

        # Visualize the first few results
        if i < 2:  # Show only first two images
            vis_path = os.path.join(output_folder, f"segmentation_vis_{i}.png")
            visualize_segmentation(image, enhanced_segmented, refined_mask, 
                                 show=False, save_path=vis_path)
    
    print(f"Segmentation complete. Segmented images saved to {segmented_dir}")
