"""
Object segmentation using "GrabCut" algorithm.
This module handles the extraction of objects from background.
"""

import os
import cv2
import numpy as np
from src.utils.image_utils import load_images_from_folder, create_directory
from src.utils.visualization import visualize_segmentation
import multiprocessing
from functools import partial
from tqdm import tqdm 

def segment_object_grabcut(image):
    """
    Segment the object using GrabCut algorithm.
    Assumes object is roughly in the center of the image.
    
    Args:
        image: Input image
        
    Returns:
        Tuple of (segmented image, mask)
    """
    mask = np.zeros(image.shape[:2], np.uint8) # Initial mask
    
    height, width = image.shape[:2]
    margin_x = width // 4
    margin_y = height // 4
    rect = (margin_x, margin_y, width - 2*margin_x, height - 2*margin_y) # Initial rectangle
    
    # --- MODIFICATION POINT: GrabCut parameters and model initialization ---
    # Original parameter setup:
    # # 13 is fixed by OpenCV according to the documentation
    # NUM_PARAMS = 13
    # num_iters = 5
    # bgd_model = np.zeros((1, num_iters * NUM_PARAMS), np.float64)
    # fgd_model = np.zeros((1, num_iters * NUM_PARAMS), np.float64)

    # MODIFIED parameter setup:
    # Using standard model sizes for OpenCV Python's GrabCut.
    # The number of iterations passed to cv2.grabCut is the key parameter.
    bgd_model = np.zeros((1, 65), np.float64) 
    fgd_model = np.zeros((1, 65), np.float64)
    # Original num_iters was 5. My previous optimization suggestion was 3.
    # Let's try restoring to 5 or even increasing slightly if quality is an issue and time permits.
    # num_iters = 3 # Previous suggestion
    num_iters = 5 # Restoring to original, or try 7-10 for potentially better quality at cost of time.
    # --- END MODIFICATION POINT ---

    # doing GrabCut
    # Initialize mask with GC_INIT_WITH_RECT.
    # Pixels outside rect are GC_BGD (0). Inside rect are GC_PR_FGD (3).
    try:
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, num_iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error as e:
        print(f"Warning: cv2.grabCut failed for an image: {e}. Using rectangle as foreground.")
        cv2.rectangle(mask, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), cv2.GC_FGD, -1)
    
    # --- MODIFICATION POINT: Mask interpretation ---
    # Original mask interpretation:
    # # background: 0, 2 -> 0
    # # foreground: 1, 3 -> 1
    # mask_returned = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # MODIFIED mask interpretation:
    # After cv2.grabCut, mask contains:
    # 0 (cv2.GC_BGD): Definite background
    # 1 (cv2.GC_FGD): Definite foreground
    # 2 (cv2.GC_PR_BGD): Probable background
    # 3 (cv2.GC_PR_FGD): Probable foreground
    # To get a binary mask, we treat definite and probable foreground as 1, others as 0.
    mask_returned = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
    # --- END MODIFICATION POINT ---
    
    segmented = image * mask_returned[:, :, np.newaxis]
    
    return segmented, mask_returned

def refine_mask_with_morphology(mask, kernel_size=5):
    """
    Refine the mask using morphological operations.
    
    Args:
        mask: Binary mask (0 for background, 1 for foreground)
        kernel_size: Size of morphological kernel
        
    Returns:
        Refined mask
    """
    # --- MODIFICATION POINT: Morphological operations tuning ---
    # Original kernel_size = 5.
    # If light color patches are not fully segmented (i.e., holes in the foreground mask),
    # MORPH_CLOSE needs to fill them. A larger kernel might help, or more iterations.
    # If grey patches are small noise, MORPH_OPEN removes them.
    #
    # Let's try a slightly larger kernel for closing, or add iterations.
    #
    # Original:
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # # fill holes
    # refined_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    # # remove noise
    # refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)

    # MODIFIED:
    # Option 1: Slightly larger kernel for closing if holes are an issue
    # kernel_close_size = kernel_size + 2 # e.g., 7 if original was 5
    # kernel_open_size = kernel_size      # Keep open kernel same or slightly smaller
    
    # Option 2: Keep kernel size, but iterate MORPH_CLOSE if holes are stubborn
    # Option 3: Experiment with kernel shape (e.g., cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size)))

    # For now, let's stick to Option 1: adjust kernel sizes if default `kernel_size` isn't working well.
    # You can pass different kernel_sizes from `process_images` if you make it a parameter.
    # Or, we can make them slightly different here:
    
    # Effective kernel for closing (to fill holes in the object)
    # If the object has fine structures, a large closing kernel can distort them.
    # If light color is being missed (holes), closing should help.
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # Iterations can also help MORPH_CLOSE fill larger holes without excessively large kernels
    # refined_mask_closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close, iterations=1) # Original implicitly 1 iter
    refined_mask_closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close, iterations=2) # Try 2 iterations

    # Effective kernel for opening (to remove small noise/grey patches if they are background)
    # If grey patches are actual parts of the object being misclassified as noise,
    # then opening might remove them, which is bad. This assumes grey patches are noise.
    kernel_open_size = max(3, kernel_size - 2) # Make open kernel a bit smaller than close
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_open_size, kernel_open_size))
    refined_mask_opened = cv2.morphologyEx(refined_mask_closed, cv2.MORPH_OPEN, kernel_open, iterations=1)
    
    refined_mask = refined_mask_opened
    # --- END MODIFICATION POINT ---
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
    if np.all(segmented_image < 10): 
        return segmented_image

    lab = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    object_mask = np.any(segmented_image > 10, axis=2)
    
    if np.any(object_mask):
        l_values = l_channel[object_mask]
        if l_values.size > 0:
            avg_lightness = np.mean(l_values)
            if avg_lightness < min_brightness and avg_lightness > 1e-6: 
                brightness_factor = min_brightness / avg_lightness
                l_channel[object_mask] = np.clip(l_channel[object_mask] * brightness_factor, 0, 255)
                enhanced_lab = cv2.merge([l_channel, a, b]) 
                enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB) 
                result = np.zeros_like(segmented_image)
                result[object_mask] = enhanced_rgb[object_mask] 
                return result
    return segmented_image


def _process_single_image_for_mp_worker(image_tuple_with_output_paths):
    """
    Worker function to process a single image for segmentation.
    Args:
        image_tuple_with_output_paths: (idx, image, filename, masks_dir_path, segmented_dir_path, output_folder_vis_path)
    """
    idx, image, filename, masks_dir, segmented_dir, output_folder_for_vis = image_tuple_with_output_paths
    
    try:
        segmented, mask = segment_object_grabcut(image) # Calls the modified version
        # --- MODIFICATION POINT: Parameter for refine_mask_with_morphology ---
        # Original call in serial loop didn't specify kernel_size, so it used default 5.
        # We can pass it or use a default here.
        # Let's use a default, but this could be a parameter.
        # kernel_for_refine = 5 # Original default
        kernel_for_refine = 7 # Experiment: try a slightly larger kernel if holes are the main issue
        # --- END MODIFICATION POINT ---
        refined_mask = refine_mask_with_morphology(mask, kernel_size=kernel_for_refine)
        
        refined_segmented = image.copy()
        refined_segmented[refined_mask == 0] = 0
        enhanced_segmented = ensure_content_visible(refined_segmented)
        
        mask_filename = os.path.join(masks_dir, f"mask_{filename}")
        segmented_filename = os.path.join(segmented_dir, f"segmented_{filename}")

        cv2.imwrite(mask_filename, refined_mask * 255)
        cv2.imwrite(segmented_filename, cv2.cvtColor(enhanced_segmented, cv2.COLOR_RGB2BGR))

        vis_data = None
        if idx < 2: 
            vis_data = {
                'original': image, 
                'enhanced_segmented_bgr_path': segmented_filename, 
                'refined_mask_path': mask_filename, 
                'save_path': os.path.join(output_folder_for_vis, f"segmentation_vis_{idx}.png")
            }
        return {'filename': filename, 'success': True, 'vis_data': vis_data, 'idx': idx} # Added idx for sorting
    except Exception as e:
        return {'filename': filename, 'success': False, 'error': str(e), 'vis_data': None, 'idx': idx}


def process_images(folder_path, output_folder):
    """
    Process all images in a folder and save the segmented objects.
    This function is MODIFIED to use multiprocessing.
    The original serial loop is commented out.
    
    Args:
        folder_path: Path to the folder containing original images
        output_folder: Path to save results (e.g., "results/")
                       Subdirs "segmented/masks" and "segmented/segmented" will be created.
    """
    base_segmented_dir = os.path.join(output_folder, 'segmented') 
    create_directory(base_segmented_dir) 

    masks_dir = os.path.join(base_segmented_dir, 'masks')      
    segmented_rgb_dir = os.path.join(base_segmented_dir, 'segmented') 
    create_directory(masks_dir)
    create_directory(segmented_rgb_dir)
    
    images, filenames = load_images_from_folder(folder_path)    
    if not images:
        print(f"No images found in {folder_path}. Segmentation cannot proceed.")
        return
    
    print(f"Processing {len(images)} images for segmentation using multiprocessing...")

    # Original serial loop:
    # for i, (image, filename) in enumerate(zip(images, filenames)):
    #     print(f"Processing image {i+1}/{len(images)}: {filename}")
    #     segmented, mask = segment_object_grabcut(image)
    #     refined_mask = refine_mask_with_morphology(mask) # Used default kernel_size=5
    #     refined_segmented = image.copy()
    #     refined_segmented[refined_mask == 0] = 0
    #     enhanced_segmented = ensure_content_visible(refined_segmented)
    #     mask_filename = os.path.join(masks_dir, f"mask_{filename}")
    #     segmented_filename = os.path.join(segmented_rgb_dir, f"segmented_{filename}") 
    #     cv2.imwrite(mask_filename, refined_mask * 255)
    #     cv2.imwrite(segmented_filename, cv2.cvtColor(enhanced_segmented, cv2.COLOR_RGB2BGR))
    #     if i < 2: 
    #         vis_path = os.path.join(output_folder, f"segmentation_vis_{i}.png") 
    #         visualize_segmentation(image, enhanced_segmented, refined_mask, 
    #                              show=False, save_path=vis_path)
    
    tasks_for_mp = []
    for i, (image, filename) in enumerate(zip(images, filenames)):
        tasks_for_mp.append((i, image, filename, masks_dir, segmented_rgb_dir, output_folder)) 

    num_cores = multiprocessing.cpu_count()
    pool_size = max(1, num_cores - 2 if num_cores > 2 else 1) 

    mp_results = []
    # --- MODIFICATION POINT: Use functools.partial for worker function ---
    # The worker function `_process_single_image_for_mp_worker` expects a single argument which is the tuple.
    # No partial needed here as the tuple itself contains all necessary changing and fixed args for one call.
    # Original (if `base_output_folder` was fixed for all):
    # worker_func = partial(_process_single_image_for_mp_worker, base_output_folder=output_folder)
    # with multiprocessing.Pool(processes=pool_size) as pool:
    #     with tqdm(total=len(tasks_for_mp_simple), desc="Segmenting Images (MP)") as pbar: # tasks_for_mp_simple would contain only (idx,img,fname)
    #         for result in pool.imap_unordered(worker_func, tasks_for_mp_simple): #
    #             mp_results.append(result)
    #             pbar.update(1)

    # MODIFIED: Pass the full tuple directly if worker expects it
    with multiprocessing.Pool(processes=pool_size) as pool:
        with tqdm(total=len(tasks_for_mp), desc="Segmenting Images (MP)") as pbar:
            for result in pool.imap_unordered(_process_single_image_for_mp_worker, tasks_for_mp):
                mp_results.append(result)
                pbar.update(1)
    # --- END MODIFICATION POINT ---
    
    # Sort results by original index before visualization
    mp_results.sort(key=lambda r: r['idx']) # Ensure idx is returned by worker

    visualized_count = 0
    for res in mp_results:
        if visualized_count >= 2:
            break
        if res['success'] and res['vis_data']:
            vd = res['vis_data']
            try:
                mask_to_vis = cv2.imread(vd['refined_mask_path'], cv2.IMREAD_GRAYSCALE)
                segmented_to_vis_bgr = cv2.imread(vd['enhanced_segmented_bgr_path'])
                if mask_to_vis is not None and segmented_to_vis_bgr is not None:
                    segmented_to_vis_rgb = cv2.cvtColor(segmented_to_vis_bgr, cv2.COLOR_BGR2RGB)
                    visualize_segmentation(vd['original'], segmented_to_vis_rgb, mask_to_vis,
                                         show=False, save_path=vd['save_path'])
                    visualized_count +=1
                else:
                    print(f"Warning: Could not load saved mask/segmented image for visualization: {res['filename']}")
            except Exception as e_vis:
                print(f"Error during post-MP visualization for {res['filename']}: {e_vis}")
        elif not res['success']:
            print(f"Segmentation failed for {res['filename']}: {res.get('error', 'Unknown error')}")

    print(f"Segmentation complete. Segmented images and masks saved into {base_segmented_dir}")