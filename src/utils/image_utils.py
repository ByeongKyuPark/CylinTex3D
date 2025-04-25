"""
Image utility functions for loading, preprocessing, and general image operations.
"""

import os
import cv2
import numpy as np

def create_directory(directory):
    """this creates dir if doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_images_from_folder(folder_path):
    """
    Load all images from a folder in "SORTED" order.

    Order matters for the pipeline !!
    This function assumes that the images are named in a way that sorting them
    lexicographically will yield the correct order (CCW or CW).
    
    Args:
        folder_path: Path to the folder containing images
        
    Returns:
        Tuple of (images, filenames)
    """
    images = []
    filenames = []
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        # Try alternative path (parent directory)
        alternative_path = os.path.dirname(folder_path)
        if os.path.exists(alternative_path):
            folder_path = alternative_path
        else:
            print(f"Warning: Folder not found: {folder_path}")
            return images, filenames
    
    # order matters...!
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(('.png', '.jpg', '.jpeg')): # wonder if we need to add .bmp ?
            img_path = os.path.join(folder_path, filename) # join automatically adds the separator '/' 
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                filenames.append(filename)
    
    return images, filenames

def resize_to_match_height(source_img, target_height):
    """
    match the height of the source image to the target height keeping the aspect ratio.
    
    Args:
        source_img: Source image to resize
        target_height: Target height in pixels
        
    Returns:
        Resized image
    """
    if source_img.shape[0] == target_height:
        return source_img
    
    # calc "NEW" width based on the original aspect ratio
    aspect_ratio = source_img.shape[1] / source_img.shape[0]
    new_width = int(aspect_ratio * target_height)

    return cv2.resize(source_img, (new_width, target_height), interpolation=cv2.INTER_AREA)

def get_bounding_box(image):
    """
    get the simple bounding box 

    here's how it works:
    1. convert to grayscale
    2. threshold to find non-black pixels
    3. find bounding box (defined by ul,br corners)

    Args:
        image: Input image
        
    Returns:
        Bounding box as [x_min, y_min, x_max, y_max]
    """
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # threshold to find non-black pixels
    # first param is the threshold value, which we set to 10
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # finding bbox
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    
    if np.any(rows) and np.any(cols):
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return [x_min, y_min, x_max, y_max]
    else:
        # default to full image if no object detecte
        h, w = image.shape[:2]
        return [0, 0, w, h]
