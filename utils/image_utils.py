"""
Image utility functions for loading, preprocessing, and general image operations.
"""

import os
import cv2
import numpy as np


def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_images_from_folder(folder_path):
    """
    Load all images from a folder in sorted order.
    
    Args:
        folder_path: Path to the folder containing images
        
    Returns:
        Tuple of (images, filenames)
    """
    images = []
    filenames = []
    
    # order matters...
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                filenames.append(filename)
    
    return images, filenames


def resize_to_match_height(source_img, target_height):
    """
    Resize an image to match the target height while preserving aspect ratio.
    
    Args:
        source_img: Source image to resize
        target_height: Target height in pixels
        
    Returns:
        Resized image
    """
    if source_img.shape[0] == target_height:
        return source_img
    
    # Calculate new width to maintain aspect ratio
    aspect_ratio = source_img.shape[1] / source_img.shape[0]
    new_width = int(aspect_ratio * target_height)
    
    # Resize image
    resized = cv2.resize(source_img, (new_width, target_height), interpolation=cv2.INTER_AREA)
    return resized


def detect_object_region(image):
    """
    Detect the region containing the object (non-black pixels).
    
    Args:
        image: Input image
        
    Returns:
        Bounding box as [x_min, y_min, x_max, y_max]
    """
    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # threshold to find non-black pixels
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # find bounding box
    rows = np.any(binary, axis=1)
    cols = np.any(binary, axis=0)
    
    if np.any(rows) and np.any(cols):
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return [x_min, y_min, x_max, y_max]
    else:
        # default to full image if no object detected
        h, w = image.shape[:2]
        return [0, 0, w, h]
