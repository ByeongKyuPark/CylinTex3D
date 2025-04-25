import os
import cv2
import numpy as np

def create_panorama(segmented_images, output_dir):
    """
    Create a panorama from segmented object images
    
    Args:
        segmented_images: List of paths to segmented object images
        output_dir: Directory to save the panorama output
        
    Returns:
        Path to the generated panorama image
    """
    # Ensure we have images to process
    if len(segmented_images) < 2:
        print("Need at least 2 images to create a panorama")
        return None
    
    # Load images
    images = []
    for img_path in segmented_images:
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    
    # Create a Stitcher object
    try:
        # OpenCV 4.x
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    except:
        # OpenCV 3.x
        stitcher = cv2.createStitcher() if cv2.__version__.startswith('3') else cv2.Stitcher_create()
    
    # Stitch images together
    status, panorama = stitcher.stitch(images)
    
    if status == cv2.Stitcher_OK:
        # Save the panorama
        panorama_path = os.path.join(output_dir, "panorama.jpg")
        cv2.imwrite(panorama_path, panorama)
        print(f"Panorama created successfully and saved to {panorama_path}")
        return panorama_path
    else:
        print(f"Panorama creation failed with status code {status}")
        
        # Try a simple side-by-side stitching as fallback
        print("Attempting simple side-by-side stitching as fallback")
        
        # Get max height and total width
        max_height = max([img.shape[0] for img in images])
        total_width = sum([img.shape[1] for img in images])
        
        # Create a blank canvas for the panorama
        simple_panorama = np.zeros((max_height, total_width, 3), dtype=np.uint8)
        
        # Place each image side by side
        current_x = 0
        for img in images:
            h, w = img.shape[:2]
            simple_panorama[0:h, current_x:current_x+w] = img
            current_x += w
        
        # Save the simple panorama
        simple_panorama_path = os.path.join(output_dir, "simple_panorama.jpg")
        cv2.imwrite(simple_panorama_path, simple_panorama)
        print(f"Simple panorama created and saved to {simple_panorama_path}")
        
        return simple_panorama_path
