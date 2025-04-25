import os
import cv2
import numpy as np
from pathlib import Path
import torch

def segment_objects(input_dir, output_dir):
    """
    Extract objects from input images using segmentation
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save segmented images
        
    Returns:
        List of paths to segmented object images
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load segmentation model (e.g., using Detectron2 or other segmentation models)
    try:
        # Try to import and use Detectron2 if available
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog
        
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)
        
        use_detectron = True
    except ImportError:
        print("Detectron2 not found, using a simplified segmentation approach")
        use_detectron = False
    
    segmented_paths = []
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        
        if use_detectron:
            # Use Detectron2 for segmentation
            outputs = predictor(img)
            instances = outputs["instances"].to("cpu")
            
            if len(instances) > 0:
                # Get the mask of the most confident object
                mask = instances.pred_masks[0].numpy().astype(np.uint8) * 255
                
                # Apply mask to extract object
                masked_img = cv2.bitwise_and(img, img, mask=mask)
                
                # Save the segmented image
                output_path = os.path.join(output_dir, f"segmented_{img_file}")
                cv2.imwrite(output_path, masked_img)
                segmented_paths.append(output_path)
        else:
            # Simplified segmentation using GrabCut or other methods
            mask = np.zeros(img.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Assuming the object is roughly in the center
            rect = (img.shape[1]//4, img.shape[0]//4, img.shape[1]//2, img.shape[0]//2)
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Get the foreground mask
            mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
            
            # Apply mask to extract object
            masked_img = img * mask2[:, :, np.newaxis]
            
            # Save the segmented image
            output_path = os.path.join(output_dir, f"segmented_{img_file}")
            cv2.imwrite(output_path, masked_img)
            segmented_paths.append(output_path)
    
    print(f"Segmented {len(segmented_paths)} objects from {len(image_files)} images")
    return segmented_paths
