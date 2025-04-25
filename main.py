#!/usr/bin/env python3
"""
Object Panorama Generation Pipeline - Main Script

This script runs the full pipeline:
1. Segment objects from images
2. Create panorama from segmented images
3. Generate 3D model from segmented objects
4. Apply textures to the 3D model
"""

import os
import argparse
from segmentation import segment_objects
from panorama import create_panorama
from modeling import create_3d_model
from texturing import apply_textures

def main():
    parser = argparse.ArgumentParser(description='Object Panorama Generation Pipeline')
    parser.add_argument('--input_dir', required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', required=True, help='Directory for output files')
    parser.add_argument('--skip_steps', nargs='+', choices=['segmentation', 'panorama', 'modeling', 'texturing'],
                        help='Steps to skip in the pipeline', default=[])
    args = parser.parse_args()
    
    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    segmented_dir = os.path.join(args.output_dir, 'segmented')
    panorama_dir = os.path.join(args.output_dir, 'panorama')
    model_dir = os.path.join(args.output_dir, 'model')
    textured_dir = os.path.join(args.output_dir, 'textured')
    
    for directory in [segmented_dir, panorama_dir, model_dir, textured_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Step 1: Photos → Segmentation
    if 'segmentation' not in args.skip_steps:
        print("Step 1: Segmenting objects from images...")
        segmented_images = segment_objects(args.input_dir, segmented_dir)
    else:
        segmented_images = [os.path.join(segmented_dir, f) for f in os.listdir(segmented_dir) if f.endswith(('.jpg', '.png'))]
    
    # Step 2: Segmented objects → Panorama
    if 'panorama' not in args.skip_steps:
        print("Step 2: Creating panorama from segmented objects...")
        panorama_image = create_panorama(segmented_images, panorama_dir)
    
    # Step 3: Segmented objects → 3D Model (with plane removal)
    if 'modeling' not in args.skip_steps:
        print("Step 3: Generating 3D model from segmented objects...")
        model_path = create_3d_model(segmented_images, model_dir)
    else:
        model_path = os.path.join(model_dir, 'model.obj')  # Assume default name
    
    # Step 4: Apply texturing to the model
    if 'texturing' not in args.skip_steps:
        print("Step 4: Applying textures to the 3D model...")
        textured_model = apply_textures(model_path, segmented_images, textured_dir)
        print(f"Textured model saved to: {textured_model}")
    
    print("Processing complete!")

if __name__ == "__main__":
    main()