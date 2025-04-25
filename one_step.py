#!/usr/bin/env python3
"""
One-step script to create a cylindrical panorama and 3D model from images
with optimized defaults.
"""

import os
import argparse
import time
from src.utils.image_utils import create_directory
from src.segmentation.grabcut import process_images
from src.panorama.stitching import create_panorama
from src.panorama.strip_matching import generate_parameter_combinations
from src.modeling.reconstruct_3d import reconstruct_3d_model

def main():
    """Main execution function with simplified options."""
    parser = argparse.ArgumentParser(
        description='Create a cylindrical panorama and 3D model in one step')
    
    # Simplified options
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input images')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory for output files')
    
    parser.add_argument('--volume', type=int, default=400, 
                        help='Resolution of 3D volume (default: 400)')
    
    parser.add_argument('--quality', choices=['low', 'medium', 'high', 'ultra'], default='high',
                        help='Overall quality setting (default: high)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Set quality presets
    if args.quality == 'low':
        volume_size = (200, 200, 200)
        use_poisson = False
        subdivide = False
    elif args.quality == 'medium':
        volume_size = (300, 300, 300)
        use_poisson = True
        subdivide = False
    elif args.quality == 'high':
        volume_size = (args.volume, args.volume, args.volume)
        use_poisson = True
        subdivide = False
    else:  # ultra
        volume_size = (args.volume, args.volume, args.volume)
        use_poisson = True
        subdivide = True
    
    # Create output directory
    create_directory(args.output_dir)
    
    # STEP 1: Segment objects
    print("\n=== Step 1: Segmenting Objects ===")
    process_images(args.input_dir, args.output_dir)
    
    # Segmented directory path
    segmented_dir = os.path.join(args.output_dir, 'segmented', 'segmented')
    panorama_path = os.path.join(args.output_dir, 'panorama.png')
    
    # STEP 2: Create panorama
    print("\n=== Step 2: Creating Panorama ===")
    match_widths = [20, 25, 30, 35, 40]
    center_percents = [0.1, 0.15, 0.2]
    y_offset_range = (-10, 10)
    
    # Create parameter combinations
    param_combinations = generate_parameter_combinations(
        match_widths,
        center_percents,
        [y_offset_range]
    )
    
    # Create panorama
    create_panorama(
        segmented_dir,
        args.output_dir,
        param_combinations,
        adjacent_strip_width=45,
        blend_width=5
    )
    
    # STEP 3: Reconstruct 3D model
    print("\n=== Step 3: Reconstructing 3D Model ===")
    reconstruct_3d_model(
        segmented_dir,
        panorama_path,
        args.output_dir,
        volume_size,
        texture_v_scale=0.9,
        texture_v_offset=0.05,
        skip_hull=False,
        model_path=None,
        remove_planes=True,  # Always remove planes
        plane_threshold=0.85,
        post_process=True,
        fill_holes_size=100,
        subdivide=subdivide,
        use_poisson=use_poisson
    )
    
    end_time = time.time()
    
    print("\n=== Process Complete! ===")
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to: {args.output_dir}")
    print("\nYou can view the 3D model at: " + os.path.join(args.output_dir, '3d_model', 'textured_model.obj'))
    print("Panorama image: " + panorama_path)

if __name__ == "__main__":
    main()
