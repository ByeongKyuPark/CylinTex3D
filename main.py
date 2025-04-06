#!/usr/bin/env python3
"""
Cylindrical Panorama Creator with 3D Reconstruction - Main Script

This script runs the full pipeline:
1. Segment objects from images (if needed)
2. Create panorama from segmented images
3. Reconstruct 3D model with texture mapping
"""

import os
import argparse
import time
from src.utils.image_utils import create_directory
from src.segmentation.grabcut import process_images
from src.panorama.stitching import create_panorama
from src.panorama.strip_matching import generate_parameter_combinations
from src.modeling.reconstruct_3d import reconstruct_3d_model


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cylindrical Object Panorama Creator with 3D Reconstruction')
    
    parser.add_argument('--input_dir', type=str, default='data/images',
                        help='Directory containing input images')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory for output files')
    
    parser.add_argument('--skip_segmentation', action='store_true',
                        help='Skip segmentation step (use existing segmented images)')
    
    parser.add_argument('--skip_panorama', action='store_true',
                        help='Skip panorama creation (use existing panorama)')
    
    parser.add_argument('--skip_3d', action='store_true',
                        help='Skip 3D reconstruction')
    
    parser.add_argument('--skip_hull', action='store_true',
                        help='Skip visual hull creation (just retexture existing model)')
    
    parser.add_argument('--model_path', type=str, default='results/3d_model/visual_hull.obj',
                        help='Path to existing model when using --skip_hull')
    
    parser.add_argument('--match_widths', type=str, default='20,25,30,35,40',
                        help='Comma-separated list of match widths to try')
    
    parser.add_argument('--center_percents', type=str, default='0.1,0.15,0.2',
                        help='Comma-separated list of center region percentages to try')
    
    parser.add_argument('--y_offset_range', type=str, default='-10,10',
                        help='Y-offset range to try (min,max)')
    
    parser.add_argument('--adjacent_width', type=int, default=45,
                        help='Width of adjacent strips to add to panorama')
    
    parser.add_argument('--blend_width', type=int, default=5,
                        help='Width of blending region')
    
    parser.add_argument('--volume_size', type=str, default='200,200,200',
                        help='Size of 3D volume for reconstruction (x,y,z)')
    
    parser.add_argument('--remove_planes', action='store_true',
                        help='Remove planar faces from 3D model')
        
    parser.add_argument('--plane_threshold', type=float, default=0.8,
                        help='Threshold for plane detection [0-1]')

    parser.add_argument('--texture_v_scale', type=float, default=0.9,
                        help='Scale factor for vertical texture mapping [0-1]')
    
    parser.add_argument('--texture_v_offset', type=float, default=0.05,
                        help='Offset for vertical texture mapping [0-1]')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    start_time = time.time()
    args = parse_arguments()
    
    # create output dir to save results
    create_directory(args.output_dir)
    
    # paths for "Segmented" and "Panorama" directories
    segmented_dir = os.path.join(args.output_dir, 'segmented')
    panorama_path = os.path.join(args.output_dir, 'panorama.png')
    
    # Step 1: segment objects (if not skipped)
    if not args.skip_segmentation:
        print("\n=== Step 1: Segmenting Objects ===")
        process_images(args.input_dir, args.output_dir)
    else:
        print("\n=== Step 1: Skipping Segmentation (using existing images) ===")
    
    # Step 2: create panorama (if not skipped)
    if not args.skip_panorama:
        print("\n=== Step 2: Creating Panorama ===")
        
        # parse parameter values

        # 1. how "WIDE" the matching strips are
        # "0.1,0.15,0.2" → list [0.1, 0.15, 0.2]
        match_widths = [int(x) for x in args.match_widths.split(',')]
        # 2. how "CENTERED" the matching strips are e.g. 0.1 = += 10% of the object center
        center_percents = [float(x) for x in args.center_percents.split(',')]
        # 3. how far up/down the matching strips are
        # e.g. -10,10 = [-10, 10] pixels
        y_offset_min, y_offset_max = map(int, args.y_offset_range.split(','))
        
        # mix all combinations
        param_combinations = generate_parameter_combinations(
            match_widths, 
            center_percents,
            [(y_offset_min, y_offset_max)]
        )
        
        # create the panorama
        create_panorama(
            segmented_dir,
            args.output_dir,
            param_combinations,
            args.adjacent_width,
            args.blend_width
        )
    else:
        print("\n=== Step 2: Skipping Panorama Creation (using existing panorama) ===")
    
    # Step 3: reconstruct 3D model with texture mapping (if not skipped)
    if not args.skip_3d:
        print("\n=== Step 3: Reconstructing 3D Model ===")
        
        # parse volume size

        # (1) split returns a list of strings
        #   e.g. "200,200,200" → ["200", "200", "200"]
        # (2) 'map' here is a function convert the string to int
        #   e.g. "200,200,200" → (200, 200, 200)
        #   not DATA STRUCTURE such as unsored_map or such.
        # (3) tuple to make it immutable (const)
        volume_size = tuple(map(int, args.volume_size.split(',')))
        
        # create 3D model with texture mapping
        reconstruct_3d_model(
            segmented_dir,
            panorama_path,
            args.output_dir,
            volume_size,
            texture_v_scale=args.texture_v_scale,
            texture_v_offset=args.texture_v_offset,
            skip_hull=args.skip_hull,
            model_path=args.model_path,
            remove_planes=args.remove_planes,
            plane_threshold=args.plane_threshold
        )
    else:
        print("\n=== Step 3: Skipping 3D Reconstruction ===")
    
    end_time = time.time()
    
    # benchmarking
    print("\n=== Benchmarking ===")
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()