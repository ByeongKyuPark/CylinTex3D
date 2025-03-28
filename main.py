#!/usr/bin/env python3
"""
Cylindrical Panorama Creator - Main Script

This script runs the full pipeline:
1. Segment objects from images (if needed)
2. Create panorama from segmented images
"""

import os
import argparse
import time
from utils.image_utils import ensure_dir_exists
from segmentation.grabcut import process_images
from panorama.stitching import create_panorama
from panorama.strip_matching import generate_parameter_combinations


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cylindrical Object Panorama Creator')
    
    parser.add_argument('--input_dir', type=str, default='images',
                        help='Directory containing input images')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory for output files')
    
    parser.add_argument('--skip_segmentation', action='store_true',
                        help='Skip segmentation step (use existing segmented images)')
    
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
    
    return parser.parse_args()


def main():
    """Main execution function."""
    start_time = time.time()
    args = parse_arguments()
    
    # Create output directory
    ensure_dir_exists(args.output_dir)
    
    # Path for segmented images
    segmented_dir = os.path.join(args.output_dir, 'segmented')
    
    # Step 1: Segment objects if needed
    if not args.skip_segmentation:
        print("\n=== Step 1: Segmenting Objects ===")
        process_images(args.input_dir, args.output_dir)
    else:
        print("\n=== Step 1: Skipping Segmentation (using existing images) ===")
    
    # Step 2: Create panorama
    print("\n=== Step 2: Creating Panorama ===")
    
    # Parse parameter values
    match_widths = [int(x) for x in args.match_widths.split(',')]
    center_percents = [float(x) for x in args.center_percents.split(',')]
    y_offset_min, y_offset_max = map(int, args.y_offset_range.split(','))
    
    # Generate parameter combinations
    param_combinations = generate_parameter_combinations(
        match_widths, 
        center_percents, 
        [(y_offset_min, y_offset_max)]
    )
    
    # Create the panorama
    create_panorama(
        segmented_dir,
        args.output_dir,
        param_combinations,
        args.adjacent_width,
        args.blend_width
    )
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
