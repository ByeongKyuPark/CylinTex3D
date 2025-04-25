#!/usr/bin/env python3
"""
Two-step script for cylindrical panorama and 3D model creation:
1. Images → Panorama
2. Panorama → Textured 3D Model
"""

import os
import argparse
import time
from src.utils.image_utils import create_directory
from src.segmentation.grabcut import process_images
from src.panorama.stitching import create_panorama
from src.panorama.strip_matching import generate_parameter_combinations
from src.modeling.visual_hull import create_visual_hull
from src.modeling.plane_remover import remove_planar_faces
from src.texturing.texture_mapping import generate_cylindrical_uvs, apply_texture_to_mesh
import trimesh

def step1_images_to_panorama(input_dir, output_dir):
    """
    STEP 1: Create a panorama from input images
    """
    print("\n=== STEP 1: Creating Panorama from Images ===\n")
    start_time = time.time()
    
    # Create output dirs
    create_directory(output_dir)
    
    # Segment objects
    print("Segmenting objects...")
    process_images(input_dir, output_dir)
    
    # Path to segmented images
    segmented_dir = os.path.join(output_dir, 'segmented', 'segmented')
    
    # Create panorama with default parameters
    print("\nCreating panorama...")
    match_widths = [20, 25, 30, 35, 40]
    center_percents = [0.1, 0.15, 0.2]
    y_offset_range = (-10, 10)
    
    # Generate parameters
    param_combinations = generate_parameter_combinations(
        match_widths,
        center_percents,
        [(y_offset_range[0], y_offset_range[1])]
    )
    
    # Create panorama
    success = create_panorama(
        segmented_dir,
        output_dir,
        param_combinations,
        adjacent_strip_width=45,
        blend_width=5
    )
    
    end_time = time.time()
    panorama_path = os.path.join(output_dir, 'panorama.png')
    
    if success and os.path.exists(panorama_path):
        print(f"\nPanorama creation completed in {end_time - start_time:.2f} seconds")
        print(f"Panorama saved to: {panorama_path}")
        return panorama_path
    else:
        print("\nError: Failed to create panorama!")
        return None

def step2_panorama_to_model(panorama_path, output_dir, segmented_dir=None, 
                          volume=400, subdivide=False, remove_planes=True):
    """
    STEP 2: Create a textured 3D model from panorama
    """
    print("\n=== STEP 2: Creating Textured 3D Model from Panorama ===\n")
    start_time = time.time()
    
    if not os.path.exists(panorama_path):
        print(f"Error: Panorama not found at {panorama_path}")
        return False
    
    # Create output dir
    model_dir = os.path.join(output_dir, '3d_model')
    create_directory(model_dir)
    
    # If segmented_dir not specified, try to find it
    if segmented_dir is None:
        default_path = os.path.join(output_dir, 'segmented', 'segmented')
        if os.path.exists(default_path):
            segmented_dir = default_path
        else:
            print("Error: Segmented images directory not found. Please specify with --segmented_dir")
            return False
    
    # Volume size
    volume_size = (volume, volume, volume)
    
    print(f"Creating 3D model with volume size {volume_size}...")
    
    # Getting segmented images for visual hull
    from src.utils.image_utils import load_images_from_folder
    images, filenames = load_images_from_folder(segmented_dir)
    
    if not images:
        print(f"Error: No segmented images found in {segmented_dir}")
        return False
    
    print(f"Found {len(images)} segmented images for 3D reconstruction")
    
    # Create visual hull mesh
    _, mesh = create_visual_hull(
        images, 
        model_dir, 
        volume_size,
        post_process=True,
        fill_holes_size=100,
        subdivide=subdivide,
        use_poisson=True  # Always use Poisson for best quality
    )
    
    # Remove plane faces if requested
    if remove_planes:
        print("Removing top and bottom plane faces...")
        mesh = remove_planar_faces(mesh, y_threshold=0.85)
        
        # Save cleaned mesh
        cleaned_path = os.path.join(model_dir, 'cleaned_hull.obj')
        mesh.export(cleaned_path)
        print(f"Cleaned mesh saved to {cleaned_path}")
    
    # Generate UV coordinates and apply texture
    print("Applying texture to model...")
    uvs = generate_cylindrical_uvs(mesh, v_scale=0.9, v_offset=0.05)
    textured_path = apply_texture_to_mesh(mesh, uvs, panorama_path, model_dir)
    
    end_time = time.time()
    print(f"\n3D model creation completed in {end_time - start_time:.2f} seconds")
    print(f"Textured model saved to: {textured_path}")
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Create cylindrical panorama and 3D model in two simple steps'
    )
    
    # Common arguments
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory for output files')
    
    # Step selection
    parser.add_argument('--step', type=int, default=0,
                        help='Which step to run: 1=Images→Panorama, 2=Panorama→Model, 0=Both (default)')
    
    # Step 1 arguments
    parser.add_argument('--input_dir', type=str,
                        help='Directory containing input images (required for step 1)')
    
    # Step 2 arguments
    parser.add_argument('--panorama', type=str,
                        help='Path to panorama image (required for step 2 if step 1 is skipped)')
    
    parser.add_argument('--segmented_dir', type=str,
                        help='Directory containing segmented images (optional for step 2)')
    
    parser.add_argument('--volume', type=int, default=400,
                        help='Volume resolution for 3D model (default: 400)')
    
    parser.add_argument('--subdivide', action='store_true',
                        help='Subdivide mesh for higher resolution')
    
    parser.add_argument('--keep_planes', action='store_true',
                        help='Keep top/bottom planes in 3D model')
    
    args = parser.parse_args()
    
    # Validate arguments based on which step we're running
    if args.step == 0 or args.step == 1:
        # Step 1 needs input_dir
        if not args.input_dir:
            print("Error: --input_dir is required for step 1")
            return
    
    if args.step == 2 and not args.step == 0:
        # Step 2 needs panorama path if we're only running step 2
        if not args.panorama:
            print("Error: --panorama is required when running only step 2")
            return
    
    # Process steps
    panorama_path = None
    
    # Step 1: Images to Panorama
    if args.step == 0 or args.step == 1:
        panorama_path = step1_images_to_panorama(args.input_dir, args.output_dir)
    
    # Step 2: Panorama to 3D Model
    if args.step == 0 or args.step == 2:
        # If step 1 was run, use its output; otherwise use provided panorama path
        if args.step == 0:
            if not panorama_path:
                print("Error: Step 1 did not produce a panorama. Cannot continue to step 2.")
                return
        else:
            panorama_path = args.panorama
        
        step2_panorama_to_model(
            panorama_path, 
            args.output_dir, 
            args.segmented_dir,
            args.volume,
            args.subdivide,
            not args.keep_planes  # Default is to remove planes
        )
    
    print("\nDone!")

if __name__ == "__main__":
    main()
