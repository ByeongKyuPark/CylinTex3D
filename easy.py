#!/usr/bin/env python3
"""
Simplified workflow for cylindrical panorama and 3D model creation.
Just run with no parameters to execute the entire pipeline with highest quality.
"""

import os
import sys
import time
from src.utils.image_utils import create_directory, load_images_from_folder
from src.segmentation.grabcut import process_images
from src.panorama.stitching import create_panorama
from src.panorama.strip_matching import generate_parameter_combinations
from src.modeling.visual_hull import create_visual_hull
from src.modeling.plane_remover import remove_planar_faces
from src.texturing.texture_mapping import generate_cylindrical_uvs, apply_texture_to_mesh

def find_input_directory():
    """Find input directory with images automatically"""
    # Try common locations
    potential_paths = [
        "data/images",
        "images",
        "data",
        "input"
    ]
    
    for path in potential_paths:
        if os.path.exists(path) and any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))):
            print(f"Found images in: {path}")
            return path
    
    # If not found, ask user
    print("Could not find input images directory.")
    path = input("Enter the path to your images: ")
    if os.path.exists(path):
        return path
    else:
        print(f"Error: Directory {path} not found")
        sys.exit(1)

def step1_create_panorama():
    """Create panorama from images"""
    print("\n=== Creating Panorama ===\n")
    start_time = time.time()
    
    # Find input images
    input_dir = find_input_directory()
    
    # Create output directories
    output_dir = "output"
    create_directory(output_dir)
    
    # Segment objects
    print("Segmenting objects...")
    process_images(input_dir, output_dir)
    
    # Path to segmented images
    segmented_dir = os.path.join(output_dir, 'segmented', 'segmented')
    
    # Create panorama with best parameters
    print("\nCreating panorama...")
    match_widths = [20, 25, 30, 35, 40]
    center_percents = [0.1, 0.15, 0.2]
    y_offset_range = (-10, 10)
    
    param_combinations = generate_parameter_combinations(
        match_widths,
        center_percents,
        [(y_offset_range[0], y_offset_range[1])]
    )
    
    create_panorama(
        segmented_dir,
        output_dir,
        param_combinations,
        adjacent_strip_width=45,
        blend_width=5
    )
    
    end_time = time.time()
    panorama_path = os.path.join(output_dir, 'panorama.png')
    
    if os.path.exists(panorama_path):
        print(f"\nPanorama created in {end_time - start_time:.2f} seconds")
        print(f"Panorama saved to: {panorama_path}")
        return panorama_path, segmented_dir, output_dir
    else:
        print("\nError: Failed to create panorama!")
        return None, segmented_dir, output_dir

def step2_create_model(panorama_path, segmented_dir, output_dir):
    """Create 3D model from panorama"""
    print("\n=== Creating 3D Model ===\n")
    start_time = time.time()
    
    # Create model directory
    model_dir = os.path.join(output_dir, '3d_model')
    create_directory(model_dir)
    
    # Set highest quality settings
    volume_size = (500, 500, 500)
    
    # Load segmented images for visual hull
    images, _ = load_images_from_folder(segmented_dir)
    
    if not images:
        print(f"Error: No segmented images found in {segmented_dir}")
        return False
    
    print(f"Creating high-quality 3D model with volume size {volume_size}...")
    
    # Create visual hull with highest quality settings
    _, mesh = create_visual_hull(
        images, 
        model_dir, 
        volume_size,
        post_process=True,
        fill_holes_size=100,
        subdivide=True,
        use_poisson=True
    )
    
    # Remove planes
    print("Removing top and bottom faces...")
    mesh = remove_planar_faces(mesh, y_threshold=0.85)
    
    # Generate UV coordinates and apply texture
    print("Applying texture mapping...")
    uvs = generate_cylindrical_uvs(mesh, v_scale=0.9, v_offset=0.05)
    textured_path = apply_texture_to_mesh(mesh, uvs, panorama_path, model_dir)
    
    end_time = time.time()
    print(f"\n3D model created in {end_time - start_time:.2f} seconds")
    print(f"Textured model saved to: {textured_path}")
    return True

def main():
    """Main function - run the entire pipeline"""
    total_start_time = time.time()
    
    # Step 1: Create panorama
    panorama_path, segmented_dir, output_dir = step1_create_panorama()
    if not panorama_path:
        return
    
    # Step 2: Create 3D model
    step2_create_model(panorama_path, segmented_dir, output_dir)
    
    total_end_time = time.time()
    print(f"\nTotal processing time: {total_end_time - total_start_time:.2f} seconds")
    print(f"All results saved to: {output_dir}")
    print("\nDone!")

if __name__ == "__main__":
    main()
