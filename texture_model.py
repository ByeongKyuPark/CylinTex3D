#!/usr/bin/env python3
"""
Utility to reapply texture to an existing 3D model
"""

import os
import argparse
import trimesh
import numpy as np
from src.texturing.texture_mapping import generate_cylindrical_uvs, apply_texture_to_mesh
from src.utils.image_utils import create_directory
from src.modeling.plane_remover import remove_planar_faces

def main():
    """
        instead of reconstructing a new model, this script allows you to apply a new texture mapping to an existing 3D model.
    """
    parser = argparse.ArgumentParser(description='Apply new texture mapping to existing 3D model')
    
    parser.add_argument('--model_path', type=str, default='results/3d_model/visual_hull.obj',
                        help='Path to existing model OBJ file')
    
    parser.add_argument('--panorama_path', type=str, default='results/panorama.png',
                        help='Path to panorama image')
    
    parser.add_argument('--output_dir', type=str, default='results/3d_model',
                        help='Directory to save textured model')
    
    parser.add_argument('--texture_v_scale', type=float, default=0.9,
                        help='Scale factor for vertical texture mapping [0-1]')
    
    parser.add_argument('--texture_v_offset', type=float, default=0.05,
                        help='Offset for vertical texture mapping [0-1]')
    
    parser.add_argument('--remove_planes', action='store_true',
                        help='Remove planar faces from 3D model')
        
    parser.add_argument('--plane_threshold', type=float, default=0.8,
                        help='Threshold for plane detection [0-1]')
    
    # Add mesh improvement options
    parser.add_argument('--post_process', action='store_true',
                        help='Apply mesh post-processing to improve quality')
    
    parser.add_argument('--smooth_iterations', type=int, default=5,
                        help='Number of smoothing iterations')
    
    parser.add_argument('--fill_holes_size', type=int, default=100,
                        help='Maximum size of holes to fill')
    
    parser.add_argument('--subdivide', action='store_true',
                        help='Subdivide mesh for higher resolution')
    
    # Add option to regenerate UV coordinates or reuse existing
    parser.add_argument('--reuse_uvs', action='store_true',
                        help='Reuse existing UV coordinates if available')
    
    args = parser.parse_args()
    
    # validate input files
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model file not found: {args.model_path}")
        return
    
    if not os.path.exists(args.panorama_path):
        print(f"ERROR: Panorama file not found: {args.panorama_path}")
        return
    
    # create output directory if it doesn't exist
    create_directory(args.output_dir)
    
    # load existing mesh
    print(f"Loading mesh from {args.model_path}")
    mesh = trimesh.load(args.model_path)
    if mesh is None:
        print(f"ERROR: Failed to load mesh file")
        return
    
    # Apply post-processing if requested
    if args.post_process:
        try:
            from src.modeling.mesh_processing import improve_mesh
            print("Applying mesh post-processing...")
            mesh = improve_mesh(
                mesh,
                smooth_iterations=args.smooth_iterations,
                fill_holes_size=args.fill_holes_size,
                subdivide=args.subdivide
            )
        except ImportError:
            print("Warning: Mesh processing module not available. Using raw mesh.")
    
    # Remove planes if requested
    if args.remove_planes:
        print(f"Removing planar faces with threshold {args.plane_threshold}...")
        mesh = remove_planar_faces(mesh, args.plane_threshold)
    
    # apply new texture mapping
    print(f"Applying new texture mapping with scale={args.texture_v_scale}, offset={args.texture_v_offset}")
    
    # generate "UV coordinates"
    uvs = generate_cylindrical_uvs(mesh, args.texture_v_scale, args.texture_v_offset)
    
    # apply "texture"
    textured_mesh_path = apply_texture_to_mesh(mesh, uvs, args.panorama_path, args.output_dir)
        
    print(f"Retextured mesh saved to {textured_mesh_path}")

if __name__ == "__main__":
    main()