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
        print(f"ERROR: Failed to load panorama image")
        return
    
    # apply new texture mapping
    print(f"Applying new texture mapping with scale={args.texture_v_scale}, offset={args.texture_v_offset}")
    
    # generate "UV coordinates"
    uvs = generate_cylindrical_uvs(mesh, args.texture_v_scale, args.texture_v_offset)
    
    # apply "texture"
    textured_mesh_path = apply_texture_to_mesh(mesh, uvs, args.panorama_path, args.output_dir)
        
    print(f"Retextured mesh saved to {textured_mesh_path}")

if __name__ == "__main__":
    main()