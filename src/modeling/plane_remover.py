#!/usr/bin/env python3
"""
Tool to remove planar (top/bottom) faces from a 3D model
"""

import os
import argparse
import trimesh
import numpy as np

def ensure_dir_exists(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory

def remove_planar_faces(mesh, y_threshold=0.9, output_path=None):
    """
    Remove planar faces (like top and bottom caps) from a mesh.
    
    Args:
        mesh: Trimesh object
        y_threshold: Threshold for Y component of normals [0-1]
        output_path: Path to save the cleaned mesh
        
    Returns:
        Trimesh object with planes removed
    """
    # Get face normals
    normals = mesh.face_normals
    
    # Calculate Y magnitude (absolute value)
    y_magnitude = np.abs(normals[:, 1])
    
    # Identify planes based on Y component
    planes_mask = y_magnitude > y_threshold
    
    # Keep only non-plane faces
    faces_to_keep = ~planes_mask
    new_faces = mesh.faces[faces_to_keep]
    
    if len(new_faces) < len(mesh.faces):
        print(f"Removing {len(mesh.faces) - len(new_faces)} planar faces from mesh")
        
        # Create a new mesh with only the desired faces
        new_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=new_faces)
        
        # Clean up the mesh (remove unreferenced vertices, etc.)
        new_mesh = new_mesh.process()
        
        # Save the cleaned mesh if requested
        if output_path:
            new_mesh.export(output_path)
            print(f"Cleaned mesh saved to {output_path}")
        
        return new_mesh
    else:
        print("No planes found matching the criteria")
        return mesh

def main():
    parser = argparse.ArgumentParser(description='Remove planar faces from a 3D model')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input 3D model file')
    
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save cleaned 3D model file')
    
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Threshold for Y component of normals [0-1]. Higher values remove fewer faces.')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_dir_exists(os.path.dirname(args.output))
    
    # Load mesh
    print(f"Loading mesh from {args.input}")
    mesh = trimesh.load(args.input)
    
    # Remove planes
    remove_planar_faces(mesh, args.threshold, args.output)
    
    print("Done!")

if __name__ == "__main__":
    main()