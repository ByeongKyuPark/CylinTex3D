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

def remove_planar_faces(mesh, y_threshold=0.85, top_bottom_percent=0.05):
    """
    Remove top and bottom planes from a cylindrical mesh using height-based clamping.
    
    Args:
        mesh: trimesh.Trimesh object
        y_threshold: Not used in this implementation, kept for API compatibility
        top_bottom_percent: Percentage of top and bottom to remove (0.05 = 5%)
    
    Returns:
        trimesh.Trimesh object with top and bottom planes removed
    """
    # Get all vertex heights (y-coordinates)
    heights = mesh.vertices[:, 1]
    
    # Find min and max heights
    y_min, y_max = np.min(heights), np.max(heights)
    height_range = y_max - y_min
    
    print(f"Model height range: {height_range:.2f} units from {y_min:.2f} to {y_max:.2f}")
    
    # Calculate cutoff thresholds
    bottom_cutoff = y_min + (height_range * top_bottom_percent)
    top_cutoff = y_max - (height_range * top_bottom_percent)
    
    print(f"Removing vertices below {bottom_cutoff:.2f} and above {top_cutoff:.2f}")
    
    # Create a mask for faces to keep (all vertices must be within range)
    faces_to_keep = []
    removed_count = 0
    
    for face_idx, face in enumerate(mesh.faces):
        face_vertices = mesh.vertices[face]
        face_heights = face_vertices[:, 1]
        
        # Keep face only if all its vertices are within the height range
        if np.all(face_heights > bottom_cutoff) and np.all(face_heights < top_cutoff):
            faces_to_keep.append(face_idx)
        else:
            removed_count += 1
    
    print(f"Removing {removed_count} faces out of {len(mesh.faces)} total faces")
    
    if removed_count > 0:
        # Create a new mesh with only the kept faces
        new_faces = mesh.faces[faces_to_keep]
        new_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=new_faces)
        
        # Clean up the mesh (remove unreferenced vertices, etc.)
        new_mesh = new_mesh.process()
        
        return new_mesh
    else:
        print("No faces were removed")
        return mesh

def main():
    parser = argparse.ArgumentParser(description='Remove planar faces from a 3D model')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input 3D model file')
    
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save cleaned 3D model file')
    
    parser.add_argument('--threshold', type=float, default=0.85,
                        help='Threshold for Y component of normals [0-1]. Higher values remove fewer faces.')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_dir_exists(os.path.dirname(args.output))
    
    # Load mesh
    print(f"Loading mesh from {args.input}")
    mesh = trimesh.load(args.input)
    
    # Remove planes
    cleaned_mesh = remove_planar_faces(mesh, args.threshold)
    
    # Save the result
    cleaned_mesh.export(args.output)
    print(f"Cleaned mesh saved to {args.output}")
    
    print("Done!")

if __name__ == "__main__":
    main()