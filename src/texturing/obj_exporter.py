"""
OBJ Exporter - Explicitly creates textured OBJ files with proper MTL references.
"""

import os
import numpy as np

def export_textured_obj(vertices, faces, uvs, texture_path, output_dir, prefix="textured_model"):
    """
    Export a textured mesh to OBJ format with proper MTL references.
    
    Args:
        vertices: Numpy array of vertices (Nx3)
        faces: Numpy array of face indices (Nx3)
        uvs: Numpy array of UV coordinates (Nx2)
        texture_path: Path to texture image
        output_dir: Directory to save files
        prefix: Prefix for output files
        
    Returns:
        Tuple of (obj_path, mtl_path)
    """
    # get just the filename part of the texture path
    texture_filename = os.path.basename(texture_path)
    
    # create MTL file
    mtl_path = os.path.join(output_dir, f"{prefix}.mtl")
    with open(mtl_path, 'w') as f:
        f.write(f"# Material file for {prefix}.obj\n\n")
        f.write("newmtl material_0\n")
        f.write("Ka 1.000 1.000 1.000\n")  # ambient color
        f.write("Kd 1.000 1.000 1.000\n")  # diffuse color
        f.write("Ks 0.000 0.000 0.000\n")  # specular color
        f.write("d 1.0\n")                 # transparency
        f.write("illum 2\n")               # illumination model
        f.write(f"map_Kd {texture_filename}\n")  # texture map
    
    # Create OBJ file
    obj_path = os.path.join(output_dir, f"{prefix}.obj")
    with open(obj_path, 'w') as f:
        # Header
        f.write(f"# Textured OBJ file\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")
        
        # Material reference
        f.write(f"mtllib {prefix}.mtl\n\n")
        
        # Vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        # UV coordinates
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        
        f.write("\n")
        
        # Faces with UV mapping
        f.write("usemtl material_0\n")
        for i, face in enumerate(faces):
            # OBJ is 1-indexed
            # Format is: f v1/vt1 v2/vt2 v3/vt3
            f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")
    
    return obj_path, mtl_path