"""
Texture Mapping module - Maps panorama texture to 3D model.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import trimesh
import math
from .obj_exporter import export_textured_obj

def generate_cylindrical_uvs(mesh, v_scale=0.9, v_offset=0.05):
    """
    generate UV coordinates using cylindrical mapping.
    note this function requires just a little bit of basic Computer Graphics knowledge
    to understand the mapping process. 

    Args:
        mesh: Trimesh object
        v_scale: Scale factor for vertical texture mapping [0-1] (default: 0.9)
        v_offset: Offset for vertical texture mapping [0-1] (default: 0.05)
        
    Returns:
        Array of UV coordinates for each vertex
    """
    vertices = mesh.vertices
    
    # first we align the object to the centroid
    # and then we find the axis-aligned bounding box
    center = mesh.centroid
    
    # normalize vertices to center
    centered_verts = vertices - center
    
    # generate UV coordinates using cylindrical mapping
    uvs = np.zeros((len(centered_verts), 2))
    
    # find min and max height for v-coordinate normalization
    min_y = np.min(centered_verts[:, 1])
    max_y = np.max(centered_verts[:, 1])
    height_range = max_y - min_y
    
    print("Generating UV coordinates...")
    for i, centered_vert in enumerate(tqdm(centered_verts)):
        x, y, z = centered_vert
        
        # O-mapping: object to intermediate surface
        theta = math.atan2(z, x)
        
        # S-mapping: cylindrical surface to texture coordinates
        u = ((theta + math.pi) / (2 * math.pi))
        
        # here we scale v coordinates based on the provided parameters
        v_raw = (y - min_y) / height_range if height_range > 0 else 0.5
        v = v_offset + (v_raw * v_scale)
        
        # Ensure v is within [0,1]
        v = min(max(v, 0.0), 1.0)
        
        uvs[i] = [u, v]
    
    return uvs

def apply_texture_to_mesh(mesh, uvs, panorama_path, output_dir):
    """
    apply panorama texture to mesh using the provided UV coordinates.
    
    Args:
        mesh: Trimesh object
        uvs: UV coordinates for each vertex
        panorama_path: Path to panorama image
        output_dir: Directory to save the textured mesh
        
    Returns:
        Path to saved textured mesh
    """
    
    # load panorama image
    texture = cv2.imread(panorama_path)
    if texture is None:
        raise ValueError(f"Failed to load panorama image from {panorama_path}")
    
    # if very larage, resize for performance reasons
    max_texture_size = 4096
    h, w = texture.shape[:2]
    if max(h, w) > max_texture_size:
        scale = max_texture_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        texture = cv2.resize(texture, (new_w, new_h))
    
    # save texture image
    texture_path = os.path.join(output_dir, 'texture.png')
    cv2.imwrite(texture_path, texture)
    
    # export mesh using our custom OBJ exporter
    obj_path, mtl_path = export_textured_obj(
        vertices=mesh.vertices,
        faces=mesh.faces,
        uvs=uvs,
        texture_path=texture_path,
        output_dir=output_dir,
        prefix="textured_model"
    )
    
    # also export the same mesh without texture for comparison
    mesh.export(os.path.join(output_dir, 'visual_hull.obj'))
    
    print(f"Textured mesh saved to {obj_path}")
    print(f"Material file saved to {mtl_path}")
    return obj_path

def apply_cylindrical_texture_mapping(mesh, panorama_path, output_dir, texture_v_scale=0.9, texture_v_offset=0.05):
    """
    apply cylindrical texture mapping to a 3D mesh using a panorama image.
    
    Args:
        mesh: Trimesh object
        panorama_path: Path to panorama image
        output_dir: Directory to save the textured mesh
        texture_v_scale: Scale factor for vertical texture mapping [0-1] (default: 0.9)
        texture_v_offset: Offset for vertical texture mapping [0-1] (default: 0.05)
        
    Returns:
        Path to saved textured mesh
    """
    # generate UV coordinates (using cylindrical mapping for now)
    uvs = generate_cylindrical_uvs(mesh, texture_v_scale, texture_v_offset)
    
    # mesh <- apply texture
    textured_mesh_path = apply_texture_to_mesh(mesh, uvs, panorama_path, output_dir)
    
    return textured_mesh_path