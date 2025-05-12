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
from src.utils.image_utils import create_directory

def generate_cylindrical_uvs(mesh, v_scale=0.9, v_offset=0.05):
    """
    generate UV coordinates using cylindrical mapping.
    note this function requires just a little bit of basic Computer Graphics knowledge
    to understand the mapping process. 
    MODIFIED to be vectorized.

    Args:
        mesh: Trimesh object
        v_scale: Scale factor for vertical texture mapping [0-1] (default: 0.9)
        v_offset: Offset for vertical texture mapping [0-1] (default: 0.05)
        
    Returns:
        Array of UV coordinates for each vertex, or empty array on failure
    """
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        print("Warning: Invalid or empty mesh provided to generate_cylindrical_uvs.")
        return np.array([])


    vertices = mesh.vertices.astype(np.float32) # Ensure float for precision
    
    # first we align the object to the centroid for U coordinate calculation
    center = mesh.centroid.astype(np.float32)
    centered_verts_for_u = vertices - center # For theta calculation
    
    # For V coordinate normalization, use the original mesh's Y-bounds
    # This ensures V covers the object's full extent in its original pose.
    min_y_obj_bounds = mesh.bounds[0, 1]
    max_y_obj_bounds = mesh.bounds[1, 1]
    height_range_obj = max_y_obj_bounds - min_y_obj_bounds
    
    if height_range_obj < 1e-6: # Avoid division by zero if object is flat
        height_range_obj = 1.0 
        # print("Warning: Object Y-range is near zero for V-coordinate normalization.")

    # Original loop for UV generation:
    # uvs = np.zeros((len(centered_verts), 2))
    # # find min and max height for v-coordinate normalization (from centered_verts)
    # min_y = np.min(centered_verts[:, 1]) # This was from centered_verts
    # max_y = np.max(centered_verts[:, 1]) # This was from centered_verts
    # height_range = max_y - min_y # This was from centered_verts
    # if height_range <= 1e-6: height_range = 1.0 # Avoid division by zero

    # print("Generating UV coordinates (original loop)...") # Original print
    # for i, centered_vert_loop in enumerate(tqdm(centered_verts)): # Original tqdm
    #     x_loop, y_loop, z_loop = centered_vert_loop # y_loop is from centered_verts
        
    #     # O-mapping: object to intermediate surface
    #     theta_loop = math.atan2(z_loop, x_loop) # Output is in [-pi, pi]
        
    #     # S-mapping: cylindrical surface to texture coordinates
    #     u_loop = ((theta_loop + math.pi) / (2 * math.pi)) # Normalize to [0, 1]
        
    #     # here we scale v coordinates based on the provided parameters
    #     # v_raw_loop used y_loop from centered_verts and height_range from centered_verts' Y span
    #     v_raw_loop = (y_loop - min_y) / height_range 
    #     v_loop = v_offset + (v_raw_loop * v_scale)
        
    #     # Ensure v is within [0,1]
    #     v_loop = min(max(v_loop, 0.0), 1.0)
        
    #     uvs[i] = [u_loop, v_loop] # Original V was not flipped (0 at bottom, 1 at top)
    # return uvs # End of original loop implementation

    # OPTIMIZED Vectorized UV generation:
    # print("Generating UV coordinates (vectorized)...") # tqdm might not be needed if fast

    # U coordinate (azimuthal angle based on centered X, Z)
    x_coords_centered = centered_verts_for_u[:, 0]
    z_coords_centered = centered_verts_for_u[:, 2]
    theta_vals = np.arctan2(z_coords_centered, x_coords_centered) # Range: -pi to pi
    u_coords_vec = (theta_vals + np.pi) / (2 * np.pi)         # Normalize to [0, 1]
    u_coords_vec = np.clip(u_coords_vec, 0.0, 1.0)            # Ensure strict range

    # V coordinate (normalized height based on original vertex Y and object bounds)
    # Uses original `vertices[:, 1]` for height relative to object's true min/max Y.
    v_raw_vec = (vertices[:, 1] - min_y_obj_bounds) / height_range_obj
    v_coords_vec = v_offset + (v_raw_vec * v_scale)
    v_coords_vec = np.clip(v_coords_vec, 0.0, 1.0) # Ensure strict range [0,1]

    # Combine U and V.
    # Textures usually have (0,0) at top-left.
    # Cylindrical V often goes from bottom (0) to top (1). So, flip V: 1.0 - v_coords_vec.
    # If your original code's V (0 at bottom, 1 at top) was correct for your texture, then don't flip.
    # Assuming standard texture convention (V downwards), so flip.
    uvs_vectorized = np.stack([u_coords_vec, 1.0 - v_coords_vec], axis=-1)
    
    return uvs_vectorized


def apply_texture_to_mesh(mesh, uvs, panorama_path, output_dir):
    """
    apply panorama texture to mesh using the provided UV coordinates.
    
    Args:
        mesh: Trimesh object
        uvs: UV coordinates for each vertex
        panorama_path: Path to panorama image (e.g., "results/panorama.png")
        output_dir: Directory to save the textured mesh and texture (e.g., "results/3d_model/textured_model/")
        
    Returns:
        Path to saved textured mesh OBJ file, or None on failure
    """
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        print("Error: Mesh is invalid in apply_texture_to_mesh.")
        return None
    if uvs is None or uvs.shape[0] != len(mesh.vertices):
        print(f"Error: UVs are invalid or mismatch vertex count. Verts: {len(mesh.vertices)}, UVs: {len(uvs) if uvs is not None else 'None'}.")
        return None
    
    # load panorama image
    texture_img_orig = cv2.imread(panorama_path)
    if texture_img_orig is None:
        # Original code raised ValueError. For robustness, print error and return None.
        print(f"Failed to load panorama image from {panorama_path}")
        return None 
    
    # if very large, resize for performance reasons
    max_texture_size = 4096 # Configurable
    h_tex, w_tex = texture_img_orig.shape[:2]
    texture_to_save = texture_img_orig.copy() # Work with a copy
    if max(h_tex, w_tex) > max_texture_size:
        scale = max_texture_size / float(max(h_tex, w_tex)) # Ensure float division
        new_w_tex = int(w_tex * scale)
        new_h_tex = int(h_tex * scale)
        texture_to_save = cv2.resize(texture_img_orig, (new_w_tex, new_h_tex), interpolation=cv2.INTER_AREA)
    
    # Save texture image that will be referenced by the MTL file.
    # It should be saved in the same directory as the OBJ/MTL or a relative path used.
    # Your output structure: results/3d_model/textured_model/
    # So, `output_dir` here is `results/3d_model/textured_model/`.
    # The texture file itself.
    texture_filename_for_mtl = "model_texture.png" # Consistent name for the texture file
    actual_texture_save_path = os.path.join(output_dir, texture_filename_for_mtl)
    
    # Ensure the directory for the texture (and OBJ/MTL) exists
    create_directory(os.path.dirname(actual_texture_save_path)) # Creates `output_dir` if not existing

    cv2.imwrite(actual_texture_save_path, texture_to_save) # OpenCV expects BGR
    
    # export mesh using our custom OBJ exporter
    # `texture_path` argument to `export_textured_obj` should be the relative path used in MTL.
    obj_path, mtl_path = export_textured_obj(
        vertices=mesh.vertices,
        faces=mesh.faces,
        uvs=uvs,
        texture_path=texture_filename_for_mtl, # Relative filename for MTL
        output_dir=output_dir, # Directory where OBJ, MTL, and texture are saved
        prefix="textured_model" # Prefix for OBJ and MTL files
    )
    
    # also export the same mesh without texture for comparison
    # This should go into `results/3d_model/visual_hull.obj` or similar
    # The `output_dir` for this call is `results/3d_model/textured_model/`
    # To save `visual_hull.obj` in `results/3d_model/`, need parent dir.
    # However, `create_visual_hull` already saves `visual_hull.obj` in its `output_dir`.
    # If `mesh` object was modified (e.g., by plane removal), it might be useful to save it.
    # For now, assume `visual_hull.obj` is already saved correctly by visual_hull module.
    # If you need to save the (potentially plane-removed) mesh again before texturing:
    # untextured_mesh_save_dir = os.path.dirname(output_dir) # Goes up one level, e.g. to results/3d_model/
    # if os.path.basename(output_dir) == "textured_model": # Check if it's the specific subdir
    #     mesh.export(os.path.join(untextured_mesh_save_dir, 'visual_hull_processed.obj'))

    print(f"Textured model OBJ saved to {obj_path}")
    print(f"Material file MTL saved to {mtl_path}")
    print(f"Texture image saved to {actual_texture_save_path}")
    return obj_path


def apply_cylindrical_texture_mapping(mesh, panorama_path, output_dir, texture_v_scale=0.9, texture_v_offset=0.05):
    """
    apply cylindrical texture mapping to a 3D mesh using a panorama image.
    
    Args:
        mesh: Trimesh object
        panorama_path: Path to panorama image (e.g. results/panorama.png)
        output_dir: Directory to save the textured mesh (e.g. results/3d_model/textured_model/)
        texture_v_scale: Scale factor for vertical texture mapping [0-1] (default: 0.9)
        texture_v_offset: Offset for vertical texture mapping [0-1] (default: 0.05)
        
    Returns:
        Path to saved textured mesh OBJ file, or None on failure
    """
    # generate UV coordinates (using cylindrical mapping for now)
    uvs = generate_cylindrical_uvs(mesh, texture_v_scale, texture_v_offset)
    if uvs.size == 0: # Check if UV generation failed
        print("UV generation failed in apply_cylindrical_texture_mapping.")
        return None
    
    # mesh <- apply texture
    textured_mesh_path = apply_texture_to_mesh(mesh, uvs, panorama_path, output_dir)
    
    return textured_mesh_path