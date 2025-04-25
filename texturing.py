import os
import numpy as np
import cv2
import open3d as o3d
from pathlib import Path

def apply_textures(model_path, segmented_images, output_dir):
    """
    Apply textures from the segmented images onto the 3D model
    
    Args:
        model_path: Path to the 3D model file
        segmented_images: List of paths to segmented object images
        output_dir: Directory to save the textured model output
        
    Returns:
        Path to the textured model file
    """
    print(f"Loading 3D model from {model_path}")
    
    # Load the 3D model
    mesh = o3d.io.read_triangle_mesh(model_path)
    
    # Make sure mesh has vertex normals
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    # Project textures from segmented images onto the mesh
    textured_mesh = project_textures(mesh, segmented_images)
    
    # Save the textured model
    textured_model_path = os.path.join(output_dir, "textured_model.obj")
    o3d.io.write_triangle_mesh(textured_model_path, textured_mesh)
    
    # Create a material file for the textures
    create_mtl_file(output_dir, segmented_images)
    
    print(f"Textured 3D model saved to {textured_model_path}")
    return textured_model_path

def project_textures(mesh, image_paths):
    """Project textures from the images onto the mesh"""
    print("Projecting textures onto the mesh...")
    
    # This is a simplified texturing approach
    # In reality, you'd use a more complex UV mapping technique
    
    # Create texture map
    texture_image = create_texture_atlas(mesh, image_paths)
    
    # Save texture image
    texture_path = os.path.join(os.path.dirname(image_paths[0]), "..", "textured", "texture_atlas.jpg")
    cv2.imwrite(texture_path, texture_image)
    
    # Apply texture coordinates to the mesh
    mesh.triangle_uvs = o3d.utility.Vector2dVector(generate_simple_uvs(mesh))
    
    # Set the texture
    mesh.texture = o3d.geometry.Image(texture_image)
    
    return mesh

def create_texture_atlas(mesh, image_paths):
    """Create a texture atlas from the segmented images"""
    # Load all segmented images
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    
    if not images:
        # If no valid images, return a blank texture
        return np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Determine atlas size based on number and size of images
    max_height = max([img.shape[0] for img in images])
    max_width = max([img.shape[1] for img in images])
    
    cols = min(4, len(images))  # Max 4 images per row
    rows = (len(images) + cols - 1) // cols
    
    atlas_width = cols * max_width
    atlas_height = rows * max_height
    
    # Create the atlas
    atlas = np.zeros((atlas_height, atlas_width, 3), dtype=np.uint8)
    
    # Place images in the atlas
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        
        y_start = row * max_height
        x_start = col * max_width
        
        h, w = img.shape[:2]
        atlas[y_start:y_start+h, x_start:x_start+w] = img
    
    return atlas

def generate_simple_uvs(mesh):
    """Generate simple UV coordinates for the mesh"""
    # This is a very simple UV generation, not suitable for production
    # In reality, you'd use proper UV unwrapping techniques
    
    vertices = np.asarray(mesh.vertices)
    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)
    bounds = max_bound - min_bound
    
    # Normalize X and Y to [0,1] for simple UV mapping
    uv_coords = []
    for triangle in np.asarray(mesh.triangles):
        for idx in triangle:
            vertex = vertices[idx]
            u = (vertex[0] - min_bound[0]) / bounds[0]
            v = (vertex[1] - min_bound[1]) / bounds[1]
            uv_coords.append([u, v])
    
    return np.array(uv_coords)

def create_mtl_file(output_dir, image_paths):
    """Create a material file for the textured model"""
    mtl_path = os.path.join(output_dir, "textured_model.mtl")
    texture_file = "texture_atlas.jpg"
    
    with open(mtl_path, 'w') as f:
        f.write("newmtl material0\n")
        f.write("Ka 1.000000 1.000000 1.000000\n")
        f.write("Kd 1.000000 1.000000 1.000000\n")
        f.write("Ks 0.000000 0.000000 0.000000\n")
        f.write("Tr 0.000000\n")
        f.write("illum 1\n")
        f.write("Ns 0.000000\n")
        f.write(f"map_Kd {texture_file}\n")
    
    print(f"Material file created at {mtl_path}")
