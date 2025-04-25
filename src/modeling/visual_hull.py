"""
Visual Hull Reconstruction module - Creates 3D model from silhouettes.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import trimesh
from skimage import measure

def extract_silhouettes(segmented_images):
    """
    Extract binary silhouettes from segmented images.
    
    Args:
        segmented_images: List of segmented images
        
    Returns:
        List of binary silhouette masks
    """
    silhouettes = []
    for img in segmented_images:
        # Convert to grayscale if it's a color image
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        
        # Create binary mask (white foreground, black background)
        _, silhouette = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        silhouettes.append(silhouette)
    
    return silhouettes

def create_volume_from_silhouettes(silhouettes, angles, volume_size=(100, 100, 100)):
    """
    Create a 3D volume using visual hull technique from silhouettes.
    
    Args:
        silhouettes: List of binary silhouette masks
        angles: List of angles (in degrees) for each silhouette
        volume_size: Size of the 3D volume (x, y, z)
        
    Returns:
        3D binary volume
    """
    # we're gonna carve out the background from the solid(filled) volume
    volume = np.ones(volume_size, dtype=np.uint8)
    
    # center of the volume
    cx, cy, cz = volume_size[0] // 2, volume_size[1] // 2, volume_size[2] // 2
    
    # radius of the volume (assume it fits within the volume)
    radius = min(cx, cy, cz) - 5
    
    # for each view angle, we will carve out the background from the volume
    print("Creating visual hull...")
    for i, silhouette in enumerate(tqdm(silhouettes)):
        # resize silhouette to match "volume height"
        h, w = silhouette.shape
        aspect_ratio = w / h
        new_h = volume_size[1]
        new_w = int(new_h * aspect_ratio)
        silhouette_resized = cv2.resize(silhouette, (new_w, new_h))
        
        # degree -> rad
        angle_rad = np.radians(angles[i])
        
        # for each z-slice of the volume from current angle
        # we will carve out the background from the volume
        for z in range(volume_size[2]):
            # for each pixel
            for x in range(volume_size[0]):
                for y in range(volume_size[1]):
                    # point relative to center
                    rx, ry, rz = x - cx, y - cy, z - cz
                    
                    # transform point based on the silhouette's viewing angle
                    tx = rx * np.cos(angle_rad) - rz * np.sin(angle_rad)
                    tz = rx * np.sin(angle_rad) + rz * np.cos(angle_rad)
                    ty = ry
                    
                    # map to silhouette coordinates
                    sx = int((tx / radius * 0.5 + 0.5) * new_w)
                    sy = int((ty / radius * 0.5 + 0.5) * new_h)
                    
                    # check if the point is outside the silhouette bounds
                    if sx < 0 or sx >= new_w or sy < 0 or sy >= new_h:
                        continue
                    
                    # if the point is in the background of the silhouette, carve it out of the volume
                    if silhouette_resized[sy, sx] == 0:
                        volume[x, y, z] = 0
    
    return volume

def create_mesh_from_volume(volume, level=0.5, step_size=1, allow_degenerate=False, use_poisson=True):
    """
    Create a 3D mesh from a binary volume using Marching Cubes algorithm
    with optional Poisson reconstruction for higher quality.
    
    Args:
        volume: 3D binary volume
        level: Iso-surface level (0.5 for binary volumes)
        step_size: Step size for marching cubes (1 = full resolution)
        allow_degenerate: Allow degenerate faces
        use_poisson: Use Poisson reconstruction for higher quality mesh
        
    Returns:
        Trimesh object
    """
    if use_poisson:
        try:
            # Try to use our advanced Poisson reconstruction pipeline
            from src.modeling.mesh_processing import volume_to_mesh_poisson
            return volume_to_mesh_poisson(volume, level)
        except ImportError:
            print("Warning: Mesh processing module not available. Using standard marching cubes.")
            # Fall back to standard method
            pass
    
    # Use marching cubes with parameters if Poisson is not available or not requested
    verts, faces, normals, _ = measure.marching_cubes(
        volume, 
        level=level,
        step_size=step_size,
        allow_degenerate=allow_degenerate
    )
    
    # Create a Trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # Clean up the mesh
    mesh = mesh.process(validate=True)
    
    return mesh

def generate_angles(num_images):
    """
    note that this function assumes images are taken at equal intervals around the object.
    just a pseudo solution to generate angles for now...
        e.g. 8 images -> [0, 45, 90, 135, 180, 225, 270, 315 degrees]

    Args:
        num_images: Number of images
        
    Returns:
        List of angles in degrees
    """
    return [i * (360.0 / num_images) for i in range(num_images)]

# wrapper function to create visual hull mesh
def create_visual_hull(segmented_images, output_dir, volume_size=(200, 200, 200),
                      post_process=True, smooth_iterations=None, fill_holes_size=100, 
                      subdivide=False, use_poisson=True):
    """
    Create a visual hull mesh from segmented images.
    
    Args:
        segmented_images: List of segmented images
        output_dir: Directory to save the mesh
        volume_size: Size of the 3D volume
        post_process: Whether to apply mesh improvement
        smooth_iterations: Not used anymore, kept for API compatibility
        fill_holes_size: Maximum size of holes to fill
        subdivide: Whether to subdivide the mesh for higher resolution
        use_poisson: Whether to use Poisson reconstruction for high quality mesh
        
    Returns:
        Path to saved mesh file and the mesh object
    """
    # extract silhouettes from segmented images
    silhouettes = extract_silhouettes(segmented_images)
    
    # generate angles (assuming images are taken at equal intervals around the object)
    angles = generate_angles(len(silhouettes))
    
    # create "volume" from silhouettes
    volume = create_volume_from_silhouettes(silhouettes, angles, volume_size)
    
    # create "mesh (renderable triangles)" from "volume"
    mesh = create_mesh_from_volume(volume, use_poisson=use_poisson)
    
    # Apply post-processing if requested
    if post_process:
        try:
            from src.modeling.mesh_processing import improve_mesh
            print("Applying mesh post-processing...")
            mesh = improve_mesh(
                mesh,
                fill_holes_size=fill_holes_size,
                subdivide=subdivide
            )
        except ImportError:
            print("Warning: Mesh processing module not available. Using raw mesh.")
    
    # save mesh
    mesh_path = os.path.join(output_dir, 'visual_hull.obj')
    mesh.export(mesh_path)
    
    print(f"Visual hull mesh saved to {mesh_path}")
    return mesh_path, mesh