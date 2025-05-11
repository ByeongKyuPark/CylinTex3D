"""
Visual Hull Reconstruction module - Creates 3D model from silhouettes.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import trimesh
from skimage import measure
from numba import njit

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

# OPTIMIZATION: Numba JIT compiled core function for volume carving
@njit(cache=True) # Enable caching of compiled function
def _create_volume_from_silhouettes_numba_core(
    volume_data, # Pass pre-allocated volume
    silhouettes_list_numba, # List of NumPy arrays (resized silhouettes)
    angles_rad_list_numba,  # NumPy array of angles in radians
    volume_size_tuple,      # Tuple (vx, vy, vz)
    sil_dims_list_numba,    # List of tuples (h_sil, w_sil) for each silhouette
    volume_center_tuple,    # Tuple (cx, cy, cz)
    projection_radius_numba # Float, world radius that maps to silhouette extent
    ):
    """ Numba core for visual hull carving. Modifies volume_data in place. """
    cx_n, cy_n, cz_n = volume_center_tuple
    num_silhouettes = len(silhouettes_list_numba)

    for i in range(num_silhouettes):
        silhouette_i_numba = silhouettes_list_numba[i] # Current silhouette (binary: 0 or 255)
        angle_rad_i_numba = angles_rad_list_numba[i]
        sil_h_i_numba, sil_w_i_numba = sil_dims_list_numba[i]

        cos_a_n = np.cos(angle_rad_i_numba)
        sin_a_n = np.sin(angle_rad_i_numba)

        # Iterate over all voxels in the volume
        for x_vox_n in range(volume_size_tuple[0]):
            for y_vox_n in range(volume_size_tuple[1]):
                for z_vox_n in range(volume_size_tuple[2]):
                    if volume_data[x_vox_n, y_vox_n, z_vox_n] == 0: # Already carved out
                        continue

                    # Voxel coordinates relative to volume center
                    rx_n = x_vox_n - cx_n
                    ry_n = y_vox_n - cy_n
                    rz_n = z_vox_n - cz_n
                    
                    # Transform voxel to this silhouette's camera view (rotation around Y-axis of world)
                    tx_cam_n = rx_n * cos_a_n + rz_n * sin_a_n # X in camera's image plane if Y is up
                    ty_cam_n = ry_n                            # Y in camera's image plane (world Y is camera Y)
                    
                    # Orthographic projection onto the silhouette image plane
                    # Map world coords [-proj_radius, +proj_radius] to silhouette pixel coords [0, W-1] or [0, H-1]
                    if projection_radius_numba < 1e-6 : projection_radius_numba = 1.0 # Avoid div by zero

                    # Map tx_cam_n to sx_img_n in [0, sil_w_i_numba-1]
                    # Map ty_cam_n to sy_img_n in [0, sil_h_i_numba-1]
                    # (value / world_extent_half + 1) / 2 * image_dimension
                    # world_extent_half = projection_radius_numba
                    sx_img_n = (tx_cam_n / projection_radius_numba * 0.5 + 0.5) * sil_w_i_numba
                    sy_img_n = (ty_cam_n / projection_radius_numba * 0.5 + 0.5) * sil_h_i_numba
                    
                    # Check if projected point is within this silhouette's image bounds
                    if 0 <= sx_img_n < sil_w_i_numba and \
                       0 <= sy_img_n < sil_h_i_numba:
                        
                        # Get pixel value from silhouette (nearest neighbor by int casting)
                        # Silhouette is 0 (background) or 255 (foreground)
                        sil_pixel_val = silhouette_i_numba[int(sy_img_n), int(sx_img_n)]
                        if sil_pixel_val == 0: # If voxel projects to background in this silhouette
                            volume_data[x_vox_n, y_vox_n, z_vox_n] = 0 # Carve it out
                    else: # Voxel projects outside the bounds of this silhouette image
                        volume_data[x_vox_n, y_vox_n, z_vox_n] = 0 # Carve it out
    # No explicit return needed as volume_data is modified in-place

def create_volume_from_silhouettes(silhouettes, angles, volume_size=(100, 100, 100)):
    """
    Create a 3D volume using visual hull technique from silhouettes.
    MODIFIED to use Numba JIT compiled core.
    
    Args:
        silhouettes: List of binary silhouette masks (0 for bg, 255 for fg)
        angles: List of angles (in degrees) for each silhouette
        volume_size: Size of the 3D volume (vx, vy, vz)
        
    Returns:
        3D binary volume (0 for carved, 1 for solid)
    """
    # Original Python loop implementation:
    # # we're gonna carve out the background from the solid(filled) volume
    # volume = np.ones(volume_size, dtype=np.uint8)
    
    # # center of the volume
    # cx, cy, cz = volume_size[0] // 2, volume_size[1] // 2, volume_size[2] // 2
    
    # # radius of the volume (assume it fits within the volume)
    # # This 'radius' is a scaling factor: world distance that maps to half the silhouette image width/height
    # radius = min(cx, cy, cz) - 5 
    # if radius <= 0: radius = 1.0 # Ensure positive radius

    # # for each view angle, we will carve out the background from the volume
    # print("Creating visual hull (Python loop)...") # Original print
    # for i, silhouette_orig in enumerate(tqdm(silhouettes)): # Renamed to avoid conflict
    #     # resize silhouette to match "volume height" (Y-dimension of volume)
    #     h_orig_sil, w_orig_sil = silhouette_orig.shape
    #     aspect_ratio_sil = w_orig_sil / h_orig_sil if h_orig_sil > 0 else 1.0
        
    #     new_h_sil_resized = volume_size[1] # Sil height matches volume Y-dim
    #     new_w_sil_resized = int(new_h_sil_resized * aspect_ratio_sil)
    #     if new_w_sil_resized <= 0: new_w_sil_resized = 1

    #     # Using INTER_NEAREST for binary silhouettes is usually appropriate
    #     silhouette_resized_for_loop = cv2.resize(silhouette_orig, (new_w_sil_resized, new_h_sil_resized), interpolation=cv2.INTER_NEAREST)
        
    #     angle_rad_loop = np.radians(angles[i])
    #     cos_a_loop = np.cos(angle_rad_loop)
    #     sin_a_loop = np.sin(angle_rad_loop)
        
    #     for z_vox_loop in range(volume_size[2]):
    #         for x_vox_loop in range(volume_size[0]):
    #             for y_vox_loop in range(volume_size[1]):
    #                 if volume[x_vox_loop, y_vox_loop, z_vox_loop] == 0:
    #                     continue

    #                 rx_loop, ry_loop, rz_loop = x_vox_loop - cx, y_vox_loop - cy, z_vox_loop - cz
                    
    #                 tx_loop = rx_loop * cos_a_loop - rz_loop * sin_a_loop # X in camera plane
    #                 # tz_loop = rx_loop * sin_a_loop + rz_loop * cos_a_loop # Depth (not used)
    #                 ty_loop = ry_loop # Y in camera plane (world Y up)
                    
    #                 # map to silhouette coordinates ([-radius, +radius] world -> [0, new_w/h] image)
    #                 sx_loop = int((tx_loop / radius * 0.5 + 0.5) * new_w_sil_resized)
    #                 sy_loop = int((ty_loop / radius * 0.5 + 0.5) * new_h_sil_resized)
                    
    #                 if not (0 <= sx_loop < new_w_sil_resized and 0 <= sy_loop < new_h_sil_resized):
    #                     volume[x_vox_loop, y_vox_loop, z_vox_loop] = 0 # Projects outside
    #                     continue
                    
    #                 if silhouette_resized_for_loop[sy_loop, sx_loop] == 0: # If background
    #                     volume[x_vox_loop, y_vox_loop, z_vox_loop] = 0
    # return volume # End of original Python loop implementation

    # OPTIMIZED version using Numba:
    print("Creating visual hull (Numba optimized)...")
    volume_for_numba = np.ones(volume_size, dtype=np.uint8) # Numba modifies this in-place
    
    volume_center_for_numba = (volume_size[0] // 2, volume_size[1] // 2, volume_size[2] // 2)
    
    # Projection radius: This is critical. It's the effective "world space" radius that corresponds
    # to the extent of the resized silhouette images.
    # If silhouettes are resized such that their height matches `volume_size[1]`, then a reasonable
    # `projection_radius` related to `volume_size[1]/2` (for the Y dimension) might be used.
    # The original code's `radius = min(cx, cy, cz) - 5` suggests a radius in voxel units.
    # Let's use a similar heuristic for `projection_radius_numba`.
    # It should be a float.
    projection_radius_val_numba = float(min(volume_center_for_numba) * 0.9) # e.g. 90% of min half-dimension
    if projection_radius_val_numba < 1.0: projection_radius_val_numba = 1.0

    # Prepare data for Numba (lists of NumPy arrays, simple types)
    silhouettes_list_numba_input = []
    sil_dims_list_numba_input = []

    for sil_orig_numba in silhouettes:
        h_s, w_s = sil_orig_numba.shape
        aspect_s = w_s / h_s if h_s > 0 else 1.0
        
        # Silhouettes are resized so their height generally matches volume_size[1] (Y-dimension of volume)
        # This implies object's vertical extent in world maps to this resized height.
        target_h_sil_numba = volume_size[1] 
        target_w_sil_numba = int(target_h_sil_numba * aspect_s)
        if target_w_sil_numba <= 0: target_w_sil_numba = 1

        # INTER_NEAREST is fine for binary images passed to Numba core if it does int-casting for indexing
        sil_resized_numba = cv2.resize(sil_orig_numba, (target_w_sil_numba, target_h_sil_numba), interpolation=cv2.INTER_NEAREST)
        silhouettes_list_numba_input.append(sil_resized_numba) # Already uint8 (0 or 255)
        sil_dims_list_numba_input.append((target_h_sil_numba, target_w_sil_numba))

    angles_rad_list_np_numba = np.array([np.radians(ang) for ang in angles], dtype=np.float64)

    # Call the Numba JIT compiled function. First call involves compilation time.
    _create_volume_from_silhouettes_numba_core(
        volume_for_numba, # Modified in-place
        silhouettes_list_numba_input,
        angles_rad_list_np_numba,
        volume_size, # Tuple
        sil_dims_list_numba_input,
        volume_center_for_numba, # Tuple
        projection_radius_val_numba # Float
    )
    return volume_for_numba


def create_mesh_from_volume(volume, level=0.5, step_size=1, allow_degenerate=False):
    """
    Create a 3D mesh from a binary volume using Marching Cubes algorithm.
    
    Args:
        volume: 3D binary volume
        level: Iso-surface level (0.5 for binary volumes)
        step_size: Step size for marching cubes (1 = full resolution)
        allow_degenerate: Allow degenerate faces
        
    Returns:
        Trimesh object
    """
    # OPTIMIZATION: Check if volume is empty before calling marching_cubes
    if not np.any(volume): # If volume is all zeros
        print("Warning: Input volume for marching cubes is empty. Returning an empty mesh.")
        return trimesh.Trimesh() # Return an empty Trimesh object

    # Use marching cubes with parameters
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            volume, 
            level=level, # For binary volume (0,1), level=0.5 is typical
            step_size=step_size,
            allow_degenerate=allow_degenerate
        )
    except Exception as e_mc: # Catch errors from marching_cubes (e.g. if volume is flat)
        print(f"Error during skimage.measure.marching_cubes: {e_mc}. Returning an empty mesh.")
        return trimesh.Trimesh()
    
    # OPTIMIZATION: Check if marching_cubes produced any geometry
    if verts.shape[0] == 0 or faces.shape[0] == 0:
        print("Warning: Marching cubes produced no vertices or faces. Returning an empty mesh.")
        return trimesh.Trimesh()

    # Create a Trimesh object
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=False) # process=False initially
    
    # Clean up the mesh
    try:
        mesh.process(validate=True) # This does various cleaning steps
    except Exception as e_process:
        print(f"Warning: trimesh.process() failed: {e_process}. Using mesh as is from marching_cubes.")
        # If process fails, use the mesh directly from marching_cubes output
        # (It might already be okay, or this Trimesh constructor call might be redundant)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)


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
    if num_images <= 0: return [] # Handle case of zero or negative images
    return [i * (360.0 / num_images) for i in range(num_images)]

# wrapper function to create visual hull mesh
def create_visual_hull(segmented_images, output_dir, volume_size=(200, 200, 200)):
    """
    Create a visual hull mesh from segmented images.
    
    Args:
        segmented_images: List of segmented images
        output_dir: Directory to save the mesh (e.g., "results/3d_model/")
        volume_size: Size of the 3D volume
        
    Returns:
        Tuple of (path to saved mesh file, Trimesh object) or (None, None) on failure
    """
    if not segmented_images: # Handle empty input
        print("No segmented images provided to create_visual_hull.")
        return None, None

    # extract silhouettes from segmented images
    silhouettes = extract_silhouettes(segmented_images)
    if not silhouettes: # Handle if no silhouettes extracted
        print("No silhouettes were extracted from segmented images.")
        return None, None
    
    # generate angles (assuming images are taken at equal intervals around the object)
    angles = generate_angles(len(silhouettes))
    
    # create "volume" from silhouettes (this now calls the Numba-optimized version)
    volume = create_volume_from_silhouettes(silhouettes, angles, volume_size)
    
    if volume is None or not np.any(volume): # Check if volume creation failed or is empty
        print("Visual hull volume creation failed or resulted in an empty volume.")
        return None, None

    # create "mesh (renderable triangles)" from "volume"
    mesh = create_mesh_from_volume(volume)
    
    if mesh is None or len(mesh.vertices) == 0: # Check if mesh creation failed
        print("Mesh creation from volume failed or resulted in an empty mesh.")
        return None, None
    
    # save mesh
    # Output path is `results/3d_model/visual_hull.obj`
    # `output_dir` passed here should be `results/3d_model/`
    mesh_path = os.path.join(output_dir, 'visual_hull.obj')
    try:
        create_directory(os.path.dirname(mesh_path)) # Ensure directory exists
        mesh.export(mesh_path)
        print(f"Visual hull mesh saved to {mesh_path}")
    except Exception as e_export:
        print(f"Error exporting visual hull mesh to {mesh_path}: {e_export}")
        # Return the mesh object even if export fails, path will be None
        return None, mesh 
        
    return mesh_path, mesh