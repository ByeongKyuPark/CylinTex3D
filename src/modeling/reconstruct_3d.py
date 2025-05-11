"""
3D Reconstruction module - Creates textured 3D model from panorama.
"""

import os
from src.utils.image_utils import load_images_from_folder, create_directory
from src.modeling.visual_hull import create_visual_hull
from src.modeling.plane_remover import remove_planar_faces
from src.texturing.texture_mapping import apply_cylindrical_texture_mapping
        
import trimesh

def reconstruct_3d_model(segmented_dir, panorama_path, output_dir, volume_size=(200, 200, 200),
                     texture_v_scale=0.9, texture_v_offset=0.05, skip_hull=False, 
                     model_path=None, remove_planes=False, plane_threshold=0.8):
    """
    Reconstruct a 'textured' 3D 'model' from segmented images and panorama.
    
    Args:
        segmented_dir: Directory containing segmented images
        panorama_path: Path to panorama image
        output_dir: Directory to save results
        volume_size: Size of the 3D volume for visual hull reconstruction
        texture_v_scale: Scale factor for vertical texture mapping [0-1]
        texture_v_offset: Offset for vertical texture mapping [0-1]
        skip_hull: Skip visual hull creation and use existing model
        model_path: Path to existing model if skip_hull is True
        remove_planes: Whether to remove planar faces
        plane_threshold: Threshold for plane detection
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure the base model output directory exists (e.g., "results/3d_model/")
        create_directory(output_dir) 
        
        current_mesh = None # To hold the mesh object

        if skip_hull and model_path is not None and os.path.exists(model_path):
            print(f"Skipping visual hull creation, loading existing model from: {model_path}")
            current_mesh = trimesh.load_mesh(model_path)
            if current_mesh is None or len(current_mesh.vertices) == 0:
                print(f"Error: Failed to load existing model from {model_path} or model is empty.")
                return False
        else:
            print(f"Loading segmented images from {segmented_dir} for visual hull...")
            images, _ = load_images_from_folder(segmented_dir) # filenames not needed here
            
            if not images:
                print(f"No segmented images found in {segmented_dir} for visual hull.")
                return False
            
            print(f"Found {len(images)} segmented images for visual hull.")
            
            # create_visual_hull saves `visual_hull.obj` into its `output_dir` argument.
            # Here, `output_dir` for create_visual_hull is `results/3d_model/`.
            hull_obj_path, hull_mesh_object = create_visual_hull(images, output_dir, volume_size)
            
            if hull_mesh_object is None:
                print("Visual hull creation failed.")
                return False
            current_mesh = hull_mesh_object
            print(f"Visual hull created. Raw mesh at: {hull_obj_path if hull_obj_path else 'Path not returned'}")

        # Optional: Remove planar faces from the current_mesh
        if remove_planes:
            print(f"Removing planar faces with threshold {plane_threshold}...")
            # Define path for the cleaned (plane-removed) hull, within `output_dir`
            cleaned_hull_path = os.path.join(output_dir, 'cleaned_visual_hull.obj')
            mesh_after_plane_removal = remove_planar_faces(current_mesh, plane_threshold, cleaned_hull_path)
            
            if mesh_after_plane_removal is None or len(mesh_after_plane_removal.vertices) == 0 :
                print("Warning: Plane removal resulted in an empty mesh. Using mesh before plane removal.")
            else:
                current_mesh = mesh_after_plane_removal
                print(f"Planar faces removed. Cleaned mesh saved to {cleaned_hull_path}")
        
        # Texturing
        # The textured model (OBJ, MTL, texture image) goes into `output_dir/textured_model/`
        textured_model_output_subdir = os.path.join(output_dir, "textured_model")
        create_directory(textured_model_output_subdir) # Ensure this subdir exists

        print(f"\nApplying texture to mesh. Output will be in: {textured_model_output_subdir}")
        final_textured_obj_path = apply_cylindrical_texture_mapping(
            current_mesh,
            panorama_path,
            textured_model_output_subdir, # Pass the specific subdirectory for textured outputs
            texture_v_scale,
            texture_v_offset
        )
        
        if final_textured_obj_path is None:
            print("Failed to apply texture to the mesh.")
            return False
            
        print(f"\nTextured 3D model reconstruction successful!")
        print(f"Final textured model OBJ: {final_textured_obj_path}")
        
        # Optional: Create a sample screenshot (this part is external to the core logic)
        # sample_jpg_path = os.path.join(output_dir, "sample.jpg")
        # print(f"Note: Screenshot generation (sample.jpg) not implemented here. Path would be: {sample_jpg_path}")

        return True
    
    except Exception as e:
        print(f"Error during 3D reconstruction and texturing: {e}")
        import traceback
        traceback.print_exc()
        return False