# src/modeling/reconstruct_3d.py
"""
3D Reconstruction module - Creates textured 3D model from panorama.
"""

import os
from src.utils.image_utils import load_images_from_folder, create_directory
from src.modeling.visual_hull import create_visual_hull
from src.modeling.plane_remover import remove_planar_faces
from src.texturing.texture_mapping import generate_cylindrical_uvs, apply_texture_to_mesh
        
import trimesh

def reconstruct_3d_model(segmented_dir, panorama_path, output_dir, volume_size=(200, 200, 200),
                     texture_v_scale=0.9, texture_v_offset=0.05, skip_hull=False, 
                     model_path=None, remove_planes=False, plane_threshold=0.8,
                     post_process=True, smooth_iterations=None, fill_holes_size=100,
                     subdivide=False, use_poisson=True):
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
        post_process: Whether to apply mesh improvement
        smooth_iterations: Not used anymore, kept for API compatibility
        fill_holes_size: Maximum size of holes to fill
        subdivide: Whether to subdivide the mesh for higher resolution
        use_poisson: Whether to use Poisson reconstruction for high quality mesh
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # create directory first
        model_dir = os.path.join(output_dir, '3d_model')
        create_directory(model_dir)
        
        # fetch if exist (user's call)
        if skip_hull and model_path is not None:
            print(f"Skipping visual hull creation, using existing model: {model_path}")
            mesh = trimesh.load(model_path)
            
            # Apply post-processing to the loaded mesh if requested
            if post_process:
                try:
                    from src.modeling.mesh_processing import improve_mesh
                    print("Applying mesh post-processing to existing model...")
                    mesh = improve_mesh(
                        mesh,
                        fill_holes_size=fill_holes_size,
                        subdivide=subdivide
                    )
                except ImportError:
                    print("Warning: Mesh processing module not available. Using raw mesh.")
        else:
            print(f"Loading segmented images from {segmented_dir}")
            # starting from segmented images
            images, filenames = load_images_from_folder(segmented_dir)
            
            if not images:
                print("No images found!")
                return False
            
            print(f"Found {len(images)} segmented images")
            
            # create visual hull mesh from segmented images
            _, mesh = create_visual_hull(
                images, 
                model_dir, 
                volume_size,
                post_process=post_process,
                fill_holes_size=fill_holes_size,
                subdivide=subdivide,
                use_poisson=use_poisson
            )
        
        # remove planes if requested
        if remove_planes:
            print(f"Removing planar faces with threshold {plane_threshold}...")
            mesh = remove_planar_faces(mesh, plane_threshold)
            
            # Save the cleaned mesh
            cleaned_path = os.path.join(model_dir, 'cleaned_hull.obj')
            mesh.export(cleaned_path)
            print(f"Cleaned mesh saved to {cleaned_path}")
        
        # generate UV coordinates
        uvs = generate_cylindrical_uvs(mesh, texture_v_scale, texture_v_offset)
        
        # apply texture
        textured_path = apply_texture_to_mesh(mesh, uvs, panorama_path, model_dir)
        
        print(f"\nTextured model created successfully!")
        print(f"Model: {textured_path}")
        
        return True
    
    except Exception as e:
        print(f"Error in 3D reconstruction: {e}")
        import traceback
        traceback.print_exc()
        return False