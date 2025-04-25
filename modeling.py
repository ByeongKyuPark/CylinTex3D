import os
import numpy as np
import cv2
from pathlib import Path
import open3d as o3d

def create_3d_model(segmented_images, output_dir):
    """
    Generate a 3D model from segmented object images with plane removal
    
    Args:
        segmented_images: List of paths to segmented object images
        output_dir: Directory to save the 3D model output
        
    Returns:
        Path to the generated 3D model file
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a point cloud from the segmented images
    point_cloud = generate_point_cloud(segmented_images)
    
    # Remove planar surfaces
    filtered_cloud = remove_planes(point_cloud)
    
    # Create mesh from point cloud
    mesh = create_mesh(filtered_cloud)
    
    # Save the model
    model_path = os.path.join(output_dir, "model.obj")
    o3d.io.write_triangle_mesh(model_path, mesh)
    
    print(f"3D model created successfully and saved to {model_path}")
    return model_path

def generate_point_cloud(image_paths):
    """Generate a point cloud from the segmented images"""
    # This is a simplified version - in reality, you'd need SfM or MVS
    # using libraries like COLMAP, MVSNet, or similar
    
    print("Generating point cloud from images...")
    
    # In a real implementation, you'd run a structure from motion pipeline here
    # For demonstration, we'll create a simple point cloud
    point_cloud = o3d.geometry.PointCloud()
    
    # Collect color information from images
    all_points = []
    all_colors = []
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # Create some simple 3D points (in reality, these would come from SfM)
        for y in range(0, h, 10):  # Sample every 10 pixels to keep it manageable
            for x in range(0, w, 10):
                # Only include non-black pixels (assuming black is background in segmented images)
                color = img_rgb[y, x]
                if not np.all(color == 0):
                    # Create a point with some depth variation
                    # This is just for demonstration - real points would come from SfM
                    z = (x + y) % 100 / 100.0  # Simple depth variation
                    all_points.append([x/100.0, y/100.0, z])
                    all_colors.append(color / 255.0)
    
    if len(all_points) > 0:
        point_cloud.points = o3d.utility.Vector3dVector(np.array(all_points))
        point_cloud.colors = o3d.utility.Vector3dVector(np.array(all_colors))
    
    return point_cloud

def remove_planes(point_cloud):
    """Remove planar surfaces from the point cloud"""
    print("Removing planar surfaces...")
    
    # Make a copy of the input point cloud
    filtered_cloud = o3d.geometry.PointCloud()
    filtered_cloud.points = o3d.utility.Vector3dVector(np.asarray(point_cloud.points))
    filtered_cloud.colors = o3d.utility.Vector3dVector(np.asarray(point_cloud.colors))
    
    # Downsample the point cloud for faster processing
    downsampled = filtered_cloud.voxel_down_sample(voxel_size=0.05)
    
    # Calculate normals
    downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # RANSAC plane segmentation
    remaining_cloud = downsampled
    planes_removed = 0
    
    for _ in range(3):  # Try to remove up to 3 major planes
        if len(np.asarray(remaining_cloud.points)) < 100:
            break  # Too few points left
            
        # Segment a plane
        plane_model, inliers = remaining_cloud.segment_plane(
            distance_threshold=0.02,
            ransac_n=3,
            num_iterations=1000
        )
        
        if len(inliers) < 20:
            break  # If the plane is too small, stop
        
        # Remove the plane points
        outlier_cloud = remaining_cloud.select_by_index(inliers, invert=True)
        remaining_cloud = outlier_cloud
        planes_removed += 1
    
    print(f"Removed {planes_removed} planar surfaces")
    
    return remaining_cloud

def create_mesh(point_cloud):
    """Create a mesh from the point cloud"""
    print("Creating mesh from point cloud...")
    
    # Make sure we have enough points
    if len(np.asarray(point_cloud.points)) < 10:
        print("Not enough points to create a mesh")
        return o3d.geometry.TriangleMesh()
    
    # Calculate normals if they don't exist
    if not point_cloud.has_normals():
        point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        point_cloud.orient_normals_consistent_tangent_plane(100)
    
    # Use Poisson surface reconstruction
    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        point_cloud, depth=8, width=0, scale=1.1, linear_fit=False
    )
    
    # Clean up the mesh
    mesh = mesh.filter_smooth_simple(number_of_iterations=5)
    mesh.compute_vertex_normals()
    
    return mesh
