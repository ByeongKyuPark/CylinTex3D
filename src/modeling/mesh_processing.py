"""
Mesh post-processing module - Improves mesh quality with hole filling and Poisson reconstruction.
"""

import numpy as np
import trimesh

def fill_holes(mesh, hole_size=None):
    """
    Fill holes in the mesh using trimesh's built-in function.
    
    Args:
        mesh: Trimesh object
        hole_size: Not used, kept for API compatibility
        
    Returns:
        Mesh with filled holes
    """
    print("Filling holes...")
    
    # Make a copy of the mesh
    filled_mesh = mesh.copy()
    
    # Fill holes - note that trimesh's fill_holes() doesn't take a size parameter
    filled_mesh.fill_holes()
    
    # Make sure the mesh is still valid
    filled_mesh.remove_degenerate_faces()
    filled_mesh.remove_duplicate_faces()
    
    return filled_mesh

def improve_mesh(mesh, smooth_iterations=None, lambda_factor=None, fill_holes_size=100, 
                 clean_mesh=True, remove_unreferenced=True, subdivide=False):
    """
    Apply a series of improvements to the mesh.
    
    Args:
        mesh: Input Trimesh object
        smooth_iterations: Not used, kept for API compatibility
        lambda_factor: Not used, kept for API compatibility
        fill_holes_size: Not used, kept for API compatibility
        clean_mesh: Whether to clean the mesh
        remove_unreferenced: Whether to remove unreferenced vertices
        subdivide: Whether to subdivide the mesh for higher resolution
        
    Returns:
        Improved mesh
    """
    print("Improving mesh quality...")
    
    # First, fill holes
    result_mesh = fill_holes(mesh)
    
    # Optional subdivision for higher resolution
    if subdivide:
        print("Subdividing mesh for higher resolution...")
        result_mesh = result_mesh.subdivide()
    
    # Clean up the mesh
    if clean_mesh:
        print("Cleaning mesh...")
        result_mesh.remove_degenerate_faces()
        result_mesh.remove_duplicate_faces()
        if remove_unreferenced:
            result_mesh.remove_unreferenced_vertices()
    
    # Final processing
    result_mesh.process()
    
    # Return the improved mesh
    return result_mesh

def poisson_reconstruction(points, normals, depth=9, scale=1.1, samples_per_node=1.5, point_weight=4.0):
    """
    Perform Poisson surface reconstruction.
    Requires the Open3D library for best results.
    
    Args:
        points: Point cloud vertices
        normals: Normal vectors for each point
        depth: Octree depth for reconstruction detail (higher = more detail)
        scale: Scale factor for reconstruction
        samples_per_node: Minimum number of samples per octree node
        point_weight: Weight for point constraints (higher = more adheres to input)
        
    Returns:
        Reconstructed mesh as Trimesh object
    """
    try:
        import open3d as o3d
        print(f"Performing Poisson reconstruction (depth={depth})...")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        
        # Perform Poisson reconstruction with optimized parameters
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=scale, scale=scale,
            linear_fit=True, n_threads=4)
        
        # Density-based trimming to clean up artifacts
        vertices_to_remove = densities < np.quantile(densities, 0.01)  # Remove lowest 1% density
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        # Convert back to trimesh
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        result = trimesh.Trimesh(vertices=vertices, faces=faces)
        result.remove_degenerate_faces()
        result.remove_duplicate_faces()
        result.remove_unreferenced_vertices()
        
        return result
    
    except ImportError:
        print("Warning: Open3D not available. Falling back to basic reconstruction.")
        # Create a simple mesh from points using convex hull as fallback
        cloud = trimesh.points.PointCloud(points)
        hull = cloud.convex_hull
        return hull

def volume_to_mesh_poisson(volume, level=0.5):
    """
    Create a mesh from a volume using marching cubes followed by Poisson reconstruction
    for a smoother, cleaner result.
    
    Args:
        volume: 3D binary volume
        level: Iso-surface level
        
    Returns:
        Trimesh object
    """
    from skimage import measure
    
    # First get points and normals using marching cubes
    verts, faces, normals, _ = measure.marching_cubes(volume, level=level)
    
    # Create a temporary mesh
    temp_mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    # Extract point cloud and normals
    points = temp_mesh.vertices
    point_normals = temp_mesh.vertex_normals
    
    # Clean up point cloud and normals (remove NaNs and normalize)
    valid_indices = ~np.isnan(point_normals).any(axis=1)
    cleaned_points = points[valid_indices]
    cleaned_normals = point_normals[valid_indices]
    norm = np.linalg.norm(cleaned_normals, axis=1)
    valid_norm = norm > 0
    cleaned_normals[valid_norm] = cleaned_normals[valid_norm] / norm[valid_norm, np.newaxis]
    
    # Use Poisson reconstruction for a better mesh
    print("Applying Poisson reconstruction for a smoother mesh...")
    try:
        mesh = poisson_reconstruction(cleaned_points, cleaned_normals)
        return mesh
    except Exception as e:
        print(f"Poisson reconstruction failed: {e}. Falling back to marching cubes.")
        return temp_mesh
