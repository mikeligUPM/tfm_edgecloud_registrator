import open3d as o3d
import numpy as np

def convert_ply_bgr_to_rgb(pcd):
    # Check if the point cloud has color information
    if not pcd.has_colors():
        print("Point cloud does not have color information.")
        return None
    
    # Extract colors from the point cloud (BGR format)
    colors = np.asarray(pcd.colors)
    
    # Convert BGR to RGB
    colors_rgb = colors[:, [2, 1, 0]]  # Swap B and R channels
    
    # Update the point cloud with RGB colors
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
    
    return pcd

def visualize_point_cloud(pcd, win=""):
    if pcd:
        o3d.visualization.draw_geometries([pcd], window_name=win)
    else:
        print("No point cloud to visualize.")


def main(ply_filename):
    # Read the point cloud from the ply file
    pcd = o3d.io.read_point_cloud(ply_filename)
    visualize_point_cloud(pcd, win="Original")
    pcd = convert_ply_bgr_to_rgb(pcd)
    visualize_point_cloud(pcd, win="RGB")

# Path to the input .ply file
ply_filename = "experiment__ownData_downsampled_geotransformer.ply"
main(ply_filename)