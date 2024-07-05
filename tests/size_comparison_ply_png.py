import os
import base64
import json
from datetime import datetime, timezone
import paho.mqtt.client as mqtt
import time
import numpy as np
import open3d as o3d
import random
import threading
import cv2

def view_pcd(pcd):
    x = input(f"Going to view pcd with points: {len(pcd.points)}. Y/n? ")
    if x.lower() == "y":
        o3d.visualization.draw_geometries([pcd])
        
def create_pcd_from_pngs(color_image, depth_image, K):
    """
    Create an Open3D PointCloud object from a color image, depth image, and intrinsic matrix.

    Parameters:
    - color_image (numpy.ndarray): Color image as a numpy array (shape: H x W x 3).
    - depth_image (numpy.ndarray): Depth image as a numpy array (shape: H x W).
    - K (numpy.ndarray): Intrinsic matrix (shape: 3x3).

    Returns:
    - o3d.geometry.PointCloud: Open3D PointCloud object.
    """
    
    # Create Open3D Image objects from numpy arrays
    depth_raw = o3d.geometry.Image(depth_image.astype(np.float32) / 1000.0)  # Convert depth to meters if needed
    color_raw = o3d.geometry.Image(color_image)

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, convert_rgb_to_intensity=False)

    # Create PinholeCameraIntrinsic object from K matrix
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(color_image.shape[1], color_image.shape[0], K[0][0], K[1][1], K[0][2], K[1][2])  # Fix here

    # Create PointCloud from RGBD image and intrinsics
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)

    return pcd




def save_json_to_local(payload, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(payload, json_file, indent=4)

def create_k_dict_by_camera(filepath) -> dict:
    k_dict = {}
    K = np.eye(3)
    with open(filepath, "r") as f:
        data = json.load(f)

        for _, camera in enumerate(data["cameras"]):
            # Extract camera parameters
            resolution = camera["Resolution"]
            focal = camera["Focal"]
            principal_point = camera["Principle_point"]
            camera_name = camera["Name"]

            # Create PinholeCameraIntrinsic object
            K = o3d.camera.PinholeCameraIntrinsic(
                width=resolution[0],
                height=resolution[1],
                fx=focal[0],
                fy=focal[1],
                cx=principal_point[0],
                cy=principal_point[1]
            )
            k_dict[camera_name] = K.intrinsic_matrix.tolist()
    return k_dict


def get_image_path(camera_dir, frame_id):
    if '3DMatch' in camera_dir:
        color_filename = f"{frame_id}_color.png"
        depth_filename = f"{frame_id}_depth.png"
    else:
        camera_name = camera_dir.split("\\")[-1]
        color_filename = f"{camera_name}_color_{frame_id}.png"
        depth_filename = f"{camera_name}_depth_{frame_id}.png"
    return os.path.join(camera_dir, color_filename), os.path.join(camera_dir, depth_filename)
    

# Function to process a frame
def process_frame(camera_dir, frame_id, K):
    color_path, depth_path = get_image_path(camera_dir, frame_id)

    color_image = cv2.imread(color_path)  # Example: Load color image
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # Example: Load depth image (16-bit unsigned)
    pcd = create_pcd_from_pngs(color_image, depth_image, K)
    save_point_cloud_to_ply(pcd, f"local_files\\test_{camera_dir}_{frame_id}.ply")
    # view_pcd(pcd)


def save_point_cloud_to_ply(point_cloud, file_path):
    try:
        o3d.io.write_point_cloud(file_path, point_cloud)
        print(f"Point cloud saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving point cloud to PLY: {e}")
        print(f"point_cloud class == {point_cloud.__class__}  file_path == {file_path.__class__}")
        return False  
        
    
       
       
# Main function to control the flow
def main(base_directory):
    if '3DMatch' not in base_directory:
        k_dict = create_k_dict_by_camera("cam_params.json")
    else:
        k_dict = K
    print(f"K dict class: {k_dict.__class__}")

    if True:
        camera_dirs = [os.path.join(base_directory, d) for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

        # chosen_frame = random.choice(range(1, 219))  # Assuming frames are from 1 to 219
        chosen_frame = 132
        
        if '3DMatch' in base_directory:
            chosen_frame_str = f"frame-{chosen_frame:06d}"
        else: # own data
            chosen_frame_str = f"f{chosen_frame:04d}"
        print(f"Chosen frame: f{chosen_frame:04d}")

        threads = []
        for chosen_camera_dir in camera_dirs:
            print(f"Chosen camera directory: {chosen_camera_dir}")
            cam_name = chosen_camera_dir.split("\\")[-1]
            print(f"Chosen camera NAME: {cam_name}")
            thread = threading.Thread(target=process_frame, args=(chosen_camera_dir, chosen_frame_str, k_dict[cam_name]))
            threads.append(thread)
            thread.start()
            time.sleep(0.5)

    

if __name__ == "__main__":
    base_directory = "..\\data\\png_data"
    print(f"Selected dataset dir: {base_directory}")
    main(base_directory)
    
