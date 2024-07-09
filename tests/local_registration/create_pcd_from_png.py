import base64
import numpy as np
import open3d as o3d
import cv2
import json
import threading
from threading import Timer
import time
import os
import base64
import json
from datetime import datetime, timezone
import time
import numpy as np
import random
import threading

# K = [
#     [585.0, 0.0, 320.0],
#     [0.0, 585.0, 240.0],
#     [0.0, 0.0, 1.0]
# ]

# # K = [
# #     [515.0, 0.0, 319.5],
# #     [0.0, 525.0, 239.5],
# #     [0.0, 0.0, 1.0]
# # ]

# # K = [
# #     [598.84, 0.0, 3120],
# #     [0.0, 587.62, 240],
# #     [0.0, 0.0, 1.0]
# # ]

# K = [
#     [979.7475404724926, 0.0, 953.939585967863],
#     [0.0, 978.5155360713557, 535.432112362691],
#     [0.0, 0.0, 1.0]
# ]


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

def view_pcd(pcd):
    if not isinstance(pcd, list):
        x = input(f"Going to view pcd with points: {len(pcd.points)}. Y/n? ")
        if x.lower() == "y":
            o3d.visualization.draw_geometries([pcd])
    else:
        o3d.visualization.draw_geometries(pcd)

def aaacreate_pcd_from_pngs(color_image, depth_image):
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

def create_pcd_from_pngs_data(color_image, depth_image, K):
    """
    Create an Open3D PointCloud object from a color image, depth image, and intrinsic matrices.

    Parameters:
    - color_image (numpy.ndarray): Color image as a numpy array (shape: H x W x 3).
    - depth_image (numpy.ndarray): Depth image as a numpy array (shape: H x W).
    - K_rgb (numpy.ndarray): Intrinsic matrix for the RGB camera (shape: 3x3).
    - K_depth (numpy.ndarray): Intrinsic matrix for the depth camera (shape: 3x3).

    Returns:
    - o3d.geometry.PointCloud: Open3D PointCloud object.
    """
    # Create Open3D Image objects from numpy arrays
    depth_raw = o3d.geometry.Image(depth_image.astype(np.float32) / 1000.0)  # Convert depth to meters if needed
    color_raw = o3d.geometry.Image(color_image)

    # Create RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, convert_rgb_to_intensity=False
    )


    # Create PinholeCameraIntrinsic object for the depth camera
    intrinsics_depth = o3d.camera.PinholeCameraIntrinsic()
    intrinsics_depth.set_intrinsics(
        depth_image.shape[1], depth_image.shape[0],
        K[0][0], K[1][1], K[0][2], K[1][2]
    )

    # Create PointCloud from RGBD image and intrinsics
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics_depth)

    # Flip the orientation, so it looks upright, not upside-down
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

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


# Function to process a frame
def process_frame(color_path, depth_path, camera_name, view=True):
    k_dict = create_k_dict_by_camera("cam_params.json")
    if isinstance(k_dict, dict):
        k_list = k_dict[camera_name]
    elif isinstance(k_dict, list):
        k_list = k_dict
    color_image = cv2.imread(color_path)  # Example: Load color image
    depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)  # Example: Load depth image (16-bit unsigned)
    pcd = create_pcd_from_pngs_data(color_image, depth_image, k_list)
    if view:
        view_pcd(pcd)
    return pcd




def check_file_exists(file_path):
    try:
        with open(file_path, "r") as f:
            pass
        return True
    except Exception as e:
        print(f"Error with file {file_path}: {e}")
        return False


def main(FILE_NUM=2):
    pcd_list = []
    # base_image_path = "../data/first8_frames"
    base_image_path = "data/first8_frames/"
    # file_names = [("color1.png", "depth1.png"),
    #               ("color2.png", "depth2.png")]
    file_path_list = []

    for i in range(1, FILE_NUM+1):
        file_path_list.append((f"{base_image_path}/color{i}.png", f"{base_image_path}/depth{i}.png"))

    flattened_list = [item for sublist in file_path_list for item in sublist]
    for f_path in flattened_list:
        check_file_exists(f_path)

    for index,file_pair in enumerate(file_path_list):
        pcd = process_frame(file_pair[0], file_pair[1], view=False, camera_name=str(index+1))
        if pcd:
            pcd_list.append(pcd)

    return pcd_list


    return
    # base_directory = ds_selection_prompt()
    color_pth="frame-000000_color.png"
    with open(color_pth, "r") as f:
        print("Color ok")

    depth_pth="frame-000000_depth.png"
    with open(depth_pth, "r") as f:
        print("Depth ok")

    # color_pth="c1.png"
    # with open(color_pth, "r") as f:
    #     print("Color ok")

    # depth_pth="d1.png"
    # with open(depth_pth, "r") as f:
    #     print("Depth ok")


    process_frame(color_pth, depth_pth)
if __name__ == "__main__":
    pcd_list = main()
    view_pcd(pcd_list)