import open3d as o3d
import numpy as np
import os
from datetime import datetime, timezone

# from registrator_icp import icp_p2p_registration, icp_p2l_registration
from registration_icp_ransac import icp_p2p_registration_ransac, icp_p2l_registration_ransac
from cc import geotransformer_reg
import create_pcd_from_png

from constants import BACKBONE_INIT_VOXEL_SIZE
# utc_time_str = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')

VOXEL_SIZE = 0.025 # 2.5 cm, same as used iin Geotransformer
# VOXEL_SIZE = BACKBONE_INIT_VOXEL_SIZE #0.00001 * 2.5
#VOXEL_SIZE = 0.00001 * 2.5

colors_dict = {
    "red": [1, 0, 0],
    "green": [0, 1, 0],
    "blue": [0, 0, 1],
    "yellow": [1, 1, 0],
    "magenta": [1, 0, 1],
    "cyan": [0, 1, 1],
    "pink": [1, 0.75, 0.8],
    "orange": [1, 0.5, 0],
    "purple": [0.5, 0, 0.5]
}

registration_methods = {
    #"icp_p2p": icp_p2p_registration,
    #"icp_p2l": icp_p2l_registration,
    "icp_p2p_ransac": icp_p2p_registration_ransac,
    "icp_p2l_ransac": icp_p2l_registration_ransac,
    "geotransformer": geotransformer_reg,
}

def visualize_point_cloud(file_path, downsample=True, paint=False):
    pcd = get_pcd_from_ply(file_path)

    if downsample:
        pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        print(f"#### downsampled PCD POINTS == {pcd}")
        x = input("")
    # Check if the point cloud is empty
    if pcd.is_empty():
        print(f"The point cloud file at {file_path} is empty or could not be read.")
        return

    if paint:
        pcd = paint_pcd_of_color(pcd, colors_dict["blue"])

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

def paint_pcd_of_color(pcd, color):
    num_points = len(pcd.points)
    color_array = np.tile(color, (num_points, 1))  # Create an array with the specified color repeated for all points
    pcd.colors = o3d.utility.Vector3dVector(color_array)
    # pcd.colors = o3d.utility.Vector3dVector(color_values)
    return pcd

def get_pcd_from_ply(file_path):
    with open(file_path) as f:
        print(file_path)
    # Read the point cloud from the PLY file
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"PCD POINTS == {pcd}")
    return pcd

def view_point_cloud(pcd_list, desc="No name"):
    if not isinstance(pcd_list, list):
        pcd_list = [pcd_list]

    o3d.visualization.draw_geometries(pcd_list, window_name=desc)

def save_point_cloud_to_ply(point_cloud, file_path):
    try:
        o3d.io.write_point_cloud(file_path, point_cloud)
        utc_time_str = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S:%f UTC')
        print(f"{utc_time_str} --- Point cloud saved to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving point cloud to PLY: {e}")
        print(f"point_cloud class == {point_cloud.__class__}  file_path == {file_path.__class__}")
        return False

def main(file_num = 2,
         downsample=True,
         color=True,
         target_registration="icp_p2p",
         visualization=True,
         png_source=False,
         experiment_name="exp_name"):

    # Load input
    if png_source:
        pcd_list = create_pcd_from_png.main(file_num)
    else:
        data_path =  "/home/mirazola/Documents/geotrans_v2/GeoTransformer/experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn"
        pcd_list = [pcd for i in range(file_num) if (pcd := get_pcd_from_ply(f'{data_path}/data/7-scenes-redkitchen/cloud_bin_{i}.ply'))]

    if downsample:
        pcd_list = [pcd.voxel_down_sample(voxel_size=VOXEL_SIZE) for pcd in pcd_list]

    # if visualization:
    #     for i,pcd in enumerate(pcd_list):
    #         view_point_cloud(pcd)
    #         save_point_cloud_to_ply(pcd, os.path.join("results", f"pcd_{i}.ply"))
    if color:
        color_keys = list(colors_dict.keys())
        pcd_list = [paint_pcd_of_color(pcd, colors_dict[color_keys[i % len(color_keys)]]) for i, pcd in enumerate(pcd_list)]

    # if visualization:
    #     view_point_cloud(pcd_list, desc=f"{experiment_name} - Original {file_num} pcds")
    registration_func = registration_methods.get(target_registration)
    if registration_func:
        final_fused_point_cloud = registration_func(pcd_list)
        if visualization:
            view_point_cloud(final_fused_point_cloud, desc=f"{experiment_name} - Registered pcds")
        reg_result_file_path = os.path.join("results", f"{experiment_name}.ply")
        save_point_cloud_to_ply(final_fused_point_cloud, reg_result_file_path)



if __name__ == "__main__":

    FILE_NUM = 8

    TARGET_REG = "geotransformer"

    CREATE_FROM_PNG = False #False # False for 3dmatch, True for own_data

    VISUALIZE = False

    APPLY_UNIFORM_COLOR = False

    DOWNSAMPLE_IMPUT_PCD = True


    EXPERIMENT_NAME = "experiment__"
    if CREATE_FROM_PNG:
        EXPERIMENT_NAME += "ownData_"
    else:
        EXPERIMENT_NAME += "3dmatchRedkitchen_"
    if DOWNSAMPLE_IMPUT_PCD:
        EXPERIMENT_NAME += f"downsampled_{TARGET_REG}"
    else:
        EXPERIMENT_NAME += f"{TARGET_REG}"

    ### Single test
    utc_time_str = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S:%f UTC')
    print(f"{utc_time_str} --- Starting experiment: {EXPERIMENT_NAME}")
    main(file_num=FILE_NUM,
         downsample=DOWNSAMPLE_IMPUT_PCD,
         color=APPLY_UNIFORM_COLOR,
         visualization=VISUALIZE,
         target_registration=TARGET_REG,
         png_source=CREATE_FROM_PNG,
         experiment_name=EXPERIMENT_NAME)

    ### All test sequentially (no visualization)
    # for key in registration_methods.keys():
    #     main(file_num=FILE_NUM, downsample=True, color=True, target_registration=key, visualization=False)

# FILE_NUM = 8
# for key in registration_methods.keys():
#     logger = logger_config.setup_logger(log_file=f"experiment__{key}.log")
#     utc_time_str = datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S:%f UTC')
#     print(f"\n\n\n{utc_time_str} --- STARTING EXPERIMENT: {key} WITH NUM FILES: {FILE_NUM}")
#     main(file_num=FILE_NUM, downsample=True, color=True, target_registration=key, visualization=False)
#### results/experiment__icp_p2l_ransac.log