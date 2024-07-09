import os
import torch
import numpy as np

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color
from geotransformer.utils.registration import compute_registration_error

from config import make_cfg
from model import create_model
import open3d as o3d
import re

from logger_config import logger
from constants import BACKBONE_INIT_VOXEL_SIZE
from helper_funs import calculate_registration_metrics, get_config

#VOXEL_SIZE = 0.00001 * 2.5
VOXEL_SIZE = 0.025
# VOXEL_SIZE = BACKBONE_INIT_VOXEL_SIZE


### Helper functions

def load_point_cloud(pcd):
    points = np.asarray(pcd.points)

    # Center the point cloud
    points -= np.mean(points, axis=0)

    return points


def load_data(src_pc, ref_pc, gt_file=None):
    src_points = load_point_cloud(src_pc)
    logger.debug(f"SRC PCD POINTS == {src_points}")
    ref_points = load_point_cloud(ref_pc)
    #x = input(f"REF PCD POINTS == {src_points}")
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])


    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
        "transform": np.identity(4, dtype=np.float32)
    }

    return data_dict


import json
import numpy as np

def default_serializer(obj):
    """
    Default JSON serializer for objects not serializable by default.
    """
    if isinstance(obj, (np.ndarray)):
        return obj.tolist()
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def save_dict_to_json(data_dict, file_path="data_dict.json"):
    """
    Save a dictionary to a JSON file.

    Parameters:
    - data_dict: dict, the dictionary to save
    - file_path: str, the path to the JSON file (default is "data_dict.json")
    """
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data_dict, json_file, indent=4, default=default_serializer)
        logger.debug(f"Dictionary successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while saving the dictionary to JSON: {e}")

def process_point_clouds(model, cfg, src_pc, ref_pc):
    print_check(11)
    data_dict = load_data(src_pc, ref_pc)
    print_check(12)
    # logger.debug(f"\n\n{list(data_dict.keys())}\n\n")
    neighbor_limits = [38, 36, 36, 38]  # default setting in 3DMatch
    # neighbor_limits = [100,100,100,100] # lower IR
    # neighbor_limits = [800,800,800,800] # Doesn't run, CUDA out of memory
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )
    # save_dict_to_json(data_dict, "data_dict_0.json")
    print_check(13)

    data_dict = to_cuda(data_dict)
    print_check(14)
    output_dict = model(data_dict)
    print_check(15)
    data_dict = release_cuda(data_dict)
    print_check(16)
    output_dict = release_cuda(output_dict)
    print_check(17)

    ref_points = output_dict["ref_points"]
    src_points = output_dict["src_points"]
    estimated_transform = output_dict["estimated_transform"]
    transform = data_dict["transform"] if "transform" in data_dict else None

    return ref_points, src_points, estimated_transform


def sort_key(filename):
    # Extract the numeric part of the filename using regular expressions
    numeric_part = re.search(r'\d+', filename).group()
    # Convert the numeric part to an integer for sorting
    return int(numeric_part)

def down_pc(pcd):
    logger.debug(f"POINTS BEFORE DOWN: {pcd.points}")
    pcd = pcd.voxel_down_sample(voxel_size=0.025)
    logger.debug(f"POINTS AFTER DOWN: {pcd.points}")
    return pcd

def transform_src_to_ref(src_pc, ref_pc, src_number, model, cfg):
    print_check(10)
    ref_points, src_points, estimated_transform = process_point_clouds(model, cfg, src_pc, ref_pc)
    print_check(18)
    # ref_pc = make_open3d_point_cloud(load_point_cloud(ref_path))
    # ref_pc = down_pc(ref_pc)

    print_check(19)
    ref_pc.estimate_normals()

    print_check(20)
    src_pc.estimate_normals()
    print_check(21)
    logger.debug(f"est trans class: {estimated_transform.__class__}")
    transformed_pc = src_pc.transform(estimated_transform)
    print_check(22)

    return ref_pc, transformed_pc

# Define the generator function for sequential registration
def register_and_yield_point_clouds(pcd_list, model, cfg):
    print_check(7)
    ref_pc = pcd_list[0]
    print_check(8)
    # ref_pc.estimate_normals()

    # yield ref_pc  # Yield the reference point cloud as is

    for i in range(0, len(pcd_list)):
        src_pc = pcd_list[i]
        ref_pc, transformed_pc = transform_src_to_ref(src_pc, ref_pc, i, model, cfg)

        yield transformed_pc  # Yield the transformed point cloud
            # I don't do a 'return' in order to save memory, bc I dont need the intermediate transformd cloud points

def count_pth_files(scene_path):
    # List all files in the directory
    directory = f'/home/mikel.irazola/geotrans_v1/GeoTransformer/data/3DMatch/data/{scene_path}/'
    files = os.listdir(directory)

    # Filter files that end with '.pth'
    pth_files = [file for file in files if file.endswith('.pth')]

    # Count the number of '.pth' files
    return len(pth_files)


### Scene selection TBD
def select_dataset_prompt():
    pass

def select_scene_prompt():
    pass

def print_check(num):
    logger.debug(f"CHECKPOINT {num}")

### MAIN
def geotransformer_reg(pcd_list):
    logger.info(f"Starting geotrans with pcd len == {len(pcd_list)}")
    vox_size, threshold = get_config(0)
    cfg = make_cfg()
    print_check(1)
    model = create_model(cfg).cuda()
    print_check(2)
    state_dict = torch.load('../../weights/geotransformer-3dmatch.pth.tar')
    print_check(3)
    # state_dict = torch.load('weights/geotransformer-3dmatch.pth.tar')
    model.load_state_dict(state_dict["model"])
    print_check(4)
    model.eval()
    print_check(5)

    # Create a generator for the registration process
    registered_point_cloud_generator = register_and_yield_point_clouds(pcd_list, model, cfg)
    print_check(6)

    final_fused_point_cloud = o3d.geometry.PointCloud()
    logger.info(f"Geotrans registration INIT")
    for i, pcd in enumerate(registered_point_cloud_generator):
        final_fused_point_cloud += pcd  # Use the += operator to merge point clouds
        calculate_registration_metrics(final_fused_point_cloud, pcd_list[0], threshold=threshold)
    logger.info(f"Geotrans registration END")
    print_check(98)
    calculate_registration_metrics(final_fused_point_cloud, pcd_list[0], threshold=threshold, final_result_print=True)
    logger.debug(f"TOTAL FINAL PCD POINTS == {final_fused_point_cloud.points}  with class == {final_fused_point_cloud.__class__}")

    return final_fused_point_cloud

if __name__ == "__main__":
   pass