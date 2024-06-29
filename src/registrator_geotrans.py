import torch
import numpy as np

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud
from config import make_cfg
from model import create_model
import open3d as o3d


### Helper functions
def process_point_clouds(model, cfg, src_points, ref_points):
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
        "transform": np.identity(4, dtype=np.float32)
    }

    neighbor_limits = [38, 36, 36, 38]  # default setting in 3DMatch
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
    )

    data_dict = to_cuda(data_dict)
    output_dict = model(data_dict)
    data_dict = release_cuda(data_dict)
    output_dict = release_cuda(output_dict)

    ref_points = output_dict["ref_points"]
    src_points = output_dict["src_points"]
    estimated_transform = output_dict["estimated_transform"]
    transform = data_dict["transform"] if "transform" in data_dict else None

    return ref_points, src_points, estimated_transform


def transform_src_to_ref(src_points, ref_points, model, cfg):
    ref_points, src_points, estimated_transform = process_point_clouds(model, cfg, src_points, ref_points)

    ref_pc = make_open3d_point_cloud(ref_points)
    ref_pc.estimate_normals()

    src_pc = make_open3d_point_cloud(src_points)
    src_pc.estimate_normals()

    transformed_pc = src_pc.transform(estimated_transform)

    return ref_pc, transformed_pc


# Define the generator function for sequential registration
def register_and_yield_point_clouds(point_cloud_list, model, cfg):
    ref_points = point_cloud_list[0]
    ref_pc = make_open3d_point_cloud(ref_points)
    ref_pc.estimate_normals()
    
    yield ref_pc  # Yield the reference point cloud as is

    for i in range(1, len(point_cloud_list)):
        src_points = point_cloud_list[i]
        ref_pc, transformed_pc = transform_src_to_ref(src_points, ref_points, model, cfg)
        
        yield transformed_pc  # Yield the transformed point cloud


def geotrans_registration(pcd_list):
    cfg = make_cfg()
    model = create_model(cfg).cuda()
    state_dict = torch.load('weights/geotransformer-3dmatch.pth.tar')
    model.load_state_dict(state_dict["model"])
    model.eval()

    # Create a generator for the registration process
    registered_point_cloud_generator = register_and_yield_point_clouds(pcd_list, model, cfg)

    # Collect the final fused point cloud by iterating over the generator
    final_fused_point_cloud = []
    for i, pcd in enumerate(registered_point_cloud_generator):
        final_fused_point_cloud.append(pcd)

    # Visualization
    o3d.visualization.draw_geometries(final_fused_point_cloud, window_name=f"Final Fused Point Cloud. Total_PCDs = {len(final_fused_point_cloud)}")
    
