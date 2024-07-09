import open3d as o3d
import numpy as np

# VOXEL_SIZE = 0.025 # 2.5 cm, same as used in Geotransformer
# VOXEL_SIZE = 0.00001 * 2.5 ## Own data
i = 1

from logger_config import logger
# from constants import BACKBONE_INIT_VOXEL_SIZE
from helper_funs import calculate_registration_metrics, get_config

########################    ICP P2L    ###########################################
def icp_p2l_registration_ransac(pcd_list):
    # register point cloud
    vox_size, threshold = get_config(1)
    registered_point_cloud_generator = icp_p2l_register_and_yield_point_clouds(pcd_list, vox_size)
    logger.info("P2L Generator created")
    # threshold = BACKBONE_INIT_VOXEL_SIZE * 0.4
    final_fused_point_cloud = o3d.geometry.PointCloud()

    logger.info("P2L registration INIT")
    for i, pcd in enumerate(registered_point_cloud_generator):
        final_fused_point_cloud += pcd  # Use the += operator to merge point clouds
        calculate_registration_metrics(pcd, pcd_list[0], threshold, i=i+1, final_result_print=(i+1 == 8))
    logger.info("P2L registration END")

    # calculate_registration_metrics(final_fused_point_cloud, pcd_list[0], threshold, final_result_print=True)
    return final_fused_point_cloud


def icp_p2l_register_and_yield_point_clouds(point_cloud_list, vox_size):
        ref_pc = point_cloud_list[0]
        radius_normal = vox_size * 2
        ref_pc.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        yield ref_pc  # Yield the reference point cloud as is

        for i in range(1, len(point_cloud_list)):
            src_pc = point_cloud_list[i]
            radius_normal = vox_size * 2
            src_pc.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            _, transformed_pc = icp_p2l(src_pc, ref_pc, vox_size)

            yield transformed_pc  # Yield the transformed point cloud

def icp_p2l(pc_source, pc_ref, vox_size):
    source_down, source_fpfh = preprocess_point_cloud(pc_source, vox_size)
    target_down, target_fpfh = preprocess_point_cloud(pc_ref, vox_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh, vox_size)

    result_icp = refine_registration(pc_source, pc_ref, result_ransac, p2p=False)
    pc_source_aligned = pc_source.transform(result_icp.transformation)
    return pc_ref, pc_source_aligned


########################    ICP P2P     ###########################################
def icp_p2p_registration_ransac(pcd_list):
    vox_size, threshold = get_config(1)
    # register point cloud
    registered_point_cloud_generator = icp_p2p_register_and_yield_point_clouds(pcd_list)
    logger.info("P2P Generator created")
    # threshold = vox_size * 0.4
    final_fused_point_cloud = o3d.geometry.PointCloud()

    logger.info("P2P registration INIT")
    for i, pcd in enumerate(registered_point_cloud_generator):
        final_fused_point_cloud += pcd  # Use the += operator to merge point clouds
        calculate_registration_metrics(pcd, pcd_list[0], threshold, i=i+1, final_result_print=(i+1 == 8))
    logger.info("P2P registration END")

    calculate_registration_metrics(final_fused_point_cloud, pcd_list[0], threshold, final_result_print=True)
    return final_fused_point_cloud

def icp_p2p_register_and_yield_point_clouds(point_cloud_list):
    ref_pc = point_cloud_list[0]

    yield ref_pc  # Yield the reference point cloud as is

    for i in range(1, len(point_cloud_list)):
        src_pc = point_cloud_list[i]
        # _, transformed_pc = icp_registration(src_pc, ref_pc)
        _, transformed_pc = icp_p2p(src_pc, ref_pc)

        yield transformed_pc  # Yield the transformed point cloud

def icp_p2p(pc_source, pc_ref):
    vox_size, threshold = get_config(1)
    source_down, source_fpfh = preprocess_point_cloud(pc_source, vox_size)
    target_down, target_fpfh = preprocess_point_cloud(pc_ref, vox_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh)

    result_icp = refine_registration(pc_source, pc_ref, result_ransac, p2p=True)
    pc_source_aligned = pc_source.transform(result_icp.transformation)
    return pc_ref, pc_source_aligned


########################    HELPER FUNS     ###########################################
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, p2p=True):
    vox_size, distance_threshold = get_config(1)
    # distance_threshold = vox_size * 1.5
    # logger.debug(":: RANSAC registration on downsampled point clouds.")
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def refine_registration(source, target, result_ransac, p2p=True):
    vox_size, distance_threshold = get_config(1)
    # distance_threshold = vox_size * 0.4
    if p2p:
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    else:
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result



def preprocess_point_cloud(pcd, vox_size):
    pcd_down = pcd.voxel_down_sample(vox_size)
    radius_normal = vox_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = vox_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


# def execute_global_registration(source_down, target_down, source_fpfh,
#                                 target_fpfh, vox_size):
#     distance_threshold = vox_size * 1.5
#     result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#         source_down, target_down, source_fpfh, target_fpfh, True,
#         distance_threshold,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#         3, [
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
#                 0.9),
#             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
#                 distance_threshold)
#         ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
#     return result