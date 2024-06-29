import open3d as o3d
import numpy as np

from constants import THRESHOLD



def icp_p2p_registration(pcd_list):
    # register point cloud
    registered_point_cloud_generator = icp_p2p_register_and_yield_point_clouds(pcd_list)
    print("P2P Generator created")
    
    final_fused_point_cloud = o3d.geometry.PointCloud()

    # for i, pcd in enumerate(registered_point_cloud_generator):
    #     final_fused_point_cloud = o3d.geometry.PointCloud.concatenate([final_fused_point_cloud, pcd])
    
    for i, pcd in enumerate(registered_point_cloud_generator):
        final_fused_point_cloud += pcd  # Use the += operator to merge point clouds
    
    # save pcd and upload to blob

def icp_p2l_registration(pcd_list):
        # register point cloud
    registered_point_cloud_generator = icp_p2l_register_and_yield_point_clouds(pcd_list)
    print("P2L Generator created")
    
    final_fused_point_cloud = o3d.geometry.PointCloud()

    # for i, pcd in enumerate(registered_point_cloud_generator):
    #     final_fused_point_cloud = o3d.geometry.PointCloud.concatenate([final_fused_point_cloud, pcd])
    
    for i, pcd in enumerate(registered_point_cloud_generator):
        final_fused_point_cloud += pcd  # Use the += operator to merge point clouds
    
    # save pcd and upload to blob


def icp_p2l_register_and_yield_point_clouds(point_cloud_list):
        ref_pc = point_cloud_list[0]
        
        yield ref_pc  # Yield the reference point cloud as is

        for i in range(1, len(point_cloud_list)):
            src_pc = point_cloud_list[i]
            # _, transformed_pc = icp_registration(src_pc, ref_pc)
            _, transformed_pc = icp_p2l(src_pc, ref_pc)
            
            yield transformed_pc  # Yield the transformed point cloud
            
            
def icp_p2p_register_and_yield_point_clouds(point_cloud_list):
        ref_pc = point_cloud_list[0]
        
        yield ref_pc  # Yield the reference point cloud as is

        for i in range(1, len(point_cloud_list)):
            src_pc = point_cloud_list[i]
            # _, transformed_pc = icp_registration(src_pc, ref_pc)
            _, transformed_pc = icp_p2p(src_pc, ref_pc)
            
            yield transformed_pc  # Yield the transformed point cloud

def icp_p2p(pc_source, pc_ref):
    print(f"Performing ICP registration. Len Source = {len(pc_source.points)}, Len Ref = {len(pc_ref.points)}")
    trans_init = np.eye(4)

    # print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(
        pc_ref, pc_source, THRESHOLD, trans_init)
    print(evaluation)

    # print("Applying point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pc_ref, pc_source, THRESHOLD, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # print("")

    # Transform the point cloud according to the result of ICP
    print("Going to transform source pcd")
    pc_source.transform(reg_p2p.transformation)
    print("Source pcd transformed")
    return pc_ref, pc_source


def icp_p2l(pc_source, pc_ref, max_iterations=10, THRESHOLD=0.02, transformation_epsilon=1e-8):
    """
    Perform Point-to-Plane ICP registration between source and reference point clouds.

    Parameters:
    - pc_source: Open3D PointCloud object, source point cloud
    - pc_ref: Open3D PointCloud object, reference point cloud
    - max_iterations: int, maximum number of ICP iterations
    - THRESHOLD: float, distance THRESHOLD for correspondences
    - transformation_epsilon: float, transformation stopping criteria

    Returns:
    - pc_source_aligned: Open3D PointCloud object, aligned source point cloud
    - transformation_matrix: numpy array (4x4), transformation matrix
    """
    print(f"Performing ICP registration. Len Source = {len(pc_source.points)}, Len Ref = {len(pc_ref.points)}")
    # Perform Point-to-Plane ICP registration
    icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations,
                                                                     relative_fitness=1e-6,
                                                                     relative_rmse=1e-6)
    reg_p2l = o3d.pipelines.registration.registration_icp(
        pc_source, pc_ref, THRESHOLD, np.eye(4), 
        o3d.pipelines.registration.TransformationEstimationPointToPlane(), 
        icp_criteria)
    transformation_matrix = reg_p2l.transformation
    
    # Transform the source point cloud
    pc_source_aligned = pc_source.transform(transformation_matrix)
    
    return transformation_matrix, pc_source_aligned


