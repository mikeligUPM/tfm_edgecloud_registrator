import open3d as o3d
import numpy as np

from constants import VOXEL_SIZE

def calculate_registration_metrics_v1(pc_source, pc_ref, threshold):
    # Calculate distances between corresponding points
    distances = pc_source.compute_point_cloud_distance(pc_ref)

    # Convert distances to a NumPy array for easy calculation
    distances = np.asarray(distances)

    # Calculate RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean(np.square(distances)))
    print(f"### RMSE: {rmse}")

    # Set a distance threshold to determine inliers
    distance_threshold = threshold 
    print(f"### Distance Threshold: {distance_threshold}")

    # Calculate Inlier Ratio
    inliers = distances < distance_threshold
    inlier_ratio = np.sum(inliers) / len(distances)
    print(f"### Inlier Ratio: {inlier_ratio}")

    # Calculate FMR (False Match Rate)
    false_matches = distances >= distance_threshold
    fmr = np.sum(false_matches) / len(distances)
    print(f"### False Match Rate: {fmr}")

    # Calculate Registration Recall
    # Here we assume the ground truth is that all points should be matched within the threshold
    registration_recall = inlier_ratio  # Since all points are considered for recall
    print(f"### Registration Recall: {registration_recall}")

def calculate_registration_metrics(pc_source, pc_ref, threshold, p2p_registration=True):
    if p2p_registration:
        # Calculate point-to-point RMSE
        distances_p2p = np.linalg.norm(np.asarray(pc_source.points) - np.asarray(pc_ref.points), axis=1)
        point_to_point_rmse = np.sqrt(np.mean(distances_p2p ** 2))
        print(f"### Point-to-point RMSE: {point_to_point_rmse:.4f}")
    else:
        # Calculate point-to-plane RMSE
        distances_p2l = np.abs(np.sum((np.asarray(pc_source.points) - np.asarray(pc_ref.points)) * np.asarray(pc_ref.normals), axis=1))
        point_to_plane_rmse = np.sqrt(np.mean(distances_p2l ** 2))
        print(f"### Point-to-plane RMSE: {point_to_plane_rmse:.4f}")

    # Calculate planar surface deviation
    planar_deviation = np.mean(distances_p2l)
    print(f"### Planar surface deviation: {planar_deviation:.4f}")

    # Calculate percentage of overlapping points
    _, inds = o3d.pipelines.registration.compute_point_cloud_distance(pc_source, pc_ref)
    overlap_ratio = np.sum(np.asarray(inds) < 0.1) / float(len(pc_source.points))
    print(f"### Overlap ratio: {overlap_ratio:.4f}")

    # Calculate RMSE, Inlier Ratio, False Match Rate, Registration Recall
    distances = pc_source.compute_point_cloud_distance(pc_ref)
    distances = np.asarray(distances)
    rmse = np.sqrt(np.mean(np.square(distances)))
    print(f"### RMSE: {rmse:.4f}")

    inlier_ratio = np.sum(distances < threshold) / len(distances)
    print(f"### Inlier Ratio: {inlier_ratio:.4f}")

    fmr = np.sum(distances >= threshold) / len(distances)
    print(f"### False Match Rate: {fmr:.4f}")

    registration_recall = inlier_ratio
    print(f"### Registration Recall: {registration_recall:.4f}")






def icp_p2p_registration(pcd_list):
    # register point cloud
    registered_point_cloud_generator = icp_p2p_register_and_yield_point_clouds(pcd_list)
    print("P2P Generator created")
    threshold = 5 * VOXEL_SIZE
    final_fused_point_cloud = o3d.geometry.PointCloud()

    # for i, pcd in enumerate(registered_point_cloud_generator):
    #     final_fused_point_cloud = o3d.geometry.PointCloud.concatenate([final_fused_point_cloud, pcd])
    
    for i, pcd in enumerate(registered_point_cloud_generator):
        final_fused_point_cloud += pcd  # Use the += operator to merge point clouds
        
    calculate_registration_metrics(final_fused_point_cloud, pcd_list[0], threshold)
    return final_fused_point_cloud

def icp_p2l_registration(pcd_list):
        # register point cloud
    registered_point_cloud_generator = icp_p2l_register_and_yield_point_clouds(pcd_list)
    print("P2L Generator created")
    threshold = 10 * VOXEL_SIZE
    final_fused_point_cloud = o3d.geometry.PointCloud()
    
    for i, pcd in enumerate(registered_point_cloud_generator):
        final_fused_point_cloud += pcd  # Use the += operator to merge point clouds
        
    calculate_registration_metrics(final_fused_point_cloud, pcd_list[0], threshold)
    return final_fused_point_cloud
    


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
    threshold = 5 * VOXEL_SIZE
    evaluation = o3d.pipelines.registration.evaluate_registration(
        pc_ref, pc_source, threshold, trans_init)
    print(evaluation)

    # print("Applying point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pc_ref, pc_source, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p.__class__)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # print("")

    # Transform the point cloud according to the result of ICP
    print("Going to transform source pcd")
    pc_source.transform(reg_p2p.transformation)
    print("Source pcd transformed")
    
    distances = pc_source.compute_point_cloud_distance(pc_ref)
    # Convert distances to a NumPy array for easy calculation
    distances = np.asarray(distances)

    # Calculate RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean(np.square(distances)))
    print(f"############ RMSE: {rmse}")

    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(distances))
    print(f"############ MAE: {mae}")
    return pc_ref, pc_source


def icp_p2l(pc_source, pc_ref, max_iterations=30, THRESHOLD=0.02, transformation_epsilon=1e-8):
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
    threshold = 10 * VOXEL_SIZE
    reg_p2l = o3d.pipelines.registration.registration_icp(
        pc_source, pc_ref, threshold, np.eye(4), 
        o3d.pipelines.registration.TransformationEstimationPointToPlane(), 
        icp_criteria)
    transformation_matrix = reg_p2l.transformation
    
    # Transform the source point cloud
    pc_source_aligned = pc_source.transform(transformation_matrix)
    
    return transformation_matrix, pc_source_aligned


