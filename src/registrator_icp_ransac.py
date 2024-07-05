import open3d as o3d
import numpy as np

VOXEL_SIZE = 0.025 # 2.5 cm, same as used in Geotransformer
i = 1

def calculate_registration_metrics(pc_source, pc_ref, threshold, p2p_registration=False, final_result_print=False, i=1):
    if final_result_print:
        print(f"\n########## REGISTERED FINAL POINT CLOUD METRICS ############")
    else:
        print(f"\n############## ITERATION {i} METRICS ########################")
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
    print("#################################################")


########################    ICP P2L    ###########################################
def icp_p2l_registration_ransac(pcd_list):
    # register point cloud
    registered_point_cloud_generator = icp_p2l_register_and_yield_point_clouds(pcd_list)
    print("P2L Generator created")
    # threshold = 10 * VOXEL_SIZE
    threshold = VOXEL_SIZE *0.4
    final_fused_point_cloud = o3d.geometry.PointCloud()

    for i, pcd in enumerate(registered_point_cloud_generator):
        final_fused_point_cloud += pcd  # Use the += operator to merge point clouds
        calculate_registration_metrics(pcd, pcd_list[0], threshold, i=i+1)

    calculate_registration_metrics(final_fused_point_cloud, pcd_list[0], threshold, final_result_print=True)
    return final_fused_point_cloud


def icp_p2l_register_and_yield_point_clouds(point_cloud_list):
        ref_pc = point_cloud_list[0]
        radius_normal = VOXEL_SIZE * 2
        ref_pc.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        yield ref_pc  # Yield the reference point cloud as is

        for i in range(1, len(point_cloud_list)):
            src_pc = point_cloud_list[i]
            radius_normal = VOXEL_SIZE * 2
            src_pc.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            # _, transformed_pc = icp_registration(src_pc, ref_pc)
            _, transformed_pc = icp_p2l(src_pc, ref_pc)

            yield transformed_pc  # Yield the transformed point cloud

def icp_p2l(pc_source, pc_ref):
    source_down, source_fpfh = preprocess_point_cloud(pc_source, VOXEL_SIZE)
    target_down, target_fpfh = preprocess_point_cloud(pc_ref, VOXEL_SIZE)

    result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh, VOXEL_SIZE)

    result_icp = refine_registration(pc_source, pc_ref, result_ransac, p2p=False)
    pc_source_aligned = pc_source.transform(result_icp.transformation)
    return pc_ref, pc_source_aligned


########################    ICP P2P     ###########################################
def icp_p2p_registration_ransac(pcd_list):
    # 1- Do the RANSAC global initialization
    source_down, source_fpfh = preprocess_point_cloud(pcd_list[1], VOXEL_SIZE)
    target_down, target_fpfh = preprocess_point_cloud(pcd_list[0], VOXEL_SIZE)

    result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh, VOXEL_SIZE)
    
    registered_point_cloud_generator = icp_p2p_register_and_yield_point_clouds(pcd_list, result_ransac)
    print("P2P Generator created")
    # threshold = 5 * VOXEL_SIZE
    threshold = VOXEL_SIZE * 0.4
    final_fused_point_cloud = o3d.geometry.PointCloud()

    # for i, pcd in enumerate(registered_point_cloud_generator):
    #     final_fused_point_cloud = o3d.geometry.PointCloud.concatenate([final_fused_point_cloud, pcd])

    for i, pcd in enumerate(registered_point_cloud_generator):
        final_fused_point_cloud += pcd  # Use the += operator to merge point clouds
        calculate_registration_metrics(pcd, pcd_list[0], threshold, i=i+1)

    calculate_registration_metrics(final_fused_point_cloud, pcd_list[0], threshold, final_result_print=True)
    return final_fused_point_cloud

def icp_p2p_register_and_yield_point_clouds(point_cloud_list, result_ransac):
    ref_pc = point_cloud_list[0]

    yield ref_pc  # Yield the reference point cloud as is

    for i in range(1, len(point_cloud_list)):
        src_pc = point_cloud_list[i]
        # _, transformed_pc = icp_registration(src_pc, ref_pc)
        _, transformed_pc = icp_p2p(src_pc, ref_pc, result_ransac)

        yield transformed_pc  # Yield the transformed point cloud

def icp_p2p(pc_source, pc_ref, result_ransac):
    result_icp = refine_registration(pc_source, pc_ref, result_ransac, p2p=True)
    pc_source_aligned = pc_source.transform(result_icp.transformation)
    return pc_ref, pc_source_aligned


########################    HELPER FUNS     ###########################################
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh):
    distance_threshold = VOXEL_SIZE * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % VOXEL_SIZE)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
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
    distance_threshold = VOXEL_SIZE * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    if p2p:
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    else:
        radius_normal = VOXEL_SIZE * 2
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, result_ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result



def preprocess_point_cloud(pcd, VOXEL_SIZE):
    # print(":: Downsample with a voxel size %.3f." % VOXEL_SIZE)
    pcd_down = pcd.voxel_down_sample(VOXEL_SIZE)

    radius_normal = VOXEL_SIZE * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = VOXEL_SIZE * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, VOXEL_SIZE):
    distance_threshold = VOXEL_SIZE * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % VOXEL_SIZE)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
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