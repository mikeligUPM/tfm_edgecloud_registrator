import numpy as np
from logger_config import logger

def get_config(id_dataset):
    logger.debug(f"ID DATASET: {id_dataset}")
    if id_dataset == 0: #3dmatch
        BACKBONE_INIT_VOXEL_SIZE = 0.025
        threshold = 0.01
    elif id_dataset == 1: #own data
        BACKBONE_INIT_VOXEL_SIZE = 0.00001 * 2.5
        threshold = 0.0025
        # threshold = 1e-05
    else: # use 3dmatch as default
        BACKBONE_INIT_VOXEL_SIZE = 0.025
        threshold = 0.01
    return BACKBONE_INIT_VOXEL_SIZE, threshold


def old_calculate_registration_metrics(pc_source, pc_ref, threshold, p2p_registration=False, final_result_print=False, i=1):
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
    true_positives = np.sum(inliers)
    total_correspondences = len(distances)  # Assuming the ground truth is that all points should be matched
    registration_recall = true_positives / total_correspondences
    print(f"### Registration Recall: {registration_recall}")

    print("#################################################")




def calculate_registration_metrics(pc_source, pc_ref, threshold, p2p_registration=False, final_result_print=False, i=1):
    """
    This function calculates the Root Mean Squared Error (RMSE), Inlier Ratio (IR), Recall Ratio (RR), 
    and False Match Ratio (FMR) between two point clouds represented by Open3d PointCloud objects.

    Args:
        source_pc: An open3d.cpu.pybind.geometry.PointCloud representing the source point cloud.
        ref_pc: An open3d.cpu.pybind.geometry.PointCloud representing the reference point cloud.

    Returns:
        A dictionary containing the following keys:
            rmse: The root mean squared error between the two point clouds.
            inlier_ratio: The ratio of points in the source cloud with a close correspondence in the reference cloud.
            recall_ratio: The ratio of points in the reference cloud that have a close correspondence in the source cloud.
            false_match_ratio: The ratio of points in the source cloud that do not have a close correspondence in the reference cloud.
    """
    if final_result_print:
        logger.info(f"\n########## REGISTERED FINAL POINT CLOUD METRICS ############")
    else:
        logger.info(f"\n############## ITERATION {i} METRICS ########################")
    # Convert point cloud coordinates to numpy arrays
    source_points = np.asarray(pc_source.points)
    ref_points = np.asarray(pc_ref.points)

    # Find the nearest neighbors for each point in the source point cloud
    nearest_neighbors = np.linalg.norm(source_points[:, np.newaxis] - ref_points, axis=2).argmin(axis=1)

    # Calculate distances between points and their nearest neighbors
    distances = np.linalg.norm(source_points - ref_points[nearest_neighbors], axis=1)

    # Define a threshold for considering a point a close correspondence
    # threshold = 0.1  # You can adjust this value based on your application

    # Count the number of inliers (points with distance below the threshold)
    inliers = np.count_nonzero(distances <= threshold)

    # Calculate RMSE
    rmse = np.sqrt(np.mean(distances**2))

    # Calculate Inlier Ratio (IR)
    inlier_ratio = inliers / len(source_points)

    # Calculate Recall Ratio (RR) - ratio of inliers in reference cloud
    recall_ratio = inliers / len(ref_points)

    # Calculate False Match Ratio (FMR) - ratio of outliers in source cloud
    false_match_ratio = 1 - inlier_ratio

    logger.info(
          f"### threshold: {threshold}"
          f"### RMSE: {rmse}"
          f"\n### Inlier Ratio: {inlier_ratio}"
          f"\n### Recall Ratio: {recall_ratio}"
          f"\n### False Match Ratio: {false_match_ratio}")
    # # Return the results as a dictionary
    # return {
    #     "rmse": rmse,
    #     "inlier_ratio": inlier_ratio,
    #     "recall_ratio": recall_ratio,
    #     "false_match_ratio": false_match_ratio
    # }
