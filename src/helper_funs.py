import numpy as np


def get_config(id_dataset):
    if id_dataset == 0: #3dmatch
        BACKBONE_INIT_VOXEL_SIZE = 0.025
        threshold = 0.01
    elif id_dataset == 1: #own data
        BACKBONE_INIT_VOXEL_SIZE = 0.0001 * 2.5
        threshold = 0.0025
    else: # use 3dmatch as default
        BACKBONE_INIT_VOXEL_SIZE = 0.025
        threshold = 0.01
    return BACKBONE_INIT_VOXEL_SIZE, threshold


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
    true_positives = np.sum(inliers)
    total_correspondences = len(distances)  # Assuming the ground truth is that all points should be matched
    registration_recall = true_positives / total_correspondences
    print(f"### Registration Recall: {registration_recall}")

    print("#################################################")
