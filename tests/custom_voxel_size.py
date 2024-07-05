

def get_voxel_size_from_bbox_percentage(pcd, percentage):
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox_size = bbox.get_max_bound() - bbox.get_min_bound()

    # Calculate voxel size based on percentage of bounding box size
    voxel_size = max(bbox_size) * percentage / 100.0
    
    return voxel_size


def get_voxel_size_from_bbox_fraction(pcd, fraction=0.006):
    def calculate_bounding_box(pcd):
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox_min = bbox.min_bound
        bbox_max = bbox.max_bound
        return bbox_min, bbox_max

    # fraction=0.002 --> 219k points
    bbox_min, bbox_max = calculate_bounding_box(pcd)
    bbox_dims = bbox_max - bbox_min
    max_dim = max(bbox_dims)
    return max_dim * fraction


