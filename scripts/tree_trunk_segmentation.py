from monoforce.datasets.rough import ROUGH, rough_seq_paths
from monoforce.transformations import position
import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN


def load_point_cloud():
    path = rough_seq_paths[2]
    ds = ROUGH(path=path)
    sample_i = 55
    print(f'Sample index: {sample_i}')
    points = position(ds.get_cloud(sample_i, gravity_aligned=True))
    # remove nans from the point cloud
    points = points[~np.isnan(points).any(axis=1)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


# Downsample the point cloud
def downsample_point_cloud(pcd, voxel_size=0.1):
    return pcd.voxel_down_sample(voxel_size)


# Remove ground plane using RANSAC
def remove_ground_plane(pcd, distance_threshold=0.2):
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000)
    pcd_without_ground = pcd.select_by_index(inliers, invert=True)
    return pcd_without_ground


# DBSCAN clustering
def cluster_point_cloud(pcd, eps=0.5, min_samples=10):
    points = np.asarray(pcd.points)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    return labels


# Fit cylinder to cluster
def fit_cylinder_to_cluster(cluster_points):
    # Implement cylinder fitting using least squares or Open3D
    pass

# Main pipeline
def main():
    pcd = load_point_cloud()
    pcd_down = downsample_point_cloud(pcd)
    # o3d.visualization.draw_geometries([pcd_down])
    pcd_no_ground = remove_ground_plane(pcd_down)
    # o3d.visualization.draw_geometries([pcd_no_ground])
    labels = cluster_point_cloud(pcd_no_ground)

    # visualize clusters
    cluster_pcds = []
    for label in np.unique(labels):
        if label == -1:
            continue  # Noise
        cluster_points = np.asarray(pcd_no_ground.points)[labels == label]
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
        cluster_pcd.paint_uniform_color(list(np.random.rand(3)))
        cluster_pcds.append(cluster_pcd)
    o3d.visualization.draw_geometries(cluster_pcds)


if __name__ == '__main__':
    main()