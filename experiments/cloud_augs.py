from fusionforce.datasets.rough import ROUGH, rough_seq_paths
import open3d as o3d
import numpy as np


def get_column_mask(points : np.ndarray, d_max : float, prob=0.5) -> np.ndarray:
    """ Mask out a column of points in a 3D point cloud.
    :param points: (N, 3) array of points in the point cloud.
    :param d_max: float, maximum distance from the origin to the column center.
    :param prob: float, probability of removing a column.
    :return: boolean mask of shape (N,) where True indicates that the point is outside the column.
    """
    if np.random.rand() > prob:
        # Do not remove a column
        mask = np.ones(points.shape[0], dtype=bool)
        return mask

    # Randomly select a column center (x, y) within the bounds of d_max
    x, y = np.random.uniform(-d_max*0.8, d_max*0.8, (2,))
    dx, dy = np.random.uniform(d_max*0.4, d_max*0.6, (2,))
    # Random rotation angle
    phi = np.random.uniform(-np.pi, np.pi)
    # print(f'Column center: ({x:.2f}, {y:.2f}), dx: {dx:.2f}, dy: {dy:.2f}, phi: {phi:.2f}')
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                   [np.sin(phi), np.cos(phi), 0],
                   [0, 0, 1]])

    # Rotate the points
    points_rot = points @ Rz.T

    # Rotate the mask center (x, y) into the rotated frame
    xy_rot = np.array([x, y, 0]) @ Rz.T

    # Apply the mask in the rotated frame
    column_mask = (np.abs(points_rot[:, 0] - xy_rot[0]) > dx / 2) | \
                  (np.abs(points_rot[:, 1] - xy_rot[1]) > dy / 2)
    return column_mask


def main():
    path = rough_seq_paths[0]
    # sample_i = 0
    # path = np.random.choice(rough_seq_paths)
    ds = ROUGH(path=path)
    sample_i = np.random.randint(0, len(ds))
    cloud = ds.get_cloud(sample_i)
    points = np.stack([cloud['x'], cloud['y'], cloud['z']], axis=1)

    # remove nans
    valid = np.logical_not(np.isnan(cloud['x']))
    points = points[valid]

    d_max = 6.4
    h_max = 2.0
    heightmap_mask = (points[:, 0] > -d_max) & (points[:, 0] < d_max) & \
                     (points[:, 1] > -d_max) & (points[:, 1] < d_max) & \
                     (points[:, 2] > -h_max) & (points[:, 2] < h_max)
    points = points[heightmap_mask]

    # point cloud augmentation: remove a column of points
    column_mask = get_column_mask(points, d_max, prob=1.0)
    points = points[column_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    for _ in range(10):
        main()