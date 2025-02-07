from monoforce.datasets import ROUGH, rough_seq_paths
from monoforce.transformations import position
from monoforce.cloudproc import filter_grid
from scipy.spatial import cKDTree
from tqdm import tqdm
import open3d as o3d
import numpy as np


def transform_cloud(cloud, Tr):
    assert cloud.ndim == 2
    assert cloud.shape[1] == 3  # (N, 3)
    cloud_tr = Tr[:3, :3] @ cloud.T + Tr[:3, 3:]
    return cloud_tr.T

def data_slicing():
    path = np.random.choice(rough_seq_paths)
    ds = ROUGH(path)
    ids = np.random.choice(range(len(ds)), 10)
    print(f"Selected sample ids: {ids}")
    ds_slice = ds[ids]
    print(f"Full dataset contains {len(ds.ids)} samples")
    print(f"Sliced dataset contains {len(ds_slice.ids)} samples")
    assert len(ds_slice) == len(ids)
    assert set(ds_slice.ids).issubset(set(ds.ids))
    ds_slice.get_global_cloud(vis=True, cached=False, save=False, step=1)
    # ds.get_global_cloud(vis=True, cached=False, save=False, step=10)


def create_global_cloud():
    from mayavi import mlab
    from monoforce.vis import draw_coord_frames
    """
    Create global heightmap cloud from the sequence of point clouds
    """
    dist_th = 0.1

    # paths = rough_seq_paths
    paths = [
        '/home/ruslan/data/datasets/ORU/radarize__2024-04-27-15-02-12_0',
    ]
    for path in paths:
        print('Processing sequence:', path)
        ds = ROUGH(path)

        # create global cloud
        global_cloud = None
        step = 1
        wps = np.array([ds.poses[50], ds.poses[140], ds.poses[280], ds.poses[370]])
        wps[:, :3, :3] = np.eye(3)
        for i in tqdm(range(0, 370, step)):
            try:
                cloud = ds.get_cloud(i, gravity_aligned=False)
                pose = ds.get_pose(i)
            except Exception as e:
                print(f"Error: {e}")
                continue

            points = position(cloud)
            # remove nans
            mask = np.all(np.isfinite(points), axis=1)
            points = points[mask]
            points = filter_grid(points, ds.dphys_cfg.grid_res, keep='first', log=False)
            points = transform_cloud(points, pose)
            if global_cloud is None:
                global_cloud = points
            else:
                tree = cKDTree(global_cloud)
                dists, idxs = tree.query(points, k=1)
                new_pts_mask = dists > dist_th
                new_points = points[new_pts_mask]
                global_cloud = np.vstack((global_cloud, new_points))

        # visualizations with mayavi
        mlab.figure(bgcolor=(1, 1, 1), size=(1600, 800))
        print(np.min(global_cloud[:, 2]), np.max(global_cloud[:, 2]))
        color = global_cloud[:, 2]
        mlab.points3d(global_cloud[:, 0], global_cloud[:, 1], global_cloud[:, 2], color, scale_factor=0.02, opacity=0.6, colormap='jet')
        mlab.plot3d(ds.poses[:370, 0, 3], ds.poses[:370, 1, 3], ds.poses[:370, 2, 3], color=(0, 0, 1), tube_radius=0.2)
        draw_coord_frames(wps, scale=15)
        mlab.show()


def show_map():
    map = o3d.io.read_point_cloud('/home/ruslan/data/datasets/ORU/radarize__2024-04-27-15-02-12_0/map/map.pcd')
    traj = o3d.io.read_point_cloud('/home/ruslan/data/datasets/ORU/radarize__2024-04-27-15-02-12_0/map/trajectory.pcd')

    # map = o3d.io.read_point_cloud('/home/ruslan/data/datasets/ORU/radarize__2024-05-24-13-21-28_0/map/map.pcd')
    # traj = o3d.io.read_point_cloud('/home/ruslan/data/datasets/ORU/radarize__2024-05-24-13-21-28_0/map/trajectory.pcd')

    # map = o3d.io.read_point_cloud('/home/ruslan/data/datasets/ORU/radarize__2024-02-07-10-47-13_0/map/map.pcd')
    # traj = o3d.io.read_point_cloud('/home/ruslan/data/datasets/ORU/radarize__2024-02-07-10-47-13_0/map/trajectory.pcd')

    # map = o3d.io.read_point_cloud('/home/ruslan/data/datasets/ORU/radarize__2023-08-16-11-02-33_0/map/map.pcd')
    # traj = o3d.io.read_point_cloud('/home/ruslan/data/datasets/ORU/radarize__2023-08-16-11-02-33_0/map/trajectory.pcd')

    traj.paint_uniform_color([0, 0, 0])
    o3d.visualization.draw_geometries([map, traj])

def main():
    # data_slicing()
    create_global_cloud()
    # show_map()


if __name__ == '__main__':
    main()
