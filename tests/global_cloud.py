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
    """
    Create global heightmap cloud from the sequence of point clouds
    """
    dist_th = 0.1

    for path in rough_seq_paths:
        print('Processing sequence:', path)
        ds = ROUGH(path)

        # create global cloud
        global_cloud = None
        poses = []
        for i in tqdm(range(0, len(ds), 10)):
            cloud = ds.get_cloud(i, gravity_aligned=False)
            pose = ds.get_pose(i)
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

            poses.append(pose)

        # visualize global cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(global_cloud)

        # create a coordinate frame for each pose
        pcd_poses = []
        for pose in poses:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
            frame.transform(pose)
            pcd_poses.append(frame)

        # save global cloud
        # o3d.io.write_point_cloud(os.path.join(path, 'map', 'map.pcd'), pcd)
        o3d.visualization.draw_geometries([pcd] + pcd_poses)


def main():
    data_slicing()
    create_global_cloud()


if __name__ == '__main__':
    main()
