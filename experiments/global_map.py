import numpy as np
import open3d as o3d
import os


def main():
    seq_path = '../data/ROUGH/marv_2025-03-19-15-22-35/'
    # clouds_path = os.path.join(seq_path, 'clouds')
    clouds_path = '../data/marv_2025-03-19-15-22-35/clouds'

    def get_poses():
        poses_data = np.loadtxt(os.path.join(seq_path, 'poses', 'lidar_poses.csv'), delimiter=',', skiprows=1)
        pose_stamps = poses_data[:, 0]
        poses = poses_data[:, 1:].reshape((-1, 3, 4))
        # add 4th row to poses
        poses = np.concatenate([poses, np.tile(np.array([[0, 0, 0, 1]]), (poses.shape[0], 1, 1))], axis=1)
        return pose_stamps, poses

    def get_cloud(sample_i):
        cloud_file = sorted(os.listdir(clouds_path))[sample_i]
        cloud_stamp = float(cloud_file[:-4].replace('_', '.'))
        cloud = np.load(os.path.join(clouds_path, cloud_file))['cloud']
        points = np.stack([cloud['x'], cloud['y'], cloud['z']], axis=1)
        mask = np.isnan(cloud['x'])
        points = points[~mask]
        return cloud_stamp, points

    pose_stamps, poses = get_poses()

    def get_pose(cloud_stamp):
        pose_i = np.argmin(np.abs(pose_stamps - cloud_stamp))
        pose_stamp = pose_stamps[pose_i]
        assert np.allclose(pose_stamp, pose_stamp, atol=0.1), f'Cloud stamp: {cloud_stamp}, Pose stamp: {pose_stamp}'
        pose = poses[pose_i]
        return pose

    pcds, pose_frames = [], []
    for sample_i in range(0, 50, 10):
        cloud_stamp, points = get_cloud(sample_i)
        pose = get_pose(cloud_stamp)

        pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

        points = points @ pose[:3, :3].T + pose[:3, 3:4].T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.transform(pose)

        pose_frame.transform(pose)

        pcds.append(pcd)
        pose_frames.append(pose_frame)

    o3d.visualization.draw_geometries(pcds)
    # o3d.visualization.draw_geometries(pose_frames)


def main_rough():
    from monoforce.datasets.rough import ROUGH, rough_seq_paths

    seq_path = '../data/ROUGH/marv_2025-03-19-14-47-44/'
    # seq_path = rough_seq_paths[0]
    ds = ROUGH(seq_path)
    print(len(ds))

    pcds = []
    for sample_i in range(300, 400, 10):
        cloud = ds.get_cloud(sample_i)
        points = np.stack([cloud['x'], cloud['y'], cloud['z']], axis=1)
        mask = np.isnan(cloud['x'])
        points = points[~mask]
        cloud_stamp = float(ds.ids[sample_i].replace('_', '.'))

        pose = ds.get_pose(sample_i)
        pose_id = np.argmin(np.abs(ds.poses_ts - cloud_stamp))
        pose_stamp = ds.poses_ts[pose_id]
        stamp_diff = abs(pose_stamp - cloud_stamp)
        if stamp_diff > 0.1:
            print(f'Cloud stamp: {cloud_stamp}, Pose stamp: {pose_stamp}, Diff: {stamp_diff}')
            continue

        points = points @ pose[:3, :3].T + pose[:3, 3:4].T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        pcds.append(pcd)

    o3d.visualization.draw_geometries(pcds)


def stamps_test():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from monoforce.utils import read_yaml
    mpl.use('TkAgg')


    seq_path = '../data/ROUGH/marv_2025-03-19-14-47-44/'
    clouds_path = os.path.join(seq_path, 'clouds')

    cloud_files = os.listdir(clouds_path)
    cloud_stamps = np.sort([float(f[:-4].replace('_', '.')) for f in cloud_files])

    poses_data = np.loadtxt(os.path.join(seq_path, 'poses', 'lidar_poses.csv'), delimiter=',', skiprows=1)
    pose_stamps = poses_data[:, 0]
    assert np.diff(pose_stamps).min() > 0, 'Pose stamps are not sorted'
    poses = poses_data[:, 1:].reshape((-1, 3, 4))
    poses = np.concatenate([poses, np.tile(np.array([[0, 0, 0, 1]]), (poses.shape[0], 1, 1))], axis=1)

    # calib = read_yaml(os.path.join(seq_path, 'calibration/transformations.yaml'))
    # T_robot_lidar = np.asarray(calib['T_base_link__os_sensor']['data'], dtype=float).reshape((4, 4))

    # ids = np.searchsorted(pose_stamps, cloud_stamps)
    # diffs = pose_stamps[ids] - cloud_stamps
    # print(np.max(diffs), np.min(diffs))
    # plt.plot(cloud_stamps, diffs, '.')
    # plt.grid()
    # plt.show()

    # plt.plot(cloud_stamps[::100], cloud_stamps[::100], '.', label='Cloud stamps')
    # plt.plot(pose_stamps[::100], pose_stamps[::100], '.', label='Pose stamps')
    # plt.grid()
    # plt.legend()
    # plt.show()

    # create a global cloud map
    cloud_map = []
    for cloud_file in sorted(cloud_files):
        cloud_stamp = float(cloud_file[:-4].replace('_', '.'))
        cloud = np.load(os.path.join(clouds_path, cloud_file))['cloud']
        points = np.stack([cloud['x'], cloud['y'], cloud['z']], axis=1)
        mask = np.isnan(cloud['x'])
        points = points[~mask]

        pose_id = np.argmin(np.abs(pose_stamps - cloud_stamp))
        pose_stamp = pose_stamps[pose_id]
        pose = poses[pose_id]
        stamp_diff = abs(pose_stamp - cloud_stamp)

        if stamp_diff > 0.1:
            continue
        print(f'Cloud stamp: {cloud_stamp}, Pose stamp: {pose_stamp}, Diff: {stamp_diff}')

        points = np.matmul(points, pose[:3, :3].T) + pose[:3, 3:4].T

        cloud_map.append(points)
        if len(cloud_map) == 10:
            break
    cloud_map = np.concatenate(cloud_map)
    print(cloud_map.shape)

    # plt.figure(figsize=(8, 8))
    # # plt.plot(cloud_map[:, 0], cloud_map[:, 1], '.', label='Cloud map')
    # plt.plot(poses[:, 0, 3], poses[:, 1, 3], '.-', label='Pose trajectory')
    # plt.grid()
    # plt.axis('equal')
    # plt.show()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud_map)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    # main()
    # main_rough()
    stamps_test()