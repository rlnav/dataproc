import numpy as np
import open3d as o3d
import os


def main():
    from monoforce.datasets.rough import rough_seq_paths
    from monoforce.utils import read_yaml
    from tqdm import tqdm

    seq_path = rough_seq_paths[0]
    # seq_path = np.random.choice(rough_seq_paths)
    clouds_path = os.path.join(seq_path, 'clouds')
    cloud_files = sorted(os.listdir(clouds_path))
    poses_path = os.path.join(seq_path, 'poses', 'lidar_poses.csv')
    calib_path = os.path.join(seq_path, 'calibration')
    calib = read_yaml(os.path.join(calib_path, 'transformations.yaml'))
    T_robot_lidar = np.asarray(calib['T_base_link__os_sensor']['data'], dtype=float).reshape((4, 4))
    T_lidar_robot = np.linalg.inv(T_robot_lidar)

    def get_poses():
        poses_data = np.loadtxt(poses_path, delimiter=',', skiprows=1)
        pose_stamps = poses_data[:, 0]
        poses = poses_data[:, 1:].reshape((-1, 3, 4))
        # add 4th row to poses
        poses = np.concatenate([poses, np.tile(np.array([[0, 0, 0, 1]]), (poses.shape[0], 1, 1))], axis=1)
        return pose_stamps, poses

    def get_cloud(sample_i):
        cloud_file = cloud_files[sample_i]
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
        t_diff = pose_stamp - cloud_stamp
        # print(f'Cloud stamp: {cloud_stamp}, Pose stamp: {pose_stamp}, Diff: {t_diff}')
        pose = poses[pose_i]
        return pose, t_diff

    pcds, pose_frames = [], []
    for sample_i in tqdm(range(0, len(cloud_files), 1)):
        cloud_stamp, points0 = get_cloud(sample_i)
        pose_lidar, t_diff = get_pose(cloud_stamp)
        if abs(t_diff) > 0.020:
            continue
        print(t_diff)
        pose_robot = pose_lidar @ T_lidar_robot

        pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        pose_frame.transform(pose_robot)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points0)
        pcd.transform(pose_lidar)

        pcds.append(pcd)
        pose_frames.append(pose_frame)

    o3d.visualization.draw_geometries(pcds + pose_frames)


if __name__ == '__main__':
    main()