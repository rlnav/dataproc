import os
import cv2
import numpy as np
import torch
import yaml
from copy import copy


CLASS_LABEL_MAP = {
    "classes": (
        "unlabelled",
        "bush",
        "dirt",
        "fence",
        "grass",
        "gravel",
        "log",
        "mud",
        "other-object",
        "other-terrain",
        "rock",
        "sky",
        "structure",
        "tree-foliage",
        "tree-trunk",
        "water",
    ),
    "labels": [
        255,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14
    ]
}


class TraversabilityDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.cloud_path = os.path.join(data_dir, 'clouds')
        self.poses_path = os.path.join(data_dir, 'poses', 'lidar_poses.csv')
        self.calib_path = os.path.join(data_dir, 'calibration')
        self.calib = self.load_calib(calib_path=self.calib_path)
        self.ids = self.get_ids()
        self.poses_ts, self.poses = self.get_all_poses()
        self.raw_img_size = (1920, 1200)

    @staticmethod
    def load_calib(calib_path):
        calib = {}
        # read camera calibration
        cams_path = os.path.join(calib_path, 'cameras')
        for file in os.listdir(cams_path):
            if file.endswith('.yaml'):
                with open(os.path.join(cams_path, file), 'r') as f:
                    cam_info = yaml.load(f, Loader=yaml.FullLoader)
                    calib[file.replace('.yaml', '')] = cam_info
                f.close()
        # read cameras-lidar transformations
        trans_path = os.path.join(calib_path, 'transformations.yaml')
        with open(trans_path, 'r') as f:
            transforms = yaml.load(f, Loader=yaml.FullLoader)
        f.close()
        calib['transformations'] = transforms
        T = np.asarray(calib['transformations']['T_base_link__base_footprint']['data'],
                       dtype=np.float32).reshape((4, 4))
        calib['clearance'] = np.abs(T[2, 3])
        return calib

    def get_ids(self):
        ids = [f[:-4] for f in os.listdir(self.cloud_path)]
        ids = sorted(ids)
        return ids

    def ind_to_stamp(self, i):
        ind = self.ids[i]
        stamp = float(ind.replace('_', '.'))
        return stamp

    @staticmethod
    def pose2mat(pose):
        T = np.eye(4)
        T[:3, :4] = pose.reshape((3, 4))
        return T

    @staticmethod
    def get_only_in_img_mask(points, H, W):
        """pts should be N x 3
        """
        return (points[:, 2] > 0) & \
               (points[:, 0] > 1) & (points[:, 0] < W - 1) & \
               (points[:, 1] > 1) & (points[:, 1] < H - 1)

    def get_all_poses(self):
        data = np.loadtxt(self.poses_path, delimiter=',', skiprows=1)
        assert len(data) > 0, f'No poses found in {self.poses_path}'
        stamps, Ts = data[:, 0], data[:, 1:13]
        lidar_poses = np.asarray([self.pose2mat(pose) for pose in Ts], dtype=np.float32)
        # poses of the robot in the map frame
        Tr_robot_lidar = self.calib['transformations']['T_base_link__os_sensor']['data']
        Tr_robot_lidar = np.asarray(Tr_robot_lidar, dtype=np.float32).reshape((4, 4))
        Tr_lidar_robot = np.linalg.inv(Tr_robot_lidar)
        poses = lidar_poses @ Tr_lidar_robot
        return stamps, poses

    def get_traj(self, i, T_horizon=10.0):
        # n_frames equals to the number of future poses (trajectory length)
        dt = 0.1  # lidar frequency is 10 Hz

        # get trajectory as sequence of `n_frames` future poses
        all_poses = copy(self.poses)
        all_ts = copy(self.poses_ts)
        time_left = self.ind_to_stamp(i)
        il = np.argmin(np.abs(self.poses_ts - time_left))
        ir = np.argmin(np.abs(all_ts - (self.poses_ts[il] + T_horizon)))
        ir = min(max(ir, il + 1), len(all_ts))
        poses = all_poses[il:ir]
        stamps = np.asarray(all_ts[il:ir])

        # transform poses to the same coordinate frame as the height map
        poses = np.linalg.inv(poses[0]) @ poses
        stamps = stamps - stamps[0]

        # limit stamps and poses to the horizon
        stamps = stamps[stamps <= T_horizon]
        poses = poses[:len(stamps)]

        # make sure the trajectory has the fixed length
        n_frames = int(np.ceil(T_horizon / dt))
        if len(poses) < n_frames:
            # repeat the last pose to fill the trajectory
            poses = np.concatenate([poses, np.tile(poses[-1:], (n_frames - len(poses), 1, 1))], axis=0)
            stamps = np.concatenate([stamps, stamps[-1] + np.arange(1, n_frames - len(stamps) + 1) * dt], axis=0)
            assert len(poses) == n_frames, f'Poses and stamps have different lengths {len(poses)} != {n_frames}'
        # truncate the trajectory
        poses = poses[:n_frames]
        stamps = stamps[:n_frames]

        traj = {'stamps': stamps, 'poses': poses}
        return traj

    def get_footprint_traj_points(self, i, robot_size=(0.7, 1.0), grid_res=0.1):
        # robot footprint points grid
        width, length = robot_size
        x = np.arange(-length / 2, length / 2, grid_res)
        y = np.arange(-width / 2, width / 2, grid_res)
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)
        footprint0 = np.stack([x, y, z], axis=-1).reshape((-1, 3))
        footprint0 = np.asarray(footprint0, dtype=np.float32)

        Tr_base_link__base_footprint = np.asarray(self.calib['transformations']['T_base_link__base_footprint']['data'],
                                                  dtype=np.float32).reshape((4, 4))
        traj = self.get_traj(i)
        poses = traj['poses']
        poses_footprint = poses
        poses_footprint[:, 2, 3] -= abs(Tr_base_link__base_footprint[2, 3])  # subtract robot's clearance

        trajectory_points = []
        for Tr in poses_footprint:
            footprint = (Tr[:3, :3] @ footprint0.T + Tr[:3, 3:]).T
            trajectory_points.append(footprint)
        trajectory_points = np.concatenate(trajectory_points, axis=0)
        return trajectory_points

    def get_image(self, i, camera='camera_front'):
        ind = self.ids[i]
        img_path = os.path.join(self.data_dir, 'images', '%s_%s.png' % (ind, camera))
        assert os.path.exists(img_path), f'Image path {img_path} does not exist'
        img = cv2.imread(img_path)
        return img

    def get_seg_label(self, i, camera='camera_front'):
        id = self.ids[i]
        seg_path = os.path.join(self.data_dir, 'images/wildscenes_seg/seg/', '%s_%s.png' % (id, camera))
        assert os.path.exists(seg_path), f'Image path {seg_path} does not exist'
        seg = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        seg = cv2.resize(seg, self.raw_img_size, interpolation=cv2.INTER_NEAREST)
        return seg

    def get_seg_vis(self, i, camera='camera_front'):
        id = self.ids[i]
        seg_path = os.path.join(self.data_dir, 'images/wildscenes_seg/vis/', '%s_%s.png' % (id, camera))
        assert os.path.exists(seg_path), f'Image path {seg_path} does not exist'
        seg = cv2.imread(seg_path)
        seg = cv2.resize(seg, self.raw_img_size, interpolation=cv2.INTER_NEAREST)
        return seg

    def get_traversability_label(self, i, camera='camera_front'):
        # mask of semantic labels that are traversable
        label = self.get_seg_label(i=i, camera=camera)
        label = np.asarray(label)
        VOID = CLASS_LABEL_MAP['labels'][CLASS_LABEL_MAP['classes'].index('unlabelled')]
        # init mask with unknown labels
        traversable_mask = VOID * np.ones_like(label)
        # set untraversable labels (obstacles)
        obstacle_classes = ["tree-trunk", "rock"]
        obstacle_labels = [CLASS_LABEL_MAP['labels'][CLASS_LABEL_MAP['classes'].index(c)] for c in obstacle_classes]
        for l in obstacle_labels:
            traversable_mask[label == l] = 0
        # set traversable labels
        K = np.asarray(self.calib['camera_front']['camera_matrix']['data'], dtype=np.float32).reshape((3, 3))
        Tr = np.asarray(self.calib['transformations'][f'T_base_link__{camera}']['data'],
                        dtype=np.float32).reshape((4, 4))
        points = self.get_footprint_traj_points(i=i, grid_res=0.1)
        # transform points to the camera frame
        points = (Tr[:3, :3].T @ (points.T - Tr[:3, 3:4])).T
        # project points to the image plane
        uv = (K @ points.T).T
        uv[:, :2] /= uv[:, 2:3]
        uv = uv.astype(np.int32)
        # filter points that are in the image
        H, W = self.raw_img_size[1], self.raw_img_size[0]
        mask = self.get_only_in_img_mask(uv, H, W)
        uv = uv[mask]
        # set traversable labels from robot's footprint path
        traversable_mask[uv[:, 1], uv[:, 0]] = 1
        return traversable_mask

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = self.get_image(i=i)
        trav_mask = self.get_traversability_label(i=i)
        return img, trav_mask


def main():
    from tqdm import tqdm

    data_dir = '/media/ruslan/VRAS-DATA 4TB 2/datasets/ROUGH-v1/marv_2024-10-31-15-56-33/'
    ds = TraversabilityDataset(data_dir=data_dir)

    i = np.random.randint(0, len(ds))
    # i = 717
    img, trav_mask = ds[i]

    # convert trav_mask to RGB image
    colormap = {
        255: (0, 0, 0),
        0: (0, 0, 255),
        1: (0, 255, 0)
    }
    trav_img = np.stack(np.vectorize(colormap.get)(trav_mask), axis=2).astype(np.uint8)
    print(trav_img.shape)

    img = cv2.resize(img, (819, 512), interpolation=cv2.INTER_NEAREST)
    trav_img = cv2.resize(trav_img, (819, 512), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('img', img)
    cv2.imshow('trav_img', trav_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()