#!/usr/bin/env python

import numpy as np
import open3d as o3d
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from monoforce.datasets import ROUGH, rough_seq_paths
from monoforce.transformations import position, transform_cloud
from monoforce.dphys_config import DPhysConfig
from monoforce.cloudproc import estimate_heightmap
import matplotlib as mpl
mpl.use('TkAgg')


def points_alignment():
    ds = ROUGH(path=rough_seq_paths[0])
    # print(f'Number of samples: {len(ds)}')

    # i = np.random.choice(len(ds))
    # i = 120
    i = 294
    print(f'Sample index: {i}')
    points = position(ds.get_cloud(i))  # in robot-centric frame: base_link
    cloud_pose = np.eye(4)

    map_pose = ds.get_pose(i)
    roll, pitch, yaw = Rotation.from_matrix(map_pose[:3, :3]).as_euler('xyz', degrees=True)
    print(f"Roll: {roll} [deg], \nPitch: {pitch} [deg], \nYaw: {yaw} [deg]")
    gravity_aligned_pose = np.eye(4)
    gravity_aligned_pose[:3, :3] = Rotation.from_euler('xyz', [-roll, -pitch, 0], degrees=True).as_matrix()

    # move to gravity-aligned frame
    points = transform_cloud(points, np.linalg.inv(gravity_aligned_pose))
    cloud_pose = cloud_pose @ np.linalg.inv(gravity_aligned_pose)
    gravity_aligned_pose = gravity_aligned_pose @ np.linalg.inv(gravity_aligned_pose)

    # crop the point cloud to the region of interest
    cfg = DPhysConfig()
    grid_res, d_max, h_max = cfg.grid_res, cfg.d_max, cfg.h_max_above_ground
    mask = ((points[:, 0] > -d_max) & (points[:, 0] < d_max) &
            (points[:, 1] > -d_max) & (points[:, 1] < d_max) &
            (points[:, 2] > -h_max) & (points[:, 2] < h_max) )
    points = points[mask]

    # visualize point cloud and map pose
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    cloud_pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    cloud_pose_frame.transform(cloud_pose)

    gravity_aligned_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
    gravity_aligned_frame.transform(gravity_aligned_pose)

    o3d.visualization.draw_geometries([pcd, gravity_aligned_frame, cloud_pose_frame])


def heightmap():
    from time import time

    ds = ROUGH(path=rough_seq_paths[0])

    # crop the point cloud to the region of interest
    cfg = DPhysConfig()
    grid_res, d_max, h_max = cfg.grid_res, cfg.d_max, cfg.h_max_above_ground

    for _ in range(5):
        # i = 294
        i = np.random.choice(len(ds))
        points = torch.from_numpy(position(ds.get_cloud(i)))  # in robot-centric frame: base_link

        t0 = time()
        hm = estimate_heightmap(points=points, grid_res=grid_res, d_max=d_max, h_max=h_max, r_min=1)
        print(f'Estimating heightmap took: {time() - t0} [sec]')
        heightmap, measured_mask = hm[0], hm[1]

        t1 = time()
        hm0 = ds.get_terrain_height_map(i)
        print(f'Getting terrain heightmap took: {time() - t1} [sec]')
        heightmap0 = hm0[0]

    # Resulting heightmap
    plt.figure()
    plt.subplot(131)
    plt.imshow(heightmap.numpy(), cmap='jet', origin='lower', vmin=-h_max, vmax=h_max)

    plt.subplot(132)
    plt.imshow(heightmap0.numpy(), cmap='jet', origin='lower', vmin=-h_max, vmax=h_max)

    plt.subplot(133)
    plt.imshow(measured_mask, cmap='grey', origin='lower')

    plt.colorbar()

    plt.show()


def main():
    # points_alignment()
    heightmap()


if __name__ == '__main__':
    main()
