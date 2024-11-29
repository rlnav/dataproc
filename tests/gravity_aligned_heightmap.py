#!/usr/bin/env python

import numpy as np
import open3d as o3d
import torch
import matplotlib.pyplot as plt
from monoforce.datasets import ROUGH, rough_seq_paths
from monoforce.transformations import position
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
    robot_pose = ds.get_initial_pose_on_heightmap(i)
    gravity_aligned_pose = np.eye(4)

    # crop the point cloud to the region of interest
    cfg = DPhysConfig()
    grid_res, d_max, h_max = cfg.grid_res, cfg.d_max, cfg.h_max
    mask = ((points[:, 0] > -d_max) & (points[:, 0] < d_max) &
            (points[:, 1] > -d_max) & (points[:, 1] < d_max) &
            (points[:, 2] > -h_max) & (points[:, 2] < h_max) )
    points = points[mask]

    # visualize point cloud and map pose
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    cloud_pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    cloud_pose_frame.transform(robot_pose)

    gravity_aligned_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2)
    gravity_aligned_frame.transform(gravity_aligned_pose)

    o3d.visualization.draw_geometries([pcd, gravity_aligned_frame, cloud_pose_frame])


def heightmap():
    from time import time

    ds = ROUGH(path=rough_seq_paths[0])

    # crop the point cloud to the region of interest
    cfg = DPhysConfig()
    grid_res, d_max, h_max = cfg.grid_res, cfg.d_max, cfg.h_max

    for _ in range(1):
        i = 294
        # i = np.random.choice(len(ds))
        points = torch.from_numpy(position(ds.get_cloud(i)))  # in robot-centric frame: base_link

        t0 = time()
        hm = estimate_heightmap(points=points, grid_res=grid_res, d_max=d_max, h_max=h_max, r_min=1)
        print(f'Estimating heightmap took: {time() - t0} [sec]')
        heightmap, measured_mask = hm[0], hm[1]

        t1 = time()
        # hm0 = ds.get_terrain_height_map(i, cached=False)
        hm0 = ds.get_geom_height_map(i, cached=False)
        print(f'Getting terrain heightmap took: {time() - t1} [sec]')
        heightmap0 = hm0[0]

    # Resulting heightmap
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(heightmap.numpy(), cmap='jet', origin='lower', vmin=-h_max, vmax=h_max)

    plt.subplot(132)
    plt.imshow(heightmap0.numpy(), cmap='jet', origin='lower', vmin=-h_max, vmax=h_max)

    plt.subplot(133)
    plt.imshow(measured_mask, cmap='Greys', origin='lower')

    plt.colorbar()

    plt.show()


def main():
    points_alignment()
    heightmap()


if __name__ == '__main__':
    main()
