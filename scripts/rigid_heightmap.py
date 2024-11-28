from monoforce.datasets import ROUGH, rough_seq_paths
from monoforce.dphys_config import DPhysConfig
from monoforce.utils import read_yaml, normalize
from monoforce.cloudproc import hm_to_cloud, position
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Qt5Agg')

dphys_cfg = DPhysConfig()
lss_cfg = read_yaml('../../monoforce/monoforce/config/lss_cfg.yaml')

def segmentation_test():
    path = rough_seq_paths[0]
    ds = ROUGH(path=path, dphys_cfg=dphys_cfg, lss_cfg=lss_cfg)
    for i in np.random.choice(range(len(ds)), 1):
    # for i in [120, 294]:
        print('Data index:', i)
        seg_points, seg_colors = ds.get_semantic_cloud(i, vis=False)
        seg_colors = normalize(seg_colors)
        traj_points = ds.get_footprint_traj_points(i)
        traj_colors = np.ones_like(traj_points) * [0, 0, 1]
        points = np.concatenate((seg_points, traj_points), axis=0)
        colors = np.concatenate((seg_colors, traj_colors), axis=0)

        poses = ds.get_traj(i)['poses']

        heightmap = ds.get_terrain_height_map(i, cached=False)
        # points = hm_to_cloud(heightmap[0], cfg=dphys_cfg, mask=heightmap[1])

        # plt.figure(figsize=(10, 5))
        # plt.subplot(1, 2, 1)
        # plt.imshow(heightmap[0].squeeze().T, cmap='jet', vmin=-1, vmax=1, origin='lower')
        # plt.subplot(1, 2, 2)
        # plt.imshow(heightmap[1].squeeze().T, cmap='gray', origin='lower')
        # plt.show()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # coordinate frame
        pose_frames = []
        for pose in poses:
            pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
            # transform coordinate frame
            pose_frame.transform(pose)
            pose_frames.append(pose_frame)

        o3d.visualization.draw_geometries([pcd] + pose_frames)


def rough_test():
    path = np.random.choice(rough_seq_paths)
    ds = ROUGH(path=path, dphys_cfg=dphys_cfg, lss_cfg=lss_cfg)

    sample_i = np.random.choice(range(len(ds)))
    print('Sample index:', sample_i)
    sample = ds[sample_i]
    (imgs, rots, trans, intrins, post_rots, post_trans,
     hm_geom, hm_terrain,
     control_ts, controls,
     pose0,
     traj_ts, Xs, Xds, Rs, Omegas) = sample

    terrain_points = hm_to_cloud(hm_terrain[0], dphys_cfg, hm_terrain[1])
    geom_points = hm_to_cloud(hm_geom[0], dphys_cfg, hm_geom[1])
    points = position(ds.get_cloud(sample_i))

    # visualize
    terrain_pcd = o3d.geometry.PointCloud()
    terrain_pcd.points = o3d.utility.Vector3dVector(terrain_points)
    terrain_pcd.paint_uniform_color([0, 0, 1])

    geom_pcd = o3d.geometry.PointCloud()
    geom_pcd.points = o3d.utility.Vector3dVector(geom_points)
    geom_pcd.paint_uniform_color([1, 0, 0])

    cloud_pcd = o3d.geometry.PointCloud()
    cloud_pcd.points = o3d.utility.Vector3dVector(points)
    cloud_pcd.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw_geometries([terrain_pcd])
    o3d.visualization.draw_geometries([terrain_pcd, geom_pcd])
    o3d.visualization.draw_geometries([terrain_pcd, geom_pcd, cloud_pcd])


def main():
    segmentation_test()
    rough_test()


if __name__ == '__main__':
    main()