import sys
sys.path.append('../src')
sys.path.append('../../monoforce/monoforce/src')
from dataproc.imgproc import ego_to_cam, get_only_in_img_mask
from monoforce.datasets.wildscenes.utils2d import METAINFO
from monoforce.datasets.rough import ROUGH, rough_seq_paths
from monoforce.utils import normalize
from monoforce.transformations import position
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
import matplotlib as mpl
mpl.use('TkAgg')


def segment_cloud():
    path = rough_seq_paths[1]
    ds = ROUGH(path=path)
    sample_i = 100 #np.random.randint(0, len(ds))
    # classes = METAINFO['classes']
    classes = [c for c in METAINFO['classes'] if c not in ds.lss_cfg['soft_classes']]
    print(classes)
    ds.get_semantic_cloud(sample_i, classes=classes, vis=True)
    # return

    void_id = 133
    label_2_rgb = {cidx: p for cidx, p in zip(METAINFO['cidx'], METAINFO['palette'])}

    lidar_points = position(ds.get_cloud(sample_i, gravity_aligned=False))
    points = []
    colors = []
    n_cams = len(ds.camera_names)
    plt.figure(figsize=(n_cams * 5, 5))
    plt.axis('off')
    for cam_i in range(n_cams):
        cam = ds.camera_names[cam_i]
        rgb, K = ds.get_image(sample_i, camera=cam)

        seg_label_cam = ds.get_seg_label(sample_i, camera=cam)
        rgb = np.asarray(rgb) / 255.
        seg_label_cam = np.asarray(seg_label_cam)
        # transform segmentation labels to colors
        seg_color_cam = np.zeros(rgb.shape, dtype=np.float32)
        for cidx, c in label_2_rgb.items():
            seg_color_cam[seg_label_cam == cidx] = c
        seg_color_cam /= 255.
        rgb_seg = np.copy(rgb)
        rgb_seg[seg_label_cam != void_id] = normalize(seg_color_cam + rgb)[seg_label_cam != void_id]

        E = ds.calib['transformations'][f'T_base_link__{cam}']['data']
        E = np.asarray(E, dtype=np.float32).reshape((4, 4))

        lidar_points = torch.as_tensor(lidar_points)
        E = torch.as_tensor(E)
        K = torch.as_tensor(K)

        cam_points = ego_to_cam(lidar_points.T, E[:3, :3], E[:3, 3], K).T
        mask = get_only_in_img_mask(cam_points.T, rgb.shape[0], rgb.shape[1])
        cam_points = cam_points[mask]
        # print('Points in image:', cam_points.shape)

        # colorize point cloud with values from segmentation image
        uv = cam_points[:, :2].numpy().astype(int)
        seg_colors = seg_color_cam[uv[:, 1], uv[:, 0]]

        points.append(lidar_points[mask].numpy())
        colors.append(seg_colors)

        # visualize
        plt.subplot(1, len(ds.camera_names), cam_i + 1)
        plt.imshow(rgb_seg)
        # plt.scatter(cam_points[:, 0], cam_points[:, 1], s=1, c=lidar_points[mask, 2],
        #             cmap='jet', alpha=0.2, vmin=-1, vmax=1)

    # display color pallete with class names
    plt.figure()
    plt.axis('off')
    for i, c in enumerate(METAINFO['palette']):
        plt.subplot(len(METAINFO['palette']), 1, i + 1)
        plt.imshow(np.ones((100, 100, 3), dtype=np.float32) * c / 255.)
        plt.title(METAINFO['classes'][i])
        plt.axis('off')

    # plt.savefig('segmentation_demo.png')
    plt.show()

    points = np.vstack(points)
    colors = np.vstack(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def colorize_cloud():
    path = rough_seq_paths[2]
    ds = ROUGH(path=path)
    # ds.get_semantic_cloud(120, vis=True)

    sample_i = np.random.choice(range(len(ds)))
    # sample_i = 120
    print(f'Sample index: {sample_i}')

    lidar_points = position(ds.get_cloud(sample_i, gravity_aligned=False))
    points = []
    colors = []
    n_cams = len(ds.camera_names)
    plt.figure(figsize=(n_cams * 5, 5))
    plt.axis('off')
    for cam_i in range(n_cams):
        cam = ds.camera_names[cam_i]
        rgb, K = ds.get_image(sample_i, camera=cam)
        rgb = np.asarray(rgb) / 255.

        E = ds.calib['transformations'][f'T_base_link__{cam}']['data']
        E = np.asarray(E, dtype=np.float32).reshape((4, 4))

        lidar_points = torch.as_tensor(lidar_points)
        E = torch.as_tensor(E)
        K = torch.as_tensor(K)

        cam_points = ego_to_cam(lidar_points.T, E[:3, :3], E[:3, 3], K).T
        mask = get_only_in_img_mask(cam_points.T, rgb.shape[0], rgb.shape[1])
        cam_points = cam_points[mask]
        # print('Points in image:', cam_points.shape)

        # colorize point cloud with values from segmentation image
        uv = cam_points[:, :2].numpy().astype(int)
        seg_colors = rgb[uv[:, 1], uv[:, 0]]

        points.append(lidar_points[mask].numpy())
        colors.append(seg_colors)

        # visualize
        plt.subplot(1, len(ds.camera_names), cam_i + 1)
        plt.imshow(rgb)
        # plt.scatter(cam_points[:, 0], cam_points[:, 1], s=1, c=lidar_points[mask, 2],
        #             cmap='jet', alpha=0.2, vmin=-1, vmax=1)

    # plt.savefig('segmentation_demo.png')
    plt.show()

    points = np.vstack(points)
    colors = np.vstack(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def main():
    segment_cloud()
    # colorize_cloud()


if __name__ == '__main__':
    main()