import sys
sys.path.append('../src')
sys.path.append('../../monoforce/monoforce/src')
from dataproc.imgproc import ego_to_cam, get_only_in_img_mask
from monoforce.datasets.wildscenes.utils3d import METAINFO
from monoforce.datasets.rough import ROUGH, rough_seq_paths
from monoforce.utils import normalize
from monoforce.transformations import position
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
import matplotlib as mpl
import matplotlib.patches as mpatches
mpl.use('TkAgg')


def segment_cloud():
    path = rough_seq_paths[1]
    ds = ROUGH(path=path)
    sample_i = np.random.randint(0, len(ds))
    # classes = METAINFO['classes']
    # classes = [c for c in METAINFO['classes'] if c not in ds.lss_cfg['soft_classes']]
    # classes = ['grass']
    # ds.get_semantic_cloud(sample_i, classes=classes, vis=True)
    # return

    void_id = 255
    lidar_points = position(ds.get_cloud(sample_i, gravity_aligned=False))
    points = []
    colors = []
    n_cams = len(ds.camera_names)
    plt.figure(figsize=(n_cams * 4, 4))
    plt.axis('off')
    # no white space
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for cam_i in range(n_cams):
        cam = ds.camera_names[cam_i]
        rgb, K = ds.get_image(sample_i, camera=cam)
        rgb = np.asarray(rgb) / 255.

        seg_label = ds.get_seg_label(sample_i, camera=cam)
        seg_label = np.asarray(seg_label)
        # transform segmentation labels to colors
        seg_color = np.zeros(rgb.shape, dtype=np.float32)
        for cidx, c in zip(METAINFO['cidx'], METAINFO['palette']):
            seg_color[seg_label == cidx] = c
        seg_color /= 255.
        rgb_seg = np.copy(rgb)
        rgb_seg[seg_label != void_id] = normalize(seg_color + rgb)[seg_label != void_id]

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
        seg_color_points = seg_color[uv[:, 1], uv[:, 0]]

        points.append(lidar_points[mask].numpy())
        colors.append(seg_color_points)

        # visualize
        plt.subplot(2, len(ds.camera_names), cam_i + 1)
        plt.imshow(rgb)
        plt.scatter(cam_points[:, 0], cam_points[:, 1], s=1, c=lidar_points[mask, 2],
                    cmap='jet', alpha=0.2, vmin=-1, vmax=1)
        plt.axis('off')
        plt.subplot(2, len(ds.camera_names), len(ds.camera_names) + cam_i + 1)
        plt.imshow(rgb_seg)
        plt.axis('off')

    # display color pallet with class names
    plt.figure(figsize=(2, 10))
    patches = [mpatches.Patch(color=np.asarray(METAINFO['palette'][i])/255.,
                              label=METAINFO['classes'][i]) for i in range(len(METAINFO['cidx']))]
    plt.legend(handles=patches, loc='center', fontsize=10)
    plt.axis("off")
    plt.show()

    points = np.vstack(points)
    colors = np.vstack(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def main():
    segment_cloud()


if __name__ == '__main__':
    main()