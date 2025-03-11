import sys
sys.path.append('../src')
sys.path.append('../../monoforce/monoforce/src')
from dataproc.imgproc import ego_to_cam, get_only_in_img_mask
from monoforce.datasets.wildscenes import METAINFO as WILDSCENES_METAINFO
from monoforce.datasets.rough import ROUGH, rough_seq_paths
from monoforce.datasets.coco import COCO_CLASSES
from monoforce.utils import normalize, explore_data
from monoforce.transformations import position
from monoforce.imgproc import undistort_image
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
import matplotlib as mpl
import matplotlib.patches as mpatches
mpl.use('TkAgg')


def segment_cloud():
    path = rough_seq_paths[2]
    sample_i = 47
    void_id = 255

    ds = ROUGH(path=path)
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
        for cidx, c in zip(WILDSCENES_METAINFO['cidx'], WILDSCENES_METAINFO['palette']):
            seg_color[seg_label == cidx] = c
        seg_color /= 255.

        # undistort image and segmentation label
        D = np.asarray(ds.calib[cam]['distortion_coefficients']['data'])
        rgb, _ = undistort_image(rgb, K, D)
        seg_label, _ = undistort_image(seg_label, K, D)
        seg_color, K = undistort_image(seg_color, K, D)

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
    patches = [mpatches.Patch(color=np.asarray(WILDSCENES_METAINFO['palette'][i])/255.,
                              label=WILDSCENES_METAINFO['classes'][i]) for i in range(len(WILDSCENES_METAINFO['cidx']))]
    plt.legend(handles=patches, loc='center', fontsize=10)
    plt.axis("off")
    plt.show()

    points = np.vstack(points)
    colors = np.vstack(colors)
    pcd_rigid = o3d.geometry.PointCloud()
    pcd_rigid.points = o3d.utility.Vector3dVector(points)
    pcd_rigid.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_rigid])


def rigid_cloud():
    path = '/home/ruslan/data/datasets/ROUGH/marv_2024-10-31-15-35-05/'
    ds = ROUGH(path=path)
    sample_i = np.random.randint(0, len(ds))
    print('Sample:', sample_i)
    # explore_data(ds, sample_range=[sample_i])

    rigid_classes_coco = [c for c in COCO_CLASSES if c not in ['grass', 'snow', 'flower', 'tree']]
    rigid_classes_wildscenes = [c for c in WILDSCENES_METAINFO['classes'] if c not in ['tree-foliage', 'bush', 'grass', 'sky']]
    rigid_points_coco, _ = ds.get_semantic_cloud(sample_i, classes=rigid_classes_coco, layout='coco')
    rigid_points_wildscenes, _ = ds.get_semantic_cloud(sample_i, classes=rigid_classes_wildscenes, layout='wildscenes')
    rigid_points = np.concatenate((rigid_points_coco, rigid_points_wildscenes), axis=0)

    points = position(ds.get_cloud(sample_i))
    points = points[~np.isnan(points).any(axis=1)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd_rigid = o3d.geometry.PointCloud()
    pcd_rigid.points = o3d.utility.Vector3dVector(rigid_points)
    pcd_rigid.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([pcd, pcd_rigid])
    return


def main():
    # segment_cloud()
    rigid_cloud()


if __name__ == '__main__':
    main()