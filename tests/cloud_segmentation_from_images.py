import os
from dataproc.imgproc import ego_to_cam, get_only_in_img_mask
from monoforce.datasets.coco import COCO_CATEGORIES
from monoforce.datasets.rough import ROUGH, rough_seq_paths, segment_vegetation, lower_green, upper_green
from monoforce.utils import normalize
from monoforce.transformations import position
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
import cv2
import matplotlib as mpl
mpl.use('TkAgg')


def segment_cloud():
    path = rough_seq_paths[0]
    ds = ROUGH(path=path)
    # ds.get_semantic_cloud(120, vis=True)

    void_id = 133
    coco_classes = [i['name'].replace('-merged', '').replace('-other', '') for i in COCO_CATEGORIES] + ['void']
    coco_colors = [(np.array(color['color']) / 255).tolist() for color in COCO_CATEGORIES] + [[0., 0., 0.]]
    # selected_classes = ['person', 'tree', 'building']
    selected_classes = np.copy(coco_classes)
    # selected_classes = ['person']

    # sample_i = np.random.choice(range(len(ds)))
    sample_i = 120

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
        rgb_seg = np.copy(rgb)
        for color_i, color in enumerate(coco_colors):
            if coco_classes[color_i] not in selected_classes:
                continue
            # print(f'Colorizing {coco_classes[color_i]} with {color}')
            seg_color_cam[seg_label_cam == color_i] = color
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


def vegetation_segmentation():
    path = rough_seq_paths[2]
    ds = ROUGH(path=path)

    # sample_i = np.random.choice(range(len(ds)))
    sample_i = 55  # 55, 342
    print(f'Sample index: {sample_i}')

    n_cams = len(ds.camera_names)
    plt.figure(figsize=(n_cams * 5, 5))
    plt.axis('off')
    for cam_i in range(n_cams):
        cam = ds.camera_names[cam_i]
        rgb, K = ds.get_image(sample_i, camera=cam)
        rgb = np.asarray(rgb)

        # segment vegetation
        mask = segment_vegetation(rgb, lower_green, upper_green)

        # visualize: 1st row - RGB image, 2nd row - vegetation mask
        plt.subplot(2, len(ds.camera_names), cam_i + 1)
        plt.imshow(rgb)
        plt.subplot(2, len(ds.camera_names), len(ds.camera_names) + cam_i + 1)
        plt.imshow(mask, cmap='gray')

    plt.figure(figsize=(10, 5))
    # visualize lower and upper bounds for green in HSV
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(np.array([[lower_green]], dtype=np.uint8), cv2.COLOR_HSV2RGB))
    plt.title('Lower bound')
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(np.array([[upper_green]], dtype=np.uint8), cv2.COLOR_HSV2RGB))
    plt.title('Upper bound')

    plt.show()


def vegetation_segmentation_cloud():
    path = rough_seq_paths[2]
    ds = ROUGH(path=path)

    sample_i = 55
    print(f'Sample index: {sample_i}')

    lidar_points = position(ds.get_cloud(sample_i, gravity_aligned=False))

    # segment vegetation points in a point cloud and colorize them
    points = []
    colors = []
    n_cams = len(ds.camera_names)
    for cam_i in range(n_cams):
        cam = ds.camera_names[cam_i]
        rgb, K = ds.get_image(sample_i, camera=cam)
        rgb = np.asarray(rgb)

        E = ds.calib['transformations'][f'T_base_link__{cam}']['data']
        E = np.asarray(E, dtype=np.float32).reshape((4, 4))

        lidar_points = torch.as_tensor(lidar_points)
        E = torch.as_tensor(E)
        K = torch.as_tensor(K)

        cam_points = ego_to_cam(lidar_points.T, E[:3, :3], E[:3, 3], K).T
        mask = get_only_in_img_mask(cam_points.T, rgb.shape[0], rgb.shape[1])
        cam_points = cam_points[mask]

        # segment vegetation
        uv = cam_points[:, :2].numpy().astype(int)
        veg_mask = segment_vegetation(rgb, lower_green, upper_green)
        veg_mask = veg_mask[uv[:, 1], uv[:, 0]]
        veg_color = np.zeros((len(veg_mask), 3))
        veg_color[veg_mask] = np.array([0, 255, 0]) / 255.

        points.append(lidar_points[mask].numpy())
        colors.append(veg_color)

    points = np.vstack(points)
    colors = np.vstack(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


def choose_veg_color():
    np.random.seed(42)

    # path = np.random.choice(rough_seq_paths)
    path = rough_seq_paths[2]
    print(f'Path: {path}')
    ds = ROUGH(path=path)

    sample_i = 55
    # sample_i = np.random.choice(range(len(ds)))
    print(f'Sample index: {sample_i}')
    cam = ds.camera_names[1]
    # cam = np.random.choice(ds.camera_names)
    print(f'Camera: {cam}')
    bgr, _ = ds.get_cached_resized_img(sample_i, camera=cam)
    bgr = np.asarray(bgr)[..., ::-1]

    def nothing(x):
        pass  # Placeholder function for trackbars

    # Create a window
    cv2.namedWindow("HSV Selector")

    # Create trackbars for adjusting thresholds
    cv2.createTrackbar("Low H", "HSV Selector", lower_green[0], 179, nothing)
    cv2.createTrackbar("High H", "HSV Selector", upper_green[0], 179, nothing)
    cv2.createTrackbar("Low S", "HSV Selector", lower_green[1], 255, nothing)
    cv2.createTrackbar("High S", "HSV Selector", upper_green[1], 255, nothing)
    cv2.createTrackbar("Low V", "HSV Selector", lower_green[2], 255, nothing)
    cv2.createTrackbar("High V", "HSV Selector", upper_green[2], 255, nothing)

    while True:
        # Get values from trackbars
        low_h = cv2.getTrackbarPos("Low H", "HSV Selector")
        high_h = cv2.getTrackbarPos("High H", "HSV Selector")
        low_s = cv2.getTrackbarPos("Low S", "HSV Selector")
        high_s = cv2.getTrackbarPos("High S", "HSV Selector")
        low_v = cv2.getTrackbarPos("Low V", "HSV Selector")
        high_v = cv2.getTrackbarPos("High V", "HSV Selector")

        # Create mask based on current trackbar positions
        lower_bound = np.array([low_h, low_s, low_v])
        upper_bound = np.array([high_h, high_s, high_v])
        mask = segment_vegetation(bgr, lower_bound, upper_bound)

        # Apply mask to the original image
        result = cv2.bitwise_and(bgr, bgr, mask=np.asarray(mask, dtype=np.uint8))

        # Show results
        cv2.imshow("Original", bgr)
        cv2.imshow("Masked", result)

        # # stack images and save
        # img = np.hstack((bgr, result))
        # os.makedirs('./gen/segmentation', exist_ok=True)
        # cv2.imwrite('./gen/segmentation/veg_color_selector.png', img)

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


def green_mask():
    from monoforce.datasets.coco import COCO_CLASSES

    ds = ROUGH(path=rough_seq_paths[2])
    sample_i = 55
    print(f'Sample index: {sample_i}')
    # points = position(ds.get_cloud(sample_i, gravity_aligned=False))
    soft_classes = ds.lss_cfg['soft_classes']
    rigid_classes = [c for c in COCO_CLASSES if c not in soft_classes]
    points, _ = ds.get_semantic_cloud(sample_i, classes=rigid_classes, vis=False)
    points = torch.as_tensor(points, dtype=torch.float32)

    cam_i = 1
    cam = ds.camera_names[cam_i]
    rgb, K = ds.get_image(sample_i, camera=cam)
    rgb = np.asarray(rgb)

    E = ds.calib['transformations'][f'T_base_link__{cam}']['data']
    E = np.asarray(E, dtype=np.float32).reshape((4, 4))

    E = torch.as_tensor(E)
    K = torch.as_tensor(K)

    img_plane_points = ego_to_cam(points.T, E[:3, :3], E[:3, 3], K).T
    cam_points_mask = get_only_in_img_mask(img_plane_points.T, rgb.shape[0], rgb.shape[1])

    veg_rgb_mask = segment_vegetation(rgb, lower_green, upper_green)

    # show image with opencv
    bgr = rgb[..., ::-1]
    segm = cv2.bitwise_and(bgr, bgr, mask=np.asarray(veg_rgb_mask, dtype=np.uint8))
    result = cv2.addWeighted(bgr, 0.7, segm, 0.3, 0)

    cv2.imshow('result', result)
    cv2.waitKey(0)

    # mask of the points that belong to vegetation
    uv = img_plane_points[cam_points_mask, :2].numpy().astype(int)
    veg_rgb_mask = veg_rgb_mask[uv[:, 1], uv[:, 0]]

    veg_points = points[cam_points_mask][veg_rgb_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.numpy())
    pcd.paint_uniform_color([0, 0, 1])

    veg_pcd = o3d.geometry.PointCloud()
    veg_pcd.points = o3d.utility.Vector3dVector(veg_points.numpy())
    veg_pcd.paint_uniform_color([0, 1, 0])
    o3d.visualization.draw_geometries([pcd, veg_pcd])


def main():
    # segment_cloud()
    # colorize_cloud()
    # vegetation_segmentation()
    # vegetation_segmentation_cloud()
    choose_veg_color()
    # green_mask()


if __name__ == '__main__':
    main()