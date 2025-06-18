import os
import numpy as np
import cv2
import yaml
import open3d as o3d


def get_colored_cloud(depth: np.ndarray, K: np.ndarray, rgb=None) -> o3d.geometry.PointCloud:
    # read color and depth images
    height, width = depth.shape

    # calculate matrix 'point3d_coords' where each row is (x,y,z) for each point
    coords = np.vstack([np.indices(depth.shape).reshape((depth.ndim, -1)), np.ones(width * height)])
    K_inv = np.linalg.inv(K)
    point3d_coords = np.transpose((K_inv @ coords) * depth.flatten())

    # convert to open3d format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point3d_coords)

    if rgb is not None:
        # calculate matrix 'point3d_colors' where each row is (r,g,b) for each point
        point3d_colors = rgb.reshape(width * height, 3) / 255
        pcd.colors = o3d.utility.Vector3dVector(point3d_colors)

    return pcd


def main():
    seq_path = '/media/ruslan/VRAS-DATA 4TB 2/datasets/ROUGH/helhest_2025_06_13-15_01_10'
    image_files = sorted(os.listdir(os.path.join(seq_path, 'images', 'left')))
    depth_files = sorted([f for f in os.listdir(os.path.join(seq_path, 'defom-stereo', 'disparity')) if f.endswith('.npy')])
    calibration_path = os.path.join(seq_path, 'calibration')
    # ind = np.random.randint(0, len(image_files))
    ind = 150

    img_path = os.path.join(seq_path, 'images', 'left', image_files[ind])
    image = cv2.imread(img_path)

    disp_path = os.path.join(seq_path, 'defom-stereo', 'disparity', depth_files[ind])
    disp = np.load(disp_path)
    # # apply colormap to disparity
    # disp_vis = cv2.convertScaleAbs(disp, alpha=255/disp.max())
    # disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
    # cv2.imshow("Disparity", disp_vis)
    # cv2.waitKey(0)

    # read camera intrinsics
    calib_intr = yaml.safe_load(open(os.path.join(calibration_path, 'cameras', 'camera_left.yaml')))
    K = np.array(calib_intr['camera_matrix']['data']).reshape(3, 3)
    focal_length = K[0, 0]  # assuming fx is the first element in the camera matrix

    calib_extr = yaml.safe_load(open(os.path.join(calibration_path, 'transformations.yaml')))
    # Tr_camera_left__robot = np.array(calib_extr['Tr_camera_left__robot']['data'], dtype=float).reshape(4, 4)
    # Tr_camera_right__robot = np.array(calib_extr['Tr_camera_right__robot']['data'], dtype=float).reshape(4, 4)
    # cams_baseline = np.linalg.norm(Tr_camera_left__robot[:3, 3] - Tr_camera_right__robot[:3, 3])  # [m]
    cams_baseline = 0.15

    # depth = (cams_baseline * focal_length) / (disp + 1e-6)  # [m] * [pixels] / [pixels] = [m]
    valid_mask = disp > 1.0
    depth = np.zeros_like(disp)
    depth[valid_mask] = (cams_baseline * focal_length) / disp[valid_mask]
    depth[depth > 10.0] = 0  # optional
    print(depth)

    image = np.asarray(image)
    depth = np.asarray(depth)
    pcd = get_colored_cloud(depth, K, rgb=image)
    # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()