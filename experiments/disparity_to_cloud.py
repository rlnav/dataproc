import os
import numpy as np
import cv2
import yaml
import open3d as o3d
from depth_to_cloud import get_colored_cloud


seq_path = '/media/ruslan/VRAS-DATA 4TB 2/datasets/ROUGH/helhest_2025_06_13-15_01_10'
image_files = sorted(os.listdir(os.path.join(seq_path, 'images', 'left')))
depth_files = sorted([f for f in os.listdir(os.path.join(seq_path, 'defom-stereo', 'disparity')) if f.endswith('.npy')])
points_files = sorted(os.listdir(os.path.join(seq_path, 'luxonis', 'clouds')))
calibration_path = os.path.join(seq_path, 'calibration')


def main():
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
    # cv2.destroyWindow("Disparity")

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
    valid_mask = disp > 2.0
    depth = np.zeros_like(disp)
    depth[valid_mask] = (cams_baseline * focal_length) / disp[valid_mask]

    image = np.asarray(image)
    depth = np.asarray(depth)
    pcd_disp = get_colored_cloud(depth * 1000., K, rgb=image)
    pcd_disp, _ = pcd_disp.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    points_path = os.path.join(seq_path, 'luxonis', 'clouds', points_files[ind])
    points = np.load(points_path)['points']
    # remove NaN points
    points = points[~np.isnan(points).any(axis=1)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.visualization.draw_geometries([pcd_disp, pcd])


if __name__ == '__main__':
    main()