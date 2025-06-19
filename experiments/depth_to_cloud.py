import os
import numpy as np
from PIL import Image
import yaml
import open3d as o3d


def get_colored_cloud(depth_mm: np.ndarray, K: np.ndarray, rgb=None) -> o3d.geometry.PointCloud:
    height, width = depth_mm.shape

    # Generate pixel coordinates
    vu = np.indices(depth_mm.shape).reshape((2, -1))  # shape (2, H*W)
    uv = vu[::-1, :]  # shape (2, H*W)
    coords = np.vstack([
        uv[0, :],  # u (x-coordinate)
        uv[1, :],  # v (y-coordinate)
        np.ones(height * width)
    ])  # shape (3, H*W)

    K_inv = np.linalg.inv(K)
    depth_flat = depth_mm.flatten()
    points = (K_inv @ coords) * depth_flat  # shape (3, H*W)
    point3d_coords = points.T / 1000.0  # convert to meters, shape (H*W, 3)

    # Filter out invalid points (depth == 0)
    valid = depth_flat > 0
    point3d_coords = point3d_coords[valid]

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point3d_coords)

    if rgb is not None:
        point3d_colors = rgb.reshape(height * width, 3)[valid] / 255.0
        pcd.colors = o3d.utility.Vector3dVector(point3d_colors)

    return pcd


def main():
    seq_path = '/media/ruslan/VRAS-DATA 4TB 2/datasets/ROUGH/helhest_2025_06_13-15_01_10'
    image_files = sorted(os.listdir(os.path.join(seq_path, 'images', 'left')))
    depth_files = sorted(os.listdir(os.path.join(seq_path, 'luxonis', 'depth')))
    clouds_files = sorted(os.listdir(os.path.join(seq_path, 'luxonis', 'clouds')))
    calibration_path = os.path.join(seq_path, 'calibration')
    # ind = np.random.randint(0, len(image_files))
    ind = 150

    img_path = os.path.join(seq_path, 'images', 'left', image_files[ind])
    image = Image.open(img_path).convert('RGB')
    # image.show()

    depth_path = os.path.join(seq_path, 'luxonis', 'depth', depth_files[ind])
    depth = Image.open(depth_path)
    # depth.show()

    # read camera intrinsics
    calib = yaml.safe_load(open(os.path.join(calibration_path, 'cameras', 'camera_right.yaml')))
    K = np.array(calib['camera_matrix']['data']).reshape(3, 3)

    points_path = os.path.join(seq_path, 'luxonis', 'clouds', clouds_files[ind])
    points = np.load(points_path)['points']
    points = points[~np.isnan(points).any(axis=1)]
    print(points.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    image = np.asarray(image)
    depth = np.asarray(depth)
    pcd_depth = get_colored_cloud(depth, K, rgb=image)
    # o3d.visualization.draw_geometries([pcd, pcd_depth])
    o3d.visualization.draw_geometries([pcd_depth])


if __name__ == '__main__':
    main()