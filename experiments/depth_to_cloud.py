import os
import numpy as np
from PIL import Image
import yaml
import open3d as o3d


def get_colored_cloud(depth: np.ndarray, K: np.ndarray, image=None) -> o3d.geometry.PointCloud:
    # read color and depth images
    height, width = depth.shape

    # calculate matrix 'point3d_coords' where each row is (x,y,z) for each point
    coords = np.vstack([np.indices(depth.shape).reshape((depth.ndim, -1)), np.ones(width * height)])
    K_inv = np.linalg.inv(K)
    point3d_coords = np.transpose((K_inv @ coords) * depth.flatten())
    point3d_coords /= 1000. # convert to meters

    # convert to open3d format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point3d_coords)

    if image is not None:
        # calculate matrix 'point3d_colors' where each row is (r,g,b) for each point
        point3d_colors = image.reshape(width * height, 3) / 255
        pcd.colors = o3d.utility.Vector3dVector(point3d_colors)

    return pcd


def main():
    seq_path = '/media/ruslan/VRAS-DATA 4TB 2/datasets/ROUGH/helhest_2025_06_13-15_01_10'
    images = sorted(os.listdir(os.path.join(seq_path, 'images', 'left')))
    depths = sorted(os.listdir(os.path.join(seq_path, 'depth')))
    calibration_path = os.path.join(seq_path, 'calibration')

    img_path = os.path.join(seq_path, 'images', 'left', images[0])
    image = Image.open(img_path)
    # image.show()

    depth_path = os.path.join(seq_path, 'depth', depths[0])
    depth = Image.open(depth_path)
    # depth.show()

    # read camera intrinsics
    calib = yaml.safe_load(open(os.path.join(calibration_path, 'cameras', 'camera_left.yaml')))
    K = np.array(calib['camera_matrix']['data']).reshape(3, 3)

    image = np.asarray(image)
    depth = np.asarray(depth)
    pcd = get_colored_cloud(depth, K)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()