from fusionforce.datasets.rough import ROUGH, rough_seq_paths
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def main():
    # path = np.random.choice(rough_seq_paths)
    path = rough_seq_paths[0]
    ds = ROUGH(path=path)
    i = np.random.randint(0, len(ds))
    cloud = ds.get_cloud(i)

    points = np.stack([cloud['x'], cloud['y'], cloud['z']], axis=1)
    intensity = cloud['intensity']

    # remove nans
    valid = np.logical_not(np.isnan(cloud['x']))
    points = points[valid]
    intensity = intensity[valid]

    # normalize intensity
    intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    colormap = plt.cm.jet
    colors = colormap(intensity)[:, :3]
    print(colors.shape)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()