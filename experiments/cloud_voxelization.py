from fusionforce.models.terrain_encoder.voxelnet import VoxelNet
from fusionforce.utils import read_yaml
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')  # Use TkAgg backend for 3D plotting


def main():
    # Create a random point cloud with 1000 points
    # point_cloud = np.random.rand(3, 1000).astype(np.float32)
    point_cloud = np.load('/home/ruslan/Desktop/points.npy').T
    print(point_cloud.shape)

    # Convert the point cloud to a PyTorch tensor
    point_cloud_tensor = torch.tensor(point_cloud).unsqueeze(0)  # Add batch dimension

    # Initialize the VoxelNet model
    lss_cfg = read_yaml('../../fusionforce/fusionforce/config/lss_cfg.yaml')
    model = VoxelNet(grid_conf=lss_cfg['grid_conf'])
    model.from_pretrained('../../fusionforce/fusionforce/config/weights/voxelnet/val.pth')

    # Forward pass through the model
    output = model(point_cloud_tensor)

    # Print the output shape
    for key, value in output.items():
        print(f"{key}: {value.shape}")

    voxel_grid = model.lidar_net.voxelize(point_cloud_tensor)
    print(f"Voxel grid shape: {voxel_grid.shape}")  # Should be (B, 1, Z, X, Y) where Z, X, Y are the grid dimensions

    # Visualize the voxel grid in 3D
    voxel_grid_np = voxel_grid.squeeze().cpu().numpy()  # Remove batch and channel dimensions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = np.nonzero(voxel_grid_np)  # Get the indices of occupied voxels
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title('Voxel Grid Visualization')
    plt.show()


if __name__ == '__main__':
    main()