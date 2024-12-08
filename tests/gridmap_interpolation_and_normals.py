from monoforce.models.dphysics import DPhysics, interpolate_grid
from monoforce.dphys_config import DPhysConfig
from monoforce.datasets.rough import ROUGH, rough_seq_paths
from mayavi import mlab
import torch


def compute_heightmap_gradients(z_grid: torch.Tensor, grid_res: float) -> torch.Tensor:
    """
    Computes the gradients of a heightmap along the x and y axes using torch.gradient.

    **Note**: The input grid is assumed to use torch's "xy" indexing convention, i.e., the first dimension is the y-axis and the second dimension is the x-axis.

    Parameters:
    - z_grid: Heightmap tensor of shape (B, H, W).
    - grid_res: Resolution of the grid in meters.
    """
    B, H, W = z_grid.shape
    z_grid = z_grid.squeeze()
    gradients = torch.vstack(torch.gradient(z_grid, spacing=grid_res, dim=(1, 0), edge_order=2))
    return gradients.reshape(B, 2, H, W)


def normalized(x, eps=1e-6):
    """
    Normalizes the input tensor.

    Parameters:
    - x: Input tensor.
    - eps: Small value to avoid division by zero.

    Returns:
    - Normalized tensor.
    """
    norm = torch.norm(x, dim=-1, keepdim=True)
    return x / torch.clamp(norm, min=eps)

# @torch.compile
def surface_normals(z_grid_grads: torch.Tensor, query: torch.Tensor, max_coord: float) -> torch.Tensor:
    """
    Computes the surface normals and tangents at the queried coordinates.

    Parameters:
    - z_grid_grads: torch.Tensor of gradients of the heightmap along the x and y axes (3D array), (B, 2, H, W).
    - query: Tensor of desired point coordinates for interpolation (3D array), (B, N, 2).

    Returns:
    - Surface normals at the queried coordinates.
    """
    norm_query = query / max_coord  # Normalize to [-1, 1]
    # Query coordinates of shape (B, N, 1, 2)
    B, N = query.shape[:2]
    grid_coords = norm_query.unsqueeze(2)
    # Interpolate the grid values into shape (B, 2, N, 1)
    grad_query = torch.nn.functional.grid_sample(z_grid_grads, grid_coords, align_corners=True, mode="bilinear", padding_mode="border")
    grad_query = grad_query.squeeze(-1).permute(0, 2, 1)  # (B, N, 2)
    # Compute the surface normals
    n = torch.dstack([-grad_query, torch.ones((B, N, 1))])  # n = [-dz/dx, -dz/dy, 1]
    n = normalized(n)
    return n


# @torch.compile
def interpolate_grid1(grid: torch.Tensor, query: torch.Tensor, max_coord: float | torch.Tensor) -> torch.Tensor:
    """
    Interpolates the height at the desired (query[0], query[1]]) coordinates.

    Parameters:
    - grid: Tensor of grid values corresponding to the x and y coordinates (3D array), (B, D, D). Top-left corner is (-max_coord, -max_coord). The indexing order follows the "xy" convention, meaning the first dimension is the y-axis and the second dimension is the x-axis.
    - query: Tensor of desired point coordinates for interpolation (3D array), (B, N, 2). Range is from -max_coord to max_coord.
    Returns:
    - Interpolated grid values at the queried coordinates in shape (B, N, 1).
    """
    norm_query = query / max_coord  # Normalize to [-1, 1]
    # Query coordinates of shape (B, N, 1, 2)
    grid_coords = norm_query.unsqueeze(2)
    # Grid of shape (B, 1, H, W)
    grid_w_c = grid.unsqueeze(1)
    # Interpolate the grid values into shape (B, 1, N, 1)
    z_query = torch.nn.functional.grid_sample(grid_w_c, grid_coords, align_corners=True, mode="bilinear", padding_mode="border")
    return z_query.squeeze(1)



def gridmap_interpolation_and_normals():
    robot = 'marv'
    dphys_cfg = DPhysConfig(robot=robot)

    # heightmap defining the terrain
    x_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    y_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')

    z_grid = torch.exp(-(x_grid - 2) ** 2 / 4) * torch.exp(-(y_grid - 0) ** 2 / 2)
    z_grid = z_grid.unsqueeze(0)  # add batch dimension

    # robot point cloud
    points = dphys_cfg.robot_points[::5]
    points += torch.tensor([0, 0, 2.0])
    points = points.unsqueeze(0)  # add batch dimension

    z_points = interpolate_grid(z_grid, points[..., 0], points[..., 1],
                                d_max=dphys_cfg.d_max, grid_res=dphys_cfg.grid_res, return_normals=False)
    # z_points = interpolate_grid1(z_grid, points[..., :2], dphys_cfg.d_max)
    points_grid = points.clone()
    points_grid[..., 2] = z_points

    # gradients = compute_heightmap_gradients(z_grid, dphys_cfg.d_max)
    # normals = surface_normals(gradients, points_grid[..., :2], dphys_cfg.d_max)

    mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
    mlab.surf(x_grid.numpy(), y_grid.numpy(), z_grid.squeeze(0).numpy(), colormap='jet', opacity=0.7)
    visu_pts = mlab.points3d(points[0, :, 0], points[0, :, 1], points[0, :, 2],
                             scale_factor=0.05, color=(1, 0, 0))
    visu_pts_proj = mlab.points3d(points_grid[0, :, 0], points_grid[0, :, 1], points_grid[0, :, 2],
                                  scale_factor=0.05, color=(0, 0, 1))
    # visu_normals = mlab.quiver3d(points_grid[0, :, 0].numpy(), points_grid[0, :, 1].numpy(), points_grid[0, :, 2].numpy(),
    #                              normals[0, :, 0].numpy(), normals[0, :, 1].numpy(), normals[0, :, 2].numpy(),
    #                              scale_factor=0.5, color=(0, 1, 0), opacity=0.5)
    mlab.show()




def normals_test():
    robot = 'marv'
    dphys_cfg = DPhysConfig(robot=robot)
    dphysics = DPhysics(dphys_cfg=dphys_cfg)

    ds = ROUGH(path=rough_seq_paths[0])
    sample_i = 294
    hm = ds.get_geom_height_map(sample_i)

    # heightmap defining the terrain
    x_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    y_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid)

    # z_grid = torch.sin(x_grid) * torch.cos(y_grid)
    # z_grid = torch.exp(-(x_grid - 2) ** 2 / 4) * torch.exp(-(y_grid - 0) ** 2 / 2)
    # z_grid = torch.zeros_like(x_grid)
    z_grid = hm[0]

    mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
    mlab.surf(x_grid.numpy(), y_grid.numpy(), z_grid.numpy(), colormap='jet', opacity=0.7)

    # random points withing -d_max and d_max
    n_pts = 1000
    points_xy = torch.rand(n_pts, 2) * 2 * dphys_cfg.d_max - dphys_cfg.d_max
    z_points, normals = dphysics.interpolate_grid(z_grid.unsqueeze(0),
                                                  points_xy[..., 0].unsqueeze(0),
                                                  points_xy[..., 1].unsqueeze(0), return_normals=True)
    points_grid = torch.zeros(n_pts, 3)
    points_grid[..., :2] = points_xy
    points_grid[..., 2] = z_points.squeeze(0)
    normals = normals.squeeze(0)

    visu_pts = mlab.points3d(points_grid[:, 0], points_grid[:, 1], points_grid[:, 2],
                             scale_factor=0.05, color=(0, 0, 1))
    visu_normals = mlab.quiver3d(points_grid[:, 0].numpy(), points_grid[:, 1].numpy(), points_grid[:, 2].numpy(),
                                 normals[:, 0].numpy(), normals[:, 1].numpy(), normals[:, 2].numpy(),
                                 scale_factor=0.5, color=(0, 1, 0), opacity=0.5)
    mlab.show()


def main():
    gridmap_interpolation_and_normals()
    # normals_test()


if __name__ == '__main__':
    main()
