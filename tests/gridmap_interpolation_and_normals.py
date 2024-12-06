from monoforce.models.dphysics import DPhysics
from monoforce.dphys_config import DPhysConfig
from monoforce.datasets.rough import ROUGH, rough_seq_paths
from mayavi import mlab
import torch
import os


def gridmap_interpolation_and_normals():
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
    z_grid[~hm[1].bool()] = z_grid[hm[1].bool()].min()

    z_grid = z_grid.unsqueeze(0)  # add batch dimension

    # robot point cloud
    points0 = dphys_cfg.robot_points[::5]
    points0 += torch.tensor([0, 0, 2.0])
    points0 = points0.unsqueeze(0)  # add batch dimension

    mlab.figure(bgcolor=(1, 1, 1), size=(800, 800))
    mlab.surf(x_grid.numpy(), y_grid.numpy(), z_grid[0].numpy(), colormap='jet', opacity=0.7)

    visu_pts = mlab.points3d(points0[0, :, 0], points0[0, :, 1], points0[0, :, 2],
                             scale_factor=0.05, color=(0, 0, 1))
    visu_proj_pts = mlab.points3d(points0[0, :, 0], points0[0, :, 1], points0[0, :, 2],
                                  scale_factor=0.05, color=(1, 0, 0))
    visu_normals = mlab.quiver3d(points0[0, :, 0].numpy(), points0[0, :, 1].numpy(), points0[0, :, 2].numpy(),
                                 points0[0, :, 0].numpy(), points0[0, :, 1].numpy(), points0[0, :, 2].numpy(),
                                 scale_factor=1.0, color=(0, 1, 0))

    for x in torch.arange(-8.0, 8.0, 0.1):
        points = points0 + torch.tensor([x, x, 0]).view(1, 1, 3)

        z_points, normals = dphysics.interpolate_grid(z_grid, points[..., 0], points[..., 1], return_normals=True)
        points_grid = points.clone()
        points_grid[..., 2] = z_points

        # update plot
        visu_pts.mlab_source.set(x=points[0, :, 0].numpy(), y=points[0, :, 1].numpy(), z=points[0, :, 2].numpy())
        visu_proj_pts.mlab_source.set(x=points_grid[0, :, 0].numpy(), y=points_grid[0, :, 1].numpy(), z=points_grid[0, :, 2].numpy())
        visu_normals.mlab_source.set(x=points_grid[0, :, 0].numpy(), y=points_grid[0, :, 1].numpy(), z=points_grid[0, :, 2].numpy(),
                                     u=normals[0, :, 0].numpy(), v=normals[0, :, 1].numpy(), w=normals[0, :, 2].numpy())

        os.makedirs('gen', exist_ok=True)
        mlab.savefig(f'gen/{x}.png')

    mlab.show()


def normals():
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
    normals()


if __name__ == '__main__':
    main()
