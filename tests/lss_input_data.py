import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import numpy as np
import open3d as o3d
from monoforce.models.terrain_encoder.lss import LiftSplatShoot
from monoforce.models.terrain_encoder.utils import ego_to_cam, get_only_in_img_mask, denormalize_img
from monoforce.cloudproc import position, hm_to_cloud
from monoforce.datasets import ROUGH, rough_seq_paths
from monoforce.utils import read_yaml
from monoforce.dphys_config import DPhysConfig
mpl.use('Qt5Agg')

dphys_cfg = DPhysConfig()
lss_cfg = read_yaml('../../monoforce/monoforce/config/lss_cfg.yaml')
model = LiftSplatShoot(lss_cfg['grid_conf'], lss_cfg['data_aug_conf'], outC=1)

ds = ROUGH(path=rough_seq_paths[0], lss_cfg=lss_cfg ,dphys_cfg=dphys_cfg)


def explore_input_data(sample_i=0):
    H, W = lss_cfg['data_aug_conf']['H'], lss_cfg['data_aug_conf']['W']
    cams = ds.camera_names

    imgs, rots, trans, intrins, post_rots, post_trans = ds.get_images_data(sample_i)
    hm_terrain = ds.get_terrain_height_map(sample_i)
    pts = torch.as_tensor(position(ds.get_cloud(sample_i))).T
    height_terrain, mask_rigid = hm_terrain[0], hm_terrain[1]

    frustum_pts = model.get_geometry(rots[None], trans[None], intrins[None], post_rots[None], post_trans[None]).squeeze(0)

    n_rows, n_cols = 2, int(np.ceil(len(cams) / 2) + 2)
    img_h, img_w = imgs.shape[-2], imgs.shape[-1]
    ratio = img_h / img_w
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4 * ratio))
    gs = mpl.gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=0.0, hspace=0.0, left=0.0, right=1.0, top=1.0, bottom=0.0)

    plt.clf()
    final_ax = plt.subplot(gs[:, -1:])
    for imgi, img in enumerate(imgs):
        cam_pts = ego_to_cam(pts, rots[imgi], trans[imgi], intrins[imgi])
        mask = get_only_in_img_mask(cam_pts, H, W)
        plot_pts = post_rots[imgi].matmul(cam_pts) + post_trans[imgi].unsqueeze(1)

        ax = plt.subplot(gs[imgi // int(np.ceil(len(cams) / 2)), imgi % int(np.ceil(len(cams) / 2))])
        showimg = denormalize_img(img)

        plt.imshow(showimg)
        plt.scatter(plot_pts[0, mask], plot_pts[1, mask], c=pts[2, mask],
                    s=1, alpha=0.4, cmap='jet', vmin=-1., vmax=1.)
        plt.axis('off')
        # camera name as text on image
        plt.text(0.5, 0.9, cams[imgi].replace('_', ' '),
                 horizontalalignment='center', verticalalignment='top',
                 transform=ax.transAxes, fontsize=10)

        plt.sca(final_ax)
        plt.scatter(frustum_pts[imgi, :, :, :, 0].view(-1), frustum_pts[imgi, :, :, :, 1].view(-1),
                    label=cams[imgi].replace('_', ' '), s=0.2, alpha=0.5)

    plt.legend(loc='upper right')
    final_ax.set_aspect('equal')
    plt.title('Frustum points')

    # plot height maps
    ax = plt.subplot(gs[:, -2:-1])
    plt.imshow(height_terrain.T, origin='lower', cmap='jet', vmin=-1., vmax=1.)
    plt.title('Terrain HM')
    plt.colorbar()

    plt.show()


def frustum_points():
    sample_i = 120
    explore_input_data(sample_i)

    imgs, rots, trans, intrins, post_rots, post_trans = ds.get_images_data(sample_i)
    frustum_pts = model.get_geometry(rots[None], trans[None], intrins[None], post_rots[None], post_trans[None]).squeeze(0)
    print('Frustum points:', frustum_pts.shape)

    terrain_hm = ds.get_terrain_height_map(sample_i)
    terrain_points = hm_to_cloud(terrain_hm[0], dphys_cfg, terrain_hm[1])
    print('Terrain points:', terrain_points.shape)

    terrain_pcd = o3d.geometry.PointCloud()
    terrain_pcd.points = o3d.utility.Vector3dVector(terrain_points)
    terrain_pcd.paint_uniform_color([0, 0, 1])

    frustum_pcd = o3d.geometry.PointCloud()
    frustum_pcd.points = o3d.utility.Vector3dVector(frustum_pts.view(-1, 3).detach().numpy())

    grav_alined_pose = np.eye(4)
    grav_alined_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    grav_alined_frame.transform(grav_alined_pose)

    # camera coordinate frame
    cam_frames = []
    for tran, rot in zip(trans, rots):
        pose = np.eye(4)
        pose[:3, :3] = rot
        pose[:3, 3] = tran
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
        frame.transform(pose)
        cam_frames.append(frame)

    # o3d.visualization.draw_geometries([frustum_pcd, grav_alined_frame, terrain_pcd] + cam_frames)
    o3d.visualization.draw_geometries([grav_alined_frame, terrain_pcd] + cam_frames)


def main():
    frustum_points()


if __name__ == '__main__':
    main()