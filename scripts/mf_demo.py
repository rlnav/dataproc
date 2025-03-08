#!/usr/bin/env python

import sys
sys.path.append('../../monoforce/monoforce/src/')
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from monoforce.models.traj_predictor.dphys_config import DPhysConfig
from monoforce.models.traj_predictor.dphysics import DPhysics
from monoforce.models.terrain_encoder.lss import LiftSplatShoot
from monoforce.models.terrain_encoder.utils import ego_to_cam, get_only_in_img_mask, denormalize_img
from monoforce.utils import read_yaml, compile_data, str2bool
from monoforce.datasets import ROUGH, rough_seq_paths

np.random.seed(42)
torch.manual_seed(42)


def arg_parser():
    parser = argparse.ArgumentParser(description='Terrain encoder predictor input arguments')
    parser.add_argument('--seq', type=str, default='val', help='Data sequence')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--terrain_encoder', type=str, default='lss', help='Terrain encoder model')
    parser.add_argument('--terrain_encoder_path', type=str, default=None, help='Path to the LSS model')
    parser.add_argument('--traj_predictor', type=str, default='dphysics', help='Trajectory predictor model')
    parser.add_argument('--vis', type=str2bool, default=True, help='Visualize the results')
    return parser.parse_args()

def generate_controls(n_trajs=10,
                      time_horizon=5.0, dt=0.01,
                      w_range=(-1.0, 1.0), v=1.0):
    """
    Generates control inputs for the robot trajectories.

    Parameters:
    - n_trajs: Number of trajectories.
    - time_horizon: Time horizon for each trajectory.
    - dt: Time step.
    - w_range: Range of the rotational speed.

    Returns:
    - Linear and angular velocities for the robot trajectories: (n_trajs, time_steps, 2).
    """
    N = int(time_horizon / dt)
    ws = torch.stack([torch.linspace(w_range[0], w_range[1], n_trajs // 2),
                      torch.linspace(w_range[1], w_range[0], n_trajs//2)]).flatten()
    vs = torch.stack([v * torch.ones(n_trajs // 2),
                      -v * torch.ones(n_trajs // 2)]).flatten()

    # repeat the controls for each time step
    vs = vs.unsqueeze(1).repeat(1, N)
    ws = ws.unsqueeze(1).repeat(1, N)

    # stack the controls
    controls = torch.stack([vs, ws], dim=-1)

    return controls

class Demo:
    def __init__(self,
                 seq='val',
                 batch_size=1,
                 terrain_encoder='lss',
                 terrain_encoder_path=None,
                 traj_predictor='dphysics'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load DPhys config
        if seq in rough_seq_paths:
            robot = os.path.basename(seq).split('_')[0]
            robot = 'tradr' if robot == 'ugv' else 'marv'
        else:
            robot = 'marv'
        print(f'Robot: {robot}')
        self.dphys_cfg = DPhysConfig(robot=robot)
        self.traj_predictor = self.get_traj_pred(model=traj_predictor)

        # load LSS config
        self.lss_config = read_yaml(os.path.join('../../monoforce/monoforce', 'config/lss_cfg.yaml'))
        self.terrain_encoder = self.get_terrain_encoder(terrain_encoder_path, model=terrain_encoder)

        # load data
        self.loader = self.get_dataloader(batch_size=batch_size, seq=seq)

        # output folder to write results
        self.output_folder = (f'./gen/demo_{os.path.basename(seq)}/'
                              f'{robot}_{self.terrain_encoder.__class__.__name__}_'
                              f'{self.traj_predictor.__class__.__name__}')

    def get_terrain_encoder(self, path, model='lss'):
        if model == 'lss':
            terrain_encoder = LiftSplatShoot(self.lss_config['grid_conf'],
                                             self.lss_config['data_aug_conf']).from_pretrained(path)
        else:
            raise ValueError(f'Invalid terrain encoder model: {model}. Supported: lss')
        terrain_encoder.to(self.device)
        terrain_encoder.eval()
        return terrain_encoder

    def predict_terrain(self, batch):
        model = self.terrain_encoder.__class__.__name__
        if model == 'LiftSplatShoot':
            imgs, rots, trans, intrins, post_rots, post_trans = batch[:6]
            img_inputs = (imgs, rots, trans, intrins, post_rots, post_trans)
            terrain = self.terrain_encoder(*img_inputs)
        else:
            raise ValueError(f'Invalid terrain encoder model: {model}. Supported: LiftSplatShoot')
        return terrain

    def get_traj_pred(self, model='dphysics'):
        if model == 'dphysics':
            traj_predictor = DPhysics(self.dphys_cfg, device=self.device)
        else:
            raise ValueError(f'Invalid trajectory predictor model: {model}. Supported: dphysics')
        traj_predictor.to(self.device)
        traj_predictor.eval()
        return traj_predictor

    def predict_states(self, terrain, batch, n_trajs=32):
        model = self.traj_predictor.__class__.__name__
        if model == 'DPhysics':
            controls = generate_controls(n_trajs=n_trajs)
            controls = controls.to(self.device)
            # Xs, Xds, Rs, Omegas = batch[12:16]
            # state0 = tuple([s[:, 0] for s in [Xs.repeat(n_trajs, 1, 1), Xds.repeat(n_trajs, 1, 1),
            #                                   Rs.repeat(n_trajs, 1, 1, 1), Omegas.repeat(n_trajs, 1, 1)]])
            height, friction = terrain['terrain'], terrain['friction']
            states_pred, _ = self.traj_predictor(z_grid=height.squeeze(1).repeat(n_trajs, 1, 1),
                                                 # state=state0,
                                                 controls=controls, friction=friction.squeeze(1).repeat(n_trajs, 1, 1))
        else:
            raise ValueError(f'Invalid model: {model}. Supported: DPhysics')
        return states_pred

    def get_dataloader(self, batch_size=1, seq='val'):
        if seq != 'val':
            print('Loading dataset from:', seq)
            val_ds = ROUGH(path=seq, lss_cfg=self.lss_config, dphys_cfg=self.dphys_cfg)
        else:
            _, val_ds = compile_data(lss_cfg=self.lss_config, dphys_cfg=self.dphys_cfg)
        loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return loader

    @torch.inference_mode()
    def run(self, vis=False):
        # create output folder
        os.makedirs(self.output_folder, exist_ok=True)

        img_H, img_W = self.lss_config['data_aug_conf']['H'], self.lss_config['data_aug_conf']['W']
        cams = ['cam_left', 'cam_front', 'cam_right', 'cam_rear']

        x_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
        y_grid = torch.arange(-self.dphys_cfg.d_max, self.dphys_cfg.d_max, self.dphys_cfg.grid_res)
        x_grid, y_grid = torch.meshgrid(x_grid, y_grid)

        ratio = img_H / img_W
        fig, axes = plt.subplots(3, 4, figsize=(12, 3 * ratio + 6))
        gs = fig.add_gridspec(3, 4)
        for i, batch in enumerate(tqdm(self.loader)):
            if os.path.exists(f'{self.output_folder}/{i:04d}.png'):
                continue
            batch = [t.to(self.device) for t in batch]

            # terrain prediction
            terrain = self.predict_terrain(batch)
            H_t_pred, H_g_pred, H_diff_pred, Friction_pred = (terrain['terrain'], terrain['geom'],
                                                              terrain['diff'], terrain['friction'])
            # trajectory prediction loss: xyz and rotation
            states_pred = self.predict_states(terrain, batch)

            # visualizations
            H_t_pred = H_t_pred[0, 0].cpu()
            Friction_pred = Friction_pred[0, 0].cpu()
            Xs_pred = states_pred[0].cpu()
            grid_res = self.lss_config['grid_conf']['xbound'][2]

            # get height map points
            hm_points = torch.stack([x_grid, y_grid, H_t_pred], dim=-1)
            hm_points = hm_points.view(-1, 3).T

            batch = [t.to('cpu') for t in batch]
            # get a sample from the dataset
            (imgs, rots, trans, intrins, post_rots, post_trans,
             hm_geom, hm_terrain,
             control_ts, controls,
             pose0,
             traj_ts, Xs, Xds, Rs, Omegas) = batch

            # clear axis
            for ax in axes.flatten():
                ax.clear()
                ax.axis('off')
            for imgi, img in enumerate(imgs[0]):
                ax = axes[0, imgi]
                showimg = denormalize_img(img)
                ax.imshow(showimg)
                # camera name as text on image
                ax.text(0.5, 0.9, cams[imgi].replace('_', ' '),
                        horizontalalignment='center', verticalalignment='top',
                        transform=ax.transAxes, fontsize=10)

                # plot points projected on the image
                cam_pts = ego_to_cam(hm_points, rots[0, imgi], trans[0, imgi], intrins[0, imgi])
                mask_img = get_only_in_img_mask(cam_pts, img_H, img_W)
                plot_pts = post_rots[0, imgi].matmul(cam_pts) + post_trans[0, imgi].unsqueeze(1)
                ax.scatter(plot_pts[0, mask_img], plot_pts[1, mask_img],
                           c=Friction_pred.view(-1)[mask_img],
                           # c=hm_points[2, mask_img],
                           s=2, alpha=0.8, cmap='jet', vmin=0, vmax=1.)

                # plot trajectories projected on the image
                for traj_i in range(len(Xs_pred)):
                    cam_pts_Xs_pred = ego_to_cam(Xs_pred[traj_i, :, :3].T, rots[0, imgi], trans[0, imgi], intrins[0, imgi])
                    mask_img_Xs_pred = get_only_in_img_mask(cam_pts_Xs_pred, img_H, img_W)
                    plot_pts_Xs_pred = post_rots[0, imgi].matmul(cam_pts_Xs_pred) + post_trans[0, imgi].unsqueeze(1)
                    ax.scatter(plot_pts_Xs_pred[0, mask_img_Xs_pred], plot_pts_Xs_pred[1, mask_img_Xs_pred], c='k', s=0.5)

            # plot terrain heightmap
            ax = fig.add_subplot(gs[1:3, 0:2])
            ax.clear()
            ax.axis('off')
            ax.set_title('Terrain Heightmap')
            H_t_vis = np.fliplr(H_t_pred)
            ax.imshow(H_t_vis, origin='lower', cmap='jet', vmin=-1., vmax=1.)
            R90 = np.array([[0, 1],
                            [-1, 0]])
            for traj_i in range(len(Xs_pred)):
                Xs_pred_vis = (Xs_pred[traj_i, :, :2] @ R90 + self.dphys_cfg.d_max) / grid_res
                ax.plot(Xs_pred_vis[:, 0], Xs_pred_vis[:, 1], 'k')

            # plot friction
            ax = fig.add_subplot(gs[1:3, 2:4])
            ax.clear()
            ax.axis('off')
            ax.set_title('Friction')
            Friction_vis = np.fliplr(Friction_pred)
            ax.imshow(Friction_vis, origin='lower', cmap='jet', vmin=0., vmax=1.)
            R90 = np.array([[0, 1],
                            [-1, 0]])
            for traj_i in range(len(Xs_pred)):
                Xs_pred_vis = (Xs_pred[traj_i, :, :2] @ R90 + self.dphys_cfg.d_max) / grid_res
                ax.plot(Xs_pred_vis[:, 0], Xs_pred_vis[:, 1], 'k')

            plt.tight_layout()
            if vis:
                plt.pause(0.01)
                plt.draw()
            plt.savefig(f'{self.output_folder}/{i:04d}.png')
        plt.close(fig)


def main():
    args = arg_parser()
    print(args)
    demo = Demo(seq=args.seq,
                     batch_size=args.batch_size,
                     terrain_encoder=args.terrain_encoder,
                     terrain_encoder_path=args.terrain_encoder_path,
                     traj_predictor=args.traj_predictor)
    demo.run(vis=args.vis)


if __name__ == '__main__':
    main()
