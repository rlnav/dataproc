#!/usr/bin/env python

import sys
sys.path.append('../../monoforce/monoforce/src/')
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
import argparse
from monoforce.models.traj_predictor.dphys_config import DPhysConfig
from monoforce.models.traj_predictor.dphysics import DPhysics
from monoforce.models.terrain_encoder.lss import LiftSplatShoot
from monoforce.models.terrain_encoder.utils import ego_to_cam, get_only_in_img_mask, denormalize_img
from monoforce.utils import read_yaml, compile_data, str2bool, normalize
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
        self.dphys_cfg.vel_max = 1.0
        self.dphys_cfg.omega_max = 1.0
        self.dphys_cfg.traj_sim_time = 8.0
        self.dphys_cfg.n_sim_trajs = 32
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

    def predict_states(self, terrain, batch):
        model = self.traj_predictor.__class__.__name__
        n_trajs = self.dphys_cfg.n_sim_trajs
        T = self.dphys_cfg.traj_sim_time
        v = self.dphys_cfg.vel_max
        w = self.dphys_cfg.omega_max
        if model == 'DPhysics':
            controls = generate_controls(n_trajs=n_trajs, time_horizon=T, v=v, w_range=(-w, w))
            controls = controls.to(self.device)
            # Xs, Xds, Rs, Omegas = batch[12:16]
            # state0 = tuple([s[:, 0] for s in [Xs.repeat(n_trajs, 1, 1), Xds.repeat(n_trajs, 1, 1),
            #                                   Rs.repeat(n_trajs, 1, 1, 1), Omegas.repeat(n_trajs, 1, 1)]])
            height, friction = terrain['terrain'], terrain['friction']
            states_pred, forces_pred = self.traj_predictor(z_grid=height.squeeze(1).repeat(n_trajs, 1, 1),
                                                           # state=state0,
                                                           friction=friction.squeeze(1).repeat(n_trajs, 1, 1),
                                                           controls=controls)
            N_forces = forces_pred[0]  # (n_trajs, time_horizon, n_pts, 3)
            traj_costs = N_forces.norm(dim=-1).mean(dim=-1).std(dim=-1)
            # traj_costs = N_forces.norm(dim=-1).mean(dim=-1).max(dim=-1)[0] #- N_forces.norm(dim=-1).mean(dim=-1).min(dim=-1)[0]
        else:
            raise ValueError(f'Invalid model: {model}. Supported: DPhysics')
        return states_pred, traj_costs

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

        # Define a custom colormap from Green -> Yellow -> Red
        colors = ["green", "red"]
        custom_cmap = LinearSegmentedColormap.from_list("green_red", colors, N=self.dphys_cfg.n_sim_trajs)

        R90 = np.array([[0, 1],
                        [-1, 0]])
        for i, batch in enumerate(tqdm(self.loader)):
            # if os.path.exists(f'{self.output_folder}/{i:04d}.png'):
            #     continue
            t0 = time()
            batch = [t.to(self.device) for t in batch]

            # terrain prediction
            terrain = self.predict_terrain(batch)
            H_t_pred, H_g_pred, H_diff_pred, Friction_pred = (terrain['terrain'], terrain['geom'],
                                                              terrain['diff'], terrain['friction'])
            # trajectory prediction loss: xyz and rotation
            states_pred, traj_costs = self.predict_states(terrain, batch)
            traj_costs_norm = (traj_costs - traj_costs.min()) / (traj_costs.max() - traj_costs.min())
            traj_colors = custom_cmap(traj_costs_norm.cpu().numpy())[..., :3][:, ::-1]
            t1 = time()
            print(f'Prediction time: {t1 - t0:.2f} s')

            # visualizations
            H_t_pred = H_t_pred[0, 0].cpu()
            Friction_pred = Friction_pred[0, 0].cpu()
            Xs_pred = states_pred[0].cpu()
            grid_res = self.lss_config['grid_conf']['xbound'][2]

            # get a sample from the dataset
            batch = [t.cpu() for t in batch]
            (imgs, rots, trans, intrins, post_rots, post_trans,
             hm_geom, hm_terrain,
             control_ts, controls,
             pose0,
             traj_ts, Xs, Xds, Rs, Omegas) = batch

            # visualize images using opencv
            imgs_vis = []
            for imgi, img in enumerate(imgs[0][:3]):
                showimg = np.asarray(denormalize_img(img))
                showimg = cv2.cvtColor(showimg, cv2.COLOR_RGB2BGR)
                # project trajectory points on the image
                cam_pts_Xs_pred = ego_to_cam(Xs_pred[:, :, :3].reshape(-1, 3).T, rots[0, imgi], trans[0, imgi], intrins[0, imgi])
                mask_img_Xs_pred = get_only_in_img_mask(cam_pts_Xs_pred, img_H, img_W)
                plot_pts_Xs_pred = post_rots[0, imgi].matmul(cam_pts_Xs_pred) + post_trans[0, imgi].unsqueeze(1)

                for i in torch.where(mask_img_Xs_pred)[0]:
                    traj_i = int(i // (self.dphys_cfg.traj_sim_time // self.dphys_cfg.dt)) - 1
                    color = traj_colors[traj_i] * 255
                    uv = plot_pts_Xs_pred[:, i]
                    cv2.circle(showimg, (int(uv[0]), int(uv[1])), 2, color, -1)
                imgs_vis.append(showimg)
            # concatenate images
            img_vis = np.concatenate(imgs_vis, axis=1)
            terrain_vis = H_t_pred.cpu().numpy()
            terrain_vis = np.flipud(np.fliplr(terrain_vis))
            h, w = terrain_vis.shape
            terrain_vis = terrain_vis[:h//3*2, :]
            # add color to terrain
            terrain_vis = cv2.applyColorMap((normalize(terrain_vis) * 255).astype(np.uint8), cv2.COLORMAP_JET)
            # plot the predicted trajectory as lines
            for traj_i in range(len(Xs_pred)):
                Xs_pred_vis = (Xs_pred[traj_i, :, :2] @ R90 + self.dphys_cfg.d_max) / grid_res
                Xs_pred_vis = Xs_pred_vis.cpu().numpy().astype(np.int32)
                # color based on cost small (green) to large (red)
                color = traj_colors[traj_i] * 255
                for j in range(1, len(Xs_pred_vis)):
                    cv2.line(terrain_vis, tuple(Xs_pred_vis[j-1]), tuple(Xs_pred_vis[j]), color, 1)
            terrain_vis = cv2.resize(terrain_vis, (img_vis.shape[1], img_vis.shape[1]//3*2), interpolation=cv2.INTER_NEAREST)
            # concatenate images and terrain
            res_vis = np.concatenate([img_vis, terrain_vis], axis=0)
            res_vis = cv2.resize(res_vis, (res_vis.shape[1]//2, res_vis.shape[0]//2), interpolation=cv2.INTER_NEAREST)

            cv2.imshow('Predictions', res_vis)
            # cv2.waitKey(0)
            # break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            t2 = time()
            print(f'Visualization time: {t2 - t1:.2f} s')
        cv2.destroyAllWindows()


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
