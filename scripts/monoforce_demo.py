#!/usr/bin/env python

import sys
sys.path.append('../../monoforce/monoforce/src/')
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import os
import numpy as np
import cv2
import torch
from torch.utils.data import DataLoader
from collections import deque
import argparse
from monoforce.models.physics_engine.engine.engine import DPhysicsEngine, PhysicsState
from monoforce.configs import WorldConfig, RobotModelConfig, PhysicsEngineConfig
from monoforce.models.physics_engine.engine.engine_state import vectorize_iter_of_states as vectorize_states
from monoforce.models.physics_engine.utils.environment import make_x_y_grids
from monoforce.models.terrain_encoder.lss import LiftSplatShoot
from monoforce.models.terrain_encoder.utils import ego_to_cam, get_only_in_img_mask, denormalize_img
from monoforce.utils import read_yaml, compile_data, str2bool, normalize, timing
from monoforce.datasets import ROUGH


def arg_parser():
    parser = argparse.ArgumentParser(description='Terrain encoder predictor input arguments')
    parser.add_argument('--seq', type=str, default='val', help='Data sequence')
    parser.add_argument('--n_trajs', type=int, default=16, help='Number of predicted trajecotries')
    parser.add_argument('--terrain_encoder_path', type=str,
                        default='../../monoforce/monoforce/config/weights/lss/val.pth',
                        help='Path to the LSS model')
    parser.add_argument('--vis', type=str2bool, default=True, help='Visualize the results')
    return parser.parse_args()

def generate_vws(n_trajs=10,
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
    ws = torch.linspace(w_range[0], w_range[1], n_trajs).flatten()
    vs = v * torch.ones(n_trajs).flatten()
    # vs[::2] *= -1.  # reverse the direction of the last half of the trajectories

    # repeat the controls for each time step
    vs = vs.unsqueeze(1).repeat(1, N)
    ws = ws.unsqueeze(1).repeat(1, N)

    # stack the controls
    controls = torch.stack([vs, ws], dim=-1)

    return controls

class Demo:
    def __init__(self,
                 seq='val',
                 n_trajs=16,
                 grid_res=0.1,  # 10cm per grid cell
                 max_coord=6.4,  # meters
                 terrain_encoder_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load DPhys config
        self.robot_model = RobotModelConfig().to(self.device)
        self.n_trajs = n_trajs
        x_grid, y_grid = make_x_y_grids(max_coord=max_coord, grid_res=grid_res, num_robots=n_trajs)
        z_grid = torch.zeros_like(x_grid)
        self.world_config = WorldConfig(
            x_grid=x_grid,
            y_grid=y_grid,
            z_grid=z_grid,
            grid_res=grid_res,
            max_coord=max_coord,
        ).to(self.device)
        self.physics_config = PhysicsEngineConfig(num_robots=n_trajs).to(self.device)
        self.physics_engine = self.get_physics_engine()

        # load LSS config
        self.lss_config = read_yaml(os.path.join('../../monoforce/monoforce', 'config/lss_cfg.yaml'))
        self.terrain_encoder = self.get_terrain_encoder(terrain_encoder_path)

        # load data
        self.loader = self.get_dataloader(batch_size=1, seq=seq)

        self.controls = self.get_controls(n_trajs=n_trajs)

        # output video file to write results
        self.output_video = f'./gen/demo_{os.path.basename(os.path.normpath(seq))}.mp4'
        # create output folder
        os.makedirs('./gen/', exist_ok=True)
        self.out_size = (1248, 880)  # (width, height)
        self.video_writer = cv2.VideoWriter(self.output_video, cv2.VideoWriter_fourcc(*'mp4v'), 10, self.out_size)

    def get_controls(self, n_trajs=1):
        vws = generate_vws(n_trajs=n_trajs, time_horizon=7.0).to(self.device)
        n_trajs, n_iters = vws.shape[:2]
        vs, ws = vws[..., 0], vws[..., 1]
        flipper_vels = self.robot_model.vw_to_vels(v=vs, w=ws).reshape(n_trajs, n_iters, -1)
        flipper_angles = torch.zeros_like(flipper_vels)
        controls = torch.cat([flipper_vels, flipper_angles], dim=-1)
        assert controls.shape == (n_trajs, n_iters, 8)
        return controls

    def get_terrain_encoder(self, path, model='lss'):
        terrain_encoder = LiftSplatShoot(self.lss_config['grid_conf'],
                                         self.lss_config['data_aug_conf']).from_pretrained(path)
        terrain_encoder.to(self.device)
        terrain_encoder.eval()
        return terrain_encoder

    def predict_terrain(self, batch):
        imgs, rots, trans, intrins, post_rots, post_trans = batch[:6]
        img_inputs = (imgs, rots, trans, intrins, post_rots, post_trans)
        terrain = self.terrain_encoder(*img_inputs)
        return terrain

    def get_physics_engine(self):
        enine = DPhysicsEngine(self.physics_config, self.robot_model, self.device)
        enine.to(self.device)
        enine.eval()
        return enine

    def predict_states(self, terrain, batch):
        n_trajs, n_iters = self.controls.shape[:2]

        (imgs, rots, trans, intrins, post_rots, post_trans,
         hm_geom, hm_terrain,
         control_ts, controls,
         traj_ts, xs, xds, qs, omegas, thetas) = batch

        # Initial state
        x0 = xs[:, 0].repeat(n_trajs, 1)
        xd0 = xds[:, 0].repeat(n_trajs, 1)
        q0 = qs[:, 0].repeat(n_trajs, 1)
        omega0 = omegas[:, 0].repeat(n_trajs, 1)
        thetas0 = thetas[:, 0].repeat(n_trajs, 1)
        state0 = PhysicsState(x0, xd0, q0, omega0, thetas0, batch_size=n_trajs)

        self.world_config.z_grid = terrain['terrain'].squeeze(1).repeat(n_trajs, 1, 1)
        states_pred = deque(maxlen=n_iters)
        state = state0
        for i in range(n_iters):
            state, der, aux = self.physics_engine(state, self.controls[:, i], self.world_config)
            states_pred.append(state)
        states_pred = vectorize_states(states_pred)

        # costs: average XY angular velocity. Omega.shape == (n_iters, n_trajs, 3)
        traj_costs = states_pred.omega[:, :, :2].norm(dim=2).mean(dim=0)

        return states_pred, traj_costs

    def get_dataloader(self, batch_size=1, seq='val'):
        if seq != 'val':
            print('Loading dataset from:', seq)
            val_ds = ROUGH(path=seq, lss_cfg=self.lss_config)
        else:
            _, val_ds = compile_data(lss_cfg=self.lss_config)
        loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return loader

    @torch.inference_mode()
    def run(self, vis=False):
        img_H, img_W = self.lss_config['data_aug_conf']['H'], self.lss_config['data_aug_conf']['W']
        cams = ['cam_left', 'cam_front', 'cam_right', 'cam_rear']

        # Define a custom colormap from Green -> Red
        colors = ["green", "red"]
        custom_cmap = LinearSegmentedColormap.from_list("green_red", colors, N=self.n_trajs)
        for i, batch in enumerate(tqdm(self.loader)):
            batch = [t.to(self.device) for t in batch]

            # terrain prediction
            terrain = self.predict_terrain(batch)

            # TODO: remove this hack with newer models (that use xy indexing instead of ij)
            terrain['terrain'] = terrain['terrain'].transpose(2, 3)  # (B, 1, H, W)
            # terrain['terrain'] = batch[6][:, 0].unsqueeze(1)  # use the ground truth terrain for visualization

            # predict states
            states_pred, traj_costs = self.predict_states(terrain, batch)
            TRAJ_COST_MIN = traj_costs.min().item()
            TRAJ_COST_MAX = traj_costs.max().item()
            traj_costs_norm = (traj_costs - TRAJ_COST_MIN) / (TRAJ_COST_MAX - TRAJ_COST_MIN)
            traj_colors = custom_cmap(traj_costs_norm.cpu().numpy())[..., :3][:, ::-1]  # RGB to BGR

            # get a sample from the dataset
            batch = [t.cpu() for t in batch]
            (imgs, rots, trans, intrins, post_rots, post_trans,
             hm_geom, hm_terrain,
             control_ts, controls,
             traj_ts, xs, xds, qs, omegas, thetas) = batch

            # visualizations
            xs_pred = states_pred.x.permute(1, 0, 2).cpu()

            x_grid = self.world_config.x_grid[0].cpu()
            y_grid = self.world_config.y_grid[0].cpu()
            z_grid = terrain['terrain'][0, 0].cpu()
            hm_pts = torch.stack([x_grid, y_grid, z_grid]).view(3, -1)

            # visualize images using opencv
            imgs_vis = []
            for imgi, img in enumerate(imgs[0][:3]):
                showimg = np.asarray(denormalize_img(img))
                h, w = showimg.shape[:2]
                showimg = cv2.cvtColor(showimg, cv2.COLOR_RGB2BGR)
                # add text: name of the camera in the top-middle of the image
                cv2.putText(showimg, cams[imgi], (w//2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                # project heightmap points on the image
                cam_pts = ego_to_cam(hm_pts, rots[0, imgi], trans[0, imgi], intrins[0, imgi])
                mask_img = get_only_in_img_mask(cam_pts, img_H, img_W)
                plot_pts = post_rots[0, imgi].matmul(cam_pts[:, mask_img]) + post_trans[0, imgi].unsqueeze(1)
                for uv in plot_pts.T:
                    cv2.circle(showimg, (int(uv[0]), int(uv[1])), 1, (0., 0., 0.), -1)

                # project trajectory points on the image
                for traj_i in range(len(xs_pred)):
                    cam_pts_Xs_pred = ego_to_cam(xs_pred[traj_i, :, :3].reshape(-1, 3).T,
                                                 rots[0, imgi], trans[0, imgi], intrins[0, imgi])
                    mask_img_Xs_pred = get_only_in_img_mask(cam_pts_Xs_pred, img_H, img_W)
                    plot_pts_Xs_pred = post_rots[0, imgi].matmul(cam_pts_Xs_pred) + post_trans[0, imgi].unsqueeze(1)
                    color = traj_colors[traj_i] * 255
                    for uv in plot_pts_Xs_pred[:, mask_img_Xs_pred].T:
                        cv2.circle(showimg, (int(uv[0]), int(uv[1])), 2, color, -1)

                imgs_vis.append(showimg)

            # concatenate images
            img_vis = np.concatenate(imgs_vis, axis=1)
            # add color to terrain
            z_grid_vis = (normalize(z_grid).numpy() * 255).astype(np.uint8)
            z_grid_vis = cv2.applyColorMap(z_grid_vis, cv2.COLORMAP_JET)
            # plot the predicted trajectory as lines
            for traj_i in range(len(xs_pred)):
                xs_pred_vis = (xs_pred[traj_i, :, :2] + self.world_config.max_coord) / self.world_config.grid_res
                xs_pred_vis = xs_pred_vis.cpu().numpy().astype(np.int32)
                # color based on cost small (green) to large (red)
                color = traj_colors[traj_i] * 255
                for j in range(1, len(xs_pred_vis)):
                    cv2.line(z_grid_vis, tuple(xs_pred_vis[j-1]), tuple(xs_pred_vis[j]), color, 1)
            z_grid_vis = cv2.resize(z_grid_vis, (img_vis.shape[1], img_vis.shape[1]), interpolation=cv2.INTER_NEAREST)

            # vertival flip to have Y axis up
            z_grid_vis = cv2.flip(z_grid_vis, 0)

            # rotate 90 degrees counterclockwise to have forward direction up
            z_grid_vis = cv2.rotate(z_grid_vis, cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.putText(z_grid_vis, 'Elevation', (img_vis.shape[1] // 2, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # do not show bottom half of the terrain
            h, w = z_grid_vis.shape[:2]
            z_grid_vis = z_grid_vis[:h // 2, :]

            # concatenate images and terrain
            res_vis = np.concatenate([img_vis, z_grid_vis], axis=0)
            # resize to output size for video writing
            res_vis = cv2.resize(res_vis, self.out_size, interpolation=cv2.INTER_NEAREST)

            if vis:
                cv2.imshow('Predictions', res_vis)
                # cv2.waitKey(0)
                # break
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                self.video_writer.write(res_vis)

        cv2.destroyAllWindows()


def main():
    args = arg_parser()
    print(args)
    demo = Demo(seq=args.seq,
                n_trajs=args.n_trajs,
                terrain_encoder_path=args.terrain_encoder_path)
    demo.run(vis=args.vis)


if __name__ == '__main__':
    main()
