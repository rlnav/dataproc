import matplotlib.pyplot as plt
import torch
from monoforce.models.traj_predictor.dphysics import DPhysics
from monoforce.models.traj_predictor.dphys_config import DPhysConfig
import matplotlib as mpl
mpl.use('Qt5Agg')


def debug():
    robot = 'tradr'
    dphys_cfg = DPhysConfig(robot=robot)
    dphys_cfg.n_sim_trajs = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    B = dphys_cfg.n_sim_trajs
    T = dphys_cfg.traj_sim_time
    dt = dphys_cfg.dt

    # instantiate the simulator
    dphysics = DPhysics(dphys_cfg, device=device)

    # terrain properties
    H, W = int(2 * dphys_cfg.d_max / dphys_cfg.grid_res), int(2 * dphys_cfg.d_max / dphys_cfg.grid_res)
    # z_grid = torch.rand(B, H, W, device=device)
    z_grid = torch.zeros(B, H, W, device=device)

    controls = torch.stack([torch.tensor([[1.0, 0.0]] * int(T / dt))]).repeat(B, 1, 1)
    controls = torch.as_tensor(controls, dtype=torch.float32, device=device)

    # put tensors to device
    z_grid = z_grid.to(device)
    controls = controls.to(device)

    # simulate the rigid body dynamics
    with torch.no_grad():
        states, forces = dphysics(z_grid=z_grid, controls=controls, vis=True)
        print('xyz shape:', states[0].shape)

    with torch.no_grad():
        states1, forces1 = dphysics(z_grid=z_grid[:1], controls=controls[:1], vis=True)
        print('xyz1 shape:', states1[0].shape)

        plt.figure()
        xyz = states[0][:1].cpu().numpy()
        xyz1 = states1[0].cpu().numpy()
        plt.plot(xyz[0, ::8, 0], xyz[0, ::8, 1], 'r.')
        plt.plot(xyz1[0, ::10, 0], xyz1[0, ::10, 1], 'b.')
        plt.legend(['xyz', 'xyz1'])
        plt.show()

        if torch.allclose(states[0][:1], states1[0], atol=1e-3):
            print('Success!')
        else:
            print('xyz1 != xyz')
            # raise ValueError('xyz1 != xyz')


if __name__ == '__main__':
    debug()
