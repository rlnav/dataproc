import torch
from time import time
import matplotlib.pyplot as plt
from monoforce.vis import set_axes_equal
from monoforce.models.dphysics import DPhysics, generate_control_inputs
from monoforce.dphys_config import DPhysConfig
from monoforce.datasets.rough import ROUGH, rough_seq_paths
import matplotlib as mpl
mpl.use('Qt5Agg')


def shoot_multiple():
    # simulation parameters
    dphys_cfg = DPhysConfig()
    dphys_cfg.n_sim_trajs = 128
    dt = dphys_cfg.dt
    T = dphys_cfg.traj_sim_time
    num_trajs = dphys_cfg.n_sim_trajs
    vel_max, omega_max = dphys_cfg.vel_max, dphys_cfg.omega_max
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    # instantiate the simulator
    dphysics = DPhysics(dphys_cfg, device=device, use_odeint=True)

    # initial state
    x = torch.tensor([[0.0, 0.0, 0.0]], device=device).repeat(num_trajs, 1)
    xd = torch.zeros_like(x)
    R = torch.eye(3, device=device).repeat(x.shape[0], 1, 1)
    omega = torch.zeros_like(x)

    # terrain properties
    x_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res).to(device)
    y_grid = torch.arange(-dphys_cfg.d_max, dphys_cfg.d_max, dphys_cfg.grid_res).to(device)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid, indexing='ij')

    # z_grid = torch.exp(-(x_grid - 2) ** 2 / 4) * torch.exp(-(y_grid - 0) ** 2 / 2)
    # z_grid = torch.sin(x_grid) * torch.cos(y_grid)
    # z_grid = torch.zeros_like(x_grid)
    ds = ROUGH(path=rough_seq_paths[0])
    z_grid = ds.get_geom_height_map(294)[0]

    # repeat the heightmap for each rigid body
    x_grid = x_grid.repeat(num_trajs, 1, 1)
    y_grid = y_grid.repeat(num_trajs, 1, 1)
    z_grid = z_grid.repeat(num_trajs, 1, 1)

    # control inputs in m/s and rad/s
    controls_front, _ = generate_control_inputs(n_trajs=num_trajs // 2,
                                                v_range=(vel_max / 2, vel_max), w_range=(-omega_max, omega_max),
                                                time_horizon=T, dt=dt)
    controls_back, _ = generate_control_inputs(n_trajs=num_trajs // 2,
                                               v_range=(-vel_max, -vel_max / 2), w_range=(-omega_max, omega_max),
                                               time_horizon=T, dt=dt)
    controls = torch.cat([controls_front, controls_back], dim=0)
    controls = torch.as_tensor(controls, dtype=torch.float32, device=device)

    # initial state
    state0 = (x, xd, R, omega)

    # put tensors to device
    state0 = tuple([s.to(device) for s in state0])
    z_grid = z_grid.to(device)
    controls = controls.to(device)

    # simulate the rigid body dynamics
    with torch.no_grad():
        t0 = time()
        states, forces = dphysics(z_grid=z_grid, controls=controls, state=state0)
        t1 = time()
        Xs, Xds, Rs, Omegas = states
        print(f'Simulation of {num_trajs} trajs (T={T} [sec] long) took {(t1-t0):.3f} [sec] on device: {device}')

    # visualize
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # plot heightmap
    ax.plot_surface(x_grid[0].cpu().numpy(), y_grid[0].cpu().numpy(), z_grid[0].cpu().numpy(), alpha=0.6, cmap='terrain')
    set_axes_equal(ax)
    for i in range(num_trajs):
        X = Xs[i].cpu().numpy()
        ax.plot(X[:, 0], X[:, 1], X[:, 2], label=f'Traj {i}', c='g')
    ax.set_title(f'Simulation of {num_trajs} trajs (T={T} [sec] long) took {(t1-t0):.3f} [sec] on device: {device}')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    plt.show()


def main():
    shoot_multiple()


if __name__ == '__main__':
    main()
