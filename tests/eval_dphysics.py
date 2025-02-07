import sys
sys.path.append('../../monoforce/monoforce/src/')
from time import time
import numpy as np
from monoforce.models.traj_predictor.dphysics import DPhysics, DPhysConfig
import torch


def main():
    # shoot_multiple()
    plot_results()


def shoot_multiple():
    # simulation parameters
    dphys_cfg = DPhysConfig()
    dphys_cfg.use_odeint = False

    num_trajs = 2048
    n_sims = 5
    # for device in ['cpu', 'cuda']:
    for device in ['cpu']:
        for T in [1., 2., 4., 6., 8., 10.]:
            dphys_cfg.traj_sim_time = T
            num_tsteps = int(T / dphys_cfg.dt)

            # instantiate the simulator
            dphysics = DPhysics(dphys_cfg, device=device)

            # terrain properties
            x_grid, y_grid = dphys_cfg.x_grid, dphys_cfg.y_grid
            z_grid = torch.zeros_like(x_grid)
            # repeat the heightmap for each rigid body
            z_grid = z_grid.repeat(num_trajs, 1, 1)

            # control inputs in m/s and rad/s
            controls = torch.ones((num_trajs, num_tsteps, 2))

            # put tensors to device
            z_grid = z_grid.to(device)
            controls = controls.to(device)

            # simulate the rigid body dynamics
            with torch.no_grad():
                t_avg = 0.
                for _ in range(n_sims):
                    t0 = time()
                    dphysics(z_grid=z_grid, controls=controls, vis=False)
                    t1 = time()
                    t_avg += (t1 - t0)
                t_avg /= n_sims
                print(f'N trajs: {num_trajs}. Traj horizon={T:.1f} [sec]. Simulation took {t_avg:.3f} [sec] on device: {device}')


def plot_results():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.use('TkAgg')
    mpl.rcParams['font.size'] = 26

    n_trajs = 2048
    dphys_runtimes = {
        'odeint':
            {
                'GPU': [0.415, 0.749, 1.444, 2.133, 2.834, 3.557],
                # 'CPU': [7.633, 14.713, 29.078, 42.766, 54.415, 71.97]  # 2048 trajs
            },
        'auto-diff':
            {
                'GPU': [0.438, 0.795, 1.519, 2.247, 2.986, 3.749],
                # 'CPU': [6.403, 12.753, 25.162, 70.441, 151.082, np.nan]  # 2048 trajs
            }
    }

    # n_trajs = 512
    # dphys_runtimes = {
    #     'odeint':
    #         {
    #             'GPU': [0.338, 0.673, 1.288, 1.938, 2.551, 3.164],
    #             'CPU': [1.565, 3.006, 6.100, 8.874, 11.877, 14.304]
    #         },
    #     'auto-diff':
    #         {
    #             'GPU': [0.439, 0.719, 1.393, 2.07, 3.061, 3.931],
    #             'CPU': [1.633, 3.105, 6.02, 9.159, 12.604, 16.059]
    #         }
    # }

    traj_horizons = [1., 2., 4., 6., 8., 10.]

    plt.figure(figsize=(10, 8))
    plt.plot(traj_horizons, 1000. * np.asarray(dphys_runtimes['odeint']['GPU']) / n_trajs,
            'k-', marker='o', label='Neural ODE (GPU)')
    # plt.plot(traj_horizons, 1000. * np.asarray(dphys_runtimes['odeint']['CPU']) / n_trajs,
    #          'b-', marker='o', label='Neural ODE (CPU)')
    plt.plot(traj_horizons, 1000. * np.asarray(dphys_runtimes['auto-diff']['GPU']) / n_trajs,
             'k--', marker='x', label='Auto-diff (GPU)')
    # plt.plot(traj_horizons, 1000. * np.asarray(dphys_runtimes['auto-diff']['CPU']) / n_trajs,
    #          'b--', marker='x', label='Auto-diff (CPU)')
    plt.xlabel('Trajectory horizon [sec]')
    plt.ylabel('Runtime [msec] / trajectory')
    plt.title(f'Physics Engine Runtime')
    plt.grid()
    plt.legend()
    plt.xticks(traj_horizons)
    plt.yticks(np.arange(0, 4.1, 0.5))
    plt.savefig('/home/ruslan/Desktop/dphys_runtime.png')
    plt.show()


if __name__ == '__main__':
    main()