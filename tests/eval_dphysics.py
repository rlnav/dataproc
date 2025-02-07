import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
mpl.rcParams['font.size'] = 18


def main():
    n_trajs = 512
    traj_horizons = [1., 2., 4., 6., 8., 10.]
    dphys_runtimes = {
        'odeint':
            {
                'GPU': [0.338, 0.673, 1.288, 1.938, 2.551, 3.164],
                'CPU': [1.565, 3.006, 6.100, 8.874, 11.877, 14.304]
            },
        'auto-diff':
            {
                'GPU': [0.439, 0.719, 1.393, 2.07, 3.061, 3.931],
                'CPU': [1.633, 3.105, 6.02, 9.159, 12.604, 16.059]
            }
    }

    plt.figure(figsize=(10, 8))
    plt.plot(traj_horizons, dphys_runtimes['odeint']['GPU'], 'k-', marker='o', label='Nueral ODE (GPU)')
    plt.plot(traj_horizons, dphys_runtimes['odeint']['CPU'], 'b-', marker='o', label='Neural ODE (CPU)')
    plt.plot(traj_horizons, dphys_runtimes['auto-diff']['GPU'], 'k--', marker='x', label='Auto-diff (GPU)')
    plt.plot(traj_horizons, dphys_runtimes['auto-diff']['CPU'], 'b--', marker='x', label='Auto-diff (CPU)')
    plt.xlabel('Trajectory horizon [sec]')
    plt.ylabel('Runtime [sec]')
    plt.title(f'Physics Engine Runtime:\n{n_trajs} simulated trajectories')
    plt.grid()
    plt.legend()
    plt.xticks(traj_horizons)
    plt.yticks(range(0, 16, 2))
    plt.savefig('/home/ruslan/Desktop/dphys_runtime.png')
    plt.show()


if __name__ == '__main__':
    main()