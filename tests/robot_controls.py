import os
import torch
from monoforce.datasets import ROUGH, rough_seq_paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')


def main():
    plt.figure(figsize=(20, 5))
    for seq_i in range(len(rough_seq_paths)):
        ds = ROUGH(path=rough_seq_paths[seq_i], is_train=False)
        for sample_i in tqdm(range(len(ds))):
            traj_ts, states = ds.get_states_traj(sample_i)
            ts, controls = ds.get_controls(sample_i)
            Xs = states[0]
            print('Xs shape:', Xs.shape)
            print('Traj dt:', torch.diff(traj_ts).mean())

            # visualization
            plt.clf()

            plt.subplot(141)
            plt.title('Trajectory Z(t)')
            plt.plot(traj_ts, Xs[:, 2], '.b')
            plt.xlabel('Time [s]')
            plt.ylabel('Z [m]')
            plt.grid()
            plt.ylim(-1, 1)
            plt.xlim(-0.1, 5.1)

            plt.subplot(142)
            plt.title('Trajectory Y(X)')
            plt.plot(Xs[:, 0], Xs[:, 1], '.b')
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')
            plt.grid()
            plt.ylim(-6.4, 6.4)
            plt.xlim(-6.4, 6.4)

            plt.subplot(143)
            plt.title('Controls: V(t)')
            plt.plot(ts, controls[:, 0], '.b')
            plt.xlabel('Time [s]')
            plt.ylabel('V [m/s]')
            plt.ylim(-1.1, 1.1)
            plt.grid()

            plt.subplot(144)
            plt.title('Controls: Omega(t)')
            plt.plot(ts, controls[:, 1], '.b')
            plt.xlabel('Time [s]')
            plt.ylabel('Omega [rad/s]')
            plt.ylim(-1.6, 1.6)
            plt.grid()

            plt.pause(0.1)
            plt.draw()

            os.makedirs('./gen/control_tests', exist_ok=True)
            plt.savefig(f'./gen/control_tests/seq_{seq_i}_sample_{sample_i}.png')


if __name__ == '__main__':
    main()
