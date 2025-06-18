import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')


def main():
    colormap = {
        'LiftSplatShoot': 'red',
        'VoxelNet': 'green',
        'BEVFusion': 'blue',
    }

    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(15, 10), sharex=True, sharey=True)
    terrain_encoders = ['LiftSplatShoot', 'VoxelNet', 'BEVFusion']
    for i, terrain_encoder in enumerate(terrain_encoders):
        for traj_predictor in ['DPhysics']:
            path = f'/home/ruslan/workspaces/ros1/traversability_ws/src/fusionforce/fusionforce/scripts/gen/eval_val/{terrain_encoder}_{traj_predictor}/'
            df = pd.read_csv(path + 'losses.csv')

            # plot histogram of losses
            for j, col in enumerate(df.columns[1:]):
                errs = df[col].values
                metric = col.replace(' loss', '')
                if 'Rot' not in metric:
                    errs = np.sqrt(errs)  # convert loss to error
                mean, std = np.mean(errs), np.std(errs)
                print(f'{terrain_encoder}: {metric}: mean: {mean}, std: {std}')
                axs[i, j].hist(errs, edgecolor='k', alpha=0.6, bins=10, color=colormap[terrain_encoder])
                meas = '[rad]' if 'rot' in col.lower() else '[m]'
                axs[i, j].set_xlabel(f'{col.replace('loss', 'err')} {meas}')
                axs[i, j].set_ylabel('N samples')
                axs[i, j].grid()
            print('\n' + '-' * 50 + '\n')

    # # random index of a sample with certain loss value
    # idx = df.loc[(df['H_t loss'] < 0.01)].sample().index[0]
    # loss_geom = df['H_g loss'].values[idx]
    # loss_terrain = df['H_t loss'].values[idx]
    # loss_xyz = df['XYZ loss'].values[idx]
    # loss_rot = df['Rot loss'].values[idx]
    #
    # fig = plt.imread(path + f'{idx:04d}.png')
    # plt.figure(figsize=(20, 10))
    # plt.title(f"idx: {idx}, geom_err: {np.sqrt(loss_geom):.3f}[m], trrain_err: {np.sqrt(loss_terrain):.3f}[m],"
    #           f"xyz_err: {np.sqrt(loss_xyz):.3f}[m], rot_err: {np.sqrt(loss_rot):.3f}[m]")
    # plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # plt.imshow(fig)
    # plt.axis('off')
    #
    plt.show()


def plot_xyz_error_histogram():
    # figure font size
    mpl.rcParams['font.size'] = 28

    categories = ['RGB', 'Points', 'RGB + Points']
    column1 = [0.043, 0.042, 0.045]
    column2 = [0.128, 0.129, 0.125]

    # Set bar width and positions
    x = np.arange(len(categories))
    width = 0.25

    plt.figure(figsize=(10, 8))
    plt.bar(x - width, column1, width, label=r'$\nabla$Physics')
    plt.bar(x, column2, width, label='TrajLSTM')
    # plt.xlabel('Sensor Input')
    plt.ylabel('Translational Error [m]')
    plt.xticks(x, categories)
    plt.ylim([0, 0.20])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_rot_error_histogram():
    # figure font size
    mpl.rcParams['font.size'] = 28

    categories = ['RGB', 'Points', 'RGB + Points']
    column1 = [2.013, 2.123, 2.254]
    column2 = [3.494, 3.369, 3.067]

    # Set bar width and positions
    x = np.arange(len(categories))
    width = 0.25

    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.bar(x - width, column1, width, label=r'$\nabla$Physics')
    plt.bar(x, column2, width, label='TrajLSTM')

    # Add labels, legend, and title
    # plt.xlabel('Sensor Input')
    plt.ylabel('Rotation Error [deg]')
    plt.xticks(x, categories)
    plt.ylim([0, 5])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # main()
    plot_xyz_error_histogram()
    plot_rot_error_histogram()