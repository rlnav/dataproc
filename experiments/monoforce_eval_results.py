import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('TkAgg')


def main():
    terrain_encoder = 'LiftSplatShoot'
    traj_predictor = 'DPhysics'
    semantics = ['seem', 'wildscenes']
    plt.figure(figsize=(20, 5))
    for sem in semantics:
        path = f'/home/ruslan/workspaces/traversability_ws/src/monoforce/monoforce/scripts/gen/eval_{sem}_semantics/marv_{terrain_encoder}_{traj_predictor}/'
        df = pd.read_csv(path + 'losses.csv')
        print(df.head())

        # plot histogram of losses
        for col in df.columns[1:]:
            plt.subplot(1, 4, list(df.columns).index(col))
            plt.hist(np.sqrt(df[col].values), edgecolor='k', alpha=0.6, label=sem)
            meas = '[rad]' if 'rot' in col.lower() else '[m]'
            plt.xlabel(f'{col.replace('loss', 'err')} {meas}')
            plt.ylabel('N samples')
            plt.legend()
            plt.grid()

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

    plt.show()


if __name__ == '__main__':
    main()