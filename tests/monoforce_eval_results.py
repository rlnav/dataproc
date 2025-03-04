import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Qt5Agg')


def main():
    terrain_encoder = 'LiftSplatShoot'
    traj_predictor = 'DPhysics'
    path = f'/home/ruslan/workspaces/traversability_ws/src/monoforce/monoforce/scripts/gen/eval1/marv_{terrain_encoder}_{traj_predictor}/'
    df = pd.read_csv(path + 'losses.csv')
    print(df.head())

    # plot histogram of losses
    plt.figure(figsize=(10, 10))
    for col in df.columns[1:]:
        plt.subplot(2, 2, list(df.columns).index(col))
        plt.hist(df[col].values, color='b', edgecolor='k', alpha=0.7)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid()

    # random index of a sample with certain loss value
    idx = df.loc[(df['H_t loss'] < 0.001) & (df['H_g loss'] > 0.01)].sample().index[0]
    loss_geom = df['H_g loss'].values[idx]
    loss_terrain = df['H_t loss'].values[idx]
    loss_xyz = df['XYZ loss'].values[idx]
    loss_rot = df['Rot loss'].values[idx]

    fig = plt.imread(path + f'{idx:04d}.png')
    plt.figure(figsize=(20, 10))
    plt.title(f"idx: {idx}, geom_err: {np.sqrt(loss_geom):.3f}[m], trrain_err: {np.sqrt(loss_terrain):.3f}[m],"
              f"xyz_err: {np.sqrt(loss_xyz):.3f}[m], rot_err: {np.sqrt(loss_rot):.3f}[m]")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(fig)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()