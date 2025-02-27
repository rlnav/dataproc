import matplotlib.pyplot as plt
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
    for col in df.columns[1:]:
        plt.figure()
        plt.hist(df[col].values, color='b', edgecolor='k', alpha=0.7)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid()

    # random index of a sample with certain loss value
    idx = df.loc[df['H_t loss'] < 0.002].sample().index[0]
    print(f"idx: {idx}, loss: {df['H_t loss'].values[idx]}")

    fig = plt.imread(path + f'{idx:04d}.png')
    plt.figure()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.imshow(fig)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()