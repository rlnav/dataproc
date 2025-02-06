from monoforce.datasets.rough import ROUGH, rough_seq_paths
from monoforce.models.dphysics import DPhysics
import matplotlib.pyplot as plt
import matplotlib as mpl
from monoforce.losses import physics_loss
import torch
mpl.use('Qt5Agg')


def physics_loss_test():
    # N_gt = 10
    # gt_ts = torch.linspace(0, 1.2, N_gt)
    # X_gt = torch.stack([
    #     gt_ts,
    #     torch.sin(2 * 3.14 * gt_ts),
    #     torch.zeros(N_gt)
    # ], dim=1)
    #
    # N_pred = 100
    # pred_ts = torch.linspace(0, 0.9, N_pred)
    # X_pred = torch.stack([
    #     pred_ts,
    #     torch.sin(2 * 3.14 * pred_ts),
    #     torch.zeros(N_pred)
    # ], dim=1) #+ 0.01 * torch.randn(N_pred, 3)

    ds = ROUGH(path=rough_seq_paths[0])
    # sample_i = np.random.choice(len(ds))
    sample_i = 120
    sample = ds[sample_i]
    (imgs, rots, trans, intrins, post_rots, post_trans,
     hm_geom, hm_terrain,
     control_ts, controls,
     pose0,
     traj_ts, Xs, Xds, Rs, Omegas) = sample
    X_gt= Xs
    gt_ts = traj_ts
    pred_ts = control_ts

    dphysics = DPhysics(device='cpu')
    z_grid = torch.zeros_like(hm_terrain[0])
    friction = 0.7 * torch.ones_like(z_grid)
    states_pred, _ = dphysics(z_grid=z_grid[None], controls=controls[None], friction=friction[None])
    X_pred = states_pred[0].squeeze(0)

    loss = physics_loss([X_pred[None]], [X_gt[None]], pred_ts[None], gt_ts[None])
    ts_ids = torch.argmin(torch.abs(pred_ts[None].unsqueeze(1) - gt_ts[None].unsqueeze(2)), dim=2)
    X_pred_gt_ts = X_pred[None][torch.arange(X_gt[None].shape[0]).unsqueeze(1), ts_ids].squeeze(0)
    print('Physics loss:', loss.item())
    assert X_pred_gt_ts.shape == X_gt.shape, f'X_pred_gt_ts shape: {X_pred_gt_ts.shape}, X_gt shape: {X_gt.shape}'

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.title(f'Physics loss: {loss.item()}')
    plt.plot(gt_ts, X_gt[..., 2], 'kx', label='Ground truth')
    plt.plot(pred_ts, X_pred[..., 2], 'r.', label='Prediction')
    plt.plot(gt_ts, X_pred_gt_ts[..., 2], 'g.', label='Prediction at GT ts')
    # plot corresponding lines
    for i in range(0, len(X_gt), 5):
        plt.plot([gt_ts[i], gt_ts[i]], [X_gt[i, 2], X_pred_gt_ts[i, 2]], 'b-', alpha=0.5)
    plt.legend()
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('States: Z (t)')

    plt.subplot(1, 2, 2)
    plt.xlabel('States: X')
    plt.ylabel('States: Y')
    plt.plot(X_gt[..., 0], X_gt[..., 1], 'kx', label='Ground truth')
    plt.plot(X_pred[..., 0], X_pred[..., 1], 'r.', label='Prediction')
    plt.plot(X_pred_gt_ts[..., 0], X_pred_gt_ts[..., 1], 'g.', label='Prediction at GT ts')
    # plot corresponding lines
    for i in range(0, len(X_gt), 5):
        plt.plot([X_gt[i, 0], X_pred_gt_ts[i, 0]], [X_gt[i, 1], X_pred_gt_ts[i, 1]], 'b-', alpha=0.5)
    plt.legend()
    plt.grid()
    plt.axis('equal')

    plt.show()


if __name__ == '__main__':
    physics_loss_test()
