import pytest
import os
from monoforce.datasets.rough import ROUGH, rough_seq_paths
import numpy as np


paths = rough_seq_paths

def test_sequences_exist():
    for path in paths:
        assert os.path.isdir(path), f'Data path {path} does not exist'

        ds = ROUGH(path, is_train=False)
        assert len(ds) > 0, f'No sequences found in {path}'

        assert os.path.exists(ds.controls_path), f'Controls path {ds.controls_path} does not exist'
        assert os.path.exists(ds.poses_path), f'Poses path {ds.poses_path} does not exist'
        assert os.path.exists(ds.calib_path), f'Calibration path {ds.calib_path} does not exist'
        imgs_path = os.path.join(path, 'images')
        assert os.path.exists(imgs_path), f'Images path {imgs_path} does not exist'
        assert os.path.exists(ds.cloud_path), f'Clouds path {ds.cloud_path} does not exist'
        img_files = [f for f in os.listdir(imgs_path) if f.endswith('.png')]
        assert len(img_files) > 0, f'No images found in {imgs_path}'
        # print(f'Found {len(img_files)} images in {imgs_path}')
        seg_path = os.path.join(imgs_path, 'wildscenes_seg/seg/')
        assert os.path.exists(seg_path), f'Segmentation path {seg_path} does not exist'
        seg_files = [f for f in os.listdir(seg_path) if f.endswith('.png')]
        assert len(seg_files) == len(img_files), f'Number of segmentation files does not match number of images'


def are_overlapping_stamps(ts1, ts2):
    # make sure timestamps are increasing
    assert np.all(np.diff(ts1) > 0), 'Timestamps are not increasing'
    assert np.all(np.diff(ts2) > 0), 'Timestamps are not increasing'

    # make sure pose timestamps are within control timestamps
    u_min, u_max = ts1.min(), ts1.max()
    p_min, p_max = ts2.min(), ts2.max()
    MIN = max(u_min, p_min)
    MAX = min(u_max, p_max)
    assert MIN <= MAX, f'There is no overlap between timestamps'


def test_timestamps():
    for path in paths:
        ds = ROUGH(path, is_train=False)
        u_ts, _ = ds.get_all_controls()
        assert len(u_ts), f'No timestamps found in {path}'
        p_ts, _ = ds.get_all_poses(return_stamps=True)
        assert len(p_ts), f'No timestamps found in {path}'

        are_overlapping_stamps(u_ts, p_ts)

        # # find the closest control timestamp for each pose timestamp
        # u_ts = u_ts[(u_ts >= MIN) & (u_ts <= MAX)]
        # p_ts = p_ts[(p_ts >= MIN) & (p_ts <= MAX)]
        # idx = np.searchsorted(u_ts, p_ts)
        # idx = np.clip(idx, 0, len(u_ts) - 1)
        # diff = u_ts[idx] - p_ts
        # assert diff.max() <= ds.dphys_cfg.traj_sim_time,\
        #     f'Closest control timestamp is too far from pose timestamp, diff={diff.max()} [sec]'


def test_ids():
    for path in paths:
        ds = ROUGH(path, is_train=False)
        ids = ds.get_ids()
        assert len(ids), f'No ids found in {path}'
        assert len(ids) == len(set(ids)), f'Ids are not unique'

        cloud_stamps = np.asarray([ds.ind_to_stamp(i) for i in range(len(ids))])
        # make sure timestamps are increasing
        assert len(cloud_stamps) == len(ids), f'Cloud stamps are not unique'
        assert np.all(np.diff(cloud_stamps) > 0), 'Cloud timestamps are not increasing'
        control_ts, _ = ds.get_all_controls()
        assert len(control_ts), f'No timestamps found in {path}'

        are_overlapping_stamps(control_ts, cloud_stamps)
