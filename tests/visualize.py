#!/usr/bin/env python

import os
from monoforce.datasets import rough_seq_paths, ROUGH
from monoforce.utils import explore_data
import matplotlib as mpl
mpl.use('Qt5Agg')

def visualize_rough():
    for path in rough_seq_paths:
        assert os.path.isdir(path), 'Data path %s does not exist' % path
        ds = ROUGH(path, is_train=False)
        explore_data(ds, sample_range='random', save=False)


def visualize_wildscenes():
    from monoforce.datasets.wildscenes import WildScenes, wild_seq_names

    for seq in wild_seq_names:
        ds = WildScenes(seq, is_train=False)
        explore_data(ds, sample_range='random', save=False)


def main():
    visualize_rough()
    # visualize_wildscenes()


if __name__ == '__main__':
    main()
