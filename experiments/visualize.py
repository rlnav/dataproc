#!/usr/bin/env python

import os
from monoforce.datasets import rough_seq_paths, ROUGH
from monoforce.utils import explore_data
import matplotlib as mpl
mpl.use('Qt5Agg')

def visualize_rough():
    # paths = rough_seq_paths
    paths = [
        '/media/ruslan/VRAS-DATA 4TB 2/datasets/ROUGH/marv_2025-03-19-14-41-19/'
    ]
    for path in paths:
        assert os.path.isdir(path), 'Data path %s does not exist' % path
        ds = ROUGH(path, is_train=False)
        # ds.get_global_cloud(vis=True, cached=False, step=10)
        explore_data(ds, sample_range='random', save=False)


def main():
    visualize_rough()


if __name__ == '__main__':
    main()
