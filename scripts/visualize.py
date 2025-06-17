#!/usr/bin/env python

import os
from fusionforce.datasets import rough_seq_paths, ROUGH
from fusionforce.utils import explore_data
import matplotlib as mpl
try:
    mpl.use('Qt5Agg')
except:
    mpl.use('TkAgg')

def visualize_rough():
    paths = rough_seq_paths
    # paths = [f for f in rough_seq_paths if 'marv_2025-03-19-' in f]

    for path in paths:
        assert os.path.isdir(path), 'Data path %s does not exist' % path
        ds = ROUGH(path, is_train=False)
        explore_data(ds, sample_range='random', save=False)


def main():
    visualize_rough()


if __name__ == '__main__':
    main()
