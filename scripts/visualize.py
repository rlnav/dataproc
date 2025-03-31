#!/usr/bin/env python

import os
from monoforce.datasets import rough_seq_paths, ROUGH
from monoforce.utils import explore_data
import matplotlib as mpl
try:
    mpl.use('Qt5Agg')
except:
    mpl.use('TkAgg')

def visualize_rough():
    paths = [f for f in rough_seq_paths if 'marv_2025-03-19-' in f]
    # paths = sorted([os.path.join('../data/ROUGH/', f) for f in os.listdir('../data/ROUGH/') if f.startswith('marv_2025-03-19-')])
        
    for path in paths:
        assert os.path.isdir(path), 'Data path %s does not exist' % path
        ds = ROUGH(path, is_train=False)
        explore_data(ds, sample_range='random', save=False)


def main():
    visualize_rough()


if __name__ == '__main__':
    main()
