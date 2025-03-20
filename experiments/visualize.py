#!/usr/bin/env python

import os
from monoforce.datasets import rough_seq_paths, ROUGH
from monoforce.utils import explore_data
import matplotlib as mpl
mpl.use('Qt5Agg')

def visualize_rough():
    # paths = rough_seq_paths
    paths = sorted([os.path.join('../data/ROUGH/', f) for f in os.listdir('../data/ROUGH/') if f.startswith('marv_2025-03-19-')])
        
    for path in paths:
        if 'marv_2025-03-19-15-03-51' in path:  # no slam poses
            continue
        if 'marv_2025-03-19-14-41-19' in path:  # no slam poses
            continue
        if 'marv_2025-03-19-14-45-10' in path:  # no controls
            continue
        if 'marv_2025-03-19-15-20-04' in path:  # no controls
            continue
        if 'marv_2025-03-19-15-33-00' in path:  # no slam poses
            continue
        if 'marv_2025-03-19-14-45-10' in path:
            print('Possible control/lidar time stamp issue')
            
        assert os.path.isdir(path), 'Data path %s does not exist' % path
        ds = ROUGH(path, is_train=False)
        # ds.get_global_cloud(vis=True, cached=False, step=10)
        explore_data(ds, sample_range='random', save=False)


def main():
    visualize_rough()


if __name__ == '__main__':
    main()
