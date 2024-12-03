#!/usr/bin/env python

import os
from monoforce.datasets import rough_seq_paths, ROUGH
from monoforce.utils import explore_data
import matplotlib as mpl
mpl.use('TkAgg')

def visualize():
    # ids = [120, 294, 532, 573, 926, 2620]
    for path in rough_seq_paths:
        assert os.path.isdir(path), 'Data path %s does not exist' % path
        ds = ROUGH(path, is_train=False)
        explore_data(ds, sample_range='random', save=False)
        # for sample in tqdm(ds):
        #     pass

def main():
    visualize()


if __name__ == '__main__':
    main()
