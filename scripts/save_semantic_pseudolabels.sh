#!/bin/bash

DATA_PATH=/mnt/personal/agishrus/data

SEQUENCES=(
            'robingas/data/22-09-27-unhost/husky/husky_2022-09-27-15-01-44'
            'robingas/data/22-09-27-unhost/husky/husky_2022-09-27-10-33-15'
            'robingas/data/22-10-27-unhost-final-demo/husky_2022-10-27-15-33-57'
            'robingas/data/22-09-23-unhost/husky/husky_2022-09-23-12-38-31'
            'robingas/data/22-06-30-cimicky_haj/husky_2022-06-30-15-58-37'
            'robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34'
            'robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-16-37-03'
            'robingas/data/22-06-30-cimicky_haj/ugv_2022-06-30-11-30-57'
            'robingas/data/22-10-20-unhost/ugv_2022-10-20-13-58-22'
            'robingas/data/22-10-20-unhost/ugv_2022-10-20-14-05-42'
            'robingas/data/22-10-20-unhost/ugv_2022-10-20-14-30-57'
)

# shellcheck disable=SC2068
for SEQ in ${SEQUENCES[@]};
do
  IMG_PATH=$DATA_PATH/$SEQ/'images'
  python ../packages/SEEM/demo_code/app.py --imgs-path $IMG_PATH
done
echo "Done saving semantic pseudolabels."

