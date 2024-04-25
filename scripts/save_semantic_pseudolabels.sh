#!/bin/bash

DATA_PATH=/mnt/personal/agishrus/data/datasets
#DATA_PATH=/media/ruslan/data/datasets

SEQUENCES=(
            'RobinGas/husky_oru/radarize__2024-02-07-10-47-13_0'
            'RobinGas/husky_oru/radarize__2023-08-16-11-24-37_0'
            'RobinGas/husky_oru/radarize__2023-08-16-11-09-06_0'
            'RobinGas/husky_oru/radarize__2023-08-16-11-02-33_0'
            'RobinGas/husky_oru/radarize__2023-08-16-11-44-56_0'
            'RobinGas/husky_oru/radarize__2023-08-16-11-37-14_0'
            'RobinGas/husky_oru/radarize__2023-08-16-11-54-42_0'
            'Robingas/husky/husky_2022-09-27-15-01-44'
            'RobinGas/husky/husky_2022-09-27-10-33-15'
            'RobinGas/husky/husky_2022-10-27-15-33-57'
            'RobinGas/husky/husky_2022-09-23-12-38-31'
            'RobinGas/husky/husky_2022-06-30-15-58-37'
            'RobinGas/marv/ugv_2022-08-12-15-18-34'
            'RobinGas/marv/ugv_2022-08-12-16-37-03'
            'RobinGas/tradr/ugv_2022-06-30-11-30-57'
            'RobinGas/tradr/ugv_2022-10-20-13-58-22'
            'RobinGas/tradr/ugv_2022-10-20-14-05-42'
            'RobinGas/tradr/ugv_2022-10-20-14-30-57'
)

# shellcheck disable=SC2068
for SEQ in ${SEQUENCES[@]};
do
  IMG_PATH=$DATA_PATH/$SEQ/'images'
  echo "Processing images path $IMG_PATH"
  python ../packages/SEEM/demo_code/app.py --imgs-path $IMG_PATH
done
echo "Done saving semantic pseudolabels."

