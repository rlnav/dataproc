#!/bin/bash

# This script synchronizes data sequences from a remote server
#DATA_PATH=/media/ruslan/data/datasets
DATA_PATH=/media/ruslan/SSD/data/datasets

# list of sequences to process
SEQUENCES=(
            'RobinGas/husky_oru/radarize__2023-08-16-11-24-37_0/'
            'RobinGas/husky_oru/radarize__2024-02-07-10-47-13_0/'
            'RobinGas/husky/husky_2022-09-27-15-01-44/'
            'RobinGas/husky/husky_2022-09-27-10-33-15/'
            'RobinGas/huky/husky_2022-10-27-15-33-57/'
            'RobinGas/husky/husky_2022-09-23-12-38-31/'
            'RobinGas/huky/husky_2022-06-30-15-58-37/'
            'RobinGas/marv/ugv_2022-08-12-15-18-34/'
            'RobinGas/marv/ugv_2022-08-12-16-37-03/'
            'RobinGas/huky/ugv_2022-06-30-11-30-57/'
            'RobinGas/tradr/ugv_2022-10-20-13-58-22/'
            'RobinGas/tradr/ugv_2022-10-20-14-05-42/'
            'RobinGas/tradr/ugv_2022-10-20-14-30-57/'
)

USER_NAME=agishrus
SERVER=login3.rci.cvut.cz

# loop through sequences
# shellcheck disable=SC2068
for SEQ in ${SEQUENCES[@]};
do
    SOURCE_PATH=${USER_NAME}@$SERVER:/mnt/personal/agishrus/data/$SEQ
    TARGET_PATH=${DATA_PATH}/$SEQ
#    TARGET_PATH=${USER_NAME}@$SERVER:/mnt/personal/agishrus/data/$SEQ
#    SOURCE_PATH=${DATA_PATH}/$SEQ

    # if target path does not exist, create it
    if [ ! -d "$TARGET_PATH" ]; then
        mkdir -p $TARGET_PATH
    fi

    echo "Synchronizing from source path ${SOURCE_PATH}"
    echo "to target path $TARGET_PATH"

    rsync -r --progress --ignore-existing --exclude='*.bag' ${SOURCE_PATH} ${TARGET_PATH}
done
echo "Done synchronizing data."
