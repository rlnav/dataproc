#!/bin/bash

# This script synchronizes data DATASETS from a remote server
DATA_PATH=/media/ruslan/SSD/data
#DATA_PATH=/home/ruslan/data

# list of DATASETS to process
DATASETS=(
            'datasets/RobinGas/'
#            'datasets/Rellis3D/'
)

USER_NAME=agishrus
SERVER=login3.rci.cvut.cz

# loop through DATASETS
# shellcheck disable=SC2068
for DS in ${DATASETS[@]};
do
    SOURCE_PATH=${USER_NAME}@$SERVER:/mnt/personal/agishrus/data/$DS
    TARGET_PATH=${DATA_PATH}/$DS
#    SOURCE_PATH=${DATA_PATH}/$DS
#    TARGET_PATH=${USER_NAME}@$SERVER:/mnt/personal/agishrus/data/$DS

#    # if target path does not exist, create it
#    if [ ! -d "$TARGET_PATH" ]; then
#        mkdir -p "$TARGET_PATH"
#    fi

    echo "Synchronizing from source path ${SOURCE_PATH}"
    echo "to target path $TARGET_PATH"

    # synchronize data
    rsync -r --progress --ignore-existing --exclude="*visuals*" --exclude="*.mp4" --exclude='*terrain*' --exclude='*.bag' "${SOURCE_PATH}" "${TARGET_PATH}"
done
echo "Done synchronizing data."
