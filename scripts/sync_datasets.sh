#!/bin/bash

# This script synchronizes data sequences from a remote server
DATA_PATH=/media/ruslan/data
#DATA_PATH=/media/ruslan/SSD/data

# list of sequences to process
SEQUENCES=(
            'robingas/data/22-09-27-unhost/husky/husky_2022-09-27-15-01-44/'
            'robingas/data/22-09-27-unhost/husky/husky_2022-09-27-10-33-15/'
            'robingas/data/22-10-27-unhost-final-demo/husky_2022-10-27-15-33-57/'
            'robingas/data/22-09-23-unhost/husky/husky_2022-09-23-12-38-31/'
            'robingas/data/22-06-30-cimicky_haj/husky_2022-06-30-15-58-37/'
            'robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-15-18-34/'
            'robingas/data/22-08-12-cimicky_haj/marv/ugv_2022-08-12-16-37-03/'
            'robingas/data/22-10-20-unhost/ugv_2022-10-20-13-58-22/'
            'robingas/data/22-10-20-unhost/ugv_2022-10-20-14-05-42/'
            'robingas/data/22-10-20-unhost/ugv_2022-10-20-14-30-57/'
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
    echo "Synchronizing from source path ${SOURCE_PATH}"
    echo "to target path $TARGET_PATH"

    rsync -r --progress --ignore-existing --exclude='*.bag' ${SOURCE_PATH} ${TARGET_PATH}
done
echo "Done synchronizing data."
