#!/bin/bash

#DATA_PATH=/mnt/personal/agishrus/data/datasets
#DATA_PATH=/media/ruslan/data/datasets
DATA_PATH=/media/ruslan/VRAS-DATA\ 4TB\ 2/datasets/ROUGH-v2

SEQUENCES=($(ls "$DATA_PATH"))

# shellcheck disable=SC2068
for SEQ in ${SEQUENCES[@]};
do
  IMG_PATH="${DATA_PATH}"/$SEQ/images
  echo "Processing images path $IMG_PATH"
  python ../packages/SEEM/demo_code/app.py --imgs-path "$IMG_PATH"
done
echo "Done saving semantic pseudolabels."

