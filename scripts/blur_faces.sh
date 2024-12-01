#!/bin/bash

DATA_PATH=/media/ruslan/VRAS-DATA\ 4TB\ 2/datasets/ROUGH-v2

# find all sequences in the data path
SEQUENCES=($(ls "$DATA_PATH"))

# shellcheck disable=SC2068
for SEQ in ${SEQUENCES[@]};
do
  IMGS_PATH="$DATA_PATH"/$SEQ/'images/'
  echo "Processing images path $IMGS_PATH"
  deface "$IMGS_PATH"
  rename -f 's/_anonymized//' "$IMGS_PATH/"*
done
echo "Done blurring faces."
