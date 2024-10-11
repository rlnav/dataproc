#!/bin/bash

#DATA_PATH=/mnt/personal/agishrus/data/datasets
#DATA_PATH=/media/ruslan/data/datasets
DATA_PATH=/home/ruslan/Desktop/outdoor_dataset/

SEQUENCES=(
            '24-10-05-pokor-petrin/marv_2024-10-05-12-34-53'
            '24-10-05-pokor-petrin/marv_2024-10-05-12-35-00'
            '24-10-05-pokor-petrin/marv_2024-10-05-13-01-40'
            '24-10-05-pokor-petrin/marv_2024-10-05-13-17-08'
            '24-10-05-pokor-petrin/marv_2024-10-05-13-29-39'
            '24-10-05-pokor-petrin/marv_2024-10-05-13-43-21'
            '24-10-05-pokor-petrin/marv_2024-10-05-13-57-57'
            '24-10-05-pokor-petrin/marv_2024-10-05-14-12-29'
            '24-10-05-pokor-petrin/marv_2024-10-05-14-22-10'
            '24-10-05-pokor-petrin/marv_2024-10-05-14-28-15'
)

# shellcheck disable=SC2068
for SEQ in ${SEQUENCES[@]};
do
  IMG_PATH=$DATA_PATH/$SEQ/'images'
  echo "Processing images path $IMG_PATH"
  python ../packages/SEEM/demo_code/app.py --imgs-path $IMG_PATH
done
echo "Done saving semantic pseudolabels."

