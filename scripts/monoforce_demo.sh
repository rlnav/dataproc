#!/bin/bash

SEQ=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/data/ROUGH/24-08-14-monoforce-long_drive
TERRAIN_ENCODERS=(lss)
VIS=False

for TERRAIN_ENCODER in "${TERRAIN_ENCODERS[@]}"
do
  WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/${TERRAIN_ENCODER}/val.pth
  echo "Terrain encoder ${TERRAIN_ENCODER} with trajectory predictor ${TRAJ_PREDICTOR}..."
  python monoforce_demo.py --terrain_encoder ${TERRAIN_ENCODER} \
                           --terrain_encoder_path ${WEIGHTS} \
                           --seq ${SEQ} \
                           --vis ${VIS}
done
echo "Done."
