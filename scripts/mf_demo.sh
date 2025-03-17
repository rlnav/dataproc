#!/bin/bash

SEQ=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/data/ROUGH/ugv_2024-10-05-16-24-48
TERRAIN_ENCODERS=(lss)
TRAJ_PREDICTORS=(dphysics)
VIS=False

for TERRAIN_ENCODER in "${TERRAIN_ENCODERS[@]}"
do
  for TRAJ_PREDICTOR in "${TRAJ_PREDICTORS[@]}"
  do
    WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/${TERRAIN_ENCODER}/val.pth
    echo "Terrain encoder ${TERRAIN_ENCODER} with trajectory predictor ${TRAJ_PREDICTOR}..."
    ./mf_demo.py --terrain_encoder ${TERRAIN_ENCODER} \
                 --terrain_encoder_path ${WEIGHTS} \
                 --traj_predictor ${TRAJ_PREDICTOR} \
                 --seq ${SEQ} \
                 --vis ${VIS}
  done
done

echo "Done."
