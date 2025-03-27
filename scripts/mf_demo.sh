#!/bin/bash

SEQ=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/data/ROUGH/marv_2024-09-26-13-54-43/
TERRAIN_ENCODERS=(lss)
TRAJ_PREDICTORS=(dphysics)
VIS=True

for TERRAIN_ENCODER in "${TERRAIN_ENCODERS[@]}"
do
  for TRAJ_PREDICTOR in "${TRAJ_PREDICTORS[@]}"
  do
#    WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/${TERRAIN_ENCODER}/val.pth
    WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/tb_runs/rough/lss_2025_03_25_11_28_40/val.pth
    echo "Terrain encoder ${TERRAIN_ENCODER} with trajectory predictor ${TRAJ_PREDICTOR}..."
    ./mf_demo.py --terrain_encoder ${TERRAIN_ENCODER} \
                 --terrain_encoder_path ${WEIGHTS} \
                 --traj_predictor ${TRAJ_PREDICTOR} \
                 --seq ${SEQ} \
                 --vis ${VIS}
  done
done

echo "Done."
