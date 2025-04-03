#!/bin/bash

SEQ=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/data/ROUGH/marv_2024-09-26-13-54-43/
TERRAIN_ENCODERS=(lss)
VIS=True

for TERRAIN_ENCODER in "${TERRAIN_ENCODERS[@]}"
do
#  WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/weights/${TERRAIN_ENCODER}/val.pth
  WEIGHTS=$HOME/workspaces/traversability_ws/src/monoforce/monoforce/config/tb_runs/rough/lss_2025_03_25_11_28_40/val.pth
  echo "Terrain encoder ${TERRAIN_ENCODER} with trajectory predictor ${TRAJ_PREDICTOR}..."
  python mf_demo.py --terrain_encoder ${TERRAIN_ENCODER} \
                    --terrain_encoder_path ${WEIGHTS} \
                    --seq ${SEQ} \
                    --vis ${VIS}
done
echo "Done."
