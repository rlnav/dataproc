#!/bin/bash

SEQ=$HOME/workspaces/ros2/traversability_ws/src/fusionforce/fusionforce/data/ROUGH/marv_2024-09-26-13-54-43
TERRAIN_ENCODER=lss
VIS=True

WEIGHTS=$HOME/workspaces/ros2/traversability_ws/src/fusionforce/fusionforce/config/weights/${TERRAIN_ENCODER}/val.pth
echo "Terrain encoder ${TERRAIN_ENCODER} with trajectory predictor ${TRAJ_PREDICTOR}..."
python fusionforce_demo.py --terrain_encoder ${TERRAIN_ENCODER} \
                         --terrain_encoder_path ${WEIGHTS} \
                         --seq ${SEQ} \
                         --vis ${VIS}
echo "Done."
