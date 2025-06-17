# Data Processing

Data processing tools from bag files to data sequences.
The dataset is used to train
[FusionForce](https://github.com/ctu-vras/fusionforce) traversability estimation models.

## Usage

Make sure to adjust the paths and data topics.

- To save lidar clouds, corresponding camera images, and calibration (extrinsics and intrinsics) from a bag file:
    ```commandline
    ros2 launch dataproc save_data.launch bag_path:=/path/to/bag output_path:=/path/to/save/data
    ```

- To save control inputs:
    ```commandline
    cd ./scripts/
    ./add_cmd_vels
    ```

- RGB data anonymization using the [Deface](https://github.com/ORB-HD/deface) package:
    ```commandline
    cd ./scripts/
    ./blur_faces.sh
    ```

- Save semantic pseudo labels using the [SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once) model:
    ```commandline
    cd ./scripts/
    ./save_semantic_pseudolabels.sh
    ```
  
- Save semantic pseudo labels using the [WildScenes](https://github.com/csiro-robotics/WildScenes) models:
    ```commandline
    cd ./scripts/
    ./save_semantic_pseudolabels_wildscenes
    ```
  The script is based on the [mmsegmentation](https://mmsegmentation.readthedocs.io/en/main/user_guides/3_inference.html)
  inference tutorial. Please make sure to [install](https://github.com/csiro-robotics/WildScenes/blob/main/installation.md) the required dependencies
  and download the WildScenes [pretrained models](https://github.com/csiro-robotics/WildScenes/tree/main?tab=readme-ov-file#trained-models).
  
- Save localization (lidar poses). The [norlab_icp_mapper](https://github.com/norlab-ulaval/norlab_icp_mapper) SLAM was used to obtain the poses:
    ```commandline
    cd ./scripts/
    ./add_lidar_poses
    ```