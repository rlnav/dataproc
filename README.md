# Data Processing

Data processing tools from bag files to data sequences.
The dataset is used to train
[MonoForce](https://github.com/ctu-vras/monoforce) traversability estimation models.

The bag files are available at:
- [http://subtdata.felk.cvut.cz/outdoor-dataset/](http://subtdata.felk.cvut.cz/outdoor-dataset/)
- [http://subtdata.felk.cvut.cz/robingas/](http://subtdata.felk.cvut.cz/robingas/)

## Usage

Make sure to adjust the paths and data topics.

- To save lidar clouds, corresponding camera images, and calibration (extrinsics and intrinsics) from a bag file:
    ```commandline
    OUTPUT_PATH=/path/to/save/data/sequence
    roslaunch dataproc dataproc.launch output_path:=${OUTPUT_PATH} img_topics:=[] lidar_topics:=[] camera_info_topics:=[]
    ```

- To save control inputs:
    ```commandline
    cd ./scripts/
    ./add_controls
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
  
- Save localization (lidar poses). The [norlab_icp_mapper](https://github.com/norlab-ulaval/norlab_icp_mapper) SLAM was used to obtain the poses:
    ```commandline
    cd ./scripts/
    ./add_lidar_poses
    ```