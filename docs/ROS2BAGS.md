# ROS1 -> ROS2 bags

How to convert ROS1 bag files to ROS2 format.

1. Input: `*.bag` (ROS1 format), output: `*.db3` and `metadata.yaml` (ROS2 `sqlite3` format). The [rosbags](https://ternaris.gitlab.io/rosbags/) tools is required.
    ```bash
    pip install rosbags
    rosbags-convert --src marv_2025-03-19-15-35-24.bag --dst marv_2025-03-19-15-35-24
    ```
    The output-folder `marv_2025-03-19-15-35-24` should contain:
    ```bash
    ls marv_2025-03-19-15-35-24
    marv_2025-03-19-15-35-24.db3  metadata.yaml
    ```
2. Convert from `sqlite3` format (ROS2) to `mcap` format (ROS2).
   - create a yaml-file `out.yaml` describing the output bag format with the following content:
   ```aiignore
   ---
   output_bags:
   - uri: marv_2025-03-19-15-35-24_mcap
     storage_id: mcap
     all: true
     compression_mode: file
     compression_format: zstd
   ```
   - perform the conversion to ROS2 [MCAP](https://mcap.dev/) format:
   ```bash
   ros2 bag convert -i marv_2025-03-19-15-35-24 -o out.yaml
   ```
3. As a result you should have the folder `marv_2025-03-19-15-35-24_mcap` generated with the following content:
    ```bash
    ls marv_2025-03-19-15-35-24_mcap
    marv_2025-03-19-15-35-24_mcap_0.mcap.zstd  metadata.yaml
    ```
    Make sure the correction was correct:
    ```bash
    ros2 bag info marv_2025-03-19-15-35-24_mcap 
    ```