# Data Proc

## Data generation

The bag files are available at [http://subtdata.felk.cvut.cz/robingas/data/](http://subtdata.felk.cvut.cz/robingas/data/).
In order to generate the traversability data from a prerecorded bag file, please run
(*note, that the topic names could be different depending on a bag file):

```commandline
cd ./scripts/data/
./create_dataset --bag-paths /path/to/data/sequence/<sequence-name>.bag \
                               /path/to/data/sequence/<sequence-name>_loc.bag \
                               --cloud-topics /os_cloud_node/destaggered_points \
                               --camera-topics /camera_rear/image_color/compressed \
                                               /camera_front/image_color/compressed \
                                               /camera_right/image_color/compressed \
                                               /camera_left/image_color/compressed \
                               --camera-info-topics /camera_front/camera_info \
                                                    /camera_rear/camera_info \
                                                    /camera_right/camera_info \
                                                    /camera_left/camera_info \
                               --robot-model 'Box()' --discard-model 'Box()' \
                               --input-step 50 --visualize False --save-data True
```
