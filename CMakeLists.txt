cmake_minimum_required(VERSION 3.0.2)
project(dataproc)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED)
catkin_python_setup()
catkin_package()

include_directories()

catkin_install_python(PROGRAMS
  scripts/bagmerge
  scripts/correct_camera_frames
  scripts/save_sensor_data
  nodes/play_tf_static
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
    launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)