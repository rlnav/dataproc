cmake_minimum_required(VERSION 3.0.2)
project(dataproc)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED)
catkin_python_setup()
catkin_package()

include_directories()

catkin_install_python(PROGRAMS
  nodes/dataproc
  nodes/play_tf_static
  nodes/cloud_proc
  nodes/gridmap_masker
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
    launch/
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)