#!/usr/bin/env python3

import os
import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

import rclpy
import rclpy.time
from rclpy.executors import ExternalShutdownException
from rclpy.impl.logging_severity import LoggingSeverity
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros


def append_to_yaml(yaml_path, data_dict):
    if not os.path.exists(yaml_path):
        with open(yaml_path, 'w') as f:
            yaml.dump(data_dict, f)
    else:
        with open(yaml_path, 'r') as f:
            print('Updating yaml file: %s' % yaml_path)
            cur_yaml = yaml.load(f, Loader=yaml.FullLoader)
            cur_yaml.update(data_dict)

        with open(yaml_path, 'w') as f:
            yaml.safe_dump(cur_yaml, f)  # Also note the safe_dump


class DataProcessor(Node):

    def __init__(self):
        super().__init__('depth_estimation_node')
        self._logger.set_level(LoggingSeverity.DEBUG)

        self.declare_parameter('img_topics',
                               ['/camera_left/image_raw', '/camera_right/image_raw', '/depth/image_raw'])
        self.declare_parameter('camera_info_topics',
                                 ['/camera_left/camera_info', '/camera_right/camera_info', '/depth/camera_info'])
        self.declare_parameter('point_cloud_topic', '/depth/points')
        self.declare_parameter('max_msgs_delay', 0.1)
        self.declare_parameter('period', 1.0)
        self.declare_parameter('output_path', './')
        self.declare_parameter('robot_frame', 'base_link')

        self.img_topics = self.get_parameter('img_topics').get_parameter_value().string_array_value
        self.camera_info_topics = self.get_parameter('camera_info_topics').get_parameter_value().string_array_value
        self.point_cloud_topic = self.get_parameter('point_cloud_topic').get_parameter_value().string_value

        self.max_msgs_delay = self.get_parameter('max_msgs_delay').get_parameter_value().double_value
        self.period = float(self.get_parameter('period').get_parameter_value().double_value)
        self.robot_frame = self.get_parameter('robot_frame').get_parameter_value().string_value

        self.output_path = self.get_parameter('output_path').get_parameter_value().string_value
        self.cloud_path = os.path.join(self.output_path, 'clouds')
        os.makedirs(self.cloud_path, exist_ok=True)
        self.img_path = os.path.join(self.output_path, 'images')
        os.makedirs(os.path.join(self.img_path, 'left'), exist_ok=True)
        os.makedirs(os.path.join(self.img_path, 'right'), exist_ok=True)
        self.depth_path = os.path.join(self.output_path, 'depth')
        os.makedirs(self.depth_path, exist_ok=True)
        self.calib_path = os.path.join(self.output_path, 'calibration')
        os.makedirs(self.calib_path, exist_ok=True)

        self.cv_bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.saved_calibration = False
        self.prev_time = self.get_clock().now()

    def get_transform(self, from_frame, to_frame, time=None):
        """
        Retrieve a transformation matrix between two frames using TF2.
        """
        if time is None:
            time = rclpy.time.Time()
        timeout = rclpy.time.Duration(seconds=1.0)
        try:
            tf = self.tf_buffer.lookup_transform(to_frame, from_frame,
                                                 time=time, timeout=timeout)
        except Exception as ex:
            tf = self.tf_buffer.lookup_transform(to_frame, from_frame,
                                                 time=rclpy.time.Time(), timeout=timeout)
            self._logger.warning(
                f"Could not find transform from {from_frame} to {to_frame} at time {time}, using latest available transform: {ex}"
            )
        # Convert TF2 transform message to a 4x4 transformation matrix
        translation = [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z]
        quat = [tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w]
        T = np.eye(4)
        R = Rotation.from_quat(quat).as_matrix()
        T[:3, 3] = translation
        T[:3, :3] = R
        return T

    def spin(self):
        try:
            rclpy.spin(self)
        except (KeyboardInterrupt, ExternalShutdownException):
            self.get_logger().info('Keyboard interrupt, shutting down...')
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def start(self):
        # subscribe to topics with approximate time synchronization
        subs = []
        for topic in self.img_topics:
            self._logger.info('Subscribing to %s' % topic)
            subs.append(Subscriber(self, Image, topic))
        for topic in self.camera_info_topics:
            self._logger.info('Subscribing to %s' % topic)
            subs.append(Subscriber(self, CameraInfo, topic))
        subs.append(Subscriber(self, PointCloud2, self.point_cloud_topic))
        sync = ApproximateTimeSynchronizer(subs, queue_size=10, slop=self.max_msgs_delay)
        sync.registerCallback(self.callback)

    def callback(self, *msgs):
        now = self.get_clock().now()
        if (now - self.prev_time).nanoseconds / 1e9 < self.period:  # 1 second = 1e9 ns
            return  # Skip this callback
        self.prev_time = now
        self._logger.debug('Received %d messages' % len(msgs))

        # TODO: do not assume a fixed order of messages:
        # msgs[0] is left image, msgs[1] is right image, msgs[2] is depth image
        # msgs[3] is left camera info, msgs[4] is right camera info, msgs[5] is depth camera info
        # msgs[6] is point cloud
        (imgL_msg, imgR_msg, depth_msg,
         left_camera_info_msg, right_camera_info_msg, depth_camera_info_msg,
         point_cloud_msg) = msgs

        imgL = self.cv_bridge.imgmsg_to_cv2(imgL_msg, desired_encoding='passthrough')
        imgR = self.cv_bridge.imgmsg_to_cv2(imgR_msg, desired_encoding='passthrough')
        depth_img = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        self._logger.debug(f'imgL.shape = {imgL.shape}')
        self._logger.debug(f'imgR.shape = {imgR.shape}')
        self._logger.debug(f'depth_img.shape = {depth_img.shape}')

        points = pc2.read_points_numpy(point_cloud_msg, skip_nans=False)
        self._logger.debug(f'PointCloud2 has {points.shape} shape')

        # save sensor data
        stamp = imgL_msg.header.stamp
        ind = f'{stamp.sec:010d}_{stamp.nanosec:09d}'

        imgL_filename = os.path.join(self.img_path, 'left', f'{ind}.png')
        cv2.imwrite(imgL_filename, imgL)
        imgR_filename = os.path.join(self.img_path, 'right', f'{ind}.png')
        cv2.imwrite(imgR_filename, imgR)

        depth_filename = os.path.join(self.depth_path, f'{ind}.png')
        cv2.imwrite(depth_filename, depth_img)

        cloud_filename = os.path.join(self.cloud_path, f'{ind}.npz')
        np.savez(cloud_filename, points=points)

        # save calibration data
        if not self.saved_calibration:
            Tr_camleft_robot = self.get_transform(imgL_msg.header.frame_id, self.robot_frame, time=stamp)
            yaml_data_dict = {'Tr_camleft_robot':
                                  {'rows': 4, 'cols': 4,
                                   'data': ['%.3f' % x for x in Tr_camleft_robot.flatten()]}}
            append_to_yaml(os.path.join(self.calib_path, 'transformations.yaml'), yaml_data_dict)

            Tr_camright_robot = self.get_transform(imgR_msg.header.frame_id, self.robot_frame, time=stamp)
            yaml_data_dict = {'Tr_camright_robot':
                                  {'rows': 4, 'cols': 4,
                                   'data': ['%.3f' % x for x in Tr_camright_robot.flatten()]}}
            append_to_yaml(os.path.join(self.calib_path, 'transformations.yaml'), yaml_data_dict)

            Tr_camdepth_robot = self.get_transform(depth_msg.header.frame_id, self.robot_frame, time=stamp)
            yaml_data_dict = {'Tr_camdepth_robot':
                                  {'rows': 4, 'cols': 4,
                                   'data': ['%.3f' % x for x in Tr_camdepth_robot.flatten()]}}
            append_to_yaml(os.path.join(self.calib_path, 'transformations.yaml'), yaml_data_dict)

            Tr_points_robot = self.get_transform(point_cloud_msg.header.frame_id, self.robot_frame, time=stamp)
            yaml_data_dict = {'Tr_points_robot':
                                  {'rows': 4, 'cols': 4,
                                   'data': ['%.3f' % x for x in Tr_points_robot.flatten()]}}
            append_to_yaml(os.path.join(self.calib_path, 'transformations.yaml'), yaml_data_dict)

            self._logger.debug('Saved calibration data to %s' % self.calib_path)

            self.saved_calibration = True


def main(args=None):
    rclpy.init(args=args)
    node = DataProcessor()
    node.start()
    node.spin()


if __name__ == '__main__':
    main()
