#!/usr/bin/env python3

import os
import cv2
import numpy as np

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

        self.img_topics = self.get_parameter('img_topics').get_parameter_value().string_array_value
        self.camera_info_topics = self.get_parameter('camera_info_topics').get_parameter_value().string_array_value
        self.point_cloud_topic = self.get_parameter('point_cloud_topic').get_parameter_value().string_value

        self.max_msgs_delay = self.get_parameter('max_msgs_delay').get_parameter_value().double_value
        self.period = float(self.get_parameter('period').get_parameter_value().double_value)

        self.cv_bridge = CvBridge()
        self._tf_buffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.prev_time = self.get_clock().now()

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

    def safe_lookup_transform(self, target_frame, source_frame, time):
        try:
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                time
            )
        except tf2_ros.ExtrapolationException:
            return self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time()
            )

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

        stamp = msgs[0].header.stamp

        # TODO: do not assume a fixed order of messages:
        # msgs[0] is left image, msgs[1] is right image, msgs[2] is depth image
        # msgs[3] is left camera info, msgs[4] is right camera info, msgs[5] is depth camera info
        # msgs[6] is point cloud
        imgL = self.cv_bridge.imgmsg_to_cv2(msgs[0], desired_encoding='passthrough')
        imgR = self.cv_bridge.imgmsg_to_cv2(msgs[1], desired_encoding='passthrough')
        depth_img = self.cv_bridge.imgmsg_to_cv2(msgs[2], desired_encoding='passthrough')
        self._logger.debug(f'imgL.shape = {imgL.shape}')
        self._logger.debug(f'imgR.shape = {imgR.shape}')
        self._logger.debug(f'depth_img.shape = {depth_img.shape}')

        points = pc2.read_points_numpy(msgs[6], skip_nans=False)
        self._logger.debug(f'PointCloud2 has {points.shape} shape')

        # save data
        ind = f'{stamp.sec:010d}_{stamp.nanosec:09d}'
        imgL_filename = os.path.join(self.img_path, 'left', f'{ind}.png')
        cv2.imwrite(imgL_filename, imgL)
        imgR_filename = os.path.join(self.img_path, 'right', f'{ind}.png')
        cv2.imwrite(imgR_filename, imgR)
        depth_filename = os.path.join(self.depth_path, f'{ind}.png')
        cv2.imwrite(depth_filename, depth_img)

        cloud_filename = os.path.join(self.cloud_path, f'{ind}.npz')
        np.savez(cloud_filename, points=points)


def main(args=None):
    rclpy.init(args=args)
    node = DataProcessor()
    node.start()
    node.spin()


if __name__ == '__main__':
    main()
