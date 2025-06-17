#!/usr/bin/env python3

import cv2
import numpy as np

import rclpy
import rclpy.time
from rclpy.executors import ExternalShutdownException
from rclpy.impl.logging_severity import LoggingSeverity
from rclpy.node import Node

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
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

        self.img_topics = self.get_parameter('img_topics').get_parameter_value().string_array_value
        self.camera_info_topics = self.get_parameter('camera_info_topics').get_parameter_value().string_array_value
        self.point_cloud_topic = self.get_parameter('point_cloud_topic').get_parameter_value().string_value

        self.max_msgs_delay = self.get_parameter('max_msgs_delay').get_parameter_value().double_value
        self.period = float(self.get_parameter('period').get_parameter_value().double_value)

        self.cv_bridge = CvBridge()
        self._tf_buffer = tf2_ros.Buffer()
        self._listener = tf2_ros.TransformListener(self._tf_buffer, self)
        self.prev_time = self.get_clock().now()

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

        for msg in msgs:
            if isinstance(msg, Image):
                cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                self._logger.debug(f'Image shape: {cv_image.shape}, dtype: {cv_image.dtype}')
            elif isinstance(msg, CameraInfo):
                self._logger.debug(f'CameraInfo K matrix: {msg.k}')
            elif isinstance(msg, PointCloud2):
                self._logger.debug(f'PointCloud2 with {msg.width * msg.height} points')


def main(args=None):
    rclpy.init(args=args)
    node = DataProcessor()
    node.start()
    node.spin()


if __name__ == '__main__':
    main()
