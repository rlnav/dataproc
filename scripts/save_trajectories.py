import rospy
from sensor_msgs.msg import PointCloud2
from ros_numpy import numpify
from tf2_ros import BufferCore, TransformException
from rosbag import Bag, ROSBagException, Compression

import os
import numpy as np
from tqdm import tqdm
import open3d as o3d


def load_buffer(bag_paths):
    tf_topics = ['/tf', '/tf_static']
    # buffer = BufferCore(cache_time=rospy.Duration(2**31 - 1))
    # buffer = BufferCore(cache_time=rospy.Duration(24 * 60 * 60))
    buffer = BufferCore(rospy.Duration(24 * 60 * 60))
    for path in bag_paths:
        try:
            with Bag(path, 'r') as bag:
                for topic, msg, stamp in tqdm(bag.read_messages(topics=tf_topics),
                                              desc='%s: reading transforms' % path.split('/')[-1],
                                              total=bag.get_message_count(topic_filters=tf_topics)):
                    if topic == '/tf':
                        for tf in msg.transforms:
                            buffer.set_transform(tf, 'bag')
                    elif topic == '/tf_static':
                        for tf in msg.transforms:
                            buffer.set_transform_static(tf, 'bag')
        except ROSBagException as ex:
            print('Could not read %s: %s' % (path, ex))

    return buffer

def slots(msg):
    """Return message attributes (slots) as list."""
    return [getattr(msg, var) for var in msg.__slots__]

def get_point_cloud_in_window(bag, target_time, time_window, topic):
    """
    Retrieves the closest PointCloud2 message within a given time window around the target_time.

    :param bag: The opened rosbag file.
    :param target_time: Desired timestamp (in seconds).
    :param time_window: Allowed time window (in seconds) around target_time.
    :param topic: The ROS topic to search in.
    :return: The closest PointCloud2 message within the window, or None if no match is found.
    """
    closest_msg = None
    closest_time_diff = float('inf')

    lower_bound = rospy.Time.from_sec(target_time - time_window)
    upper_bound = rospy.Time.from_sec(target_time + time_window)

    for topic, msg, t in bag.read_messages(topics=[topic], start_time=lower_bound, end_time=upper_bound):
        t_sec = t.to_sec()
        time_diff = abs(t_sec - target_time)

        if time_diff < closest_time_diff:
            closest_time_diff = time_diff
            closest_msg = msg

        # Stop early if an exact match is found
        if time_diff == 0:
            break
    return closest_msg


def process_bag(bagfile, cloud_times):
    bag = Bag(bagfile, 'r')

    buffer = load_buffer([bagfile])

    time_horizon = [0., 10.]  # seconds
    time_search_window = 0.2  # seconds
    cloud_topic = "/points_filtered_kontron"
    fixed_frame = "odom"
    robot_frame = "base_link"
    time_step = 1.0
    n = [int(np.floor(h / time_step)) for h in time_horizon]

    for cloud_time in tqdm(cloud_times):
        pcd_msg = get_point_cloud_in_window(bag, cloud_time,
                                            time_window=time_search_window,
                                            topic=cloud_topic)
        if pcd_msg is None:
            print(f"No point cloud found for time {cloud_time}")
            continue
        pcd_msg = PointCloud2(*slots(pcd_msg))
        cloud = numpify(pcd_msg).flatten()
        nan_mask = np.isnan(cloud['x'])
        cloud = cloud[~nan_mask]
        points = np.stack([cloud['x'], cloud['y'], cloud['z']], axis=1)
        print(points.shape)

        # Find transform from input cloud to fixed frame.
        try:
            input_to_fixed = buffer.lookup_transform_core(fixed_frame, pcd_msg.header.frame_id, pcd_msg.header.stamp)
        except TransformException as ex:
            print('Could not transform from %s to %s at %.3f s.' % (
            pcd_msg.header.frame_id, fixed_frame, pcd_msg.header.stamp.to_sec()))
            continue
        input_to_fixed = numpify(input_to_fixed.transform)
        # print(input_to_fixed.shape)

        # Find transforms from input cloud to robot positions within the horizon.
        input_to_robot_tfs = []
        start = pcd_msg.header.stamp.to_sec()
        for t in np.linspace(start - n[0] * time_step, start + n[1] * time_step, sum(n) + 1):
            try:
                tf = buffer.lookup_transform_full_core(robot_frame, rospy.Time.from_seconds(t),
                                                       pcd_msg.header.frame_id, pcd_msg.header.stamp,
                                                       fixed_frame)
            except TransformException as ex:
                print('Could not transform from %s to %s at %.3f s.' % (pcd_msg.header.frame_id, robot_frame, t))
                continue
            tf = numpify(tf.transform)
            # print(tf.shape)
            input_to_robot_tfs.append(tf)

        # visualization
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.transform(input_to_fixed)

        fixed_to_robots = []
        pose_frames = []
        for input_to_robot in input_to_robot_tfs:
            fixed_to_robot = input_to_fixed @ np.linalg.inv(input_to_robot)
            fixed_to_robots.append(fixed_to_robot)

            pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            pose_frame.transform(fixed_to_robot)
            pose_frames.append(pose_frame)

        o3d.visualization.draw_geometries(pose_frames + [pcd])
        break
    bag.close()
    print("Processing complete.")


def to_sec(file_name):
    # 1723650624_685964108.npz -> 1723650624.685964108
    seconds = file_name.split('.')[0]
    seconds = float(seconds.replace('_', '.'))
    return seconds


if __name__ == "__main__":
    # bag_file = '/media/ruslan/VRAS-DATA 4TB 2/outdoor_dataset/24-08-initial_tests_marv/24-08-14-monoforce-silly_drive.bag'
    bag_file = "/media/ruslan/VRAS-DATA 4TB 2/outdoor_dataset/25-03-19-petrin/marv_2025-03-19-15-35-24.bag"

    seq = f"../data/ROUGH/{bag_file.split('/')[-1].split('.')[0]}"
    clouds_path = os.path.join(seq, "clouds")
    cloud_files = sorted(os.listdir(clouds_path))
    cloud_stamps = [to_sec(f) for f in cloud_files]

    process_bag(bag_file, cloud_stamps)
