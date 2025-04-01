#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from ros_numpy import numpify
from tf2_ros import BufferCore, TransformException
from rosbag import Bag, ROSBagException

import os
import numpy as np
from tqdm import tqdm
import open3d as o3d
import matplotlib.pyplot as plt


def load_tf_buffer(bag: Bag):
    tf_topics = ['/tf', '/tf_static']
    buffer = BufferCore(rospy.Duration(24 * 60 * 60))
    for topic, msg, stamp in tqdm(bag.read_messages(topics=tf_topics),
                                  desc='reading transforms',
                                  total=bag.get_message_count(topic_filters=tf_topics)):
        if topic == '/tf':
            for tf in msg.transforms:
                buffer.set_transform(tf, 'bag')
        elif topic == '/tf_static':
            for tf in msg.transforms:
                buffer.set_transform_static(tf, 'bag')
    return buffer


def load_control_buffer(bag: Bag, control_topics: list):
    class ControlBuffer(object):
        def __init__(self):
            self.stamps = []
            self.vels = []
            self.omegas = []

        def append(self, msg, stamp=None):
            assert isinstance(msg, Twist)
            if stamp is None:
                stamp = msg.header.stamp

            self.stamps.append(stamp.to_sec())
            self.vels.append(msg.linear.x)
            self.omegas.append(msg.angular.z)

        def __len__(self):
            return len(self.stamps)

        def load(self):
            for topic, msg, stamp in tqdm(bag.read_messages(topics=control_topics),
                                          desc='reading controls',
                                          total=bag.get_message_count(topic_filters=control_topics)):
                msg = Twist(*slots(msg))
                self.append(msg, stamp)

        def get(self, time_left, time_right):
            """Get control values for the given time window."""
            left_idx = np.searchsorted(self.stamps, time_left)
            right_idx = np.searchsorted(self.stamps, time_right)
            return self.stamps[left_idx:right_idx], self.vels[left_idx:right_idx], self.omegas[left_idx:right_idx]

        def get_interpolated(self, time_left, time_right, step=0.01):
            """Get control values for the given time window."""
            t, vels, omegas = self.get(time_left, time_right)
            t_interp = np.arange(time_left, time_right, step)
            if len(t) == 0:
                # If no data is available, return zero control values
                vels_interp = np.zeros_like(t_interp)
                omegas_interp = np.zeros_like(t_interp)
            else:
                # # Interpolate the control values to the new time points
                vels_interp = np.interp(t_interp, t, vels, left=0., right=0.)
                # omegas_interp = np.interp(t_interp, t, omegas, left=0., right=0.)

                # vels_interp = np.zeros_like(t_interp)
                omegas_interp = np.zeros_like(t_interp)
                indices = np.searchsorted(t_interp, t).clip(0, len(t_interp) - 1)
                # vels_interp[indices] = vels
                omegas_interp[indices] = omegas

            return t_interp, vels_interp, omegas_interp

    control_buffer = ControlBuffer()
    control_buffer.load()

    return control_buffer


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


def process_bag(bag_file, cloud_times):
    try:
        bag = Bag(bag_file, 'r')
    except ROSBagException as ex:
        print(f"Error opening bag file: {ex}")
        return

    tf_buffer = load_tf_buffer(bag)
    control_buffer = load_control_buffer(bag, ['/marv/cartesian_controller/cmd_vel'])

    time_horizon = [0., 10.]  # seconds
    time_search_window = 0.2  # seconds
    cloud_topic = "/points_filtered_kontron"
    fixed_frame = "map"
    robot_frame = "base_link"
    time_step = 0.5  # seconds
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

        # Find transform from input cloud to fixed frame.
        try:
            input_to_fixed = tf_buffer.lookup_transform_core(fixed_frame, pcd_msg.header.frame_id, pcd_msg.header.stamp)
        except TransformException as ex:
            print('Could not transform from %s to %s at %.3f s.' % (
            pcd_msg.header.frame_id, fixed_frame, pcd_msg.header.stamp.to_sec()))
            continue
        input_to_fixed = numpify(input_to_fixed.transform)
        # print(input_to_fixed.shape)

        # Find transforms from input cloud to robot positions within the horizon.
        input_to_robot_tfs = []
        start = pcd_msg.header.stamp.to_sec()
        traj_ts = np.linspace(start + n[0] * time_step, start + n[1] * time_step, (n[1] - n[0]) + 1)
        for t in traj_ts:
            try:
                tf = tf_buffer.lookup_transform_full_core(robot_frame, rospy.Time.from_seconds(t),
                                                          pcd_msg.header.frame_id, pcd_msg.header.stamp,
                                                          fixed_frame)
            except TransformException as ex:
                print('Could not transform from %s to %s at %.3f s.' % (pcd_msg.header.frame_id, robot_frame, t))
                continue
            tf = numpify(tf.transform)
            # print(tf.shape)
            input_to_robot_tfs.append(tf)
        input_to_robot_tfs = np.array(input_to_robot_tfs)
        fixed_to_robot_tfs = input_to_fixed @ np.linalg.inv(input_to_robot_tfs)

        # get robot commanded velocities
        time_left = start + time_horizon[0]
        time_right = start + time_horizon[1]
        control_ts, vels, omegas = control_buffer.get(time_left, time_right)
        control_ts_interp, vels_interp, omegas_interp = control_buffer.get_interpolated(time_left, time_right, step=0.01)
        print(points.shape, len(control_ts_interp), len(vels_interp), len(traj_ts), len(fixed_to_robot_tfs))

        if np.random.random() < 0.05:
            visualize(control_ts_interp, vels_interp, omegas_interp,
                      points, fixed_to_robot_tfs, input_to_fixed)
            motion(vels_interp, omegas_interp, points, fixed_to_robot_tfs[0], input_to_fixed)
            # motion(vels, omegas, points, fixed_to_robot_tfs[0], input_to_fixed)
    bag.close()
    print("Processing complete.")


def visualize(control_ts, vels, omegas, points, fixed_to_robot_tfs, input_to_fixed):
    plt.figure(figsize=(12, 8))
    plt.plot(control_ts, vels, '.', label='v(t)')
    plt.plot(control_ts, omegas, '.', label='w(t)')
    # plt.plot(control_ts_interp, vels_interp, label='v(t) interp')
    # plt.plot(control_ts_interp, omegas_interp, label='w(t) interp')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s or rad/s)')
    plt.title('Robot Velocities')
    plt.legend()
    plt.grid()
    plt.show()

    # visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.transform(input_to_fixed)

    fixed_to_robots = []
    pose_frames = []
    for i, fixed_to_robot in enumerate(fixed_to_robot_tfs):
        fixed_to_robots.append(fixed_to_robot)

        pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        pose_frame.transform(fixed_to_robot)
        pose_frames.append(pose_frame)

    o3d.visualization.draw_geometries(pose_frames + [pcd])


def motion(vels, omegas, points_input, fixed_to_robot, input_to_fixed):
    import sys
    sys.path.append('../../monoforce/monoforce/src')
    import torch
    from monoforce.configs import (
        WorldConfig,
        RobotModelConfig,
        PhysicsEngineConfig,
    )
    from monoforce.models.physics_engine.engine.engine import DPhysicsEngine, PhysicsState
    from monoforce.models.physics_engine.utils.geometry import unit_quaternion
    from monoforce.models.physics_engine.engine.engine_state import vectorize_iter_of_states as vectorize_states
    from monoforce.models.physics_engine.vis.animator import animate_trajectory
    from monoforce.models.physics_engine.utils.environment import make_x_y_grids
    from monoforce.cloudproc import estimate_heightmap
    from collections import deque

    n_robots = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Heightmap setup
    grid_res = 0.1  # 10cm per grid cell
    max_coord = 6.4  # meters
    x_grid, y_grid = make_x_y_grids(max_coord=max_coord, grid_res=grid_res, num_robots=n_robots)

    # Estimate heightmap
    input_to_robot = np.linalg.inv(fixed_to_robot) @ input_to_fixed
    points = points_input @ input_to_robot[:3, :3].T + input_to_robot[:3, 3]  # Transform points to robot frame
    points = torch.from_numpy(points).float()
    z_grid = estimate_heightmap(points, grid_res=grid_res, d_max=max_coord, h_max=1.0)[0].repeat(n_robots, 1, 1)

    # Instantiate the configs
    robot_model = RobotModelConfig()
    world_config = WorldConfig(
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        grid_res=grid_res,
        max_coord=max_coord,
    )
    physics_config = PhysicsEngineConfig(num_robots=n_robots)
    for cfg in [robot_model, world_config, physics_config]:
        cfg.to(device)

    # Controls
    n_iters = len(vels)
    vels = torch.as_tensor(vels, dtype=torch.float32, device=device)
    omegas = torch.as_tensor(omegas, dtype=torch.float32, device=device)
    controls = robot_model.vw_to_vels(vels, omegas)
    flipper_controls = torch.zeros_like(controls)
    controls_all = torch.cat((controls, flipper_controls), dim=-1).repeat(n_robots, 1, 1)
    assert controls_all.shape == (n_robots, n_iters, robot_model.num_driving_parts * 2)

    # Instantiate the physics engine
    engine = DPhysicsEngine(physics_config, robot_model, device)

    # Initial state
    x0 = torch.tensor([0.0, 0.0, 0.1]).to(device).repeat(n_robots, 1)
    xd0 = torch.zeros_like(x0)
    q0 = unit_quaternion(batch_size=n_robots, device=device)
    omega0 = torch.zeros_like(x0)
    thetas0 = torch.zeros(n_robots, robot_model.num_driving_parts).to(device)
    init_state = PhysicsState(x0, xd0, q0, omega0, thetas0)

    states = deque(maxlen=n_iters)
    auxs = deque(maxlen=n_iters)

    state = init_state
    for i in range(n_iters):
        state, der, aux = engine(state, controls_all[:, i], world_config)
        states.append(state)
        auxs.append(aux)

    states_vec = vectorize_states(states)
    print(states_vec.x.shape)

    # visualization
    animate_trajectory(
        world_config,
        physics_config,
        states,
        auxs,
    )


def to_sec(file_name):
    # "1723650624_685964108.npz" -> 1723650624.685964108 [seconds]
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
