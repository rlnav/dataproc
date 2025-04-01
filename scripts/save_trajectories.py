#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from sensor_msgs.msg import PointCloud2
from ros_numpy import numpify
from tf2_ros import BufferCore, TransformException
from rosbag import Bag, ROSBagException

import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from collections import deque

import sys
sys.path.append('../../monoforce/monoforce/src')
from monoforce.models.physics_engine.engine.engine import DPhysicsEngine, PhysicsState
from monoforce.models.physics_engine.utils.geometry import unit_quaternion
from monoforce.models.physics_engine.engine.engine_state import vectorize_iter_of_states as vectorize_states
from monoforce.models.physics_engine.vis.animator import animate_trajectory
from monoforce.models.physics_engine.utils.environment import make_x_y_grids
from monoforce.cloudproc import estimate_heightmap
from monoforce.configs import RobotModelConfig, WorldConfig, PhysicsEngineConfig
import matplotlib as mpl
mpl.use('Qt5Agg')


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


def load_control_buffer(bag: Bag, cmd_vel_topic: str, joint_states_topic: str):
    assert cmd_vel_topic in bag.get_type_and_topic_info()[1], f"Topic {cmd_vel_topic} not found in bag."
    assert joint_states_topic in bag.get_type_and_topic_info()[1], f"Topic {joint_states_topic} not found in bag."

    class ControlBuffer(object):
        def __init__(self):
            self.cmd_vel_stamps = []
            self.vels = []
            self.omegas = []

            self.js_stamps = []
            self.fl_flipper_angles = []
            self.fr_flipper_angles = []
            self.rl_flipper_angles = []
            self.rr_flipper_angles = []

            self.robot_model = RobotModelConfig()

        def __len__(self):
            return len(self.cmd_vel_stamps)

        def load(self):
            for topic, msg, stamp in tqdm(bag.read_messages(topics=[cmd_vel_topic, joint_states_topic]),
                                          desc='reading controls',
                                          total=bag.get_message_count(topic_filters=[cmd_vel_topic, joint_states_topic])):
                if topic == cmd_vel_topic:
                    cmd_vel_msg = Twist(*slots(msg))
                    self.cmd_vel_stamps.append(stamp.to_sec())
                    self.vels.append(cmd_vel_msg.linear.x)
                    self.omegas.append(cmd_vel_msg.angular.z)

                if topic == joint_states_topic:
                    js_msg = JointState(*slots(msg))
                    self.js_stamps.append(js_msg.header.stamp.to_sec())
                    self.fl_flipper_angles.append(js_msg.position[0])
                    self.fr_flipper_angles.append(js_msg.position[1])
                    self.rl_flipper_angles.append(js_msg.position[2])
                    self.rr_flipper_angles.append(js_msg.position[3])

        def get_vws(self, time_left, time_right):
            """Get cmd vel values for the given time window."""
            left_idx = np.searchsorted(self.cmd_vel_stamps, time_left)
            right_idx = np.searchsorted(self.cmd_vel_stamps, time_right)
            ts = self.cmd_vel_stamps[left_idx:right_idx]
            vels = self.vels[left_idx:right_idx]
            omegas = self.omegas[left_idx:right_idx]
            return ts, vels, omegas

        def get_js(self, time_left, time_right):
            """Get joint state values for the given time window."""
            left_idx = np.searchsorted(self.js_stamps, time_left)
            right_idx = np.searchsorted(self.js_stamps, time_right)
            ts = self.js_stamps[left_idx:right_idx]
            fl_flipper_angles = self.fl_flipper_angles[left_idx:right_idx]
            fr_flipper_angles = self.fr_flipper_angles[left_idx:right_idx]
            rl_flipper_angles = self.rl_flipper_angles[left_idx:right_idx]
            rr_flipper_angles = self.rr_flipper_angles[left_idx:right_idx]
            flipper_angles = np.stack([
                fl_flipper_angles,
                fr_flipper_angles,
                rl_flipper_angles,
                rr_flipper_angles
            ]).T  # (4, N) -> (N, 4)
            return ts, flipper_angles

        def get_controls(self, time_left, time_right, step=0.01):
            """Get control values for the given time window."""
            ts_cmd_vel, vels, omegas = self.get_vws(time_left, time_right)
            ts_js, flipper_angles = self.get_js(time_left, time_right)

            ts_interp = np.arange(time_left, time_right, step)
            controls_interp = np.zeros((len(ts_interp), self.robot_model.num_driving_parts * 2))
            if len(ts_cmd_vel) != 0 and len(ts_js) != 0:
                flipper_vels = self.robot_model.vw_to_vels(v=torch.as_tensor(vels),
                                                           w=torch.as_tensor(omegas)).cpu().numpy()
                # Interpolate the control values to the new time points
                for i in range(controls_interp.shape[1]):
                    if i < self.robot_model.num_driving_parts:
                        controls_interp[:, i] = np.interp(ts_interp, ts_cmd_vel, flipper_vels[:, i])
                    else:
                        controls_interp[:, i] = np.interp(ts_interp, ts_js, flipper_angles[:, i - self.robot_model.num_driving_parts])
            return ts_interp, controls_interp

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


def process_bag(bag_file, cloud_times,
                time_horizon=[0., 5.],
                robot_frame='base_link',
                fixed_frame='odom',
                time_step=1.0,
                time_search_window=0.2,
                cloud_topic='/points',
                cmd_vel_topic='/cmd_vel',
                joint_states_topic='/joint_states'):
    try:
        bag = Bag(bag_file, 'r')
    except ROSBagException as ex:
        print(f"Error opening bag file: {ex}")
        return

    tf_buffer = load_tf_buffer(bag)
    control_buffer = load_control_buffer(bag, cmd_vel_topic, joint_states_topic)

    # plt.figure(figsize=(12, 8))
    # plt.plot(control_buffer.cmd_vel_stamps, control_buffer.vels, 'r', label='v(t)')
    # # plt.plot(control_buffer.cmd_vel_stamps, control_buffer.omegas, 'b', label='w(t)')
    # plt.plot(control_buffer.js_stamps, control_buffer.fl_flipper_angles, 'g', label='fl flipper angles')
    # # plt.plot(control_buffer.js_stamps, control_buffer.fr_flipper_angles, 'y', label='fr flipper angles')
    # # plt.plot(control_buffer.js_stamps, control_buffer.rl_flipper_angles, 'c', label='rl flipper angles')
    # # plt.plot(control_buffer.js_stamps, control_buffer.rr_flipper_angles, 'm', label='rr flipper angles')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Control values')
    # plt.legend()
    # plt.grid()
    # plt.show()

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
            input_to_robot_tfs.append(tf)
        input_to_robot_tfs = np.array(input_to_robot_tfs)
        fixed_to_robot_tfs = input_to_fixed @ np.linalg.inv(input_to_robot_tfs)

        # get robot commanded velocities
        time_left = start + time_horizon[0]
        time_right = start + time_horizon[1]
        control_ts, controls = control_buffer.get_controls(time_left, time_right, step=0.01)
        print(points.shape, len(control_ts), len(controls), len(traj_ts), len(fixed_to_robot_tfs))

        if np.random.random() < 0.05:
            ts, vels, omegas = control_buffer.get_vws(time_left, time_right)
            plt.figure(figsize=(12, 8))
            plt.plot(ts, vels, 'r', label='v(t)')
            plt.plot(ts, omegas, 'b', label='w(t)')
            plt.xlabel('Time (s)')
            plt.ylabel('Control values')
            plt.legend()
            plt.grid()
            plt.show()

            show_cloud_and_path(points, fixed_to_robot_tfs, input_to_fixed)

            motion(controls, points, fixed_to_robot_tfs[0], input_to_fixed)
    bag.close()
    print("Processing complete.")


def show_cloud_and_path(points, fixed_to_robot_tfs, input_to_fixed):
    import open3d as o3d

    # cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.transform(input_to_fixed)
    # path
    pose_frames = []
    for i, fixed_to_robot in enumerate(fixed_to_robot_tfs):
        pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        pose_frame.transform(fixed_to_robot)
        pose_frames.append(pose_frame)
    o3d.visualization.draw_geometries(pose_frames + [pcd])


def motion(controls, points_input, fixed_to_robot, input_to_fixed):
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
    z_grid = estimate_heightmap(points, grid_res=grid_res, d_max=max_coord, h_max=1.0, r_min=1.0)[0].repeat(n_robots, 1, 1)
    # z_grid = torch.zeros_like(x_grid)

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
    n_iters = len(controls)
    controls = torch.as_tensor(controls).float().to(device).repeat(n_robots, 1, 1)
    assert controls.shape == (n_robots, n_iters, robot_model.num_driving_parts * 2)

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
        state, der, aux = engine(state, controls[:, i], world_config)
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
    # bag_file = '/media/ruslan/VRAS-DATA 4TB 2/outdoor_dataset/24-08-initial_tests_marv/24-08-14-monoforce-long_drive.bag'
    # bag_file = "/media/ruslan/VRAS-DATA 4TB 2/outdoor_dataset/25-03-19-petrin/marv_2025-03-19-15-35-24.bag"
    bag_file = "/media/ruslan/VRAS-DATA 4TB 2/outdoor_dataset/24-10-31-petrin/marv_2024-10-31-15-52-07.bag"

    seq = f"../data/ROUGH/{bag_file.split('/')[-1].split('.')[0]}"
    clouds_path = os.path.join(seq, "clouds")
    cloud_files = sorted(os.listdir(clouds_path))
    cloud_stamps = [to_sec(f) for f in cloud_files]

    process_bag(bag_file, cloud_stamps,
                time_horizon=[0., 5.],
                fixed_frame='odom',
                time_step=0.5,
                time_search_window=0.2,
                cloud_topic='/points_filtered_kontron',
                cmd_vel_topic='/marv/cartesian_controller/cmd_vel',
                joint_states_topic='/marv/flippers/joint_states')
