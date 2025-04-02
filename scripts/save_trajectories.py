#!/usr/bin/env python

"""
This script processes a ROS bag file to extract point cloud data, robot trajectory, and control information.
It visualizes the data and simulates the robot's motion using a physics engine.
It estimates a heightmap from the point cloud to be used for the simulation.

We find for each point cloud the closest corresponding robot trajectory within a time horizon.
Additionally, we find the corresponding robot control commands (cmd_vel and joint states) for the given time horizon.
The data could be visualized with a probability `vis_prob`.

We save the robot trajectory and control commands to files for later use (MonoForce model training).
"""

# ROS bag processing imports
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from sensor_msgs.msg import PointCloud2
from ros_numpy import numpify
from tf2_ros import BufferCore, TransformException
from rosbag import Bag, ROSBagException
# common utils imports
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from collections import deque
from dataproc.io import write_to_csv, append_to_csv
from dataproc.cloudproc import estimate_heightmap
# visualization imports
import sys
sys.path.append('../../monoforce/monoforce/src')
from monoforce.models.physics_engine.engine.engine import DPhysicsEngine, PhysicsState
from monoforce.models.physics_engine.utils.geometry import unit_quaternion
from monoforce.models.physics_engine.vis.animator import animate_trajectory
from monoforce.models.physics_engine.utils.environment import make_x_y_grids
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
    # assert cmd_vel_topic in bag.get_type_and_topic_info()[1], f"Topic {cmd_vel_topic} not found in bag."
    # assert joint_states_topic in bag.get_type_and_topic_info()[1], f"Topic {joint_states_topic} not found in bag."

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

            self.fl_flipper_ws = []
            self.fr_flipper_ws = []
            self.rl_flipper_ws = []
            self.rr_flipper_ws = []

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

                    self.fl_flipper_ws.append(js_msg.velocity[0])
                    self.fr_flipper_ws.append(js_msg.velocity[1])
                    self.rl_flipper_ws.append(js_msg.velocity[2])
                    self.rr_flipper_ws.append(js_msg.velocity[3])

        def get_vws(self, time_left, time_right):
            """Get cmd vel values for the given time window."""
            left_idx = np.searchsorted(self.cmd_vel_stamps, time_left)
            right_idx = np.searchsorted(self.cmd_vel_stamps, time_right)
            ts = self.cmd_vel_stamps[left_idx:right_idx]
            vels = self.vels[left_idx:right_idx]
            omegas = self.omegas[left_idx:right_idx]
            return ts, vels, omegas

        def get_flipper_angles(self, t_des):
            """Get flipper angle values for the given time moment."""
            idx = np.searchsorted(self.js_stamps, t_des)
            t = self.js_stamps[idx]
            fl_flipper_angle = self.fl_flipper_angles[idx]
            fr_flipper_angle = self.fr_flipper_angles[idx]
            rl_flipper_angle = self.rl_flipper_angles[idx]
            rr_flipper_angle = self.rr_flipper_angles[idx]
            flipper_angles = np.array([
                fl_flipper_angle,
                fr_flipper_angle,
                rl_flipper_angle,
                rr_flipper_angle
            ])  # (4,)
            return t, flipper_angles

        def get_flipper_ws(self, time_left, time_right):
            """Get flipper angular velocity values for the given time window."""
            left_idx = np.searchsorted(self.js_stamps, time_left)
            right_idx = np.searchsorted(self.js_stamps, time_right)
            ts = self.js_stamps[left_idx:right_idx]
            fl_flipper_ws = self.fl_flipper_ws[left_idx:right_idx]
            fr_flipper_ws = self.fr_flipper_ws[left_idx:right_idx]
            rl_flipper_ws = self.rl_flipper_ws[left_idx:right_idx]
            rr_flipper_ws = self.rr_flipper_ws[left_idx:right_idx]
            flipper_ws = np.stack([
                fl_flipper_ws,
                fr_flipper_ws,
                rl_flipper_ws,
                rr_flipper_ws
            ]).T  # (4, N) -> (N, 4)
            return ts, flipper_ws

        def get_controls(self, time_left, time_right, step=0.01):
            """Get control values for the given time window."""
            ts_cmd_vel, vels, omegas = self.get_vws(time_left, time_right)
            ts_js, flipper_ws = self.get_flipper_ws(time_left, time_right)

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
                        controls_interp[:, i] = np.interp(ts_interp, ts_js, flipper_ws[:, i - self.robot_model.num_driving_parts])
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


def process_bag(bag_path, cloud_files,
                traj_time_horizon=[0., 10.],
                robot_frame='base_link',
                fixed_frame='odom',
                traj_time_step=1.0,
                cloud_time_search_window=0.2,
                cloud_topic='/points',
                cmd_vel_topic='/cmd_vel',
                joint_states_topic='/joint_states',
                vis_prob=0.0,
                save=False):
    try:
        bag = Bag(bag_path, 'r')
    except ROSBagException as ex:
        print(f"Error opening bag file: {ex}")
        return
    # Check if the topics exist in the bag: control commands and point clouds
    if not cloud_topic in bag.get_type_and_topic_info()[1]:
        print(f"Topic {cloud_topic} not found in bag.")
        return
    if not cmd_vel_topic in bag.get_type_and_topic_info()[1]:
        print(f"Topic {cmd_vel_topic} not found in bag.")
        return
    if not joint_states_topic in bag.get_type_and_topic_info()[1]:
        print(f"Topic {joint_states_topic} not found in bag.")
        return

    # Get the time of the point clouds
    cloud_files = sorted(cloud_files)
    cloud_times = [fname_to_sec(f) for f in cloud_files]
    assert np.all(np.diff(cloud_times) > 0), "Cloud files are not sorted by time."

    seq_path = os.path.basename(bag_path.replace('.bag', ''))
    seq_path = os.path.join('/media/ruslan/VRAS-DATA 4TB 2/datasets/ROUGH/', seq_path)
    trajs_path = os.path.join(seq_path, 'trajectories')
    controls_path = os.path.join(seq_path, 'controls')
    for path in [trajs_path, controls_path]:
        os.makedirs(path, exist_ok=True)

    tf_buffer = load_tf_buffer(bag)
    control_buffer = load_control_buffer(bag, cmd_vel_topic, joint_states_topic)

    n = [int(np.floor(h / traj_time_step)) for h in traj_time_horizon]

    for cloud_i, cloud_time in tqdm(enumerate(cloud_times), desc='Going through clouds stamps', total=len(cloud_times)):
        pcd_msg = get_point_cloud_in_window(bag, cloud_time,
                                            time_window=cloud_time_search_window,
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
        traj_ts = np.linspace(start + n[0] * traj_time_step, start + n[1] * traj_time_step, (n[1] - n[0]) + 1)
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
        # robot trajectory is in the fixed frame
        fixed_to_robot_tfs = input_to_fixed @ np.linalg.inv(input_to_robot_tfs)
        # flipper angles at trajectory timestamps
        traj_flipper_anles = []
        for t in traj_ts:
            t, flipper_angles = control_buffer.get_flipper_angles(t)
            traj_flipper_anles.append(flipper_angles)

        # get robot commanded velocities
        time_left = start + traj_time_horizon[0]
        time_right = start + traj_time_horizon[1]
        control_ts, controls = control_buffer.get_controls(time_left, time_right, step=0.01)

        if save:
            """Save the data"""
            # trajectory: poses
            traj_path = os.path.join(trajs_path,
                                     f'traj_{robot_frame}_{cloud_files[cloud_i]}'.replace('.npz', '.csv'))
            write_to_csv(traj_path, 'stamp, T00, T01, T02, T03, T10, T11, T12, T13, T20, T21, T22, T23\n')
            assert len(traj_ts) == len(fixed_to_robot_tfs),\
                f"Trajectory timestamps and transforms length mismatch: {len(traj_ts)} != {len(fixed_to_robot_tfs)}"
            for pose_i in range(len(fixed_to_robot_tfs)):
                pose = fixed_to_robot_tfs[pose_i]
                append_to_csv(traj_path,
                              '%.9f, %s\n' % (traj_ts[pose_i], ', '.join(['%.3f' % x for x in pose[:3, :].flatten()])))
            # trajectory: flipper angles
            traj_flipper_path = os.path.join(trajs_path,
                                                f'traj_flipper_angles_{cloud_files[cloud_i]}'.replace('.npz', '.csv'))
            assert len(traj_ts) == len(traj_flipper_anles),\
                f"Trajectory timestamps and flipper angles length mismatch: {len(traj_ts)} != {len(traj_flipper_anles)}"
            write_to_csv(traj_flipper_path, 'stamp, fl_a, fr_a, rl_a, rr_a\n')
            for angle_i in range(len(traj_flipper_anles)):
                flipper_angles = traj_flipper_anles[angle_i]
                append_to_csv(traj_flipper_path,
                              '%.9f, %s\n' % (traj_ts[angle_i], ', '.join(['%.3f' % x for x in flipper_angles])))

            # controls
            control_path = os.path.join(controls_path,
                                        f'flipper_vs_ws_{cloud_files[cloud_i]}'.replace('.npz', '.csv'))
            assert len(control_ts) == len(controls), "Control timestamps and values length mismatch."
            write_to_csv(control_path, 'stamp, fl_v, fr_v, rl_v, rr_v, fl_w, fr_w, rl_w, rr_w\n')
            for control_i in range(len(control_ts)):
                append_to_csv(control_path,
                              '%.9f, %s\n' % (control_ts[control_i], ', '.join(['%.3f' % c for c in controls[control_i]])))

        # Visualize the data with probability vis_prob
        if np.random.random() < vis_prob:
            ts, vels, omegas = control_buffer.get_vws(time_left, time_right)
            plt.figure(figsize=(16, 8))
            plt.subplot(121)
            plt.plot(ts, vels, 'r', label='v(t)')
            plt.plot(ts, omegas, 'b', label='w(t)')
            plt.xlabel('Time (s)')
            plt.ylabel('Cmd vels [m/s] or [rad/s]')
            plt.legend()
            plt.grid()

            plt.subplot(122)
            ts_js, flipper_ws = control_buffer.get_flipper_ws(time_left, time_right)
            plt.plot(ts_js, flipper_ws[:, 0], 'g', label='fl flipper ws')
            plt.plot(ts_js, flipper_ws[:, 1], 'y', label='fr flipper ws')
            plt.plot(ts_js, flipper_ws[:, 2], 'c', label='rl flipper ws')
            plt.plot(ts_js, flipper_ws[:, 3], 'm', label='rr flipper ws')
            plt.xlabel('Time (s)')
            plt.ylabel('Joint states vels [rad/s]')
            plt.legend()
            plt.grid()
            plt.show()

            show_cloud_and_path(points, fixed_to_robot_tfs, input_to_fixed)

            n_robots = 1
            x0 = torch.tensor([0.0, 0.0, 0.1]).repeat(n_robots, 1)
            xd0 = torch.zeros_like(x0)
            q0 = unit_quaternion(batch_size=n_robots)
            omega0 = torch.zeros_like(x0)
            thetas0 = torch.as_tensor(traj_flipper_anles[0]).float().repeat(n_robots, 1)
            state0 = PhysicsState(x0, xd0, q0, omega0, thetas0)

            motion(controls, points, fixed_to_robot_tfs[0], input_to_fixed, state0=state0)
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


def motion(controls, points_input, fixed_to_robot, input_to_fixed, state0: PhysicsState | None = None):
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
    if state0 is None:
        x0 = torch.tensor([0.0, 0.0, 0.1]).repeat(n_robots, 1)
        xd0 = torch.zeros_like(x0)
        q0 = unit_quaternion(batch_size=n_robots)
        omega0 = torch.zeros_like(x0)
        thetas0 = torch.zeros(n_robots, robot_model.num_driving_parts)
        state0 = PhysicsState(x0, xd0, q0, omega0, thetas0)
    state0 = state0.to(device)

    states = deque(maxlen=n_iters)
    auxs = deque(maxlen=n_iters)

    state = state0
    for i in range(n_iters):
        state, der, aux = engine(state, controls[:, i], world_config)
        states.append(state)
        auxs.append(aux)

    # visualization
    animate_trajectory(
        world_config,
        physics_config,
        states,
        auxs,
    )


def fname_to_sec(file_name):
    # "1723650624_685964108.npz" -> 1723650624.685964108 [seconds]
    seconds = file_name.split('.')[0]
    seconds = float(seconds.replace('_', '.'))
    return seconds


def main():
    from glob import glob

    data_dir = "/media/ruslan/VRAS-DATA 4TB 2/outdoor_dataset/"
    bag_paths_glob = glob(os.path.join(data_dir, "*", "*.bag"))
    bag_paths = []
    seq_paths = []
    for bag_path in tqdm(bag_paths_glob):
        seq_path = f"../data/ROUGH/{os.path.basename(bag_path.replace('.bag', ''))}"
        # if not already in the list append bag path
        if bag_path not in bag_paths and os.path.exists(seq_path) and "marv" in bag_path:
            bag_paths.append(bag_path)
            seq_paths.append(seq_path)
    seq_paths = sorted(seq_paths)
    bag_paths = sorted(bag_paths)
    print(f"Found {len(bag_paths)} bags.")
    # bag_paths = ["/media/ruslan/VRAS-DATA 4TB 2/outdoor_dataset/25-03-19-petrin/marv_2025-03-19-15-35-24.bag"]

    for seq_path, bag_path in zip(seq_paths, bag_paths):
        cloud_files = os.listdir(os.path.join(seq_path, "clouds"))
        print(f"Processing {bag_path}...")
        process_bag(bag_path, cloud_files,
                    traj_time_horizon=[0., 10.],
                    fixed_frame='map',
                    traj_time_step=0.5,
                    cloud_time_search_window=0.2,
                    cloud_topic='/points_filtered_kontron',
                    cmd_vel_topic='/marv/cartesian_controller/cmd_vel',
                    joint_states_topic='/marv/joint_states',
                    vis_prob=0.0,
                    save=True)


if __name__ == "__main__":
    main()