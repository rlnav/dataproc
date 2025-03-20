import torch
from flipper_training.configs import (
    WorldConfig,
    RobotModelConfig,
    PhysicsEngineConfig,
)
from flipper_training.engine.engine import DPhysicsEngine, PhysicsState
from flipper_training.utils.geometry import unit_quaternion
from flipper_training.engine.engine_state import (
    vectorize_iter_of_states as vectorize_states,
)
from collections import deque


def main():
    num_robots = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Heightmap setup - use torch's XY indexing !!!!!
    grid_res = 0.05  # 5cm per grid cell
    max_coord = 3.2  # meters
    DIM = int(2 * max_coord / grid_res)
    xint = torch.linspace(-max_coord, max_coord, DIM)
    yint = torch.linspace(-max_coord, max_coord, DIM)
    x, y = torch.meshgrid(xint, yint, indexing="xy")

    # gaussian hm
    z = (
            (
                    1.0 * torch.exp(-0.5 * ((x - 0) ** 2 + (y - 4) ** 2))
                    + 5.0 * torch.exp(-0.3 * ((x - 1) ** 2 + (y + 2) ** 2))
                    + 2.0 * torch.exp(-0.1 * ((x + max_coord) ** 2 + (y + max_coord) ** 2))
            )
            + 0.01 * torch.randn_like(x)
            + torch.exp(-0.03 * ((x + 5) ** 2 + (y + 5) ** 2))
    )
    x_grid = x.repeat(num_robots, 1, 1)
    y_grid = y.repeat(num_robots, 1, 1)
    z_grid = z.repeat(num_robots, 1, 1)

    # Instatiate the physics config
    robot_model = RobotModelConfig(robot_type="marv")
    world_config = WorldConfig(
        x_grid=x_grid,
        y_grid=y_grid,
        z_grid=z_grid,
        grid_res=grid_res,
        max_coord=max_coord,
        k_stiffness=40000,
    )
    physics_config = PhysicsEngineConfig(num_robots=num_robots)

    # Controls
    traj_length = 10.0  # seconds
    n_iters = int(traj_length / physics_config.dt)
    speed = 1.0  # m/s forward
    omega = 0.5  # rad/s yaw
    controls = robot_model.vw_to_vels(speed, omega)
    flipper_controls = torch.zeros_like(controls)

    for cfg in [robot_model, world_config, physics_config]:
        cfg.to(device)

    engine = DPhysicsEngine(physics_config, robot_model, device)

    x0 = torch.tensor([-3, -2.5, 0.1]).to(device).repeat(num_robots, 1)
    xd0 = torch.zeros_like(x0)
    q0 = unit_quaternion(num_robots, device=device)
    omega0 = torch.zeros_like(x0)
    thetas0 = torch.zeros(num_robots, robot_model.num_driving_parts).to(device)
    controls_all = torch.cat((controls, flipper_controls), dim=-1).repeat(n_iters, num_robots, 1).to(device)

    # Set the flippers to an fixed position at the beginning
    angles_deg = torch.tensor([-0.0, -0.0, -0.0, -0.0], device=device)
    angles = torch.deg2rad(angles_deg)
    max_joint_vel = robot_model.joint_max_pivot_vels
    min_joint_vel = -robot_model.joint_max_pivot_vels
    times = (angles >= 0).float() * (angles / max_joint_vel) + (angles < 0).float() * (angles / min_joint_vel)
    steps = (times / physics_config.dt).int()
    vels = angles / times
    vels[torch.isnan(vels)] = 0.0
    controls_all[: steps[0], :, robot_model.num_driving_parts] = vels[0]
    controls_all[: steps[1], :, robot_model.num_driving_parts + 1] = vels[1]
    controls_all[: steps[2], :, robot_model.num_driving_parts + 2] = vels[2]
    controls_all[: steps[3], :, robot_model.num_driving_parts + 3] = vels[3]

    init_state = PhysicsState(x0, xd0, q0, omega0, thetas0)

    states = deque(maxlen=n_iters)
    dstates = deque(maxlen=n_iters)
    auxs = deque(maxlen=n_iters)

    state = init_state
    for i in range(n_iters):
        state, der, aux = engine(state, controls_all[i], world_config)
        states.append(state)
        dstates.append(der)
        auxs.append(aux)

    states_vec = vectorize_states(states)
    dstates_vec = vectorize_states(dstates)
    aux_vec = vectorize_states(auxs)

    # visualize heightmap and trajectory with matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib as mpl
    mpl.use('TkAgg')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='terrain')
    # for i in range(num_robots):
    for i in range(1):
        xs = states_vec.x[:, i].cpu().numpy()
        print(xs.shape)
        ax.plot(xs[:, 0], xs[:, 1], xs[:, 2])
    plt.show()


if __name__ == '__main__':
    main()