import numpy as np
import pyvista as pv
import time


marv_obj = "/home/ruslan/workspaces/traversability_ws/src/monoforce/monoforce/config/meshes/marv.obj"
robot_mesh = pv.read(marv_obj)

# Create a 2D grid (Heightmap)
grid_size = 50
x = np.linspace(-5, 5, grid_size)
y = np.linspace(-5, 5, grid_size)
X, Y = np.meshgrid(x, y)
Z = np.sin(X ** 2 + Y ** 2)  # Example height function

# Create PyVista surface
surface = pv.StructuredGrid(X, Y, Z)

# Create the moving point (initial position)
robot_points = robot_mesh.points
point_cloud = pv.PolyData(robot_points)

# Setup Plotter
plotter = pv.Plotter()
plotter.add_mesh(surface, cmap="terrain", show_edges=False, opacity=0.7)
point_actor = plotter.add_mesh(point_cloud, color="red", point_size=10)

# Show in interactive mode
plotter.show(auto_close=False, interactive=False)

# Animate the point moving in a circular path
num_steps = 100
for i in range(num_steps):
    robot_points[:, 0] += 0.1  # Move the point in the x-direction

    # Update point position
    point_cloud.points = robot_points

    plotter.update()
    plotter.render()  # Refresh the plot
    time.sleep(0.05)  # Adjust speed

plotter.close()
