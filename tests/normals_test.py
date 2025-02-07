import torch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
pio.renderers.default = 'browser'
import matplotlib as mpl
mpl.use('Qt5Agg')


grid_res = 0.1
max_coord = 1.0

# Convention follows that the coordinate system has -max, -max at the top left corner
DIM = int(2 * max_coord / grid_res)
x = torch.linspace(-max_coord, max_coord, DIM)
y = torch.linspace(-max_coord, max_coord, DIM)
x, y = torch.meshgrid(x, y, indexing='xy')  # SUPER IMPORTANT TO USE INDEXING='XY'

# fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
# x_im = ax[0].contourf(x, y, x, cmap='gray', levels=100)
# ax[0].invert_yaxis()
# ax[0].set_title("X")
# ax[0].xaxis.tick_top()  # Move X-axis ticks to the top
# ax[0].tick_params(axis='x', labeltop=True, labelbottom=False)  # Show labels on top and hide them on bottom
#
# y_im = ax[1].contourf(x, y, y, cmap='gray', levels=100)
# ax[1].invert_yaxis()
# ax[1].set_title("Y")
# ax[1].xaxis.tick_top()  # Move X-axis ticks to the top
# ax[1].tick_params(axis='x', labeltop=True, labelbottom=False)  # Show labels on top and hide them on bottom
#
# plt.colorbar(x_im, ax=ax[0])
# plt.colorbar(y_im, ax=ax[1])

# z = torch.zeros((DIM, DIM))
# z[-DIM // 2:, -DIM // 2:] = 1.0
z = torch.exp(-(x - 2) ** 2 / 4) * torch.exp(-(y - 0) ** 2 / 2)

# plt.figure(figsize=(8, 6), dpi=100)
# plt.contourf(x, y, z, cmap='gray', levels=100)
# plt.colorbar()
# plt.title("Z")

# fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'surface'}]])
# fig.add_trace(
#     go.Surface(x=x, y=y, z=z, colorscale='Viridis', showscale=False),
#     row=1, col=1
# )
# fig.update_layout(
#     scene=dict(
#         xaxis_title='X',
#         yaxis_title='Y',
#         zaxis_title='Height (Z)',
#         camera_eye=dict(x=1.25, y=1.25, z=1.25),
#         aspectmode='manual',
#         aspectratio=dict(
#             x=1.,
#             y=1.,
#             z=z.max().item() / (2 * max_coord)
#         ),
#     ),
#     title_text='3D Heightmap',
# )
# fig.update_layout(
#     width=1000,
#     height=1000,
#     margin=dict(l=20, r=20, t=20, b=20)
# )

def compute_heightmap_gradients(z_grid: torch.Tensor, grid_res: float) -> torch.Tensor:
    """
    Computes the gradients of a heightmap along the x and y axes using torch.gradient.

    **Note**: The input grid is assumed to use torch's "xy" indexing convention, i.e., the first dimension is the y-axis and the second dimension is the x-axis.

    Parameters:
    - z_grid: Heightmap tensor of shape (B, H, W).
    - grid_res: Resolution of the grid in meters.
    """
    B, H, W = z_grid.shape
    z_grid = z_grid.squeeze()
    gradients = torch.vstack(torch.gradient(z_grid, spacing=grid_res, dim=(1, 0), edge_order=2))
    return gradients.reshape(B, 2, H, W)

# Gradients from custom function
gradients = compute_heightmap_gradients(z.unsqueeze(0), grid_res)
dz_dx, dz_dy = gradients[0, 0], gradients[0, 1]
fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
fig.suptitle("Gradients from custom function")
dz_dx_im = ax[0].contourf(x, y, dz_dx, cmap='gray', levels=100)
ax[0].set_title("Gradient X")
dz_dy_im = ax[1].contourf(x, y, dz_dy, cmap='gray', levels=100)
ax[1].set_title("Gradient Y")
plt.colorbar(dz_dx_im, ax=ax[0])
plt.colorbar(dz_dy_im, ax=ax[1])


def normalized(x, eps=1e-6):
    """
    Normalizes the input tensor.

    Parameters:
    - x: Input tensor.
    - eps: Small value to avoid division by zero.

    Returns:
    - Normalized tensor.
    """
    norm = torch.norm(x, dim=-1, keepdim=True)
    return x / torch.clamp(norm, min=eps)

def surface_normals(z_grid_grads: torch.Tensor, query: torch.Tensor, max_coord: float) -> torch.Tensor:
    """
    Computes the surface normals and tangents at the queried coordinates.

    Parameters:
    - z_grid_grads: torch.Tensor of gradients of the heightmap along the x and y axes (3D array), (B, 2, H, W).
    - query: Tensor of desired point coordinates for interpolation (3D array), (B, N, 2).

    Returns:
    - Surface normals at the queried coordinates.
    """
    norm_query = query / max_coord  # Normalize to [-1, 1]
    # Query coordinates of shape (B, N, 1, 2)
    B, N = query.shape[:2]
    grid_coords = norm_query.unsqueeze(2)
    # Interpolate the grid values into shape (B, 2, N, 1)
    grad_query = torch.nn.functional.grid_sample(z_grid_grads, grid_coords, align_corners=True, mode="bilinear", padding_mode="border").squeeze(-1).permute(0, 2, 1)  # (B, N, 2)
    # Compute the surface normals
    n = torch.dstack([-grad_query, torch.ones((B, N, 1))])  # n = [-dz/dx, -dz/dy, 1]
    n = normalized(n)
    return n


stacked_grid = torch.stack([x, y, z], dim=2)
flat_grid = stacked_grid.view(-1, 3)
normals = surface_normals(gradients, flat_grid[..., :2].unsqueeze(0), max_coord)
normals_grid = normals.view(DIM, DIM, 3)
n = (normals_grid[..., :2]**2).sum(dim=2).sqrt()
fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
n_im = ax.contourf(x, y, n, cmap='gray', levels=100)
plt.colorbar(n_im, ax=ax)
plt.title("Surface Normals")

plt.show()
