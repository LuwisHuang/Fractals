# reference: https://en.wikipedia.org/wiki/Lorenz_system
# Lorenz attractor core equations:
# dx/dt = sigma * (y - x)
# dy/dt = x * (rho - z) - y
# dz/dt = x * y - beta * z

import torch
import numpy as np
import plotly.graph_objects as go # using ploty to draw 3d images on website


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameter
sigma = 10.0
beta = 8/3
dt = 0.01
num_steps = 50000
x0, y0, z0 = 1.0, 1.0, 1.0

rho_vals = torch.linspace(20, 40, 9, device=device)  # 9 different tracks (9 rhos)
num_rho = len(rho_vals)

# initialize the tensor: shape = (num_rho, num_steps, 3)
traj = torch.zeros((num_rho, num_steps, 3), device=device)
traj[:, 0, :] = torch.tensor([x0, y0, z0], device=device)

# core iteration using Euler method, computing all rhos at one time (Using GPU)
for i in range(1, num_steps):
    x, y, z = traj[:, i-1, 0], traj[:, i-1, 1], traj[:, i-1, 2]
    dx = sigma * (y - x)
    dy = x * (rho_vals - z) - y
    dz = x * y - beta * z
    traj[:, i, 0] = x + dx * dt
    traj[:, i, 1] = y + dy * dt
    traj[:, i, 2] = z + dz * dt


traj_np = traj.cpu().numpy() # copy the number to cpu

# draw the graph
colors = np.linspace(0, 1, num_steps)
fig = go.Figure()

for i in range(num_rho):
    fig.add_trace(go.Scatter3d(
        x=traj_np[i, :, 0],
        y=traj_np[i, :, 1],
        z=traj_np[i, :, 2],
        mode='lines',
        line=dict(color=colors, colorscale='Viridis', width=3),
        visible=(i==0)
    ))

# slider
steps = []
for i, rho in enumerate(rho_vals.cpu().numpy()):
    step = dict(
        method='update',
        args=[{"visible":[j==i for j in range(num_rho)]},
              {"title":f"Lorenz Attractor σ={sigma}, ρ={rho:.1f}"}],
        label=f"ρ={rho:.1f}"
    )
    steps.append(step)

sliders = [dict(active=0, pad={"t":50}, steps=steps)]
fig.update_layout(
    sliders=sliders,
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
)
fig.show()
