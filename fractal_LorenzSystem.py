# reference: https://en.wikipedia.org/wiki/Lorenz_system
# Lorenz attractor core equations:
# dx/dt = sigma * (y - x)
# dy/dt = x * (rho - z) - y
# dz/dt = x * y - beta * z

import torch
import numpy as np
import plotly.graph_objects as go

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
sigma = 10.0
beta = 8/3
dt = 0.01
num_steps = 50000
x0, y0, z0 = 1.0, 1.0, 1.0  # initial state

rho_vals = np.linspace(20, 40, 9)  # changable rho

trajectories = []

for rho in rho_vals:
    # initialize x, y, z
    x_vals = torch.zeros(num_steps, device=device)
    y_vals = torch.zeros(num_steps, device=device)
    z_vals = torch.zeros(num_steps, device=device)
    
    x_vals[0], y_vals[0], z_vals[0] = x0, y0, z0

    # core loop
    for i in range(1, num_steps):
        
        dx = sigma*(y_vals[i-1] - x_vals[i-1])
        dy = x_vals[i-1]*(rho - z_vals[i-1]) - y_vals[i-1]
        dz = x_vals[i-1]*y_vals[i-1] - beta*z_vals[i-1]
        
        # Euler intergration
        x_vals[i] = x_vals[i-1] + dx*dt
        y_vals[i] = y_vals[i-1] + dy*dt
        z_vals[i] = z_vals[i-1] + dz*dt

    # save to GPU
    trajectories.append((
        x_vals.cpu().numpy(),
        y_vals.cpu().numpy(),
        z_vals.cpu().numpy()
    ))

# draw the initial trace
colors = np.linspace(0,1,num_steps)
fig = go.Figure()
x_vals, y_vals, z_vals = trajectories[0]
fig.add_trace(go.Scatter3d(
    x=x_vals, 
    y=y_vals, 
    z=z_vals,
    mode='lines',
    line=dict(color=colors, colorscale='Viridis', width=3)
))

# add slider
steps = []
for i, rho in enumerate(rho_vals):
    step = dict(
        method='update',
        args=[{"visible":[j==i for j in range(len(rho_vals))]},
              {"title":f"Lorenz Attractor σ={sigma}, ρ={rho:.1f}"}],
        label=f"ρ={rho:.1f}"
    )
    steps.append(step)
sliders = [dict(active=0, pad={"t":50}, steps=steps)]

# add other traces
for i in range(1, len(rho_vals)):
    x_vals, y_vals, z_vals = trajectories[i]
    fig.add_trace(go.Scatter3d(
        x=x_vals, 
        y=y_vals, 
        z=z_vals,
        mode='lines',
        line=dict(color=colors, colorscale='Viridis', width=3),
        visible=False
    ))

# update the plot
fig.update_layout(
    sliders=sliders,
    scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
)

fig.show()
