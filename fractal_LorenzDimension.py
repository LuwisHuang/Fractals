# Grassberger–Procaccia correlation dimension estimation
# 1. Generate Lorenz trajectory for a given rho
# 2. Compute all pairwise distances between points in trajectory
# 3. For a range of radii r, count fraction of pairs with distance < r (correlation sum)
# 4. Plot log(C(r)) vs log(r) and fit a line; slope ≈ correlation dimension

import torch
import numpy as np
from numpy import log, polyfit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sigma = 10.0
beta = 8/3
rho = 28.0  # choose rho = 28
dt = 0.01
num_steps = 30000  
x0, y0, z0 = 1.0, 1.0, 1.0

# initialize tensor
traj = torch.zeros((num_steps, 3), device=device)
traj[0] = torch.tensor([x0, y0, z0], device=device)

# generate track
for i in range(1, num_steps):
    x, y, z = traj[i-1]
    dx = sigma*(y - x)
    dy = x*(rho - z) - y
    dz = x*y - beta*z
    traj[i] = traj[i-1] + dt*torch.tensor([dx, dy, dz], device=device)

# estimate dimension of Lorenz System
# 1. compute distance
diff = traj.unsqueeze(0) - traj.unsqueeze(1)  # shape (num_steps, num_steps, 3)
dist = torch.norm(diff, dim=2)  # shape (num_steps, num_steps)

# 2. choose r values
r_vals = torch.logspace(-2, 1, steps=20, device=device)

C_r = []
for r in r_vals:
    # count values less than r
    count = torch.sum(dist < r).item()
    # except itself
    count -= num_steps  
    C_r.append(count / (num_steps*(num_steps-1)))

C_r = np.array(C_r)
r_vals_np = r_vals.cpu().numpy()

# 3. apply polyfit to log(C(r)) vs log(r) ,slope is the dimension of lorenz system


log_r = np.log(r_vals_np)
log_C = np.log(C_r + 1e-12)  # avoid 0
slope, _ = polyfit(log_r, log_C, 1)

print(f"Estimated correlation dimension ≈ {slope:.3f}")
