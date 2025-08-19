import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
sigma = 10.0
beta = 8/3
rho = 28.0   # ✅ 经典 Lorenz 吸引子
dt = 0.01
num_steps = 100000
x0, y0, z0 = 1.0, 1.0, 1.0

# initialize x, y, z
x_vals = torch.zeros(num_steps, device=device)
y_vals = torch.zeros(num_steps, device=device)
z_vals = torch.zeros(num_steps, device=device)

x_vals[0], y_vals[0], z_vals[0] = x0, y0, z0

# integration loop
for i in range(1, num_steps):
    dx = sigma*(y_vals[i-1] - x_vals[i-1])
    dy = x_vals[i-1]*(rho - z_vals[i-1]) - y_vals[i-1]
    dz = x_vals[i-1]*y_vals[i-1] - beta*z_vals[i-1]
    x_vals[i] = x_vals[i-1] + dx*dt
    y_vals[i] = y_vals[i-1] + dy*dt
    z_vals[i] = z_vals[i-1] + dz*dt

# move to CPU numpy & drop transients
x_vals = x_vals.cpu().numpy()[5000::10]  # ✅ 丢掉前5000步, 下采样
y_vals = y_vals.cpu().numpy()[5000::10]
z_vals = z_vals.cpu().numpy()[5000::10]

# --- Fractal dimension via box-counting ---
def fractal_dimension_boxcount(x, y, z, num_scales=15):
    coords = np.vstack([x, y, z]).T
    min_vals = coords.min(axis=0)
    max_vals = coords.max(axis=0)
    norm_coords = (coords - min_vals) / (max_vals - min_vals + 1e-12)
    
    sizes = np.logspace(0, -3, num_scales)  # ✅ 合理缩放范围
    N = []
    for size in sizes:
        indices = np.floor(norm_coords / size).astype(int)
        N.append(len(np.unique(indices, axis=0)))
    
    coeffs = np.polyfit(np.log(1/sizes), np.log(N), 1)
    return coeffs[0]

dim = fractal_dimension_boxcount(x_vals, y_vals, z_vals)
print(f"Approximate fractal dimension when ρ = {rho}: {dim:.3f}")

