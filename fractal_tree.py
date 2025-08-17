import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置绘图参数
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.axis('off')

# 颜色渐变函数：绿色到紫色
def color_gradient(level, max_level):
    return (level / max_level, 0, 1 - level / max_level)

# 递归绘制毕达哥拉斯树
def draw_tree(x, y, size, angle, level, max_level):
    if level > max_level or size < 0.01:
        return
    
    # 计算四个顶点坐标
    # 左下角 (x, y)
    # 左上角 (x1, y1)
    dx = size * torch.cos(angle)
    dy = size * torch.sin(angle)
    x1 = x + dx
    y1 = y + dy
    
    # 正方形的另外两个顶点
    perp_angle = angle + torch.tensor(np.pi/2)
    dx2 = size * torch.cos(perp_angle)
    dy2 = size * torch.sin(perp_angle)
    x2 = x1 + dx2
    y2 = y1 + dy2
    x3 = x + dx2
    y3 = y + dy2
    
    # 绘制方块
    xs = [x.item(), x1.item(), x2.item(), x3.item()]
    ys = [y.item(), y1.item(), y2.item(), y3.item()]
    ax.fill(xs, ys, color=color_gradient(level, max_level), edgecolor='black')
    
    # 递归绘制左子树
    new_size = size * 0.7
    new_angle_left = angle + torch.tensor(np.pi/4)
    draw_tree(x3, y3, new_size, new_angle_left, level+1, max_level)
    
    # 递归绘制右子树
    new_angle_right = angle - torch.tensor(np.pi/4)
    draw_tree(x2, y2, new_size, new_angle_right, level+1, max_level)

# 初始化树参数
max_level = 8
size = torch.tensor(1.0, device=device)
x0 = torch.tensor(0.0, device=device)
y0 = torch.tensor(0.0, device=device)
angle0 = torch.tensor(np.pi/2, device=device)

# 绘制树
draw_tree(x0, y0, size, angle0, level=0, max_level=max_level)

plt.show()
