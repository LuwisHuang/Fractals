# reference: https://en.wikipedia.org/wiki/Dragon_curve
import torch
import matplotlib.pyplot as plt

# -----------------------------
# 1) L-system for Heighway Dragon
# -----------------------------
def generate_dragon(iterations):
    """Generate Heighway Dragon instructions using L-system."""
    seq = "FX"  # starting axiom
    for _ in range(iterations):
        seq_new = ""
        for c in seq:
            if c == "X":
                seq_new += "X+YF+"
            elif c == "Y":
                seq_new += "-FX-Y"
            else:
                seq_new += c
        seq = seq_new
    return seq

# -----------------------------
# 2) Interpret L-system to coordinates
# -----------------------------
def dragon_to_coords(seq, step=1.0):
    """Convert L-system instructions to 2D coordinates using PyTorch tensors."""
    x = [0.0]
    y = [0.0]
    angle = 0.0  # start facing right
    stack = []

    for c in seq:
        if c == "F":
            # move forward
            x.append(x[-1] + step * torch.cos(torch.tensor(angle)))
            y.append(y[-1] + step * torch.sin(torch.tensor(angle)))
        elif c == "+":
            # turn left 90°
            angle += torch.pi/2
        elif c == "-":
            # turn right 90°
            angle -= torch.pi/2
        # ignore X and Y
    return torch.tensor(x), torch.tensor(y)

# -----------------------------
# 3) Generate Dragon Curve
# -----------------------------
iterations = 12  # 可以调整迭代次数
seq = generate_dragon(iterations)
x, y = dragon_to_coords(seq, step=1.0)

# -----------------------------
# 4) Plot using matplotlib
# -----------------------------
plt.figure(figsize=(10,10))
plt.plot(x.numpy(), y.numpy(), color='blue', linewidth=1)
plt.axis('equal')
plt.title(f"Heighway Dragon Fractal ({iterations} iterations)")
plt.show()

