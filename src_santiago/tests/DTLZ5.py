import matplotlib.pyplot as plt
import numpy as np
from optimizer import MultiObjectiveOptimizer
from problems import dtlz5_f1, dtlz5_f2, dtlz5_f3
import time

# Degenerate front/ works not bad, i have to add the PCA


time_start = time.time()
funcs = [dtlz5_f1, dtlz5_f2, dtlz5_f3]
N_VARS = 12
bounds = [(0.0, 1.0)] * N_VARS


opt = MultiObjectiveOptimizer(funcs, bounds)
opt.fit()


MAX_ITER = 20
plt.ion()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
time_end = time.time()
print(time_end - time_start)

for i in range(MAX_ITER):
    input()
    print(f"--- Iter {i+1} ---")

    model = opt.estimate_front_model(kernel="linear")
    target, gain = opt.predict_next_best_point(model)

    ax.clear()

    u = np.linspace(0, np.pi / 2, 20)
    v = np.linspace(0, np.pi / 2, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.1)

    pts = opt.pareto_front_y
    if not np.isnan(pts).any():
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="red", s=30, label="Solutions")
    else:
        pass

    ax.scatter(target[0], target[1], target[2], c="lime", s=200, marker="*")

    ax.set_title(f"DTLZ5 - Iter {i+1}")
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    ax.view_init(30, 45)

    plt.draw()
    plt.pause(1)
    opt.compute_and_add_point(target)

plt.ioff()
plt.show()
