import matplotlib.pyplot as plt
import numpy as np
from optimizer import MultiObjectiveOptimizer
from problems import dtlz1_f1, dtlz1_f2, dtlz1_f3
import time

# Multimodal/ works mew

time_start = time.time()

N_OBJS = 3
k = 5
N_VARS = N_OBJS + k - 1
bounds = [(0.0, 1.0)] * N_VARS
funcs = [dtlz1_f1, dtlz1_f2, dtlz1_f3]

opt = MultiObjectiveOptimizer(funcs, bounds)
opt.fit()

MAX_ITER = 20
HV_TOLERANCE = 1e-5

plt.ion()
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

time_end = time.time()
print(time_end - time_start)


for i in range(MAX_ITER):
    input()
    print(f"\n--- Iteration {i+1}/{MAX_ITER} ---")

    opt.prune_front()
    model = opt.estimate_front_model(kernel="linear")

    target, expected_gain = opt.predict_next_best_point(model)
    print(f"  > Expected Gain: {expected_gain:.6f}")

    if expected_gain < HV_TOLERANCE:
        print("  > [STOP] Converged.")
        break

    ax.clear()

    x = np.linspace(0, 0.5, 10)
    y = np.linspace(0, 0.5, 10)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 - X - Y
    Z[Z < 0] = np.nan

    ax.plot_surface(X, Y, Z, alpha=0.2, color="blue", edgecolor="none")
    ax.plot([0.5, 0], [0, 0.5], [0, 0], "b--", lw=1)
    ax.plot([0, 0], [0.5, 0], [0, 0.5], "b--", lw=1)
    ax.plot([0, 0.5], [0, 0], [0.5, 0], "b--", lw=1)

    pts = opt.pareto_front_y
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c="red", s=30, label=f"Solutions")

    ax.scatter(
        target[0], target[1], target[2], c="lime", s=250, marker="*", edgecolors="black"
    )

    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)
    ax.set_zlim(0, 0.6)
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    ax.set_title(f"DTLZ1 (Linear Front) - Iter {i+1}")
    ax.view_init(elev=30, azim=45)

    plt.draw()
    plt.pause(0.5)
    opt.compute_and_add_point(target)

plt.ioff()
plt.show()
