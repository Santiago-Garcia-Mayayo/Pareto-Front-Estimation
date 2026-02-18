import matplotlib.pyplot as plt
import numpy as np
from optimizer import MultiObjectiveOptimizer
from problems import dtlz7_f1, dtlz7_f2, dtlz7_f3
import time

# Separate Pareto Front/not great

time_start = time.time()
N_OBJS = 3
k = 20
N_VARS = N_OBJS - 1 + k
bounds = [(0.0, 1.0)] * N_VARS
funcs = [dtlz7_f1, dtlz7_f2, dtlz7_f3]

ref_points = []
for _ in range(2000):
    r_f1 = np.random.rand()
    r_f2 = np.random.rand()
    h = 3.0 - (
        r_f1 * (1.0 + np.sin(3.0 * np.pi * r_f1))
        + r_f2 * (1.0 + np.sin(3.0 * np.pi * r_f2))
    )
    if h >= 0:
        ref_points.append([r_f1, r_f2, h])
ref_points = np.array(ref_points)

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

    model = opt.estimate_front_model(kernel="thin_plate_spline")

    target, expected_gain = opt.predict_next_best_point(model)
    print(f"  > Expected Gain: {expected_gain:.6f}")

    if expected_gain < HV_TOLERANCE:
        print("  > [STOP] Converged.")
        break

    ax.clear()

    ax.scatter(
        ref_points[:, 0],
        ref_points[:, 1],
        ref_points[:, 2],
        color="gray",
        alpha=0.05,
        s=5,
        label="True Regions",
    )

    pts = opt.pareto_front_y
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2], c="red", s=30, label=f"Solutions ({len(pts)})"
    )

    ax.scatter(
        target[0],
        target[1],
        target[2],
        c="lime",
        s=250,
        marker="*",
        edgecolors="black",
        zorder=10,
    )

    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    ax.set_title(f"DTLZ7 (Disconnected) - Iter {i+1}")

    ax.view_init(elev=20, azim=135)

    plt.draw()
    plt.pause(0.5)

    opt.compute_and_add_point(target)

plt.ioff()
plt.show()
