import matplotlib.pyplot as plt
import numpy as np
from optimizer import MultiObjectiveOptimizer
from problems import dtlz6_f1, dtlz6_f2, dtlz6_f3
import time

# Gradient/ fails to generate the model, needed to do a PCA before?

time_start = time.time()
N_OBJS = 3
k = 10
N_VARS = N_OBJS + k - 1
bounds = [(0.0, 1.0)] * N_VARS
funcs = [dtlz6_f1, dtlz6_f2, dtlz6_f3]

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

    if np.any(target < -1) or np.any(target > 20):
        print("  > [AVISO] El modelo predijo un valor fuera de rango (Outlier visual).")

    if expected_gain < HV_TOLERANCE:
        print("  > [STOP] Converged.")
        break
    ax.clear()
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_zlim(0, 8)

    u = np.linspace(0, np.pi / 2, 20)
    v = np.linspace(0, np.pi / 2, 20)
    x_sph = np.outer(np.cos(u), np.sin(v))
    y_sph = np.outer(np.sin(u), np.sin(v))
    z_sph = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(
        x_sph, y_sph, z_sph, color="gray", alpha=0.1, rstride=2, cstride=2
    )

    pts = opt.pareto_front_y
    ax.scatter(
        pts[:, 0], pts[:, 1], pts[:, 2], c="red", s=30, label=f"Pareto Arc ({len(pts)})"
    )

    if np.all(target > -1) and np.all(target < 15):
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
    ax.set_title(f"DTLZ6 - Iter {i+1}")

    ax.legend()
    ax.view_init(elev=30, azim=45)

    plt.draw()
    plt.pause(0.5)

    opt.compute_and_add_point(target)

plt.ioff()
plt.show()
