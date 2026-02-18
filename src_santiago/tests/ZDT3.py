import matplotlib.pyplot as plt
import numpy as np
from optimizer import MultiObjectiveOptimizer
from problems import zdt3_f1, zdt3_f2

# Disconnected Pareto Front. Better coverage will be neccesary. Maybe the KKT can
# help for NGSA-II convergence approach. More potential in NGSAII but better now
# the normal is better.


N_OBJS = 2
N_VARS = 10
bounds = [(0.0, 1.0)] * N_VARS
funcs = [zdt3_f1, zdt3_f2]


opt = MultiObjectiveOptimizer(funcs, bounds)


if not hasattr(opt, "n_dim"):
    opt.n_dim = len(funcs)
if not hasattr(opt, "constraints"):
    opt.constraints = ()

opt.fit()

print("Press ENTER")

MAX_ITER = 100
HV_TOLERANCE = 1e-5

plt.ion()
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111)

for i in range(MAX_ITER):
    print(f"\n--- Iteration {i+1}/{MAX_ITER} ---")

    opt.prune_front()

    model = opt.estimate_front_model(kernel="cubic")

    target, expected_gain = opt.predict_next_best_point(model)

    if expected_gain < HV_TOLERANCE:
        print("STOP Converged.")
        break

    ax.clear()

    f1_true = np.linspace(0, 1, 300)
    f2_true = 1.0 - np.sqrt(f1_true) - f1_true * np.sin(10.0 * np.pi * f1_true)
    ax.plot(
        f1_true, f2_true, "k--", linewidth=1.0, alpha=0.4, label="Boundary Function"
    )

    f1_grid = np.linspace(0, 1, 150).reshape(-1, 1)
    try:
        f2_pred = model(f1_grid)
        ax.plot(
            f1_grid, f2_pred, color="dodgerblue", linewidth=2, label="Estimated Model"
        )
    except:
        pass

    pts = opt.pareto_front_y
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c="firebrick",
        s=40,
        label=f"Solutions ({len(pts)})",
        zorder=5,
    )

    ax.scatter(
        target[0],
        target[1],
        c="lime",
        s=250,
        marker="*",
        edgecolors="black",
        label="Next Target",
        zorder=10,
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-1.0, 1.5)

    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_title(f"ZDT3 (Disconnected) - Iter {i+1}")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.draw()
    plt.pause(0.5)

    opt.compute_and_add_point(target)

plt.ioff()
plt.show()
