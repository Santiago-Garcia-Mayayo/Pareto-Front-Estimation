import matplotlib.pyplot as plt
import numpy as np
from optimizer import MultiObjectiveOptimizer
from problems import zdt2_f1, zdt2_f2

# Concave, same as ZDT1, performs great, but it takes too much time. Weird stop.
# Modify way of curve (more smooth and no repetition) to improve NGSA-II convergence approach.


N_OBJS = 2
N_VARS = 10
bounds = [(0.0, 1.0)] * N_VARS
funcs = [zdt2_f1, zdt2_f2]
opt = MultiObjectiveOptimizer(funcs, bounds)

if not hasattr(opt, "n_dim"):
    opt.n_dim = len(funcs)
if not hasattr(opt, "constraints"):
    opt.constraints = ()

opt.fit()

print("Press enter")
MAX_ITER = 20
HV_TOLERANCE = 1e-5

plt.ion()
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111)

for i in range(MAX_ITER):
    print(f"\nIteration {i+1}/{MAX_ITER}")

    opt.prune_front()

    model = opt.estimate_front_model(kernel="cubic")

    target, expected_gain = opt.predict_next_best_point(model)
    # print(f"Expected Gain: {expected_gain:.6f}")

    if expected_gain < HV_TOLERANCE:
        print("STOP Converged.")
        break

    ax.clear()

    f1_true = np.linspace(0, 1, 200)
    f2_true = 1.0 - np.power(f1_true, 2)
    ax.plot(
        f1_true, f2_true, "k--", linewidth=1.5, alpha=0.6, label="True Front (Theory)"
    )

    f1_grid = np.linspace(0, 1, 100).reshape(-1, 1)
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
    ax.set_ylim(-0.05, 1.4)
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_title(f"ZDT2 (Concave) - Iter {i+1}")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.draw()
    plt.pause(0.5)

    opt.compute_and_add_point(target)

plt.ioff()
plt.show()
