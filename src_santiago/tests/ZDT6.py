import matplotlib.pyplot as plt
import numpy as np
from optimizer2 import MultiObjectiveOptimizer
from problems import zdt6_f1, zdt6_f2

# ==========================================
# CONFIGURACIÓN ZDT6 (Densidad Sesgada)
# ==========================================
N_OBJS = 2
N_VARS = 10
bounds = [(0.0, 1.0)] * N_VARS
funcs = [zdt6_f1, zdt6_f2]

print("=== RUNNING ZDT6 (BIASED DENSITY / CONCAVE) ===")
print("El reto: Evitar que los puntos se amontonen y lograr un 'spread' uniforme.")

# 1. Fase Inicial (Auto-Grid PBI)
opt = MultiObjectiveOptimizer(funcs, bounds)

# Aseguramos atributos
if not hasattr(opt, "n_dim"):
    opt.n_dim = len(funcs)
if not hasattr(opt, "constraints"):
    opt.constraints = ()

opt.fit()

print(f" > Frente inicial generado con {len(opt.pareto_front_y)} puntos.")

# ==========================================
# PAUSA MANUAL
# ==========================================
print("\n" + "=" * 50)
input(
    " [PAUSA] Inicialización completa. Presiona ENTER para ver la lucha por la distribución..."
)
print("=" * 50 + "\n")

# 2. Configuración del Bucle
MAX_ITER = 200
HV_TOLERANCE = 1e-5

plt.ion()
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111)

for i in range(MAX_ITER):
    print(f"\n--- Iteration {i+1}/{MAX_ITER} ---")

    opt.prune_front()

    # Kernel 'cubic' va excelente para la curva f2 = 1 - f1^2
    model = opt.estimate_front_model(kernel="cubic")

    target, expected_gain = opt.predict_next_best_point(model)
    print(f"  > Expected Gain: {expected_gain:.6f}")

    if expected_gain < HV_TOLERANCE:
        print("  > [STOP] Converged. Gain is negligible.")
        break

    # VISUALIZACIÓN
    ax.clear()

    # 1. Frente Teórico (Gris discontinuo)
    # Geométricamente es idéntico a ZDT2: f2 = 1 - f1^2
    f1_true = np.linspace(0, 1, 200)
    f2_true = 1.0 - np.power(f1_true, 2)
    ax.plot(f1_true, f2_true, "k--", linewidth=1.5, alpha=0.4, label="Analytical Front")

    # 2. MODELO ESTIMADO (Soporta múltiples islas si hay huecos muy grandes)
    if hasattr(model, "islands"):
        for idx, island in enumerate(model.islands):
            b_min = island["bounds"][0][0]
            b_max = island["bounds"][0][1]

            grid_x = np.linspace(b_min, b_max, 50).reshape(-1, 1)
            try:
                pred_y = island["model"](grid_x)
                label = "Estimated Model" if idx == 0 else ""
                ax.plot(grid_x, pred_y, color="dodgerblue", linewidth=2.5, label=label)
            except:
                pass

    # 3. Puntos Reales (Rojos)
    pts = opt.pareto_front_y
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c="firebrick",
        s=40,
        label=f"Solutions ({len(pts)})",
        zorder=5,
    )

    # 4. Target (Verde)
    ax.scatter(
        target[0],
        target[1],
        c="lime",
        s=250,
        marker="*",
        edgecolors="black",
        label="Next Target (EHVI)",
        zorder=10,
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.4)
    ax.set_xlabel("Objective 1 (f1)")
    ax.set_ylabel("Objective 2 (f2)")
    ax.set_title(f"ZDT6 (Biased Density) - Iter {i+1}")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.draw()
    plt.pause(0.5)

    opt.compute_and_add_point(target)

plt.ioff()
print("\n=== Optimization Finished ===")
plt.show()
