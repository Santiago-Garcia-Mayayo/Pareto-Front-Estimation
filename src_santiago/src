import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution
from scipy.special import comb
from scipy.interpolate import RBFInterpolator
from itertools import combinations

# Añade esto junto a tus otros imports de scipy
from scipy.cluster.hierarchy import linkage, fcluster


class MultiObjectiveOptimizer:
    def __init__(self, objectives, bounds, constraints=None):
        """
        Initializes the Multi-Objective Optimizer.

        Args:
            objectives (list/tuple): list of objective functions
            bounds (list/tuple): list of tuples with variable bounds
            constraints (list/tuple, optional): constraints for the optimizer. If it is not the correct type we correct it. Defaults to None.
        """
        self.funcs = objectives
        self.bounds = bounds
        self.n_dim = len(objectives)

        # Done because some solvers need a list or a tuple of constraints, even if it's just one. If None, we set it to an empty tuple.
        if constraints is not None:
            if not isinstance(constraints, (list, tuple)):
                self.constraints = (constraints,)
            else:
                self.constraints = constraints
        else:
            self.constraints = ()

        self.ideal_point = None
        self.nadir_point = None
        self.denom = None
        self.anchor_x = []
        self.pareto_front_x = []
        self.pareto_front_y = []

    def _get_optimal_divisions(self):
        """
        Computes the minimal number of divisions that creates a grid of at least 10 times
        the dimension of the objective space. This is a common heuristic to ensure a good
        approximation of the Pareto front without creating an excessively large number of points.
        """
        min_target = 10 * self.n_dim
        p = 1
        while True:
            count = comb(self.n_dim + p - 1, p, exact=True)
            if count >= min_target:
                return p, count
            p += 1

    def _generate_grid_weights(self, divisions):
        """
        Generates the set of vectors to initialise the Pareto Front. Are equidistant to
        each other. It is totally dependent to the previous function, wehre the number of
        divisions is calculated. It returns a list of weights where all of them sum 1.
        """
        n = self.n_dim

        total_slots = divisions + n - 1

        weights = []

        for cuts in combinations(range(total_slots), n - 1):

            augmented_cuts = (-1,) + cuts + (total_slots,)

            point = []
            for i in range(n):
                val = augmented_cuts[i + 1] - augmented_cuts[i] - 1
                point.append(val)

            weights.append(point)

        return np.array(weights) / divisions

    def fit(self):
        """
        We compute here the limits of our Pareto Front, and we generate the other points
        of the front using a Tchebycheff scalarization with the weights generated in the
        previous function.
        """
        start_time = time.time()

        n_vars = len(self.bounds)

        sampler = np.random.uniform(
            low=[b[0] for b in self.bounds],
            high=[b[1] for b in self.bounds],
            size=(100, n_vars),
        )
        sample_vals = np.array([[f(x) for f in self.funcs] for x in sampler])

        approx_ideal = np.min(sample_vals, axis=0)
        approx_nadir = np.max(sample_vals, axis=0)
        denom = approx_nadir - approx_ideal
        denom[denom < 1e-6] = 1e-6

        self.anchor_x = []
        payoff_matrix = []

        for i in range(self.n_dim):

            w = np.full(self.n_dim, 1e6)
            w[i] = 1.0

            def hierarchical_obj(x):
                vals = np.array([f(x) for f in self.funcs])
                return np.sum(w * vals)

            res = differential_evolution(
                hierarchical_obj,
                self.bounds,
                constraints=self.constraints,
                strategy="best1bin",
                maxiter=800,
                popsize=20,
                polish=True,
                tol=1e-6,
            )

            if res.success or res.x is not None:
                true_vals = np.array([f(res.x) for f in self.funcs])
                self.anchor_x.append(res.x)
                payoff_matrix.append(true_vals)
                print(f"Anchor {i+1}: {np.round(true_vals, 4)}")
            else:
                print(f"No anchor in position {i+1}")

        if not payoff_matrix:
            payoff_matrix.append(approx_ideal)

        payoff_matrix = np.array(payoff_matrix)

        self.ideal_point = np.min(payoff_matrix, axis=0)
        self.nadir_point = np.max(payoff_matrix, axis=0)

        self.denom = self.nadir_point - self.ideal_point
        self.denom[self.denom < 1e-6] = 1e-6

        p, count = self._get_optimal_divisions()
        print(f"\nCreating {count} different points")
        weights = self._generate_grid_weights(p)

        self.pareto_front_x = []
        self.pareto_front_y = []

        for i, w in enumerate(weights):
            w_norm = w / np.linalg.norm(w)

            def scalar_func(x):
                vals = np.array([f(x) for f in self.funcs])
                diff = vals - self.ideal_point

                d1 = np.dot(diff, w_norm)

                d2 = np.linalg.norm(diff - (d1 * w_norm))

                theta = 5.0
                return d1 + theta * d2

            res = differential_evolution(
                scalar_func,
                self.bounds,
                constraints=self.constraints,
                strategy="best1bin",
                popsize=25,
                maxiter=400,
                polish=True,
                tol=1e-5,
            )

            if res.success or res.x is not None:
                sol_y = np.array([f(res.x) for f in self.funcs])
                self.pareto_front_x.append(res.x)
                self.pareto_front_y.append(sol_y)

                if i % 10 == 0 or i == len(weights) - 1:
                    print(f"Point {i+1}/{len(weights)} computed")

        self.pareto_front_x = np.array(self.pareto_front_x)
        self.pareto_front_y = np.array(self.pareto_front_y)
        return self.pareto_front_x, self.pareto_front_y

    def estimate_front_model(self, kernel="linear"):
        """
        Builds RBF model we could predict the last objective from the others.
        This is a simple surrogate model that captures the structure of the Pareto front.
        We use a small smoothing factor to ensure numerical stability, especially if points are very close to each other.
        """
        candidates = np.unique(np.round(self.pareto_front_y, 9), axis=0)
        is_efficient = np.ones(candidates.shape[0], dtype=bool)

        for i, c in enumerate(candidates):
            if is_efficient[i]:
                is_dominated = np.any(
                    np.all(candidates <= c, axis=1) & np.any(candidates < c, axis=1)
                )
                if is_dominated:
                    is_efficient[i] = False

        pareto_y = candidates[is_efficient]

        X_train = pareto_y[:, :-1]
        y_train = pareto_y[:, -1]
        try:
            rbf = RBFInterpolator(X_train, y_train, kernel=kernel, smoothing=1e-8)
        except np.linalg.LinAlgError:
            rbf = RBFInterpolator(X_train, y_train, kernel=kernel, smoothing=1e-5)

        def predictor(partial_objectives):
            partial_objectives = np.array(partial_objectives)
            is_single = partial_objectives.ndim == 1
            if is_single:
                partial_objectives = partial_objectives.reshape(1, -1)

            result = rbf(partial_objectives)

            if is_single:
                return result[0]
            else:
                return result

        return predictor

    def get_hypervolume_contribution(self, candidate_point, ref_point, mc_samples=5000):
        """
        Monte Carlo estimation of Hypervolume Contribution. We count it by sampling
        random points in the objective space and checking how many are dominated by
        the candidate but not by the existing front. This gives us an estimate of the
        contribution of the candidate point to the hypervolume.
        """
        rng = np.random.default_rng(42)
        min_b = self.ideal_point
        max_b = ref_point
        samples = rng.uniform(low=min_b, high=max_b, size=(mc_samples, self.n_dim))

        dominated_by_existing = np.zeros(mc_samples, dtype=bool)
        for pt in self.pareto_front_y:
            is_dom = np.all(pt <= samples, axis=1)
            dominated_by_existing = dominated_by_existing | is_dom

        dominated_by_candidate = np.all(candidate_point <= samples, axis=1)
        contribution_mask = dominated_by_candidate & (~dominated_by_existing)
        return np.sum(contribution_mask)

    def predict_next_best_point(self, model):
        """
        Finds target coordinates that maximize Hypervolume  using a genetic algorithm
        over the prediction model.
        """
        ref_point = self.nadir_point * 1.1
        current_data = self.pareto_front_y[:, :-1]
        search_bounds = []
        for i in range(current_data.shape[1]):
            mn, mx = np.min(current_data[:, i]), np.max(current_data[:, i])
            span = mx - mn
            search_bounds.append((mn - 0.05 * span, mx + 0.05 * span))

        def hv_objective(x_input):
            f_last = model(x_input)
            candidate = np.append(x_input, f_last)
            contrib = self.get_hypervolume_contribution(
                candidate, ref_point, mc_samples=2000
            )
            return -contrib

        res = differential_evolution(
            hv_objective,
            bounds=search_bounds,
            strategy="best1bin",
            maxiter=100,
            popsize=15,
            tol=1e-3,
        )

        best_input = res.x
        best_last = model(best_input)
        best_point = np.append(best_input, best_last)
        expected_gain = -res.fun

        return best_point, expected_gain

    def compute_and_add_point(self, target_coords):
        """
        Finds REAL x variables for target using Local Search + Warm Start.
        """
        dists = np.linalg.norm(self.pareto_front_y - target_coords, axis=1)
        idx_nearest = np.argmin(dists)

        x0 = self.pareto_front_x[idx_nearest]
        x0 = np.array(
            [np.random.uniform(b[0], b[1]) for b in self.bounds]
        )  # No warm start

        def goal_attainment(x):
            vals = np.array([f(x) for f in self.funcs])
            return np.sum((vals - target_coords) ** 2)

        res = minimize(
            goal_attainment,
            x0,
            method="SLSQP",  # We can try our quasi newton here
            bounds=self.bounds,
            constraints=self.constraints,
            options={"maxiter": 50, "ftol": 1e-4},
        )

        if res.success or res.x is not None:
            real_x = res.x
            real_y = np.array([f(res.x) for f in self.funcs])

            old_count = len(self.pareto_front_y)
            self.pareto_front_x = np.vstack([self.pareto_front_x, real_x])
            self.pareto_front_y = np.vstack([self.pareto_front_y, real_y])
            new_count = len(self.pareto_front_y)

            dist_error = np.linalg.norm(real_y - target_coords)
            print(f"  > Success! Point Added. Error vs Target: {dist_error:.4f}")

            return real_y

        return None

    def prune_front(self):
        """
        Eliminate dominated points from the Pareto front.
        """
        candidates = self.pareto_front_y
        n_points = len(candidates)
        is_efficient = np.ones(n_points, dtype=bool)

        for i in range(n_points):
            if is_efficient[i]:
                for j in range(n_points):
                    if i == j:
                        continue
                    if np.all(candidates[j] <= candidates[i]) and np.any(
                        candidates[j] < candidates[i]
                    ):
                        is_efficient[i] = False
                        break

        if not np.all(is_efficient):
            self.pareto_front_y = self.pareto_front_y[is_efficient]
            self.pareto_front_x = self.pareto_front_x[is_efficient]
