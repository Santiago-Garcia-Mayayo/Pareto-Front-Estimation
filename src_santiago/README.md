# Quasi-Newton Pareto Front Estimator

This directory contains the original source code, test suites, and iterative approximation models developed for my Master's Thesis.

**Work in Progress...**

The module is organized as follows:

* **/src** - Contains the core implementation in the file optimizer.py, including the `MultiObjectiveOptimizer` class, the RBF surrogate model, and a pruning strategy.
* **/tests** - Contains the verification suites to ensure the mathematical and algorithmic correctness of the estimators.

**1. Running the Optimizer:**
The core logic relies heavily on vectorized `NumPy` operations and `SciPy` optimization routines (`differential_evolution`, `minimize` with SLSQP). To initialize and run the estimator:
```python
from src.optimizer import MultiObjectiveOptimizer

# Initialize with your objective functions and bounds
optimizer = MultiObjectiveOptimizer(objectives=[f1, f2], bounds=[(0, 1), (0, 1)])
pareto_x, pareto_y = optimizer.fit()


**2. Running the Tests (Software Verification):**

Rigorous verification and validation are critical for the reliability of the Quasi-Newton optimizer and the surrogate models. The test suite is specifically designed to ensure mathematical correctness, validate edge cases in the objective space, and prevent regressions when integrating with the upstream exact Hypervolume framework.

To execute the automated test suite, ensure you have `pytest` installed. Run the following command from the root of this module (the `quasi_newton_estimator/` directory):

```bash
> pytest tests/ -v

> pytest tests/ --cov=src/
