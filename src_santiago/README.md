# Quasi-Newton Pareto Front Estimator

This directory contains the original source code, test suites, and iterative approximation models developed for my Master's Thesis.

**Work in Progress...**

## Organisation
The module is organized as follows:

* **/src** - Contains the core implementation in the file optimizer.py, including the `MultiObjectiveOptimizer` class, the RBF surrogate model, and a pruning strategy.
* **/tests** - Contains the executable benchmark test files to test the algorithm.

## Running the Optimizer:
First, to pip install the requirements of the global project, or just execute:
```bash
> pip install numpy scipy matplotlib
```
The best way to evaluate this module is by running one of the provided experiment examples. These scripts will execute the iterative optimization loop and launch a live interactive plot showing the algorithm approximating the Pareto Front step-by-step. You need to execute:
```bash
> python test_to_run.py
```

## Future work
As this module is actively being developed for my Master's Thesis, the following objectives are planned:

* **Quasi-Newton Integration:** The directional search along the approximated Pareto Front will be fully powered by the custom Quasi-Newton algorithm developed in the broader scope of this thesis.
* **Advanced Front Management:** Further enhancements regarding the treatment of the Pareto Front, including clustering techniques, sparsity control, and rigorous convergence criteria.
* **Extensive Benchmarking:** Comprehensive performance evaluations against other state-of-the-art multi-objective optimization algorithms to rigorously validate the estimator's efficiency.
