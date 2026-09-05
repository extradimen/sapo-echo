"""Scalable multimodal many-objective DTLZ-style benchmarks."""

import itertools

import numpy as np
import pymoo.gradient.toolbox as anp

from pymoo.problems.many.dtlz import DTLZ, generic_sphere


class MMDTLZ2(DTLZ):
    """DTLZ2 with multiple equivalent Pareto sets in decision space.

    The angular variables retain the regular DTLZ2 Pareto front. Each distance
    variable contributes ``sin(frequency*pi*x)^2`` to ``g`` and therefore has
    ``frequency + 1`` globally optimal values. With ``k`` distance variables,
    the problem contains ``(frequency + 1) ** k`` equivalent Pareto subsets.
    """

    def __init__(self, n_obj=5, k=2, frequency=3, n_var=None, **kwargs):
        self.frequency = max(1, int(frequency))
        if n_var is None:
            n_var = int(n_obj) - 1 + int(k)
        super().__init__(n_var=n_var, n_obj=n_obj, **kwargs)
        self.n_basins = (self.frequency + 1) ** self.k

    def _evaluate(self, x, out, *args, **kwargs):
        X_, X_M = x[:, :self.n_obj - 1], x[:, self.n_obj - 1:]
        g = anp.sum(anp.sin(self.frequency * anp.pi * X_M) ** 2, axis=1)
        out["F"] = self.obj_func(X_, g, alpha=1)

    def _calc_pareto_front(self, ref_dirs=None):
        if ref_dirs is None:
            rng = np.random.default_rng(1)
            ref_dirs = rng.dirichlet(np.ones(self.n_obj), size=1000)
        return generic_sphere(ref_dirs)

    def pareto_set_by_basin(self, points_per_basin=100, seed=1):
        rng = np.random.default_rng(seed)
        angles = rng.random((int(points_per_basin), self.n_obj - 1))
        optima = np.linspace(0.0, 1.0, self.frequency + 1)
        groups = []
        for tail in itertools.product(optima, repeat=self.k):
            X = np.empty((len(angles), self.n_var), dtype=float)
            X[:, :self.n_obj - 1] = angles
            X[:, self.n_obj - 1:] = np.asarray(tail)
            groups.append(X)
        return groups

    def _calc_pareto_set(self, n_pareto_points=1000):
        per_basin = max(1, int(n_pareto_points) // self.n_basins)
        return np.vstack(self.pareto_set_by_basin(per_basin))
