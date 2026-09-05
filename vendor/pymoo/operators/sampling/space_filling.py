"""Deterministic-space-filling samplers for bounded real problems."""

import numpy as np
from scipy.stats import qmc

from pymoo.core.sampling import Sampling


class ScrambledSobolSampling(Sampling):
    """Scrambled Sobol points, seeded by pymoo's run-level RNG."""

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        seed = int(random_state.integers(0, np.iinfo(np.uint32).max))
        exponent = int(np.ceil(np.log2(max(1, n_samples))))
        unit = qmc.Sobol(problem.n_var, scramble=True, seed=seed).random_base2(exponent)[:n_samples]
        xl, xu = problem.bounds()
        return qmc.scale(unit, np.asarray(xl), np.asarray(xu))


class MaximinPoolSampling(Sampling):
    """Select a farthest-first design from an unevaluated random candidate pool."""

    def __init__(self, pool_multiplier=5):
        super().__init__()
        self.pool_multiplier = max(1, int(pool_multiplier))

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        n_pool = self.pool_multiplier * n_samples
        pool = random_state.random((n_pool, problem.n_var))
        selected = [int(random_state.integers(n_pool))]
        nearest_sq = np.sum((pool - pool[selected[0]]) ** 2, axis=1)
        nearest_sq[selected] = -np.inf
        while len(selected) < n_samples:
            idx = int(np.argmax(nearest_sq))
            selected.append(idx)
            distance_sq = np.sum((pool - pool[idx]) ** 2, axis=1)
            nearest_sq = np.minimum(nearest_sq, distance_sq)
            nearest_sq[selected] = -np.inf
        xl, xu = problem.bounds()
        return np.asarray(xl) + pool[selected] * (np.asarray(xu) - np.asarray(xl))
