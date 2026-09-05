"""Space-filling sampling and variation for fixed-cardinality categories."""

import numpy as np
from scipy.stats import qmc

from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling


class ScrambledSobolCategoricalSampling(Sampling):
    """Map scrambled Sobol coordinates to uniformly sized category bins."""

    def _do(self, problem, n_samples, *args, random_state=None, **kwargs):
        seed = int(random_state.integers(0, np.iinfo(np.uint32).max))
        exponent = int(np.ceil(np.log2(max(1, n_samples))))
        unit = qmc.Sobol(problem.n_var, scramble=True, seed=seed).random_base2(exponent)
        cardinality = np.asarray(problem.xu, dtype=int) - np.asarray(problem.xl, dtype=int) + 1
        return (
            np.asarray(problem.xl, dtype=int)
            + np.floor(unit[:n_samples] * cardinality).astype(int)
        )


class CategoricalRandomMutation(Mutation):
    """Random-reset mutation that never leaves a selected category unchanged."""

    def _do(self, problem, X, random_state=None, **kwargs):
        X = np.asarray(X, dtype=int).copy()
        probability = self.get_prob_var(problem, size=len(X))
        lower = np.asarray(problem.xl, dtype=int)
        cardinality = np.asarray(problem.xu, dtype=int) - lower + 1
        for row in range(len(X)):
            mutate = random_state.random(problem.n_var) < probability[row]
            if not np.any(mutate):
                mutate[int(random_state.integers(problem.n_var))] = True
            columns = np.where(mutate)[0]
            offsets = random_state.integers(1, cardinality[columns])
            X[row, columns] = lower[columns] + (
                X[row, columns] - lower[columns] + offsets
            ) % cardinality[columns]
        return X
