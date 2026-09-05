"""Python implementation of CPDEA for reproducible MMOP comparisons."""

import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.core.selection import Selection
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def _dominance_matrix(F):
    less_equal = F[:, None, :] <= F[None, :, :]
    strictly_less = F[:, None, :] < F[None, :, :]
    # D[i,j] = 1 when j dominates i, matching CPDEA's LocalC convention.
    return (np.all(less_equal.transpose(1, 0, 2), axis=2)
            & np.any(strictly_less.transpose(1, 0, 2), axis=2)).astype(float)


def _cpd_scores(problem, pop, k=3, eta=2.0):
    X, F = np.asarray(pop.get("X")), np.asarray(pop.get("F"))
    n = len(pop)
    distance = cdist(X, X)
    volume = float(np.prod(np.asarray(problem.xu) - np.asarray(problem.xl)))
    radius = eta * (volume / n) ** (1.0 / problem.n_var)
    kernel = np.exp(-(distance ** 2) / (2 * radius ** 2)) / (radius * np.sqrt(2 * np.pi))
    local = np.sum(_dominance_matrix(F) * kernel.T, axis=1)
    transformed = distance / (1.0 + (local[:, None] + local[None, :]) / 2.0)
    np.fill_diagonal(transformed, np.inf)
    nearest = np.partition(transformed, min(k, n - 1) - 1, axis=0)[:min(k, n - 1)]
    return 1.0 / (1.0 + np.sum(nearest, axis=0))


def _archive_update(pop, size, k=3):
    nd = NonDominatedSorting().do(pop.get("F"), only_non_dominated_front=True)
    pop = pop[nd]
    if len(pop) <= size:
        pop.set("density", _double_density(pop, k))
        return pop
    X, F = pop.get("X"), pop.get("F")
    chosen = np.ones(len(pop), dtype=bool)
    while np.sum(chosen) > size:
        indices = np.where(chosen)[0]
        d_obj, d_dec = cdist(F[indices], F[indices]), cdist(X[indices], X[indices])
        np.fill_diagonal(d_obj, np.inf); np.fill_diagonal(d_dec, np.inf)
        kk = min(k, len(indices) - 1)
        obj_sum = np.partition(d_obj, kk - 1, axis=0)[:kk].sum(axis=0)
        dec_sum = np.partition(d_dec, kk - 1, axis=0)[:kk].sum(axis=0)
        ao, ad = np.mean(obj_sum), np.mean(dec_sum)
        score = 1.0 / (1.0 + obj_sum / (ao if ao else np.inf)
                       + dec_sum / (ad if ad else np.inf))
        chosen[indices[int(np.argmax(score))]] = False
    pop = pop[chosen]
    pop.set("density", _double_density(pop, k))
    return pop


def _double_density(pop, k=3):
    if len(pop) <= k:
        return np.zeros(len(pop))
    d_obj, d_dec = cdist(pop.get("F"), pop.get("F")), cdist(pop.get("X"), pop.get("X"))
    np.fill_diagonal(d_obj, np.inf); np.fill_diagonal(d_dec, np.inf)
    obj_sum = np.partition(d_obj, k - 1, axis=0)[:k].sum(axis=0)
    dec_sum = np.partition(d_dec, k - 1, axis=0)[:k].sum(axis=0)
    ao, ad = np.mean(obj_sum), np.mean(dec_sum)
    return 1.0 / (1.0 + obj_sum / (ao if ao else np.inf)
                  + dec_sum / (ad if ad else np.inf))


class _CPDTournament(Selection):
    def _do(self, problem, pop, n_select, n_parents, random_state=None, **kwargs):
        score = np.asarray(pop.get("cpd"), dtype=float)
        candidates = random_state.integers(0, len(pop), size=(n_select, n_parents, 2))
        return np.where(score[candidates[:, :, 0]] <= score[candidates[:, :, 1]],
                        candidates[:, :, 0], candidates[:, :, 1])


class CPDEA(GeneticAlgorithm):
    """Convergence-Penalized Density Evolutionary Algorithm."""

    def __init__(self, pop_size=200, k=3, eta=2.0, sampling=None, **kwargs):
        self.k, self.eta = int(k), float(eta)
        self.external_archive = Population()
        super().__init__(pop_size=pop_size,
                         sampling=FloatRandomSampling() if sampling is None else sampling,
                         selection=_CPDTournament(), crossover=SBX(prob=1.0, eta=20),
                         mutation=PM(eta=20), n_offsprings=1,
                         eliminate_duplicates=False, **kwargs)

    def _initialize_infill(self):
        return self.initialization.do(self.problem, self.pop_size + 1,
                                      algorithm=self, random_state=self.random_state)

    def _initialize_advance(self, infills=None, **kwargs):
        self.external_archive = _archive_update(infills, self.pop_size, self.k)
        scores = _cpd_scores(self.problem, infills, self.k, self.eta)
        delete = int(np.argmax(scores))
        self.pop = infills[np.arange(len(infills)) != delete]
        self.pop.set("cpd", np.delete(scores, delete))

    def _advance(self, infills=None, **kwargs):
        merged = Population.merge(self.pop, infills)
        scores = _cpd_scores(self.problem, merged, self.k, self.eta)
        delete = int(np.argmax(scores))
        self.pop = merged[np.arange(len(merged)) != delete]
        self.pop.set("cpd", np.delete(scores, delete))
        self.external_archive = _archive_update(
            Population.merge(self.external_archive, infills), self.pop_size, self.k)

    def _infill(self):
        max_eval = getattr(self.termination, "n_max_evals", np.inf)
        late = np.isfinite(max_eval) and self.evaluator.n_eval > .5 * max_eval
        if late and len(self.external_archive) > self.k and self.random_state.random() > .5:
            density = self.external_archive.get("density")
            candidates = np.where(np.isclose(density, np.min(density)))[0]
            p1 = int(self.random_state.choice(candidates))
            pool = Population.merge(self.external_archive, self.pop)
            distances = cdist(self.external_archive[p1].get("X")[None, :], pool.get("X"))[0]
            nearest = np.argsort(distances)[1:min(self.problem.n_var + 1, len(pool))]
            p2 = int(self.random_state.choice(nearest))
            parents = Population.merge(self.external_archive[p1:p1 + 1], pool[p2:p2 + 1])
            off = self.mating.crossover.do(self.problem, parents,
                                            parents=np.array([[0, 1]]),
                                            random_state=self.random_state)
            off = off[self.random_state.integers(len(off)):][:1]
            return self.mating.mutation.do(self.problem, off,
                                           random_state=self.random_state)
        return super()._infill()

    def _finalize(self):
        self.pop = self.external_archive
