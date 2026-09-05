"""Python implementation of the published HREA hierarchy-ranking method."""

import numpy as np
from scipy.spatial.distance import cdist

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.population import Population
from pymoo.core.selection import Selection
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def _crowding(X):
    X = np.asarray(X, dtype=float)
    if len(X) <= 1:
        return np.full(len(X), np.inf)
    scale = np.ptp(X, axis=0); scale[scale == 0] = 1.0
    distance = np.sort(cdist((X - X.min(axis=0)) / scale,
                             (X - X.min(axis=0)) / scale), axis=1)[:, 1:]
    reciprocal = np.divide(1.0, distance, out=np.full_like(distance, np.inf), where=distance > 0)
    return (len(X) - 1) / np.sum(reciprocal, axis=1)


def _environmental(pop, size):
    X, F = pop.get("X"), pop.get("F")
    distance = cdist(X, X)
    radius = .2 * np.prod(np.maximum(np.ptp(X, axis=0), 1e-32)) ** (1.0 / X.shape[1])
    dominates = (np.all(F[:, None] <= F[None, :], axis=2)
                 & np.any(F[:, None] < F[None, :], axis=2))
    neighborhood = distance < radius
    local = np.sum(dominates.T & neighborhood, axis=1) / np.maximum(neighborhood.sum(axis=1), 1)
    nearest = np.sort(distance, axis=0)[:min(3, len(pop))].sum(axis=0)
    order = np.lexsort((-nearest, local))[:size]
    selected = pop[order]
    selected.set("crowding", _crowding(selected.get("X")))
    return selected


def _archive_update(pop, size, epsilon, progress):
    """Translate the author's ``ArchiveUpdate.m`` selection order.

    In particular, the remaining candidates must be re-ranked after close
    solutions are deleted.  Precomputing all fronts once is not equivalent:
    deleting an earlier front can promote a later solution into the current
    first front.  The final crowding weights also preserve the author's
    preference for under-filled hierarchy levels.
    """
    effective = epsilon
    if epsilon != 1 and progress < .5:
        effective = 2 * (1 - epsilon) / (2 * progress + 1) + 2 * epsilon - 1
    first_indices = NonDominatedSorting().do(
        pop.get("F"), only_non_dominated_front=True)
    first = pop[first_indices]
    archive = first
    remaining_mask = np.ones(len(pop), dtype=bool)
    remaining_mask[first_indices] = False
    remaining = pop[remaining_mask]
    radius = .2 * np.prod(np.ptp(pop.get("X"), axis=0)) ** (
        1.0 / pop.get("X").shape[1])

    while len(remaining):
        if len(archive):
            keep = np.min(cdist(archive.get("X"), remaining.get("X")), axis=0) >= radius
            remaining = remaining[keep]
        if not len(remaining):
            break
        front_indices = NonDominatedSorting().do(
            remaining.get("F"), only_non_dominated_front=True)
        candidates = remaining[front_indices]
        ranks = NonDominatedSorting().do(np.vstack((
            candidates.get("F") * (1 - effective), first.get("F"))),
            return_rank=True)[1][:len(candidates)]
        archive = Population.merge(archive, candidates[ranks == 0])
        keep = np.ones(len(remaining), dtype=bool)
        keep[front_indices] = False
        remaining = remaining[keep]
        if np.max(ranks) > 0:
            break

    if len(archive) > size:
        ranks = NonDominatedSorting().do(archive.get("F"), return_rank=True)[1]
        kept = []
        weights = []
        per_front = int(np.ceil(size / (np.max(ranks) + 1)))
        overflow = []
        for rank in range(np.max(ranks) + 1):
            ids = np.where(ranks == rank)[0]
            if len(ids) < per_front:
                kept.extend(ids.tolist())
                weights.extend([per_front - len(ids) + 1] * len(ids))
            else:
                overflow.extend(ids.tolist())
        while len(kept) + len(overflow) > size:
            distance = np.sort(cdist(
                archive.get("X")[overflow], archive.get("X")[overflow]), axis=0)
            delete = int(np.argmin(
                distance[:min(3, len(overflow))].sum(axis=0)))
            del overflow[delete]
        archive = archive[kept + overflow]
        weights.extend([1] * len(overflow))
        archive.set("crowding", _crowding(archive.get("X")) * np.asarray(weights))
    else:
        archive.set("crowding", _crowding(archive.get("X")))
    return archive


class _CrowdingTournament(Selection):
    def _do(self, problem, pop, n_select, n_parents, random_state=None, **kwargs):
        score = np.asarray(pop.get("crowding"), dtype=float)
        c = random_state.integers(0, len(pop), size=(n_select, n_parents, 2))
        return np.where(score[c[:, :, 0]] >= score[c[:, :, 1]], c[:, :, 0], c[:, :, 1])


class HREA(GeneticAlgorithm):
    def __init__(self, pop_size=200, epsilon=.3, sampling=None, **kwargs):
        self.epsilon = float(epsilon)
        self.external_archive = Population()
        super().__init__(pop_size=pop_size,
                         sampling=FloatRandomSampling() if sampling is None else sampling,
                         selection=_CrowdingTournament(), crossover=SBX(prob=1.0, eta=20),
                         mutation=PM(eta=20), eliminate_duplicates=False, **kwargs)

    def _initialize_advance(self, infills=None, **kwargs):
        self.pop = _environmental(infills, self.pop_size)
        self.external_archive = _archive_update(infills, self.pop_size, self.epsilon, 0.)

    def _infill(self):
        source = self.pop
        max_eval = getattr(self.termination, "n_max_evals", np.inf)
        progress = self.evaluator.n_eval / max_eval if np.isfinite(max_eval) else 0.
        if progress >= .5 and self.random_state.random() < .5 and len(self.external_archive):
            source = self.external_archive
        return self.mating.do(self.problem, source, self.n_offsprings,
                              algorithm=self, random_state=self.random_state)

    def _advance(self, infills=None, **kwargs):
        self.pop = _environmental(Population.merge(self.pop, infills), self.pop_size)
        max_eval = getattr(self.termination, "n_max_evals", self.evaluator.n_eval)
        progress = min(1., self.evaluator.n_eval / max_eval)
        self.external_archive = _archive_update(
            Population.merge(self.external_archive, infills), self.pop_size,
            self.epsilon, progress)

    def _finalize(self):
        self.pop = self.external_archive
