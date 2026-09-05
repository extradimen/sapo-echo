"""Offline portfolio selectors for SAPO/ECHO attribution experiments.

The functions in this module never evaluate, mate, or alter search solutions.
They select a fixed-cardinality output from an already evaluated candidate
pool.  This separation lets several portfolio policies share exactly the same
search trajectory and objective-evaluation budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymoo.algorithms.moo.nsga3 import ReferenceDirectionSurvival
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions


SELECTORS = (
    "objective_only",
    "random_quality_matched",
    "edge_jaccard",
    "minimum_common_edge",
    "kmedoids_edge",
    "common_vertex_overlap",
    "xu2025_paper_vertex",
    "xu2025_official_edge",
    "echo",
)


@dataclass(frozen=True)
class PortfolioSelection:
    """A deterministic description of one selected portfolio."""

    X: np.ndarray
    F: np.ndarray
    source_indices: np.ndarray
    quality_core_size: int
    selector: str


def _jaccard(left, right):
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = np.count_nonzero(left | right)
    return 0.0 if union == 0 else 1.0 - np.count_nonzero(left & right) / union


def _common_fraction(left, right):
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    denominator = min(np.count_nonzero(left), np.count_nonzero(right))
    if denominator == 0:
        return 0.0
    return np.count_nonzero(left & right) / denominator


def _same_objective(F, left, right):
    return np.allclose(F[left], F[right], atol=1e-12, rtol=0.0)


def _deduplicated_nondominated_pool(problem, X, F):
    X = np.asarray(X)
    F = np.asarray(F, dtype=float)
    if len(X) != len(F):
        raise ValueError("X and F must contain the same number of candidates")
    if len(X) == 0:
        raise ValueError("candidate pool is empty")
    identities = np.asarray(problem.decision_identity(X), dtype=str)
    _, first = np.unique(identities, return_index=True)
    first = np.sort(first)
    X, F = X[first], F[first]
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return X[front], F[front], first[front]


def _objective_core(problem, X, F, size, ref_dirs, random_seed):
    if size <= 0:
        return []
    if size >= len(X):
        return list(range(len(X)))
    population = Population.new(X=X, F=F)
    survival = ReferenceDirectionSurvival(ref_dirs)
    core = survival.do(
        problem,
        population,
        n_survive=size,
        random_state=np.random.default_rng(random_seed),
    )
    lookup = {tuple(np.asarray(row).tolist()): index for index, row in enumerate(X)}
    return [lookup[tuple(np.asarray(row).tolist())] for row in core.get("X")]


def _echo_score(events, F, selected, candidates, index):
    group = [other for other in selected if _same_objective(F, index, other)]
    peers = [other for other in candidates
             if other != index and _same_objective(F, index, other)]
    if group:
        before = events[group]
        before_score = np.count_nonzero(
            np.any(before, axis=0) & ~np.all(before, axis=0))
        after = events[group + [index]]
        after_score = np.count_nonzero(
            np.any(after, axis=0) & ~np.all(after, axis=0))
        realized = int(after_score - before_score)
    else:
        realized = 0
    potential = max(
        (int(np.count_nonzero(events[index] != events[other])) for other in peers),
        default=0,
    )
    return realized, potential


def _distance_score(embedding, F, selected, candidates, index, distance):
    group = [other for other in selected if _same_objective(F, index, other)]
    if group:
        return min(distance(embedding[index], embedding[other]) for other in group)
    peers = [other for other in candidates
             if other != index and _same_objective(F, index, other)]
    return max(
        (distance(embedding[index], embedding[other]) for other in peers),
        default=0.0,
    )


def _facility_location_score(events, F, selected, candidates, index):
    """Greedy k-medoids/facility-location improvement within an F group."""
    universe = [other for other in candidates
                if _same_objective(F, index, other)]
    group = [other for other in selected if _same_objective(F, index, other)]
    if not universe:
        return 0.0
    if not group:
        # The first medoid is the most central member of its exact-F group.
        return -sum(_jaccard(events[index], events[other]) for other in universe)
    improvement = 0.0
    for other in universe:
        before = min(_jaccard(events[other], events[medoid]) for medoid in group)
        after = min(before, _jaccard(events[other], events[index]))
        improvement += before - after
    return improvement


def _xu2025_rank_scores(contains, counts=None, denominators=None,
                        include_self=True):
    """Return the fixed within-rank path-similarity scores of Xu et al.

    ``contains`` describes the resources present in path i. ``counts``
    describes resource occurrences in path j.  With binary counts this is the
    vertex formula printed as Eq. (3); with directed-edge occurrence counts
    and ``include_self=False`` it follows the authors' Java ``NSort`` code.
    Lower values are preferred.  This is a rank-wide score, not a greedy
    distance from the solutions already selected.
    """
    contains = np.asarray(contains, dtype=bool)
    counts = contains.astype(float) if counts is None else np.asarray(
        counts, dtype=float)
    if contains.shape != counts.shape:
        raise ValueError("contains and counts must have identical shapes")
    overlap = contains.astype(float) @ counts.T
    if not include_self:
        overlap[np.diag_indices_from(overlap)] = 0.0
    if denominators is None:
        denominators = np.count_nonzero(contains, axis=1)
    denominators = np.asarray(denominators, dtype=float)
    if denominators.shape != (len(contains),):
        raise ValueError("denominators must contain one value per path")
    return overlap.sum(axis=1) / np.maximum(1.0, denominators)


def select_portfolio(problem, X, F, size, *, selector="echo",
                     quality_fraction=.7, random_seed=0,
                     vertex_embedding=None, directed_edge_counts=None,
                     path_vertex_counts=None, ref_dirs=None,
                     pad_to_size=True):
    """Select a quality-controlled portfolio from one evaluated pool.

    All semantic selectors compare alternatives only within exact objective
    groups.  This mirrors the frozen integer-objective ECHO definition and
    prevents objective degradation from masquerading as semantic diversity.
    """
    if selector not in SELECTORS:
        raise ValueError(f"unknown selector {selector!r}; choose from {SELECTORS}")
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0.0 < quality_fraction <= 1.0:
        raise ValueError("quality_fraction must be in (0, 1]")

    X, F, source = _deduplicated_nondominated_pool(problem, X, F)
    if ref_dirs is None:
        ref_dirs = get_reference_directions(
            "energy", problem.n_obj, max(1, int(size)), seed=1)
    core_size = min(len(X), size, int(round(quality_fraction * size)))
    selected = _objective_core(
        problem, X, F, core_size, ref_dirs, random_seed)
    selected_set = set(selected)
    candidates = [index for index in range(len(X)) if index not in selected_set]
    events = np.asarray(problem.failure_event_embedding(X), dtype=bool)
    if selector in {"common_vertex_overlap", "xu2025_paper_vertex"} and vertex_embedding is None:
        transform = getattr(problem, "vertex_event_embedding", None)
        if callable(transform):
            vertex_embedding = transform(X)
    if vertex_embedding is not None:
        vertex_embedding = np.asarray(vertex_embedding, dtype=bool)
        if len(vertex_embedding) != len(X):
            raise ValueError("vertex_embedding must align with the deduplicated pool")
    if selector == "xu2025_official_edge" and directed_edge_counts is None:
        transform = getattr(problem, "directed_edge_usage_embedding", None)
        if callable(transform):
            directed_edge_counts, path_vertex_counts = transform(X)
    if directed_edge_counts is not None:
        directed_edge_counts = np.asarray(directed_edge_counts)
        path_vertex_counts = np.asarray(path_vertex_counts)
        if len(directed_edge_counts) != len(X) or len(path_vertex_counts) != len(X):
            raise ValueError("official-code embeddings must align with the pool")

    xu_scores = None
    if selector == "xu2025_paper_vertex":
        if vertex_embedding is None:
            raise ValueError("Xu paper selector requires vertex_embedding")
        xu_scores = _xu2025_rank_scores(vertex_embedding, include_self=True)
    elif selector == "xu2025_official_edge":
        if directed_edge_counts is None:
            raise ValueError("Xu official-code selector requires directed edges")
        xu_scores = _xu2025_rank_scores(
            directed_edge_counts > 0,
            counts=directed_edge_counts,
            denominators=path_vertex_counts,
            include_self=False,
        )

    rng = np.random.default_rng(random_seed)
    random_order = {
        index: rank for rank, index in enumerate(rng.permutation(candidates))
    }
    while candidates and len(selected) < size:
        if selector == "objective_only":
            # Continue ordinary reference-direction survival to K.
            selected = _objective_core(
                problem, X, F, min(size, len(X)), ref_dirs, random_seed)
            candidates = []
            break
        if selector == "random_quality_matched":
            chosen = min(candidates, key=lambda index: random_order[index])
        elif selector == "echo":
            chosen = max(
                candidates,
                key=lambda index: (*_echo_score(
                    events, F, selected, candidates, index), -index),
            )
        elif selector == "edge_jaccard":
            chosen = max(
                candidates,
                key=lambda index: (
                    _distance_score(
                        events, F, selected, candidates, index, _jaccard),
                    -index,
                ),
            )
        elif selector == "minimum_common_edge":
            chosen = max(
                candidates,
                key=lambda index: (
                    _distance_score(
                        events, F, selected, candidates, index,
                        lambda left, right: 1.0 - _common_fraction(left, right),
                    ),
                    -index,
                ),
            )
        elif selector == "kmedoids_edge":
            chosen = max(
                candidates,
                key=lambda index: (
                    _facility_location_score(
                        events, F, selected, candidates, index),
                    -index,
                ),
            )
        elif selector in {"xu2025_paper_vertex", "xu2025_official_edge"}:
            # Xu et al. use a fixed, rank-wide similarity value and retain the
            # lowest-scoring members.  The shared objective core is retained
            # here to keep the offline-composer comparison quality matched.
            chosen = min(candidates, key=lambda index: (xu_scores[index], index))
        else:
            if vertex_embedding is None:
                raise ValueError(
                    "common_vertex_overlap requires vertex_embedding or "
                    "problem.vertex_event_embedding")
            chosen = max(
                candidates,
                key=lambda index: (
                    _distance_score(
                        vertex_embedding, F, selected, candidates, index,
                        lambda left, right: 1.0 - _common_fraction(left, right),
                    ),
                    -index,
                ),
            )
        selected.append(chosen)
        candidates.remove(chosen)

    # Fixed-cardinality reporting: repeated plans consume output slots if the
    # unique nondominated pool is smaller than K, matching the MAPF protocol.
    unique_count = len(selected)
    if pad_to_size:
        while len(selected) < size:
            selected.append(selected[len(selected) % unique_count])
    selected = np.asarray(selected[:size], dtype=int)
    return PortfolioSelection(
        X=np.asarray(X[selected]),
        F=np.asarray(F[selected], dtype=float),
        source_indices=np.asarray(source[selected], dtype=int),
        quality_core_size=core_size,
        selector=selector,
    )
