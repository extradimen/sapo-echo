"""Strict portfolio-level ECHO definitions for the IEEE Access revision.

This module intentionally does not change the historical implementation in
``echo_portfolio.py``.  It provides the event-coverage function stated in the
submitted manuscript and keeps the post-review correction auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from echo_portfolio import _deduplicated_nondominated_pool, _objective_core
from pymoo.util.ref_dirs import get_reference_directions


@dataclass(frozen=True)
class StrictPortfolioSelection:
    X: np.ndarray
    F: np.ndarray
    source_indices: np.ndarray
    quality_core_size: int
    selector: str
    shortfall: int


def fixed_event_universe(pool_events: np.ndarray) -> np.ndarray:
    """Return pool-supported physical-edge events shared by all selectors."""
    events = np.asarray(pool_events, dtype=bool)
    if events.ndim != 2 or events.shape[0] == 0:
        raise ValueError("pool_events must be a nonempty candidate-by-event matrix")
    return np.any(events, axis=0)


def portfolio_coverage_counts(selected_events: np.ndarray,
                              universe: np.ndarray) -> tuple[int, int, int, int]:
    """Return survived/total singleton and unordered two-event counts."""
    events = np.asarray(selected_events, dtype=bool)
    universe = np.asarray(universe, dtype=bool)
    if events.ndim != 2 or events.shape[1] != len(universe):
        raise ValueError("selected events and universe are not aligned")
    event_indices = np.flatnonzero(universe)
    q = len(event_indices)
    if q == 0:
        return 0, 0, 0, 0
    used = events[:, event_indices]
    failed_single = np.all(used, axis=0)
    survived_single = q - int(np.count_nonzero(failed_single))
    total_pairs = q * (q - 1) // 2
    if total_pairs == 0:
        return survived_single, q, 0, 0

    # K is 30 in the paper, so one uint64 bit mask represents every selected
    # plan.  Fall back to direct Boolean evaluation for larger portfolios.
    k = len(events)
    if k <= 64:
        masks = np.zeros(q, dtype=np.uint64)
        for position in range(k):
            masks |= used[position].astype(np.uint64) << np.uint64(position)
        all_plans = (np.uint64(1) << np.uint64(k)) - np.uint64(1) if k < 64 else np.uint64(~0)
        # Many map edges induce the same selected-plan usage mask.  Collapse
        # them before counting pairs; this changes neither the event universe
        # nor the denominator and avoids an O(Q^2) loop over thousands of map
        # edges for every selector.
        unique_masks, counts = np.unique(masks, return_counts=True)
        failed_pairs = 0
        for left in range(len(unique_masks)):
            if unique_masks[left] == all_plans:
                failed_pairs += int(counts[left] * (counts[left] - 1) // 2)
            if left + 1 < len(unique_masks):
                matches = np.bitwise_or(
                    unique_masks[left], unique_masks[left + 1:]) == all_plans
                failed_pairs += int(counts[left] * counts[left + 1:][matches].sum())
    else:
        failed_pairs = 0
        for left in range(q - 1):
            failed_pairs += sum(
                bool(np.all(used[:, left] | used[:, right]))
                for right in range(left + 1, q))
    return survived_single, q, total_pairs - failed_pairs, total_pairs


def portfolio_coverage_metrics(selected_events: np.ndarray,
                               universe: np.ndarray) -> dict[str, float | int]:
    s1, n1, s2, n2 = portfolio_coverage_counts(selected_events, universe)
    return {
        "portfolio_psc1": float(s1 / n1) if n1 else 0.0,
        "portfolio_psc2": float(s2 / n2) if n2 else 0.0,
        "portfolio_psc1_survived_events": s1,
        "portfolio_psc1_total_events": n1,
        "portfolio_psc2_survived_events": s2,
        "portfolio_psc2_total_events": n2,
    }


def _failed_pair_count_from_masks(masks: np.ndarray, all_plans: np.uint64) -> int:
    unique_masks, counts = np.unique(np.asarray(masks, dtype=np.uint64),
                                     return_counts=True)
    failed = 0
    for left in range(len(unique_masks)):
        if unique_masks[left] == all_plans:
            failed += int(counts[left] * (counts[left] - 1) // 2)
        if left + 1 < len(unique_masks):
            matches = np.bitwise_or(
                unique_masks[left], unique_masks[left + 1:]) == all_plans
            failed += int(counts[left] * counts[left + 1:][matches].sum())
    return failed


def _psc2_marginal_gain(events: np.ndarray, universe: np.ndarray,
                        selected: list[int], index: int) -> int:
    """Count currently failed two-edge events rescued by one candidate."""
    if not selected:
        return 0
    event_indices = np.flatnonzero(universe)
    selected_usage = events[np.asarray(selected, dtype=int)][:, event_indices]
    masks = np.zeros(len(event_indices), dtype=np.uint64)
    for position in range(len(selected)):
        masks |= (selected_usage[position].astype(np.uint64)
                  << np.uint64(position))
    avoided = ~events[index, event_indices]
    if np.count_nonzero(avoided) < 2:
        return 0
    all_selected = ((np.uint64(1) << np.uint64(len(selected))) - np.uint64(1)
                    if len(selected) < 64 else np.uint64(~0))
    return _failed_pair_count_from_masks(masks[avoided], all_selected)


def select_strict_coverage_portfolio(problem, X, F, size, *,
                                     quality_fraction=.7, random_seed=0,
                                     ref_dirs=None,
                                     use_psc2_tiebreak=True):
    """Greedily maximize true portfolio-level closure survival.

    The objective-quality core is frozen first.  Every residual step chooses a
    maximum primary PSC-1 marginal gain; PSC-2 is used only among primary ties.
    No duplicate padding is performed.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0.0 < quality_fraction <= 1.0:
        raise ValueError("quality_fraction must be in (0, 1]")
    X, F, source = _deduplicated_nondominated_pool(problem, X, F)
    if ref_dirs is None:
        ref_dirs = get_reference_directions(
            "energy", problem.n_obj, max(1, int(size)), seed=1)
    core_size = min(len(X), size, int(round(quality_fraction * size)))
    selected = _objective_core(problem, X, F, core_size, ref_dirs, random_seed)
    selected_set = set(selected)
    candidates = [index for index in range(len(X)) if index not in selected_set]
    events = np.asarray(problem.failure_event_embedding(X), dtype=bool)
    universe = fixed_event_universe(events)
    while candidates and len(selected) < size:
        common = np.all(events[np.asarray(selected, dtype=int)], axis=0)
        primary = {
            index: int(np.count_nonzero(universe & common & ~events[index]))
            for index in candidates
        }
        best_primary = max(primary.values())
        ties = [index for index in candidates
                if primary[index] == best_primary]
        if use_psc2_tiebreak and len(ties) > 1:
            secondary = {
                index: _psc2_marginal_gain(
                    events, universe, selected, index)
                for index in ties
            }
            best_secondary = max(secondary.values())
            ties = [index for index in ties
                    if secondary[index] == best_secondary]
        chosen = min(ties)
        selected.append(chosen)
        candidates.remove(chosen)
    chosen = np.asarray(selected[:size], dtype=int)
    return StrictPortfolioSelection(
        X=np.asarray(X[chosen]),
        F=np.asarray(F[chosen], dtype=float),
        source_indices=np.asarray(source[chosen], dtype=int),
        quality_core_size=core_size,
        selector=("strict_coverage_psc1_psc2_tie" if use_psc2_tiebreak
                  else "strict_coverage_psc1_only"),
        shortfall=max(0, int(size) - len(chosen)),
    )

