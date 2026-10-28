"""Exact corrected-coverage oracle audit for the IEEE Access revision."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from analyze_mapf_dense_representation import independently_validate
from echo_portfolio import _deduplicated_nondominated_pool, _objective_core
from echo_revision import (
    fixed_event_universe,
    portfolio_coverage_metrics,
    select_strict_coverage_portfolio,
)
from pymoo.problems.multi.movingai_mapf import (
    PrioritizedMAPFProblem,
    load_movingai_map,
    load_movingai_scenarios,
)
from pymoo.util.ref_dirs import get_reference_directions
from research_archive import atomic_json


ROOT = Path(__file__).resolve().parent
OUTPUT = (ROOT / "research_archive" / "analyses" /
          "2026-10-28_ieee-revision_corrected-oracle_v1")
PROTOCOL = (ROOT / "research_archive" / "plans" /
            "2026-10-28_ieee-access-revision-experiment-protocol_v1.md")
MAP = ROOT / "data/movingai_mapf/maps/random-32-32-20.map"
SCENARIO = (ROOT / "data/movingai_mapf/scenarios/scen-even/"
            "random-32-32-20-even-1.scen")
PAIRS = ((1, 4), (1, 19), (3, 4), (3, 5), (3, 8))
PORTFOLIO_SIZES = (3, 6, 10)
MAX_GLOBAL_COMBINATIONS = 500_000
MAX_RESIDUAL_COMBINATIONS = 2_000_000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"no rows for {path}")
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def boolean_bits(values):
    packed = np.packbits(np.asarray(values, dtype=np.uint8), bitorder="little")
    return int.from_bytes(packed.tobytes(), "little")


def failure_masks(events, universe):
    usage = np.asarray(events, dtype=bool)[:, np.asarray(universe, dtype=bool)]
    q = usage.shape[1]
    left, right = np.triu_indices(q, 1)
    single = [boolean_bits(row) for row in usage]
    pair = [boolean_bits(row[left] | row[right]) for row in usage]
    return single, pair, q, len(left)


def objective_tie_score(objectives, indices):
    low = np.min(objectives, axis=0)
    span = np.maximum(1.0, np.max(objectives, axis=0) - low)
    return float(np.sum((objectives[list(indices)] - low) / span))


def subset_key(single_masks, pair_masks, objectives, indices):
    single_failed = single_masks[indices[0]]
    pair_failed = pair_masks[indices[0]]
    for index in indices[1:]:
        single_failed &= single_masks[index]
        pair_failed &= pair_masks[index]
    return (
        -single_failed.bit_count(),
        -pair_failed.bit_count(),
        -objective_tie_score(objectives, indices),
        tuple(-index for index in indices),
    )


def exact_best(single_masks, pair_masks, objectives, combinations):
    best_key = best_indices = None
    count = 0
    started = time.perf_counter()
    for indices in combinations:
        indices = tuple(indices)
        key = subset_key(single_masks, pair_masks, objectives, indices)
        count += 1
        if best_key is None or key > best_key:
            best_key, best_indices = key, indices
    return best_indices, count, time.perf_counter() - started


def metrics(events, universe, indices):
    return portfolio_coverage_metrics(
        events[np.asarray(indices, dtype=int)], universe)


def main():
    if (OUTPUT / "SHA256SUMS").exists():
        print(f"ALREADY COMPLETE {OUTPUT}")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    grid = load_movingai_map(MAP)
    tasks = load_movingai_scenarios(SCENARIO)
    rows = []
    for pair in PAIRS:
        problem = PrioritizedMAPFProblem(
            grid, [tasks[index] for index in pair], preference_levels=21)
        raw_x, raw_f = [], []
        for priority in ((0, 1), (1, 0)):
            for preferences in itertools.product(range(21), repeat=2):
                genotype = np.asarray(priority + preferences, dtype=int)
                decoded = problem.decode(genotype)
                if decoded["feasible"]:
                    independently_validate(
                        grid, problem.tasks, problem.risk_grid, decoded)
                    raw_x.append(genotype)
                    raw_f.append(decoded["objectives"])
        X, F, _ = _deduplicated_nondominated_pool(
            problem, np.asarray(raw_x), np.asarray(raw_f))
        events = np.asarray(problem.failure_event_embedding(X), dtype=bool)
        universe = fixed_event_universe(events)
        single_masks, pair_masks, q1, q2 = failure_masks(events, universe)
        lookup = {tuple(row.tolist()): index for index, row in enumerate(X)}
        for size in PORTFOLIO_SIZES:
            core_size = min(len(X), size, int(round(.70 * size)))
            residual = size - core_size
            residual_count = (math.comb(max(0, len(X) - core_size), residual)
                              if len(X) >= size else 0)
            global_count = math.comb(len(X), size) if len(X) >= size else 0
            base = {
                "pair": f"{pair[0]}-{pair[1]}",
                "feasible_genotypes": len(raw_x),
                "nondominated_unique_plans": len(X),
                "event_universe_single": q1,
                "event_universe_pairs": q2,
                "K": size,
                "core_size": core_size,
                "residual_slots": residual,
                "residual_combination_count": residual_count,
                "global_combination_count": global_count,
            }
            if len(X) < size:
                rows.append({
                    **base, "status": "insufficient_unique_pool",
                    "greedy_psc1": "", "greedy_psc2": "",
                    "fixed_core_oracle_psc1": "", "fixed_core_oracle_psc2": "",
                    "global_oracle_psc1": "", "global_oracle_psc2": "",
                    "greedy_fixed_core_ratio": "", "greedy_global_ratio": "",
                    "fixed_core_oracle_seconds": "", "global_oracle_seconds": "",
                })
                continue
            ref_dirs = get_reference_directions("energy", 2, size, seed=1)
            core = _objective_core(
                problem, X, F, core_size, ref_dirs, random_seed=20261028)
            candidates = [index for index in range(len(X))
                          if index not in set(core)]
            greedy = select_strict_coverage_portfolio(
                problem, X, F, size, quality_fraction=.70,
                random_seed=20261028, ref_dirs=ref_dirs,
                use_psc2_tiebreak=True)
            greedy_indices = tuple(
                lookup[tuple(row.tolist())] for row in greedy.X)
            greedy_metrics = metrics(events, universe, greedy_indices)

            fixed_indices = None
            fixed_seconds = float("nan")
            if residual_count <= MAX_RESIDUAL_COMBINATIONS:
                completions = (
                    tuple(core) + tuple(completion)
                    for completion in itertools.combinations(candidates, residual))
                fixed_indices, _, fixed_seconds = exact_best(
                    single_masks, pair_masks, F, completions)
            global_indices = None
            global_seconds = float("nan")
            if global_count <= MAX_GLOBAL_COMBINATIONS:
                global_indices, _, global_seconds = exact_best(
                    single_masks, pair_masks, F,
                    itertools.combinations(range(len(X)), size))
            fixed_metrics = (metrics(events, universe, fixed_indices)
                             if fixed_indices is not None else None)
            global_metrics = (metrics(events, universe, global_indices)
                              if global_indices is not None else None)
            rows.append({
                **base,
                "status": ("complete_both_oracles" if global_metrics is not None
                           else "complete_fixed_core_global_capped"),
                "greedy_psc1": greedy_metrics["portfolio_psc1"],
                "greedy_psc2": greedy_metrics["portfolio_psc2"],
                "fixed_core_oracle_psc1": (
                    fixed_metrics["portfolio_psc1"] if fixed_metrics else ""),
                "fixed_core_oracle_psc2": (
                    fixed_metrics["portfolio_psc2"] if fixed_metrics else ""),
                "global_oracle_psc1": (
                    global_metrics["portfolio_psc1"] if global_metrics else ""),
                "global_oracle_psc2": (
                    global_metrics["portfolio_psc2"] if global_metrics else ""),
                "greedy_fixed_core_ratio": (
                    greedy_metrics["portfolio_psc1"] /
                    fixed_metrics["portfolio_psc1"]
                    if fixed_metrics and fixed_metrics["portfolio_psc1"] else 1.0),
                "greedy_global_ratio": (
                    greedy_metrics["portfolio_psc1"] /
                    global_metrics["portfolio_psc1"]
                    if global_metrics and global_metrics["portfolio_psc1"] else ""),
                "fixed_core_oracle_seconds": fixed_seconds,
                "global_oracle_seconds": global_seconds,
            })
            print(f"COMPLETE pair={pair} K={size}", flush=True)
    write_csv(OUTPUT / "corrected_oracle_results.csv", rows)
    valid_fixed = [row for row in rows
                   if row["fixed_core_oracle_psc1"] != ""]
    valid_global = [row for row in rows if row["global_oracle_psc1"] != ""]
    atomic_json(OUTPUT / "summary.json", {
        "schema": "ieee-revision-corrected-oracle-summary-v1",
        "fixed_core_cells": len(valid_fixed),
        "global_oracle_cells": len(valid_global),
        "minimum_greedy_fixed_core_ratio": min(
            (float(row["greedy_fixed_core_ratio"]) for row in valid_fixed),
            default=None),
        "minimum_greedy_global_ratio": min(
            (float(row["greedy_global_ratio"]) for row in valid_global),
            default=None),
        "claim_boundary": (
            "Fixed-core residual and unconstrained fixed-K oracles solve "
            "different problems; only the former is the theorem's comparator."),
    })
    atomic_json(OUTPUT / "provenance.json", {
        "schema": "ieee-revision-corrected-oracle-v1",
        "evidence_status": "post-review revision analysis",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "map": str(MAP.relative_to(ROOT)),
        "map_sha256": sha256(MAP),
        "scenario": str(SCENARIO.relative_to(ROOT)),
        "scenario_sha256": sha256(SCENARIO),
        "pairs": PAIRS,
        "portfolio_sizes": PORTFOLIO_SIZES,
        "maximum_global_combinations": MAX_GLOBAL_COMBINATIONS,
        "maximum_residual_combinations": MAX_RESIDUAL_COMBINATIONS,
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
    })
    checksum = OUTPUT / "SHA256SUMS"
    files = sorted(path for path in OUTPUT.iterdir()
                   if path.is_file() and path != checksum)
    checksum.write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in files))
    print(f"COMPLETE {OUTPUT}")


if __name__ == "__main__":
    main()
