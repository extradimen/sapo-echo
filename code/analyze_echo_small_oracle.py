"""Exact residual portfolio audit for the frozen ECHO small-instance protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from analyze_mapf_dense_representation import independently_validate
from echo_portfolio import (
    _deduplicated_nondominated_pool,
    _objective_core,
    select_portfolio,
)
from pymoo.problems.multi.movingai_mapf import (
    PrioritizedMAPFProblem,
    load_movingai_map,
    load_movingai_scenarios,
)
from pymoo.util.ref_dirs import get_reference_directions


PAIRS = ((1, 4), (1, 19), (3, 4), (3, 5), (3, 8))
PORTFOLIO_SIZES = (3, 6, 10)
MAX_COMBINATIONS = 2_000_000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def psc1(events, objectives, indices):
    survived = exposed = 0
    groups = {}
    for index in indices:
        groups.setdefault(tuple(objectives[index]), []).append(index)
    for group in groups.values():
        used = events[group].astype(bool)
        union = np.any(used, axis=0)
        common = np.all(used, axis=0)
        survived += int(np.count_nonzero(union & ~common))
        exposed += int(np.count_nonzero(union))
    return survived / exposed if exposed else 0.0


def psc2(events, objectives, indices):
    survived = exposed = 0
    groups = {}
    for index in indices:
        groups.setdefault(tuple(objectives[index]), []).append(index)
    for group in groups.values():
        used = events[group].astype(bool)
        union_indices = np.flatnonzero(np.any(used, axis=0))
        if len(union_indices) < 2:
            continue
        exposed += len(union_indices) * (len(union_indices) - 1) // 2
        for left, right in itertools.combinations(union_indices, 2):
            if np.any(~(used[:, left] | used[:, right])):
                survived += 1
    return survived / exposed if exposed else 0.0


def objective_tie_score(objectives, indices):
    low = np.min(objectives, axis=0)
    span = np.maximum(1.0, np.max(objectives, axis=0) - low)
    return float(np.sum((objectives[list(indices)] - low) / span))


def exact_completion(events, objectives, core, candidates, residual):
    best = None
    started = time.perf_counter()
    for completion in itertools.combinations(candidates, residual):
        indices = tuple(core) + completion
        primary = psc1(events, objectives, indices)
        key = (
            primary,
            psc2(events, objectives, indices),
            -objective_tie_score(objectives, indices),
            tuple(-index for index in indices),
        )
        if best is None or key > best[0]:
            best = (key, indices)
    return best[1], time.perf_counter() - started


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--archive-root", default="research_archive")
    parser.add_argument("--analysis-id", required=True)
    args = parser.parse_args()
    target = Path(args.archive_root) / "analyses" / args.analysis_id
    target.mkdir(parents=True, exist_ok=False)

    grid = load_movingai_map(args.map)
    tasks = load_movingai_scenarios(args.scenario)
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

        for size in PORTFOLIO_SIZES:
            core_size = min(len(X), size, int(round(.70 * size)))
            residual = size - core_size
            combination_count = (
                math.comb(max(0, len(X) - core_size), residual)
                if len(X) >= size else 0)
            base = {
                "pair": f"{pair[0]}-{pair[1]}",
                "canonical_genotypes": 882,
                "feasible_genotypes": len(raw_x),
                "nondominated_unique_plans": len(X),
                "K": size,
                "core_size": core_size,
                "residual_slots": residual,
                "combination_count": combination_count,
            }
            if len(X) < size or combination_count > MAX_COMBINATIONS:
                rows.append({
                    **base, "status": "omitted_by_frozen_cap",
                    "greedy_psc1": "", "oracle_psc1": "", "ratio": "",
                    "absolute_gap": "", "greedy_psc2": "", "oracle_psc2": "",
                    "exact_match": "", "oracle_seconds": "",
                })
                continue
            ref_dirs = get_reference_directions("energy", 2, size, seed=1)
            core = _objective_core(
                problem, X, F, core_size, ref_dirs, random_seed=20260903)
            candidates = [index for index in range(len(X)) if index not in set(core)]
            greedy = select_portfolio(
                problem, X, F, size, selector="echo", quality_fraction=.70,
                random_seed=20260903, ref_dirs=ref_dirs, pad_to_size=False)
            lookup = {tuple(row.tolist()): index for index, row in enumerate(X)}
            greedy_indices = tuple(lookup[tuple(row.tolist())] for row in greedy.X)
            oracle_indices, elapsed = exact_completion(
                events, F, core, candidates, residual)
            greedy_primary = psc1(events, F, greedy_indices)
            oracle_primary = psc1(events, F, oracle_indices)
            rows.append({
                **base, "status": "complete",
                "greedy_psc1": greedy_primary,
                "oracle_psc1": oracle_primary,
                "ratio": greedy_primary / oracle_primary if oracle_primary else 1.0,
                "absolute_gap": oracle_primary - greedy_primary,
                "greedy_psc2": psc2(events, F, greedy_indices),
                "oracle_psc2": psc2(events, F, oracle_indices),
                "exact_match": set(greedy_indices) == set(oracle_indices),
                "oracle_seconds": elapsed,
            })

    results = target / "oracle_results.csv"
    write_csv(results, rows)
    valid = [row for row in rows if row["status"] == "complete"]
    summary = {
        "schema": "echo-small-decoder-universe-oracle-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_status": "confirmatory",
        "valid_cells": len(valid),
        "omitted_cells": len(rows) - len(valid),
        "minimum_ratio": min((float(row["ratio"]) for row in valid), default=None),
        "median_ratio": float(np.median([float(row["ratio"]) for row in valid])) if valid else None,
        "all_above_one_minus_inverse_e": all(
            float(row["ratio"]) + 1e-12 >= 1.0 - 1.0 / math.e for row in valid),
        "median_at_least_095": bool(valid) and float(np.median(
            [float(row["ratio"]) for row in valid])) >= .95,
        "claim_boundary": (
            "Exact residual selection over the complete finite prioritized-decoder "
            "universe, conditional on a fixed objective core; not all mathematical MAPF paths."
        ),
    }
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance = {
        "protocol": str(Path(
            "research_archive/plans/2026-09-03_echo_small_oracle_protocol_v1.md"
        ).resolve()),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(__file__),
        "map": str(Path(args.map).resolve()), "map_sha256": sha256(args.map),
        "scenario": str(Path(args.scenario).resolve()),
        "scenario_sha256": sha256(args.scenario),
        "pairs": PAIRS, "portfolio_sizes": PORTFOLIO_SIZES,
        "maximum_combinations": MAX_COMBINATIONS,
    }
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    files = sorted(path for path in target.iterdir() if path.is_file())
    (target / "SHA256SUMS").write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in files))
    print(json.dumps({"target": str(target), **summary}, indent=2))


if __name__ == "__main__":
    main()
