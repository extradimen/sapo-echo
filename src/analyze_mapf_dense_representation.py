"""Exhaustively audit a denser prioritized-MAPF representation.

This is a decoder gate, not an optimizer experiment.  For two robots the
canonical search space contains two priority orders times L squared preference
combinations, where L is fixed before execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pymoo.problems.multi.movingai_mapf import (
    PrioritizedMAPFProblem,
    joint_paths_are_collision_free,
    load_movingai_map,
    load_movingai_scenarios,
)


PAIRS = ((1, 4), (1, 19), (3, 4), (3, 5), (3, 8))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def official_archive(root, pair):
    left, right = pair
    return Path(root) / "experiments" / (
        f"2026-08-31_official-mocbs_random32-tasks{left}-{right}_exact_v1"
    )


def independently_validate(grid, tasks, risk_grid, decoded):
    """Recompute path legality and objectives without trusting decoder flags."""
    paths = decoded["paths"]
    if paths is None or len(paths) != len(tasks):
        raise AssertionError("decoder did not return one path per task")
    for task, path in zip(tasks, paths):
        if tuple(path[0]) != tuple(task.start) or tuple(path[-1]) != tuple(task.goal):
            raise AssertionError("decoded endpoint mismatch")
        for x, y in path:
            if not (0 <= y < grid.shape[0] and 0 <= x < grid.shape[1] and grid[y, x]):
                raise AssertionError("decoded path enters an invalid cell")
        for source, target in zip(path, path[1:]):
            if abs(source[0] - target[0]) + abs(source[1] - target[1]) > 1:
                raise AssertionError("decoded path contains an invalid move")
    if not joint_paths_are_collision_free(paths):
        raise AssertionError("decoded paths collide")
    recomputed = (
        sum(len(path) - 1 for path in paths),
        sum(risk_grid[y, x] for path in paths for x, y in path[1:]),
    )
    reported = tuple(map(int, decoded["objectives"]))
    if tuple(map(int, recomputed)) != reported:
        raise AssertionError("independently recomputed objectives differ")
    return reported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--archive-root", default="research_archive")
    parser.add_argument("--analysis-id", required=True)
    parser.add_argument("--preference-levels", type=int, default=21)
    args = parser.parse_args()

    if args.preference_levels != 21:
        raise ValueError("This pre-registered audit fixes preference_levels=21")
    target = Path(args.archive_root) / "analyses" / args.analysis_id
    if target.exists():
        raise FileExistsError(target)
    target.mkdir(parents=True)

    grid = load_movingai_map(args.map)
    all_tasks = load_movingai_scenarios(args.scenario)
    rows = []
    witnesses = {}
    for pair in PAIRS:
        reference_path = official_archive(args.archive_root, pair) / "raw" / "validated_solutions.json"
        reference = json.loads(reference_path.read_text())
        official = {tuple(map(int, cost)) for cost in reference["independently_recomputed_costs"]}
        problem = PrioritizedMAPFProblem(
            grid,
            [all_tasks[index] for index in pair],
            preference_levels=args.preference_levels,
        )
        reached = {}
        feasible = 0
        for priority in ((0, 1), (1, 0)):
            for preferences in itertools.product(range(args.preference_levels), repeat=2):
                genotype = np.asarray(priority + preferences, dtype=int)
                decoded = problem.decode(genotype)
                if not decoded["feasible"]:
                    continue
                feasible += 1
                cost = independently_validate(
                    grid, problem.tasks, problem.risk_grid, decoded
                )
                reached.setdefault(cost, genotype.tolist())
        covered = official & set(reached)
        missing = official - set(reached)
        label = f"{pair[0]}-{pair[1]}"
        witnesses[label] = {
            ",".join(map(str, cost)): reached[cost]
            for cost in sorted(covered)
        }
        rows.append({
            "pair": label,
            "canonical_genotypes": 2 * args.preference_levels ** 2,
            "feasible_genotypes": feasible,
            "unique_cost_vectors": len(reached),
            "official_cost_vectors": len(official),
            "covered_official_cost_vectors": len(covered),
            "coverage": len(covered) / len(official),
            "missing": json.dumps(sorted(missing)),
        })

    with (target / "representation_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(target / "coverage_witnesses.json", witnesses)
    passes = all(row["coverage"] == 1.0 for row in rows)
    write_json(target / "registered_decision.json", {
        "schema": "mapf-dense-preference-representation-gate-v1",
        "preference_levels": args.preference_levels,
        "pairs": [row["pair"] for row in rows],
        "all_official_cost_vectors_representable": passes,
        "gate_passed": passes,
        "optimizer_runs": 0,
    })
    write_json(target / "provenance.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(__file__),
        "map": str(Path(args.map).resolve()),
        "map_sha256": sha256(args.map),
        "scenario": str(Path(args.scenario).resolve()),
        "scenario_sha256": sha256(args.scenario),
        "official_archives": [str(official_archive(args.archive_root, pair)) for pair in PAIRS],
        "selection_or_tuning": "none; exhaustive decoder-only gate",
        "independent_validation": "endpoints, traversability, moves, collisions, and both objectives recomputed",
    })
    files = sorted(path for path in target.iterdir() if path.is_file())
    (target / "SHA256SUMS").write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in files
    ))
    print(json.dumps({"target": str(target), "gate_passed": passes, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
