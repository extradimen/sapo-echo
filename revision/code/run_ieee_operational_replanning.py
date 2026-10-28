"""Frozen small-instance stored-portfolio versus official MO-CBS edge replanning."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np

from echo_portfolio import _deduplicated_nondominated_pool, _objective_core
from echo_revision import select_strict_coverage_portfolio
from external_adapters.momapf.official_mocbs import (
    OFFICIAL_COMMIT, build_official_mocbs, run_official_mocbs,
    write_mocbs_instance,
)
from pymoo.problems.multi.movingai_mapf import (
    PrioritizedMAPFProblem, load_movingai_map, load_movingai_scenarios,
)
from pymoo.util.ref_dirs import get_reference_directions
from research_archive import atomic_json


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config/ieee_revision_operational_replan_v1.json"
PROTOCOL = ROOT / "research_archive/plans/2026-10-28_ieee-operational-replanning-protocol_v1.md"
OUTPUT = ROOT / "research_archive/experiments/2026-10-28_ieee-operational-replanning_v1"
OFFICIAL_SOURCE = ROOT / "third_party/public_cppmomapf/cpp"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def eligible_pool(problem, levels):
    started = time.perf_counter()
    raw_x, raw_f = [], []
    for priority in ((0, 1), (1, 0)):
        for preferences in itertools.product(range(levels), repeat=2):
            genotype = np.asarray(priority + preferences, dtype=int)
            decoded = problem.decode(genotype)
            if decoded["feasible"]:
                raw_x.append(genotype)
                raw_f.append(decoded["objectives"])
    X, F, _ = _deduplicated_nondominated_pool(
        problem, np.asarray(raw_x), np.asarray(raw_f))
    return X, F, len(raw_x), time.perf_counter() - started


def selected_portfolios(problem, X, F, config):
    size = int(config["fixed_output_size"])
    ref_dirs = get_reference_directions("energy", 2, size, seed=1)
    started = time.perf_counter()
    objective_indices = _objective_core(
        problem, X, F, size, ref_dirs, random_seed=20261028)
    objective_seconds = time.perf_counter() - started
    started = time.perf_counter()
    echo = select_strict_coverage_portfolio(
        problem, X, F, size,
        quality_fraction=float(config["quality_core_fraction"]),
        random_seed=20261028, ref_dirs=ref_dirs)
    echo_seconds = time.perf_counter() - started
    if echo.shortfall or len(objective_indices) != size:
        raise AssertionError("frozen K=3 output cannot be filled")
    lookup = {tuple(row.tolist()): i for i, row in enumerate(X)}
    echo_indices = [lookup[tuple(row.tolist())] for row in echo.X]
    return objective_indices, echo_indices, objective_seconds, echo_seconds


def frozen_events(pool_events, objective_indices, pair, config):
    universe = np.flatnonzero(np.any(pool_events, axis=0))
    exposure = np.sum(pool_events[np.asarray(objective_indices)], axis=0)
    n_random = int(config["events_per_pair_uniform"])
    n_critical = int(config["events_per_pair_high_exposure"])
    if len(universe) < max(n_random, n_critical):
        raise ValueError("insufficient pool-supported physical edges")
    seed = int(config["event_seed"]) + pair[0] * 100 + pair[1]
    rng = np.random.default_rng(seed)
    random_edges = sorted(rng.choice(universe, n_random, replace=False).tolist())
    critical_edges = sorted(universe.tolist(),
                            key=lambda i: (-int(exposure[i]), i))[:n_critical]
    return [("uniform", i) for i in random_edges] + [
        ("high_exposure", i) for i in critical_edges]


def main():
    if (OUTPUT / "SHA256SUMS").exists():
        print(f"ALREADY COMPLETE {OUTPUT}", flush=True)
        return
    config = json.loads(CONFIG.read_text())
    pairs = [tuple(pair) for pair in config["task_pairs_zero_based"]]
    map_path = ROOT / config["map"]
    scenario_path = ROOT / config["scenario"]
    grid = load_movingai_map(map_path)
    tasks = load_movingai_scenarios(scenario_path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    unit_dir = OUTPUT / "units"
    unit_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ieee_operational_replan_") as temp:
        binary = Path(temp) / "mocbs_cli"
        build = build_official_mocbs(OFFICIAL_SOURCE, binary)
        if build["official_commit"] != OFFICIAL_COMMIT:
            raise AssertionError("official MO-CBS commit changed")
        all_rows = []
        pair_records = []
        for pair in pairs:
            name = f"tasks{pair[0]}-{pair[1]}"
            problem = PrioritizedMAPFProblem(
                grid, [tasks[i] for i in pair],
                preference_levels=int(config["preference_levels"]))
            X, F, feasible_count, pool_seconds = eligible_pool(
                problem, int(config["preference_levels"]))
            objective_indices, echo_indices, objective_seconds, echo_seconds = (
                selected_portfolios(problem, X, F, config))
            pool_events = np.asarray(problem.failure_event_embedding(X), dtype=bool)
            event_specs = frozen_events(pool_events, objective_indices, pair, config)
            edge_by_index = {index: edge for edge, index in
                             problem.physical_edge_index.items()}
            pair_records.append({
                "pair": name, "feasible_genotypes": feasible_count,
                "eligible_unique_plans": len(X),
                "pool_seconds": pool_seconds,
                "objective_composer_seconds": objective_seconds,
                "echo_composer_seconds": echo_seconds,
                "objective_indices": objective_indices,
                "echo_indices": echo_indices,
                "eligible_objectives": np.asarray(F).tolist(),
            })
            for event_class, edge_index in event_specs:
                unit_file = unit_dir / f"{name}_{event_class}_edge{edge_index}.json"
                if unit_file.exists():
                    row = json.loads(unit_file.read_text())
                    if (row["pair"] != name or row["event_class"] != event_class
                            or row["edge_index"] != edge_index):
                        raise AssertionError("resumed unit identity mismatch")
                    all_rows.append(row)
                    continue
                edge = edge_by_index[edge_index]
                raw_survivors = np.flatnonzero(~pool_events[:, edge_index])
                objective_survivors = [i for i in objective_indices
                                       if not pool_events[i, edge_index]]
                echo_survivors = [i for i in echo_indices
                                  if not pool_events[i, edge_index]]
                instance = Path(temp) / f"{name}_{event_class}_{edge_index}.txt"
                started = time.perf_counter()
                instance_grid, instance_tasks = write_mocbs_instance(
                    map_path, scenario_path, 2, instance, task_indices=pair,
                    blocked_edges=[edge])
                summary, solutions, validated = run_official_mocbs(
                    binary, instance,
                    float(config["mocbs_time_limit_seconds_per_event"]),
                    instance_grid, instance_tasks, require_complete=False,
                    blocked_edges=[edge])
                response_seconds = time.perf_counter() - started
                row = {
                    "pair": name,
                    "event_class": event_class,
                    "edge_index": int(edge_index),
                    "closed_edge": [list(edge[0]), list(edge[1])],
                    "eligible_pool_survivors": int(len(raw_survivors)),
                    "objective_survivors": int(len(objective_survivors)),
                    "echo_survivors": int(len(echo_survivors)),
                    "objective_immediate_success": bool(objective_survivors),
                    "echo_immediate_success": bool(echo_survivors),
                    "mocbs_replan_success": bool(validated["solution_count"]),
                    "mocbs_complete_pareto": bool(summary["find_all_pareto"]),
                    "mocbs_solution_count": int(validated["solution_count"]),
                    "mocbs_replan_objectives": validated["costs"].tolist(),
                    "mocbs_response_seconds": response_seconds,
                    "mocbs_search_seconds": float(summary["runtime_search"]),
                    "mocbs_expanded": int(summary["n_expanded"]),
                    "mocbs_low_level_expansions": int(summary["low_level_expansions"]),
                }
                atomic_json(unit_file, row)
                all_rows.append(row)
                print(f"COMPLETE {name} {event_class} edge {edge_index}", flush=True)
        expected = len(pairs) * (
            int(config["events_per_pair_uniform"]) +
            int(config["events_per_pair_high_exposure"]))
        if len(all_rows) != expected:
            raise AssertionError(f"expected {expected} event rows, got {len(all_rows)}")
        all_rows.sort(key=lambda row: (row["pair"], row["event_class"], row["edge_index"]))
        with (OUTPUT / "event_results.csv").open("w", newline="") as stream:
            fields = [key for key in all_rows[0]
                      if key not in {"closed_edge", "mocbs_replan_objectives"}]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in all_rows)
        atomic_json(OUTPUT / "pair_records.json", pair_records)
        atomic_json(OUTPUT / "provenance.json", {
            "schema": "ieee-operational-replanning-v1",
            "config": str(CONFIG.relative_to(ROOT)),
            "config_sha256": sha256(CONFIG),
            "protocol": str(PROTOCOL.relative_to(ROOT)),
            "protocol_sha256": sha256(PROTOCOL),
            "script_sha256": sha256(__file__),
            "adapter_python_sha256": sha256(ROOT / "external_adapters/momapf/official_mocbs.py"),
            "adapter_cpp_sha256": sha256(ROOT / "external_adapters/momapf/mocbs_cli.cpp"),
            "map_sha256": sha256(map_path),
            "scenario_sha256": sha256(scenario_path),
            "official_commit": OFFICIAL_COMMIT,
            "build": build,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "event_rows": expected,
        })
    checksum = OUTPUT / "SHA256SUMS"
    checksum.write_text("".join(
        f"{sha256(path)}  {path.relative_to(OUTPUT)}\n"
        for path in sorted(OUTPUT.rglob("*"))
        if path.is_file() and path != checksum))
    print(f"SEALED {OUTPUT} {len(all_rows)} events", flush=True)


if __name__ == "__main__":
    main()
