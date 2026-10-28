"""Run frozen E7 equal-call ECHO versus reactive blocked-edge replanning."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_echo_final_multifamily import (
    EXPERIMENTS,
    RUN_GROUPS,
    load_generator_pool,
)
from movingai_mapf_experiment import make_problem
from pymoo.operators.sampling.categorical import ScrambledSobolCategoricalSampling
from research_archive import atomic_json


ROOT = Path(__file__).resolve().parent
SOURCE_ANALYSIS = (
    ROOT / "research_archive" / "analyses" /
    "2026-09-04_echo-final-multifamily_selector-confirmation_v1"
)
OUTPUT = (
    ROOT / "research_archive" / "experiments" /
    "2026-09-05_echo-equal-budget-replanning_E7_v2"
)
SCENARIOS = ("wh10b", "wh20a", "random32", "maze32b")
SEEDS = tuple(range(1401, 1411))
EVENTS_PER_UNIT = 10
REACTIVE_CALLS_PER_EVENT = 15
GENERATOR = "NSGA3-ObjectiveArchive"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_group(scenario, seed):
    matches = [group for group in RUN_GROUPS
               if group[0] == scenario and group[2] <= seed <= group[3]]
    if len(matches) != 1:
        raise RuntimeError(f"source mapping mismatch: {scenario}/{seed}")
    return matches[0]


def load_portfolio(scenario, seed, selector):
    path = SOURCE_ANALYSIS / "portfolios" / (
        f"{scenario}_{GENERATOR}_seed{seed:03d}_{selector}_r00.npz")
    with np.load(path, allow_pickle=False) as saved:
        return np.asarray(saved["X"], dtype=int), np.asarray(saved["F"], dtype=float), path


def best_score(F, ideal, scale):
    if len(F) == 0:
        return float("nan")
    return float(np.min(np.sum((np.asarray(F) - ideal) / scale, axis=1)))


def main():
    if (OUTPUT / "SHA256SUMS").exists():
        print(f"ALREADY COMPLETE {OUTPUT}", flush=True)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    unit_directory = OUTPUT / "units"
    unit_directory.mkdir(exist_ok=True)
    rows = []
    units = []
    sampling = ScrambledSobolCategoricalSampling()
    for scenario in SCENARIOS:
        for seed in SEEDS:
            unit_file = unit_directory / f"{scenario}_seed{seed:04d}.json"
            if unit_file.exists():
                saved_unit = json.loads(unit_file.read_text())
                rows.extend(saved_unit["rows"])
                units.append(saved_unit["provenance"])
                print(f"SKIP COMPLETE {scenario} seed {seed}", flush=True)
                continue
            _, directory, _, _ = source_group(scenario, seed)
            experiment = EXPERIMENTS / directory
            config = json.loads((experiment / "config.json").read_text())
            manifest = json.loads((experiment / "manifest.json").read_text())
            run_id = f"{GENERATOR}_seed{seed:03d}"
            completed = {record["run_id"] for record in manifest.get("runs", [])
                         if record.get("status") == "complete"}
            if run_id not in completed:
                raise RuntimeError(f"incomplete source run: {directory}/{run_id}")
            run_path = experiment / "runs" / run_id
            source_metrics = json.loads((run_path / "metrics.json").read_text())
            problem = make_problem(
                config["map"], config["scenario"], int(config["agents"]),
                task_indices=config.get("task_indices"),
                preference_levels=int(config.get("preference_levels", 5)),
                independent_neighbor_orders=bool(
                    config.get("independent_neighbor_orders", False)))
            pool_x, pool_f, snapshot = load_generator_pool(
                problem, run_path, GENERATOR, 2 * int(config["population"]))
            ideal = np.min(pool_f, axis=0)
            scale = np.where(np.ptp(pool_f, axis=0) > 0, np.ptp(pool_f, axis=0), 1.0)
            objective_x, objective_f, objective_file = load_portfolio(
                scenario, seed, "objective_only")
            echo_x, echo_f, echo_file = load_portfolio(scenario, seed, "echo")
            objective_events = problem.failure_event_embedding(objective_x).astype(bool)
            echo_events = problem.failure_event_embedding(echo_x).astype(bool)
            exposure = np.sum(objective_events, axis=0)
            candidates = np.flatnonzero(exposure > 0)
            order = sorted(candidates.tolist(), key=lambda index: (-exposure[index], index))
            selected_edges = order[:EVENTS_PER_UNIT]
            if len(selected_edges) != EVENTS_PER_UNIT:
                raise RuntimeError(f"fewer than ten exposed edges: {scenario}/{seed}")
            echo_storage_waypoints = sum(
                len(path) for row in echo_x for path in problem.decode(row)["paths"])

            unit_rows = []
            for event_rank, edge_index in enumerate(selected_edges, start=1):
                edge = next(key for key, value in problem.physical_edge_index.items()
                            if value == edge_index)
                lookup_started = time.perf_counter()
                objective_survive = ~objective_events[:, edge_index]
                echo_survive = ~echo_events[:, edge_index]
                echo_lookup_seconds = time.perf_counter() - lookup_started

                blocked = make_problem(
                    config["map"], config["scenario"], int(config["agents"]),
                    horizon=problem.horizon,
                    task_indices=config.get("task_indices"),
                    preference_levels=int(config.get("preference_levels", 5)),
                    independent_neighbor_orders=bool(
                        config.get("independent_neighbor_orders", False)),
                    blocked_edges=[edge])
                reactive_seed = seed * 100000 + edge_index
                X = sampling.do(
                    blocked, REACTIVE_CALLS_PER_EVENT,
                    random_state=np.random.default_rng(reactive_seed)).get("X")
                started = time.perf_counter()
                evaluated = blocked.evaluate(X, return_as_dictionary=True)
                reactive_seconds = time.perf_counter() - started
                feasible = np.asarray(evaluated["G"]).reshape(-1) <= 0
                reactive_f = np.asarray(evaluated["F"], dtype=float)[feasible]

                unit_rows.append({
                    "scenario": scenario,
                    "seed": seed,
                    "event_rank": event_rank,
                    "physical_edge_index": edge_index,
                    "edge_source": repr(edge[0]),
                    "edge_target": repr(edge[1]),
                    "objective_exposure_count": int(exposure[edge_index]),
                    "objective_immediate_success": bool(np.any(objective_survive)),
                    "echo_immediate_success": bool(np.any(echo_survive)),
                    "reactive_success": bool(np.any(feasible)),
                    "objective_survivor_count": int(np.count_nonzero(objective_survive)),
                    "echo_survivor_count": int(np.count_nonzero(echo_survive)),
                    "reactive_feasible_count": int(np.count_nonzero(feasible)),
                    "objective_best_normalized_score": best_score(
                        objective_f[objective_survive], ideal, scale),
                    "echo_best_normalized_score": best_score(
                        echo_f[echo_survive], ideal, scale),
                    "reactive_best_normalized_score": best_score(
                        reactive_f, ideal, scale),
                    "echo_lookup_seconds": echo_lookup_seconds,
                    "reactive_seconds": reactive_seconds,
                    "reactive_calls": REACTIVE_CALLS_PER_EVENT,
                    "source_precompute_seconds": float(source_metrics["elapsed_seconds"]),
                    "echo_storage_plans": len(echo_x),
                    "echo_storage_genotype_integers": int(echo_x.size),
                    "echo_storage_waypoints": echo_storage_waypoints,
                })
            unit_provenance = {
                "scenario": scenario, "seed": seed,
                "source_experiment": directory,
                "source_run": run_id,
                "source_snapshot": str(snapshot.relative_to(experiment)),
                "objective_portfolio_sha256": sha256(objective_file),
                "echo_portfolio_sha256": sha256(echo_file),
            }
            if len(unit_rows) != EVENTS_PER_UNIT:
                raise AssertionError(f"E7 unit row-count mismatch: {scenario}/{seed}")
            atomic_json(unit_file, {
                "provenance": unit_provenance,
                "rows": unit_rows,
            })
            rows.extend(unit_rows)
            units.append(unit_provenance)
            print(f"COMPLETE {scenario} seed {seed}", flush=True)

    if len(rows) != 400 or len(units) != 40:
        raise AssertionError("E7 row-count mismatch")
    write_csv(OUTPUT / "event_results.csv", rows)
    data = pd.DataFrame(rows)
    summary = data.groupby("scenario").agg(
        units=("seed", "nunique"),
        events=("event_rank", "size"),
        objective_immediate_success=("objective_immediate_success", "mean"),
        echo_immediate_success=("echo_immediate_success", "mean"),
        reactive_success=("reactive_success", "mean"),
        echo_survivor_count=("echo_survivor_count", "mean"),
        reactive_feasible_count=("reactive_feasible_count", "mean"),
        echo_lookup_seconds=("echo_lookup_seconds", "mean"),
        reactive_seconds=("reactive_seconds", "mean"),
        source_precompute_seconds=("source_precompute_seconds", "mean"),
    ).reset_index()
    summary.to_csv(OUTPUT / "scenario_summary.csv", index=False)
    atomic_json(OUTPUT / "units.json", units)
    provenance = {
        "schema": "echo-equal-budget-reactive-replanning-E7-v1",
        "protocol": "research_archive/plans/2026-09-05_echo_equal_budget_replanning_protocol_v1.md",
        "source_analysis": str(SOURCE_ANALYSIS),
        "source_analysis_checksums_sha256": sha256(SOURCE_ANALYSIS / "SHA256SUMS"),
        "scenarios": SCENARIOS,
        "seeds": SEEDS,
        "events_per_unit": EVENTS_PER_UNIT,
        "reactive_calls_per_event": REACTIVE_CALLS_PER_EVENT,
        "matched_total_calls": EVENTS_PER_UNIT * REACTIVE_CALLS_PER_EVENT,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
    }
    atomic_json(OUTPUT / "provenance.json", provenance)
    checksum = OUTPUT / "SHA256SUMS"
    files = sorted(path for path in OUTPUT.iterdir()
                   if path.is_file() and path != checksum)
    checksum.write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in files))


if __name__ == "__main__":
    main()
