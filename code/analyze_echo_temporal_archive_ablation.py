"""Compose the frozen five-cell ECHO temporal-archive ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from analyze_echo_selector_replay import fast_portfolio_metrics
from analyze_movingai_mapf_experiment import empirical_quality
from echo_portfolio import select_portfolio
from movingai_mapf_experiment import make_problem
from pymoo.util.ref_dirs import get_reference_directions


CELLS = {
    "no_archive_objective": ("NSGA3", False, "objective_only"),
    "objective_archive_objective": (
        "NSGA3-ObjectiveArchive", True, "objective_only"),
    "echo_archive_objective": ("NSGA3-ECHO", True, "objective_only"),
    "objective_archive_echo": ("NSGA3-ObjectiveArchive", True, "echo"),
    "full_echo": ("NSGA3-ECHO", True, "echo"),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_snapshot(run_path):
    paths = sorted((run_path / "trajectory").glob("generation_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no snapshots in {run_path}")
    return paths[-1]


def load_pool(run_path, include_archive):
    snapshot = latest_snapshot(run_path)
    with np.load(snapshot, allow_pickle=False) as saved:
        live_x = np.asarray(saved["X"], dtype=int)
        live_f = np.asarray(saved["F"], dtype=float)
        blocks_x, blocks_f = [live_x], [live_f]
        if include_archive and "archive_X" in saved.files and len(saved["archive_X"]):
            blocks_x.append(np.asarray(saved["archive_X"], dtype=int))
            blocks_f.append(np.asarray(saved["archive_F"], dtype=float))
    return np.row_stack(blocks_x), np.row_stack(blocks_f), live_x, live_f, snapshot


def write_csv(path, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="append", required=True,
                        help="LABEL=experiment-directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-fraction", type=float, default=.7)
    parser.add_argument(
        "--evidence-status", choices=("development", "confirmatory"),
        default="development")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    portfolio_dir = args.output / "portfolios"
    portfolio_dir.mkdir()
    rows, sources, invariants = [], [], []

    for value in args.experiment:
        label, raw_path = value.split("=", 1)
        experiment = Path(raw_path).resolve()
        config_path, manifest_path = experiment / "config.json", experiment / "manifest.json"
        config = json.loads(config_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"incomplete experiment: {experiment}")
        available = set(config["algorithms"])
        archive_algorithms = {"NSGA3-ObjectiveArchive", "NSGA3-ECHO"}
        if not archive_algorithms.issubset(available):
            raise ValueError(f"missing required archive algorithms in {experiment}")
        live_control = "NSGA3" if "NSGA3" in available else "NSGA3-ObjectiveArchive"
        required = archive_algorithms | {live_control}
        problem = make_problem(
            config["map"], config["scenario"], config["agents"],
            task_indices=config.get("task_indices"),
            preference_levels=config.get("preference_levels", 5),
            independent_neighbor_orders=bool(config.get("independent_neighbor_orders", False)),
        )
        ref_dirs = get_reference_directions(
            "energy", problem.n_obj, config["population"], seed=1)
        for seed in config["seeds"]:
            live = {}
            for algorithm in required:
                run_path = experiment / "runs" / f"{algorithm}_seed{seed:03d}"
                _, _, live_x, live_f, snapshot = load_pool(run_path, False)
                live[algorithm] = (live_x, live_f, snapshot)
            reference_x, reference_f, _ = live[live_control]
            for algorithm in sorted(required - {live_control}):
                same_x = np.array_equal(reference_x, live[algorithm][0])
                same_f = np.array_equal(reference_f, live[algorithm][1])
                invariants.append({
                    "problem": label, "seed": seed, "algorithm": algorithm,
                    "same_live_X_as_nsga3": same_x,
                    "same_live_F_as_nsga3": same_f,
                })
                if not (same_x and same_f):
                    raise AssertionError(
                        f"passive archive changed live population: {label}/{seed}/{algorithm}")

            seed_rows = []
            for cell, (algorithm, include_archive, selector) in CELLS.items():
                if algorithm == "NSGA3" and algorithm not in available:
                    algorithm = live_control
                run_path = experiment / "runs" / f"{algorithm}_seed{seed:03d}"
                X, F, _, _, snapshot = load_pool(run_path, include_archive)
                started = time.perf_counter()
                selected = select_portfolio(
                    problem, X, F, config["population"], selector=selector,
                    quality_fraction=args.quality_fraction,
                    random_seed=int(seed), ref_dirs=ref_dirs,
                )
                elapsed = time.perf_counter() - started
                name = f"{label}_seed{seed:03d}_{cell}.npz"
                np.savez_compressed(
                    portfolio_dir / name, X=selected.X, F=selected.F,
                    source_indices=selected.source_indices)
                row = {
                    "problem": label, "seed": seed, "cell": cell,
                    "archive_policy": "none" if not include_archive else (
                        "echo" if algorithm == "NSGA3-ECHO" else "objective_only"),
                    "composer": selector,
                    "raw_pool_size": len(X),
                    "selection_seconds": elapsed,
                    "snapshot": str(snapshot.relative_to(experiment)),
                    "portfolio_file": f"portfolios/{name}",
                    **fast_portfolio_metrics(problem, selected.X, selected.F),
                }
                rows.append(row)
                seed_rows.append(row)
            quality_inputs = {}
            for row in seed_rows:
                with np.load(args.output / row["portfolio_file"], allow_pickle=False) as saved:
                    quality_inputs[row["cell"]] = np.asarray(saved["F"], dtype=float)
            quality = empirical_quality(quality_inputs)
            for row in seed_rows:
                row.update(quality[row["cell"]])
        sources.append({
            "label": label, "experiment": str(experiment),
            "config_sha256": sha256(config_path),
            "manifest_sha256": sha256(manifest_path),
        })

    write_csv(args.output / "ablation_results.csv", rows)
    write_csv(args.output / "live_population_invariants.csv", invariants)
    metric_names = (
        "exact_group_undirected_edge_coverage",
        "exact_group_undirected_two_edge_coverage",
        "unique_plan_count", "nondominated_output_size",
        "mean_sum_of_costs", "mean_clearance_risk", "selection_seconds",
        "igd_empirical", "hv_ratio_empirical",
    )
    summary = []
    for problem in sorted({row["problem"] for row in rows}):
        for cell in CELLS:
            subset = [row for row in rows if row["problem"] == problem and row["cell"] == cell]
            summary.append({
                "problem": problem, "cell": cell, "seeds": len(subset),
                **{f"{metric}_mean": float(np.mean([row[metric] for row in subset]))
                   for metric in metric_names},
            })
    write_csv(args.output / "ablation_summary.csv", summary)
    provenance = {
        "schema": "echo-temporal-archive-five-cell-ablation-v1",
        "evidence_status": args.evidence_status,
        "cells": CELLS,
        "quality_fraction": args.quality_fraction,
        "sources": sources,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(__file__),
        "invariant": (
            "all archived variants must have bitwise-identical live X/F; "
            "ordinary NSGA3 is used when present, otherwise the passive "
            "objective-archive run is the live control"),
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    checksum = args.output / "SHA256SUMS"
    files = sorted(path for path in args.output.rglob("*")
                   if path.is_file() and path != checksum)
    checksum.write_text("".join(
        f"{sha256(path)}  {path.relative_to(args.output)}\n" for path in files))


if __name__ == "__main__":
    main()
