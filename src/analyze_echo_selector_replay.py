"""Replay competing portfolio selectors on identical archived ECHO pools."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from echo_portfolio import SELECTORS, select_portfolio
from analyze_movingai_mapf_experiment import empirical_quality
from movingai_mapf_experiment import make_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_input(value):
    label, path = value.split("=", 1)
    return label, Path(path).resolve()


def latest_snapshot(run_path):
    snapshots = sorted((run_path / "trajectory").glob("generation_*.npz"))
    if not snapshots:
        raise FileNotFoundError(f"no generation snapshot in {run_path}")
    return snapshots[-1]


def load_precomposition_pool(run_path):
    snapshot = latest_snapshot(run_path)
    with np.load(snapshot, allow_pickle=False) as saved:
        blocks_x = [np.asarray(saved["X"], dtype=int)]
        blocks_f = [np.asarray(saved["F"], dtype=float)]
        if "archive_X" in saved.files and len(saved["archive_X"]):
            blocks_x.append(np.asarray(saved["archive_X"], dtype=int))
            blocks_f.append(np.asarray(saved["archive_F"], dtype=float))
    return np.row_stack(blocks_x), np.row_stack(blocks_f), snapshot


def write_csv(path, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fast_portfolio_metrics(problem, X, F):
    """Score exact edge-failure endpoints without unrelated path diagnostics."""
    X = np.asarray(X)
    F = np.asarray(F, dtype=float)
    events = np.asarray(problem.failure_event_embedding(X), dtype=bool)
    identities = np.asarray(problem.decision_identity(X), dtype=str)
    groups = {}
    for index, objective in enumerate(F):
        groups.setdefault(tuple(objective), []).append(index)
    survived_single = exposed_single = 0
    survived_double = exposed_double = 0
    for indices in groups.values():
        group_events = events[indices]
        active = np.any(group_events, axis=0)
        common = np.all(group_events, axis=0)
        survived_single += int(np.count_nonzero(active & ~common))
        exposed_single += int(np.count_nonzero(active))
        active_events = group_events[:, active]
        event_count = active_events.shape[1]
        if event_count < 2:
            continue
        masks = []
        for column in range(event_count):
            mask = 0
            for position in np.flatnonzero(active_events[:, column]):
                mask |= 1 << int(position)
            masks.append(mask)
        counts = Counter(masks)
        all_plans = (1 << len(indices)) - 1
        exposed_double += event_count * (event_count - 1) // 2
        failed = 0
        values = sorted(counts)
        for left_position, left in enumerate(values):
            if (left | left) == all_plans:
                failed += counts[left] * (counts[left] - 1) // 2
            for right in values[left_position + 1:]:
                if (left | right) == all_plans:
                    failed += counts[left] * counts[right]
        survived_double += event_count * (event_count - 1) // 2 - failed
    pairwise = []
    for left in range(len(events)):
        for right in range(left + 1, len(events)):
            union = np.count_nonzero(events[left] | events[right])
            pairwise.append(
                0.0 if union == 0 else
                1.0 - np.count_nonzero(events[left] & events[right]) / union
            )
    return {
        "exact_group_undirected_edge_coverage": (
            survived_single / exposed_single if exposed_single else 0.0),
        "exact_group_undirected_two_edge_coverage": (
            survived_double / exposed_double if exposed_double else 0.0),
        "unique_plan_count": len(set(identities.tolist())),
        "nondominated_output_size": len(
            NonDominatedSorting().do(F, only_non_dominated_front=True)),
        "mean_sum_of_costs": float(np.mean(F[:, 0])),
        "mean_clearance_risk": float(np.mean(F[:, 1])),
        "mean_pairwise_edge_jaccard": (
            float(np.mean(pairwise)) if pairwise else 0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="append", required=True,
                        help="LABEL=experiment-directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--algorithm", default="NSGA3-EdgeCoverage")
    parser.add_argument("--selectors", type=lambda value: value.split(","),
                        default=list(SELECTORS))
    parser.add_argument("--quality-fraction", type=float, default=.7)
    parser.add_argument("--random-replicates", type=int, default=20)
    args = parser.parse_args()
    unknown = sorted(set(args.selectors) - set(SELECTORS))
    if unknown:
        raise ValueError(f"unknown selectors: {unknown}")
    args.output.mkdir(parents=True, exist_ok=False)
    portfolio_dir = args.output / "portfolios"
    portfolio_dir.mkdir()

    rows = []
    sources = []
    for label, experiment in map(parse_input, args.experiment):
        config_path = experiment / "config.json"
        manifest_path = experiment / "manifest.json"
        config = json.loads(config_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if manifest["status"] != "complete":
            raise ValueError(f"incomplete experiment: {experiment}")
        if args.algorithm not in config["algorithms"]:
            raise ValueError(f"{args.algorithm} absent from {experiment}")
        problem = make_problem(
            config["map"], config["scenario"], config["agents"],
            task_indices=config.get("task_indices"),
            preference_levels=config.get("preference_levels", 5),
            independent_neighbor_orders=bool(
                config.get("independent_neighbor_orders", False)),
        )
        ref_dirs = get_reference_directions(
            "energy", problem.n_obj, config["population"], seed=1)
        for seed in config["seeds"]:
            run_path = experiment / "runs" / f"{args.algorithm}_seed{seed:03d}"
            X, F, snapshot = load_precomposition_pool(run_path)
            selectors = []
            for selector in args.selectors:
                repeats = (args.random_replicates
                           if selector == "random_quality_matched" else 1)
                selectors.extend((selector, repeat) for repeat in range(repeats))
            for selector, repeat in selectors:
                selector_seed = int(seed * 1000 + repeat)
                started = time.perf_counter()
                result = select_portfolio(
                    problem, X, F, config["population"],
                    selector=selector,
                    quality_fraction=args.quality_fraction,
                    random_seed=selector_seed,
                    ref_dirs=ref_dirs,
                )
                selection_seconds = time.perf_counter() - started
                metrics = fast_portfolio_metrics(problem, result.X, result.F)
                key = f"{label}_seed{seed:03d}_{selector}_r{repeat:02d}"
                portfolio_path = portfolio_dir / f"{key}.npz"
                np.savez_compressed(
                    portfolio_path,
                    X=result.X,
                    F=result.F,
                    source_indices=result.source_indices,
                )
                rows.append({
                    "problem": label,
                    "seed": seed,
                    "selector": selector,
                    "selector_repeat": repeat,
                    "source_algorithm": args.algorithm,
                    "raw_pool_size": len(X),
                    "quality_core_size": result.quality_core_size,
                    "selection_seconds": selection_seconds,
                    "portfolio_file": str(portfolio_path.relative_to(args.output)),
                    **metrics,
                })
        sources.append({
            "label": label,
            "experiment": str(experiment),
            "config_sha256": sha256(config_path),
            "manifest_sha256": sha256(manifest_path),
        })

    for problem in sorted({row["problem"] for row in rows}):
        for seed in sorted({row["seed"] for row in rows if row["problem"] == problem}):
            subset = [row for row in rows
                      if row["problem"] == problem and row["seed"] == seed]
            outputs = {}
            for row in subset:
                key = f"{row['selector']}_r{int(row['selector_repeat']):02d}"
                with np.load(args.output / row["portfolio_file"], allow_pickle=False) as saved:
                    outputs[key] = np.asarray(saved["F"], dtype=float)
            quality = empirical_quality(outputs)
            for row in subset:
                key = f"{row['selector']}_r{int(row['selector_repeat']):02d}"
                row.update(quality[key])

    write_csv(args.output / "selector_results.csv", rows)
    summary = []
    metric_names = (
        "exact_group_undirected_edge_coverage",
        "exact_group_undirected_two_edge_coverage",
        "unique_plan_count",
        "nondominated_output_size",
        "mean_sum_of_costs",
        "mean_clearance_risk",
        "mean_pairwise_edge_jaccard",
        "selection_seconds",
        "igd_empirical",
        "hv_ratio_empirical",
        "empirical_reference_size",
    )
    for problem in sorted({row["problem"] for row in rows}):
        for selector in args.selectors:
            subset = [row for row in rows
                      if row["problem"] == problem and row["selector"] == selector]
            summary.append({
                "problem": problem,
                "selector": selector,
                "rows": len(subset),
                "independent_seeds": len({row["seed"] for row in subset}),
                **{
                    f"{metric}_mean": float(np.mean([row[metric] for row in subset]))
                    for metric in metric_names
                },
            })
    write_csv(args.output / "selector_summary.csv", summary)
    provenance = {
        "schema": "echo-identical-final-pool-selector-replay-v1",
        "evidence_status": "development; post-hoc selector comparison",
        "source_pool": (
            "Final live population plus the passive failure-coverage archive "
            "from the last saved pre-composition snapshot. This gives every "
            "final selector identical candidates but does not compare "
            "counterfactual temporal archive policies because early ordinary "
            "NSGA-III generations were not always recorded."
        ),
        "selectors": args.selectors,
        "quality_fraction": args.quality_fraction,
        "random_replicates": args.random_replicates,
        "sources": sources,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(__file__),
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    files = sorted(path for path in args.output.rglob("*") if path.is_file())
    (args.output / "SHA256SUMS").write_text("".join(
        f"{sha256(path)}  {path.relative_to(args.output)}\n" for path in files
    ))


if __name__ == "__main__":
    main()
