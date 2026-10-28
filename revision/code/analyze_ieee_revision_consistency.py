"""Post-review consistency replay for Access-2026-44964.

The analysis reuses immutable pre-submission candidate pools.  It never alters
search trajectories and writes resumable unit records into a new append-only
analysis directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_echo_final_multifamily import (
    EXPERIMENTS,
    RUN_GROUPS,
    load_generator_pool,
)
from analyze_echo_selector_replay import fast_portfolio_metrics
from analyze_movingai_mapf_experiment import empirical_quality
from echo_portfolio import _deduplicated_nondominated_pool
from echo_revision import (
    fixed_event_universe,
    portfolio_coverage_metrics,
    select_strict_coverage_portfolio,
)
from movingai_mapf_experiment import make_problem
from pymoo.util.ref_dirs import get_reference_directions
from research_archive import atomic_json


ROOT = Path(__file__).resolve().parent
SOURCE = (ROOT / "research_archive" / "analyses" /
          "2026-09-04_echo-final-multifamily_selector-confirmation_v1")
OUTPUT = (ROOT / "research_archive" / "analyses" /
          "2026-10-28_ieee-revision_portfolio-consistency_v1")
PROTOCOL = (ROOT / "research_archive" / "plans" /
            "2026-10-28_ieee-access-revision-experiment-protocol_v1.md")
OLD_SELECTORS = (
    "objective_only",
    "echo",
    "edge_jaccard",
    "minimum_common_edge",
    "kmedoids_edge",
    "common_vertex_overlap",
    "xu2025_paper_vertex",
    "xu2025_official_edge",
)
STRICT_SELECTORS = (
    "strict_coverage_psc1_only",
    "strict_coverage_psc1_psc2_tie",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_portfolio(scenario, algorithm, seed, selector):
    path = SOURCE / "portfolios" / (
        f"{scenario}_{algorithm}_seed{seed:03d}_{selector}_r00.npz")
    with np.load(path, allow_pickle=False) as saved:
        return (np.asarray(saved["X"], dtype=int),
                np.asarray(saved["F"], dtype=float), path)


def family(scenario: str) -> str:
    if scenario.startswith("wh"):
        return "warehouse"
    if scenario.startswith("maze"):
        return "maze"
    return scenario.rstrip("32")


def direct_signature_check(problem, selected_x, universe):
    """Cross-check edge signatures against decoded physical paths."""
    event_indices = np.flatnonzero(universe)
    checked = event_indices[:min(32, len(event_indices))]
    embedding = np.asarray(problem.failure_event_embedding(selected_x), dtype=bool)
    for row, decision in enumerate(selected_x):
        decoded = problem.decode(decision)
        used = set()
        for path in decoded["paths"]:
            for source, target in zip(path[:-1], path[1:]):
                if source == target:
                    continue
                edge = (source, target) if source < target else (target, source)
                used.add(problem.physical_edge_index[edge])
        for event in checked:
            if bool(embedding[row, event]) != (int(event) in used):
                raise AssertionError("edge-signature/direct-path mismatch")
    return int(len(checked) * len(selected_x))


def unit_rows(problem, scenario, seed, algorithm, raw_x, raw_f,
              size, ref_dirs, search_seconds, source_times, portfolio_dir):
    eligible_x, eligible_f, _ = _deduplicated_nondominated_pool(
        problem, raw_x, raw_f)
    pool_events = np.asarray(
        problem.failure_event_embedding(eligible_x), dtype=bool)
    universe = fixed_event_universe(pool_events)
    outputs = {}
    timings = {}
    portfolio_sources = {}
    for selector in OLD_SELECTORS:
        X, F, path = source_portfolio(scenario, algorithm, seed, selector)
        label = "legacy_group_heuristic" if selector == "echo" else selector
        outputs[label] = (X, F)
        key = (scenario, algorithm, seed, selector)
        timings[label] = float(source_times.get(key, np.nan))
        portfolio_sources[label] = str(path.relative_to(ROOT))
    for use_psc2, label in (
        (False, "strict_coverage_psc1_only"),
        (True, "strict_coverage_psc1_psc2_tie"),
    ):
        started = time.perf_counter()
        selected = select_strict_coverage_portfolio(
            problem, raw_x, raw_f, size,
            quality_fraction=.7,
            random_seed=seed * 1000,
            ref_dirs=ref_dirs,
            use_psc2_tiebreak=use_psc2,
        )
        timings[label] = time.perf_counter() - started
        outputs[label] = (selected.X, selected.F)
        path = portfolio_dir / f"{scenario}_{algorithm}_seed{seed:03d}_{label}.npz"
        np.savez_compressed(
            path, X=selected.X, F=selected.F,
            source_indices=selected.source_indices,
            quality_core_size=np.asarray([selected.quality_core_size]),
            shortfall=np.asarray([selected.shortfall]),
        )
        portfolio_sources[label] = str(path.relative_to(ROOT))

    quality = empirical_quality({label: value[1]
                                 for label, value in outputs.items()})
    rows = []
    direct_checks = 0
    for label, (X, F) in outputs.items():
        selected_events = np.asarray(
            problem.failure_event_embedding(X), dtype=bool)
        global_metrics = portfolio_coverage_metrics(selected_events, universe)
        legacy_metrics = fast_portfolio_metrics(problem, X, F)
        if label.startswith("strict_coverage"):
            direct_checks += direct_signature_check(problem, X, universe)
        rows.append({
            "scenario": scenario,
            "family": family(scenario),
            "seed": int(seed),
            "generator": algorithm,
            "selector": label,
            "raw_pool_size": int(len(raw_x)),
            "eligible_pool_size": int(len(eligible_x)),
            "pool_event_universe_size": int(np.count_nonzero(universe)),
            "selected_plan_count": int(len(X)),
            "unique_plan_count": int(legacy_metrics["unique_plan_count"]),
            "cardinality_shortfall": int(max(0, size - len(X))),
            "selection_seconds": float(timings[label]),
            "search_seconds": float(search_seconds),
            "search_plus_selection_seconds": float(
                search_seconds + timings[label]),
            "legacy_group_edge_diversity": float(
                legacy_metrics["exact_group_undirected_edge_coverage"]),
            "legacy_group_two_edge_diversity": float(
                legacy_metrics["exact_group_undirected_two_edge_coverage"]),
            "mean_sum_of_costs": float(legacy_metrics["mean_sum_of_costs"]),
            "mean_clearance_risk": float(
                legacy_metrics["mean_clearance_risk"]),
            "mean_pairwise_edge_jaccard": float(
                legacy_metrics["mean_pairwise_edge_jaccard"]),
            "nondominated_output_size": int(
                legacy_metrics["nondominated_output_size"]),
            "igd_empirical": float(quality[label]["igd_empirical"]),
            "hv_ratio_empirical": float(quality[label]["hv_ratio_empirical"]),
            "portfolio_source": portfolio_sources[label],
            **global_metrics,
        })
    return rows, direct_checks


def main():
    if (OUTPUT / "SHA256SUMS").exists():
        print(f"ALREADY COMPLETE {OUTPUT}", flush=True)
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    units_dir = OUTPUT / "units"
    units_dir.mkdir(exist_ok=True)
    portfolio_dir = OUTPUT / "portfolios"
    portfolio_dir.mkdir(exist_ok=True)
    source_table = pd.read_csv(SOURCE / "selector_results.csv")
    source_times = {
        (str(row.scenario), str(row.generator), int(row.seed), str(row.selector)):
        float(row.selection_seconds)
        for row in source_table.itertuples()
        if int(row.selector_repeat) == 0
    }
    all_rows = []
    source_records = []
    seen = set()
    direct_checks = 0
    for scenario, directory, first_seed, last_seed in RUN_GROUPS:
        experiment = EXPERIMENTS / directory
        config = json.loads((experiment / "config.json").read_text())
        manifest = json.loads((experiment / "manifest.json").read_text())
        problem = make_problem(
            config["map"], config["scenario"], int(config["agents"]),
            task_indices=config.get("task_indices"),
            preference_levels=int(config.get("preference_levels", 5)),
            independent_neighbor_orders=bool(
                config.get("independent_neighbor_orders", False)),
        )
        size = int(config["population"])
        ref_dirs = get_reference_directions(
            "energy", problem.n_obj, size, seed=1)
        completed = {item["run_id"] for item in manifest.get("runs", [])
                     if item.get("status") == "complete"}
        for seed in range(first_seed, last_seed + 1):
            for algorithm in ("NSGA3-ObjectiveArchive", "NSGA2"):
                key = (scenario, seed, algorithm)
                if key in seen:
                    raise RuntimeError(f"duplicate unit {key}")
                seen.add(key)
                unit_file = units_dir / f"{scenario}_{algorithm}_seed{seed:03d}.json"
                if unit_file.exists():
                    saved = json.loads(unit_file.read_text())
                    all_rows.extend(saved["rows"])
                    direct_checks += int(saved["direct_signature_checks"])
                    print(f"SKIP {scenario} {algorithm} seed {seed}", flush=True)
                    continue
                run_id = f"{algorithm}_seed{seed:03d}"
                if run_id not in completed:
                    raise RuntimeError(f"incomplete source {directory}/{run_id}")
                run_path = experiment / "runs" / run_id
                raw_x, raw_f, snapshot = load_generator_pool(
                    problem, run_path, algorithm, 2 * size)
                run_metrics = json.loads((run_path / "metrics.json").read_text())
                rows, checks = unit_rows(
                    problem, scenario, seed, algorithm, raw_x, raw_f,
                    size, ref_dirs, float(run_metrics["elapsed_seconds"]),
                    source_times, portfolio_dir)
                payload = {
                    "source_experiment": directory,
                    "source_run": run_id,
                    "source_snapshot": str(snapshot.relative_to(experiment)),
                    "source_snapshot_sha256": sha256(snapshot),
                    "direct_signature_checks": checks,
                    "rows": rows,
                }
                atomic_json(unit_file, payload)
                all_rows.extend(rows)
                direct_checks += checks
                print(f"COMPLETE {scenario} {algorithm} seed {seed}", flush=True)
        source_records.append({
            "scenario": scenario,
            "experiment": directory,
            "seed_start": first_seed,
            "seed_end": last_seed,
            "config_sha256": sha256(experiment / "config.json"),
            "manifest_sha256": sha256(experiment / "manifest.json"),
        })
    if len(seen) != 320:
        raise AssertionError(f"expected 320 units, got {len(seen)}")
    expected_rows = 320 * (len(OLD_SELECTORS) + len(STRICT_SELECTORS))
    if len(all_rows) != expected_rows:
        raise AssertionError(f"expected {expected_rows} rows, got {len(all_rows)}")
    write_csv(OUTPUT / "selector_results.csv", all_rows)
    summary = []
    metrics = (
        "portfolio_psc1", "portfolio_psc2", "hv_ratio_empirical",
        "igd_empirical", "unique_plan_count", "selection_seconds",
        "search_plus_selection_seconds", "legacy_group_edge_diversity",
    )
    for scenario in sorted({row["scenario"] for row in all_rows}):
        for generator in ("NSGA3-ObjectiveArchive", "NSGA2"):
            for selector in tuple(OLD_SELECTORS[:-7]) + (
                    "legacy_group_heuristic", "edge_jaccard",
                    "minimum_common_edge", "kmedoids_edge",
                    "common_vertex_overlap", "xu2025_paper_vertex",
                    "xu2025_official_edge", *STRICT_SELECTORS):
                subset = [row for row in all_rows
                          if row["scenario"] == scenario
                          and row["generator"] == generator
                          and row["selector"] == selector]
                if not subset:
                    continue
                summary.append({
                    "scenario": scenario,
                    "generator": generator,
                    "selector": selector,
                    "seeds": len(subset),
                    **{f"{metric}_mean": float(np.mean(
                        [float(row[metric]) for row in subset]))
                       for metric in metrics},
                })
    write_csv(OUTPUT / "selector_summary.csv", summary)
    atomic_json(OUTPUT / "provenance.json", {
        "schema": "ieee-access-revision-portfolio-consistency-v1",
        "evidence_status": "post-review revision analysis",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "source_analysis": str(SOURCE.relative_to(ROOT)),
        "source_checksums_sha256": sha256(SOURCE / "SHA256SUMS"),
        "source_records": source_records,
        "selectors": list(OLD_SELECTORS) + list(STRICT_SELECTORS),
        "portfolio_event_universe": "pool-supported undirected physical edges",
        "direct_signature_checks": direct_checks,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "strict_module_sha256": sha256(ROOT / "echo_revision.py"),
    })
    checksum = OUTPUT / "SHA256SUMS"
    files = sorted(path for path in OUTPUT.rglob("*")
                   if path.is_file() and path != checksum)
    checksum.write_text("".join(
        f"{sha256(path)}  {path.relative_to(OUTPUT)}\n" for path in files))
    print(f"COMPLETE {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
