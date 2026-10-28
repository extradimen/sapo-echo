"""Analyze the frozen larger-fleet revision panel under pool-fixed coverage.

This script is intentionally separate from the search runner.  It reads only
sealed experiment units, compares objective-only and corrected coverage
composition on the identical eligible pool, and writes append-only unit files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from analyze_echo_final_multifamily import load_generator_pool
from analyze_echo_selector_replay import fast_portfolio_metrics
from analyze_movingai_mapf_experiment import empirical_quality
from echo_portfolio import _deduplicated_nondominated_pool, select_portfolio
from echo_revision import (
    fixed_event_universe,
    portfolio_coverage_metrics,
    select_strict_coverage_portfolio,
)
from movingai_mapf_experiment import make_problem
from pymoo.util.ref_dirs import get_reference_directions
from research_archive import atomic_json


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "research_archive" / "experiments"
OUTPUT = (ROOT / "research_archive" / "analyses" /
          "2026-10-28_ieee-revision_larger-fleet_v1")
PROTOCOL = (ROOT / "research_archive" / "plans" /
            "2026-10-28_ieee-access-revision-experiment-protocol_v1.md")
SCENARIOS = ("wh20a", "random32", "room32", "maze32b")
FLEETS = (18, 24, 36)


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


def load_unit(scenario: str, agents: int, seed: int):
    experiment_id = (
        f"2026-10-28_ieee-revision-scaling_{scenario}_"
        f"agents{agents}_seed{seed}_v1")
    experiment = EXPERIMENTS / experiment_id
    manifest_path = experiment / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise RuntimeError(f"source experiment is not complete: {experiment_id}")
    config = json.loads((experiment / "config.json").read_text())
    problem = make_problem(
        config["map"], config["scenario"], int(config["agents"]),
        task_indices=config.get("task_indices"),
        preference_levels=int(config.get("preference_levels", 5)),
        independent_neighbor_orders=bool(
            config.get("independent_neighbor_orders", False)),
    )
    run_id = f"NSGA3-ObjectiveArchive_seed{seed:03d}"
    complete_runs = {
        record["run_id"] for record in manifest.get("runs", [])
        if record.get("status") == "complete"
    }
    if run_id not in complete_runs:
        raise RuntimeError(f"missing complete run: {experiment_id}/{run_id}")
    run_path = experiment / "runs" / run_id
    raw_x, raw_f, snapshot = load_generator_pool(
        problem, run_path, "NSGA3-ObjectiveArchive", 60)
    metadata = json.loads((run_path / "metadata.json").read_text())
    metrics = json.loads((run_path / "metrics.json").read_text())
    search_seconds = float(
        metadata.get("elapsed_seconds", metrics.get("elapsed_seconds", np.nan)))
    return experiment, problem, raw_x, raw_f, snapshot, search_seconds


def analyze_unit(scenario: str, agents: int, seed: int) -> list[dict]:
    experiment, problem, raw_x, raw_f, snapshot, search_seconds = load_unit(
        scenario, agents, seed)
    size = 30
    ref_dirs = get_reference_directions("energy", problem.n_obj, size, seed=1)
    eligible_x, eligible_f, _ = _deduplicated_nondominated_pool(
        problem, raw_x, raw_f)
    universe = fixed_event_universe(problem.failure_event_embedding(eligible_x))
    outputs = {}
    times = {}
    started = time.perf_counter()
    baseline = select_portfolio(
        problem, raw_x, raw_f, size, selector="objective_only",
        quality_fraction=.7, random_seed=seed * 1000, ref_dirs=ref_dirs)
    times["objective_only"] = time.perf_counter() - started
    outputs["objective_only"] = (baseline.X, baseline.F)
    started = time.perf_counter()
    corrected = select_strict_coverage_portfolio(
        problem, raw_x, raw_f, size, quality_fraction=.7,
        random_seed=seed * 1000, ref_dirs=ref_dirs,
        use_psc2_tiebreak=True)
    times["strict_coverage_psc1_psc2_tie"] = time.perf_counter() - started
    outputs["strict_coverage_psc1_psc2_tie"] = (corrected.X, corrected.F)
    quality = empirical_quality({name: value[1] for name, value in outputs.items()})
    rows = []
    for selector, (X, F) in outputs.items():
        legacy = fast_portfolio_metrics(problem, X, F)
        global_metrics = portfolio_coverage_metrics(
            problem.failure_event_embedding(X), universe)
        rows.append({
            "scenario": scenario,
            "agents": agents,
            "seed": seed,
            "selector": selector,
            "raw_pool_size": len(raw_x),
            "eligible_pool_size": len(eligible_x),
            "pool_event_universe_size": int(np.count_nonzero(universe)),
            "selected_plan_count": len(X),
            "unique_plan_count": int(legacy["unique_plan_count"]),
            "cardinality_shortfall": max(0, size - len(X)),
            "mean_sum_of_costs": float(legacy["mean_sum_of_costs"]),
            "mean_clearance_risk": float(legacy["mean_clearance_risk"]),
            "igd_empirical": float(quality[selector]["igd_empirical"]),
            "hv_ratio_empirical": float(quality[selector]["hv_ratio_empirical"]),
            "search_seconds": search_seconds,
            "selection_seconds": times[selector],
            "search_plus_selection_seconds": search_seconds + times[selector],
            "source_experiment": str(experiment.relative_to(ROOT)),
            "source_snapshot": str(snapshot.relative_to(ROOT)),
            **global_metrics,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--fleets", default="24,36")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    fleets = tuple(int(value) for value in args.fleets.split(","))
    scenarios = tuple(value for value in args.scenarios.split(",") if value)
    if not set(fleets) <= set(FLEETS):
        raise ValueError(f"unsupported fleet size: {fleets}")
    if not set(scenarios) <= set(SCENARIOS):
        raise ValueError(f"unsupported scenario: {scenarios}")
    if (OUTPUT / "SHA256SUMS").exists():
        print(f"ALREADY COMPLETE {OUTPUT}")
        return
    units = OUTPUT / "units"
    units.mkdir(parents=True, exist_ok=True)
    for scenario in scenarios:
        for agents in fleets:
            for seed in seeds:
                path = units / f"{scenario}_agents{agents}_seed{seed}.json"
                if path.exists():
                    print(f"SKIP COMPLETE {path.name}", flush=True)
                    continue
                rows = analyze_unit(scenario, agents, seed)
                atomic_json(path, {"schema": "ieee-revision-scaling-unit-v1",
                                   "rows": rows})
                print(f"COMPLETE {path.name}", flush=True)
    if not args.finalize:
        return
    expected = {f"{scenario}_agents{agents}_seed{seed}.json"
                for scenario in scenarios for agents in fleets for seed in seeds}
    observed = {path.name for path in units.glob("*.json")}
    missing = expected - observed
    if missing:
        raise RuntimeError(f"cannot finalize; missing {len(missing)} units")
    rows = []
    for path in sorted(units.glob("*.json")):
        rows.extend(json.loads(path.read_text())["rows"])
    write_csv(OUTPUT / "scaling_results.csv", rows)
    atomic_json(OUTPUT / "provenance.json", {
        "schema": "ieee-revision-scaling-analysis-v1",
        "evidence_status": "post-review revision analysis",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "seeds": seeds, "fleets": fleets, "scenarios": scenarios,
        "platform": platform.platform(),
    })
    checksum = OUTPUT / "SHA256SUMS"
    files = sorted(path for path in OUTPUT.rglob("*")
                   if path.is_file() and path != checksum)
    checksum.write_text("".join(
        f"{sha256(path)}  {path.relative_to(OUTPUT)}\n" for path in files))
    print(f"COMPLETE {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
