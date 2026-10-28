"""Audit whether sealed E7 closures were represented in the frozen pool."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_echo_final_multifamily import (
    EXPERIMENTS,
    RUN_GROUPS,
    load_generator_pool,
)
from echo_portfolio import _deduplicated_nondominated_pool
from movingai_mapf_experiment import make_problem
from research_archive import atomic_json


ROOT = Path(__file__).resolve().parent
E7 = (ROOT / "research_archive" / "experiments" /
      "2026-09-05_echo-equal-budget-replanning_E7_v2")
PORTFOLIOS = (ROOT / "research_archive" / "analyses" /
              "2026-09-04_echo-final-multifamily_selector-confirmation_v1" /
              "portfolios")
OUTPUT = (ROOT / "research_archive" / "analyses" /
          "2026-09-05_echo-E7_pool-support-audit_v2")
PROTOCOL = (ROOT / "research_archive" / "plans" /
            "2026-09-05_echo_E7_pool-support_audit_protocol_v1.md")
GENERATOR = "NSGA3-ObjectiveArchive"
SELECTORS = (
    "objective_only",
    "edge_jaccard",
    "minimum_common_edge",
    "kmedoids_edge",
    "common_vertex_overlap",
    "xu2025_paper_vertex",
    "xu2025_official_edge",
    "echo",
)


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


def source_group(scenario, seed):
    matches = [group for group in RUN_GROUPS
               if group[0] == scenario and group[2] <= seed <= group[3]]
    if len(matches) != 1:
        raise RuntimeError(f"source mapping mismatch: {scenario}/{seed}")
    return matches[0]


def load_portfolio(scenario, seed, selector):
    path = PORTFOLIOS / (
        f"{scenario}_{GENERATOR}_seed{seed:03d}_{selector}_r00.npz")
    with np.load(path, allow_pickle=False) as saved:
        return np.asarray(saved["X"], dtype=int), path


def best_targeted_coverage(core_survival, candidate_survival, slots):
    """Exact event-mask DP; returns the most E7 events coverable."""
    event_count = core_survival.shape[1]
    covered = 0
    for event in range(event_count):
        if np.any(core_survival[:, event]):
            covered |= 1 << event
    masks = set()
    for row in candidate_survival:
        mask = 0
        for event in np.flatnonzero(row):
            mask |= 1 << int(event)
        masks.add(mask)
    states = {covered}
    for _ in range(slots):
        states |= {state | mask for state in tuple(states) for mask in masks}
    return max(mask.bit_count() for mask in states)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    rows = []
    unit_rows = []
    sources = []

    for unit_file in sorted((E7 / "units").glob("*.json")):
        sealed = json.loads(unit_file.read_text())
        event_rows = sealed["rows"]
        provenance = sealed["provenance"]
        scenario = provenance["scenario"]
        seed = int(provenance["seed"])
        _, directory, _, _ = source_group(scenario, seed)
        experiment = EXPERIMENTS / directory
        config = json.loads((experiment / "config.json").read_text())
        problem = make_problem(
            config["map"], config["scenario"], int(config["agents"]),
            task_indices=config.get("task_indices"),
            preference_levels=int(config.get("preference_levels", 5)),
            independent_neighbor_orders=bool(
                config.get("independent_neighbor_orders", False)),
        )
        run_path = experiment / "runs" / f"{GENERATOR}_seed{seed:03d}"
        raw_x, raw_f, snapshot = load_generator_pool(
            problem, run_path, GENERATOR, 2 * int(config["population"]))
        eligible_x, eligible_f, _ = _deduplicated_nondominated_pool(
            problem, raw_x, raw_f)
        raw_usage = problem.failure_event_embedding(raw_x).astype(bool)
        eligible_usage = problem.failure_event_embedding(eligible_x).astype(bool)

        portfolio_data = {}
        portfolio_files = {}
        for selector in SELECTORS:
            X, path = load_portfolio(scenario, seed, selector)
            portfolio_data[selector] = problem.failure_event_embedding(X).astype(bool)
            portfolio_files[selector] = path

        edge_indices = [int(row["physical_edge_index"]) for row in event_rows]
        echo_x, _ = load_portfolio(scenario, seed, "echo")
        quality_core_size = int(round(.7 * int(config["population"])))
        # The ECHO portfolio stores its fixed qK core first.  Ordinary
        # reference-direction survival to K is not nested with survival to qK,
        # so the first qK rows of objective-only output are not a valid core
        # identity check.
        core_usage = problem.failure_event_embedding(
            echo_x[:quality_core_size]).astype(bool)
        core_survival = ~core_usage[:, edge_indices]
        candidate_survival = ~eligible_usage[:, edge_indices]
        targeted_ceiling = best_targeted_coverage(
            core_survival, candidate_survival,
            int(config["population"]) - quality_core_size)

        classifications = []
        for event in event_rows:
            edge_index = int(event["physical_edge_index"])
            raw_survivors = int(np.count_nonzero(~raw_usage[:, edge_index]))
            eligible_survivors = int(np.count_nonzero(
                ~eligible_usage[:, edge_index]))
            record = {
                "scenario": scenario,
                "seed": seed,
                "event_rank": int(event["event_rank"]),
                "physical_edge_index": edge_index,
                "edge_source": event["edge_source"],
                "edge_target": event["edge_target"],
                "raw_pool_size": len(raw_x),
                "raw_pool_survivor_count": raw_survivors,
                "raw_pool_support": raw_survivors > 0,
                "eligible_pool_size": len(eligible_x),
                "eligible_pool_survivor_count": eligible_survivors,
                "eligible_pool_support": eligible_survivors > 0,
                "reactive_success": bool(event["reactive_success"]),
            }
            for selector in SELECTORS:
                count = int(np.count_nonzero(
                    ~portfolio_data[selector][:, edge_index]))
                record[f"{selector}_survivor_count"] = count
                record[f"{selector}_success"] = count > 0
            if not record["eligible_pool_support"]:
                classification = "generation_miss"
            elif record["echo_success"]:
                classification = "echo_success"
            else:
                classification = "selection_miss"
            record["classification"] = classification
            classifications.append(classification)
            rows.append(record)

        unit_rows.append({
            "scenario": scenario,
            "seed": seed,
            "events": len(event_rows),
            "eligible_supported_events": sum(
                row["scenario"] == scenario and row["seed"] == seed and
                row["eligible_pool_support"] for row in rows),
            "generation_misses": classifications.count("generation_miss"),
            "selection_misses": classifications.count("selection_miss"),
            "echo_successes": classifications.count("echo_success"),
            "targeted_max_events_with_frozen_core": targeted_ceiling,
        })
        sources.append({
            "scenario": scenario,
            "seed": seed,
            "source_experiment": directory,
            "source_snapshot": str(snapshot.relative_to(experiment)),
            "e7_unit": str(unit_file.relative_to(ROOT)),
            "e7_unit_sha256": sha256(unit_file),
            "portfolio_sha256": {
                selector: sha256(path)
                for selector, path in portfolio_files.items()
            },
        })
        print(f"AUDITED {scenario} seed {seed}", flush=True)

    if len(rows) != 400 or len(unit_rows) != 40:
        raise AssertionError("E7 audit row-count mismatch")
    write_csv(OUTPUT / "event_pool_support.csv", rows)
    write_csv(OUTPUT / "unit_summary.csv", unit_rows)
    data = pd.DataFrame(rows)
    summaries = []
    for scenario, group in data.groupby("scenario", sort=True):
        supported = group[group["eligible_pool_support"]]
        summary = {
            "scenario": scenario,
            "events": len(group),
            "pool_supported_events": int(group["eligible_pool_support"].sum()),
            "pool_support_rate": float(group["eligible_pool_support"].mean()),
            "generation_misses": int((group["classification"] == "generation_miss").sum()),
            "selection_misses": int((group["classification"] == "selection_miss").sum()),
            "echo_successes": int(group["echo_success"].sum()),
        }
        for selector in SELECTORS:
            summary[f"{selector}_success_all"] = float(
                group[f"{selector}_success"].mean())
            summary[f"{selector}_success_given_pool_support"] = (
                float(supported[f"{selector}_success"].mean())
                if len(supported) else float("nan"))
        summaries.append(summary)
    write_csv(OUTPUT / "scenario_summary.csv", summaries)
    atomic_json(OUTPUT / "provenance.json", {
        "schema": "echo-E7-pool-support-audit-v2",
        "interpretation": "post-hoc causal diagnostic; no new optimizer run",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "sealed_e7": str(E7.relative_to(ROOT)),
        "sealed_e7_sha256s_sha256": sha256(E7 / "SHA256SUMS"),
        "selector_analysis": str(PORTFOLIOS.parent.relative_to(ROOT)),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "sources": sources,
    })
    checksum = OUTPUT / "SHA256SUMS"
    files = sorted(path for path in OUTPUT.iterdir()
                   if path.is_file() and path != checksum)
    checksum.write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in files))


if __name__ == "__main__":
    main()
