"""Replay the corrected selector on the already frozen E7/E7b events.

No search is rerun.  The script uses the strict portfolios from the revision
consistency replay and the exact critical events/resource classes frozen before
this revision, so the comparison remains candidate-pool matched.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_echo_final_multifamily import EXPERIMENTS, RUN_GROUPS
from movingai_mapf_experiment import make_problem
from research_archive import atomic_json


ROOT = Path(__file__).resolve().parent
STRICT_SOURCE = (ROOT / "research_archive" / "analyses" /
                 "2026-10-28_ieee-revision_portfolio-consistency_v1")
E7 = (ROOT / "research_archive" / "experiments" /
      "2026-09-05_echo-equal-budget-replanning_E7_v2")
E7B = (ROOT / "research_archive" / "analyses" /
       "2026-09-05_echo-E7b-supported-critical-resources_v1")
OUTPUT = (ROOT / "research_archive" / "analyses" /
          "2026-10-28_ieee-revision_critical-edge-replay_v1")
PROTOCOL = (ROOT / "research_archive" / "plans" /
            "2026-10-28_ieee-access-revision-experiment-protocol_v1.md")
STRICT = "strict_coverage_psc1_psc2_tie"


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


def source_group(scenario: str, seed: int):
    matches = [group for group in RUN_GROUPS
               if group[0] == scenario and group[2] <= seed <= group[3]]
    if len(matches) != 1:
        raise RuntimeError(f"source mapping mismatch: {scenario}/{seed}")
    return matches[0]


def problem_for(scenario: str, seed: int):
    _, directory, _, _ = source_group(scenario, seed)
    config = json.loads((EXPERIMENTS / directory / "config.json").read_text())
    return make_problem(
        config["map"], config["scenario"], int(config["agents"]),
        task_indices=config.get("task_indices"),
        preference_levels=int(config.get("preference_levels", 5)),
        independent_neighbor_orders=bool(
            config.get("independent_neighbor_orders", False)),
    )


def strict_x(scenario: str, generator: str, seed: int) -> np.ndarray:
    path = STRICT_SOURCE / "portfolios" / (
        f"{scenario}_{generator}_seed{seed:03d}_{STRICT}.npz")
    with np.load(path, allow_pickle=False) as saved:
        return np.asarray(saved["X"], dtype=int)


def score_usage(usage: np.ndarray) -> dict:
    survivor_counts = np.count_nonzero(~usage, axis=0)
    pair_success = [bool(np.any(~usage[:, i] & ~usage[:, j]))
                    for i in range(usage.shape[1])
                    for j in range(i + 1, usage.shape[1])]
    return {
        "class_success_rate": float(np.mean(survivor_counts > 0)),
        "mean_survivor_count": float(np.mean(survivor_counts)),
        "minimum_survivor_count": int(np.min(survivor_counts)),
        "two_class_success_rate": (float(np.mean(pair_success))
                                   if pair_success else float("nan")),
    }


def main() -> None:
    if (OUTPUT / "SHA256SUMS").exists():
        print(f"ALREADY COMPLETE {OUTPUT}")
        return
    if not (STRICT_SOURCE / "SHA256SUMS").exists():
        raise RuntimeError("strict consistency source is not sealed")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # E7: the originally frozen highest-exposure physical edges.
    e7_rows = []
    for unit_file in sorted((E7 / "units").glob("*.json")):
        payload = json.loads(unit_file.read_text())
        scenario = payload["provenance"]["scenario"]
        seed = int(payload["provenance"]["seed"])
        problem = problem_for(scenario, seed)
        X = strict_x(scenario, "NSGA3-ObjectiveArchive", seed)
        usage = np.asarray(problem.failure_event_embedding(X), dtype=bool)
        for event in payload["rows"]:
            edge = int(event["physical_edge_index"])
            survivors = int(np.count_nonzero(~usage[:, edge]))
            e7_rows.append({
                "scenario": scenario, "seed": seed,
                "event_rank": int(event["event_rank"]),
                "physical_edge_index": edge,
                "selector": STRICT,
                "survivor_count": survivors,
                "success": survivors > 0,
                "reactive_replanning_success": bool(event["reactive_success"]),
            })
    write_csv(OUTPUT / "e7_strict_event_results.csv", e7_rows)

    # E7b: ten frozen actionable high-exposure resource classes per pool.
    old_rows = pd.read_csv(E7B / "selector_unit_results.csv").to_dict("records")
    strict_rows = []
    for unit_file in sorted((E7B / "units").glob("*.json")):
        payload = json.loads(unit_file.read_text())
        meta = payload["metadata"]
        scenario = str(meta["scenario"])
        seed = int(meta["seed"])
        generator = str(meta["generator"])
        representatives = [int(item["representative_edge_index"])
                           for item in payload["classes"]]
        if not representatives:
            continue
        problem = problem_for(scenario, seed)
        X = strict_x(scenario, generator, seed)
        usage = np.asarray(problem.failure_event_embedding(X), dtype=bool)
        strict_rows.append({
            "scenario": scenario,
            "family": ("warehouse" if scenario.startswith("wh") else
                       "maze" if scenario.startswith("maze") else
                       scenario.rstrip("32")),
            "seed": seed, "generator": generator, "selector": STRICT,
            "selector_repeat": 0,
            "eligible_pool_size": int(meta["eligible_pool_size"]),
            "total_actionable_classes": int(meta["total_actionable_classes"]),
            "selected_resource_classes": len(representatives),
            **score_usage(usage[:, representatives]),
        })
    combined = old_rows + strict_rows
    write_csv(OUTPUT / "e7b_selector_results_with_strict.csv", combined)
    frame = pd.DataFrame(combined)
    summary = frame.groupby(
        ["scenario", "generator", "selector"], as_index=False
    ).agg(
        seeds=("seed", "nunique"),
        class_success_rate=("class_success_rate", "mean"),
        mean_survivor_count=("mean_survivor_count", "mean"),
        minimum_survivor_count=("minimum_survivor_count", "mean"),
        two_class_success_rate=("two_class_success_rate", "mean"),
    )
    summary.to_csv(OUTPUT / "e7b_scenario_summary.csv", index=False)
    atomic_json(OUTPUT / "summary.json", {
        "schema": "ieee-revision-critical-edge-replay-summary-v1",
        "e7_events": len(e7_rows),
        "e7_success_rate": float(np.mean([row["success"] for row in e7_rows])),
        "e7_mean_survivor_count": float(np.mean(
            [row["survivor_count"] for row in e7_rows])),
        "e7b_strict_units": len(strict_rows),
    })
    atomic_json(OUTPUT / "provenance.json", {
        "schema": "ieee-revision-critical-edge-replay-v1",
        "evidence_status": "post-review replay; no new search",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "strict_source": str(STRICT_SOURCE.relative_to(ROOT)),
        "strict_source_checksums_sha256": sha256(STRICT_SOURCE / "SHA256SUMS"),
        "e7_source": str(E7.relative_to(ROOT)),
        "e7_source_checksums_sha256": sha256(E7 / "SHA256SUMS"),
        "e7b_source": str(E7B.relative_to(ROOT)),
        "e7b_source_checksums_sha256": sha256(E7B / "SHA256SUMS"),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
    })
    checksum = OUTPUT / "SHA256SUMS"
    files = sorted(path for path in OUTPUT.rglob("*")
                   if path.is_file() and path != checksum)
    checksum.write_text("".join(
        f"{sha256(path)}  {path.relative_to(OUTPUT)}\n" for path in files))
    print(f"COMPLETE {OUTPUT}")


if __name__ == "__main__":
    main()
