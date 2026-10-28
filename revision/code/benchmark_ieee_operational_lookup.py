"""Measure post-closure stored-portfolio lookup on the frozen MO-CBS event panel."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np

from echo_portfolio import _deduplicated_nondominated_pool
from echo_revision import select_strict_coverage_portfolio
from pymoo.problems.multi.movingai_mapf import (
    PrioritizedMAPFProblem, load_movingai_map, load_movingai_scenarios,
)
from pymoo.util.ref_dirs import get_reference_directions
from research_archive import atomic_json


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "research_archive/experiments/2026-10-28_ieee-operational-replanning_v1"
OUTPUT = ROOT / "research_archive/analyses/2026-10-28_ieee-operational-lookup_v1"
CONFIG = ROOT / "config/ieee_revision_operational_replan_v1.json"
PROTOCOL = ROOT / "research_archive/plans/2026-10-28_ieee-operational-lookup-protocol_v1.md"
REPEATS = 1000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if (OUTPUT / "SHA256SUMS").exists():
        print(f"ALREADY COMPLETE {OUTPUT}")
        return
    config = json.loads(CONFIG.read_text())
    grid = load_movingai_map(ROOT / config["map"])
    tasks = load_movingai_scenarios(ROOT / config["scenario"])
    pair_records = json.loads((SOURCE / "pair_records.json").read_text())
    with (SOURCE / "event_results.csv").open(newline="") as stream:
        events = list(csv.DictReader(stream))
    rows = []
    for record in pair_records:
        first, second = [int(value) for value in record["pair"][5:].split("-")]
        problem = PrioritizedMAPFProblem(
            grid, [tasks[first], tasks[second]],
            preference_levels=int(config["preference_levels"]))
        raw_x, raw_f = [], []
        for priority in ((0, 1), (1, 0)):
            for prefs in itertools.product(
                    range(int(config["preference_levels"])), repeat=2):
                x = np.asarray(priority + prefs, dtype=int)
                decoded = problem.decode(x)
                if decoded["feasible"]:
                    raw_x.append(x)
                    raw_f.append(decoded["objectives"])
        X, F, _ = _deduplicated_nondominated_pool(
            problem, np.asarray(raw_x), np.asarray(raw_f))
        if len(X) != record["eligible_unique_plans"]:
            raise AssertionError("eligible pool changed")
        ref_dirs = get_reference_directions("energy", 2, 3, seed=1)
        selected = select_strict_coverage_portfolio(
            problem, X, F, 3, quality_fraction=.7,
            random_seed=20261028, ref_dirs=ref_dirs)
        lookup = {tuple(row.tolist()): i for i, row in enumerate(X)}
        indices = [lookup[tuple(row.tolist())] for row in selected.X]
        if indices != record["echo_indices"]:
            raise AssertionError("ECHO selection changed")
        signature = np.asarray(
            problem.failure_event_embedding(X[indices]), dtype=bool)
        for event in (row for row in events if row["pair"] == record["pair"]):
            edge = int(event["edge_index"])
            timings = []
            for _ in range(REPEATS):
                started = time.perf_counter_ns()
                success = bool(np.any(~signature[:, edge]))
                timings.append(time.perf_counter_ns() - started)
            if success != (event["echo_immediate_success"] == "True"):
                raise AssertionError("lookup result differs from sealed experiment")
            rows.append({
                "pair": record["pair"], "event_class": event["event_class"],
                "edge_index": edge, "success": success,
                "lookup_median_ns": int(np.median(timings)),
                "lookup_p95_ns": int(np.quantile(timings, .95)),
            })
    if len(rows) != 100:
        raise AssertionError("expected 100 event lookups")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "lookup_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    medians = np.asarray([row["lookup_median_ns"] for row in rows])
    atomic_json(OUTPUT / "summary.json", {
        "schema": "ieee-operational-lookup-v1",
        "events": len(rows), "repeats_per_event": REPEATS,
        "median_of_event_medians_ns": int(np.median(medians)),
        "p95_of_event_medians_ns": int(np.quantile(medians, .95)),
        "scope": "precomputed Boolean edge-signature lookup only; excludes offline pool generation, portfolio composition, loading, and dispatch",
        "source_sha256s_sha256": sha256(SOURCE / "SHA256SUMS"),
        "config_sha256": sha256(CONFIG),
        "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(__file__),
        "platform": platform.platform(),
        "python": platform.python_version(),
    })
    checksum = OUTPUT / "SHA256SUMS"
    checksum.write_text("".join(
        f"{sha256(path)}  {path.name}\n"
        for path in sorted(OUTPUT.iterdir())
        if path.is_file() and path != checksum))
    print(f"SEALED {OUTPUT}")


if __name__ == "__main__":
    main()
