"""Validate Moving AI warehouse maps/scenarios and write an inventory."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np

from pymoo.problems.multi.movingai_mapf import (
    load_movingai_map,
    load_movingai_scenarios,
    validate_tasks,
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    maps = Path(args.maps)
    scenarios = Path(args.scenarios)
    rows = []
    for map_path in sorted(maps.glob("warehouse-*.map")):
        grid = load_movingai_map(map_path)
        for family in ("random", "even"):
            directory = scenarios / f"scen-{family}"
            pattern = f"{map_path.stem}-{family}-*.scen"
            for scenario_path in sorted(directory.glob(pattern)):
                tasks = load_movingai_scenarios(scenario_path)
                validate_tasks(grid, tasks, map_path.name)
                lengths = np.asarray([
                    task.optimal_single_agent_length for task in tasks
                ], dtype=float)
                rows.append({
                    "map": map_path.name,
                    "family": family,
                    "scenario": scenario_path.name,
                    "tasks": len(tasks),
                    "width": grid.shape[1],
                    "height": grid.shape[0],
                    "traversable_cells": int(np.sum(grid)),
                    "mean_single_agent_length": float(np.mean(lengths)),
                    "max_single_agent_length": float(np.max(lengths)),
                    "map_sha256": sha256(map_path),
                    "scenario_sha256": sha256(scenario_path),
                })
    if not rows:
        raise RuntimeError("No warehouse scenarios found")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def parser():
    root = Path(__file__).resolve().parent
    value = argparse.ArgumentParser()
    value.add_argument("--maps", default=root / "data" / "movingai_mapf" / "maps")
    value.add_argument("--scenarios", default=root / "data" / "movingai_mapf" / "scenarios")
    value.add_argument("--output", default=root / "data" / "movingai_mapf" / "warehouse_inventory.csv")
    return value


if __name__ == "__main__":
    run(parser().parse_args())
