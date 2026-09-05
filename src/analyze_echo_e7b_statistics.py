"""Exploratory inference for the frozen E7b supported-resource replay.

E7b is a post-hoc diagnostic.  It asks how frozen selectors behave when a
closed resource is demonstrably represented by both using and avoiding
candidates in the quality-eligible nondominated pool.  Scenario is the
primary bootstrap unit and seed is the secondary unit.  Random selection has
already been averaged within each seed by the replay script.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = (
    ROOT / "results" / "04_supported_resource_stress" / "replay"
)
INPUT = INPUT_DIR / "selector_unit_collapsed.csv"
OUTPUT = (
    ROOT / "build" / "reproduced_results" / "04_supported_resource_statistics"
)
DRAW_COUNT = 20_000
BASE_SEED = 20260905
METRICS = (
    "class_success_rate",
    "mean_survivor_count",
    "minimum_survivor_count",
    "two_class_success_rate",
)
PANELS = {
    "warehouse": ("wh10a", "wh10b", "wh20a", "wh20b"),
    "boundary": ("random32", "room32", "maze32a", "maze32b"),
    "all": (
        "wh10a", "wh10b", "wh20a", "wh20b",
        "random32", "room32", "maze32a", "maze32b",
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def holm(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=float)
    order = np.argsort(values_array)
    adjusted = np.empty(len(values_array), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values_array) - rank) * values_array[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def rank_biserial(delta: np.ndarray) -> float:
    nonzero = np.asarray(delta, dtype=float)
    nonzero = nonzero[~np.isclose(nonzero, 0.0)]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    return float(
        (np.sum(ranks[nonzero > 0]) - np.sum(ranks[nonzero < 0]))
        / np.sum(ranks)
    )


def hierarchical_interval(
        groups: dict[str, np.ndarray], *, draws: int, seed: int,
) -> tuple[float, float]:
    """Equal-scenario, within-scenario seed bootstrap, vectorized by stratum."""
    labels = sorted(groups)
    if not labels:
        return float("nan"), float("nan")
    arrays = [np.asarray(groups[label], dtype=float) for label in labels]
    if any(not len(array) for array in arrays):
        raise ValueError("empty scenario stratum")
    rng = np.random.default_rng(seed)
    stratum_count = len(arrays)
    selected = rng.integers(0, stratum_count, size=(draws, stratum_count))
    bootstrap_means = np.empty((stratum_count, draws, stratum_count))
    for source_index, array in enumerate(arrays):
        sample_indices = rng.integers(
            0, len(array), size=(draws, stratum_count, len(array)))
        bootstrap_means[source_index] = array[sample_indices].mean(axis=2)
    draw_index = np.arange(draws)[:, None]
    position = np.arange(stratum_count)[None, :]
    sampled_means = bootstrap_means[selected, draw_index, position]
    values = sampled_means.mean(axis=1)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(INPUT)
    unit_keys = ["scenario", "family", "generator", "seed"]
    if data[unit_keys].drop_duplicates().shape[0] != 320:
        raise AssertionError("E7b input must contain exactly 320 source units")
    selectors = sorted(data.selector.unique())
    if "echo" not in selectors:
        raise AssertionError("ECHO selector is absent")
    competitors = [selector for selector in selectors if selector != "echo"]

    paired_rows: list[dict] = []
    delta_table: dict[tuple[str, str, str, str], dict[int, float]] = {}
    for metric in METRICS:
        pivot = data.pivot_table(
            index=unit_keys, columns="selector", values=metric, aggfunc="first")
        for comparator in competitors:
            paired = pivot[["echo", comparator]].dropna().copy()
            paired["improvement"] = paired["echo"] - paired[comparator]
            for (scenario, family, generator), block in paired.groupby(
                    level=["scenario", "family", "generator"]):
                delta = block.improvement.to_numpy(dtype=float)
                pvalue = 1.0 if np.allclose(delta, 0.0) else float(
                    wilcoxon(delta, zero_method="pratt").pvalue)
                paired_rows.append({
                    "metric": metric,
                    "comparator": comparator,
                    "scenario": scenario,
                    "family": family,
                    "generator": generator,
                    "paired_seeds": len(delta),
                    "mean_improvement": float(np.mean(delta)),
                    "median_improvement": float(np.median(delta)),
                    "positive": int(np.count_nonzero(delta > 1e-12)),
                    "ties": int(np.count_nonzero(np.isclose(delta, 0.0))),
                    "negative": int(np.count_nonzero(delta < -1e-12)),
                    "rank_biserial": rank_biserial(delta),
                    "wilcoxon_p": pvalue,
                })
                delta_table[(metric, comparator, scenario, generator)] = {
                    int(index[-1]): float(value)
                    for index, value in block.improvement.items()
                }

    # Adjust the 16 scenario-generator tests separately for each endpoint and
    # comparator.  This is descriptive support for a post-hoc analysis, not a
    # conversion of E7b into a confirmatory experiment.
    for metric in METRICS:
        for comparator in competitors:
            positions = [
                index for index, row in enumerate(paired_rows)
                if row["metric"] == metric and row["comparator"] == comparator
            ]
            adjusted = holm([paired_rows[index]["wilcoxon_p"] for index in positions])
            for index, value in zip(positions, adjusted):
                paired_rows[index]["holm_p_within_metric_comparator"] = float(value)
    write_csv(OUTPUT / "paired_tests.csv", paired_rows)

    hierarchy_rows: list[dict] = []
    hierarchy_seed = BASE_SEED
    for metric in METRICS:
        for comparator in competitors:
            for panel, requested_scenarios in PANELS.items():
                for generator in ("NSGA3-ObjectiveArchive", "NSGA2"):
                    groups = {}
                    for scenario in requested_scenarios:
                        values = delta_table.get((metric, comparator, scenario, generator), {})
                        if values:
                            groups[scenario] = np.asarray(list(values.values()), dtype=float)
                    low, high = hierarchical_interval(
                        groups, draws=DRAW_COUNT, seed=hierarchy_seed)
                    hierarchy_seed += 1
                    hierarchy_rows.append({
                        "metric": metric,
                        "comparator": comparator,
                        "panel": panel,
                        "generator": generator,
                        "scenario_count": len(groups),
                        "paired_seed_count": int(sum(len(value) for value in groups.values())),
                        "min_seeds_per_scenario": min(map(len, groups.values())),
                        "max_seeds_per_scenario": max(map(len, groups.values())),
                        "equal_scenario_mean_improvement": float(np.mean([
                            np.mean(value) for value in groups.values()])),
                        "hierarchical_ci_low": low,
                        "hierarchical_ci_high": high,
                    })

                combined_groups = {}
                for scenario in requested_scenarios:
                    first = delta_table.get(
                        (metric, comparator, scenario, "NSGA3-ObjectiveArchive"), {})
                    second = delta_table.get(
                        (metric, comparator, scenario, "NSGA2"), {})
                    common_seeds = sorted(set(first) & set(second))
                    if common_seeds:
                        combined_groups[scenario] = np.asarray([
                            (first[seed] + second[seed]) / 2.0
                            for seed in common_seeds
                        ], dtype=float)
                low, high = hierarchical_interval(
                    combined_groups, draws=DRAW_COUNT, seed=hierarchy_seed)
                hierarchy_seed += 1
                hierarchy_rows.append({
                    "metric": metric,
                    "comparator": comparator,
                    "panel": panel,
                    "generator": "paired-generator-average",
                    "scenario_count": len(combined_groups),
                    "paired_seed_count": int(sum(
                        len(value) for value in combined_groups.values())),
                    "min_seeds_per_scenario": min(map(len, combined_groups.values())),
                    "max_seeds_per_scenario": max(map(len, combined_groups.values())),
                    "equal_scenario_mean_improvement": float(np.mean([
                        np.mean(value) for value in combined_groups.values()])),
                    "hierarchical_ci_low": low,
                    "hierarchical_ci_high": high,
                })
    write_csv(OUTPUT / "hierarchical_intervals.csv", hierarchy_rows)

    all_combined = [
        row for row in hierarchy_rows
        if row["panel"] == "all" and row["generator"] == "paired-generator-average"
    ]
    compact = sorted(
        all_combined,
        key=lambda row: (row["metric"], row["comparator"]),
    )
    write_csv(OUTPUT / "all_panel_summary.csv", compact)

    success_rows = [row for row in compact if row["metric"] == "class_success_rate"]
    strongest = min(success_rows, key=lambda row: row["equal_scenario_mean_improvement"])
    decision = {
        "schema": "echo-e7b-supported-resource-exploratory-decision-v1",
        "status": "post_hoc_exploratory",
        "source_units": 320,
        "source_units_with_actionable_resource_classes": int(
            data.loc[data.total_actionable_classes > 0, unit_keys]
            .drop_duplicates().shape[0]),
        "bootstrap_draws": DRAW_COUNT,
        "primary_bootstrap_unit": "scenario",
        "secondary_bootstrap_unit": "seed",
        "random_replicates": "collapsed within seed by the E7b replay",
        "strongest_class_success_comparator_by_observed_mean": {
            "selector": strongest["comparator"],
            "echo_minus_comparator": strongest["equal_scenario_mean_improvement"],
            "ci_low": strongest["hierarchical_ci_low"],
            "ci_high": strongest["hierarchical_ci_high"],
        },
        "interpretation_rule": (
            "Positive differences favor ECHO. Because E7b was motivated by the E7 "
            "diagnosis, intervals quantify an exploratory mechanism check and must "
            "not be described as preregistered confirmation. Ties are retained."
        ),
    }
    (OUTPUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n")
    provenance = {
        "schema": "echo-e7b-supported-resource-exploratory-statistics-v1",
        "input": str(INPUT.relative_to(ROOT)),
        "input_sha256": sha256(INPUT),
        "input_scope": "curated public replay table",
        "protocol": (
            "protocols/04_supported_resource_stress/"
            "supported_resource_stress.md"
        ),
        "bootstrap_draws": DRAW_COUNT,
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
    }
    (OUTPUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    checksum = OUTPUT / "SHA256SUMS"
    files = sorted(path for path in OUTPUT.iterdir() if path.is_file())
    checksum.write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in files))
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
