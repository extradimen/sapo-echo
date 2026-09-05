"""Confirmatory inference for the frozen final ECHO multi-family panel."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
INPUT = (
    ROOT / "results" / "03_multifamily_and_sensitivity" / "frozen_panel" /
    "selector_results.csv"
)
OUTPUT = (
    ROOT / "build" / "reproduced_results" / "03_multifamily_statistics"
)
METRICS = {
    "exact_group_undirected_edge_coverage": 1.0,
    "exact_group_undirected_two_edge_coverage": 1.0,
    "hv_ratio_empirical": 1.0,
    "igd_empirical": -1.0,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def rank_biserial(delta):
    delta = np.asarray(delta, dtype=float)
    delta = delta[delta != 0]
    if not len(delta):
        return 0.0
    ranks = rankdata(np.abs(delta))
    return float(
        (np.sum(ranks[delta > 0]) - np.sum(ranks[delta < 0]))
        / np.sum(ranks)
    )


def hierarchical_interval(groups, draws=20000, seed=20260904):
    labels = sorted(groups)
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = rng.choice(labels, len(labels), replace=True)
        means = []
        for label in sampled:
            delta = np.asarray(groups[label], dtype=float)
            means.append(float(np.mean(rng.choice(delta, len(delta), replace=True))))
        values[draw] = np.mean(means)
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    data = pd.read_csv(INPUT)
    data = data[data.selector.isin(["echo", "objective_only"])].copy()
    if len(data) != 640:
        raise AssertionError(f"expected 640 deterministic rows, got {len(data)}")

    paired_rows = []
    delta_table = {}
    for metric, direction in METRICS.items():
        pivot = data.pivot_table(
            index=["scenario", "family", "generator", "seed"],
            columns="selector", values=metric, aggfunc="first")
        if pivot.isna().any().any() or len(pivot) != 320:
            raise AssertionError(f"incomplete pairs for {metric}")
        pivot["improvement"] = direction * (pivot["echo"] - pivot["objective_only"])
        for (scenario, family, generator), block in pivot.groupby(
                level=["scenario", "family", "generator"]):
            delta = block["improvement"].to_numpy(dtype=float)
            pvalue = 1.0 if np.allclose(delta, 0) else float(
                wilcoxon(delta, zero_method="pratt").pvalue)
            paired_rows.append({
                "metric": metric,
                "scenario": scenario,
                "family": family,
                "generator": generator,
                "seeds": len(delta),
                "mean_improvement": float(np.mean(delta)),
                "median_improvement": float(np.median(delta)),
                "positive": int(np.count_nonzero(delta > 0)),
                "ties": int(np.count_nonzero(delta == 0)),
                "negative": int(np.count_nonzero(delta < 0)),
                "rank_biserial": rank_biserial(delta),
                "wilcoxon_p": pvalue,
            })
            delta_table[(metric, scenario, generator)] = delta

    for metric in METRICS:
        positions = [index for index, row in enumerate(paired_rows)
                     if row["metric"] == metric]
        adjusted = holm([paired_rows[index]["wilcoxon_p"] for index in positions])
        for index, value in zip(positions, adjusted):
            paired_rows[index]["holm_p_within_metric"] = float(value)
    write_csv(OUTPUT / "paired_tests.csv", paired_rows)

    panels = {
        "warehouse": ["wh10a", "wh10b", "wh20a", "wh20b"],
        "boundary": ["random32", "room32", "maze32a", "maze32b"],
        "all": ["wh10a", "wh10b", "wh20a", "wh20b",
                "random32", "room32", "maze32a", "maze32b"],
    }
    hierarchy = []
    for metric in METRICS:
        for panel, scenarios in panels.items():
            for generator in ["NSGA3-ObjectiveArchive", "NSGA2"]:
                groups = {scenario: delta_table[(metric, scenario, generator)]
                          for scenario in scenarios}
                low, high = hierarchical_interval(
                    groups, seed=20260904 + len(hierarchy))
                hierarchy.append({
                    "metric": metric,
                    "panel": panel,
                    "generator": generator,
                    "scenario_count": len(groups),
                    "seed_count_per_scenario": 20,
                    "equal_scenario_mean_improvement": float(np.mean(
                        [np.mean(value) for value in groups.values()])),
                    "hierarchical_ci_low": low,
                    "hierarchical_ci_high": high,
                })
            combined = {
                scenario: np.mean(np.row_stack([
                    delta_table[(metric, scenario, "NSGA3-ObjectiveArchive")],
                    delta_table[(metric, scenario, "NSGA2")],
                ]), axis=0)
                for scenario in scenarios
            }
            low, high = hierarchical_interval(
                combined, seed=20261004 + len(hierarchy))
            hierarchy.append({
                "metric": metric,
                "panel": panel,
                "generator": "paired-generator-average",
                "scenario_count": len(combined),
                "seed_count_per_scenario": 20,
                "equal_scenario_mean_improvement": float(np.mean(
                    [np.mean(value) for value in combined.values()])),
                "hierarchical_ci_low": low,
                "hierarchical_ci_high": high,
            })
    write_csv(OUTPUT / "hierarchical_intervals.csv", hierarchy)

    psc1 = "exact_group_undirected_edge_coverage"
    warehouse = [row for row in hierarchy if row["metric"] == psc1
                 and row["panel"] == "warehouse"]
    decision = {
        "primary_endpoint": psc1,
        "warehouse_primary_pass_by_generator": {
            row["generator"]: row["hierarchical_ci_low"] > 0
            for row in warehouse
        },
        "hv_guardrail": {
            "threshold": -0.005,
            "all_scenario_generator_mean_improvements_above_threshold": all(
                row["mean_improvement"] >= -0.005
                for row in paired_rows if row["metric"] == "hv_ratio_empirical"),
            "worst_scenario_generator_mean_improvement": min(
                row["mean_improvement"] for row in paired_rows
                if row["metric"] == "hv_ratio_empirical"),
        },
        "interpretation": (
            "Positive improvement favors ECHO for every metric; IGD signs are reversed. "
            "Scenario is the primary bootstrap unit."
        ),
    }
    (OUTPUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n")
    provenance = {
        "schema": "echo-final-multifamily-confirmatory-inference-v1",
        "input": str(INPUT),
        "input_sha256": sha256(INPUT),
        "bootstrap_draws": 20000,
        "multiple_testing": "Holm within each metric across 16 scenario-generator tests",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
    }
    (OUTPUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    checksum = OUTPUT / "SHA256SUMS"
    files = sorted(path for path in OUTPUT.iterdir()
                   if path.is_file() and path != checksum)
    checksum.write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in files))
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
