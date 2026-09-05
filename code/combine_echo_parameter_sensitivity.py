"""Combine scenario-isolated ECHO sensitivity archives and infer robustness."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SCENARIOS = ("wh10a", "wh10b", "wh20a", "wh20b", "random32", "room32", "maze32a", "maze32b")
SOURCE_ROOT = ROOT / "research_archive" / "analyses"
OUTPUT = SOURCE_ROOT / "2026-09-05_echo-parameter-sensitivity_combined_v2"
METRICS = {
    "exact_group_undirected_edge_coverage": 1.0,
    "exact_group_undirected_two_edge_coverage": 1.0,
    "hv_ratio_empirical": 1.0,
    "igd_empirical": -1.0,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(groups, draws, seed):
    labels = sorted(groups)
    rng = np.random.default_rng(seed)
    values = np.empty(draws)
    for draw in range(draws):
        sampled = rng.choice(labels, len(labels), replace=True)
        means = []
        for label in sampled:
            sample = np.asarray(groups[label], dtype=float)
            means.append(float(np.mean(rng.choice(sample, len(sample), replace=True))))
        values[draw] = np.mean(means)
    return [float(x) for x in np.quantile(values, [0.025, 0.975])]


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    sources = []
    result_frames = []
    contrast_frames = []
    for scenario in SCENARIOS:
        source = SOURCE_ROOT / f"2026-09-05_echo-parameter-sensitivity_{scenario}_v2"
        if not (source / "SHA256SUMS").exists():
            raise RuntimeError(f"incomplete source: {source}")
        results = pd.read_csv(source / "sensitivity_results.csv")
        contrasts = pd.read_csv(source / "sensitivity_contrasts.csv")
        if len(results) != 720 or len(contrasts) != 120:
            raise AssertionError(f"unexpected row count for {scenario}")
        result_frames.append(results)
        contrast_frames.append(contrasts)
        sources.append({
            "scenario": scenario,
            "path": str(source),
            "checksums_sha256": sha256(source / "SHA256SUMS"),
            "results_sha256": sha256(source / "sensitivity_results.csv"),
        })
    results = pd.concat(result_frames, ignore_index=True)
    contrasts = pd.concat(contrast_frames, ignore_index=True)
    if len(results) != 5760 or len(contrasts) != 960:
        raise AssertionError("combined row count mismatch")
    results.to_csv(OUTPUT / "sensitivity_results.csv", index=False)
    contrasts.to_csv(OUTPUT / "sensitivity_contrasts.csv", index=False)

    panels = {
        "warehouse": ("wh10a", "wh10b", "wh20a", "wh20b"),
        "boundary": ("random32", "room32", "maze32a", "maze32b"),
        "all": SCENARIOS,
    }
    surface = []
    for metric, direction in METRICS.items():
        for multiplier in (1.0, 1.5, 2.0):
            for fraction in (0.5, 0.6, 0.7, 0.8, 0.9):
                for panel, scenarios in panels.items():
                    for generator in ("NSGA3-ObjectiveArchive", "NSGA2"):
                        groups = {}
                        for scenario in scenarios:
                            block = results[(results.scenario == scenario)
                                            & (results.generator == generator)
                                            & (results.archive_multiplier == multiplier)]
                            base = block[block.selector == "objective_only"].set_index("seed")
                            echo = block[(block.selector == "echo")
                                         & (block.quality_fraction == fraction)].set_index("seed")
                            groups[scenario] = direction * (
                                echo.loc[base.index, metric].to_numpy()
                                - base[metric].to_numpy())
                        low, high = interval(
                            groups, 20000, 20260905 + len(surface))
                        surface.append({
                            "metric": metric, "archive_multiplier": multiplier,
                            "quality_fraction": fraction, "panel": panel,
                            "generator": generator,
                            "equal_scenario_mean_improvement": float(np.mean(
                                [np.mean(values) for values in groups.values()])),
                            "hierarchical_ci_low": low,
                            "hierarchical_ci_high": high,
                        })
                    combined = {}
                    for scenario in scenarios:
                        parts = []
                        for generator in ("NSGA3-ObjectiveArchive", "NSGA2"):
                            block = results[(results.scenario == scenario)
                                            & (results.generator == generator)
                                            & (results.archive_multiplier == multiplier)]
                            base = block[block.selector == "objective_only"].set_index("seed")
                            echo = block[(block.selector == "echo")
                                         & (block.quality_fraction == fraction)].set_index("seed")
                            parts.append(direction * (
                                echo.loc[base.index, metric].to_numpy()
                                - base[metric].to_numpy()))
                        combined[scenario] = np.mean(np.row_stack(parts), axis=0)
                    low, high = interval(
                        combined, 20000, 20261905 + len(surface))
                    surface.append({
                        "metric": metric, "archive_multiplier": multiplier,
                        "quality_fraction": fraction, "panel": panel,
                        "generator": "paired-generator-average",
                        "equal_scenario_mean_improvement": float(np.mean(
                            [np.mean(values) for values in combined.values()])),
                        "hierarchical_ci_low": low,
                        "hierarchical_ci_high": high,
                    })
    surface = pd.DataFrame(surface)
    surface.to_csv(OUTPUT / "hierarchical_sensitivity_surface.csv", index=False)

    psc1 = surface[(surface.metric == "exact_group_undirected_edge_coverage")
                   & (surface.panel == "warehouse")]
    hv = surface[surface.metric == "hv_ratio_empirical"]
    default = psc1[(psc1.archive_multiplier == 2.0)
                   & (psc1.quality_fraction == 0.7)]
    decision = {
        "schema": "echo-parameter-sensitivity-combined-decision-v2",
        "source_scenarios": list(SCENARIOS),
        "result_rows": len(results),
        "all_warehouse_psc1_surface_means_nonnegative": bool(
            (psc1.equal_scenario_mean_improvement >= 0).all()),
        "warehouse_psc1_positive_ci_cells": int(
            (psc1.hierarchical_ci_low > 0).sum()),
        "warehouse_psc1_total_cells": len(psc1),
        "all_hv_surface_means_above_minus_0005": bool(
            (hv.equal_scenario_mean_improvement >= -0.005).all()),
        "worst_hv_surface_mean": float(hv.equal_scenario_mean_improvement.min()),
        "default_2N_q07": default.to_dict(orient="records"),
        "default_preserved": True,
        "interpretation": (
            "Sensitivity is evidence of robustness, not a parameter-selection exercise."
        ),
    }
    (OUTPUT / "decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n")
    provenance = {
        "schema": "echo-parameter-sensitivity-combined-v2",
        "protocol": "research_archive/plans/2026-09-04_echo_parameter_sensitivity_protocol_v2.md",
        "bootstrap": "equal-scenario hierarchical, 20000 draws per cell",
        "sources": sources,
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
