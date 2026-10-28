"""Paired inference for the post-review portfolio-consistency replay."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from research_archive import atomic_json


ROOT = Path(__file__).resolve().parent
SOURCE = (ROOT / "research_archive" / "analyses" /
          "2026-10-28_ieee-revision_portfolio-consistency_v1")
OUTPUT = (ROOT / "research_archive" / "analyses" /
          "2026-10-28_ieee-revision_portfolio-statistics_v1")
PRIMARY = "strict_coverage_psc1_psc2_tie"
COMPARATORS = (
    "objective_only",
    "legacy_group_heuristic",
    "edge_jaccard",
    "minimum_common_edge",
    "xu2025_official_edge",
)
METRICS = ("portfolio_psc1", "portfolio_psc2", "hv_ratio_empirical",
           "unique_plan_count", "search_plus_selection_seconds")
PANELS = {
    "warehouse": ("wh10a", "wh10b", "wh20a", "wh20b"),
    "boundary": ("random32", "room32", "maze32a", "maze32b"),
    "all": ("wh10a", "wh10b", "wh20a", "wh20b",
            "random32", "room32", "maze32a", "maze32b"),
}


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


def hierarchical_interval(groups, draws=20000, seed=20261028):
    labels = sorted(groups)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    for draw in range(draws):
        chosen = rng.choice(labels, size=len(labels), replace=True)
        scenario_means = []
        for label in chosen:
            values = np.asarray(groups[label], dtype=float)
            scenario_means.append(float(np.mean(
                rng.choice(values, size=len(values), replace=True))))
        samples[draw] = float(np.mean(scenario_means))
    return (float(np.mean([np.mean(groups[label]) for label in labels])),
            float(np.quantile(samples, .025)),
            float(np.quantile(samples, .975)))


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    total = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (total - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def main():
    if (OUTPUT / "SHA256SUMS").exists():
        print(f"ALREADY COMPLETE {OUTPUT}")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE / "selector_results.csv")
    required = {PRIMARY, *COMPARATORS}
    if not required <= set(data.selector):
        raise RuntimeError("required selectors missing")

    # Collapse the two candidate generators inside each scenario/seed before
    # primary inference; generator-specific results are retained separately.
    collapsed = data.groupby(
        ["scenario", "family", "seed", "selector"], as_index=False
    )[list(METRICS)].mean()

    effects = []
    scenario_tests = []
    for metric in METRICS:
        pivot = collapsed.pivot_table(
            index=["scenario", "seed"], columns="selector", values=metric)
        for comparator in COMPARATORS:
            delta = (pivot[PRIMARY] - pivot[comparator]).dropna()
            grouped = {
                scenario: block.to_numpy(dtype=float)
                for scenario, block in delta.groupby(level="scenario")
            }
            for panel, scenarios in PANELS.items():
                panel_groups = {key: value for key, value in grouped.items()
                                if key in scenarios}
                mean, low, high = hierarchical_interval(
                    panel_groups, seed=20261028 + len(effects))
                effects.append({
                    "metric": metric,
                    "primary": PRIMARY,
                    "comparator": comparator,
                    "panel": panel,
                    "scenario_count": len(panel_groups),
                    "paired_seed_units": int(sum(len(v) for v in panel_groups.values())),
                    "equal_scenario_mean_difference": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "interval_strictly_positive": bool(low > 0),
                    "interval_strictly_negative": bool(high < 0),
                })
            raw_tests = []
            for scenario, values in grouped.items():
                pvalue = (1.0 if np.allclose(values, 0) else
                          float(wilcoxon(values, zero_method="pratt").pvalue))
                raw_tests.append((scenario, values, pvalue))
            adjusted = holm([item[2] for item in raw_tests])
            for (scenario, values, pvalue), corrected in zip(raw_tests, adjusted):
                scenario_tests.append({
                    "metric": metric,
                    "primary": PRIMARY,
                    "comparator": comparator,
                    "scenario": scenario,
                    "paired_seeds": len(values),
                    "mean_difference": float(np.mean(values)),
                    "median_difference": float(np.median(values)),
                    "wins": int(np.count_nonzero(values > 0)),
                    "ties": int(np.count_nonzero(values == 0)),
                    "losses": int(np.count_nonzero(values < 0)),
                    "wilcoxon_p": pvalue,
                    "holm_p_within_contrast_metric": float(corrected),
                })
    write_csv(OUTPUT / "hierarchical_effects.csv", effects)
    write_csv(OUTPUT / "scenario_paired_tests.csv", scenario_tests)

    saturation = []
    for selector in sorted(required):
        subset = data[data.selector == selector]
        for metric in ("portfolio_psc1", "portfolio_psc2"):
            values = subset[metric].to_numpy(dtype=float)
            saturation.append({
                "selector": selector,
                "metric": metric,
                "rows": len(values),
                "mean": float(np.mean(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "fraction_equal_one": float(np.mean(np.isclose(values, 1.0))),
            })
    write_csv(OUTPUT / "saturation_audit.csv", saturation)

    runtime_rows = []
    for generator in sorted(data.generator.unique()):
        for selector in sorted(required):
            subset = data[(data.generator == generator) &
                          (data.selector == selector)]
            search = subset.search_seconds.to_numpy(dtype=float)
            composer = subset.selection_seconds.to_numpy(dtype=float)
            total = subset.search_plus_selection_seconds.to_numpy(dtype=float)
            runtime_rows.append({
                "generator": generator,
                "selector": selector,
                "runs": len(subset),
                "search_seconds_mean": float(np.mean(search)),
                "search_seconds_median": float(np.median(search)),
                "composer_seconds_mean": float(np.mean(composer)),
                "composer_seconds_median": float(np.median(composer)),
                "composer_seconds_q05": float(np.quantile(composer, .05)),
                "composer_seconds_q95": float(np.quantile(composer, .95)),
                "end_to_end_seconds_mean": float(np.mean(total)),
                "composer_over_search_percent_mean": float(np.mean(
                    100.0 * composer / np.maximum(search, 1e-12))),
            })
    write_csv(OUTPUT / "runtime_audit.csv", runtime_rows)

    def sysctl(name):
        try:
            return subprocess.check_output(
                ["sysctl", "-n", name], text=True).strip()
        except Exception:
            return "unavailable"

    def effect(metric, comparator, panel):
        matches = [row for row in effects if row["metric"] == metric
                   and row["comparator"] == comparator and row["panel"] == panel]
        if len(matches) != 1:
            raise AssertionError((metric, comparator, panel))
        return matches[0]

    primary_objective = effect("portfolio_psc1", "objective_only", "all")
    primary_jaccard = effect("portfolio_psc1", "edge_jaccard", "all")
    decision = {
        "schema": "ieee-revision-portfolio-statistics-decision-v1",
        "evidence_status": "post-review revision analysis",
        "primary_selector": PRIMARY,
        "portfolio_psc1_vs_objective_all": primary_objective,
        "portfolio_psc1_vs_edge_jaccard_all": primary_jaccard,
        "corrected_primary_advantage_supported": bool(
            primary_objective["ci95_low"] > 0 and
            primary_jaccard["ci95_low"] > 0),
        "interpretation_rule": (
            "If false, withdraw universal corrected-PSC superiority; retain "
            "only effects whose scenario-level evidence supports them."),
    }
    atomic_json(OUTPUT / "decision.json", decision)
    atomic_json(OUTPUT / "provenance.json", {
        "schema": "ieee-revision-portfolio-statistics-v1",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_checksums_sha256": sha256(SOURCE / "SHA256SUMS"),
        "bootstrap_draws": 20000,
        "primary_replication_unit": "scenario",
        "seed_pairing": True,
        "generator_handling": "mean within scenario-seed before inference",
        "runtime_environment": {
            "platform": platform.platform(),
            "processor": sysctl("machdep.cpu.brand_string"),
            "logical_cpu_count": sysctl("hw.ncpu"),
            "memory_bytes": sysctl("hw.memsize"),
            "runtime_scope": (
                "archived search wall time plus corrected offline composer; "
                "excludes exhaustive failure-injection analysis"),
        },
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
