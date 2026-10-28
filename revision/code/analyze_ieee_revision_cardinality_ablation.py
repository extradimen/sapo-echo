"""Cardinality-matched archive attribution requested during peer review."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_echo_selector_replay import fast_portfolio_metrics
from analyze_echo_temporal_archive_ablation import load_pool
from analyze_movingai_mapf_experiment import empirical_quality
from echo_portfolio import (
    _deduplicated_nondominated_pool,
    select_portfolio,
)
from echo_revision import (
    fixed_event_universe,
    portfolio_coverage_metrics,
    select_strict_coverage_portfolio,
)
from movingai_mapf_experiment import make_problem
from pymoo.util.ref_dirs import get_reference_directions
from research_archive import atomic_json


ROOT = Path(__file__).resolve().parent
OUTPUT = (ROOT / "research_archive" / "analyses" /
          "2026-10-28_ieee-revision_cardinality-matched-archive_v1")
PROTOCOL = (ROOT / "research_archive" / "plans" /
            "2026-10-28_ieee-access-revision-experiment-protocol_v1.md")
EXPERIMENTS = (
    ("even8-development",
     "2026-09-03_echo-temporal-archive_even8_dev20_v1"),
    ("even12-confirmation",
     "2026-09-03_echo-temporal-archive_even12_confirm20_v1"),
    ("even13-confirmation",
     "2026-09-03_echo-temporal-archive_even13_confirm20_v1"),
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


def main():
    if (OUTPUT / "SHA256SUMS").exists():
        print(f"ALREADY COMPLETE {OUTPUT}")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    units = OUTPUT / "units"
    units.mkdir(exist_ok=True)
    portfolios = OUTPUT / "portfolios"
    portfolios.mkdir(exist_ok=True)
    rows = []
    sources = []
    for label, directory in EXPERIMENTS:
        experiment = ROOT / "research_archive" / "experiments" / directory
        config = json.loads((experiment / "config.json").read_text())
        problem = make_problem(
            config["map"], config["scenario"], int(config["agents"]),
            task_indices=config.get("task_indices"),
            preference_levels=int(config.get("preference_levels", 5)),
            independent_neighbor_orders=bool(
                config.get("independent_neighbor_orders", False)),
        )
        for seed in config["seeds"]:
            unit_file = units / f"{label}_seed{seed:04d}.json"
            if unit_file.exists():
                rows.extend(json.loads(unit_file.read_text())["rows"])
                continue
            live_algorithm = ("NSGA3" if "NSGA3" in config["algorithms"]
                              else "NSGA3-ObjectiveArchive")
            live_run = experiment / "runs" / f"{live_algorithm}_seed{seed:03d}"
            archive_run = (experiment / "runs" /
                           f"NSGA3-ObjectiveArchive_seed{seed:03d}")
            live_x, live_f, _, _, live_snapshot = load_pool(live_run, False)
            archive_x, archive_f, _, _, archive_snapshot = load_pool(
                archive_run, True)
            eligible_live_x, eligible_live_f, _ = _deduplicated_nondominated_pool(
                problem, live_x, live_f)
            eligible_archive_x, eligible_archive_f, _ = _deduplicated_nondominated_pool(
                problem, archive_x, archive_f)
            paired_k = min(int(config["population"]), len(eligible_live_x),
                           len(eligible_archive_x))
            if paired_k <= 0:
                raise RuntimeError(f"empty paired pool {label}/{seed}")
            ref_dirs = get_reference_directions(
                "energy", problem.n_obj, paired_k, seed=1)
            selected = {}
            selected["no_archive_objective"] = select_portfolio(
                problem, live_x, live_f, paired_k, selector="objective_only",
                quality_fraction=.7, random_seed=seed, ref_dirs=ref_dirs,
                pad_to_size=False)
            selected["objective_archive_objective"] = select_portfolio(
                problem, archive_x, archive_f, paired_k,
                selector="objective_only", quality_fraction=.7,
                random_seed=seed, ref_dirs=ref_dirs, pad_to_size=False)
            selected["no_archive_strict"] = select_strict_coverage_portfolio(
                problem, live_x, live_f, paired_k, quality_fraction=.7,
                random_seed=seed, ref_dirs=ref_dirs,
                use_psc2_tiebreak=True)
            selected["objective_archive_strict"] = select_strict_coverage_portfolio(
                problem, archive_x, archive_f, paired_k, quality_fraction=.7,
                random_seed=seed, ref_dirs=ref_dirs,
                use_psc2_tiebreak=True)

            # The archive pool is a superset of the live eligible pool and
            # defines one common event universe for all four cells.
            universe = fixed_event_universe(
                problem.failure_event_embedding(eligible_archive_x))
            quality = empirical_quality({key: value.F
                                         for key, value in selected.items()})
            unit_rows = []
            for cell, result in selected.items():
                identity_count = len(set(
                    problem.decision_identity(result.X).tolist()))
                if len(result.X) != paired_k or identity_count != paired_k:
                    raise AssertionError(
                        f"cardinality mismatch {label}/{seed}/{cell}")
                path = portfolios / f"{label}_seed{seed:04d}_{cell}.npz"
                np.savez_compressed(
                    path, X=result.X, F=result.F,
                    source_indices=result.source_indices)
                legacy = fast_portfolio_metrics(problem, result.X, result.F)
                global_metrics = portfolio_coverage_metrics(
                    problem.failure_event_embedding(result.X), universe)
                unit_rows.append({
                    "problem": label,
                    "seed": int(seed),
                    "cell": cell,
                    "paired_unique_cardinality": int(paired_k),
                    "live_eligible_pool_size": int(len(eligible_live_x)),
                    "archive_eligible_pool_size": int(len(eligible_archive_x)),
                    "selected_plan_count": int(len(result.X)),
                    "unique_plan_count": int(identity_count),
                    "portfolio_psc1": global_metrics["portfolio_psc1"],
                    "portfolio_psc2": global_metrics["portfolio_psc2"],
                    "legacy_group_edge_diversity": legacy[
                        "exact_group_undirected_edge_coverage"],
                    "legacy_group_two_edge_diversity": legacy[
                        "exact_group_undirected_two_edge_coverage"],
                    "hv_ratio_empirical": quality[cell]["hv_ratio_empirical"],
                    "igd_empirical": quality[cell]["igd_empirical"],
                    "portfolio_file": str(path.relative_to(ROOT)),
                })
            atomic_json(unit_file, {
                "problem": label,
                "seed": int(seed),
                "live_snapshot": str(live_snapshot.relative_to(experiment)),
                "archive_snapshot": str(archive_snapshot.relative_to(experiment)),
                "paired_unique_cardinality": int(paired_k),
                "rows": unit_rows,
            })
            rows.extend(unit_rows)
            print(f"COMPLETE {label} seed {seed}", flush=True)
        sources.append({
            "problem": label,
            "experiment": directory,
            "config_sha256": sha256(experiment / "config.json"),
            "manifest_sha256": sha256(experiment / "manifest.json"),
        })
    if len(rows) != 60 * 4:
        raise AssertionError(f"expected 240 rows, got {len(rows)}")
    write_csv(OUTPUT / "cardinality_matched_results.csv", rows)
    contrasts = []
    comparisons = (
        ("archive_content_objective", "objective_archive_objective",
         "no_archive_objective"),
        ("archive_content_strict", "objective_archive_strict",
         "no_archive_strict"),
        ("strict_composer_with_archive", "objective_archive_strict",
         "objective_archive_objective"),
    )
    for problem in sorted({row["problem"] for row in rows}):
        for name, treatment, control in comparisons:
            for metric in ("portfolio_psc1", "portfolio_psc2",
                           "legacy_group_edge_diversity",
                           "hv_ratio_empirical"):
                treatment_values = {
                    row["seed"]: float(row[metric]) for row in rows
                    if row["problem"] == problem and row["cell"] == treatment}
                control_values = {
                    row["seed"]: float(row[metric]) for row in rows
                    if row["problem"] == problem and row["cell"] == control}
                seeds = sorted(set(treatment_values) & set(control_values))
                delta = np.asarray([treatment_values[s] - control_values[s]
                                    for s in seeds])
                contrasts.append({
                    "problem": problem,
                    "contrast": name,
                    "treatment": treatment,
                    "control": control,
                    "metric": metric,
                    "paired_seeds": len(seeds),
                    "mean_difference": float(np.mean(delta)),
                    "median_difference": float(np.median(delta)),
                    "wins": int(np.count_nonzero(delta > 0)),
                    "ties": int(np.count_nonzero(delta == 0)),
                    "losses": int(np.count_nonzero(delta < 0)),
                })
    write_csv(OUTPUT / "cardinality_matched_contrasts.csv", contrasts)
    atomic_json(OUTPUT / "provenance.json", {
        "schema": "ieee-revision-cardinality-matched-archive-v1",
        "evidence_status": "post-review revision analysis",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": sha256(PROTOCOL),
        "sources": sources,
        "fixed_cardinality_rule": (
            "per seed, min(N, unique nondominated live pool, unique "
            "nondominated objective-archive pool)"),
        "common_event_universe": "objective-archive eligible pool",
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
