"""Frozen identical-pool selector analysis for the final ECHO MAPF panel."""

from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from analyze_echo_selector_replay import fast_portfolio_metrics
from analyze_movingai_mapf_experiment import empirical_quality
from echo_portfolio import SELECTORS, select_portfolio
from movingai_mapf_experiment import make_problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions


ROOT = Path(__file__).resolve().parent
EXPERIMENTS = ROOT / "research_archive" / "experiments"
OUTPUT = (
    ROOT / "research_archive" / "analyses" /
    "2026-09-04_echo-final-multifamily_selector-confirmation_v1"
)
SELECTOR_NAMES = tuple(SELECTORS)
RANDOM_REPLICATES = 20
QUALITY_FRACTION = 0.7

# scenario, experiment directory, inclusive first seed, inclusive last seed
RUN_GROUPS = (
    ("wh10a", "2026-09-03_echo-final-multifamily_wh10a_even14_seed1401-1420_v1", 1401, 1420),
    ("wh10b", "2026-09-03_echo-final-multifamily_wh10b_even14_seed1401-1420_v1", 1401, 1420),
    ("wh20a", "2026-09-03_echo-final-multifamily_wh20a_even14_seed1401-1420_v1", 1401, 1402),
    ("wh20a", "2026-09-03_echo-final-multifamily_wh20a_even14_seed1403-1420_recovery_v1", 1403, 1420),
    ("wh20b", "2026-09-04_echo-final-multifamily_wh20b_even14_seed1401-1410_recovery_v2", 1401, 1410),
    ("wh20b", "2026-09-04_echo-final-multifamily_wh20b_even14_seed1411-1420_recovery_v2", 1411, 1420),
    ("random32", "2026-09-04_echo-final-multifamily_random32_even14_seed1401-1410_chunk_v2", 1401, 1410),
    ("random32", "2026-09-04_echo-final-multifamily_random32_even14_seed1411-1420_chunk_v2", 1411, 1420),
    ("room32", "2026-09-04_echo-final-multifamily_room32_even14_seed1401-1410_chunk_v1", 1401, 1410),
    ("room32", "2026-09-04_echo-final-multifamily_room32_even14_seed1411-1420_chunk_v1", 1411, 1420),
    ("maze32a", "2026-09-04_echo-final-multifamily_maze32a_even14_seed1401-1410_chunk_v1", 1401, 1410),
    ("maze32a", "2026-09-04_echo-final-multifamily_maze32a_even14_seed1411-1420_chunk_v1", 1411, 1420),
    ("maze32b", "2026-09-04_echo-final-multifamily_maze32b_even14_seed1401-1410_chunk_v1", 1401, 1410),
    ("maze32b", "2026-09-04_echo-final-multifamily_maze32b_even14_seed1411-1420_chunk_v1", 1411, 1420),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
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


def snapshots(run_path: Path) -> list[Path]:
    result = sorted((run_path / "trajectory").glob("generation_*.npz"))
    if not result:
        raise FileNotFoundError(f"no trajectory snapshots: {run_path}")
    return result


def load_live(snapshot: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(snapshot, allow_pickle=False) as saved:
        return (
            np.asarray(saved["X"], dtype=int),
            np.asarray(saved["F"], dtype=float),
        )


def objective_crowding_keep(F: np.ndarray, capacity: int) -> np.ndarray:
    """Deterministic reference-independent NSGA-II-style truncation."""
    F = np.asarray(F, dtype=float)
    if len(F) <= capacity:
        return np.arange(len(F), dtype=int)
    distance = np.zeros(len(F), dtype=float)
    for objective in range(F.shape[1]):
        order = np.lexsort((np.arange(len(F)), F[:, objective]))
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        scale = F[order[-1], objective] - F[order[0], objective]
        if scale > 0 and len(order) > 2:
            distance[order[1:-1]] += (
                F[order[2:], objective] - F[order[:-2], objective]
            ) / scale
    chosen = sorted(
        range(len(F)),
        key=lambda index: (
            0 if np.isinf(distance[index]) else 1,
            -distance[index] if np.isfinite(distance[index]) else 0.0,
            *F[index].tolist(),
            index,
        ),
    )[:capacity]
    return np.asarray(chosen, dtype=int)


def update_nsga2_archive(problem, archive_x, archive_f, X, F, capacity):
    if archive_x is None:
        pool_x, pool_f = np.asarray(X), np.asarray(F)
    else:
        pool_x = np.row_stack([X, archive_x])
        pool_f = np.row_stack([F, archive_f])
    identities = np.asarray(problem.decision_identity(pool_x), dtype=str)
    _, first = np.unique(identities, return_index=True)
    first = np.sort(first)
    pool_x, pool_f = pool_x[first], pool_f[first]
    front = NonDominatedSorting().do(pool_f, only_non_dominated_front=True)
    pool_x, pool_f = pool_x[front], pool_f[front]
    keep = objective_crowding_keep(pool_f, capacity)
    return pool_x[keep], pool_f[keep]


def load_generator_pool(problem, run_path: Path, algorithm: str, capacity: int):
    paths = snapshots(run_path)
    final_x, final_f = load_live(paths[-1])
    if algorithm == "NSGA3-ObjectiveArchive":
        with np.load(paths[-1], allow_pickle=False) as saved:
            archive_x = np.asarray(saved["archive_X"], dtype=int)
            archive_f = np.asarray(saved["archive_F"], dtype=float)
        return np.row_stack([final_x, archive_x]), np.row_stack([final_f, archive_f]), paths[-1]
    if algorithm != "NSGA2":
        raise ValueError(algorithm)
    archive_x = archive_f = None
    for snapshot in paths:
        X, F = load_live(snapshot)
        archive_x, archive_f = update_nsga2_archive(
            problem, archive_x, archive_f, X, F, capacity)
    return np.row_stack([final_x, archive_x]), np.row_stack([final_f, archive_f]), paths[-1]


def richness(problem, X, F):
    identities = np.asarray(problem.decision_identity(X), dtype=str)
    _, first = np.unique(identities, return_index=True)
    first = np.sort(first)
    X, F = np.asarray(X)[first], np.asarray(F)[first]
    front = NonDominatedSorting().do(F, only_non_dominated_front=True)
    F = F[front]
    groups = Counter(tuple(row.tolist()) for row in F)
    repeated = sum(size for size in groups.values() if size > 1)
    return {
        "nondominated_unique_candidates": len(F),
        "exact_objective_groups": len(groups),
        "repeated_objective_groups": sum(size > 1 for size in groups.values()),
        "candidates_in_repeated_groups": repeated,
        "repeated_group_candidate_fraction": repeated / len(F) if len(F) else 0.0,
        "maximum_exact_objective_group_size": max(groups.values(), default=0),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=False)
    portfolio_dir = OUTPUT / "portfolios"
    portfolio_dir.mkdir()
    rows = []
    pool_rows = []
    sources = []
    seen_units = set()

    for scenario, directory, first_seed, last_seed in RUN_GROUPS:
        experiment = EXPERIMENTS / directory
        config_path = experiment / "config.json"
        manifest_path = experiment / "manifest.json"
        config = json.loads(config_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        problem = make_problem(
            config["map"], config["scenario"], int(config["agents"]),
            task_indices=config.get("task_indices"),
            preference_levels=int(config.get("preference_levels", 5)),
            independent_neighbor_orders=bool(config.get("independent_neighbor_orders", False)),
        )
        size = int(config["population"])
        ref_dirs = get_reference_directions("energy", problem.n_obj, size, seed=1)
        completed = {
            record["run_id"] for record in manifest.get("runs", [])
            if record.get("status") == "complete"
        }
        for seed in range(first_seed, last_seed + 1):
            for algorithm in ("NSGA3-ObjectiveArchive", "NSGA2"):
                unit = (scenario, seed, algorithm)
                if unit in seen_units:
                    raise RuntimeError(f"duplicate analysis unit: {unit}")
                seen_units.add(unit)
                run_id = f"{algorithm}_seed{seed:03d}"
                if run_id not in completed:
                    raise RuntimeError(f"incomplete source run: {directory}/{run_id}")
                run_path = experiment / "runs" / run_id
                X, F, snapshot = load_generator_pool(
                    problem, run_path, algorithm, 2 * size)
                pool_rows.append({
                    "scenario": scenario,
                    "family": "warehouse" if scenario.startswith("wh") else (
                        "maze" if scenario.startswith("maze") else scenario.rstrip("32")),
                    "seed": seed,
                    "generator": algorithm,
                    "raw_pool_size": len(X),
                    **richness(problem, X, F),
                })
                seed_rows = []
                quality_inputs = {}
                for selector in SELECTOR_NAMES:
                    repeats = RANDOM_REPLICATES if selector == "random_quality_matched" else 1
                    for repeat in range(repeats):
                        selector_seed = seed * 1000 + repeat
                        started = time.perf_counter()
                        selected = select_portfolio(
                            problem, X, F, size,
                            selector=selector,
                            quality_fraction=QUALITY_FRACTION,
                            random_seed=selector_seed,
                            ref_dirs=ref_dirs,
                        )
                        elapsed = time.perf_counter() - started
                        key = f"{scenario}_{algorithm}_seed{seed:03d}_{selector}_r{repeat:02d}"
                        portfolio_file = portfolio_dir / f"{key}.npz"
                        np.savez_compressed(
                            portfolio_file, X=selected.X, F=selected.F,
                            source_indices=selected.source_indices)
                        row = {
                            "scenario": scenario,
                            "family": "warehouse" if scenario.startswith("wh") else (
                                "maze" if scenario.startswith("maze") else scenario.rstrip("32")),
                            "seed": seed,
                            "generator": algorithm,
                            "selector": selector,
                            "selector_repeat": repeat,
                            "raw_pool_size": len(X),
                            "quality_core_size": selected.quality_core_size,
                            "selection_seconds": elapsed,
                            "source_experiment": directory,
                            "source_snapshot": str(snapshot.relative_to(experiment)),
                            "portfolio_file": str(portfolio_file.relative_to(OUTPUT)),
                            **fast_portfolio_metrics(problem, selected.X, selected.F),
                        }
                        rows.append(row)
                        seed_rows.append(row)
                        quality_inputs[f"{selector}_r{repeat:02d}"] = selected.F
                quality = empirical_quality(quality_inputs)
                for row in seed_rows:
                    qkey = f"{row['selector']}_r{int(row['selector_repeat']):02d}"
                    row.update(quality[qkey])
        sources.append({
            "scenario": scenario,
            "experiment": str(experiment),
            "seed_start": first_seed,
            "seed_end": last_seed,
            "config_sha256": sha256(config_path),
            "manifest_sha256": sha256(manifest_path),
        })

    if len(seen_units) != 320:
        raise AssertionError(f"expected 320 generator units, got {len(seen_units)}")
    write_csv(OUTPUT / "selector_results.csv", rows)
    write_csv(OUTPUT / "pool_richness.csv", pool_rows)

    summary = []
    metrics = (
        "exact_group_undirected_edge_coverage",
        "exact_group_undirected_two_edge_coverage",
        "unique_plan_count", "nondominated_output_size",
        "mean_sum_of_costs", "mean_clearance_risk",
        "mean_pairwise_edge_jaccard", "selection_seconds",
        "igd_empirical", "hv_ratio_empirical",
    )
    for scenario in sorted({row["scenario"] for row in rows}):
        for generator in ("NSGA3-ObjectiveArchive", "NSGA2"):
            for selector in SELECTOR_NAMES:
                subset = [row for row in rows if row["scenario"] == scenario
                          and row["generator"] == generator
                          and row["selector"] == selector]
                summary.append({
                    "scenario": scenario,
                    "generator": generator,
                    "selector": selector,
                    "rows": len(subset),
                    "independent_seeds": len({row["seed"] for row in subset}),
                    **{f"{metric}_mean": float(np.mean([row[metric] for row in subset]))
                       for metric in metrics},
                })
    write_csv(OUTPUT / "selector_summary.csv", summary)

    contrasts = []
    deterministic = [row for row in rows if row["selector"] in {"echo", "objective_only"}]
    for scenario in sorted({row["scenario"] for row in deterministic}):
        for generator in ("NSGA3-ObjectiveArchive", "NSGA2"):
            for metric in metrics:
                echo = {row["seed"]: float(row[metric]) for row in deterministic
                        if row["scenario"] == scenario and row["generator"] == generator
                        and row["selector"] == "echo"}
                objective = {row["seed"]: float(row[metric]) for row in deterministic
                             if row["scenario"] == scenario and row["generator"] == generator
                             and row["selector"] == "objective_only"}
                seeds = sorted(set(echo) & set(objective))
                delta = np.asarray([echo[s] - objective[s] for s in seeds])
                pvalue = 1.0 if np.allclose(delta, 0) else float(
                    wilcoxon(delta, zero_method="pratt").pvalue)
                contrasts.append({
                    "scenario": scenario,
                    "generator": generator,
                    "metric": metric,
                    "seeds": len(seeds),
                    "echo_minus_objective_mean": float(np.mean(delta)),
                    "echo_minus_objective_median": float(np.median(delta)),
                    "positive": int(np.count_nonzero(delta > 0)),
                    "ties": int(np.count_nonzero(delta == 0)),
                    "negative": int(np.count_nonzero(delta < 0)),
                    "wilcoxon_p": pvalue,
                })
    write_csv(OUTPUT / "echo_objective_contrasts.csv", contrasts)

    provenance = {
        "schema": "echo-final-multifamily-identical-pool-confirmation-v1",
        "evidence_status": "confirmatory",
        "protocol": "research_archive/plans/2026-09-03_echo_final_multifamily_protocol_v1.md",
        "selectors": SELECTOR_NAMES,
        "random_replicates": RANDOM_REPLICATES,
        "quality_fraction": QUALITY_FRACTION,
        "nsga2_archive": (
            "passive 2N objective-only archive reconstructed generation by generation; "
            "decision-identity deduplication, nondominated filtering, deterministic "
            "reference-independent crowding truncation; never re-enters search"
        ),
        "sources": sources,
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__)),
    }
    (OUTPUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    checksum = OUTPUT / "SHA256SUMS"
    files = sorted(path for path in OUTPUT.rglob("*") if path.is_file() and path != checksum)
    checksum.write_text("".join(
        f"{sha256(path)}  {path.relative_to(OUTPUT)}\n" for path in files))


if __name__ == "__main__":
    main()
