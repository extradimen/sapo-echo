"""Paired statistics and empirical-front quality for archived MAPF runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import moocore
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from pymoo.indicators.hv import HV
from movingai_mapf_experiment import make_problem, score


HIGHER_BETTER = {
    "hv_ratio_empirical",
    "backup_fraction_epsilon_001",
    "backup_fraction_epsilon_002",
    "backup_fraction_epsilon_005",
    "backup_fraction_epsilon_exact",
    "blocked_vertex_survival_epsilon_001",
    "blocked_vertex_survival_epsilon_002",
    "blocked_vertex_survival_epsilon_005",
    "blocked_vertex_survival_epsilon_exact",
    "blocked_edge_survival_epsilon_001",
    "blocked_edge_survival_epsilon_002",
    "blocked_edge_survival_epsilon_005",
    "blocked_edge_survival_epsilon_exact",
    "unique_plan_count",
    "nondominated_output_size",
    "mean_pairwise_route_distance",
    "unique_evaluated_genotypes",
    "hv_ratio_exact",
    "exact_cost_vector_coverage",
    "exact_group_undirected_edge_coverage",
    "exact_group_undirected_two_edge_coverage",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holm(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_rank_biserial(improvement):
    improvement = np.asarray(improvement, dtype=float)
    improvement = improvement[improvement != 0]
    if not len(improvement):
        return 0.0
    ranks = rankdata(np.abs(improvement))
    total = np.sum(ranks)
    return float((
        np.sum(ranks[improvement > 0]) - np.sum(ranks[improvement < 0])
    ) / total)


def empirical_quality(outputs):
    pooled = np.vstack(list(outputs.values()))
    reference = pooled[moocore.is_nondominated(pooled)]
    ideal = np.min(pooled, axis=0)
    scale = np.where(np.ptp(pooled, axis=0) > 0, np.ptp(pooled, axis=0), 1.0)
    normalized_reference = (reference - ideal) / scale
    hv = HV(ref_point=np.full(pooled.shape[1], 1.1))
    denominator = hv(normalized_reference)
    result = {}
    for algorithm, values in outputs.items():
        values = values[moocore.is_nondominated(values)]
        normalized = (values - ideal) / scale
        igd = np.mean(np.min(np.linalg.norm(
            normalized_reference[:, None, :] - normalized[None, :, :], axis=2
        ), axis=1))
        result[algorithm] = {
            "igd_empirical": float(igd),
            "hv_ratio_empirical": float(hv(normalized) / denominator),
            "empirical_reference_size": int(len(reference)),
        }
    return result


def exact_quality(values, reference):
    values = np.asarray(values, dtype=float)
    reference = np.asarray(reference, dtype=float)
    values = values[moocore.is_nondominated(values)]
    ideal = np.min(reference, axis=0)
    scale = np.where(np.ptp(reference, axis=0) > 0, np.ptp(reference, axis=0), 1.0)
    normalized_reference = (reference - ideal) / scale
    normalized = (values - ideal) / scale
    igd = np.mean(np.min(np.linalg.norm(
        normalized_reference[:, None, :] - normalized[None, :, :], axis=2
    ), axis=1))
    hv = HV(ref_point=np.full(reference.shape[1], 1.1))
    denominator = hv(normalized_reference)
    present = [
        np.any(np.all(np.isclose(values, point, atol=1e-12), axis=1))
        for point in reference
    ]
    return {
        "igd_exact": float(igd),
        "hv_ratio_exact": float(hv(normalized) / denominator),
        "exact_cost_vector_coverage": float(np.mean(present)),
    }


def load_runs(experiment, exact_reference=None):
    manifest = json.loads((experiment / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise RuntimeError(f"Experiment is not complete: {manifest['status']}")
    config = json.loads((experiment / "config.json").read_text())
    problem = make_problem(
        config["map"], config["scenario"], int(config["agents"]),
        task_indices=config.get("task_indices"),
        preference_levels=int(config.get("preference_levels", 5)),
        independent_neighbor_orders=bool(
            config.get("independent_neighbor_orders", False)),
    )
    rows = []
    outputs = {}
    for record in manifest["runs"]:
        run = experiment / "runs" / record["run_id"]
        metadata = json.loads((run / "metadata.json").read_text())
        metrics = json.loads((run / "metrics.json").read_text())
        final = np.load(run / "final.npz")
        X = np.asarray(final["declared_output_X"], dtype=int)
        F = np.asarray(final["declared_output_F"], dtype=float)
        derived = score(problem, X, F)
        metrics.update({
            name: value for name, value in derived.items() if name.endswith("_exact")
        })
        key = (int(metadata["seed"]), metadata["algorithm"])
        outputs[key] = F
        rows.append({**metadata, **metrics, "run_id": record["run_id"]})
    data = pd.DataFrame(rows)
    if data.duplicated(["algorithm", "seed"]).any():
        raise RuntimeError("Duplicate algorithm/seed rows")
    for seed in sorted(data.seed.unique()):
        seed_outputs = {
            algorithm: outputs[(int(seed), algorithm)]
            for algorithm in sorted(data.algorithm.unique())
        }
        quality = empirical_quality(seed_outputs)
        for algorithm, metrics in quality.items():
            mask = (data.seed == seed) & (data.algorithm == algorithm)
            for name, value in metrics.items():
                data.loc[mask, name] = value
        if exact_reference is not None:
            for algorithm, values in seed_outputs.items():
                mask = (data.seed == seed) & (data.algorithm == algorithm)
                for name, value in exact_quality(values, exact_reference).items():
                    data.loc[mask, name] = value
    return manifest, data


def run(args):
    experiment = Path(args.experiment).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    exact_reference = None
    exact_provenance = None
    if args.exact_reference_experiment:
        exact_path = Path(args.exact_reference_experiment).resolve()
        exact_manifest = json.loads((exact_path / "manifest.json").read_text())
        if exact_manifest["status"] != "complete":
            raise RuntimeError("Exact reference experiment is not complete")
        exact_metrics = json.loads((exact_path / "metrics.json").read_text())
        if not exact_metrics["find_all_pareto"]:
            raise RuntimeError("Exact reference did not certify the full Pareto set")
        exact_reference = np.asarray(exact_metrics["cost_vectors"], dtype=float)
        exact_provenance = {
            "path": str(exact_path),
            "manifest_sha256": sha256(exact_path / "manifest.json"),
            "checksums_sha256": sha256(exact_path / "SHA256SUMS"),
            "solution_count": len(exact_reference),
        }
    manifest, data = load_runs(experiment, exact_reference=exact_reference)
    data.to_csv(output / "combined_results.csv", index=False)

    identity = {
        "algorithm", "seed", "run_id", "population", "evaluation_budget",
        "decision_embedding_mode",
    }
    metrics = [column for column in data.columns if column not in identity]
    summary = data.groupby("algorithm")[metrics].agg(["mean", "std", "median"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / "descriptive_statistics.csv", index=False)

    algorithms = sorted(data.algorithm.unique())
    baseline = args.baseline
    if baseline not in algorithms:
        raise RuntimeError(f"Baseline {baseline!r} is absent")
    pivot = {
        name: data[data.algorithm == name].set_index("seed").sort_index()
        for name in algorithms
    }
    paired = []
    for competitor in algorithms:
        if competitor == baseline:
            continue
        for metric in metrics:
            first = pivot[baseline][metric].astype(float)
            other = pivot[competitor][metric].astype(float)
            improvement = first - other if metric in HIGHER_BETTER else other - first
            difference = first - other
            if np.allclose(difference, 0):
                pvalue = 1.0
            else:
                pvalue = float(wilcoxon(first, other, zero_method="pratt").pvalue)
            paired.append({
                "comparison": f"{baseline}_vs_{competitor}",
                "metric": metric,
                "p_raw": pvalue,
                "baseline_mean": float(first.mean()),
                "competitor_mean": float(other.mean()),
                "median_improvement": float(np.median(improvement)),
                "paired_rank_biserial_positive_is_baseline_better": (
                    paired_rank_biserial(improvement)
                ),
                "wins": int(np.sum(improvement > 0)),
                "ties": int(np.sum(improvement == 0)),
                "losses": int(np.sum(improvement < 0)),
            })
    paired = pd.DataFrame(paired)
    paired["p_holm"] = np.nan
    for metric, indices in paired.groupby("metric").groups.items():
        paired.loc[indices, "p_holm"] = holm(paired.loc[indices, "p_raw"])
    paired["significant_holm_005"] = paired.p_holm < .05
    paired.to_csv(output / "paired_statistics.csv", index=False)

    friedman = []
    if len(algorithms) >= 3:
        for metric in metrics:
            samples = [pivot[name][metric].astype(float).to_numpy() for name in algorithms]
            if all(np.allclose(samples[0], sample) for sample in samples[1:]):
                statistic, pvalue = 0.0, 1.0
            else:
                statistic, pvalue = friedmanchisquare(*samples)
            friedman.append({
                "metric": metric,
                "statistic": float(statistic),
                "p_value": float(pvalue),
                "algorithms": len(algorithms),
                "paired_seeds": len(samples[0]),
            })
    pd.DataFrame(friedman).to_csv(output / "friedman_statistics.csv", index=False)

    provenance = {
        "source_experiment": str(experiment),
        "source_manifest_sha256": sha256(experiment / "manifest.json"),
        "source_checksums_sha256": sha256(experiment / "SHA256SUMS"),
        "baseline": baseline,
        "quality_reference": (
            "Per-seed empirical nondominated union of all compared declared outputs; "
            "not an exact Pareto front and not confirmatory evidence."
        ),
        "paired_test": "two-sided Wilcoxon signed-rank with Pratt zeros",
        "multiple_testing": "Holm correction across competitors within each metric",
        "effect_size": "paired rank-biserial; positive means baseline is better",
        "friedman": "paired seeds as blocks",
        "protocol": manifest["protocol"],
        "exact_reference": exact_provenance,
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    files = sorted(path for path in output.iterdir() if path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in files
    ))


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--experiment", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--baseline", default="PNSGA3-v2")
    value.add_argument("--exact-reference-experiment")
    return value


if __name__ == "__main__":
    run(parser().parse_args())
