"""Paired inference for the cardinality-matched archive attribution."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from research_archive import atomic_json


ROOT = Path(__file__).resolve().parent
SOURCE = (ROOT / "research_archive" / "analyses" /
          "2026-10-28_ieee-revision_cardinality-matched-archive_v1")
OUTPUT = (ROOT / "research_archive" / "analyses" /
          "2026-10-28_ieee-revision_cardinality-statistics_v1")
CONTRASTS = (
    ("archive_content_objective", "objective_archive_objective",
     "no_archive_objective"),
    ("archive_content_strict", "objective_archive_strict",
     "no_archive_strict"),
    ("strict_composer_with_archive", "objective_archive_strict",
     "objective_archive_objective"),
)
METRICS = ("portfolio_psc1", "portfolio_psc2", "hv_ratio_empirical")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values: np.ndarray, seed: int, draws: int = 20000):
    rng = np.random.default_rng(seed)
    sampled = np.mean(rng.choice(values, (draws, len(values)), replace=True), axis=1)
    return float(np.mean(values)), float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))


def main() -> None:
    if (OUTPUT / "SHA256SUMS").exists():
        print(f"ALREADY COMPLETE {OUTPUT}")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE / "cardinality_matched_results.csv")
    rows = []
    for contrast_index, (name, treatment, control) in enumerate(CONTRASTS):
        for metric_index, metric in enumerate(METRICS):
            for problem in sorted(data.problem.unique()):
                block = data[data.problem == problem].pivot(
                    index="seed", columns="cell", values=metric)
                delta = (block[treatment] - block[control]).dropna().to_numpy(float)
                mean, low, high = interval(
                    delta, 20261028 + contrast_index * 100 + metric_index * 10 + len(rows))
                pvalue = (1.0 if np.allclose(delta, 0) else
                          float(wilcoxon(delta, zero_method="pratt").pvalue))
                rows.append({
                    "problem": problem, "contrast": name,
                    "treatment": treatment, "control": control,
                    "metric": metric, "paired_seeds": len(delta),
                    "mean_difference": mean, "ci95_low": low,
                    "ci95_high": high, "wilcoxon_p": pvalue,
                    "wins": int(np.count_nonzero(delta > 0)),
                    "ties": int(np.count_nonzero(delta == 0)),
                    "losses": int(np.count_nonzero(delta < 0)),
                })
    write_csv(OUTPUT / "paired_inference.csv", rows)
    atomic_json(OUTPUT / "decision.json", {
        "schema": "ieee-revision-cardinality-statistics-decision-v1",
        "interpretation": (
            "Archive filling and retained-content effects are distinct.  "
            "Only paired equal-cardinality contrasts in paired_inference.csv "
            "support retained-content attribution."),
        "all_strict_psc2_nonnegative": all(
            row["losses"] == 0 for row in rows
            if row["contrast"] == "archive_content_strict" and
            row["metric"] == "portfolio_psc2"),
    })
    atomic_json(OUTPUT / "provenance.json", {
        "schema": "ieee-revision-cardinality-statistics-v1",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_checksums_sha256": sha256(SOURCE / "SHA256SUMS"),
        "bootstrap_draws": 20000,
        "paired_by": "problem and seed",
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
    })
    checksum = OUTPUT / "SHA256SUMS"
    files = sorted(path for path in OUTPUT.iterdir()
                   if path.is_file() and path != checksum)
    checksum.write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in files))
    print(f"COMPLETE {OUTPUT}")


if __name__ == "__main__":
    main()
