"""Audit and summarize the frozen operational replanning event panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from research_archive import atomic_json


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "research_archive/experiments/2026-10-28_ieee-operational-replanning_v1"
OUTPUT = ROOT / "research_archive/analyses/2026-10-28_ieee-operational-replanning-summary_v1"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_source():
    lines = (SOURCE / "SHA256SUMS").read_text().splitlines()
    for line in lines:
        digest, relative = line.split("  ", 1)
        path = SOURCE / relative
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"source checksum mismatch: {relative}")
    if len(lines) != 103:
        raise AssertionError(f"expected 100 units and 3 summary files, got {len(lines)}")


def main():
    if (OUTPUT / "SHA256SUMS").exists():
        print(f"ALREADY COMPLETE {OUTPUT}")
        return
    audit_source()
    data = pd.read_csv(SOURCE / "event_results.csv")
    if len(data) != 100 or data["pair"].nunique() != 5:
        raise AssertionError("event panel size mismatch")
    if not set(data["event_class"]) == {"uniform", "high_exposure"}:
        raise AssertionError("event classes changed")
    if not (data.groupby(["pair", "event_class"]).size() == 10).all():
        raise AssertionError("pair/class event count mismatch")
    pairs = json.loads((SOURCE / "pair_records.json").read_text())
    if len(pairs) != 5:
        raise AssertionError("pair records changed")
    data["eligible_pool_support"] = data["eligible_pool_survivors"] > 0
    data["echo_minus_objective"] = (
        data["echo_immediate_success"].astype(int) -
        data["objective_immediate_success"].astype(int))
    rows = []
    rng = np.random.default_rng(20261028)
    for label, subset in [("all", data)] + [
        (name, data[data["event_class"] == name])
        for name in ("uniform", "high_exposure")]:
        effects = subset.groupby("pair")["echo_minus_objective"].mean().to_numpy()
        draws = np.mean(rng.choice(effects, size=(20_000, len(effects)),
                                   replace=True), axis=1)
        supported = subset[subset["eligible_pool_support"]]
        rows.append({
            "event_class": label,
            "events": int(len(subset)),
            "eligible_pool_supported": int(len(supported)),
            "eligible_pool_unsupported": int(len(subset) - len(supported)),
            "objective_success": int(subset["objective_immediate_success"].sum()),
            "echo_success": int(subset["echo_immediate_success"].sum()),
            "mocbs_replan_success": int(subset["mocbs_replan_success"].sum()),
            "mocbs_complete_pareto": int(subset["mocbs_complete_pareto"].sum()),
            "objective_supported_success": int(
                supported["objective_immediate_success"].sum()),
            "echo_supported_success": int(
                supported["echo_immediate_success"].sum()),
            "mocbs_response_median_seconds": float(
                subset["mocbs_response_seconds"].median()),
            "mocbs_response_p95_seconds": float(
                subset["mocbs_response_seconds"].quantile(.95)),
            "mocbs_search_median_seconds": float(
                subset["mocbs_search_seconds"].median()),
            "echo_minus_objective_rate": float(effects.mean()),
            "echo_minus_objective_cluster_bootstrap_95": [
                float(np.quantile(draws, .025)),
                float(np.quantile(draws, .975))],
        })
    by_pair = data.groupby(["pair", "event_class"], as_index=False).agg(
        events=("edge_index", "size"),
        supported=("eligible_pool_support", "sum"),
        objective_success=("objective_immediate_success", "sum"),
        echo_success=("echo_immediate_success", "sum"),
        mocbs_success=("mocbs_replan_success", "sum"),
        mocbs_median_seconds=("mocbs_response_seconds", "median"),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    by_pair.to_csv(OUTPUT / "by_pair_class.csv", index=False)
    atomic_json(OUTPUT / "summary.json", {
        "schema": "ieee-operational-replanning-summary-v1",
        "source_sha256s_sha256": sha256(SOURCE / "SHA256SUMS"),
        "pair_count": 5,
        "event_count": 100,
        "bootstrap_unit": "task pair; paired difference; 20000 resamples",
        "event_class_overlap": "uniform and high-exposure samples may share a physical edge; treated as separate protocol events",
        "results": rows,
        "offline_pool_generation_seconds_median": float(np.median([
            record["pool_seconds"] for record in pairs])),
        "offline_echo_composer_seconds_median": float(np.median([
            record["echo_composer_seconds"] for record in pairs])),
        "offline_objective_composer_seconds_median": float(np.median([
            record["objective_composer_seconds"] for record in pairs])),
        "limitation": "two agents on one random map; MO-CBS is a published online replanner here, not a dedicated edge-fault-tolerant planner",
    })
    atomic_json(OUTPUT / "provenance.json", {
        "script_sha256": sha256(__file__),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256s_sha256": sha256(SOURCE / "SHA256SUMS"),
    })
    checksum = OUTPUT / "SHA256SUMS"
    checksum.write_text("".join(
        f"{sha256(path)}  {path.name}\n"
        for path in sorted(OUTPUT.iterdir())
        if path.is_file() and path != checksum))
    print(f"SEALED {OUTPUT}")


if __name__ == "__main__":
    main()
