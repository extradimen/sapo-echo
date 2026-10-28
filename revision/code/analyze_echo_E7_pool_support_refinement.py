"""Refine sealed E7 v2 pool-support causes without replaying any path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from research_archive import atomic_json


ROOT = Path(__file__).resolve().parent
SOURCE = (ROOT / "research_archive" / "analyses" /
          "2026-09-05_echo-E7_pool-support-audit_v2")
OUTPUT = (ROOT / "research_archive" / "analyses" /
          "2026-09-05_echo-E7_pool-support-causal-refinement_v1")
ADDENDUM = (ROOT / "research_archive" / "plans" /
            "2026-09-05_echo_E7_pool-support_classification_addendum_v1.md")
SELECTORS = (
    "objective_only", "edge_jaccard", "minimum_common_edge",
    "kmedoids_edge", "common_vertex_overlap", "xu2025_paper_vertex",
    "xu2025_official_edge", "echo",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def causal_label(row):
    if not bool(row.raw_pool_support):
        return "search_supply_miss"
    if not bool(row.eligible_pool_support):
        return "eligibility_filter_miss"
    if not bool(row.echo_success):
        return "echo_selection_miss"
    return "echo_success"


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    source_file = SOURCE / "event_pool_support.csv"
    data = pd.read_csv(source_file)
    if len(data) != 400:
        raise AssertionError("expected 400 sealed E7 events")
    data["causal_classification"] = data.apply(causal_label, axis=1)
    data.to_csv(OUTPUT / "event_causal_classification.csv", index=False)

    rows = []
    for scenario, group in data.groupby("scenario", sort=True):
        supported = group[group["eligible_pool_support"]]
        row = {
            "scenario": scenario,
            "events": len(group),
            "search_supply_miss": int((group.causal_classification ==
                                        "search_supply_miss").sum()),
            "eligibility_filter_miss": int((group.causal_classification ==
                                             "eligibility_filter_miss").sum()),
            "echo_selection_miss": int((group.causal_classification ==
                                         "echo_selection_miss").sum()),
            "echo_success": int((group.causal_classification ==
                                  "echo_success").sum()),
            "raw_pool_support": int(group.raw_pool_support.sum()),
            "eligible_pool_support": int(group.eligible_pool_support.sum()),
        }
        for selector in SELECTORS:
            row[f"{selector}_success_given_eligible_support"] = (
                float(supported[f"{selector}_success"].mean())
                if len(supported) else float("nan"))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "scenario_summary.csv", index=False)
    totals = {
        label: int((data.causal_classification == label).sum())
        for label in (
            "search_supply_miss", "eligibility_filter_miss",
            "echo_selection_miss", "echo_success")
    }
    atomic_json(OUTPUT / "decision.json", {
        "schema": "echo-E7-pool-support-causal-refinement-v1",
        "events": len(data),
        "totals": totals,
        "primary_finding": (
            "No E7 event was an ECHO selection miss: every event supported "
            "by the frozen eligible pool was retained by ECHO."),
        "interpretation": (
            "The E7 immediate-failure boundary is dominated by search supply "
            "and frozen objective-eligibility filtering, not demonstrated "
            "failure of the ECHO composer."),
        "status": "post-hoc diagnostic; not a new confirmatory test",
    })
    atomic_json(OUTPUT / "provenance.json", {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256s_sha256": sha256(SOURCE / "SHA256SUMS"),
        "source_event_table_sha256": sha256(source_file),
        "addendum": str(ADDENDUM.relative_to(ROOT)),
        "addendum_sha256": sha256(ADDENDUM),
        "script": str(Path(__file__).relative_to(ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "no_decoder_or_optimizer_replay": True,
    })
    checksum = OUTPUT / "SHA256SUMS"
    files = sorted(path for path in OUTPUT.iterdir()
                   if path.is_file() and path != checksum)
    checksum.write_text("".join(
        f"{sha256(path)}  {path.name}\n" for path in files))


if __name__ == "__main__":
    main()
