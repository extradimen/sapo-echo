# Operational comparison after a physical-edge closure

Post-review exploratory experiment for IEEE Access manuscript
Access-2026-44964. The frozen protocol is
`2026-10-28_ieee-operational-replanning-protocol_v1.md` and the exact source
config is `config/ieee_revision_operational_replan_v1.json`.

## What was compared

Five previously declared two-agent task pairs on the Moving AI
`random-32-32-20` map. For each pair, the same enumerated candidate pool fed
an objective-only K=3 selector and the corrected strict ECHO K=3 selector.
Each pair received ten uniformly sampled pool-used edge closures and ten
highest objective-only-exposure closures. The same closed physical edge was
then supplied to the pinned official MO-CBS planner, with a local build overlay
filtering that undirected edge from graph adjacency. The path validator
independently rejected any returned path crossing the closed edge. A separate
smoke test caught and fixed an initial graph-copy issue before evidence
execution; the failed experiment startup produced no event records.

| Event stratum | Events | Eligible pool has a survivor | Objective-only immediate | Strict ECHO immediate | MO-CBS replans |
|---|---:|---:|---:|---:|---:|
| Uniform | 50 | 41 | 39 | 41 | 50 |
| High exposure | 50 | 12 | 0 | 12 | 50 |
| All protocol events | 100 | 53 | 39 | 53 | 100 |

The ECHO-minus-objective success-rate difference is 14 percentage points in
the 100 protocol events. A paired task-pair cluster bootstrap with 20,000
resamples gives a 95% interval of [0, 38] percentage points. There are only
five task-pair clusters; the interval includes zero. Two task pairs account
for all ECHO gains in the high-exposure stratum. Uniform and high-exposure
strata may include the same physical edge as two protocol events and are not
independent random samples of operational failures.

MO-CBS completed and returned a valid replan for all 100 events. Its median
response wall time was 10.37 ms, including instance serialization and process
startup, and its 95th percentile was 24.47 ms. The median offline candidate
generation time was 0.369 s per pair, followed by a median strict ECHO
composition time of 0.00085 s. A separate 1,000-repeat-per-event component
benchmark measured a median of event-median precomputed signature lookup time
of 1.083 microseconds. The latter excludes file loading and dispatch; it is
not a full operational response time.

## Interpretation and manuscript boundary

ECHO used every closure-surviving option present in its eligible pool on this
panel. Its failure to respond to 47 events was a candidate-supply failure.
Online MO-CBS solved those cases quickly on these small two-agent instances.
Thus this study supports a narrow portfolio-composition claim and offers no
evidence that precomputation beats online replanning in overall cost or
practical response time at small scale. MO-CBS is a published multiobjective
planner run again after closure, not a dedicated robust edge-fault planner.
Its objective is the same travel/risk pair, but it searches a different
solution space after observing the closure. The larger E7 study remains the
stress boundary; its 15-call reactive sampler is not relabeled as MO-CBS.

The paper should present this as an operational comparator and retain the
explicit limitation that an exact, like-for-like implementation of a dedicated
robust edge-closure planner has not been established. It must not claim
statistically secure superiority from five task pairs or a matched
end-to-end planning budget.

## Immutable evidence

- Event units, per-pair records, code/config/adapter hashes and source-commit
  provenance: `research_archive/experiments/2026-10-28_ieee-operational-replanning_v1/`.
- Pair-class table and paired cluster bootstrap:
  `research_archive/analyses/2026-10-28_ieee-operational-replanning-summary_v1/`.
- Component lookup timings:
  `research_archive/analyses/2026-10-28_ieee-operational-lookup_v1/`.
- Each directory has a complete `SHA256SUMS` file. All three checksum
  manifests were verified after child wait status reported successful
  completion. The official MO-CBS source commit is
  `80bc741d4b9d2284df7cdc4ad4404de1ce4ac3c6`.
