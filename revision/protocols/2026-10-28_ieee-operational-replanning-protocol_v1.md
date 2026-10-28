# IEEE Access operational replanning comparison v1

Frozen before evidence execution on 2026-10-28. This is post-review evidence,
not a revision of the submitted search algorithm. Five previously declared
random-32-32-20 two-agent task pairs and the fixed map are specified in
`config/ieee_revision_operational_replan_v1.json`.

For each pair, enumerate the same 882 priority/preference genotypes used by the
small-instance oracle study. Deduplicate joint trajectories and retain the
nondominated eligible pool. Compose K=3 plans with the corrected frozen-core
ECHO selector, and K=3 plans with objective-only NSGA-III reference-direction
selection from that identical pool. Freeze the event universe to physical edges
used by at least one eligible plan. Select 10 edges uniformly without
replacement using the frozen seed and 10 edges with greatest objective-only
portfolio exposure (edge index breaks ties), preserving overlap as a separate
event-class label. All methods receive the same physical-edge closure.

For each event, independently validate stored-plan survival and run the
published MO-CBS implementation from the pinned local official source commit,
using a temporary build overlay that filters that undirected edge from graph
successors and predecessors. The official source tree is never modified;
the source and overlay hashes are recorded. The solver is run from a new process after the
closure with a 10-second wall-clock search limit. Report solver completion,
valid replans, response latency, search expansions and objective vectors.
Report offline pool generation and portfolio composition separately, and
classify whether the precomputed eligible pool contains any surviving plan.
Do not interpret 10 seconds of MO-CBS search as a matched compute budget to
150 function evaluations or as an implementation of a fault-tolerant planner.

This bounded two-agent study answers the operational comparison directly for
small instances. The existing E7 400-event panel remains the larger-instance
feasibility boundary and its online sampler is not relabeled as MO-CBS.
Incomplete or failed MO-CBS calls remain in the denominator as timeouts or
failures, not omitted. Every pair is an independently resumable unit. The
checksum manifest is written only after all five units complete.
