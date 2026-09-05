# ECHO equal-budget reactive replanning protocol v1 (E7)

Frozen before executing any E7 cell.

Execution note: the first v1 launch passed an integer where the Sobol API
requires a NumPy generator and stopped before completing any replanning call.
The empty v1 directory is retained for audit. The corrected, scientifically
identical run uses the v2 experiment identifier.

## Question

When ten physical edges fail one at a time, what does a prevalidated ECHO
portfolio provide relative to a reactive planner with the same total number of
joint planning calls?

## Panel and budget

- Scenarios: `wh10b`, `wh20a`, `random32`, and `maze32b`.
- Seeds: 1401--1410.
- Source generator: the frozen NSGA-III objective-archive run, 150 joint-path
  evaluations, `N=K=30`, followed by the frozen ECHO composer (`2N`, 0.7).
- For each scenario/seed, take the ten most frequently used undirected physical
  edges in the objective-only output; ties use the fixed physical-edge index.
  This selects consequential, method-independent exposure events without
  inspecting ECHO success.
- ECHO performs no failure-time planning: it immediately looks up a stored plan
  that avoids the closed edge.
- The reactive baseline receives 15 fresh scrambled-Sobol priority/preference
  genotypes after each closure. Ten events times 15 calls equals the 150 calls
  used once by the precomputed candidate generator.
- The blocked-edge planner forbids traversal in either direction while keeping
  both endpoint vertices available.

## Outcomes

- Immediate success of objective-only and ECHO portfolios.
- Reactive success after 15 calls, failed replans, and wall-clock latency.
- Best feasible normalized objective score after closure, using the original
  source pool's ideal/range only as a fixed reporting scale.
- ECHO lookup latency, original precomputation time, reactive total time, and
  storage (30 plans/genotypes versus no precomputed portfolio).

This is an amortized equal-call comparison over a ten-event contingency set,
not a claim that a finite portfolio dominates an unlimited online replanner.
