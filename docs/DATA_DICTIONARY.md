# Compact data dictionary

## Core identifiers

- `scenario`: frozen map/task block; the primary aggregation unit.
- `seed`: paired stochastic search seed within a scenario.
- `generator` or `algorithm`: candidate-pool generator.
- `selector`: fixed-cardinality output composition policy.

## Main endpoints

- `exact_group_undirected_edge_coverage` / PSC-1: proportion of declared
  single physical-edge closures for which an exact-objective group contains
  both an exposed and a surviving selected plan.
- `exact_group_undirected_two_edge_coverage` / PSC-2: analogous coverage under
  simultaneous two-edge closures; not directly optimized by ECHO.
- `hv_ratio_empirical`: selected-set hypervolume divided by the empirical
  candidate-pool reference value.
- `igd_empirical`: selected-set IGD to the empirical candidate-pool reference.
- `unique_plan_count`: number of distinct decoded joint plans delivered.

## E7b endpoints

- `class_success_rate`: fraction of actionable resource-response classes with
  at least one surviving selected plan.
- `mean_survivor_count`: mean number of selected survivors per closure class.
- `minimum_survivor_count`: worst survivor count over the audited classes.
- `two_class_success_rate`: survival coverage for simultaneous pairs of
  actionable resource classes.

Positive paired differences favor ECHO except for IGD, whose sign is reversed
by the inference scripts so that positive still denotes improvement.
