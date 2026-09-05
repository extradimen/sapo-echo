# ECHO parameter sensitivity protocol v2

Frozen before inspecting any sensitivity result. This replaces v1, which is
retained for audit but withdrawn before execution because NSGA-III's recorded
trajectory does not contain the complete pre-survival pool used at every
archive update. Exact counterfactual temporal archives therefore cannot be
reconstructed honestly without rerunning search.

## Purpose

Test whether the frozen default (`2N` available passive archive, 70% objective
core and 30% ECHO completion) is a narrow tuned point. No default will be
changed merely because another cell has the largest sample mean.

## Inputs and factors

- Reuse all eight untouched `even-14` scenarios, seeds 1401--1420, and both
  candidate generators from the final multi-family confirmation.
- Search trajectories and the authoritative final passive `2N` archives remain
  fixed; no path evaluation is repeated and no archive re-enters search.
- Available archive budgets are `N`, `1.5N`, and `2N`. Smaller budgets are
  deterministic objective-only coresets of the authoritative final archive;
  the final live population is then added unchanged.
- Objective-core fractions are 0.5, 0.6, 0.7, 0.8, and 0.9.
- Output remains exactly `K=N=30`. Each ECHO cell is compared with
  objective-only output from its identical capped pool.

## Endpoints and interpretation

- Primary: PSC-1 improvement in the four warehouse scenarios.
- Secondary: PSC-2, empirical HV, empirical IGD, and selection time.
- Guardrail: no scenario-generator cell may have mean empirical-HV loss below
  -0.005.
- Report the full surface. Retain `2N/0.7` unless contradicted; do not tune on
  boundary maps.
- Use equal-scenario 20,000-draw hierarchical bootstrap intervals with scenario
  as the primary replication unit.
- This experiment supports claims about available passive-memory budget, not
  the unobserved counterfactual history of a smaller online archive.
