# ECHO small-instance portfolio-oracle protocol v1

Frozen: 2026-09-03, before executing the oracle analysis.

## Purpose and claim boundary

This experiment measures the loss of greedy ECHO relative to an exact
fixed-cardinality portfolio selector. It does not claim that the prioritized
decoder enumerates every mathematically possible MAPF path. The exact universe
is the complete finite decoder space declared below; the already archived
official MO-CBS solutions are used only to verify objective-front coverage.

## Instances and candidate universe

- Public Moving AI `random-32-32-20` map and `even-1` scenario.
- Five task pairs fixed by the earlier MO-CBS audit: `(1,4)`, `(1,19)`,
  `(3,4)`, `(3,5)`, and `(3,8)`, using zero-based scenario rows.
- Two priority orders and all `21^2` route-preference combinations: exactly
  882 canonical genotypes per task pair.
- Independently recheck endpoints, traversability, legal moves, collision
  freedom, sum of costs, and clearance risk.
- Collapse duplicate decoded joint paths, then retain the empirical
  nondominated candidate pool.

## Portfolio comparison

For each task pair and `K in {3, 6, 10}` when the exact residual enumeration
contains no more than 2,000,000 combinations (larger cells are reported as
predeclared computational omissions, not silently approximated):

1. Freeze the same NSGA-III reference-direction objective core used by ECHO
   (`round(0.70 K)` members).
2. Select the residual slots with greedy ECHO.
3. Exhaustively enumerate every residual completion of that fixed core and
   maximize the reported PSC-1 definition.
4. Apply deterministic ties in this order: higher PSC-2, lower summed
   normalized objective value, then lexicographically smaller candidate IDs.
5. Report greedy/oracle PSC-1 ratio, absolute gap, PSC-2, runtime, pool size,
   combination count, and exact-match indicator.

The exact guarantee and approximation statement apply only to the residual
problem conditional on the frozen quality core and finite eligible pool.

## Decision rule

The small-oracle audit supports the selector if all valid cells have
greedy/oracle PSC-1 ratio at least `1 - 1/e` and the median ratio is at least
0.95. Any lower cell remains in the main or supplementary results and triggers
an explanation of the greedy failure mode; instances are not replaced after
inspection.
