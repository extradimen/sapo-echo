# IEEE Access resubmission experiment protocol v1

Frozen on 2026-10-28 before running revision analyses.

## Purpose

This post-review protocol addresses the implementation, metric, attribution,
cost, and scaling concerns raised for manuscript Access-2026-44964.  It does
not overwrite any pre-submission archive.  All outputs are append-only and are
labelled post-review revision evidence.

## Audit findings that motivate the protocol

The submitted implementation named `echo` maximizes a diversity heuristic
inside exact-objective groups.  The submitted PSC-1/PSC-2 implementation also
aggregates group-conditioned coverage.  These are not the portfolio-level
event-coverage function stated in the manuscript.  Consequently, the standard
monotone-submodular greedy guarantee cannot be attributed to that historical
implementation.

## Frozen definitions

For one candidate pool, the single-edge event universe is the set of undirected
physical edges used by at least one eligible candidate.  A selected portfolio
survives an event if at least one selected joint plan uses none of the closed
edges.  PSC-1 is the fraction of singleton events survived; PSC-2 is the
fraction of unordered two-edge events survived.  The event universe is fixed
from the eligible pool and is identical for all selectors in one comparison.

The corrected selector first freezes the same NSGA-III reference-direction
quality core.  Each residual step maximizes marginal portfolio-level PSC-1.
Ties are broken by marginal portfolio-level PSC-2 and then by source index.
The `(1-1/e)` statement applies only to the primary PSC-1 residual objective on
the fixed eligible pool and fixed core.  PSC-2 is a tie-break and has no stated
approximation guarantee.

The historical implementation is retained under the label
`legacy_group_heuristic`.  Historical group-conditioned metrics are retained
only as legacy diagnostics and are not called PSC-1 or PSC-2 in the revision.

## Work packages

1. **R1 consistency replay.** Reuse the 320 frozen scenario-seed-generator
   candidate pools from the eight-scenario final panel.  Compare corrected
   coverage greedy, PSC-1-only greedy, legacy group heuristic, objective-only,
   edge Jaccard, and minimum-common-edge under identical pools and fixed output
   cardinality.  Compute pool-fixed PSC-1/PSC-2, objective safeguards, output
   cardinality, and selection time.
2. **R2 paired inference.** Use scenario-then-seed hierarchical bootstrap with
   20,000 draws and paired scenario-specific effects.  Never average away an
   adverse family.
3. **R3 cardinality-matched archive attribution.** Reanalyze the frozen
   archive ablation at the largest paired unique cardinality available in both
   cells.  Separate portfolio filling from retained-content value.
4. **R4 exact oracle.** On bounded small eligible pools, report both (a) optimal
   residual selection with the quality core fixed and (b) optimal fixed-K
   selection without a fixed core.  Identify which gap belongs to which
   problem.
5. **R5 final-method cost.** Measure the corrected composer separately and add
   it to archived generator wall time.  Report distribution, hardware/software
   environment, and distinguish this from exhaustive failure injection.
6. **R6 scale/generalization.** Only after R1--R5 pass integrity checks, run a
   bounded larger-fleet panel on warehouse and boundary map families under the
   same evaluation and output budgets.  Preserve negative results.

## Decision rules

- If portfolio-level PSC-1 is saturated, state that result and do not use the
  legacy group-conditioned statistic as a substitute.
- If corrected ECHO does not beat objective-only or structural comparators,
  narrow or withdraw the corresponding superiority claim.
- A pool with fewer than K unique eligible plans reports a shortfall.  It is not
  padded with duplicates for fixed-cardinality attribution.
- Existing candidate pools and pre-submission files are immutable.

