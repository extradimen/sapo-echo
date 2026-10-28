# E7b supported critical-resource selector protocol v1

Frozen before computing E7b outcomes. This is a post-hoc exploratory analysis
of the already sealed final multi-family pools. It does not rerun search, alter
ECHO, or replace the registered PSC-1/PSC-2 confirmation.

## Question

When the common eligible candidate pool contains both plans that use and plans
that avoid a highly exposed resource, which fixed-size selector retains the
strongest critical-resource contingency portfolio?

## Panel and fairness

- Reuse all eight final scenarios, seeds 1401--1420, and both NSGA-III and
  NSGA-II candidate generators: 320 pool units.
- Use the frozen deduplicated nondominated pool available to every saved
  composer and the saved `K=30` outputs.
- Compare objective-only, 20 repeated quality-matched random selections,
  edge Jaccard, minimum-common-edge, edge k-medoids, common-vertex overlap,
  both transparent Xu-mechanism adaptations, and ECHO.
- Do not inspect any selector output when constructing resource events.

## Supported resource classes

For each pool, encode every undirected physical edge by its binary usage vector
over the eligible candidates. Exclude all-zero classes (unused) and all-one
classes (no candidate can avoid the edge). Merge edges with identical usage
vectors into one resource-response class, so consecutive segments traversed by
exactly the same candidates cannot be counted as independent evidence.

Rank the remaining classes by eligible-pool exposure fraction, descending,
with the lowest physical-edge index as the deterministic tie break. Use the
first ten classes, or every class if fewer than ten exist. A unit with no such
class is retained explicitly and contributes no conditional selector outcome.

## Outcomes

- Primary diagnostic: fraction of selected resource classes for which at least
  one delivered plan survives.
- Resilience depth: mean, minimum, and fraction of the 30 delivered plans that
  survive each class.
- Transfer: fraction of unordered pairs of selected classes for which at least
  one delivered plan avoids both.
- Report class size (number of physically distinct edges sharing one candidate
  response) and exposure so merged corridors remain auditable.

Random replicates are collapsed within pool before scenario aggregation.
Scenario is the primary unit for any later hierarchical bootstrap. No new
confirmatory claim is permitted from this post-hoc panel alone. If a material,
multi-scenario ECHO advantage appears, an untouched protocol must be frozen
before running new search seeds; if methods tie, no new runs are triggered.

