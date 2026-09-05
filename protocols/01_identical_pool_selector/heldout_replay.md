# ECHO identical-pool held-out replay protocol v1

Frozen: 2026-09-03, after inspecting only the even-8 development replay and
before running any selector on even-9, even-10, or even-11.

## Scope

This is a held-out archived-pool confirmation of the **final portfolio
composer**, not a confirmation of counterfactual temporal archive policies.
Each input run supplies its last saved live population plus pre-composition
passive archive. All selectors receive the identical deduplicated,
nondominated candidate pool, the same 70% objective core, and exactly 30 output
slots. Random selection uses 20 deterministic within-seed repetitions that are
averaged before inference.

## Frozen inputs

| Scenario | Input SHA256SUMS file hash |
|---|---|
| even-9 | `9477747b55970b961d33f7d8d895c8c85d537c077138425c90a544bb4a19ab74` |
| even-10 | `c3973568803441cd45b4f257f5986b9956267d1c640129a382935ba1088d832a` |
| even-11 | `50c0c34fbcce8e83b9a40dd7c2068eb49aa32a199edd3d2a08fcb5ff7ff27bc0` |

Each scenario contains 20 paired seeds. Source algorithm label is the archived
compatibility name `NSGA3-EdgeCoverage`; its search is ordinary
single-population NSGA-III and its passive mechanism is the same numerical
construction now named ECHO.

## Frozen selectors

1. objective-only NSGA-III reference-direction output;
2. random quality-matched fill;
3. exact-objective edge-Jaccard farthest fill;
4. exact-objective minimum-common-edge fill;
5. greedy edge-signature facility-location (`kmedoids_edge`) proxy;
6. exact-objective common-vertex-overlap fill;
7. ECHO marginal single-edge failure-coverage fill.

The common-vertex selector is a transparent generic overlap comparator. It is
not yet labeled as a faithful reproduction of Xu et al. (2025); that requires
a separate formula/code audit.

## Frozen implementation hashes

- `echo_portfolio.py`: `3491fbc25b0b36b30f10a578451c049a050d8b485b75e8a4b32d8ecc9413d2e3`
- `analyze_echo_selector_replay.py`: `8f9d5af570a7cf96b534c7ae90f2eddd309d79ccc83ff1ea259f5cd383a4558c`
- `analyze_echo_selector_decision.py`: `fdedb3f01fa0697aaed11854e92a03a3e56d712951c3c1bd00476758ad0c65fd`

## Endpoints and gates

Primary endpoint: exact-objective uniform single-edge portfolio survival
coverage (`PSC-1`).

Secondary/out-of-objective endpoint: simultaneous two-edge survival coverage
(`PSC-2`). Quality safeguards are empirical HV ratio and IGD from the pooled
selector outputs within each scenario and seed.

The composer-level confirmation passes only if all conditions hold:

1. ECHO has the highest mean PSC-1 on at least two of three scenarios;
2. ECHO's equal-scenario PSC-1 mean exceeds every deterministic generic
   selector;
3. a hierarchical paired bootstrap over scenario then seed has a lower 95%
   bound above zero for ECHO PSC-1 versus the strongest deterministic generic
   selector in the held-out panel;
4. ECHO's equal-scenario PSC-2 is no more than 0.005 below the best selector;
5. ECHO's equal-scenario empirical-HV ratio is no more than 0.005 below the
   best selector.

Wilcoxon tests with Holm correction, effect sizes, and win/tie/loss counts are
secondary. Scenario is the external-validity unit; 60 seeds must not be treated
as 60 independent problem instances.

## Interpretation limit

Passing this protocol would show that ECHO's final composer outperforms the
implemented generic output selectors on held-out ECHO-retained candidate
pools. It would not establish superiority to a faithfully reproduced published
method, a selector-specific temporal archive, all-optimal MAPF, or online
replanning. Those remain required experiments in the main paper plan.

