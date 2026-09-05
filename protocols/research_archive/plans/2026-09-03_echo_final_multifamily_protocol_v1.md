# ECHO final untouched multi-family confirmation protocol v1

Frozen: 2026-09-03 before executing any listed scenario.

## Hypotheses

- Primary: on warehouse scenarios, ECHO final composition improves PSC-1 over
  objective-only output on the identical passive objective archive without an
  empirical HV loss larger than 0.005.
- Transfer: the PSC-1 selector may improve PSC-2, but PSC-2 cannot rescue a
  failed primary endpoint.
- Boundary: random, room, and maze families are expected to contain less
  semantic headroom and are retained regardless of sign.
- Generator robustness: the selector effect is evaluated with both NSGA-III
  and NSGA-II candidate generation; no generator is dropped after inspection.

## Frozen scenario units

All use the first 12 task rows, `N=K=30`, 21 preference levels, 150 path
evaluations, seeds 1401--1420, duplicate-genotype elimination, and a snapshot
of every generation.

| Label | Family | Map SHA-256 | Scenario (`even-14`) SHA-256 |
|---|---|---|---|
| wh10a | warehouse | `c8d1b2f24788ed6bd1ccf45065b96b4ce82d65f88c72de750e03e2758637bff0` | `eca354a02fc1ace6c9ca2f4cd29b735c9aa85a53220dfdb1047a6f5dca020fbd` |
| wh10b | warehouse | `4f06e82c2b87238daa8e308086afdba701112bf023e94e740e5bf9508a6adec3` | `544d0aa9f2ce6446e94777faccd2ff0ebf7e84f6615b15daa2e168649be2c25b` |
| wh20a | warehouse | `bd3bec2d1c20a8bbf900583cb4c2cf9dc16101470fbfae93b272ee0fb575d3ec` | `30096a9adaf81dbe6d8c1e3d4b14c5171db8850879e8af29f1144736b10db1cc` |
| wh20b | warehouse | `eae2a3f5298b1e113bfc32b5b4404f5de5105365d237df63cc9dc9e2d1c73713` | `1494ea661294823c0d4f2f0b5fe52dfd1cf52a405bee8879dc5b75f8152d2c6d` |
| random32 | random | `4240fddfa77d88b72ce779e02acf46a5ff056a3b04af5a4e35f7bc86cdfba3ec` | `cf7b12fd5ccf8c9bd8f45399c1e3ea06238c6b0d885a982d102bbf18c1ee6017` |
| room32 | room | `f107fefbf63c7a1a3753e4a033452301a1cdd7043cab7f45e4ed0e7dec1e90a9` | `c9ef67f780e64b19d98eb9d8ea3f1f360715208788f379c63cfaedc0f4401c40` |
| maze32a | maze | `5c549328775ce530072cb05eda8f9010235a7e29294806d6aebb0ad667479cd3` | `b7fec1ef3bf88a3d813ee042d023bb0ac5afe842db69b0b6d33d60a302e31e5f` |
| maze32b | maze | `7ff67aa59f71933b8cf2605e12631b8a28d9ebcfb9b941de3afdc7dce3123fee` | `93a133ceef3676e33a682e45b947fae44966b45465a0f76c7cffc1f708bfc7eb` |

The archive audit found none of these `even-14` scenario files in an earlier
experiment configuration before this protocol was frozen.

## Search and offline comparison

- Run `NSGA3-ObjectiveArchive` and ordinary `NSGA2` once for every
  `(scenario, seed)` pair: 320 base searches in total.
- NSGA-III uses the final method's passive objective-only `2N` archive.
- For NSGA-II, reconstruct a passive objective-only `2N` archive from its
  every-generation trajectory using only objective values and deterministic
  crowding/reference-independent tie handling; it never re-enters search.
- On each generator-specific identical pool, compare objective-only, repeated
  quality-matched random, edge Jaccard, minimum common edge, common vertex,
  edge k-medoids proxy, the two Xu et al. adaptations, and ECHO.
- Report PSC-1, PSC-2, HV/IGD safeguards, unique plans, selector time, search
  wall time, peak memory where available, and semantic-richness diagnostics.

## Inference

Scenario is the primary replication unit. Use equal-scenario hierarchical
paired bootstrap intervals (20,000 draws), with seed-paired Wilcoxon/Holm tests
as secondary evidence. Report warehouse, boundary, generator, and full-panel
effects separately; do not average away an adverse family.

