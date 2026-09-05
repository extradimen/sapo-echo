# ECHO temporal-archive five-cell ablation protocol v1

Frozen: 2026-09-03, before running the new archive-policy experiment.

## Question

Under one bitwise-identical NSGA-III live search trajectory, which part creates
the failure-coverage benefit: passive temporal retention, semantic retention,
or ECHO final composition?

## Development experiment

- Moving AI `warehouse-10-20-10-2-1-even-8.scen`;
- first 12 tasks, 21 preference levels;
- population/output `N=K=30`, 150 path evaluations;
- seeds 821--840;
- archive capacity `2N`, update every 5 generations;
- every generation recorded;
- all archive variants passive, with no search re-entry;
- final ECHO composer uses a 70% objective core and 30% semantic fill.

## Five frozen cells

1. no archive + objective-only output;
2. objective-only `2N` archive + objective-only output;
3. ECHO `2N` archive + objective-only output;
4. objective-only `2N` archive + ECHO output;
5. ECHO `2N` archive + ECHO output.

The objective-only archive is a deterministic objective-space coreset. ECHO's
archive has its frozen 50% objective core and 50% failure-coverage fill.

## Mandatory invariant

For every seed, final live `X` and `F` must be bitwise identical among ordinary
NSGA-III, NSGA-III with the objective archive, and NSGA-III with ECHO archive.
Any violation invalidates the attribution experiment.

## Endpoints

- primary: exact-objective-group single-undirected-edge survival `PSC-1`;
- secondary transfer: simultaneous two-edge survival `PSC-2`;
- safeguards: empirical HV ratio, IGD, unique plans and nondominated output;
- cost: full run wall time from the run archive and offline composer time.

This is a development ablation. A later scenario-level confirmation will be
frozen only after this implementation passes invariants and tests.

