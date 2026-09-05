# ECHO temporal-archive confirmation protocol v1

Frozen: 2026-09-03, before running or examining `even-12`/`even-13` results.

## Inputs

- Moving AI warehouse `warehouse-10-20-10-2-1-even-12.scen`, SHA-256
  `1e7e07577badeff027c6d4e5f7e2ec3e411b4ae90b10ff391e29a3c913bb382e`;
- Moving AI warehouse `warehouse-10-20-10-2-1-even-13.scen`, SHA-256
  `c223792d656b7719c28c168137db362bcd162959848278c0e45bdbc889abc65a`;
- first 12 task rows, 21 preference levels;
- 20 paired seeds per scenario: 1201--1220 and 1301--1320;
- population/output `N=K=30`, 150 path evaluations;
- archive `2N`, update every five generations, no re-entry;
- every generation recorded.

Only the objective-only and ECHO archive searches are executed. The no-archive
cell is composed from their bitwise-identical live population, reducing compute
without removing any candidate. Both archive policies receive the same two
final composers, yielding the same five logical cells as development.

## Validity condition

Within every `(scenario, seed)`, objective-archive and ECHO-archive final live
`X/F` must be bitwise identical. Otherwise the result is invalid.

## Frozen decision

Adopt the simpler objective-only temporal archive in the final algorithm if:

1. ECHO final composition on the objective archive has positive mean PSC-1 on
   both scenarios and the equal-scenario hierarchical 95% interval is above 0;
2. full ECHO is within 0.002 PSC-1 of objective-archive + ECHO composition in
   equal-scenario mean;
3. the semantic archive does not have a positive hierarchical PSC-1 interval
   against the objective archive under the same ECHO composer;
4. objective-archive + ECHO output remains within 0.005 of the best HV ratio.

If condition 1 fails, ECHO composition is not confirmed. If conditions 2 or 3
fail, keep semantic archive retention and report its scenario dependence.

PSC-2 is a transfer endpoint and cannot override a failure on primary PSC-1.

