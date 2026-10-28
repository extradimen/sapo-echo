# IEEE Access resubmission evidence

This is the paper-aligned public snapshot for manuscript Access-2026-44964.
The original September release remains in the repository for provenance, but
its group-wise selector and its larger reported effects are **historical**.
The revised paper evaluates the portfolio-level selector
`strict_coverage_psc1_psc2_tie` in
[`code/echo_revision.py`](code/echo_revision.py). The previously submitted
heuristic remains in [`code/echo_portfolio.py`](code/echo_portfolio.py), clearly
separated from the corrected method.

## Follow the revised Results section

| Paper topic | Public evidence | Main finding |
|---|---|---|
| Identical-pool corrected replay | [`01_corrected_portfolio_replay`](results/01_corrected_portfolio_replay/) | 320 frozen pools; small overall PSC gain |
| Matched-cardinality archive ablation | [`02_archive_attribution`](results/02_archive_attribution/) | Retained content separated from portfolio filling |
| Exact small-pool oracles | [`03_exact_oracles`](results/03_exact_oracles/) | Fixed-core residual and unconstrained fixed-K questions kept distinct |
| Larger-fleet boundary | [`04_larger_fleets`](results/04_larger_fleets/) | All 12 pilot cells fell short of K=30 at 150 evaluations |
| Critical closures, E7/E7b | [`05_critical_closures`](results/05_critical_closures/) | Most targeted closures lacked an eligible survivor |
| Published-planner response after closure | [`06_operational_replanning`](results/06_operational_replanning/) | ECHO 53/100, objective-only 39/100, MO-CBS 100/100 on five two-agent pairs |
| Final revision audit | [`07_final_audit`](results/07_final_audit/) | Counts, hashes and negative findings |

The operational study uses one random map and targeted as well as uniformly
sampled pool-used closures. MO-CBS is a published multiobjective planner rerun
after observing the same physical-edge closure. It is an operational response
comparator, not a dedicated robust edge-fault planner or a matched selector.
Its response median was 10.37 ms; the ECHO minus objective-only task-pair
bootstrap interval was 0 to 38 percentage points and includes zero. The study
does not establish an end-to-end latency advantage or generalization to large
fleets.

## Code and protocol

- [`code/`](code/) contains the exact corrected selector, analysis scripts,
  Moving AI decoder and MO-CBS edge-filter adapter included in the revision
  supplement. The official third-party MO-CBS source is obtained separately
  from [public_cppmomapf](https://github.com/rap-lab-org/public_cppmomapf) at
  commit `80bc741d4b9d2284df7cdc4ad4404de1ce4ac3c6`; it is not
  redistributed here.
- [`protocols/`](protocols/) contains the frozen revision and operational
  protocols. The operational configuration is
  [`code/config/ieee_revision_operational_replan_v1.json`](code/config/ieee_revision_operational_replan_v1.json).
- [`paper/table09_operational.tex`](paper/table09_operational.tex) is the exact
  source for the new operational-results Table 6. Its filename is an internal
  source identifier; the compiled manuscript number is Table 6.
- [`SOURCE_MAP.csv`](SOURCE_MAP.csv) maps every copied file to its unchanged
  working-archive path and SHA-256. Each evidence directory also retains its
  original `SHA256SUMS` manifest.

The public snapshot includes sealed analysis units, event records, statistics,
code and protocols. Large pre-submission candidate trajectories used as replay
inputs are not mirrored here; the local append-only archive preserves them and
they can be supplied to editors or reviewers on reasonable request. Therefore
the included material independently audits the reported calculations and
version identity, while a complete regeneration of every candidate pool also
requires those retained trajectory files and the licensed/public benchmark
inputs identified in the original release.
