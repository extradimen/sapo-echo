# Source code

This directory is a frozen source snapshot for the manuscript.

- `echo_portfolio.py`: identical-pool portfolio selectors, including ECHO.
- `pnsga3_mapf_edge_coverage.py`: final one-population NSGA-III-ECHO builder
  and compatibility aliases for historical experiments.
- `movingai_mapf_experiment.py`: MAPF experiment runner.
- `analyze_echo_*.py`: paper analyses and audits.
- `../vendor/pymoo/`: pinned pymoo dependency used by the experiments,
  including the passive-archive extension and MAPF problem implementation.

Historical function names are retained only when provenance files call them.
The authoritative paper-facing settings are in
[`../configs/nsga3_echo_frozen.yaml`](../configs/nsga3_echo_frozen.yaml).
