# Moving AI multi-agent path-finding benchmark

Official source: <https://movingai.com/benchmarks/mapf.html>.

The benchmark page provides the maps and random/even start-goal scenario sets
under the Open Data Commons Attribution License and asks papers to cite:

R. Stern et al., “Multi-Agent Pathfinding: Definitions, Variants, and
Benchmarks,” SoCS 2019, pp. 151–158.

Downloaded archives are not committed. The initial robotics extension uses
warehouse maps, with maze and room maps retained as geometry-shift checks.
Robot identity symmetry, collision feasibility, and wait actions must be
canonicalized before measuring path-set diversity. These files are only data
preparation at this stage; no robotics result is evidence until a frozen MAPF
encoding and dedicated comparator protocol has been archived.

The selected dedicated comparators are the paper authors' MO-CBS and MOM*
implementations from <https://github.com/rap-lab-org/public_cppmomapf>, pinned
at commit `80bc741d4b9d2284df7cdc4ad4404de1ce4ac3c6`. The source is limited to
academic/non-commercial use, so it remains a separately downloaded dependency.
The local smoke build and output are described in
`momapf_baseline_provenance.json`.

The publication adapter in `external_adapters/momapf` builds a small command
line wrapper against that pinned checkout; it does not vendor or redistribute
the restricted source or compiled binary. `run_official_mocbs_experiment.py`
archives the exact input, raw solver output, independent validation report,
dependency commit, and SHA-256 hashes. The first frozen exact-reference panel
uses four two-agent random-map task pairs. It shows equal complete Pareto-cost
coverage for every algorithm and more distinct PNSGA-III joint paths, but no
consistent disruption-survival advantage. The data therefore remain a narrow
small-instance confirmation rather than a general robot-performance claim.

The in-tree evolutionary adapter now provides collision-free prioritized
space-time A* decoding, canonical joint-path identity, physical blockage
metrics, and a public warehouse experiment runner. Its first unseen-scenario
confirmation is a documented negative result; neither the tested larger
archive multiplicity nor physical-occupancy embedding is part of the frozen
PNSGA-III v2 definition.

Downloaded 2026-08-31 archive SHA-256 values:

- `mapf-map.zip`: `9da2e4c5ce03aa4e063b3a283ce874590b36cc4f31a297fe7ecb00d105abf288`
- `mapf-scen-random.zip`: `20b7838f7a51f13e90a63ee138e9435fb4e41b0381becbc7313b7d3a7d859276`
- `mapf-scen-even.zip`: `249896aaf15ef2d9beb378f954f0b7ca17189c6dec1b76a78965bbdbe714ad75`

All three archives passed `unzip -t`. The map archive contains four official
warehouse layouts (`warehouse-10-20-10-2-{1,2}` and
`warehouse-20-40-10-2-{1,2}`), plus maze and room shift checks.
