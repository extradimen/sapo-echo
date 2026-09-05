# Benchmark data provenance

The experiments use the public Moving AI MAPF benchmark. To avoid
redistributing upstream archives, this repository contains an exact inventory
of the selected warehouse inputs rather than copies of the map archives.

- Official source: <https://movingai.com/benchmarks/mapf.html>
- Citation: R. Stern et al., “Multi-Agent Pathfinding: Definitions, Variants,
  and Benchmarks,” SoCS 2019, pp. 151-158.
- License stated by the benchmark host: Open Data Commons Attribution License.
- Selected input inventory: [`warehouse_inventory.csv`](warehouse_inventory.csv)

Downloaded and verified archive hashes:

| Archive | SHA-256 |
|---|---|
| `mapf-map.zip` | `9da2e4c5ce03aa4e063b3a283ce874590b36cc4f31a297fe7ecb00d105abf288` |
| `mapf-scen-random.zip` | `20b7838f7a51f13e90a63ee138e9435fb4e41b0381becbc7313b7d3a7d859276` |
| `mapf-scen-even.zip` | `249896aaf15ef2d9beb378f954f0b7ca17189c6dec1b76a78965bbdbe714ad75` |

The processed data reported by the manuscript are under [`results/`](../results/).
The source-to-public mapping is [`docs/SOURCE_MAP.csv`](../docs/SOURCE_MAP.csv).
