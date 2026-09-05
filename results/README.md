# Results index

The numbered folders follow the order of the manuscript's Results section.
CSV files are machine-readable evidence; JSON files record decisions,
provenance and frozen analysis settings.

| Manuscript result | Evidence directory | Main paper item |
|---|---|---|
| Identical-pool selector confirmation | [`01_identical_pool_selector`](01_identical_pool_selector/) | Fig. 3; Table IV |
| Closest published mechanism adaptations | [`02_published_mechanism_comparison`](02_published_mechanism_comparison/) | Fig. 4; Table V |
| Multi-family and parameter robustness | [`03_multifamily_and_sensitivity`](03_multifamily_and_sensitivity/) | robustness text and intervals |
| Supported-resource stress audit | [`04_supported_resource_stress`](04_supported_resource_stress/) | Table VIII |
| Component attribution | [`05_component_attribution`](05_component_attribution/) | Fig. 5; Table VI |
| Cost, candidate supply and failure boundary | [`06_cost_boundary_and_support`](06_cost_boundary_and_support/) | Fig. 6; Table VII |

These are curated copies. Original dated identifiers are preserved in
[`docs/SOURCE_MAP.csv`](../docs/SOURCE_MAP.csv) and in each provenance file.

Two paper statistics can be regenerated directly from the included processed
data (outputs are written under ignored `build/`):

```bash
python src/analyze_echo_final_statistics.py
python src/analyze_echo_e7b_statistics.py
```
