# SAPO-ECHO

Reproducibility package for **SAPO-ECHO: Failure-Semantic Portfolio
Optimization for Resilient Multi-Agent Path Finding**.

SAPO is a general semantic-aware portfolio framework. ECHO is its concrete
edge/resource-closure mechanism for selecting a bounded set of solutions that
remain meaningfully different under resource failures. ECHO-MAPF applies the
mechanism to collision-free multi-agent path finding (MAPF).

The method first generates quality-controlled candidate plans with NSGA-III,
then performs passive portfolio selection using physical-edge overlap and
failure coverage. The archive never feeds back into the evolutionary search.

## What this release contains

- `code/`: core ECHO/NSGA-III implementation and analysis entry points;
- `protocols/`: frozen experimental and audit protocols;
- `results/`: compact processed CSV/JSON evidence for the reported claims,
  ablations, sensitivity analysis, and boundary tests;
- `reference_audit/`: dated authenticity audit for the cited literature;
- `assets/graphical_abstract.png`: visual summary;
- `MANIFEST.csv` and `SHA256SUMS`: package inventory and integrity hashes.

The directory names under `results/` intentionally retain the original dated
archive identifiers so every reported result can be traced to its protocol.

## Quick start

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements.txt
python code/analyze_echo_final_statistics.py --help
```

The compact release is primarily intended for audit and re-analysis. Some
large candidate portfolios are omitted, so not every search run can be
regenerated from this package alone.

## Data availability

This GitHub repository contains the complete source code, frozen protocols,
and compact processed evidence used to inspect the manuscript's main results.
The append-only raw experimental archive is approximately 3.9 GB and includes
large candidate portfolios and intermediate arrays. It is preserved by the
author and is available to editors and reviewers upon reasonable request. A
DOI-bearing archival deposit will be linked here when available.

To verify the compact release:

```bash
sha256sum -c SHA256SUMS
```

On macOS, install GNU coreutils and use `gsha256sum`, or compare the values
with `shasum -a 256`.

## Scope of the evidence

The experiments support a targeted claim: failure-semantic portfolio
selection can improve the recovery of genuinely distinct backup plans when
candidate pools contain such alternatives. The evidence does **not** support
a universal claim over all MAPF topologies. Warehouse instances show the
clearest benefit; random, room, and maze families expose important boundaries.

## Citation

The associated manuscript is under review. Until a bibliographic record is
available, cite this repository and its release tag. Citation metadata is also
provided in `CITATION.cff`.

## Author

Jinpeng Wang, Guangzhou City University of Technology, Guangzhou 510800,
China. ORCID: [0009-0003-7324-2211](https://orcid.org/0009-0003-7324-2211).

## License

Apache License 2.0. See `LICENSE`.
