# SAPO-ECHO

Official code-and-data snapshot for **SAPO-ECHO: Failure-Semantic Portfolio
Optimization for Resilient Multi-Agent Path Finding**.

SAPO separates objective quality, `F(x)`, from application semantics, `H(x)`.
ECHO instantiates `H(x)` with undirected physical-edge closures and selects a
fixed-size portfolio whose members are less likely to fail together.
NSGA-III-ECHO uses ordinary single-population NSGA-III, a passive objective-only
`2N` history, and a final `N`-plan composer with a 70% objective core. The
archive never re-enters search.

## Main result in one paragraph

On held-out warehouse candidate pools, ECHO achieved PSC-1 of 0.79512 versus
0.78940 for edge-Jaccard selection and 0.77940 for objective-only selection.
The paired hierarchical PSC-1 difference from edge Jaccard was 0.00572 (95%
interval [0.00217, 0.01113]). Objective safeguards tied (empirical HV ratio
1.0; empirical IGD 0). Benefits were near zero on several random, room and
maze instances, so the claim is deliberately topology-qualified rather than
universal.

## Repository map

| Path | Purpose |
|---|---|
| [`paper/`](paper/) | paper-to-figure and paper-to-evidence navigation |
| [`results/`](results/) | processed experimental evidence in Results-section order |
| [`protocols/`](protocols/) | frozen protocols in the same numbered order |
| [`configs/`](configs/) | final paper configuration; one island, no migration |
| [`src/`](src/) | SAPO-ECHO algorithm, experiment and analysis source |
| [`vendor/`](vendor/) | pinned in-tree pymoo dependency used by the experiments |
| [`data/`](data/) | public benchmark provenance and exact inventory |
| [`docs/`](docs/) | source mapping, reference audit and release policy |
| [`assets/`](assets/) | graphical summary |

Internal dates, pilots and temporary run names are intentionally absent from
the navigation tree. Their immutable mapping is retained in
[`docs/SOURCE_MAP.csv`](docs/SOURCE_MAP.csv). The distinction between the
private append-only workspace and this public snapshot is documented in
[`docs/WORKSPACE_AND_RELEASE.md`](docs/WORKSPACE_AND_RELEASE.md).

## Follow the paper

Start with [`results/README.md`](results/README.md). Its six numbered folders
follow the manuscript's Results subsections and link each claim to the
corresponding figure or table. Figures 1-6 are available in
[`paper/figures/`](paper/figures/).

## Quick verification

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e vendor
python tools/verify_release.py
PYTHONPATH=src:vendor python -c "from echo_portfolio import SELECTORS; print(SELECTORS)"
```

Regenerate the inventory after an intentional release change:

```bash
python tools/refresh_manifest.py
```

## Data availability

This repository includes the processed CSV/JSON evidence used by the paper,
the frozen protocols, scenario inventory, figures and executable source. Large
candidate trajectories and intermediate arrays are kept in the checksum-sealed
local research archive and can be provided to editors and reviewers on
reasonable request. They are not mixed into the public navigation tree.

The underlying Moving AI maps and scenarios remain with their official host;
their source, license, archive hashes and selected-file inventory are recorded
in [`data/README.md`](data/README.md).

## Releasing a new snapshot

Experiments run only in the local append-only workspace. After scientific and
completeness audit, selected files are copied through the source map, checked,
committed and assigned a new immutable tag. Submission ZIP files are built from
that fixed Git commit, never edited as a second source of truth. For example:

```bash
python tools/build_release_zip.py --revision <fixed-tag> --output sapo-echo.zip
```

## Citation and license

Citation metadata are in [`CITATION.cff`](CITATION.cff). Project code is
released under Apache-2.0; benchmark and third-party components retain their
respective licenses and attribution requirements.

Author: Jinpeng Wang, Guangzhou City University of Technology, Guangzhou
510800, China. [ORCID 0009-0003-7324-2211](https://orcid.org/0009-0003-7324-2211).
