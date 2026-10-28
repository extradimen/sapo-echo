# SAPO-ECHO

Code-and-data repository for **SAPO-ECHO: Failure-Semantic Portfolio
Optimization for Resilient Multi-Agent Path Finding**. The current IEEE Access
resubmission evidence is indexed in [`revision/`](revision/). The original
September snapshot is retained below as historical material.

SAPO separates objective quality, `F(x)`, from application semantics, `H(x)`.
ECHO instantiates `H(x)` with undirected physical-edge closures and selects a
fixed-size portfolio whose members are less likely to fail together.
NSGA-III-ECHO uses ordinary single-population NSGA-III, a passive objective-only
`2N` history, and a final `N`-plan composer with a 70% objective core. The
archive never re-enters search.

## Revised result in one paragraph

The corrected strict portfolio-level rule was replayed on 320 frozen pools.
Its overall single-edge coverage gain over objective-only selection was
0.00073 (95% hierarchical interval [0.00015, 0.00155]), and over edge-Jaccard
selection was 0.00021 [0.00001, 0.00050]. The earlier larger group-wise effects
are withdrawn. In a separate small post-review operational study, ECHO retained
a surviving stored plan for 53 of 100 closure events, objective-only for 39,
and a published MO-CBS planner rerun after closure found a new plan for all
100. The five-task-pair interval for the ECHO minus objective-only rate includes
zero. Candidate supply, map family and fleet size limit the claim.

## Repository map

| Path | Purpose |
|---|---|
| [`revision/`](revision/) | current corrected code, data, protocols and paper-result map |
| [`paper/`](paper/) | paper-to-figure and paper-to-evidence navigation |
| [`results/`](results/) | historical September processed evidence |
| [`protocols/`](protocols/) | historical September protocols |
| [`configs/`](configs/) | original configuration; one island, no migration |
| [`src/`](src/) | historical source snapshot; corrected selector is in `revision/code/` |
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

Start with [`revision/README.md`](revision/README.md). The six older numbered
folders in `results/` follow the original submission and are no longer the
authority for revised numerical claims. Figures 1-6 are available in
[`paper/figures/`](paper/figures/).

## Quick verification

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e vendor
python tools/verify_release.py
PYTHONPATH=revision/code:vendor python -c "from echo_revision import select_strict_coverage_portfolio; print(select_strict_coverage_portfolio.__name__)"
```

Regenerate the inventory after an intentional release change:

```bash
python tools/refresh_manifest.py
```

## Data availability

This repository includes the revised analysis units, closure-event records,
statistics, frozen protocols, scenario inventory, figures and source. Large
candidate trajectories used to generate or replay the frozen pools remain in
the checksum-sealed local research archive and can be provided to editors and
reviewers on reasonable request. They are not mixed into the public navigation
tree. See [`revision/README.md`](revision/README.md) for the exact boundary of
what the public snapshot alone can reproduce.

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
