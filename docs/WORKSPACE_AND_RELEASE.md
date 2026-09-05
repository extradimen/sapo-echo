# Working archive versus public release

This repository is a curated, frozen publication snapshot. It is not a mirror
of the author's local experimental workspace.

## Local experimental workspace

The private working archive retains append-only run directories, dates,
development labels, pilots, interrupted units, logs, intermediate arrays and
environment snapshots. Those names remain unchanged because analysis and
recovery scripts use them as provenance identifiers.

## Public GitHub release

The public tree contains only material needed to understand, audit and
reproduce the manuscript. Results and protocols are grouped in the same order
as the Results section. Human-readable names replace internal batch names.
`docs/SOURCE_MAP.csv` records the immutable mapping back to the original local
archive identifiers.

The publication workflow is one-way:

```text
append-only local runs
        -> completeness and scientific audit
        -> explicit source-to-public mapping
        -> curated GitHub snapshot
        -> SHA-256 manifest and fixed Git tag
        -> submission/release ZIP from that tag
```

No experiment should write directly into this repository. New evidence is
first completed and sealed locally, then appended through a new mapping and a
new public version. A published tag is never repurposed.
