# Vendored optimization dependency

`pymoo/` is the pinned source tree used for the archived experiments. It is
kept separate from the paper-facing SAPO-ECHO scripts in `src/` so readers can
distinguish the contribution from its optimization infrastructure.

The upstream project is pymoo by Julian Blank and contributors. The vendored
tree retains its Apache-2.0 license in [`LICENSE`](LICENSE). It also contains
the audited passive-archive extension and MAPF problem code needed by the
paper snapshot.
