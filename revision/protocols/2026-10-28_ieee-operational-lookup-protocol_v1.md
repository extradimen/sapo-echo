# Frozen stored-portfolio lookup timing protocol

Frozen before running on 2026-10-28. Reconstruct the five exact portfolios
from `2026-10-28_ieee-operational-replanning_v1`, verify selection identities,
and perform the precomputed Boolean edge-signature lookup 1,000 times for each
of its 100 closure event rows. Report each event's median and 95th percentile
time and the median of event medians. The timing excludes candidate generation,
portfolio composition, file loading, and dispatch; report those separately.
This is a component microbenchmark, not an end-to-end latency comparison.
