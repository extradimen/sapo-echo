# Protocol v1 implementation correction

After the held-out replay completed, the first downstream decision analysis
revealed that `analyze_echo_selector_decision.py` iterated over metric names
without refreshing each metric's `higher_is_better` flag. The lingering flag
from IGD inverted coverage/HV improvement signs. No portfolio selection,
failure score, empirical quality value, input, hypothesis, endpoint, threshold,
or decision gate changed. The invalid result is retained with `INVALID.md`.

The correction is limited to iterating over `METRICS.items()` in the decision
loop. Corrected script SHA-256:
`29ae527772c04886a74cdb2f93ac6a6da64edcef4e67af1dd7ca81b0f8004a3e`.
Corrected output must use a new `_decision_v2` directory.
