# E7 pool-support causal-classification addendum v1

This post-hoc addendum was created after the v2 support table showed that raw
evaluated-pool support and deduplicated nondominated composer-pool support can
differ. It does not alter the sealed E7 experiment or v2 support calculations.

The v2 label `generation_miss` is too coarse for causal interpretation. The
event-level values are therefore reclassified without rerunning any decoder:

- `search_supply_miss`: no saved raw evaluated candidate avoids the edge;
- `eligibility_filter_miss`: a raw candidate avoids it, but none remains after
  the frozen deduplicated nondominated eligibility filter;
- `echo_selection_miss`: the eligible composer pool contains a survivor but
  the saved ECHO portfolio does not;
- `echo_success`: the saved ECHO portfolio contains a survivor.

All four categories are mutually exclusive and exhaustive. The addendum is a
diagnostic refinement prompted by audit results, not preregistered evidence.

