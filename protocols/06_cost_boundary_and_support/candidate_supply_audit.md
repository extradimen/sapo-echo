# E7 candidate-pool support audit protocol v1

Frozen before computing any pool-support result. This is a post-hoc causal
diagnostic of the already sealed E7 experiment, not a new confirmatory trial
and not a basis for changing the frozen ECHO parameters.

Execution note: audit attempt v1 stopped before producing result tables because
it imposed an invalid nested-reference-survival assertion. The empty/incomplete
v1 output is retained with `FAILURE.md`; corrected execution uses analysis id
v2 and the actual frozen ECHO quality core. Scientific definitions below are
unchanged.

## Question

For each of the 400 physical-edge closures fixed by E7, distinguish failure of
candidate generation from failure of fixed-cardinality portfolio selection.

## Inputs

- Sealed E7 v2 unit records and event identities.
- The corresponding frozen NSGA-III final-population plus passive objective
  archive pools used by the final multi-family selector analysis.
- Saved size-30 portfolios for objective-only, generic deterministic selectors,
  the two Xu-mechanism adaptations, and ECHO.
- No optimizer or path-evaluation run is repeated.

## Predefined classifications

For each event, report support in (a) the raw final-plus-archive pool and (b)
the deduplicated nondominated pool actually available to the composers.

- `generation_miss`: the actual composer pool has no plan avoiding the closed
  edge. No offline selector can succeed.
- `selection_miss`: the actual composer pool has a survivor but the selected
  ECHO portfolio has none.
- `echo_success`: the ECHO portfolio has at least one survivor.

The reactive E7 result is retained only as an end-to-end operational reference;
it is not used to relabel an unsupported offline pool as an ECHO selection
failure.

## Outputs

- Event-level pool survivor counts and saved-selector survivor counts.
- Scenario summaries of generation support and conditional selector success.
- A deterministic targeted upper-bound diagnostic using the frozen 70% ECHO
  quality core and at most nine residual selections over the ten E7 events.
- Provenance and SHA-256 manifests in a new append-only analysis directory.
