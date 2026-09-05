"""SAPO/ECHO constructions for failure-complementary MAPF portfolios.

``NSGA-III-ECHO`` is the paper-facing method.  The historical edge-coverage
builder names remain aliases so archived experiment configurations can still
be replayed without rewriting their provenance.
"""

from pnsga3_candidate_v2 import build_pnsga3_candidate_v2
from pymoo.algorithms.moo.nsga3 import ParallelNSGA3
from pymoo.operators.sampling.space_filling import ScrambledSobolSampling
from pymoo.util.ref_dirs import get_reference_directions


def build_nsga3_echo(problem, pop_size, *, sampling=None,
                     crossover=None, mutation=None,
                     eliminate_duplicates=None):
    """Build NSGA-III-ECHO for failure-complementary MAPF portfolios.

    Search is one ordinary NSGA-III population.  Archive capacity, archive
    admission, objective-core fraction, final cardinality and no-reentry
    semantics implement the ECHO layer. ``ParallelNSGA3`` with one island is
    used only to reuse the audited passive-archive implementation; it is not a
    multi-island search.
    """
    if not callable(getattr(problem, "failure_event_embedding", None)):
        raise ValueError(
            "NSGA-III-ECHO requires problem.failure_event_embedding"
        )
    ref_dirs = get_reference_directions("energy", problem.n_obj, pop_size, seed=1)
    operator_kwargs = {}
    if crossover is not None:
        operator_kwargs["crossover"] = crossover
    if mutation is not None:
        operator_kwargs["mutation"] = mutation
    if eliminate_duplicates is not None:
        operator_kwargs["eliminate_duplicates"] = eliminate_duplicates
    return ParallelNSGA3(
        ref_dirs,
        pop_size=pop_size,
        n_islands=1,
        main_population_fraction=None,
        migration_interval=-1,
        archive_update_interval=5,
        direction_replicas=1,
        decision_diversity=0.0,
        quality_gated_diversity=False,
        basin_archive_size=2 * pop_size,
        exploration_memory_size=0,
        declared_output_size=pop_size,
        declared_output_objective_fraction=.7,
        declared_output_local_fronts=False,
        archive_injection=False,
        decision_space_islands=False,
        archive_direction_replicas=1,
        archive_quality_mode="strict",
        basin_archive_selection="failure_coverage",
        sampling=ScrambledSobolSampling() if sampling is None else sampling,
        **operator_kwargs,
    )


def build_nsga3_objective_archive(problem, pop_size, *, sampling=None,
                                  crossover=None, mutation=None,
                                  eliminate_duplicates=None):
    """Build the passive objective-only 2N archive ablation."""
    ref_dirs = get_reference_directions("energy", problem.n_obj, pop_size, seed=1)
    operator_kwargs = {}
    if crossover is not None:
        operator_kwargs["crossover"] = crossover
    if mutation is not None:
        operator_kwargs["mutation"] = mutation
    if eliminate_duplicates is not None:
        operator_kwargs["eliminate_duplicates"] = eliminate_duplicates
    return ParallelNSGA3(
        ref_dirs,
        pop_size=pop_size,
        n_islands=1,
        main_population_fraction=None,
        migration_interval=-1,
        archive_update_interval=5,
        direction_replicas=1,
        decision_diversity=0.0,
        quality_gated_diversity=False,
        basin_archive_size=2 * pop_size,
        exploration_memory_size=0,
        declared_output_size=pop_size,
        declared_output_objective_fraction=.7,
        declared_output_local_fronts=False,
        archive_injection=False,
        decision_space_islands=False,
        archive_direction_replicas=1,
        archive_quality_mode="strict",
        basin_archive_selection="objective_only",
        sampling=ScrambledSobolSampling() if sampling is None else sampling,
        **operator_kwargs,
    )


def build_nsga3_mapf_edge_coverage(problem, pop_size, **kwargs):
    """Compatibility alias for archived ``NSGA3-EdgeCoverage`` runs."""
    return build_nsga3_echo(problem, pop_size, **kwargs)


def build_pnsga3_mapf_edge_coverage(problem, pop_size, *,
                                    main_population_fraction=.7,
                                    **operators):
    """Build PNSGA-III-EC for complementary MAPF edge-failure portfolios.

    Search is the frozen PNSGA-III v2 search. The passive 2N archive and final
    N-set use exact-objective physical-edge failure complementarity. Archived
    plans never re-enter search.
    """
    if not callable(getattr(problem, "failure_event_embedding", None)):
        raise ValueError(
            "PNSGA-III-EC requires problem.failure_event_embedding"
        )
    return build_pnsga3_candidate_v2(
        problem,
        pop_size,
        basin_archive_selection="failure_coverage",
        main_population_fraction=main_population_fraction,
        **operators,
    )
