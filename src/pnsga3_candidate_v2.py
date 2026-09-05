"""Construction point for the confirmed PNSGA-III v2 retention candidate."""

from pymoo.algorithms.moo.nsga3 import ParallelNSGA3
from pymoo.operators.sampling.space_filling import ScrambledSobolSampling
from pymoo.util.ref_dirs import get_reference_directions


def build_pnsga3_candidate_v2(problem, pop_size, *, sampling=None,
                              crossover=None, mutation=None,
                              eliminate_duplicates=None,
                              archive_direction_replicas=1,
                              archive_quality_mode="strict",
                              basin_archive_selection="directional_basin",
                              main_population_fraction=.7):
    """Build v2: frozen search plus a passive 2N retention archive.

    The archive never re-enters search.  Population, evaluation budget and
    declared output remain N; only passive history capacity is 2N.
    """
    ref_dirs = get_reference_directions("energy", problem.n_obj, pop_size, seed=1)
    operator_kwargs = {}
    if crossover is not None:
        operator_kwargs["crossover"] = crossover
    if mutation is not None:
        operator_kwargs["mutation"] = mutation
    if eliminate_duplicates is not None:
        operator_kwargs["eliminate_duplicates"] = eliminate_duplicates
    return ParallelNSGA3(
        ref_dirs, pop_size=pop_size, n_islands=4,
        main_population_fraction=main_population_fraction, migration_interval=-1,
        explorer_to_main_exchange=False, archive_update_interval=5,
        direction_replicas=2, decision_diversity=.7,
        quality_gated_diversity=True, basin_archive_size=2 * pop_size,
        exploration_memory_size=0, declared_output_size=pop_size,
        exploration_memory_local_fronts=False,
        exploration_memory_reentry=False,
        exploration_memory_objective_stratified=False,
        declared_output_local_fronts=True, basin_radius=None,
        archive_injection=False, decision_space_islands=False,
        archive_direction_replicas=archive_direction_replicas,
        archive_quality_mode=archive_quality_mode,
        basin_archive_selection=basin_archive_selection,
        sampling=ScrambledSobolSampling() if sampling is None else sampling,
        **operator_kwargs,
    )
