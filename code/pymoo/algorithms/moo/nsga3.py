import warnings

import numpy as np
from numpy.linalg import LinAlgError

import os
import time
import math
import copy

from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.core.survival import Survival
from pymoo.core.population import Population
try:
    from pymoo.docs import parse_doc_string
except ImportError:
    def parse_doc_string(*args, **kwargs):
        pass
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.space_filling import ScrambledSobolSampling
from pymoo.operators.selection.tournament import TournamentSelection, compare
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.functions import load_function
from pymoo.util.misc import intersect, has_feasible
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util import default_random_state


# =========================================================================================================
# Implementation
# =========================================================================================================

@default_random_state
def comp_by_cv_then_random(pop, P, random_state=None, **kwargs):
    S = np.full(P.shape[0], np.nan)

    for i in range(P.shape[0]):
        a, b = P[i, 0], P[i, 1]

        # if at least one solution is infeasible
        if pop[a].CV > 0.0 or pop[b].CV > 0.0:
            S[i] = compare(a, pop[a].CV, b, pop[b].CV, method='smaller_is_better', return_random_if_equal=True)

        # both solutions are feasible just set random
        else:
            S[i] = random_state.choice([a, b])

    return S[:, None].astype(int)


def _decision_embedding(problem, X, population_scaled=False):
    """Return a numeric embedding suitable for decision-space distances.

    Real-valued problems retain the historical normalization exactly. A
    discrete problem can expose ``decision_space_embedding`` to prevent
    arbitrary category labels from being treated as ordered Euclidean values.
    """
    X = np.asarray(X)
    transform = getattr(problem, "decision_space_embedding", None)
    if callable(transform):
        embedded = np.asarray(transform(X), dtype=float)
        if embedded.ndim != 2 or len(embedded) != len(X):
            raise ValueError("decision_space_embedding must return a 2-D row-aligned array")
        return embedded

    X = np.asarray(X, dtype=float)
    if population_scaled:
        lower, upper = np.min(X, axis=0), np.max(X, axis=0)
    else:
        lower = np.asarray(problem.xl, dtype=float)
        upper = np.asarray(problem.xu, dtype=float)
    return (X - lower) / np.where(upper > lower, upper - lower, 1.0)


class NSGA3(GeneticAlgorithm):

    def __init__(self,
                 ref_dirs,
                 pop_size=None,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                 crossover=SBX(eta=30, prob=1.0),
                 mutation=PM(eta=20),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 output=MultiObjectiveOutput(),
                 **kwargs):
        """

        Parameters
        ----------

        ref_dirs : {ref_dirs}
        pop_size : int (default = None)
            By default the population size is set to None which means that it will be equal to the number of reference
            line. However, if desired this can be overwritten by providing a positive number.
        sampling : {sampling}
        selection : {selection}
        crossover : {crossover}
        mutation : {mutation}
        eliminate_duplicates : {eliminate_duplicates}
        n_offsprings : {n_offsprings}

        """

        self.ref_dirs = ref_dirs

        # in case of R-NSGA-3 they will be None - otherwise this will be executed
        if self.ref_dirs is not None:

            if pop_size is None:
                pop_size = len(self.ref_dirs)

            if pop_size < len(self.ref_dirs):
                print(
                    f"WARNING: pop_size={pop_size} is less than the number of reference directions ref_dirs={len(self.ref_dirs)}.\n"
                    "This might cause unwanted behavior of the algorithm. \n"
                    "Please make sure pop_size is equal or larger than the number of reference directions. ")

        if 'survival' in kwargs:
            survival = kwargs['survival']
            del kwargs['survival']
        else:
            survival = ReferenceDirectionSurvival(ref_dirs)

        super().__init__(pop_size=pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         survival=survival,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         output=output,
                         advance_after_initial_infill=True,
                         **kwargs)

    def _setup(self, problem, **kwargs):

        if self.ref_dirs is not None:
            if self.ref_dirs.shape[1] != problem.n_obj:
                raise Exception(
                    "Dimensionality of reference points must be equal to the number of objectives: %s != %s" %
                    (self.ref_dirs.shape[1], problem.n_obj))

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            if len(self.survival.opt):
                self.opt = self.survival.opt


class ParallelNSGA3(NSGA3):

    def __init__(self,
                 ref_dirs,
                 pop_size=None,
                 n_islands=4,
                 migration_interval=10,
                 migration_rate=0.1,
                 sampling=FloatRandomSampling(),
                 selection=TournamentSelection(func_comp=comp_by_cv_then_random),
                 crossover=SBX(eta=30, prob=1.0),
                 mutation=PM(eta=20),
                 eliminate_duplicates=True,
                 n_offsprings=None,
                 output=MultiObjectiveOutput(),
                 focus_alpha=0.0,
                 direction_replicas=1,
                 decision_diversity=0.0,
                 quality_gated_diversity=False,
                 basin_archive_size=0,
                 basin_radius=None,
                 archive_direction_replicas=1,
                 archive_injection=False,
                 archive_injection_rate=1.0,
                 random_immigrant_rate=0.0,
                 explorer_random_immigrant_rate=0.0,
                 adaptive_explorer_immigrants=False,
                 explorer_immigrant_rate=0.2,
                 explorer_collapse_patience=3,
                 explorer_collapse_ratio=0.5,
                 explorer_novelty_rate=0.0,
                 explorer_novelty_pool_multiplier=10,
                 explorer_incubator_interval=0,
                 explorer_incubator_max_restarts=0,
                 island_mutation_etas=None,
                 adaptive_immigrants=False,
                 archive_stagnation_intervals=3,
                 decision_space_islands=False,
                 initial_island_assignment="sequential",
                 main_population_fraction=None,
                 restart_exploration_islands=False,
                 explorer_stagnation_intervals=3,
                 guided_explorer_mating=False,
                 adaptive_explorer_niches=False,
                 explorer_overlap_patience=0,
                 explorer_overlap_threshold=0.5,
                 archive_incumbent_priority=False,
                 exploration_memory_size=0,
                 declared_output_size=None,
                 declared_output_objective_fraction=None,
                 exploration_memory_local_fronts=False,
                 exploration_memory_reentry=False,
                 exploration_memory_objective_stratified=False,
                 declared_output_local_fronts=False,
                 explorer_to_main_exchange=False,
                 archive_update_interval=None,
                 archive_quality_mode="strict",
                 basin_archive_selection="directional_basin",
                 temporal_coreset_objective_fraction=0.5,
                 structure_aware_protection=False,
                 structure_observation_interval=10,
                 structure_minimum_ratio=3.0,
                 structure_persistence=3,
                 structure_protection_mode="candidate_pulse",
                 structure_aware_reproduction=False,
                 **kwargs):
        """

        Parameters
        ----------

        ref_dirs : {ref_dirs}
        pop_size : int (default = None)
            Total population size across all islands. If None, it is set to
            the number of reference directions.
        n_islands : int
            Number of parallel sub-populations (islands).
        migration_interval : int
            Perform migration every `migration_interval` generations. If set to
            None or a non-positive value, migration is disabled.
        migration_rate : float
            Fraction of each island population to migrate during a migration
            step. For example, 0.1 migrates 10%% of each island.
        direction_replicas : int
            Number of islands assigned to every global reference direction.
            Values above one allow equivalent decision-space basins to be
            explored under the same objective-space trade-off.
        decision_diversity : float
            Weight in [0, 1] given to decision-space novelty when a reference
            niche is already occupied.
        quality_gated_diversity : bool
            Reward decision-space novelty only among candidates whose
            normalized objectives are locally equivalent to the best candidate
            in the same reference niche. The equivalence radius is estimated
            from within-niche objective nearest-neighbour distances.
        basin_archive_size : int
            Maximum passive dual-space archive size. Zero disables it.
        basin_radius : float or None
            Minimum normalized decision-space separation between archived
            representatives associated with the same reference direction.
            If None, the radius is re-estimated at every archive update as the
            median within-niche nearest-neighbour distance. This uses only the
            current search population and no true Pareto-set information.
        archive_direction_replicas : int
            Capacity planning factor for the output archive. One preserves all
            search directions (default). Values above one deliberately use a
            coarser archive direction set to reserve alternatives per trade-off
            and should be treated as a separate output-resolution ablation.
        archive_injection : bool
            Whether archived solutions re-enter global environmental selection.
            False is safer because the archive is intended primarily for output.
        archive_injection_rate : float
            Fraction of the archive offered to each global exchange when
            archive injection is enabled.
        decision_space_islands : bool
            Re-form balanced islands around farthest-first decision-space
            centers after global exchange, subject to direction ownership.
        initial_island_assignment : {"sequential", "farthest"}
            Assign the initial samples in generation order, or form balanced
            decision-space regions around farthest-first centers.
        main_population_fraction : float or None
            If provided, activate asymmetric main-plus-exploration mode. Island
            zero receives this fraction of the total population and all
            reference directions; the remaining individuals form small
            exploration islands. Explorers can offer solutions to the main
            population at exchange, but the main population is never split.
        restart_exploration_islands : bool
            In main-plus-exploration mode, replace an explorer's normal
            offspring by a fresh scrambled Sobol batch after it repeatedly has
            no representative in the feasible nondominated quality archive.
            The restart consumes the island's existing offspring budget.
        explorer_stagnation_intervals : int
            Consecutive archive checks without representation before restart.
        guided_explorer_mating : bool
            On exchange generations, mate every exploration-island parent with
            a main-population guide from the same objective reference niche.
            This changes parent choice but not offspring or evaluation counts.
        adaptive_explorer_niches : bool
            At exchange generations, identify separated high-quality regions
            online and re-form the exploration islands around distinct regions.
            The true Pareto set and true number of modalities are never used.
        random_immigrant_rate : float
            Optional fraction of each island's existing offspring budget spent
            on random samples. This does not increase objective evaluations.
        adaptive_immigrants : bool
            Activate immigrants only after the archive size stagnates. This is
            experimental and disabled by default.
        explorer_novelty_rate : float
            Fraction of each exploration island's existing offspring budget
            replaced by decision-space maximin candidates. Proposals are not
            objectively evaluated before selection.
        explorer_novelty_pool_multiplier : int
            Unevaluated scrambled-Sobol proposals per novelty offspring slot.
        explorer_incubator_interval : int
            If positive, restart successive exploration islands at this fixed
            generation interval. Restarted candidates replace an existing
            explorer offspring batch and are incubated by ordinary isolated
            island evolution thereafter.
        explorer_incubator_max_restarts : int
            Maximum number of scheduled explorer-island restarts.

        Other parameters follow the standard NSGA3 interface.

        """

        self.n_islands = max(1, int(n_islands))
        self.migration_interval = migration_interval if migration_interval is not None else -1
        self.migration_rate = float(migration_rate)
        self.hall_of_fame = Population()
        # Optional per-island reference-direction tilt.
        self.focus_alpha = float(focus_alpha)
        self.direction_replicas = min(self.n_islands, max(1, int(direction_replicas)))
        self.decision_diversity = float(np.clip(decision_diversity, 0.0, 1.0))
        self.quality_gated_diversity = bool(quality_gated_diversity)
        self.basin_archive_size = max(0, int(basin_archive_size))
        self.basin_radius = None if basin_radius is None else max(0.0, float(basin_radius))
        self.archive_direction_replicas = max(1, int(archive_direction_replicas))
        self.current_basin_radius = self.basin_radius
        self.current_objective_equivalence_radius = np.nan
        self.archive_injection = bool(archive_injection)
        self.archive_injection_rate = float(np.clip(archive_injection_rate, 0.0, 1.0))
        self.random_immigrant_rate = float(np.clip(random_immigrant_rate, 0.0, 1.0))
        self.explorer_random_immigrant_rate = float(np.clip(
            explorer_random_immigrant_rate, 0.0, 1.0))
        self.adaptive_explorer_immigrants = bool(adaptive_explorer_immigrants)
        self.explorer_immigrant_rate = float(np.clip(
            explorer_immigrant_rate, 0.0, 1.0))
        self.explorer_collapse_patience = max(1, int(explorer_collapse_patience))
        self.explorer_collapse_ratio = float(np.clip(
            explorer_collapse_ratio, 0.0, 1.0))
        self.explorer_novelty_rate = float(np.clip(
            explorer_novelty_rate, 0.0, 1.0))
        self.explorer_novelty_pool_multiplier = max(
            1, int(explorer_novelty_pool_multiplier))
        self.explorer_novelty_evaluations = 0
        self.explorer_incubator_interval = max(
            0, int(explorer_incubator_interval))
        self.explorer_incubator_max_restarts = max(
            0, int(explorer_incubator_max_restarts))
        self.explorer_incubator_restarts = 0
        self.explorer_incubator_evaluations = 0
        self.explorer_immigrant_evaluations = 0
        self.explorer_immigrant_events = 0
        self._explorer_spread_max = np.zeros(self.n_islands, dtype=float)
        self._explorer_collapse_streak = np.zeros(self.n_islands, dtype=int)
        self._explorer_immigrant_next = np.zeros(self.n_islands, dtype=bool)
        if island_mutation_etas is not None and len(island_mutation_etas) != self.n_islands:
            raise ValueError("island_mutation_etas must contain one eta per island")
        self.island_mutation_etas = (
            None if island_mutation_etas is None
            else tuple(float(value) for value in island_mutation_etas))
        self.adaptive_immigrants = bool(adaptive_immigrants)
        self.archive_stagnation_intervals = max(1, int(archive_stagnation_intervals))
        self.decision_space_islands = bool(decision_space_islands)
        if initial_island_assignment not in ("sequential", "farthest"):
            raise ValueError("initial_island_assignment must be 'sequential' or 'farthest'.")
        self.initial_island_assignment = initial_island_assignment
        self.main_population_fraction = (
            None if main_population_fraction is None
            else float(np.clip(main_population_fraction, 0.5, 1.0))
        )
        self.restart_exploration_islands = bool(restart_exploration_islands)
        self.explorer_stagnation_intervals = max(1, int(explorer_stagnation_intervals))
        self.guided_explorer_mating = bool(guided_explorer_mating)
        self.guided_mating_events = 0
        self.adaptive_explorer_niches = bool(adaptive_explorer_niches)
        self.explorer_niche_reassignments = 0
        self.explorer_overlap_patience = max(0, int(explorer_overlap_patience))
        self.explorer_overlap_threshold = float(np.clip(explorer_overlap_threshold, 0, 1))
        self.explorer_overlap_streak = 0
        self.current_explorer_overlap = 0.0
        self.archive_incumbent_priority = bool(archive_incumbent_priority)
        self.exploration_memory_size = max(0, int(exploration_memory_size))
        self.declared_output_size = (
            self.basin_archive_size if declared_output_size is None
            else max(0, int(declared_output_size))
        )
        if (declared_output_objective_fraction is not None
                and not 0.0 < float(declared_output_objective_fraction) <= 1.0):
            raise ValueError(
                "declared_output_objective_fraction must be in (0, 1]")
        self.declared_output_objective_fraction = (
            None if declared_output_objective_fraction is None
            else float(declared_output_objective_fraction)
        )
        self.exploration_memory = Population()
        self.exploration_memory_local_fronts = bool(exploration_memory_local_fronts)
        self.exploration_memory_reentry = bool(exploration_memory_reentry)
        self.exploration_memory_objective_stratified = bool(
            exploration_memory_objective_stratified)
        self.declared_output_local_fronts = bool(declared_output_local_fronts)
        self.explorer_to_main_exchange = bool(explorer_to_main_exchange)
        self.explorer_to_main_exchange_events = 0
        self.archive_update_interval = (
            self.migration_interval if archive_update_interval is None
            else int(archive_update_interval)
        )
        self.archive_update_events = 0
        if archive_quality_mode not in ("strict", "local_equivalence"):
            raise ValueError("archive_quality_mode must be 'strict' or 'local_equivalence'")
        self.archive_quality_mode = archive_quality_mode
        if basin_archive_selection not in (
                "directional_basin", "temporal_coreset", "objective_only",
                "failure_coverage"):
            raise ValueError(
                "basin_archive_selection must be 'directional_basin' or "
                "'temporal_coreset' or 'objective_only' or 'failure_coverage'")
        self.basin_archive_selection = basin_archive_selection
        if not 0.0 < float(temporal_coreset_objective_fraction) <= 1.0:
            raise ValueError(
                "temporal_coreset_objective_fraction must be in (0, 1]")
        self.temporal_coreset_objective_fraction = float(
            temporal_coreset_objective_fraction)
        self.structure_aware_protection = bool(structure_aware_protection)
        self.structure_observation_interval = max(
            1, int(structure_observation_interval))
        self.structure_minimum_ratio = float(structure_minimum_ratio)
        self.structure_persistence = max(1, int(structure_persistence))
        if structure_protection_mode not in ("candidate_pulse", "anchored_safe"):
            raise ValueError(
                "structure_protection_mode must be 'candidate_pulse' or 'anchored_safe'")
        self.structure_protection_mode = structure_protection_mode
        self.structure_aware_reproduction = bool(structure_aware_reproduction)
        self._structure_tracker = None
        self._structure_anchor_threshold = 0.0
        self.structure_observation_events = 0
        self.structure_protection_events = 0
        self.structure_protected_survivors = 0
        self.structure_reproduction_events = 0
        self.structure_reserved_offspring = 0
        self.current_structure_branch_count = 0
        self.current_structure_active_count = 0
        self.current_structure_gap_ratio = 1.0
        self._explorer_stagnation = np.zeros(self.n_islands, dtype=int)
        self._restart_next = np.zeros(self.n_islands, dtype=bool)
        self.explorer_restart_counts = np.zeros(self.n_islands, dtype=int)
        self._archive_stagnation = 0
        self._previous_archive_size = 0
        self.basin_archive = Population()

        # Search keeps the full direction set. The output archive reserves
        # capacity for equivalent designs by using approximately
        # archive_size / direction_replicas well-spread objective directions.
        # This budget follows directly from the declared output capacity and
        # alternatives-per-trade-off target rather than a benchmark parameter.
        n_archive_dirs = min(
            len(ref_dirs),
            max(1, int(np.ceil(self.basin_archive_size / self.archive_direction_replicas)))
            if self.basin_archive_size > 0 else len(ref_dirs),
        )
        self._archive_ref_dirs = _farthest_direction_subset(ref_dirs, n_archive_dirs)
        # Fill the declared output budget without coupling archive resolution
        # to the number of islands that own a search direction.  With fewer
        # global directions than output slots, the spare slots are legitimate
        # decision-space alternatives for those directions.  Deliberate
        # archive compression is controlled only by
        # ``archive_direction_replicas`` above.
        self._archive_per_direction_limit = max(
            1, int(np.ceil(self.basin_archive_size / len(self._archive_ref_dirs)))
        ) if self.basin_archive_size > 0 else 1

        # Each global direction can be owned by several islands.  Replication
        # lets different decision-space basins compete under the same objective
        # trade-off, which a disjoint direction split cannot preserve.
        if self.main_population_fraction is not None:
            self._direction_owners = [tuple(range(self.n_islands)) for _ in ref_dirs]
        else:
            self._direction_owners = [
                tuple((j + replica) % self.n_islands for replica in range(self.direction_replicas))
                for j in range(len(ref_dirs))
            ]
        # Small islands should cover complementary parts of the Pareto front,
        # rather than each trying to cover every global reference direction.
        self._island_ref_dirs = []
        for island in range(self.n_islands):
            indices = [j for j, owners in enumerate(self._direction_owners) if island in owners]
            dirs = np.array(ref_dirs[indices], copy=True)
            if self.focus_alpha > 0.0 and len(dirs):
                weights = np.ones(dirs.shape[1])
                weights[np.arange(dirs.shape[1]) % self.n_islands == island] += self.focus_alpha
                dirs *= weights
                dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            self._island_ref_dirs.append(dirs)
        if any(len(dirs) == 0 for dirs in self._island_ref_dirs):
            raise ValueError("n_islands must not exceed the number of reference directions.")
        # Normalization contains generation-to-generation state, so it must not
        # be shared by independently evolving islands.
        self._island_survivals = []
        for island, dirs in enumerate(self._island_ref_dirs):
            is_main = self.main_population_fraction is not None and island == 0
            self._island_survivals.append(ReferenceDirectionSurvival(
                dirs,
                decision_diversity=0.0 if is_main else self.decision_diversity,
                quality_gated_diversity=False if is_main else self.quality_gated_diversity,
            ))

        # determine total population size (same semantics as NSGA3)
        if pop_size is None and ref_dirs is not None:
            pop_size = len(ref_dirs)

        if pop_size is None or pop_size <= 0:
            raise ValueError("pop_size must be a positive integer.")

        self.total_pop_size = int(pop_size)

        # population size per island (last island may have +1 if not divisible)
        if self.main_population_fraction is not None and self.n_islands > 1:
            main_size = int(round(self.total_pop_size * self.main_population_fraction))
            main_size = min(self.total_pop_size - (self.n_islands - 1), main_size)
            explorer_total = self.total_pop_size - main_size
            base, rem = divmod(explorer_total, self.n_islands - 1)
            self._island_sizes = [main_size] + [
                base + (1 if i < rem else 0) for i in range(self.n_islands - 1)
            ]
        else:
            base = self.total_pop_size // self.n_islands
            rem = self.total_pop_size % self.n_islands
            self._island_sizes = [base + (1 if i < rem else 0) for i in range(self.n_islands)]
        self.max_island_size = max(self._island_sizes)

        if n_offsprings is None:
            n_offsprings = self.total_pop_size

        if "survival" not in kwargs:
            kwargs["survival"] = ReferenceDirectionSurvival(
                ref_dirs,
                decision_diversity=self.decision_diversity,
                quality_gated_diversity=self.quality_gated_diversity,
            )

        super().__init__(ref_dirs=ref_dirs,
                         pop_size=self.total_pop_size,
                         sampling=sampling,
                         selection=selection,
                         crossover=crossover,
                         mutation=mutation,
                         eliminate_duplicates=eliminate_duplicates,
                         n_offsprings=n_offsprings,
                         output=output,
                         **kwargs)
        self._island_matings = [self.mating for _ in range(self.n_islands)]
        if self.island_mutation_etas is not None:
            self._island_matings = []
            for eta in self.island_mutation_etas:
                mating = copy.deepcopy(self.mating)
                mating.mutation = PM(eta=eta)
                self._island_matings.append(mating)

    def _initialize_infill(self):
        pop = super()._initialize_infill()

        # assign island ids to each individual
        island_ids = []
        for i, size in enumerate(self._island_sizes):
            island_ids.extend([i] * size)
        island_ids = np.array(island_ids, dtype=int)

        if len(island_ids) != len(pop):
            # in case initialization produced a slightly different number (due to duplicate elimination),
            # fall back to truncation or padding with last island id.
            if len(island_ids) > len(pop):
                island_ids = island_ids[:len(pop)]
            else:
                pad = np.full(len(pop) - len(island_ids), self.n_islands - 1, dtype=int)
                island_ids = np.concatenate([island_ids, pad])

        if self.main_population_fraction is not None and len(pop) >= self.n_islands:
            embedded = _decision_embedding(self.problem, pop.get("X"))
            island_ids = _main_exploration_initial_islands(
                embedded, self._island_sizes,
                np.zeros(embedded.shape[1]), np.ones(embedded.shape[1]),
                self.random_state,
            )
        elif self.initial_island_assignment == "farthest" and len(pop) >= self.n_islands:
            embedded = _decision_embedding(self.problem, pop.get("X"))
            island_ids = _balanced_farthest_islands(
                embedded, self._island_sizes,
                np.zeros(embedded.shape[1]), np.ones(embedded.shape[1]),
                self.random_state,
            )
        pop.set("island", island_ids)
        return pop

    def _infill(self):
        # generate offspring independently for each island
        off_all = []

        for island in range(self.n_islands):
            mask = self.pop.get("island") == island
            sub = self.pop[mask]
            if len(sub) == 0:
                continue

            n_off = len(sub)
            restart = (
                self.main_population_fraction is not None
                and self.n_islands > 1
                and island > 0
                and self._restart_next[island]
            )
            incubator_restart = (
                self.main_population_fraction is not None
                and self.n_islands > 1
                and island > 0
                and self.explorer_incubator_interval > 0
                and self.explorer_incubator_restarts
                    < self.explorer_incubator_max_restarts
                and self.n_iter is not None
                and self.n_iter % self.explorer_incubator_interval == 0
                and island == 1 + (
                    self.explorer_incubator_restarts % (self.n_islands - 1))
            )
            restart = restart or incubator_restart
            if restart:
                off = ScrambledSobolSampling().do(
                    self.problem, n_off, random_state=self.random_state
                )
                off.set("restart_candidate", np.ones(len(off), dtype=bool))
                self._restart_next[island] = False
                self.explorer_restart_counts[island] += 1
                if incubator_restart:
                    self.explorer_incubator_restarts += 1
                    self.explorer_incubator_evaluations += len(off)
                off.set("island", np.full(len(off), island, dtype=int))
                off_all.append(off)
                continue
            immigrant_rate = self.random_immigrant_rate
            if self.main_population_fraction is not None and island > 0:
                immigrant_rate = max(
                    immigrant_rate, self.explorer_random_immigrant_rate)
                if (self.adaptive_explorer_immigrants
                        and self._explorer_immigrant_next[island]):
                    immigrant_rate = max(immigrant_rate, self.explorer_immigrant_rate)
                    self._explorer_immigrant_next[island] = False
                    self.explorer_immigrant_events += 1
            if self.adaptive_immigrants and self._archive_stagnation < self.archive_stagnation_intervals:
                immigrant_rate = 0.0
            n_immigrants = min(n_off, int(np.floor(immigrant_rate * n_off)))
            n_novelty = 0
            if self.main_population_fraction is not None and island > 0:
                n_novelty = min(
                    n_off - n_immigrants,
                    int(np.floor(self.explorer_novelty_rate * n_off)))
            if self.main_population_fraction is not None and island > 0:
                self.explorer_immigrant_evaluations += n_immigrants + n_novelty
                self.explorer_novelty_evaluations += n_novelty
            structure_offspring = self._anchored_lineage_offspring(
                island, sub, max(0, n_off-n_immigrants-n_novelty))
            n_mated = n_off - n_immigrants - n_novelty - len(structure_offspring)

            parts = []
            if len(structure_offspring):
                parts.append(structure_offspring)
            if n_mated > 0:
                guided = (
                    self.guided_explorer_mating
                    and self.main_population_fraction is not None
                    and island > 0
                    and self.migration_interval > 0
                    and self.n_iter is not None
                    and self.n_iter % self.migration_interval == 0
                )
                if guided:
                    main = self.pop[self.pop.get("island") == 0]
                    parts.append(self._guided_explorer_offspring(sub, main, n_mated))
                    self.guided_mating_events += 1
                else:
                    parts.append(self._island_matings[island].do(
                        self.problem, sub, n_mated, algorithm=self,
                        random_state=self.random_state
                    ))
            if n_immigrants > 0:
                parts.append(FloatRandomSampling().do(
                    self.problem, n_immigrants, random_state=self.random_state
                ))
            if n_novelty > 0:
                parts.append(self._novelty_immigrants(n_novelty))
            off = parts[0] if len(parts) == 1 else Population.merge(*parts)

            if len(off) == 0:
                continue

            off.set("island", np.full(len(off), island, dtype=int))
            off.set("restart_candidate", np.zeros(len(off), dtype=bool))
            off_all.append(off)

        if not off_all:
            self.termination.force_termination = True
            return None

        # if only a single island has offspring just return it directly
        if len(off_all) == 1:
            return off_all[0]

        return Population.merge(*off_all)

    def _novelty_immigrants(self, count):
        """Choose Sobol proposals farthest from live and archived decisions."""
        proposal_count = max(count, count * self.explorer_novelty_pool_multiplier)
        proposals = ScrambledSobolSampling().do(
            self.problem, proposal_count, random_state=self.random_state)
        candidates = np.asarray(proposals.get("X"), dtype=float)
        anchors = [np.asarray(self.pop.get("X"), dtype=float)]
        if len(self.basin_archive):
            anchors.append(np.asarray(self.basin_archive.get("X"), dtype=float))
        anchors = np.vstack(anchors)
        xl, xu = np.asarray(self.problem.xl), np.asarray(self.problem.xu)
        scale = np.where(xu > xl, xu - xl, 1.0)
        candidate_n = (candidates - xl) / scale
        anchor_n = (anchors - xl) / scale
        minimum = np.min(np.linalg.norm(
            candidate_n[:, None, :] - anchor_n[None, :, :], axis=2), axis=1)
        selected = []
        for _ in range(count):
            index = int(np.argmax(minimum))
            selected.append(index)
            minimum = np.minimum(
                minimum,
                np.linalg.norm(candidate_n - candidate_n[index], axis=1))
            minimum[selected] = -np.inf
        return proposals[np.asarray(selected, dtype=int)]

    def _guided_explorer_offspring(self, explorer, main, n_offsprings):
        if len(main) == 0:
            return self.mating.do(
                self.problem, explorer, n_offsprings, algorithm=self,
                random_state=self.random_state,
            )
        n_matings = math.ceil(n_offsprings / self.mating.crossover.n_offsprings)
        local = self.mating.selection(
            self.problem, explorer, n_matings, n_parents=1,
            random_state=self.random_state, algorithm=self,
        ).reshape(-1)
        main_niches = main.get("niche")
        main_distance = main.get("dist_to_niche")
        pairs = np.empty((n_matings, 2), dtype=object)
        for k, parent in enumerate(local):
            niche = parent.get("niche")
            eligible = []
            if niche is not None and main_niches is not None:
                eligible = [i for i, value in enumerate(main_niches)
                            if value is not None and int(value) == int(niche)]
            if eligible:
                if main_distance is None:
                    guide_idx = eligible[0]
                else:
                    guide_idx = min(
                        eligible,
                        key=lambda i: float(main_distance[i])
                        if main_distance[i] is not None else np.inf,
                    )
            else:
                guide_idx = int(self.random_state.integers(len(main)))
            pairs[k, 0], pairs[k, 1] = parent, main[guide_idx]
        return self.mating.do(
            self.problem, explorer, n_offsprings, parents=pairs,
            algorithm=self, random_state=self.random_state,
        )

    def _advance(self, infills=None, **kwargs):
        _timing = os.environ.get("PYMOO_TIMING", "").strip() == "1"

        t_merge_init = 0.0
        t_survival_loop = 0.0
        t_merge_final = 0.0
        t_migrate = 0.0
        t_set_opt = 0.0

        t0 = time.perf_counter() if _timing else 0

        pop = self.pop
        if infills is not None:
            pop = Population.merge(self.pop, infills)
        self._mark_structure_representatives(pop)
        if _timing:
            t_merge_init = time.perf_counter() - t0

        t1 = time.perf_counter() if _timing else 0
        new_islands = []
        global_exchange = (
            self.migration_interval is not None
            and self.migration_interval > 0
            and self.n_iter is not None
            and self.n_iter % self.migration_interval == 0
        )
        archive_refresh = (
            self.archive_update_interval is not None
            and self.archive_update_interval > 0
            and self.n_iter is not None
            and self.n_iter % self.archive_update_interval == 0
        )

        if global_exchange and self.main_population_fraction is not None:
            # Explorers offer discoveries to the main population, which keeps
            # its full size and standard NSGA-III survival. Explorers retain
            # independent histories instead of being globally repartitioned.
            if self.explorer_to_main_exchange:
                main_pool = pop
                self.explorer_to_main_exchange_events += 1
            else:
                # Historical main-plus-exploration behavior retained for exact
                # reproduction: islands remain isolated despite the nominal
                # exchange interval.
                main_pool = pop[pop.get("island") == 0]
            main_selected = self._island_survivals[0].do(
                self.problem, main_pool, n_survive=self._island_sizes[0],
                algorithm=self, random_state=self.random_state, **kwargs,
            )
            if self.explorer_to_main_exchange:
                # An explorer representative can also survive in its source
                # island.  Separate object identity before assigning the main
                # label; otherwise setting an explorer label later mutates the
                # same Individual already stored in the main population.
                main_selected = Population([
                    individual.copy(deep=True) for individual in main_selected
                ])
            main_selected.set("island", np.zeros(len(main_selected), dtype=int))
            new_islands = [main_selected]
            explorer_groups = None
            if self.adaptive_explorer_niches and self.n_islands > 1:
                should_reassign = True
                if self.explorer_overlap_patience > 0:
                    self.current_explorer_overlap = self._explorer_overlap(pop)
                    if self.current_explorer_overlap >= self.explorer_overlap_threshold:
                        self.explorer_overlap_streak += 1
                    else:
                        self.explorer_overlap_streak = 0
                    should_reassign = self.explorer_overlap_streak >= self.explorer_overlap_patience
                if should_reassign:
                    explorer_groups = self._adaptive_explorer_groups(pop)
                    self.explorer_niche_reassignments += 1
                    self.explorer_overlap_streak = 0
            for island in range(1, self.n_islands):
                sub = (explorer_groups[island - 1] if explorer_groups is not None
                       else pop[pop.get("island") == island])
                if (self.exploration_memory_reentry
                        and len(self.exploration_memory)
                        and self.exploration_memory.has("island")):
                    memory_island = np.asarray(
                        self.exploration_memory.get("island"), dtype=int)
                    historical = self.exploration_memory[memory_island == island]
                    if len(historical):
                        sub = Population.merge(sub, historical)
                if len(sub) == 0:
                    continue
                restart_values = sub.get("restart_candidate")
                if restart_values is not None:
                    restart_mask = np.asarray([
                        bool(v) if v is not None else False for v in restart_values
                    ])
                    if np.sum(restart_mask) >= self._island_sizes[island]:
                        sub = sub[restart_mask]
                surv = self._structure_aware_island_survival(
                    island, sub, self._island_sizes[island], **kwargs)
                surv.set("island", np.full(len(surv), island, dtype=int))
                new_islands.append(surv)
        elif global_exchange:
            # Periodically perform one global environmental selection on the
            # complete parent+offspring pool.  This exchanges useful genetic
            # material without spending extra objective evaluations and avoids
            # irreversible local decisions accumulating across islands.
            exchange_pool = pop
            if self.archive_injection and self.basin_archive_size > 0 and len(self.basin_archive):
                n_inject = max(1, int(np.ceil(self.archive_injection_rate * len(self.basin_archive))))
                if n_inject < len(self.basin_archive):
                    indices = self.random_state.choice(
                        len(self.basin_archive), size=n_inject, replace=False
                    )
                    injected = self.basin_archive[indices]
                else:
                    injected = self.basin_archive
                exchange_pool = Population.merge(pop, injected)
            selected = self.survival.do(
                self.problem, exchange_pool, n_survive=self.total_pop_size,
                algorithm=self, random_state=self.random_state, **kwargs,
            )
            niches = selected.get("niche")
            assigned = self._assign_exchange_islands(selected, niches)
            selected.set("island", assigned)
            new_islands = [selected[assigned == i] for i in range(self.n_islands)]
        else:
            for island in range(self.n_islands):
                mask = pop.get("island") == island
                sub = pop[mask]
                if len(sub) == 0:
                    continue

                target_size = self._island_sizes[island]
                restart_values = sub.get("restart_candidate")
                if restart_values is not None:
                    restart_mask = np.asarray([
                        bool(v) if v is not None else False
                        for v in restart_values
                    ])
                    if np.sum(restart_mask) >= target_size:
                        sub = sub[restart_mask]
                surv = self._structure_aware_island_survival(
                    island, sub, target_size, **kwargs)
                surv.set("island", np.full(len(surv), island, dtype=int))
                new_islands.append(surv)
        if _timing:
            t_survival_loop = time.perf_counter() - t1

        if not new_islands:
            self.termination.force_termination = True
            return False

        t2 = time.perf_counter() if _timing else 0
        if len(new_islands) == 1:
            self.pop = new_islands[0]
        else:
            self.pop = Population.merge(*new_islands)
        if ((self.structure_aware_protection
             and self.structure_protection_mode == "anchored_safe")
                or self.structure_aware_reproduction):
            self._observe_stable_structure(self.pop)
        # Archive maintenance is intentionally synchronized with global
        # exchange.  Updating it every generation adds quadratic overhead but
        # contributes little because adjacent populations are highly similar.
        if self.basin_archive_size > 0 and (archive_refresh or len(self.basin_archive) == 0):
            if self.exploration_memory_size > 0:
                self._update_exploration_memory(pop)
            self._update_basin_archive(pop)
            self.archive_update_events += 1
            if archive_refresh:
                self._update_explorer_restart_state()
                self._update_adaptive_explorer_immigrant_state()
        if _timing:
            t_merge_final = time.perf_counter() - t2

        t3 = time.perf_counter() if _timing else 0
        # The global exchange above supersedes copy-based migration.  Keep
        # _migrate available for callers that explicitly invoke ring migration.
        if _timing:
            t_migrate = time.perf_counter() - t3

        t4 = time.perf_counter() if _timing else 0
        self._set_optimum()
        if _timing:
            t_set_opt = time.perf_counter() - t4
            self.data["advance_timing"] = dict(
                merge_init=t_merge_init,
                survival_loop=t_survival_loop,
                merge_final=t_merge_final,
                migrate=t_migrate,
                set_optimum=t_set_opt,
            )

        return True

    def _anchored_lineage_offspring(self, island, sub, capacity):
        """Reserve existing offspring slots for active post-survival lineages."""
        if (not self.structure_aware_reproduction or capacity <= 0
                or self._structure_tracker is None or len(sub) == 0):
            return Population()
        anchors = [branch for branch in self._structure_tracker.previous
                   if branch.active]
        if not anchors:
            return Population()
        feasible = np.ones(len(sub), dtype=bool)
        if sub.has("FEAS"):
            feasible = np.asarray(sub.get("FEAS"), dtype=bool).reshape(-1)
        ranks = sub.get("rank")
        if ranks is None:
            ranks = np.zeros(len(sub), dtype=int)
        else:
            ranks = np.asarray([
                int(value) if value is not None else np.iinfo(np.int32).max
                for value in ranks], dtype=int)
        eligible_indices = np.where(feasible & (ranks == 0))[0]
        if len(eligible_indices) == 0:
            return Population()
        embedded = _decision_embedding(
            self.problem, sub[eligible_indices].get("X"))
        centers = np.asarray([branch.centroid for branch in anchors])
        distances = np.linalg.norm(
            embedded[:, None, :]-centers[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        nearest_distance = distances[np.arange(len(embedded)), nearest]
        radii = np.asarray([
            max(self._structure_anchor_threshold, 2.0*branch.radius90,
                np.finfo(float).eps) for branch in anchors
        ])
        accepted = nearest_distance <= radii[nearest]
        offspring = []
        for anchor_index, anchor in enumerate(anchors):
            if len(offspring) >= capacity:
                break
            local = np.where(accepted & (nearest == anchor_index))[0]
            if len(local) == 0:
                continue
            ordered = local[np.argsort(nearest_distance[local])]
            first = sub[eligible_indices[ordered[0]]]
            second = (sub[eligible_indices[ordered[1]]]
                      if len(ordered) > 1 else first)
            parents = np.empty((1, 2), dtype=object)
            parents[0, 0], parents[0, 1] = first, second
            created = self._island_matings[island].do(
                self.problem, sub, 1, parents=parents, algorithm=self,
                random_state=self.random_state)
            if len(created):
                offspring.append(created)
        if not offspring:
            return Population()
        result = offspring[0] if len(offspring) == 1 \
            else Population.merge(*offspring)
        self.structure_reproduction_events += 1
        self.structure_reserved_offspring += len(result)
        return result

    def _mark_structure_representatives(self, pop):
        """Nominate persistent, nondominated decision branches in explorers.

        This hook is benchmark blind: it uses only evaluated decision vectors,
        feasibility and objective values.  It runs at the same ten-generation
        observation cadence used by the preregistered v3 shadow study.
        """
        pop.set("structure_protected", np.zeros(len(pop), dtype=bool))
        pop.set("structure_track_id", np.full(len(pop), -1, dtype=int))
        if (self.structure_aware_protection
                and self.structure_protection_mode == "anchored_safe"):
            self._mark_anchored_structure_representatives(pop)
            return
        if (not self.structure_aware_protection
                or self.n_iter is None
                or self.n_iter % self.structure_observation_interval != 0
                or len(pop) < 2):
            return

        feasible = np.ones(len(pop), dtype=bool)
        if pop.has("FEAS"):
            feasible = np.asarray(pop.get("FEAS"), dtype=bool).reshape(-1)
        feasible_indices = np.where(feasible)[0]
        if len(feasible_indices) < 2:
            return

        # Import lazily so the frozen v1/v2 path has no new runtime dependency.
        from pnsga3_structure import PersistentBranchTracker

        embedded = _decision_embedding(
            self.problem, pop[feasible_indices].get("X"))
        if self._structure_tracker is None:
            self._structure_tracker = PersistentBranchTracker(
                np.zeros(embedded.shape[1]), np.ones(embedded.shape[1]),
                minimum_ratio=self.structure_minimum_ratio,
                persistence=self.structure_persistence,
            )
        branches, _, ratio = self._structure_tracker.update(embedded)
        self.structure_observation_events += 1
        self.current_structure_branch_count = len(branches)
        self.current_structure_active_count = sum(
            branch.active for branch in branches)
        self.current_structure_gap_ratio = float(ratio)

        F = np.asarray(pop[feasible_indices].get("F"), dtype=float)
        rank = NonDominatedSorting().do(F, return_rank=True)[1]
        island = np.asarray(pop.get("island"), dtype=int)
        ideal, nadir = np.min(F, axis=0), np.max(F, axis=0)
        _, distance, _ = associate_to_niches(
            F, self.ref_dirs, ideal, nadir)

        protected = np.zeros(len(pop), dtype=bool)
        track_ids = np.full(len(pop), -1, dtype=int)
        for branch in branches:
            members = feasible_indices[branch.members]
            track_ids[members] = branch.track_id
            if not branch.active:
                continue
            local = branch.members
            eligible = local[(rank[local] == 0) & (island[members] > 0)]
            if len(eligible) == 0:
                continue
            chosen_local = int(eligible[np.argmin(distance[eligible])])
            protected[feasible_indices[chosen_local]] = True

        pop.set("structure_protected", protected)
        pop.set("structure_track_id", track_ids)

    def _observe_stable_structure(self, pop):
        """Update v3b anchors from the post-survival population only."""
        if (self.n_iter is None
                or self.n_iter % self.structure_observation_interval != 0
                or len(pop) < 2):
            return
        feasible = np.ones(len(pop), dtype=bool)
        if pop.has("FEAS"):
            feasible = np.asarray(pop.get("FEAS"), dtype=bool).reshape(-1)
        stable = pop[feasible]
        if len(stable) < 2:
            return
        from pnsga3_structure import PersistentBranchTracker
        embedded = _decision_embedding(self.problem, stable.get("X"))
        if self._structure_tracker is None:
            self._structure_tracker = PersistentBranchTracker(
                np.zeros(embedded.shape[1]), np.ones(embedded.shape[1]),
                minimum_ratio=self.structure_minimum_ratio,
                persistence=self.structure_persistence,
            )
        branches, threshold, ratio = self._structure_tracker.update(embedded)
        self._structure_anchor_threshold = float(threshold)
        self.structure_observation_events += 1
        self.current_structure_branch_count = len(branches)
        self.current_structure_active_count = sum(
            branch.active for branch in branches)
        self.current_structure_gap_ratio = float(ratio)

    def _mark_anchored_structure_representatives(self, pop):
        """Map candidates to active post-survival anchors without reclustering."""
        if self._structure_tracker is None or len(pop) == 0:
            return
        anchors = [branch for branch in self._structure_tracker.previous
                   if branch.active]
        if not anchors:
            return
        feasible = np.ones(len(pop), dtype=bool)
        if pop.has("FEAS"):
            feasible = np.asarray(pop.get("FEAS"), dtype=bool).reshape(-1)
        feasible_indices = np.where(feasible)[0]
        if len(feasible_indices) == 0:
            return
        embedded = _decision_embedding(
            self.problem, pop[feasible_indices].get("X"))
        centers = np.asarray([branch.centroid for branch in anchors])
        distances = np.linalg.norm(
            embedded[:, None, :]-centers[None, :, :], axis=2)
        nearest = np.argmin(distances, axis=1)
        nearest_distance = distances[np.arange(len(embedded)), nearest]
        radii = np.asarray([
            max(self._structure_anchor_threshold, 2.0*branch.radius90,
                np.finfo(float).eps) for branch in anchors
        ])
        accepted = nearest_distance <= radii[nearest]
        track_ids = np.full(len(pop), -1, dtype=int)
        for local_index in np.where(accepted)[0]:
            track_ids[feasible_indices[local_index]] = \
                anchors[nearest[local_index]].track_id

        F = np.asarray(pop[feasible_indices].get("F"), dtype=float)
        rank = NonDominatedSorting().do(F, return_rank=True)[1]
        island = np.asarray(pop.get("island"), dtype=int)
        ideal, nadir = np.min(F, axis=0), np.max(F, axis=0)
        _, niche_distance, _ = associate_to_niches(
            F, self.ref_dirs, ideal, nadir)
        protected = np.zeros(len(pop), dtype=bool)
        for anchor in anchors:
            local = np.where(
                (track_ids[feasible_indices] == anchor.track_id)
                & (rank == 0)
                & (island[feasible_indices] > 0))[0]
            if len(local):
                chosen = local[np.argmin(niche_distance[local])]
                protected[feasible_indices[chosen]] = True
        pop.set("structure_track_id", track_ids)
        pop.set("structure_protected", protected)

    def _structure_aware_island_survival(self, island, sub, target_size,
                                         **kwargs):
        """Reserve existing explorer slots for nominated branch exemplars."""
        survival = self._island_survivals[island]
        if not self.structure_aware_protection or island == 0:
            return survival.do(
                self.problem, sub, n_survive=target_size,
                algorithm=self, random_state=self.random_state, **kwargs)

        if self.structure_protection_mode == "anchored_safe":
            ordinary = survival.do(
                self.problem, sub, n_survive=target_size,
                algorithm=self, random_state=self.random_state, **kwargs)
            values = sub.get("structure_protected")
            protected_mask = (np.zeros(len(sub), dtype=bool) if values is None
                              else np.asarray([
                                  bool(value) if value is not None else False
                                  for value in values], dtype=bool))
            nominees = sub[protected_mask]
            replacements = 0
            for nominee in nominees:
                track = nominee.get("structure_track_id")
                survivor_tracks = np.asarray([
                    individual.get("structure_track_id")
                    if individual.get("structure_track_id") is not None else -1
                    for individual in ordinary], dtype=int)
                if np.any(survivor_tracks == track):
                    continue
                counts = {value: int(np.sum(survivor_tracks == value))
                          for value in np.unique(survivor_tracks) if value >= 0}
                donor_indices = [
                    index for index, value in enumerate(survivor_tracks)
                    if value < 0 or counts.get(value, 0) > 1
                ]
                if not donor_indices:
                    continue
                def donor_key(index):
                    rank = ordinary[index].get("rank")
                    distance = ordinary[index].get("dist_to_niche")
                    return (float(rank) if rank is not None else -1.0,
                            float(distance) if distance is not None else -1.0)
                donor = max(donor_indices, key=donor_key)
                ordinary[donor] = nominee
                replacements += 1
            if replacements:
                self.structure_protection_events += 1
                self.structure_protected_survivors += replacements
            return ordinary

        values = sub.get("structure_protected")
        if values is None:
            protected_mask = np.zeros(len(sub), dtype=bool)
        else:
            protected_mask = np.asarray([
                bool(value) if value is not None else False for value in values
            ], dtype=bool)
        protected = sub[protected_mask]
        if len(protected) == 0:
            return survival.do(
                self.problem, sub, n_survive=target_size,
                algorithm=self, random_state=self.random_state, **kwargs)

        if len(protected) > target_size:
            protected = survival.do(
                self.problem, protected, n_survive=target_size,
                algorithm=self, random_state=self.random_state, **kwargs)
        remaining_slots = target_size - len(protected)
        ordinary_pool = sub[~protected_mask]
        if remaining_slots > 0 and len(ordinary_pool):
            ordinary = survival.do(
                self.problem, ordinary_pool, n_survive=remaining_slots,
                algorithm=self, random_state=self.random_state, **kwargs)
            selected = Population.merge(protected, ordinary)
        else:
            selected = protected
        self.structure_protection_events += 1
        self.structure_protected_survivors += len(protected)
        return selected

    def _update_exploration_memory(self, candidates):
        """Keep a bounded shadow of feasible first/second-front candidates."""
        pool = candidates
        if pool.has("FEAS"):
            pool = pool[np.asarray(pool.get("FEAS")).reshape(-1)]
        if len(pool) == 0:
            return
        selected = self._exploration_memory_indices(pool)
        pool = pool[np.asarray(selected, dtype=int)]
        if len(self.exploration_memory):
            pool = Population.merge(pool, self.exploration_memory)
        keys = np.column_stack((pool.get("X"), pool.get("F")))
        _, unique = np.unique(keys, axis=0, return_index=True)
        pool = pool[np.sort(unique)]
        selected = self._exploration_memory_indices(pool)
        pool = pool[np.asarray(selected, dtype=int)]
        if len(pool) > self.exploration_memory_size:
            X = np.asarray(pool.get("X"), dtype=float)
            F = np.asarray(pool.get("F"), dtype=float)
            xscale = np.where(self.problem.xu > self.problem.xl,
                              self.problem.xu - self.problem.xl, 1.0)
            Xn = (X - self.problem.xl) / xscale
            fscale = np.where(np.ptp(F, axis=0) > 0, np.ptp(F, axis=0), 1.0)
            Fn = (F - np.min(F, axis=0)) / fscale
            if self.exploration_memory_objective_stratified:
                pool = self._stratified_exploration_memory(pool, Xn)
                self.exploration_memory = Population(
                    [individual.copy(deep=True) for individual in pool])
                return
            chosen = [int(np.argmin(np.linalg.norm(Fn, axis=1)))]
            minimum = np.linalg.norm(Xn - Xn[chosen[0]], axis=1)
            while len(chosen) < self.exploration_memory_size:
                minimum[chosen] = -np.inf
                index = int(np.argmax(minimum))
                chosen.append(index)
                minimum = np.minimum(
                    minimum, np.linalg.norm(Xn - Xn[index], axis=1))
            pool = pool[chosen]
        self.exploration_memory = Population(
            [individual.copy(deep=True) for individual in pool])

    def _stratified_exploration_memory(self, pool, Xn):
        """Spread passive memory along objectives and within decision modes."""
        n_directions = min(
            len(self.ref_dirs),
            max(1, int(np.ceil(self.exploration_memory_size / self.n_islands))),
        )
        directions = _farthest_direction_subset(self.ref_dirs, n_directions)
        F = np.asarray(pool.get("F"), dtype=float)
        ideal, nadir = np.min(F, axis=0), np.max(F, axis=0)
        niche, objective_distance, _ = associate_to_niches(
            F, directions, ideal, nadir)
        chosen = []
        for direction in range(n_directions):
            candidates = np.where(niche == direction)[0].tolist()
            if not candidates:
                continue
            first = min(candidates, key=lambda i: objective_distance[i])
            local = [first]
            candidates.remove(first)
            while candidates and len(local) < self.n_islands:
                distances = np.min(np.linalg.norm(
                    Xn[candidates, None, :] - Xn[local][None, :, :], axis=2), axis=1)
                local.append(candidates.pop(int(np.argmax(distances))))
            chosen.extend(local)
        chosen_set = set(chosen)
        remaining = [i for i in range(len(pool)) if i not in chosen_set]
        while remaining and len(chosen) < self.exploration_memory_size:
            distances = np.min(np.linalg.norm(
                Xn[remaining, None, :] - Xn[chosen][None, :, :], axis=2), axis=1)
            chosen.append(remaining.pop(int(np.argmax(distances))))
        if len(chosen) > self.exploration_memory_size:
            chosen = sorted(chosen, key=lambda i: objective_distance[i])[
                :self.exploration_memory_size]
        return pool[chosen]

    def _exploration_memory_indices(self, population):
        if not self.exploration_memory_local_fronts or not population.has("island"):
            fronts = NonDominatedSorting().do(population.get("F"))
            return (np.concatenate(fronts[:2]) if len(fronts)
                    else np.array([], dtype=int))
        island_values = np.asarray(population.get("island"), dtype=int)
        selected = []
        for island in range(1, self.n_islands):
            indices = np.where(island_values == island)[0]
            if not len(indices):
                continue
            fronts = NonDominatedSorting().do(population[indices].get("F"))
            if len(fronts):
                selected.extend(indices[np.concatenate(fronts[:2])].tolist())
        return np.asarray(selected, dtype=int)

    def _explorer_overlap(self, population):
        """Benchmark-blind overlap score based on local decision-space scale."""
        xl, xu = np.asarray(self.problem.xl), np.asarray(self.problem.xu)
        scale = np.where(xu > xl, xu - xl, 1.0)
        island_values = np.asarray(population.get("island"), dtype=int)
        groups = []
        for island in range(1, self.n_islands):
            sub = population[island_values == island]
            if sub.has("FEAS"):
                feasible = np.asarray(sub.get("FEAS")).reshape(-1).astype(bool)
                if np.any(feasible):
                    sub = sub[feasible]
            if len(sub):
                nd = NonDominatedSorting().do(sub.get("F"), only_non_dominated_front=True)
                if len(nd) >= 2:
                    sub = sub[nd]
            groups.append((np.asarray(sub.get("X"), dtype=float) - xl) / scale)
        pair_scores = []
        for i, A in enumerate(groups):
            if len(A) < 2:
                continue
            within_a = np.linalg.norm(A[:, None, :] - A[None, :, :], axis=2)
            np.fill_diagonal(within_a, np.inf)
            scale_a = np.median(np.min(within_a, axis=1))
            if not np.isfinite(scale_a) or scale_a <= 0:
                continue
            for B in groups[i + 1:]:
                if not len(B):
                    continue
                cross = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
                score_a = np.mean(np.min(cross, axis=1) <= scale_a)
                if len(B) > 1:
                    within_b = np.linalg.norm(B[:, None, :] - B[None, :, :], axis=2)
                    np.fill_diagonal(within_b, np.inf)
                    scale_b = np.median(np.min(within_b, axis=1))
                    score_b = np.mean(np.min(cross, axis=0) <= scale_b)
                else:
                    score_b = 0.0
                pair_scores.append(float((score_a + score_b) / 2))
        return max(pair_scores, default=0.0)

    def _adaptive_explorer_groups(self, population):
        """Re-form explorers around separated online decision-space regions."""
        explorer = population[np.asarray(population.get("island"), dtype=int) > 0]
        n_groups = self.n_islands - 1
        if len(explorer) < n_groups:
            return [population[population.get("island") == i]
                    for i in range(1, self.n_islands)]
        candidate = explorer
        if explorer.has("FEAS"):
            feasible = np.asarray(explorer.get("FEAS")).reshape(-1).astype(bool)
            if np.sum(feasible) >= n_groups:
                candidate = explorer[feasible]
        nd = NonDominatedSorting().do(candidate.get("F"), only_non_dominated_front=True)
        if len(nd) >= n_groups:
            candidate = candidate[nd]
        xl, xu = np.asarray(self.problem.xl), np.asarray(self.problem.xu)
        scale = np.where(xu > xl, xu - xl, 1.0)
        centers_pool = (np.asarray(candidate.get("X"), dtype=float) - xl) / scale
        centers = [int(self.random_state.integers(len(centers_pool)))]
        while len(centers) < n_groups:
            distance = np.min(np.linalg.norm(
                centers_pool[:, None, :] - centers_pool[centers][None, :, :], axis=2
            ), axis=1)
            distance[centers] = -np.inf
            centers.append(int(np.argmax(distance)))
        center_x = centers_pool[centers]
        X = (np.asarray(explorer.get("X"), dtype=float) - xl) / scale
        distances = np.linalg.norm(X[:, None, :] - center_x[None, :, :], axis=2)
        capacities = np.asarray(self._island_sizes[1:], dtype=int) * 2
        assigned = np.full(len(explorer), -1, dtype=int)
        for index in np.argsort(np.min(distances, axis=1)):
            choices = np.where(capacities > 0)[0]
            if not len(choices):
                choices = np.arange(n_groups)
            group = int(choices[np.argmin(distances[index, choices])])
            assigned[index] = group
            capacities[group] -= 1
        groups = []
        for group in range(n_groups):
            subset = explorer[assigned == group]
            target = self._island_sizes[group + 1]
            if len(subset) < target:
                nearest = np.argsort(distances[:, group])[:target]
                subset = explorer[nearest]
            subset.set("island", np.full(len(subset), group + 1, dtype=int))
            groups.append(subset)
        return groups

    def _update_explorer_restart_state(self):
        if not (self.main_population_fraction is not None
                and self.restart_exploration_islands):
            return
        represented = set()
        if len(self.basin_archive) and self.basin_archive.has("island"):
            represented = {
                int(v) for v in self.basin_archive.get("island")
                if v is not None and int(v) > 0
            }
        for island in range(1, self.n_islands):
            if island in represented:
                self._explorer_stagnation[island] = 0
            else:
                self._explorer_stagnation[island] += 1
                if self._explorer_stagnation[island] >= self.explorer_stagnation_intervals:
                    self._restart_next[island] = True
                    self._explorer_stagnation[island] = 0

    def _update_adaptive_explorer_immigrant_state(self):
        """Schedule a small exploration pulse after island-local collapse.

        Spread is the median normalized decision-space nearest-neighbour
        distance of the island's feasible population.  Each island is compared
        only with its own historical maximum; neither other islands nor a true
        Pareto set influence the trigger.
        """
        if not (self.main_population_fraction is not None
                and self.adaptive_explorer_immigrants):
            return
        xl, xu = np.asarray(self.problem.xl, dtype=float), np.asarray(self.problem.xu, dtype=float)
        scale = np.where(xu > xl, xu - xl, 1.0)
        island_values = np.asarray(self.pop.get("island"), dtype=int)
        for island in range(1, self.n_islands):
            sub = self.pop[island_values == island]
            if sub.has("FEAS"):
                feasible = np.asarray(sub.get("FEAS")).reshape(-1).astype(bool)
                if np.sum(feasible) >= 2:
                    sub = sub[feasible]
            if len(sub) < 2:
                spread = 0.0
            else:
                X = (np.asarray(sub.get("X"), dtype=float) - xl) / scale
                distances = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
                np.fill_diagonal(distances, np.inf)
                spread = float(np.median(np.min(distances, axis=1)))
            previous_max = self._explorer_spread_max[island]
            if spread > previous_max:
                self._explorer_spread_max[island] = spread
                self._explorer_collapse_streak[island] = 0
                continue
            collapsed = (previous_max > 0.0
                         and spread < self.explorer_collapse_ratio * previous_max)
            self._explorer_collapse_streak[island] = (
                self._explorer_collapse_streak[island] + 1 if collapsed else 0)
            if self._explorer_collapse_streak[island] >= self.explorer_collapse_patience:
                self._explorer_immigrant_next[island] = True
                self._explorer_collapse_streak[island] = 0

    def _assign_exchange_islands(self, selected, niches):
        """Assign exchange survivors while preserving island capacities.

        In basin mode, farthest-first centers make islands occupy separated
        decision regions.  Direction ownership remains a hard eligibility
        constraint, so objective-space coverage is not discarded.
        """
        assigned = np.full(len(selected), -1, dtype=int)
        remaining = np.array(self._island_sizes, dtype=int)
        if not self.decision_space_islands or len(selected) < self.n_islands:
            order = np.arange(len(selected))
            centers = None
        else:
            X = np.asarray(selected.get("X"), dtype=float)
            xl, xu = np.asarray(self.problem.xl, dtype=float), np.asarray(self.problem.xu, dtype=float)
            X = (X - xl) / np.where(xu > xl, xu - xl, 1.0)
            center_indices = [int(self.random_state.integers(len(X)))]
            while len(center_indices) < self.n_islands:
                distance = np.min(
                    np.linalg.norm(X[:, None, :] - X[center_indices][None, :, :], axis=2),
                    axis=1,
                )
                distance[center_indices] = -np.inf
                center_indices.append(int(np.argmax(distance)))
            centers = X[center_indices]
            center_distance = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
            # Assign strongly separated points first; ambiguous central points
            # can then fill whichever eligible island still has capacity.
            order = np.argsort(-np.min(center_distance, axis=1))

        for idx in order:
            niche = niches[idx]
            if niche is None or (isinstance(niche, (float, np.floating)) and np.isnan(niche)):
                # Constraint handling can retain infeasible solutions before
                # the feasible region is filled. Such solutions are not
                # associated with a reference direction yet and may enter any
                # island with remaining capacity.
                candidates = range(self.n_islands)
            else:
                candidates = self._direction_owners[int(niche)]
            available = [i for i in candidates if remaining[i] > 0]
            if not available:
                available = np.where(remaining > 0)[0].tolist()
            if centers is None:
                island = max(available, key=lambda i: remaining[i])
            else:
                island = min(available, key=lambda i: np.linalg.norm(X[idx] - centers[i]))
            assigned[idx] = island
            remaining[island] -= 1
        return assigned

    def _update_basin_archive(self, candidates):
        """Keep objective-direction elites from distinct decision-space basins."""
        candidates.set("archive_incumbent", np.zeros(len(candidates), dtype=bool))
        if len(self.basin_archive):
            self.basin_archive.set(
                "archive_incumbent", np.ones(len(self.basin_archive), dtype=bool))
            pool = Population.merge(candidates, self.basin_archive)
        else:
            pool = candidates
        if pool.has("FEAS"):
            pool = pool[np.asarray(pool.get("FEAS")).reshape(-1)]
        if len(pool) == 0:
            self.basin_archive = Population()
            return

        identity = getattr(self.problem, "decision_identity", None)
        if callable(identity):
            keys = np.asarray(identity(pool.get("X")), dtype=str)
            _, unique = np.unique(keys, return_index=True)
        else:
            _, unique = np.unique(np.asarray(pool.get("X")), axis=0, return_index=True)
        pool = pool[np.sort(unique)]

        # IGDX and Pareto-set coverage are meaningful only for converged
        # solutions.  Never let merely decision-novel dominated points inflate
        # the multimodal archive or its reported coverage.
        if self.archive_quality_mode == "strict":
            non_dominated = NonDominatedSorting().do(
                pool.get("F"), only_non_dominated_front=True
            )
            pool = pool[non_dominated]
        if self.basin_archive_selection == "temporal_coreset":
            self._update_temporal_coreset_archive(pool)
            return
        if self.basin_archive_selection == "objective_only":
            self._update_objective_only_archive(pool)
            return
        if self.basin_archive_selection == "failure_coverage":
            self._update_failure_coverage_archive(pool)
            return
        F = pool.get("F")
        ideal, nadir = np.min(F, axis=0), np.max(F, axis=0)
        niche, objective_distance, _ = associate_to_niches(
            F, self._archive_ref_dirs, ideal, nadir
        )
        Fn = (F - ideal) / np.where(nadir > ideal, nadir - ideal, 1.0)
        direction_norm = self._archive_ref_dirs / np.maximum(
            np.linalg.norm(self._archive_ref_dirs, axis=1, keepdims=True), 1e-12
        )
        convergence = np.sum(Fn * direction_norm[niche], axis=1) + objective_distance

        Xn = _decision_embedding(self.problem, pool.get("X"))
        effective_radius = self._estimate_basin_radius(Xn, niche)
        self.current_basin_radius = effective_radius
        chosen = []
        basin_ids = np.full(len(pool), -1, dtype=int)
        per_direction_limit = self._archive_per_direction_limit
        objective_radii = []

        for direction in range(len(self._archive_ref_dirs)):
            indices = np.where(niche == direction)[0]
            indices = sorted(
                indices,
                key=lambda i: (
                    convergence[i] if self.archive_quality_mode == "local_equivalence"
                    else objective_distance[i]
                ),
            )
            if not indices:
                continue
            objective_radius = _median_positive_nearest_distance(Fn[indices])
            objective_radii.append(objective_radius)
            best = indices[0]
            if self.archive_incumbent_priority:
                incumbents = np.asarray(pool.get("archive_incumbent"), dtype=bool)
                indices = sorted(
                    indices,
                    key=lambda i: (not incumbents[i], objective_distance[i]),
                )
            centers = []
            for idx in indices:
                if (self.quality_gated_diversity and idx != best
                        and np.linalg.norm(Fn[idx] - Fn[best]) > objective_radius + 1e-12):
                    continue
                if centers:
                    distances = np.linalg.norm(Xn[idx] - Xn[centers], axis=1)
                    if np.min(distances) < effective_radius:
                        continue
                basin_ids[idx] = len(centers)
                centers.append(idx)
                chosen.append(idx)
                if len(centers) >= per_direction_limit:
                    break

        self.current_objective_equivalence_radius = (
            float(np.median(objective_radii)) if objective_radii else 0.0
        )

        chosen = sorted(
            chosen,
            key=lambda i: (
                convergence[i] if self.archive_quality_mode == "local_equivalence"
                else objective_distance[i]
            ),
        )[:self.basin_archive_size]
        archive = pool[chosen]
        archive.set("global_niche", niche[chosen], "basin", basin_ids[chosen])
        self.basin_archive = Population([individual.copy(deep=True) for individual in archive])
        if len(self.basin_archive) > self._previous_archive_size:
            self._archive_stagnation = 0
        else:
            self._archive_stagnation += 1
        self._previous_archive_size = len(self.basin_archive)

    def _update_objective_only_archive(self, pool):
        """Keep a deterministic 2N coreset using objective information only."""
        capacity = min(self.basin_archive_size, len(pool))
        if capacity <= 0:
            self.basin_archive = Population()
            return
        F = np.asarray(pool.get("F"), dtype=float)
        ideal, nadir = np.min(F, axis=0), np.max(F, axis=0)
        niche, objective_distance, _ = associate_to_niches(
            F, self.ref_dirs, ideal, nadir)
        Fn = (F - ideal) / np.where(nadir > ideal, nadir - ideal, 1.0)
        selected = []
        for direction in range(len(self.ref_dirs)):
            indices = np.where(niche == direction)[0]
            if len(indices):
                selected.append(int(indices[np.argmin(objective_distance[indices])]))
        if len(selected) > capacity:
            selected = sorted(
                selected, key=lambda index: (objective_distance[index], index)
            )[:capacity]
        selected_set = set(selected)
        remaining = [index for index in range(len(pool)) if index not in selected_set]
        while remaining and len(selected) < capacity:
            if selected:
                distances = np.min(np.linalg.norm(
                    Fn[remaining, None, :] - Fn[selected][None, :, :], axis=2), axis=1)
                position = max(
                    range(len(remaining)),
                    key=lambda pos: (distances[pos], -remaining[pos]),
                )
            else:
                position = min(
                    range(len(remaining)),
                    key=lambda pos: (np.linalg.norm(Fn[remaining[pos]]), remaining[pos]),
                )
            selected.append(remaining.pop(position))
        selected = np.asarray(selected, dtype=int)
        archive = pool[selected]
        archive.set(
            "global_niche", niche[selected],
            "basin", np.full(len(archive), -1, dtype=int),
        )
        self.current_basin_radius = np.nan
        self.current_objective_equivalence_radius = np.nan
        self.basin_archive = Population(
            [individual.copy(deep=True) for individual in archive])
        if len(self.basin_archive) > self._previous_archive_size:
            self._archive_stagnation = 0
        else:
            self._archive_stagnation += 1
        self._previous_archive_size = len(self.basin_archive)

    def _update_temporal_coreset_archive(self, pool):
        """Bound passive history with equal objective and decision duties.

        The first half of the 2N capacity is a deterministic objective core,
        with at most one closest representative per unchanged search reference
        direction. Any unfilled core slots maximize normalized objective-space
        distance. The second half uses normalized decision-space farthest-first
        selection. No selection here consumes the optimizer random stream or
        feeds an archived individual back into search.
        """
        capacity = min(self.basin_archive_size, len(pool))
        if capacity <= 0:
            self.basin_archive = Population()
            return
        F = np.asarray(pool.get("F"), dtype=float)
        ideal, nadir = np.min(F, axis=0), np.max(F, axis=0)
        niche, objective_distance, _ = associate_to_niches(
            F, self.ref_dirs, ideal, nadir)
        Fn = (F - ideal) / np.where(nadir > ideal, nadir - ideal, 1.0)
        objective_capacity = min(
            capacity, int(np.ceil(
                self.basin_archive_size
                * self.temporal_coreset_objective_fraction)))
        selected = []
        for direction in range(len(self.ref_dirs)):
            indices = np.where(niche == direction)[0]
            if len(indices):
                selected.append(int(indices[np.argmin(objective_distance[indices])]))
        if len(selected) > objective_capacity:
            selected = sorted(
                selected, key=lambda index: objective_distance[index]
            )[:objective_capacity]
        selected_set = set(selected)
        remaining = [index for index in range(len(pool)) if index not in selected_set]
        while remaining and len(selected) < objective_capacity:
            if selected:
                distances = np.min(np.linalg.norm(
                    Fn[remaining, None, :] - Fn[selected][None, :, :], axis=2), axis=1)
                position = int(np.argmax(distances))
            else:
                position = int(np.argmin(np.linalg.norm(Fn[remaining], axis=1)))
            selected.append(remaining.pop(position))

        Xn = _decision_embedding(self.problem, pool.get("X"))
        while remaining and len(selected) < capacity:
            distances = np.min(np.linalg.norm(
                Xn[remaining, None, :] - Xn[selected][None, :, :], axis=2), axis=1)
            selected.append(remaining.pop(int(np.argmax(distances))))

        archive = pool[np.asarray(selected, dtype=int)]
        archive.set(
            "global_niche", niche[np.asarray(selected, dtype=int)],
            "basin", np.full(len(archive), -1, dtype=int),
        )
        self.current_basin_radius = np.nan
        self.current_objective_equivalence_radius = np.nan
        self.basin_archive = Population(
            [individual.copy(deep=True) for individual in archive])
        if len(self.basin_archive) > self._previous_archive_size:
            self._archive_stagnation = 0
        else:
            self._archive_stagnation += 1
        self._previous_archive_size = len(self.basin_archive)

    def _failure_coverage_fill(self, pool, selected, candidates, capacity):
        """Greedily add plans that reduce shared exact-cost edge failures."""
        transform = getattr(self.problem, "failure_event_embedding", None)
        if not callable(transform):
            raise ValueError(
                "failure_coverage requires problem.failure_event_embedding")
        events = np.asarray(transform(pool.get("X")), dtype=bool)
        F = np.asarray(pool.get("F"), dtype=float)
        selected = list(selected)
        candidates = list(candidates)

        def same_group(left, right):
            return np.allclose(F[left], F[right], atol=1e-12, rtol=0.0)

        def realized_gain(index):
            group = [other for other in selected if same_group(index, other)]
            if not group:
                return 0
            before_union = np.any(events[group], axis=0)
            before_common = np.all(events[group], axis=0)
            before = np.sum(before_union & ~before_common)
            after = events[group + [index]]
            after_score = np.sum(np.any(after, axis=0) & ~np.all(after, axis=0))
            return int(after_score - before)

        def pair_potential(index):
            peers = [other for other in candidates
                     if other != index and same_group(index, other)]
            return max((int(np.sum(events[index] != events[other]))
                        for other in peers), default=0)

        while candidates and len(selected) < capacity:
            scores = [
                (realized_gain(index), pair_potential(index), -index)
                for index in candidates
            ]
            position = max(range(len(candidates)), key=lambda pos: scores[pos])
            selected.append(candidates.pop(position))
        return selected

    def _update_failure_coverage_archive(self, pool):
        """Retain an objective core plus exact-cost failure-complementary plans."""
        capacity = min(self.basin_archive_size, len(pool))
        if capacity <= 0:
            self.basin_archive = Population()
            return
        F = np.asarray(pool.get("F"), dtype=float)
        ideal, nadir = np.min(F, axis=0), np.max(F, axis=0)
        niche, objective_distance, _ = associate_to_niches(
            F, self.ref_dirs, ideal, nadir)
        Fn = (F - ideal) / np.where(nadir > ideal, nadir - ideal, 1.0)
        objective_capacity = min(
            capacity, int(np.ceil(
                self.basin_archive_size
                * self.temporal_coreset_objective_fraction)))
        selected = []
        for direction in range(len(self.ref_dirs)):
            indices = np.where(niche == direction)[0]
            if len(indices):
                selected.append(int(indices[np.argmin(objective_distance[indices])]))
        if len(selected) > objective_capacity:
            selected = sorted(
                selected, key=lambda index: objective_distance[index]
            )[:objective_capacity]
        selected_set = set(selected)
        remaining = [index for index in range(len(pool)) if index not in selected_set]
        while remaining and len(selected) < objective_capacity:
            if selected:
                distances = np.min(np.linalg.norm(
                    Fn[remaining, None, :] - Fn[selected][None, :, :], axis=2), axis=1)
                position = int(np.argmax(distances))
            else:
                position = int(np.argmin(np.linalg.norm(Fn[remaining], axis=1)))
            selected.append(remaining.pop(position))
        selected = self._failure_coverage_fill(
            pool, selected, remaining, capacity)
        archive = pool[np.asarray(selected, dtype=int)]
        archive.set(
            "global_niche", niche[np.asarray(selected, dtype=int)],
            "basin", np.full(len(archive), -1, dtype=int),
        )
        self.current_basin_radius = np.nan
        self.current_objective_equivalence_radius = 0.0
        self.basin_archive = Population(
            [individual.copy(deep=True) for individual in archive])
        if len(self.basin_archive) > self._previous_archive_size:
            self._archive_stagnation = 0
        else:
            self._archive_stagnation += 1
        self._previous_archive_size = len(self.basin_archive)

    def _preserve_main_output_core(self):
        """Compose a bounded quality core plus decision-diverse alternatives.

        In asymmetric mode the main-island capacity defines the size of an
        objective-quality core selected by ordinary NSGA-III survival from the
        feasible nondominated union.  Remaining output slots are filled from
        that union by farthest-first decision-space selection.  The allocation
        follows the frozen island sizes and introduces no benchmark-specific
        quota or true-solution information.
        """
        if len(self.pop) == 0 or self.declared_output_size <= 0:
            return

        current = self.pop
        if current.has("FEAS"):
            current = current[np.asarray(current.get("FEAS")).reshape(-1)]
        if len(current) == 0:
            return

        combined = Population.merge(current, self.basin_archive)
        if len(self.exploration_memory):
            combined = Population.merge(combined, self.exploration_memory)
        identity = getattr(self.problem, "decision_identity", None)
        if self.basin_archive_selection == "failure_coverage" and callable(identity):
            keys = np.asarray(identity(combined.get("X")), dtype=str)
            _, unique = np.unique(keys, return_index=True)
        else:
            keys = np.column_stack((combined.get("X"), combined.get("F")))
            _, unique = np.unique(keys, axis=0, return_index=True)
        combined = combined[np.sort(unique)]
        if self.declared_output_local_fronts and combined.has("island"):
            island_values = np.asarray(combined.get("island"), dtype=int)
            local = []
            for island in np.unique(island_values):
                indices = np.where(island_values == island)[0]
                nd = NonDominatedSorting().do(
                    combined[indices].get("F"), only_non_dominated_front=True)
                local.extend(indices[nd].tolist())
            combined = combined[np.asarray(local, dtype=int)]
        else:
            nd = NonDominatedSorting().do(
                combined.get("F"), only_non_dominated_front=True
            )
            combined = combined[nd]

        if self.declared_output_objective_fraction is None:
            objective_slots = self._island_sizes[0]
        else:
            objective_slots = int(round(
                self.declared_output_objective_fraction
                * self.declared_output_size))
        quality_size = min(
            objective_slots, self.declared_output_size, len(combined)
        )
        survival = ReferenceDirectionSurvival(self.ref_dirs)
        core = survival.do(
            self.problem, combined, n_survive=quality_size,
            algorithm=self, random_state=self.random_state,
        )
        core_keys = {
            tuple(row) for row in np.column_stack((core.get("X"), core.get("F")))
        }
        combined_keys = np.column_stack((combined.get("X"), combined.get("F")))
        is_core = np.array([tuple(row) in core_keys for row in combined_keys])
        selected = np.where(is_core)[0].tolist()
        self._last_output_quality_core_keys = core_keys

        candidates = np.where(~is_core)[0].tolist()
        if self.basin_archive_selection == "failure_coverage":
            selected = self._failure_coverage_fill(
                combined, selected, candidates, self.declared_output_size)
        else:
            Xn = _decision_embedding(self.problem, combined.get("X"))
            while candidates and len(selected) < self.declared_output_size:
                if selected:
                    distances = np.min(
                        np.linalg.norm(
                            Xn[candidates, None, :] - Xn[selected][None, :, :],
                            axis=2,
                        ),
                        axis=1,
                    )
                    position = int(np.argmax(distances))
                else:
                    position = 0
                selected.append(candidates.pop(position))

        output = combined[selected]
        self.basin_archive = Population(
            [individual.copy(deep=True) for individual in output]
        )

    def _estimate_basin_radius(self, Xn, niche):
        """Return a separation scale without using a known Pareto set.

        Nearest-neighbour distances are calculated only between solutions
        associated with the same objective-space reference direction. This
        avoids treating movement along the Pareto front as evidence of a new
        decision-space basin. Exact duplicate distances are ignored because
        the update pool can contain both an archived copy and its survivor.
        """
        if self.basin_radius is not None:
            return self.basin_radius

        nearest = []
        for direction in np.unique(niche):
            indices = np.where(niche == direction)[0]
            if len(indices) < 2:
                continue
            points = Xn[indices]
            distances = np.linalg.norm(
                points[:, None, :] - points[None, :, :], axis=2
            )
            distances[distances <= np.finfo(float).eps] = np.inf
            local = np.min(distances, axis=1)
            nearest.extend(local[np.isfinite(local)])

        if not nearest:
            return 0.0
        return float(np.median(nearest))

    def _migrate(self):
        if self.migration_rate <= 0.0 or self.n_islands <= 1:
            return

        migrants = []
        destinations = []

        # Send each island's elites to its neighbour in a deterministic ring.
        for island in range(self.n_islands):
            mask = self.pop.get("island") == island
            sub = self.pop[mask]
            if len(sub) == 0:
                continue

            island_size = len(sub)
            n_mig = max(1, int(np.floor(self.migration_rate * island_size)))

            order = np.concatenate(NonDominatedSorting().do(sub.get("F")))
            migrants.append(sub[order[:n_mig]])

            dst_island = (island + 1) % self.n_islands
            dst_mask = self.pop.get("island") == dst_island
            dst_sub = self.pop[dst_mask]
            dst_order = np.concatenate(NonDominatedSorting().do(dst_sub.get("F")))
            n_replace = min(n_mig, len(dst_sub))
            dst_global = np.where(dst_mask)[0]
            destinations.append((dst_island, dst_global[dst_order[-n_replace:]]))

        if not migrants or not destinations:
            return

        for source, (dst_island, dst_indices) in zip(migrants, destinations):
            for dst, individual in zip(dst_indices, source):
                migrant = individual.copy(deep=True)
                migrant.set("island", dst_island)
                self.pop[dst] = migrant

    def _set_optimum(self, **kwargs):
        """Expose the non-dominated union, not stale state from one island."""
        feasible = self.pop.get("FEAS")[:, 0]
        if not np.any(feasible):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            pop = self.pop[feasible]
            self.opt = pop[NonDominatedSorting().do(pop.get("F"), only_non_dominated_front=True)]


# =========================================================================================================
# Survival
# =========================================================================================================


class ReferenceDirectionSurvival(Survival):

    def __init__(self, ref_dirs, decision_diversity=0.0,
                 quality_gated_diversity=False):
        super().__init__(filter_infeasible=True)
        self.ref_dirs = ref_dirs
        self.decision_diversity = float(np.clip(decision_diversity, 0.0, 1.0))
        self.quality_gated_diversity = bool(quality_gated_diversity)
        self.opt = None
        self.norm = HyperplaneNormalization(ref_dirs.shape[1])

    def _do(self, problem, pop, n_survive, D=None, random_state=None, **kwargs):

        # optional: survival step timing when PYMOO_TIMING=1
        _timing = os.environ.get("PYMOO_TIMING", "").strip() == "1"
        _algo = kwargs.get("algorithm")

        # attributes to be set after the survival
        # Optional objective override for specialized survival callers.
        F_override = kwargs.get("F_override", None)
        if F_override is not None:
            F = np.asarray(F_override)
        else:
            F = pop.get("F")

        # calculate the fronts of the population
        t0 = time.perf_counter() if _timing else 0
        fronts, rank = NonDominatedSorting().do(F, return_rank=True, n_stop_if_ranked=n_survive)
        t_nds = (time.perf_counter() - t0) if _timing else 0
        non_dominated, last_front = fronts[0], fronts[-1]

        # update the hyperplane based boundary estimation
        hyp_norm = self.norm
        t1 = time.perf_counter() if _timing else 0
        hyp_norm.update(F, nds=non_dominated)
        t_norm = (time.perf_counter() - t1) if _timing else 0
        ideal, nadir = hyp_norm.ideal_point, hyp_norm.nadir_point

        #  consider only the population until we come to the splitting front
        I = np.concatenate(fronts)
        pop, rank, F = pop[I], rank[I], F[I]

        # update the front indices for the current population
        counter = 0
        for i in range(len(fronts)):
            for j in range(len(fronts[i])):
                fronts[i][j] = counter
                counter += 1
        last_front = fronts[-1]

        # associate individuals to niches
        t2 = time.perf_counter() if _timing else 0
        niche_of_individuals, dist_to_niche, dist_matrix = \
            associate_to_niches(F, self.ref_dirs, ideal, nadir)
        t_assoc = (time.perf_counter() - t2) if _timing else 0

        # attributes of a population
        pop.set('rank', rank,
                'niche', niche_of_individuals,
                'dist_to_niche', dist_to_niche)

        # set the optimum, first front and closest to all reference directions
        closest = np.unique(dist_matrix[:, np.unique(niche_of_individuals)].argmin(axis=0))
        self.opt = pop[intersect(fronts[0], closest)]
        if len(self.opt) == 0:
            self.opt = pop[fronts[0]]

        # if we need to select individuals to survive
        if len(pop) > n_survive:

            # if there is only one front
            if len(fronts) == 1:
                n_remaining = n_survive
                until_last_front = np.array([], dtype=int)
                niche_count = np.zeros(len(self.ref_dirs), dtype=int)

            # if some individuals already survived
            else:
                until_last_front = np.concatenate(fronts[:-1])
                niche_count = calc_niche_count(len(self.ref_dirs), niche_of_individuals[until_last_front])
                n_remaining = n_survive - len(until_last_front)

            t3 = time.perf_counter() if _timing else 0
            X = _decision_embedding(problem, pop.get("X"), population_scaled=True)
            Fn = (F - ideal) / np.where(nadir > ideal, nadir - ideal, 1.0)
            S = niching(
                pop[last_front], n_remaining, niche_count,
                niche_of_individuals[last_front], dist_to_niche[last_front],
                random_state=random_state,
                X=X[last_front] if len(X) else None,
                base_X=X[until_last_front] if len(X) else None,
                base_niches=niche_of_individuals[until_last_front],
                decision_diversity=self.decision_diversity,
                objective_F=Fn[last_front],
                base_objective_F=Fn[until_last_front],
                quality_gated_diversity=self.quality_gated_diversity,
            )
            t_niching = (time.perf_counter() - t3) if _timing else 0

            survivors = np.concatenate((until_last_front, last_front[S].tolist()))
            pop = pop[survivors]
        else:
            t_niching = 0.0 if _timing else 0

        if _timing and _algo is not None:
            st = _algo.data.get("survival_timing_last")
            if st is not None:
                st["nds"] += t_nds
                st["norm"] += t_norm
                st["associate"] += t_assoc
                st["niching"] += t_niching
            else:
                _algo.data["survival_timing_last"] = dict(nds=t_nds, norm=t_norm, associate=t_assoc, niching=t_niching)

        return pop


@default_random_state
def niching(pop, n_remaining, niche_count, niche_of_individuals, dist_to_niche,
            random_state=None, X=None, base_X=None, base_niches=None,
            decision_diversity=0.0, objective_F=None, base_objective_F=None,
            quality_gated_diversity=False):
    survivors = []

    # boolean array of elements that are considered for each iteration
    mask = np.full(len(pop), True)

    while len(survivors) < n_remaining:

        # number of individuals to select in this iteration
        n_select = n_remaining - len(survivors)

        # all niches where new individuals can be assigned to and the corresponding niche count
        next_niches_list = np.unique(niche_of_individuals[mask])
        next_niche_count = niche_count[next_niches_list]

        # the minimum niche count
        min_niche_count = next_niche_count.min()

        # all niches with the minimum niche count (truncate randomly if there are more niches than remaining individuals)
        next_niches = next_niches_list[np.where(next_niche_count == min_niche_count)[0]]
        next_niches = next_niches[random_state.permutation(len(next_niches))[:n_select]]

        for next_niche in next_niches:

            # indices of individuals that are considered and assign to next_niche
            next_ind = np.where(np.logical_and(niche_of_individuals == next_niche, mask))[0]

            # shuffle to break random tie (equal perp. dist) or select randomly
            random_state.shuffle(next_ind)

            if niche_count[next_niche] == 0:
                next_ind = next_ind[np.argmin(dist_to_niche[next_ind])]
            elif decision_diversity > 0.0 and X is not None:
                eligible = next_ind
                if quality_gated_diversity and objective_F is not None and len(next_ind) > 1:
                    niche_objectives = [objective_F[next_ind]]
                    if base_objective_F is not None and base_niches is not None:
                        base = base_objective_F[base_niches == next_niche]
                        if len(base):
                            niche_objectives.append(base)
                    radius = _median_positive_nearest_distance(
                        np.vstack(niche_objectives)
                    )
                    best = next_ind[np.argmin(dist_to_niche[next_ind])]
                    objective_gap = np.linalg.norm(
                        objective_F[next_ind] - objective_F[best], axis=1
                    )
                    gated = next_ind[objective_gap <= radius + 1e-12]
                    if len(gated):
                        eligible = gated
                same_niche = [i for i in survivors if niche_of_individuals[i] == next_niche]
                reference_X = []
                if base_X is not None and base_niches is not None:
                    reference_X.extend(base_X[base_niches == next_niche])
                if same_niche:
                    reference_X.extend(X[same_niche])
                if reference_X:
                    reference_X = np.asarray(reference_X)
                    novelty = np.min(
                        np.linalg.norm(X[eligible, None, :] - reference_X[None, :, :], axis=2),
                        axis=1,
                    )
                    convergence = dist_to_niche[eligible]
                    novelty /= max(np.max(novelty), 1e-12)
                    convergence /= max(np.max(convergence), 1e-12)
                    score = decision_diversity * novelty - (1.0 - decision_diversity) * convergence
                    next_ind = eligible[np.argmax(score)]
                else:
                    next_ind = eligible[np.argmin(dist_to_niche[eligible])]
            else:
                # already randomized through shuffling
                next_ind = next_ind[0]

            # add the selected individual to the survivors
            mask[next_ind] = False
            survivors.append(int(next_ind))

            # increase the corresponding niche count
            niche_count[next_niche] += 1

    return survivors


def _median_positive_nearest_distance(points):
    """Estimate a local equivalence scale without a known PF or PS."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    distances[distances <= np.finfo(float).eps] = np.inf
    nearest = np.min(distances, axis=1)
    nearest = nearest[np.isfinite(nearest)]
    return float(np.median(nearest)) if len(nearest) else 0.0


def _farthest_direction_subset(ref_dirs, n_select):
    """Deterministically retain a well-spread subset of reference directions."""
    directions = np.asarray(ref_dirs, dtype=float)
    if n_select >= len(directions):
        return np.array(directions, copy=True)
    normalized = directions / np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
    )
    selected = []
    for objective in range(directions.shape[1]):
        idx = int(np.argmax(normalized[:, objective]))
        if idx not in selected:
            selected.append(idx)
        if len(selected) == n_select:
            break
    while len(selected) < n_select:
        distance = np.min(
            np.linalg.norm(
                normalized[:, None, :] - normalized[selected][None, :, :], axis=2
            ),
            axis=1,
        )
        distance[selected] = -np.inf
        selected.append(int(np.argmax(distance)))
    return directions[np.asarray(selected)]


def _balanced_farthest_islands(X, capacities, xl, xu, random_state):
    """Create separated initial regions while exactly respecting capacities."""
    X = np.asarray(X, dtype=float)
    scale = np.where(np.asarray(xu) > np.asarray(xl), np.asarray(xu) - np.asarray(xl), 1.0)
    Xn = (X - np.asarray(xl)) / scale
    centers = [int(random_state.integers(len(Xn)))]
    while len(centers) < len(capacities):
        distance = np.min(
            np.linalg.norm(Xn[:, None, :] - Xn[centers][None, :, :], axis=2), axis=1
        )
        distance[centers] = -np.inf
        centers.append(int(np.argmax(distance)))
    center_distance = np.linalg.norm(Xn[:, None, :] - Xn[centers][None, :, :], axis=2)
    assigned = np.full(len(Xn), -1, dtype=int)
    remaining = np.asarray(capacities, dtype=int).copy()
    for island, idx in enumerate(centers):
        assigned[idx] = island
        remaining[island] -= 1
    unassigned = np.where(assigned < 0)[0]
    # Points with a clear nearest center go first; ambiguous boundary points
    # fill the capacity that remains.
    sorted_dist = np.sort(center_distance[unassigned], axis=1)
    confidence = sorted_dist[:, 1] - sorted_dist[:, 0]
    for idx in unassigned[np.argsort(-confidence)]:
        available = np.where(remaining > 0)[0]
        island = int(available[np.argmin(center_distance[idx, available])])
        assigned[idx] = island
        remaining[island] -= 1
    return assigned


def _main_exploration_initial_islands(X, capacities, xl, xu, random_state):
    """Give the main population global coverage and explorers separated leftovers."""
    X = np.asarray(X, dtype=float)
    scale = np.where(np.asarray(xu) > np.asarray(xl), np.asarray(xu) - np.asarray(xl), 1.0)
    Xn = (X - np.asarray(xl)) / scale
    main_size = capacities[0]
    # Farthest-first selection makes the large main population space filling.
    main = [int(random_state.integers(len(Xn)))]
    nearest_sq = np.sum((Xn - Xn[main[0]]) ** 2, axis=1)
    nearest_sq[main] = -np.inf
    while len(main) < main_size:
        idx = int(np.argmax(nearest_sq))
        main.append(idx)
        nearest_sq = np.minimum(nearest_sq, np.sum((Xn - Xn[idx]) ** 2, axis=1))
        nearest_sq[main] = -np.inf
    assigned = np.full(len(Xn), -1, dtype=int)
    assigned[main] = 0
    remaining_idx = np.where(assigned < 0)[0]
    if len(capacities) > 1:
        local = _balanced_farthest_islands(
            X[remaining_idx], capacities[1:], xl, xu, random_state
        )
        assigned[remaining_idx] = local + 1
    return assigned


def associate_to_niches(F, niches, ideal_point, nadir_point, utopian_epsilon=0.0):
    utopian_point = ideal_point - utopian_epsilon

    denom = nadir_point - utopian_point
    denom[denom == 0] = 1e-12

    # normalize by ideal point and intercepts
    N = (F - utopian_point) / denom
    dist_matrix = load_function("calc_perpendicular_distance")(N, niches)

    niche_of_individuals = np.argmin(dist_matrix, axis=1)
    dist_to_niche = dist_matrix[np.arange(F.shape[0]), niche_of_individuals]

    return niche_of_individuals, dist_to_niche, dist_matrix


def calc_niche_count(n_niches, niche_of_individuals):
    niche_count = np.zeros(n_niches, dtype=int)
    index, count = np.unique(niche_of_individuals, return_counts=True)
    niche_count[index] = count
    return niche_count


# =========================================================================================================
# Normalization
# =========================================================================================================


class HyperplaneNormalization:

    def __init__(self, n_dim) -> None:
        super().__init__()
        self.ideal_point = np.full(n_dim, np.inf)
        self.worst_point = np.full(n_dim, -np.inf)
        self.nadir_point = None
        self.extreme_points = None

    def update(self, F, nds=None):
        # find or usually update the new ideal point - from feasible solutions
        self.ideal_point = np.min(np.vstack((self.ideal_point, F)), axis=0)
        self.worst_point = np.max(np.vstack((self.worst_point, F)), axis=0)

        # this decides whether only non-dominated points or all points are used to determine the extreme points
        if nds is None:
            nds = np.arange(len(F))

        # find the extreme points for normalization
        self.extreme_points = get_extreme_points_c(F[nds, :], self.ideal_point,
                                                   extreme_points=self.extreme_points)

        # find the intercepts for normalization and do backup if gaussian elimination fails
        worst_of_population = np.max(F, axis=0)
        worst_of_front = np.max(F[nds, :], axis=0)

        self.nadir_point = get_nadir_point(self.extreme_points, self.ideal_point, self.worst_point,
                                           worst_of_front, worst_of_population)


def get_extreme_points_c(F, ideal_point, extreme_points=None):
    # calculate the asf which is used for the extreme point decomposition
    weights = np.eye(F.shape[1])
    weights[weights == 0] = 1e6

    # add the old extreme points to never lose them for normalization
    _F = F
    if extreme_points is not None:
        _F = np.concatenate([extreme_points, _F], axis=0)

    # use __F because we substitute small values to be 0
    __F = _F - ideal_point
    __F[__F < 1e-3] = 0

    # update the extreme points for the normalization having the highest asf value each
    F_asf = np.max(__F * weights[:, None, :], axis=2)

    I = np.argmin(F_asf, axis=1)
    extreme_points = _F[I, :]

    return extreme_points


def get_nadir_point(extreme_points, ideal_point, worst_point, worst_of_front, worst_of_population):
    try:

        # find the intercepts using gaussian elimination
        M = extreme_points - ideal_point
        b = np.ones(extreme_points.shape[1])
        plane = np.linalg.solve(M, b)

        warnings.simplefilter("ignore")
        intercepts = 1 / plane

        nadir_point = ideal_point + intercepts

        # check if the hyperplane makes sense
        if not np.allclose(np.dot(M, plane), b) or np.any(intercepts <= 1e-6):
            raise LinAlgError()

        # if the nadir point should be larger than any value discovered so far set it to that value
        # NOTE: different to the proposed version in the paper
        b = nadir_point > worst_point
        nadir_point[b] = worst_point[b]

    except LinAlgError:

        # fall back to worst of front otherwise
        nadir_point = worst_of_front

    # if the range is too small set it to worst of population
    b = nadir_point - ideal_point <= 1e-6
    nadir_point[b] = worst_of_population[b]

    return nadir_point


parse_doc_string(NSGA3.__init__)
