"""Append-only equal-budget screening on public Moving AI warehouse MAPF."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np

from pnsga3_candidate_v2 import build_pnsga3_candidate_v2
from pnsga3_mapf_edge_coverage import (
    build_nsga3_echo,
    build_nsga3_mapf_edge_coverage,
    build_nsga3_objective_archive,
    build_pnsga3_mapf_edge_coverage,
)
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import (
    NSGA3,
    ParallelNSGA3,
    ReferenceDirectionSurvival,
)
from pymoo.core.population import Population
from pymoo.optimize import minimize
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.sampling.categorical import (
    CategoricalRandomMutation,
    ScrambledSobolCategoricalSampling,
)
from pymoo.problems.multi.movingai_mapf import (
    MAPFGenotypeDuplicateElimination,
    PrioritizedMAPFProblem,
    load_movingai_map,
    load_movingai_scenarios,
    validate_tasks,
)
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions
from research_archive import ExperimentArchive


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def operators(problem, eliminate_duplicates=True):
    return {
        "sampling": ScrambledSobolCategoricalSampling(),
        "crossover": UniformCrossover(),
        "mutation": CategoricalRandomMutation(),
        # Plan identity requires path planning. Using it inside variation would
        # create uncounted evaluations, so offspring use only cheap canonical
        # priority/preference identity. Exact plan identity is used at output.
        "eliminate_duplicates": (
            MAPFGenotypeDuplicateElimination(problem) if eliminate_duplicates else False
        ),
    }


def build_algorithm(name, problem, pop_size, eliminate_duplicates=True):
    ref_dirs = get_reference_directions("energy", problem.n_obj, pop_size, seed=1)
    shared = operators(problem, eliminate_duplicates=eliminate_duplicates)
    if name == "NSGA2":
        return NSGA2(pop_size=pop_size, **shared)
    if name == "NSGA3":
        return NSGA3(ref_dirs, pop_size=pop_size, **shared)
    if name in {"Islands", "Islands-Diversity"}:
        diversity = .7 if name == "Islands-Diversity" else 0.0
        return ParallelNSGA3(
            ref_dirs,
            pop_size=pop_size,
            n_islands=4,
            main_population_fraction=.7,
            migration_interval=-1,
            explorer_to_main_exchange=False,
            archive_update_interval=5,
            direction_replicas=2,
            decision_diversity=diversity,
            quality_gated_diversity=diversity > 0,
            basin_archive_size=0,
            exploration_memory_size=0,
            declared_output_size=pop_size,
            declared_output_local_fronts=True,
            archive_injection=False,
            decision_space_islands=False,
            **shared,
        )
    if name == "PNSGA3-v2":
        return build_pnsga3_candidate_v2(problem, pop_size, **shared)
    if name == "PNSGA3-Archive4":
        return build_pnsga3_candidate_v2(
            problem,
            pop_size,
            archive_direction_replicas=4,
            **shared,
        )
    if name == "PNSGA3-RouteEmbedding":
        return build_pnsga3_candidate_v2(problem, pop_size, **shared)
    if name == "PNSGA3-EdgeDescriptor":
        return build_pnsga3_candidate_v2(problem, pop_size, **shared)
    if name == "PNSGA3-EdgeCoverage":
        return build_pnsga3_mapf_edge_coverage(problem, pop_size, **shared)
    if name == "NSGA3-EdgeCoverage":
        return build_nsga3_mapf_edge_coverage(problem, pop_size, **shared)
    if name == "NSGA3-ECHO":
        return build_nsga3_echo(problem, pop_size, **shared)
    if name == "NSGA3-ObjectiveArchive":
        return build_nsga3_objective_archive(problem, pop_size, **shared)
    if name == "PNSGA3-EdgeCoverage80":
        return build_pnsga3_mapf_edge_coverage(
            problem, pop_size, main_population_fraction=.8, **shared)
    if name == "PNSGA3-EdgeCoverage90":
        return build_pnsga3_mapf_edge_coverage(
            problem, pop_size, main_population_fraction=.9, **shared)
    raise ValueError(f"Unknown algorithm: {name}")


def bounded_output(result, pop_size):
    """Return exactly N slots; repeated plans consume capacity rather than vanish."""
    if getattr(result.algorithm, "basin_archive_selection", None) == "failure_coverage":
        # The registered application-specific intervention includes a final
        # set-level composition step. Preserve it instead of applying the
        # generic pointwise farthest-first output rule a second time.
        result.algorithm._preserve_main_output_core()
        selected = result.algorithm.basin_archive
        if len(selected):
            X = np.rint(np.asarray(selected.get("X"))).astype(int).tolist()
            F = np.asarray(selected.get("F"), dtype=float).tolist()
            if len(X) < pop_size:
                for row_x, row_f in zip(result.pop.get("X"), result.pop.get("F")):
                    if len(X) >= pop_size:
                        break
                    X.append(np.rint(row_x).astype(int).tolist())
                    F.append(np.asarray(row_f, dtype=float).tolist())
            while len(X) < pop_size:
                X.append(list(X[0]))
                F.append(list(F[0]))
            return np.asarray(X[:pop_size], dtype=int), np.asarray(F[:pop_size], dtype=float)
    population = result.pop
    passive = getattr(result.algorithm, "basin_archive", None)
    if passive is not None and len(passive):
        population = Population.merge(population, passive)
    if len(population) <= pop_size:
        return (
            np.rint(np.asarray(population.get("X"))).astype(int),
            np.asarray(population.get("F"), dtype=float),
        )
    quality_size = min(int(round(.7 * pop_size)), len(population))
    survival = ReferenceDirectionSurvival(result.algorithm.ref_dirs)
    core = survival.do(
        result.problem,
        population,
        n_survive=quality_size,
        random_state=result.algorithm.random_state,
    )
    pool_X = np.rint(np.asarray(population.get("X"))).astype(int)
    core_keys = {tuple(np.rint(row).astype(int)) for row in core.get("X")}
    selected = [index for index, row in enumerate(pool_X) if tuple(row) in core_keys]
    selected = selected[:quality_size]
    selected_set = set(selected)
    candidates = [index for index in range(len(population)) if index not in selected_set]
    embedded = result.problem.decision_space_embedding(pool_X)
    while candidates and len(selected) < pop_size:
        distance = np.min(np.linalg.norm(
            embedded[candidates, None, :] - embedded[selected][None, :, :], axis=2
        ), axis=1)
        selected.append(candidates.pop(int(np.argmax(distance))))
    output = population[selected[:pop_size]]
    return (
        np.rint(np.asarray(output.get("X"))).astype(int),
        np.asarray(output.get("F"), dtype=float),
    )


def _plan_features(paths):
    physical_vertices = set()
    physical_edges = set()
    labelled_edges = set()
    undirected_physical_edges = set()
    for agent, path in enumerate(paths):
        for cell in path:
            physical_vertices.add(cell)
        for source, target in zip(path[:-1], path[1:]):
            physical_edges.add((source, target))
            labelled_edges.add((agent, source, target))
            if source != target:
                undirected_physical_edges.add(
                    (source, target) if source < target else (target, source)
                )
    return {
        "physical_vertices": physical_vertices,
        "physical_edges": physical_edges,
        "labelled_edges": labelled_edges,
        "undirected_physical_edges": undirected_physical_edges,
    }


def _jaccard_distance(left, right):
    union = left | right
    return 0.0 if not union else 1.0 - len(left & right) / len(union)


def score(problem, X, F):
    decoded = [problem.decode(row) for row in X]
    if not all(item["feasible"] for item in decoded):
        raise ValueError("Declared output contains an infeasible joint plan")
    identities = problem.decision_identity(X)
    features = [_plan_features(item["paths"]) for item in decoded]
    protected_endpoints = {
        task.start for task in problem.tasks
    } | {
        task.goal for task in problem.tasks
    }
    route_distance = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            route_distance[i, j] = route_distance[j, i] = _jaccard_distance(
                features[i]["labelled_edges"], features[j]["labelled_edges"]
            )

    baseline_steps = max(1.0, sum(task.optimal_single_agent_length for task in problem.tasks))
    scale = np.asarray([baseline_steps, 9.0 * baseline_steps])
    gap = np.max(np.abs(F[:, None, :] - F[None, :, :]) / scale, axis=2)
    np.fill_diagonal(gap, np.inf)
    distinct = identities[:, None] != identities[None, :]
    metrics = {}
    for suffix, epsilon in (("exact", 0.0), ("001", .01), ("002", .02), ("005", .05)):
        alternatives = distinct & (gap <= epsilon) & (route_distance >= .10)
        metrics[f"backup_fraction_epsilon_{suffix}"] = float(np.mean(
            np.any(alternatives, axis=1)
        ))
        vertex_survival = []
        edge_survival = []
        for i in range(len(X)):
            candidate_indices = np.where(alternatives[i])[0]
            own_vertices = features[i]["physical_vertices"] - protected_endpoints
            own_edges = features[i]["physical_edges"]
            if own_vertices:
                vertex_survival.extend(
                    any(
                        vertex not in features[j]["physical_vertices"]
                        for j in candidate_indices
                    )
                    for vertex in own_vertices
                )
            if own_edges:
                edge_survival.extend(
                    any(
                        edge not in features[j]["physical_edges"]
                        for j in candidate_indices
                    )
                    for edge in own_edges
                )
        metrics[f"blocked_vertex_survival_epsilon_{suffix}"] = (
            float(np.mean(vertex_survival)) if vertex_survival else 0.0
        )
        metrics[f"blocked_edge_survival_epsilon_{suffix}"] = (
            float(np.mean(edge_survival)) if edge_survival else 0.0
        )

    exact_groups = {}
    for index, objective in enumerate(np.asarray(F, dtype=float)):
        exact_groups.setdefault(tuple(objective), []).append(index)
    survived_events = 0
    exposed_events = 0
    survived_two_edge_events = 0
    exposed_two_edge_events = 0
    for indices in exact_groups.values():
        edge_sets = [features[index]["undirected_physical_edges"] for index in indices]
        union = set().union(*edge_sets)
        common = set.intersection(*edge_sets)
        survived_events += len(union - common)
        exposed_events += len(union)
        union_list = sorted(union)
        if len(union_list) >= 2:
            all_plans = (1 << len(edge_sets)) - 1
            masks = []
            for edge in union_list:
                mask = 0
                for position, values in enumerate(edge_sets):
                    if edge in values:
                        mask |= 1 << position
                masks.append(mask)
            total_pairs = len(masks) * (len(masks) - 1) // 2
            failed_pairs = sum(
                (masks[left] | masks[right]) == all_plans
                for left in range(len(masks))
                for right in range(left + 1, len(masks))
            )
            exposed_two_edge_events += total_pairs
            survived_two_edge_events += total_pairs - failed_pairs
    metrics["exact_group_undirected_edge_coverage"] = (
        float(survived_events / exposed_events) if exposed_events else 0.0
    )
    metrics["exact_group_undirected_two_edge_coverage"] = (
        float(survived_two_edge_events / exposed_two_edge_events)
        if exposed_two_edge_events else 0.0
    )

    nd = NonDominatedSorting().do(F, only_non_dominated_front=True)
    metrics.update({
        "declared_output_size": int(len(X)),
        "unique_plan_count": int(len(set(identities.tolist()))),
        "nondominated_output_size": int(len(nd)),
        "mean_sum_of_costs": float(np.mean(F[:, 0])),
        "mean_clearance_risk": float(np.mean(F[:, 1])),
        "mean_makespan": float(np.mean([item["makespan"] for item in decoded])),
        "mean_pairwise_route_distance": float(np.mean(
            route_distance[np.triu_indices(len(X), 1)]
        )) if len(X) > 1 else 0.0,
    })
    return metrics


def make_problem(map_path, scenario_path, agents, horizon=None,
                 decision_embedding_mode="trajectory", task_indices=None,
                 preference_levels=5, independent_neighbor_orders=False,
                 blocked_edges=None):
    grid = load_movingai_map(map_path)
    available = load_movingai_scenarios(scenario_path)
    tasks = (
        available[:agents]
        if task_indices is None
        else [available[int(index)] for index in task_indices]
    )
    if len(tasks) != agents:
        raise ValueError("task_indices length must equal agents")
    validate_tasks(grid, tasks, expected_map_name=Path(map_path).name)
    return PrioritizedMAPFProblem(
        grid,
        tasks,
        horizon=horizon,
        decision_embedding_mode=decision_embedding_mode,
        preference_levels=preference_levels,
        independent_neighbor_orders=independent_neighbor_orders,
        blocked_edges=blocked_edges,
    )


def run(args):
    config = {
        "map": str(Path(args.map).resolve()),
        "map_sha256": sha256(args.map),
        "scenario": str(Path(args.scenario).resolve()),
        "scenario_sha256": sha256(args.scenario),
        "agents": args.agents,
        "task_indices": args.task_indices,
        "algorithms": args.algorithms,
        "population": args.pop_size,
        "evaluations": args.evaluations,
        "seeds": args.seeds,
        "objectives": ["sum_of_costs", "clearance_risk"],
        "primary_backup_epsilon": .02,
        "minimum_route_jaccard_distance": .10,
        "hypothesis": args.hypothesis or (
            "PNSGA3-v2 returns more near-equivalent collision-free joint plans "
            "that survive a blocked route cell/edge under equal evaluations."
        ),
        "evidence_status": args.evidence_status,
        "eliminate_duplicate_genotypes": not args.allow_duplicate_genotypes,
        "preference_levels": args.preference_levels,
        "independent_neighbor_orders": args.independent_neighbor_orders,
    }
    archive = ExperimentArchive(
        args.archive_root, args.experiment_id, "movingai-mapf-priority-v1", config
    )
    try:
        for seed in args.seeds:
            for name in args.algorithms:
                embedding_mode = (
                    "physical_occupancy"
                    if name == "PNSGA3-RouteEmbedding"
                    else "edge_unavailability"
                    if name == "PNSGA3-EdgeDescriptor"
                    else "trajectory"
                )
                problem = make_problem(
                    args.map,
                    args.scenario,
                    args.agents,
                    args.horizon,
                    embedding_mode,
                    args.task_indices,
                    args.preference_levels,
                    args.independent_neighbor_orders,
                )
                algorithm = build_algorithm(
                    name,
                    problem,
                    args.pop_size,
                    eliminate_duplicates=not args.allow_duplicate_genotypes,
                )
                recorder = archive.new_run(
                    f"{name}_seed{seed:03d}",
                    {
                        "algorithm": name,
                        "seed": seed,
                        "population": args.pop_size,
                        "evaluation_budget": args.evaluations,
                        "decision_embedding_mode": embedding_mode,
                    },
                    snapshot_interval=args.snapshot_interval,
                )
                started = time.perf_counter()
                try:
                    result = minimize(
                        problem,
                        algorithm,
                        ("n_eval", args.evaluations),
                        seed=seed,
                        callback=recorder,
                        verbose=False,
                    )
                    X, F = bounded_output(result, args.pop_size)
                    result.archive_declared_output = (X, F)
                    metrics = score(problem, X, F)
                    metrics.update({
                        "elapsed_seconds": time.perf_counter() - started,
                        "n_eval": int(result.algorithm.evaluator.n_eval),
                        "unique_evaluated_genotypes": len(problem.evaluated_genotypes),
                        "decode_cache_size": len(problem._decode_cache),
                    })
                    recorder.complete(result, metrics)
                except Exception as error:
                    recorder.fail(error)
                    raise
        archive.finish()
    except Exception:
        archive.finish(status="failed")
        raise


def parser():
    root = Path(__file__).resolve().parent
    value = argparse.ArgumentParser()
    value.add_argument("--experiment-id", required=True)
    value.add_argument("--archive-root", default=root / "research_archive")
    value.add_argument("--map", default=(
        root / "data/movingai_mapf/maps/warehouse-10-20-10-2-1.map"
    ))
    value.add_argument("--scenario", default=(
        root / "data/movingai_mapf/scenarios/scen-random/"
        "warehouse-10-20-10-2-1-random-1.scen"
    ))
    value.add_argument("--agents", type=int, default=4)
    value.add_argument("--preference-levels", type=int, default=5)
    value.add_argument("--independent-neighbor-orders", action="store_true")
    value.add_argument(
        "--task-indices",
        type=lambda x: [int(value) for value in x.split(",")],
        help="Optional zero-based scenario row indices; length must equal --agents.",
    )
    value.add_argument("--algorithms", type=lambda x: x.split(","), default=[
        "NSGA2", "NSGA3", "Islands", "Islands-Diversity", "PNSGA3-v2"
    ])
    value.add_argument("--pop-size", type=int, default=40)
    value.add_argument("--evaluations", type=int, default=400)
    value.add_argument("--seeds", type=lambda x: [int(v) for v in x.split(",")],
                       default=[1, 2, 3])
    value.add_argument("--horizon", type=int)
    value.add_argument(
        "--allow-duplicate-genotypes",
        action="store_true",
        help="Keep duplicate offspring so fixed-size generations cannot overshoot the budget.",
    )
    value.add_argument("--snapshot-interval", type=int, default=5)
    value.add_argument(
        "--hypothesis",
        help="Pre-registered hypothesis stored verbatim in the archive config.",
    )
    value.add_argument(
        "--evidence-status",
        choices=("smoke", "development", "confirmatory"),
        default="smoke",
    )
    return value


if __name__ == "__main__":
    run(parser().parse_args())
