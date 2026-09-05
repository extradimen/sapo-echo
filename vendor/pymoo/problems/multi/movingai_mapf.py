"""Parsers and validation for the public Moving AI MAPF benchmark format."""

from __future__ import annotations

import hashlib
import heapq
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pymoo.core.duplicate import DuplicateElimination
from pymoo.core.problem import Problem


@dataclass(frozen=True)
class MAPFTask:
    bucket: int
    map_name: str
    width: int
    height: int
    start: tuple[int, int]
    goal: tuple[int, int]
    optimal_single_agent_length: float


def load_movingai_map(path):
    lines = Path(path).read_text().splitlines()
    if len(lines) < 5 or not lines[0].lower().startswith("type "):
        raise ValueError(f"Invalid Moving AI map header: {path}")
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    if lines[3].strip().lower() != "map":
        raise ValueError(f"Missing map marker: {path}")
    rows = lines[4:]
    if len(rows) != height or any(len(row) != width for row in rows):
        raise ValueError(f"Map dimensions do not match header: {path}")
    traversable_symbols = {".", "G", "S"}
    known_symbols = traversable_symbols | {"@", "O", "T", "W"}
    symbols = set("".join(rows))
    if not symbols <= known_symbols:
        raise ValueError(f"Unknown map symbols: {sorted(symbols - known_symbols)}")
    return np.asarray([
        [symbol in traversable_symbols for symbol in row] for row in rows
    ], dtype=bool)


def load_movingai_scenarios(path):
    lines = Path(path).read_text().splitlines()
    if not lines or not lines[0].lower().startswith("version "):
        raise ValueError(f"Invalid Moving AI scenario header: {path}")
    tasks = []
    for line_number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(f"Expected 9 fields at {path}:{line_number}")
        tasks.append(MAPFTask(
            bucket=int(fields[0]),
            map_name=fields[1],
            width=int(fields[2]),
            height=int(fields[3]),
            start=(int(fields[4]), int(fields[5])),
            goal=(int(fields[6]), int(fields[7])),
            optimal_single_agent_length=float(fields[8]),
        ))
    if not tasks:
        raise ValueError(f"Scenario contains no tasks: {path}")
    return tasks


def validate_tasks(grid, tasks, expected_map_name=None):
    height, width = grid.shape
    for index, task in enumerate(tasks):
        if (task.width, task.height) != (width, height):
            raise ValueError(f"Task {index} dimensions disagree with its map")
        if expected_map_name is not None and task.map_name != expected_map_name:
            raise ValueError(f"Task {index} references {task.map_name}, not {expected_map_name}")
        for label, (x, y) in (("start", task.start), ("goal", task.goal)):
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Task {index} {label} lies outside the map")
            if not grid[y, x]:
                raise ValueError(f"Task {index} {label} lies on an obstacle")
        lower_bound = float(np.hypot(
            task.start[0] - task.goal[0], task.start[1] - task.goal[1]
        ))
        if task.optimal_single_agent_length + 1e-6 < lower_bound:
            raise ValueError(f"Task {index} reported length violates Euclidean lower bound")
    return True


def warehouse_clearance_risk(grid):
    """Return an integer shelf/obstacle-proximity cost for every free cell.

    The value is one plus the number of blocked or out-of-map cells in the
    eight-cell neighbourhood.  It is deterministic, uses no fitted parameter,
    and makes narrow warehouse aisles more costly than open cells.
    """
    grid = np.asarray(grid, dtype=bool)
    height, width = grid.shape
    risk = np.zeros((height, width), dtype=np.int16)
    for y in range(height):
        for x in range(width):
            if not grid[y, x]:
                continue
            blocked = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < height and 0 <= nx < width) or not grid[ny, nx]:
                        blocked += 1
            risk[y, x] = 1 + blocked
    return risk


def canonical_joint_paths(paths):
    """Remove redundant terminal waits and return a hashable joint plan."""
    canonical = []
    for path in paths:
        path = [tuple(map(int, cell)) for cell in path]
        while len(path) > 1 and path[-2] == path[-1]:
            path.pop()
        canonical.append(tuple(path))
    return tuple(canonical)


def joint_paths_are_collision_free(paths):
    """Check vertex and opposite-edge conflicts while goals remain occupied."""
    if not paths:
        return False
    paths = canonical_joint_paths(paths)
    horizon = max(len(path) for path in paths)
    for time in range(horizon):
        positions = [path[min(time, len(path) - 1)] for path in paths]
        if len(set(positions)) != len(positions):
            return False
        if time == 0:
            continue
        previous = [path[min(time - 1, len(path) - 1)] for path in paths]
        transitions = set()
        for source, target in zip(previous, positions):
            if (target, source) in transitions and source != target:
                return False
            transitions.add((source, target))
    return True


def _position_at(path, time):
    return path[min(time, len(path) - 1)]


def _conflicts_with_plans(current, target, time, planned_paths):
    """Check a move from ``current`` at t to ``target`` at t+1."""
    for path in planned_paths:
        other_now = _position_at(path, time)
        other_next = _position_at(path, time + 1)
        if target == other_next:
            return True
        if current == other_next and target == other_now and current != target:
            return True
    return False


def _goal_safe_forever(goal, arrival_time, planned_paths, horizon):
    for time in range(arrival_time, horizon + 1):
        if any(_position_at(path, time) == goal for path in planned_paths):
            return False
    return True


def _route_alpha(route_preference, preference_levels):
    if preference_levels < 2:
        raise ValueError("preference_levels must be at least two")
    index = int(route_preference)
    if not 0 <= index < preference_levels:
        raise ValueError("route_preference lies outside preference_levels")
    return index / float(preference_levels - 1)


def _space_time_astar(grid, risk, start, goal, planned_paths, horizon,
                      route_preference, spatial_heuristic=None,
                      preference_levels=5, neighbor_order=None,
                      blocked_edges=None):
    """Plan one robot against paths already fixed by priority planning."""
    alpha = _route_alpha(route_preference, preference_levels)
    max_risk = max(1.0, float(np.max(risk)))
    minimum_step = alpha + (1.0 - alpha) / max_risk
    direction_orders = (
        ((0, 0), (1, 0), (0, 1), (-1, 0), (0, -1)),
        ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)),
        ((0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)),
        ((0, 0), (0, -1), (-1, 0), (0, 1), (1, 0)),
        ((0, 0), (1, 0), (0, -1), (-1, 0), (0, 1)),
    )
    order = route_preference if neighbor_order is None else neighbor_order
    actions = direction_orders[int(order) % len(direction_orders)]
    blocked_edges = frozenset(blocked_edges or ())

    start_state = (tuple(start), 0)
    queue = []
    counter = 0
    initial_h = (
        float(spatial_heuristic[start[1], start[0]])
        if spatial_heuristic is not None
        else (abs(start[0] - goal[0]) + abs(start[1] - goal[1])) * minimum_step
    )
    if not np.isfinite(initial_h):
        return None
    heapq.heappush(queue, (initial_h, 0.0, counter, tuple(start), 0))
    best = {start_state: 0.0}
    parent = {}
    height, width = grid.shape

    while queue:
        _, cost, _, cell, time = heapq.heappop(queue)
        if cost > best.get((cell, time), np.inf) + 1e-12:
            continue
        if cell == tuple(goal) and _goal_safe_forever(cell, time, planned_paths, horizon):
            path = [cell]
            state = (cell, time)
            while state != start_state:
                state = parent[state]
                path.append(state[0])
            return list(reversed(path))
        if time >= horizon:
            continue
        for dx, dy in actions:
            target = (cell[0] + dx, cell[1] + dy)
            x, y = target
            if not (0 <= x < width and 0 <= y < height and grid[y, x]):
                continue
            if cell != target:
                edge = (cell, target) if cell < target else (target, cell)
                if edge in blocked_edges:
                    continue
            if _conflicts_with_plans(cell, target, time, planned_paths):
                continue
            step = alpha + (1.0 - alpha) * float(risk[y, x]) / max_risk
            next_state = (target, time + 1)
            next_cost = cost + step
            if next_cost + 1e-12 >= best.get(next_state, np.inf):
                continue
            best[next_state] = next_cost
            parent[next_state] = (cell, time)
            heuristic = (
                float(spatial_heuristic[y, x])
                if spatial_heuristic is not None
                else (abs(target[0] - goal[0]) + abs(target[1] - goal[1])) * minimum_step
            )
            counter += 1
            heapq.heappush(
                queue,
                (next_cost + heuristic, next_cost, counter, target, time + 1),
            )
    return None


def _spatial_cost_to_goal(grid, risk, goal, route_preference,
                          preference_levels=5, blocked_edges=None):
    """Exact collision-free single-agent cost-to-go used as an A* heuristic."""
    alpha = _route_alpha(route_preference, preference_levels)
    max_risk = max(1.0, float(np.max(risk)))
    distance = np.full(grid.shape, np.inf, dtype=float)
    gx, gy = goal
    distance[gy, gx] = 0.0
    queue = [(0.0, gx, gy)]
    blocked_edges = frozenset(blocked_edges or ())
    height, width = grid.shape
    while queue:
        cost, x, y = heapq.heappop(queue)
        if cost > distance[y, x] + 1e-12:
            continue
        # In reverse search, each predecessor -> (x, y) pays the arrival cost
        # of the current cell.
        reverse_step = alpha + (1.0 - alpha) * float(risk[y, x]) / max_risk
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            px, py = x + dx, y + dy
            if not (0 <= px < width and 0 <= py < height and grid[py, px]):
                continue
            source, target = (px, py), (x, y)
            edge = (source, target) if source < target else (target, source)
            if edge in blocked_edges:
                continue
            candidate = cost + reverse_step
            if candidate + 1e-12 < distance[py, px]:
                distance[py, px] = candidate
                heapq.heappush(queue, (candidate, px, py))
    return distance


class PrioritizedMAPFProblem(Problem):
    """Fixed-length evolutionary interface to collision-free joint paths.

    The first ``A`` categorical genes are random priority keys for ``A``
    labelled robots. The next genes select one of ``preference_levels``
    deterministic travel/risk scalarizations (five by default). By default
    these also cycle neighbour orders for backward compatibility. With
    ``independent_neighbor_orders=True``, a final ``A`` genes select neighbour
    orders independently. Decoding uses
    space-time A* and never treats an infeasible joint plan as a backup.
    Objectives are sum of arrival steps and cumulative clearance risk, both
    minimized and additive like the cost vectors used by MO-CBS.
    """

    def __init__(self, grid, tasks, risk_grid=None, horizon=None,
                 path_embedding_samples=16,
                 decision_embedding_mode="trajectory",
                 preference_levels=5,
                 independent_neighbor_orders=False,
                 blocked_edges=None):
        grid = np.asarray(grid, dtype=bool)
        tasks = tuple(tasks)
        validate_tasks(grid, tasks)
        if len(tasks) < 2:
            raise ValueError("PrioritizedMAPFProblem requires at least two robots")
        starts = [task.start for task in tasks]
        goals = [task.goal for task in tasks]
        if len(set(starts)) != len(starts) or len(set(goals)) != len(goals):
            raise ValueError("Selected robots must have distinct starts and goals")
        self.grid = grid
        self.tasks = tasks
        self.n_agents = len(tasks)
        self.risk_grid = (
            warehouse_clearance_risk(grid) if risk_grid is None
            else np.asarray(risk_grid, dtype=float)
        )
        if self.risk_grid.shape != grid.shape or np.any(self.risk_grid[grid] <= 0):
            raise ValueError("risk_grid must be positive on every traversable cell")
        longest = max(task.optimal_single_agent_length for task in tasks)
        self.horizon = int(horizon or np.ceil(2.0 * longest + 4 * len(tasks) + 16))
        self.path_embedding_samples = int(path_embedding_samples)
        self.preference_levels = int(preference_levels)
        self.independent_neighbor_orders = bool(independent_neighbor_orders)
        if self.preference_levels < 2:
            raise ValueError("preference_levels must be at least two")
        if decision_embedding_mode not in {
                "trajectory", "physical_occupancy", "edge_unavailability"}:
            raise ValueError(
                "decision_embedding_mode must be 'trajectory', "
                "'physical_occupancy', or 'edge_unavailability'"
            )
        self.decision_embedding_mode = decision_embedding_mode
        physical_edges = []
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if not grid[y, x]:
                    continue
                for dx, dy in ((1, 0), (0, 1)):
                    nx, ny = x + dx, y + dy
                    if ny < grid.shape[0] and nx < grid.shape[1] and grid[ny, nx]:
                        physical_edges.append(((x, y), (nx, ny)))
        self.physical_edge_index = {
            edge: index for index, edge in enumerate(physical_edges)
        }
        self.blocked_edges = frozenset(
            (source, target) if source < target else (target, source)
            for source, target in (blocked_edges or ()))
        unknown_blocked = self.blocked_edges - set(self.physical_edge_index)
        if unknown_blocked:
            raise ValueError(
                "blocked_edges contains a non-traversable physical edge")
        self._decode_cache = {}
        self._heuristic_cache = {}
        self.evaluated_genotypes = set()
        gene_blocks = 3 if self.independent_neighbor_orders else 2
        lower = np.zeros(gene_blocks * self.n_agents, dtype=int)
        upper_blocks = [
            np.full(self.n_agents, self.n_agents - 1, dtype=int),
            np.full(self.n_agents, self.preference_levels - 1, dtype=int),
        ]
        if self.independent_neighbor_orders:
            upper_blocks.append(np.full(self.n_agents, 4, dtype=int))
        upper = np.concatenate(upper_blocks)
        super().__init__(
            n_var=gene_blocks * self.n_agents,
            n_obj=2,
            n_ieq_constr=1,
            xl=lower,
            xu=upper,
            vtype=int,
        )

    def _genotype(self, x):
        x = np.rint(np.asarray(x)).astype(int)
        if x.shape != (self.n_var,):
            raise ValueError(f"Expected genotype with {self.n_var} entries")
        return tuple(x.tolist())

    def decode(self, x):
        genotype = self._genotype(x)
        if genotype in self._decode_cache:
            return self._decode_cache[genotype]
        keys = np.asarray(genotype[:self.n_agents])
        preferences = np.asarray(
            genotype[self.n_agents:2 * self.n_agents])
        neighbor_orders = (
            np.asarray(genotype[2 * self.n_agents:3 * self.n_agents])
            if self.independent_neighbor_orders else preferences
        )
        priority = np.lexsort((np.arange(self.n_agents), keys))
        planned_by_agent = [None] * self.n_agents
        planned_in_priority_order = []
        feasible = True
        for agent in priority:
            task = self.tasks[int(agent)]
            heuristic_key = (int(agent), int(preferences[int(agent)]))
            if heuristic_key not in self._heuristic_cache:
                self._heuristic_cache[heuristic_key] = _spatial_cost_to_goal(
                    self.grid,
                    self.risk_grid,
                    task.goal,
                    preferences[int(agent)],
                    self.preference_levels,
                    self.blocked_edges,
                )
            path = _space_time_astar(
                self.grid,
                self.risk_grid,
                task.start,
                task.goal,
                planned_in_priority_order,
                self.horizon,
                preferences[int(agent)],
                self._heuristic_cache[heuristic_key],
                self.preference_levels,
                neighbor_orders[int(agent)],
                self.blocked_edges,
            )
            if path is None:
                feasible = False
                break
            planned_by_agent[int(agent)] = path
            planned_in_priority_order.append(path)
        if feasible:
            paths = canonical_joint_paths(planned_by_agent)
            feasible = joint_paths_are_collision_free(paths)
        else:
            paths = None
        result = {"feasible": feasible, "paths": paths, "priority": tuple(priority.tolist())}
        if feasible:
            steps = float(sum(len(path) - 1 for path in paths))
            risk = float(sum(
                self.risk_grid[y, x]
                for path in paths for x, y in path[1:]
            ))
            result.update({
                "objectives": np.asarray([steps, risk]),
                "makespan": max(len(path) - 1 for path in paths),
            })
        self._decode_cache[genotype] = result
        return result

    def _evaluate(self, X, out, *args, **kwargs):
        self.evaluated_genotypes.update(self._genotype(row) for row in np.asarray(X))
        results = [self.decode(row) for row in np.asarray(X)]
        penalty = float(self.n_agents * self.horizon * max(10, np.max(self.risk_grid)))
        out["F"] = np.asarray([
            result["objectives"] if result["feasible"] else [penalty, penalty]
            for result in results
        ])
        out["G"] = np.asarray([[0.0 if result["feasible"] else 1.0] for result in results])

    def decision_identity(self, X):
        identities = []
        for row in np.atleast_2d(X):
            result = self.decode(row)
            if not result["feasible"]:
                identities.append("infeasible:" + ",".join(map(str, self._genotype(row))))
                continue
            payload = repr(result["paths"]).encode("utf-8")
            identities.append(hashlib.sha256(payload).hexdigest())
        return np.asarray(identities, dtype=str)

    def genotype_identity(self, X):
        """Canonical cheap identity: priority permutation plus route settings."""
        identities = []
        for row in np.atleast_2d(X):
            genotype = self._genotype(row)
            keys = np.asarray(genotype[:self.n_agents])
            priority = np.lexsort((np.arange(self.n_agents), keys))
            route_settings = genotype[self.n_agents:]
            identities.append(tuple(priority.tolist()) + tuple(route_settings))
        return identities

    def decision_space_embedding(self, X):
        """Embed joint paths without exposing objective or disruption outcomes."""
        embeddings = []
        height, width = self.grid.shape
        quantiles = np.linspace(0.0, 1.0, self.path_embedding_samples)
        for row in np.atleast_2d(X):
            result = self.decode(row)
            if not result["feasible"]:
                size = (
                    height * width if self.decision_embedding_mode == "physical_occupancy"
                    else len(self.physical_edge_index)
                    if self.decision_embedding_mode == "edge_unavailability"
                    else self.n_agents * self.path_embedding_samples * 2
                )
                embeddings.append(np.zeros(size))
                continue
            if self.decision_embedding_mode == "physical_occupancy":
                values = np.zeros(height * width, dtype=float)
                for path in result["paths"]:
                    for x, y in path:
                        values[y * width + x] = 1.0
                norm = np.linalg.norm(values)
                embeddings.append(values / norm if norm > 0 else values)
                continue
            if self.decision_embedding_mode == "edge_unavailability":
                values = np.zeros(len(self.physical_edge_index), dtype=float)
                for path in result["paths"]:
                    for source, target in zip(path, path[1:]):
                        if source == target:
                            continue
                        edge = (source, target) if source < target else (target, source)
                        values[self.physical_edge_index[edge]] = 1.0
                # Deliberately unnormalized: Euclidean distance is the square
                # root of the number of closure events affecting exactly one
                # plan, so extra failure exposure is not erased by scaling.
                embeddings.append(values)
                continue
            values = []
            for path in result["paths"]:
                indices = np.rint(quantiles * (len(path) - 1)).astype(int)
                for index in indices:
                    x, y = path[index]
                    values.extend((x / max(1, width - 1), y / max(1, height - 1)))
            embeddings.append(values)
        embeddings = np.asarray(embeddings, dtype=float)
        if self.decision_embedding_mode in {"physical_occupancy", "edge_unavailability"}:
            return embeddings
        return embeddings / np.sqrt(2.0 * self.n_agents)

    def failure_event_embedding(self, X):
        """Binary undirected physical-edge failure signature for each plan."""
        embeddings = []
        for row in np.atleast_2d(X):
            values = np.zeros(len(self.physical_edge_index), dtype=np.uint8)
            result = self.decode(row)
            if result["feasible"]:
                for path in result["paths"]:
                    for source, target in zip(path, path[1:]):
                        if source == target:
                            continue
                        edge = (source, target) if source < target else (target, source)
                        values[self.physical_edge_index[edge]] = 1
            embeddings.append(values)
        return np.asarray(embeddings, dtype=np.uint8)

    def vertex_event_embedding(self, X):
        """Binary physical-vertex usage signature for overlap baselines."""
        height, width = self.grid.shape
        embeddings = []
        for row in np.atleast_2d(X):
            values = np.zeros(height * width, dtype=np.uint8)
            result = self.decode(row)
            if result["feasible"]:
                for path in result["paths"]:
                    for x, y in path:
                        values[y * width + x] = 1
            embeddings.append(values)
        return np.asarray(embeddings, dtype=np.uint8)

    def directed_edge_usage_embedding(self, X):
        """Directed-edge occurrences and path-vertex counts for Xu-code audit.

        The official MMOEA-CDP Java implementation counts shared directed
        consecutive edges but divides by the candidate path's vertex count.
        For a joint MAPF plan we concatenate the per-agent accounting without
        adding artificial edges between agents.  Waiting actions are ignored
        as edges but their vertices remain in the denominator, matching the
        published code's edge-count/path-length asymmetry as closely as the
        multi-agent adaptation permits.
        """
        edge_count = 2 * len(self.physical_edge_index)
        embeddings = []
        denominators = []
        for row in np.atleast_2d(X):
            values = np.zeros(edge_count, dtype=np.uint16)
            result = self.decode(row)
            if result["feasible"]:
                denominators.append(sum(len(path) for path in result["paths"]))
                for path in result["paths"]:
                    for source, target in zip(path, path[1:]):
                        if source == target:
                            continue
                        edge = (source, target) if source < target else (target, source)
                        offset = 0 if source < target else len(self.physical_edge_index)
                        values[offset + self.physical_edge_index[edge]] += 1
            else:
                denominators.append(0)
            embeddings.append(values)
        return (
            np.asarray(embeddings, dtype=np.uint16),
            np.asarray(denominators, dtype=np.uint32),
        )


class MAPFPlanDuplicateElimination(DuplicateElimination):
    """Eliminate genotypes that decode to the same canonical joint plan."""

    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def _do(self, pop, other, is_duplicate):
        seen = set()
        if other is not None and len(other):
            seen.update(self.problem.decision_identity(other.get("X")).tolist())
        for index, identity in enumerate(
                self.problem.decision_identity(pop.get("X")).tolist()):
            if identity in seen:
                is_duplicate[index] = True
            else:
                seen.add(identity)
        return is_duplicate


class MAPFGenotypeDuplicateElimination(DuplicateElimination):
    """Remove search-equivalent genotypes without performing path planning.

    Plan-level duplicates are still collapsed at reporting time.  This class
    deliberately avoids using decoded paths during offspring generation,
    because doing so would perform uncounted planner evaluations.
    """

    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def _do(self, pop, other, is_duplicate):
        seen = set()
        if other is not None and len(other):
            seen.update(self.problem.genotype_identity(other.get("X")))
        for index, identity in enumerate(self.problem.genotype_identity(pop.get("X"))):
            if identity in seen:
                is_duplicate[index] = True
            else:
                seen.add(identity)
        return is_duplicate
