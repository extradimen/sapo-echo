"""Build, run, and audit the authors' external MO-CBS implementation."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from pymoo.problems.multi.movingai_mapf import (
    canonical_joint_paths,
    joint_paths_are_collision_free,
    load_movingai_map,
    load_movingai_scenarios,
    validate_tasks,
    warehouse_clearance_risk,
)


OFFICIAL_COMMIT = "80bc741d4b9d2284df7cdc4ad4404de1ce4ac3c6"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_mocbs_instance(map_path, scenario_path, agents, output, task_indices=None,
                         blocked_edges=None):
    grid = load_movingai_map(map_path)
    available = load_movingai_scenarios(scenario_path)
    if task_indices is None:
        tasks = available[:int(agents)]
    else:
        task_indices = [int(index) for index in task_indices]
        if len(task_indices) != int(agents) or len(set(task_indices)) != len(task_indices):
            raise ValueError("task_indices must contain one distinct index per agent")
        tasks = [available[index] for index in task_indices]
    validate_tasks(grid, tasks, expected_map_name=Path(map_path).name)
    rows = ["".join("." if cell else "#" for cell in row) for row in grid]
    blocked_edges = tuple(blocked_edges or ())
    lines = [("PNSGA3_MOMAPF_V2" if blocked_edges else "PNSGA3_MOMAPF_V1"),
             f"{grid.shape[0]} {grid.shape[1]} {len(tasks)}"]
    lines.extend(rows)
    lines.extend(
        f"{task.start[0]} {task.start[1]} {task.goal[0]} {task.goal[1]}"
        for task in tasks
    )
    if blocked_edges:
        lines.append(str(len(blocked_edges)))
        for source, target in blocked_edges:
            source, target = tuple(source), tuple(target)
            if (abs(source[0] - target[0]) + abs(source[1] - target[1]) != 1
                    or any(not (0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]
                                and grid[y, x]) for x, y in (source, target))):
                raise ValueError(f"invalid closed physical edge: {source}, {target}")
            lines.append(f"{source[0]} {source[1]} {target[0]} {target[1]}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")
    return grid, tasks


def build_official_mocbs(source_dir, output_binary, cxx="/usr/bin/clang++"):
    source_dir = Path(source_dir).resolve()
    repo = source_dir.parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    if commit != OFFICIAL_COMMIT:
        raise RuntimeError(f"Expected official commit {OFFICIAL_COMMIT}, found {commit}")
    adapter = Path(__file__).with_name("mocbs_cli.cpp")
    sources = sorted((source_dir / "source").glob("*.cpp"))
    graph_source = source_dir / "source" / "graph.cpp"
    original = graph_source.read_text()
    target = "out.insert( this->v(ny,nx) );"
    if original.count(target) != 1:
        raise RuntimeError("MO-CBS graph.cpp edge-filter insertion point changed")
    overlay = ('extern "C" bool PNSGA3EdgeAllowed(long, long);\n' +
               original.replace(target,
                                "if (PNSGA3EdgeAllowed(v, this->v(ny,nx))) " + target))
    output_binary = Path(output_binary).resolve()
    output_binary.parent.mkdir(parents=True, exist_ok=True)
    command = [cxx, "-std=c++11", "-O2"]
    sdk = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk")
    if platform.system() == "Darwin" and sdk.is_dir():
        command += ["-isysroot", str(sdk), "-isystem", str(sdk / "usr/include/c++/v1")]
    with tempfile.TemporaryDirectory(prefix="pnsga3_mocbs_overlay_") as temp:
        overlay_file = Path(temp) / "graph.cpp"
        overlay_file.write_text(overlay)
        command += ["-I", str(source_dir / "include")]
        command += [str(overlay_file if path == graph_source else path)
                    for path in sources]
        command += [str(adapter), "-o", str(output_binary)]
        subprocess.run(command, check=True)
    return {
        "official_commit": commit,
        "official_graph_sha256": sha256(graph_source),
        "edge_filter_overlay_sha256": hashlib.sha256(overlay.encode()).hexdigest(),
        "adapter_sha256": sha256(adapter),
        "binary_sha256": sha256(output_binary),
        "compiler_command": command,
    }


def parse_mocbs_output(text):
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            records.append(json.loads(line))
    summaries = [record for record in records if record.get("type") == "summary"]
    if len(summaries) != 1:
        raise ValueError(f"Expected one summary record, found {len(summaries)}")
    return summaries[0], [
        record for record in records if record.get("type") == "solution"
    ]


def validate_mocbs_output(grid, tasks, summary, solutions, require_complete=True,
                          blocked_edges=None):
    if require_complete and not summary["find_all_pareto"]:
        raise ValueError("MO-CBS did not certify the complete Pareto set")
    if summary["solution_count"] != len(solutions):
        raise ValueError("Summary solution count does not match output")
    risk = warehouse_clearance_risk(grid)
    costs = []
    identities = set()
    closed = {frozenset((tuple(source), tuple(target)))
              for source, target in (blocked_edges or ())}
    for solution in solutions:
        paths = canonical_joint_paths(solution["paths"])
        if len(paths) != len(tasks) or not joint_paths_are_collision_free(paths):
            raise ValueError("MO-CBS returned an invalid joint path")
        for task, path in zip(tasks, paths):
            if path[0] != task.start or path[-1] != task.goal:
                raise ValueError("MO-CBS path endpoints do not match the task")
            for source, target in zip(path[:-1], path[1:]):
                if frozenset((tuple(source), tuple(target))) in closed:
                    raise ValueError("MO-CBS path crosses a closed physical edge")
                if abs(source[0] - target[0]) + abs(source[1] - target[1]) > 1:
                    raise ValueError("MO-CBS path contains a non-grid move")
                if not grid[target[1], target[0]]:
                    raise ValueError("MO-CBS path enters an obstacle")
        recomputed = np.asarray([
            sum(len(path) - 1 for path in paths),
            sum(risk[y, x] for path in paths for x, y in path[1:]),
        ], dtype=int)
        reported = np.asarray(solution["cost"], dtype=int)
        if not np.array_equal(recomputed, reported):
            raise ValueError(
                f"MO-CBS cost mismatch: reported {reported}, recomputed {recomputed}"
            )
        identity = repr(paths)
        if identity in identities:
            raise ValueError("MO-CBS returned a duplicate joint path")
        identities.add(identity)
        costs.append(reported)
    costs = np.asarray(costs, dtype=int)
    for i in range(len(costs)):
        dominated = np.all(costs <= costs[i], axis=1) & np.any(costs < costs[i], axis=1)
        if np.any(dominated):
            raise ValueError("MO-CBS output contains a dominated cost vector")
    return {"solution_count": len(solutions), "costs": costs}


def run_official_mocbs(binary, instance, time_limit, grid, tasks,
                       raw_output=None, require_complete=True, blocked_edges=None):
    process = subprocess.run(
        [str(Path(binary).resolve()), str(Path(instance).resolve()), str(float(time_limit))],
        text=True,
        capture_output=True,
        check=False,
    )
    if raw_output is not None:
        Path(raw_output).write_text(process.stdout)
    summary, solutions = parse_mocbs_output(process.stdout)
    validated = validate_mocbs_output(
        grid, tasks, summary, solutions, require_complete=require_complete,
        blocked_edges=blocked_edges,
    )
    if process.returncode not in (0, 3):
        raise RuntimeError(process.stderr or f"MO-CBS exited {process.returncode}")
    return summary, solutions, validated
