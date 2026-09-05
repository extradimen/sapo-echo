"""Append-only, publication-grade experiment archive utilities."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pymoo.core.callback import Callback


SCHEMA_VERSION = "1.0"


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True,
                                    default=_json_default) + "\n")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command, root):
    result = subprocess.run(["git", *command], cwd=root, text=True,
                            capture_output=True, check=False)
    return result.stdout.strip()


class ExperimentArchive:
    """Own one immutable experiment directory containing multiple runs."""

    def __init__(self, root, experiment_id, protocol, config):
        self.root = Path(root).resolve()
        self.path = self.root / "experiments" / experiment_id
        if self.path.exists():
            raise FileExistsError(f"Experiment already exists: {self.path}")
        self.path.mkdir(parents=True)
        (self.path / "runs").mkdir()
        (self.path / "source").mkdir()
        now = datetime.now(timezone.utc).isoformat()
        self.manifest = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "created_utc": now,
            "updated_utc": now,
            "status": "running",
            "evidence_grade": "A",
            "protocol": protocol,
            "runs": [],
        }
        atomic_json(self.path / "config.json", config)
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "executable": sys.executable,
            "numpy": np.__version__,
        }
        atomic_json(self.path / "environment.json", environment)
        repo = Path.cwd().resolve()
        atomic_json(self.path / "source" / "git.json", {
            "commit": _git(["rev-parse", "HEAD"], repo),
            "branch": _git(["branch", "--show-current"], repo),
            "status": _git(["status", "--short"], repo).splitlines(),
        })
        (self.path / "source" / "working_tree.patch").write_text(
            _git(["diff", "--binary", "HEAD"], repo) + "\n"
        )
        source_files = []
        roots = [repo / "pymoo", repo / "tests", repo / "config"]
        roots += [p for p in repo.iterdir() if p.is_file() and p.suffix in
                  {".py", ".md", ".toml", ".txt", ".yaml", ".yml"}]
        candidates = []
        for root in roots:
            if root.is_file():
                candidates.append(root)
            elif root.exists():
                candidates.extend(p for p in root.rglob("*") if p.is_file()
                                  and "__pycache__" not in p.parts)
        candidates = sorted(set(candidates))
        with tarfile.open(self.path / "source" / "source_tree.tar.gz", "w:gz") as tar:
            for source in candidates:
                relative = source.relative_to(repo)
                tar.add(source, arcname=str(relative))
                source_files.append({"path": str(relative), "sha256": sha256(source),
                                     "bytes": source.stat().st_size})
        atomic_json(self.path / "source" / "files.json", source_files)
        self._write_manifest()

    def _write_manifest(self):
        self.manifest["updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(self.path / "manifest.json", self.manifest)

    def new_run(self, run_id, metadata, snapshot_interval=1,
                snapshot_evaluation_interval=None):
        run_path = self.path / "runs" / run_id
        if run_path.exists():
            raise FileExistsError(f"Run already exists: {run_id}")
        (run_path / "trajectory").mkdir(parents=True)
        atomic_json(run_path / "metadata.json", metadata)
        record = {"run_id": run_id, "status": "running",
                  "started_utc": datetime.now(timezone.utc).isoformat()}
        self.manifest["runs"].append(record)
        self._write_manifest()
        return RunRecorder(self, run_path, record, snapshot_interval,
                           snapshot_evaluation_interval)

    def finish(self, status="complete"):
        self.manifest["status"] = status
        self._write_manifest()
        self.write_checksums()

    def write_checksums(self):
        checksum_file = self.path / "SHA256SUMS"
        files = sorted(p for p in self.path.rglob("*")
                       if p.is_file() and p != checksum_file)
        checksum_file.write_text("".join(
            f"{sha256(path)}  {path.relative_to(self.path)}\n" for path in files
        ))


class RunRecorder(Callback):
    """Record every generation plus algorithm-specific island state."""

    def __init__(self, archive, path, manifest_record, snapshot_interval=1,
                 snapshot_evaluation_interval=None):
        super().__init__()
        self.archive = archive
        self.path = Path(path)
        self.manifest_record = manifest_record
        self.snapshot_interval = max(1, int(snapshot_interval))
        self.snapshot_evaluation_interval = (
            None if snapshot_evaluation_interval is None
            else max(1, int(snapshot_evaluation_interval)))
        self._last_snapshot_evaluation = -1

    def _population_payload(self, pop, prefix=""):
        payload = {}
        if pop is None:
            return payload
        for name in ("X", "F", "G", "H", "CV", "FEAS", "island", "niche",
                     "dist_to_niche", "rank"):
            try:
                value = pop.get(name)
            except Exception:
                value = None
            if value is not None:
                array = np.asarray(value)
                if array.dtype == object and all(v is None for v in array.reshape(-1)):
                    continue
                payload[prefix + name] = array
        return payload

    def notify(self, algorithm):
        generation = int(algorithm.n_iter)
        if self.snapshot_evaluation_interval is None:
            if generation % self.snapshot_interval:
                return
        else:
            evaluation = int(algorithm.evaluator.n_eval)
            bucket = evaluation // self.snapshot_evaluation_interval
            if bucket <= self._last_snapshot_evaluation:
                return
            self._last_snapshot_evaluation = bucket
        payload = self._population_payload(algorithm.pop)
        archive = getattr(algorithm, "basin_archive", None)
        payload.update(self._population_payload(archive, "archive_"))
        payload.update(self._population_payload(
            getattr(algorithm, "external_archive", None), "external_archive_"))
        payload.update(self._population_payload(
            getattr(algorithm, "exploration_memory", None), "memory_"))
        payload["generation"] = np.asarray(generation)
        payload["n_eval"] = np.asarray(algorithm.evaluator.n_eval)
        payload["current_basin_radius"] = np.asarray(
            getattr(algorithm, "current_basin_radius", np.nan))
        payload["objective_equivalence_radius"] = np.asarray(
            getattr(algorithm, "current_objective_equivalence_radius", np.nan))
        payload["explorer_restart_counts"] = np.asarray(
            getattr(algorithm, "explorer_restart_counts", []))
        payload["explorer_niche_reassignments"] = np.asarray(
            getattr(algorithm, "explorer_niche_reassignments", 0))
        payload["explorer_to_main_exchange_events"] = np.asarray(
            getattr(algorithm, "explorer_to_main_exchange_events", 0))
        payload["explorer_immigrant_evaluations"] = np.asarray(
            getattr(algorithm, "explorer_immigrant_evaluations", 0))
        payload["explorer_novelty_evaluations"] = np.asarray(
            getattr(algorithm, "explorer_novelty_evaluations", 0))
        payload["explorer_incubator_restarts"] = np.asarray(
            getattr(algorithm, "explorer_incubator_restarts", 0))
        payload["explorer_incubator_evaluations"] = np.asarray(
            getattr(algorithm, "explorer_incubator_evaluations", 0))
        payload["explorer_immigrant_events"] = np.asarray(
            getattr(algorithm, "explorer_immigrant_events", 0))
        payload["explorer_spread_max"] = np.asarray(
            getattr(algorithm, "_explorer_spread_max", []))
        payload["explorer_collapse_streak"] = np.asarray(
            getattr(algorithm, "_explorer_collapse_streak", []))
        payload["archive_update_events"] = np.asarray(
            getattr(algorithm, "archive_update_events", 0))
        payload["explorer_overlap_streak"] = np.asarray(
            getattr(algorithm, "explorer_overlap_streak", 0))
        payload["current_explorer_overlap"] = np.asarray(
            getattr(algorithm, "current_explorer_overlap", np.nan))
        payload["structure_observation_events"] = np.asarray(
            getattr(algorithm, "structure_observation_events", 0))
        payload["structure_protection_events"] = np.asarray(
            getattr(algorithm, "structure_protection_events", 0))
        payload["structure_protected_survivors"] = np.asarray(
            getattr(algorithm, "structure_protected_survivors", 0))
        payload["structure_reproduction_events"] = np.asarray(
            getattr(algorithm, "structure_reproduction_events", 0))
        payload["structure_reserved_offspring"] = np.asarray(
            getattr(algorithm, "structure_reserved_offspring", 0))
        payload["structure_branch_count"] = np.asarray(
            getattr(algorithm, "current_structure_branch_count", 0))
        payload["structure_active_count"] = np.asarray(
            getattr(algorithm, "current_structure_active_count", 0))
        payload["structure_gap_ratio"] = np.asarray(
            getattr(algorithm, "current_structure_gap_ratio", np.nan))
        np.savez_compressed(
            self.path / "trajectory" / f"generation_{generation:06d}.npz", **payload
        )
        with open(self.path / "events.jsonl", "a") as stream:
            stream.write(json.dumps({
                "event": "generation_snapshot", "generation": generation,
                "n_eval": int(algorithm.evaluator.n_eval),
                "island_counts": {
                    str(i): int(np.sum(payload.get("island", []) == i))
                    for i in np.unique(payload.get("island", [])).astype(int)
                } if "island" in payload else {},
                "archive_size": len(archive) if archive is not None else 0,
            }) + "\n")

    def complete(self, result, metrics):
        payload = self._population_payload(result.pop)
        archive = getattr(result.algorithm, "basin_archive", None)
        payload.update(self._population_payload(archive, "archive_"))
        payload.update(self._population_payload(
            getattr(result.algorithm, "external_archive", None),
            "external_archive_"))
        payload.update(self._population_payload(
            getattr(result.algorithm, "exploration_memory", None), "memory_"))
        if result.X is not None:
            payload["result_X"] = np.asarray(result.X)
        if result.F is not None:
            payload["result_F"] = np.asarray(result.F)
        # Preserve the exact declared approximation set after any lazy bounded
        # output composition, separately from the evolving final population.
        try:
            # Some benchmark protocols (notably CEC local Pareto-set tasks)
            # need a pre-registered output rule that deliberately retains
            # locally efficient but globally dominated alternatives.  Allow a
            # runner to preserve that exact scored set without changing the
            # generic global-nondominated reporting rule.
            declared = getattr(result, "archive_declared_output", None)
            if declared is None:
                from multimodal_experiment import _solution_set
                output_x, output_f = _solution_set(result)
            else:
                output_x, output_f = declared
            payload["declared_output_X"] = np.asarray(output_x)
            payload["declared_output_F"] = np.asarray(output_f)
        except Exception:
            pass
        np.savez_compressed(self.path / "final.npz", **payload)
        atomic_json(self.path / "metrics.json", metrics)
        self.manifest_record.update({
            "status": "complete",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "n_eval": int(result.algorithm.evaluator.n_eval),
        })
        self.archive._write_manifest()

    def fail(self, error):
        self.manifest_record.update({
            "status": "failed", "error": repr(error),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
        self.archive._write_manifest()
