"""Discrete hardware-aware NAS problem backed by public benchmark tables."""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

from pymoo.core.problem import Problem
from pymoo.core.duplicate import DuplicateElimination


NASBENCH201_OPERATIONS = (
    "none",
    "skip_connect",
    "nor_conv_1x1",
    "nor_conv_3x3",
    "avg_pool_3x3",
)
_OPERATION_INDEX = {name: index for index, name in enumerate(NASBENCH201_OPERATIONS)}


def parse_architecture(architecture):
    """Convert a NAS-Bench-201 architecture string to six categorical genes."""
    operations = re.findall(r"([a-z0-9_]+)~[0-2]", architecture)
    if len(operations) != 6:
        raise ValueError(f"Expected six edge operations, found {len(operations)}")
    try:
        return tuple(_OPERATION_INDEX[operation] for operation in operations)
    except KeyError as error:
        raise ValueError(f"Unknown NAS-Bench-201 operation: {error.args[0]}") from error


def load_hw_nas_table(path, dataset="cifar100"):
    """Load architecture encodings and hardware columns from HW-NAS-Bench."""
    with Path(path).open("rb") as stream:
        payload = pickle.load(stream)
    table = payload["nasbench201"][dataset]
    architectures = np.asarray(
        [parse_architecture(config["arch_str"]) for config in table["config"]],
        dtype=np.int8,
    )
    metrics = {
        name: np.asarray(values, dtype=float)
        for name, values in table.items()
        if name != "config"
    }
    return architectures, metrics


class HWNASBenchProblem(Problem):
    """Lookup problem for accuracy, latency, energy, and related objectives.

    ``accuracy`` must be aligned with the architecture indices in the official
    HW-NAS-Bench table. Objectives are minimized; ``error`` is represented as
    ``100 - accuracy``. The full architecture table is intentionally accepted
    as an argument so unit tests and smoke runs do not depend on a large
    external benchmark download.
    """

    def __init__(self, architectures, accuracy, hardware_metrics,
                 objectives=("error", "edgegpu_latency", "edgegpu_energy"),
                 canonical_structures=None):
        architectures = np.asarray(architectures, dtype=int)
        if architectures.ndim != 2 or architectures.shape[1] != 6:
            raise ValueError("architectures must have shape (n, 6)")
        if np.any((architectures < 0) | (architectures >= len(NASBENCH201_OPERATIONS))):
            raise ValueError("architecture genes must be operation indices 0..4")
        accuracy = np.asarray(accuracy, dtype=float)
        if accuracy.shape != (len(architectures),):
            raise ValueError("accuracy must contain one value per architecture")

        available = {"error": 100.0 - accuracy}
        for name, values in hardware_metrics.items():
            values = np.asarray(values, dtype=float)
            if values.shape != (len(architectures),):
                raise ValueError(f"metric {name!r} has the wrong length")
            available[name] = values
        missing = [name for name in objectives if name not in available]
        if missing:
            raise ValueError(f"Unknown objectives: {missing}")

        self.architectures = architectures
        if canonical_structures is None:
            canonical_structures = [",".join(map(str, row)) for row in architectures]
        canonical_structures = np.asarray(canonical_structures, dtype=str)
        if canonical_structures.shape != (len(architectures),):
            raise ValueError("canonical_structures must contain one signature per architecture")
        self.canonical_structures = canonical_structures
        representative = {}
        for architecture, signature in zip(architectures, canonical_structures):
            key = tuple(architecture)
            if signature not in representative or key < representative[signature]:
                representative[signature] = key
        self.canonical_architectures = np.asarray(
            [representative[signature] for signature in canonical_structures], dtype=int
        )
        self.objective_names = tuple(objectives)
        self.objective_table = np.column_stack([available[name] for name in objectives])
        self._index = {tuple(row): index for index, row in enumerate(architectures)}
        self.queried_indices = set()
        if len(self._index) != len(architectures):
            raise ValueError("benchmark contains duplicate architecture encodings")

        super().__init__(
            n_var=6,
            n_obj=len(objectives),
            xl=np.zeros(6, dtype=int),
            xu=np.full(6, len(NASBENCH201_OPERATIONS) - 1, dtype=int),
            vtype=int,
        )

    def architecture_indices(self, X):
        X = np.rint(np.asarray(X)).astype(int)
        if X.ndim == 1:
            X = X[None, :]
        indices = []
        for row in X:
            try:
                indices.append(self._index[tuple(row)])
            except KeyError as error:
                raise ValueError(f"Architecture is outside the benchmark table: {tuple(row)}") from error
        return np.asarray(indices, dtype=int)

    def _evaluate(self, X, out, *args, **kwargs):
        indices = self.architecture_indices(X)
        self.queried_indices.update(indices.tolist())
        out["F"] = self.objective_table[indices]

    def decision_space_embedding(self, X):
        """One-hot embedding of a fixed representative of each effective graph."""
        X = np.rint(np.asarray(X)).astype(int)
        if X.ndim == 1:
            X = X[None, :]
        X = self.canonical_architectures[self.architecture_indices(X)]
        eye = np.eye(len(NASBENCH201_OPERATIONS), dtype=float)
        embedding = eye[X].reshape(len(X), -1)
        return embedding / np.sqrt(2.0 * X.shape[1])

    def decision_identity(self, X):
        return self.canonical_structures[self.architecture_indices(X)]

    def pareto_front(self, *args, **kwargs):
        from moocore import is_nondominated
        indices = np.where(is_nondominated(self.objective_table))[0]
        return self.objective_table[indices]

    def pareto_set(self, *args, **kwargs):
        from moocore import is_nondominated
        indices = np.where(is_nondominated(self.objective_table))[0]
        return self.architectures[indices]


class CanonicalArchitectureDuplicateElimination(DuplicateElimination):
    """Treat encodings of the same effective computation graph as duplicates."""

    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def _do(self, pop, other, is_duplicate):
        seen = set()
        if other is not None and len(other):
            seen.update(self.problem.decision_identity(other.get("X")).tolist())
        for index, signature in enumerate(
                self.problem.decision_identity(pop.get("X")).tolist()):
            if signature in seen:
                is_duplicate[index] = True
            else:
                seen.add(signature)
        return is_duplicate
