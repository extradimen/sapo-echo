"""Thin pymoo adapter for the pinned official BONO-Bench package."""

from __future__ import annotations

import numpy as np

from pymoo.core.problem import Problem


BONO_CLASSES = {
    8: "MultimodalAxisAlignedSphere",
    9: "MultimodalAxisAlignedEllipsoid",
    10: "MultimodalConvexFrontEllipsoid",
    11: "MultimodalLinearFrontEllipsoid",
    12: "MultimodalConcaveFrontEllipsoid",
    13: "MultimodalFreeEllipsoid",
    14: "MultimodalSteppedEllipsoid",
    15: "FewSpheres",
    16: "ManySpheres",
    17: "SteppedManySpheres",
    18: "FewEllipsoids",
    19: "ManyEllipsoids",
    20: "SteppedManyEllipsoids",
}


class BONOBenchProblem(Problem):
    """Expose one deterministic official BONO8--20 instance to pymoo.

    The external package owns all equations and instance generation.  This
    adapter supplies only pymoo bounds and array-shape conversion.
    """

    def __init__(self, number, n_var=10, instance_seed=0):
        number = int(number)
        if number not in BONO_CLASSES:
            raise ValueError("BONO number must be in 8..20")
        try:
            from bonobench import benchmark_sets
        except ImportError as error:
            raise ImportError(
                "Install the pinned official bonobench package in .venv-bono"
            ) from error
        generator = getattr(benchmark_sets, BONO_CLASSES[number])(int(n_var))
        self.bono_problem = generator.generate_instance(int(instance_seed))
        self.number = number
        self.instance_seed = int(instance_seed)
        super().__init__(n_var=int(n_var), n_obj=2,
                         xl=np.full(int(n_var), -5.0),
                         xu=np.full(int(n_var), 5.0))

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.column_stack(self.bono_problem.bulk_eval(np.asarray(X)))

    def active_peak_pairs(self, X):
        """Return the objective-wise active official peak index per row."""
        X = np.asarray(X, dtype=float)
        objectives = (self.bono_problem.get_f1(),
                      self.bono_problem.get_f2())
        active = []
        for objective in objectives:
            raw = np.vstack([peak.bulk_eval(X)
                             for peak in objective.get_peaks()])
            active.append(np.argmin(raw, axis=0))
        return np.column_stack(active)

    def approximate_reference(self, delta_hv=1e-4, delta_r2=1e-5):
        return self.bono_problem.approximate_indicators(
            delta_hv=delta_hv, delta_r2=delta_r2,
            keep_dec=True, log=False)
