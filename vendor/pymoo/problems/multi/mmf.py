"""Public multimodal multi-objective MMF benchmark problems.

The mathematical definitions follow the MMF suite introduced by Yue et al.
The implementations below are independent NumPy implementations and expose
the known, equivalent Pareto subsets in decision space for IGDX/coverage tests.

Reference
---------
Yue, C., Qu, B., Liang, J. (2018). A Multiobjective Particle Swarm Optimizer
Using Ring Topology for Solving Multimodal Multiobjective Problems. IEEE
Transactions on Evolutionary Computation, 22(5), 805-817.
"""

import numpy as np

from pymoo.core.problem import Problem


class _MMF(Problem):
    n_basins = None

    def __init__(self, xl, xu):
        super().__init__(n_var=2, n_obj=2, xl=np.asarray(xl), xu=np.asarray(xu))

    def pareto_set_by_basin(self, points_per_basin=200):
        """Return one contiguous block of points for each Pareto subset."""
        t = np.linspace(0.0, 1.0, points_per_basin)
        return np.vstack(self._pareto_subsets(t))

    def _calc_pareto_set(self, n_pareto_points=500):
        per_basin = max(2, int(np.ceil(n_pareto_points / self.n_basins)))
        return self.pareto_set_by_basin(per_basin)[:n_pareto_points]

    def _calc_pareto_front(self, n_pareto_points=500):
        t = np.linspace(0.0, 1.0, n_pareto_points)
        return self._front(t)

    @staticmethod
    def _upper_branch(values):
        """Keep points at the piecewise boundary on the upper branch."""
        values = np.asarray(values).copy()
        values[np.isclose(values, 1.0, atol=1e-14)] = np.nextafter(1.0, 2.0)
        return values


class MMF1(_MMF):
    n_basins = 2

    def __init__(self):
        super().__init__([1.0, -1.0], [3.0, 1.0])

    def _evaluate(self, X, out, *args, **kwargs):
        u = np.abs(X[:, 0] - 2.0)
        y = X[:, 1] - np.sin(6.0 * np.pi * u + np.pi)
        out["F"] = np.column_stack((u, 1.0 - np.sqrt(u) + 2.0 * y ** 2))

    def _pareto_subsets(self, u):
        y = np.sin(6.0 * np.pi * u + np.pi)
        return [np.column_stack((2.0 - u, y)), np.column_stack((2.0 + u, y))]

    def _front(self, u):
        return np.column_stack((u, 1.0 - np.sqrt(u)))


class MMF2(_MMF):
    n_basins = 2

    def __init__(self):
        super().__init__([0.0, 0.0], [1.0, 2.0])

    def _evaluate(self, X, out, *args, **kwargs):
        x1, x2 = X[:, 0], X[:, 1]
        target = np.where(x2 <= 1.0, np.sqrt(x1), 1.0 + np.sqrt(x1))
        y = x2 - target
        penalty = 4.0 * y ** 2 - 2.0 * np.cos(20.0 * np.pi * y / np.sqrt(2.0)) + 2.0
        out["F"] = np.column_stack((x1, 1.0 - np.sqrt(x1) + 2.0 * penalty))

    def _pareto_subsets(self, x1):
        root = np.sqrt(x1)
        return [np.column_stack((x1, root)),
                np.column_stack((x1, self._upper_branch(1.0 + root)))]

    def _front(self, x1):
        return np.column_stack((x1, 1.0 - np.sqrt(x1)))


class MMF4(_MMF):
    # The negative and positive x1 branches meet at x1=0.  They are two
    # pieces of the same connected Pareto set, not separate modalities.
    n_basins = 2

    def __init__(self):
        super().__init__([-1.0, 0.0], [1.0, 2.0])

    def _evaluate(self, X, out, *args, **kwargs):
        x1, x2 = X[:, 0], X[:, 1]
        u = np.abs(x1)
        folded_x2 = np.where(x2 <= 1.0, x2, x2 - 1.0)
        y = folded_x2 - np.sin(np.pi * u)
        out["F"] = np.column_stack((u, 1.0 - x1 ** 2 + 2.0 * y ** 2))

    def _pareto_subsets(self, u):
        x1 = np.linspace(-1.0, 1.0, len(u))
        y = np.sin(np.pi * np.abs(x1))
        return [np.column_stack((x1, y)),
                np.column_stack((x1, self._upper_branch(y + 1.0)))]

    def _front(self, u):
        return np.column_stack((u, 1.0 - u ** 2))


class MMF5(_MMF):
    n_basins = 2

    def __init__(self):
        super().__init__([1.0, -1.0], [3.0, 3.0])

    def _evaluate(self, X, out, *args, **kwargs):
        x1, x2 = X[:, 0], X[:, 1]
        u = np.abs(x1 - 2.0)
        folded_x2 = np.where(x2 <= 1.0, x2, x2 - 2.0)
        y = folded_x2 - np.sin(6.0 * np.pi * u + np.pi)
        out["F"] = np.column_stack((u, 1.0 - np.sqrt(u) + 2.0 * y ** 2))

    def _pareto_subsets(self, u):
        x1 = np.linspace(1.0, 3.0, len(u))
        y = np.sin(6.0 * np.pi * np.abs(x1 - 2.0) + np.pi)
        return [np.column_stack((x1, y)),
                np.column_stack((x1, self._upper_branch(y + 2.0)))]

    def _front(self, u):
        return np.column_stack((u, 1.0 - np.sqrt(u)))


class MMF7(_MMF):
    n_basins = 2

    def __init__(self):
        super().__init__([1.0, -1.0], [3.0, 1.0])

    @staticmethod
    def _target(u):
        return (0.3 * u ** 2 * np.cos(24.0 * np.pi * u + 4.0 * np.pi) + 0.6 * u) * \
            np.sin(6.0 * np.pi * u + np.pi)

    def _evaluate(self, X, out, *args, **kwargs):
        u = np.abs(X[:, 0] - 2.0)
        y = X[:, 1] - self._target(u)
        out["F"] = np.column_stack((u, 1.0 - np.sqrt(u) + y ** 2))

    def _pareto_subsets(self, u):
        y = self._target(u)
        return [np.column_stack((2.0 - u, y)), np.column_stack((2.0 + u, y))]

    def _front(self, u):
        return np.column_stack((u, 1.0 - np.sqrt(u)))


class MMF1E(MMF1):
    def __init__(self):
        # The CEC 2020 technical report defines a=e and
        # x2 in [-a^3, a^3].  The MATLAB function's [-1,1] comment is stale.
        _MMF.__init__(self, [1.0, -np.e ** 3], [3.0, np.e ** 3])

    def _evaluate(self, X, out, *args, **kwargs):
        x1, x2 = X[:, 0], X[:, 1]
        u = np.abs(x1 - 2.0)
        target = np.where(x1 < 2.0, np.sin(6 * np.pi * u + np.pi),
                          np.exp(x1) * np.sin(6 * np.pi * u + np.pi))
        out["F"] = np.column_stack((u, 1 - np.sqrt(u) + 2 * (x2 - target) ** 2))

    def _pareto_subsets(self, u):
        wave = np.sin(6.0 * np.pi * u + np.pi)
        return [np.column_stack((2.0 - u, wave)),
                np.column_stack((2.0 + u, np.exp(2.0 + u) * wave))]


class MMF8(_MMF):
    n_basins = 2

    def __init__(self):
        super().__init__([-np.pi, 0.0], [np.pi, 9.0])

    def _evaluate(self, X, out, *args, **kwargs):
        u = np.abs(X[:, 0]); folded = np.where(X[:, 1] > 4, X[:, 1] - 4, X[:, 1])
        f1 = np.sin(u)
        out["F"] = np.column_stack((f1, np.sqrt(1 - f1 ** 2)
                                     + 2 * (folded - (f1 + u)) ** 2))

    def _pareto_subsets(self, t):
        x1 = np.linspace(-.5 * np.pi, .5 * np.pi, len(t))
        u = np.abs(x1)
        y = np.sin(u) + u
        return [np.column_stack((x1, y)),
                np.column_stack((x1, y + 4.0))]

    def _front(self, t):
        f1 = np.sin(.5 * np.pi * t)
        return np.column_stack((f1, np.sqrt(1 - f1 ** 2)))


class _MMFLocal2D(Problem):
    def __init__(self, kind, include_local=False):
        self.kind, self.include_local = int(kind), bool(include_local)
        n_var = 3 if kind == 13 else 2
        lower = np.full(n_var, .1) if kind in (10, 11, 13) else np.zeros(n_var)
        upper = np.full(n_var, 1.1) if kind in (10, 11, 13) else np.ones(n_var)
        super().__init__(n_var=n_var, n_obj=2, xl=lower, xu=upper)
        self.n_basins = 2 if include_local else 1

    def _evaluate(self, X, out, *args, **kwargs):
        x1, x2 = X[:, 0], X[:, 1]
        if self.kind == 10:
            g = 2 - np.exp(-((x2 - .2) / .004) ** 2) - .8 * np.exp(-((x2 - .6) / .4) ** 2)
            F = np.column_stack((x1, g / x1))
        elif self.kind in (11, 12):
            g = 2 - np.sin(2 * np.pi * x2) ** 6 * np.exp(-2 * np.log10(2) * ((x2 - .1) / .8) ** 2)
            if self.kind == 11:
                F = np.column_stack((x1, g / x1))
            else:
                h = 1 - (x1 / g) ** 2 - (x1 / g) * np.sin(8 * np.pi * x1)
                F = np.column_stack((x1, g * h))
        else:
            z = x2 + np.sqrt(X[:, 2])
            g = 2 - np.exp(-2 * np.log10(2) * ((z - .1) / .8) ** 2) * np.sin(2 * np.pi * z) ** 6
            F = np.column_stack((x1, g / x1))
        out["F"] = F


class MMF10(_MMFLocal2D):
    def __init__(self): super().__init__(10, False)
class MMF10L(_MMFLocal2D):
    def __init__(self): super().__init__(10, True)
class MMF11(_MMFLocal2D):
    def __init__(self): super().__init__(11, False)
class MMF11L(_MMFLocal2D):
    def __init__(self): super().__init__(11, True)
class MMF12(_MMFLocal2D):
    def __init__(self): super().__init__(12, False)
class MMF12L(_MMFLocal2D):
    def __init__(self): super().__init__(12, True)
class MMF13(_MMFLocal2D):
    def __init__(self): super().__init__(13, False)
class MMF13L(_MMFLocal2D):
    def __init__(self): super().__init__(13, True)


def _dtlz_sphere(X, g, n_obj):
    angles = X[:, :n_obj - 1] * np.pi / 2
    F = np.ones((len(X), n_obj)) * (1 + g[:, None])
    for objective in range(n_obj):
        for j in range(n_obj - objective - 1):
            F[:, objective] *= np.cos(angles[:, j])
        if objective > 0:
            F[:, objective] *= np.sin(angles[:, n_obj - objective - 1])
    return F


class _MMFScalable(Problem):
    def __init__(self, kind, n_obj=3, n_var=None, num_peaks=2,
                 nonlinear=False, include_local=False, local_counts=None):
        self.kind, self.num_peaks = int(kind), int(num_peaks)
        self.nonlinear, self.include_local = bool(nonlinear), bool(include_local)
        self.local_counts = local_counts
        n_var = n_obj if n_var is None else int(n_var)
        super().__init__(n_var=n_var, n_obj=n_obj, xl=np.zeros(n_var), xu=np.ones(n_var))
        self.n_basins = (sum(local_counts) if local_counts else
                         (num_peaks if kind == 14 else (2 if include_local else 1)))

    def _evaluate(self, X, out, *args, **kwargs):
        z = X[:, -1]
        if self.nonlinear:
            transformed = z - .5 * np.sin(np.pi * X[:, -2])
        else:
            transformed = z
        if self.kind == 14:
            phase = transformed + (1 / (2 * self.num_peaks) if self.nonlinear else 0)
            g = 2 - np.sin(self.num_peaks * np.pi * phase) ** 2
        elif self.kind == 15:
            phase = transformed + (1 / (2 * self.num_peaks) if self.nonlinear else 0)
            g = 2 - np.exp(-2 * np.log10(2) * ((phase - .1) / .8) ** 2) * np.sin(self.num_peaks * np.pi * phase) ** 2
        else:
            global_count, local_count = self.local_counts
            g = np.where(z < .5,
                2 - np.sin(2 * global_count * np.pi * z) ** 2,
                2 - np.exp(-2 * np.log10(2) * ((z - .1) / .8) ** 2)
                    * np.sin(2 * local_count * np.pi * z) ** 2)
        out["F"] = _dtlz_sphere(X, g, self.n_obj)


class MMF14(_MMFScalable):
    def __init__(self, n_obj=3, n_var=None, num_peaks=2): super().__init__(14, n_obj, n_var, num_peaks)
class MMF14A(_MMFScalable):
    def __init__(self, n_obj=3, n_var=None, num_peaks=2): super().__init__(14, n_obj, n_var, num_peaks, nonlinear=True)
class MMF15(_MMFScalable):
    def __init__(self, n_obj=3, n_var=None, num_peaks=2): super().__init__(15, n_obj, n_var, num_peaks)
class MMF15L(_MMFScalable):
    def __init__(self, n_obj=3, n_var=None, num_peaks=2): super().__init__(15, n_obj, n_var, num_peaks, include_local=True)
class MMF15A(_MMFScalable):
    def __init__(self, n_obj=3, n_var=None, num_peaks=2): super().__init__(15, n_obj, n_var, num_peaks, nonlinear=True)
class MMF15AL(_MMFScalable):
    def __init__(self, n_obj=3, n_var=None, num_peaks=2): super().__init__(15, n_obj, n_var, num_peaks, nonlinear=True, include_local=True)
class MMF16L1(_MMFScalable):
    def __init__(self, n_obj=3, n_var=None): super().__init__(16, n_obj, n_var, local_counts=(2, 1))
class MMF16L2(_MMFScalable):
    def __init__(self, n_obj=3, n_var=None): super().__init__(16, n_obj, n_var, local_counts=(1, 2))
class MMF16L3(_MMFScalable):
    def __init__(self, n_obj=3, n_var=None): super().__init__(16, n_obj, n_var, local_counts=(2, 2))
