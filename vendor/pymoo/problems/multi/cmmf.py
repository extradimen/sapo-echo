"""Independent NumPy implementation of the official CEC 2023 CMMF1--5 suite.

The definitions are translated from the organizer-provided PlatEMO files.
Constraints follow pymoo/PlatEMO convention: a point is feasible when every
entry in ``G`` is less than or equal to zero.
"""

import numpy as np

from pymoo.core.problem import Problem


def _theta_abs(X):
    theta = np.ones(len(X))
    nz = X[:, 0] != 0
    theta[nz] = 2 / np.pi * np.arctan(np.abs(X[nz, 1]) / np.abs(X[nz, 0]))
    return theta


def _safe_ratio_theta(y, x):
    """Match atan(abs(y/x)) while defining the axis limit without warnings."""
    ratio = np.divide(np.abs(y), np.abs(x), out=np.full_like(y, np.inf), where=x != 0)
    ratio[(x == 0) & (y == 0)] = 0
    return 2 / np.pi * np.arctan(ratio)


class CMMF(Problem):
    """Base class for the two-objective, ten-variable CMMF problems."""

    problem_number = None
    n_constraints = None

    def __init__(self, n_var=10):
        if n_var < 2:
            raise ValueError("CMMF requires at least two variables")
        super().__init__(n_var=n_var, n_obj=2,
                         n_ieq_constr=self.n_constraints,
                         xl=-np.ones(n_var), xu=np.ones(n_var))

    @staticmethod
    def _tail_distance(X):
        return np.sum((X[:, 2:] - 0.2) ** 2, axis=1)


class CMMF1(CMMF):
    problem_number, n_constraints = 1, 6

    def _evaluate(self, X, out, *args, **kwargs):
        P = X.copy()
        P[P[:, 0] <= 0, 0] += 1
        sx = np.sum(P[:, :2] ** 2, axis=1)
        theta = _safe_ratio_theta(P[:, 1], P[:, 0])
        t = (0.98 - sx) ** 2 + self._tail_distance(X)
        out["F"] = np.column_stack((1 - theta, theta)) * (1 + t[:, None])
        G = np.zeros((len(X), 6))
        for i, (x, y, s, th) in enumerate(zip(X[:, 0], X[:, 1], sx, theta)):
            G[i, :2] = s - .36, -s + .04
            if s <= .36:
                continue
            if x < 0 and y < 0:
                if th <= .5:
                    G[i, 2:] = th - .25, -s + .96, s - 1, 0
                else:
                    G[i, 2:] = -th + .5, th - .75, -s + .96, s - 1
            elif x >= 0 and y >= 0:
                if th <= .5:
                    G[i, 2:] = -th, th - .25, -s + .96, s - 1
                else:
                    G[i, 2:] = -th + .5, th - .75, -s + .96, s - 1
            else:
                if th <= .5:
                    G[i, 2:] = -th + .25, s - 1, -s + .96, 0
                else:
                    G[i, 2:] = -th + .75, s - 1, -s + .96, 0
        out["G"] = G


class CMMF2(CMMF):
    problem_number, n_constraints = 2, 3

    def _evaluate(self, X, out, *args, **kwargs):
        P = X.copy()
        P[P[:, 0] <= 0, 0] = np.abs(P[P[:, 0] <= 0, 0])
        sx = np.sum(X[:, :2] ** 2, axis=1)
        theta = _safe_ratio_theta(P[:, 1], P[:, 0])
        Z = P[:, 2:] - .2
        h = np.sum(100 * (Z[:, :-1] ** 2 - Z[:, 1:]) ** 2 + Z[:, :-1] ** 2, axis=1)
        target = np.where(X[:, 0] < 0,
                          (1 / 16 - X[:, 0] ** 2 - .25 * X[:, 1] ** 2) ** 2,
                          (.25 - X[:, 0] ** 2 - .25 * X[:, 1] ** 2) ** 2)
        out["F"] = np.column_stack((np.cos(np.pi / 2 * theta),
                                     np.sin(np.pi / 2 * theta))) * (1 + target[:, None] + h[:, None])
        G = np.zeros((len(X), 3))
        for i, (x, y, s, th) in enumerate(zip(X[:, 0], X[:, 1], sx, theta)):
            if x < 0 and y >= 0:
                if th <= .5:
                    G[i] = -s - .04, s - .25, -th
                else:
                    q = (x + .2) ** 2 + (y - .8) ** 2
                    G[i] = q - .04, -q, 0
            elif x < 0 and y < 0:
                G[i] = -s - .04, s - .25, -th + .5
            elif x >= 0 and y >= 0:
                G[i] = -th + .5, -s + .36, s - 1
            else:
                G[i] = th - .5, -s + .16, s - .49
        out["G"] = G


class CMMF3(CMMF):
    problem_number, n_constraints = 3, 3

    def _evaluate(self, X, out, *args, **kwargs):
        sx = np.sum(X[:, :2] ** 2, axis=1)
        theta = np.empty(len(X))
        for i, (x, y) in enumerate(X[:, :2]):
            if x > -.5:
                theta[i] = 2 / np.pi * np.arctan(np.abs(y) / np.abs(x)) if x != 0 else 1
            elif x == -.5:
                theta[i] = 1
            else:
                theta[i] = 2 / np.pi * np.arctan(np.abs(y) / np.abs(x + .5))
        h = self._tail_distance(X)
        target = np.where(X[:, 0] < 0,
                          (-.96 - X[:, 0] - X[:, 1]) ** 2,
                          (.81 - X[:, 0] - X[:, 1]) ** 2)
        base = 1 - np.column_stack((np.cos(np.pi / 2 * theta),
                                    np.sin(np.pi / 2 * theta)))
        out["F"] = base * (1 + target[:, None] + h[:, None])
        G = np.zeros((len(X), 3))
        for i, (x, y, s, th) in enumerate(zip(X[:, 0], X[:, 1], sx, theta)):
            if x < 0 and y >= 0:
                G[i] = -s, s - .25, 0
            elif x <= 0 and y <= 0:
                G[i] = x + y + .9, -x - y - 1, x + .5
            elif x >= 0 and y >= 0:
                G[i] = -th + .5, x + y - 1, -x - y + .64
            else:
                G[i] = -s, s - .36, 0
        out["G"] = G


class CMMF4(CMMF):
    problem_number, n_constraints = 4, 3

    def _evaluate(self, X, out, *args, **kwargs):
        theta = np.empty(len(X))
        for i, (x, y) in enumerate(X[:, :2]):
            if x > 0:
                theta[i] = 2 / np.pi * np.arctan(np.abs(y) / np.abs(x))
            elif x == 0 or x == -1:
                theta[i] = 1
            else:
                theta[i] = 2 / np.pi * np.arctan(np.abs(y) / (x + 1))
        radius = np.where(X[:, 0] >= 0,
                          X[:, 0] ** 2 + X[:, 1] ** 2,
                          (X[:, 0] + 1) ** 2 + X[:, 1] ** 2)
        t = (.96 - radius) ** 2 + self._tail_distance(X)
        base = 1 - np.column_stack((np.cos(np.pi / 2 * theta),
                                    np.sin(np.pi / 2 * theta)))
        out["F"] = base * (1 + t[:, None])
        G = np.zeros((len(X), 3))
        G[:, 0] = -X[:, 1]
        for i, r in enumerate(radius):
            G[i, 1:] = (r - .36, -r + .04) if r <= .49 else (r - 1, -r + .64)
        out["G"] = G


class CMMF5(CMMF):
    problem_number, n_constraints = 5, 5

    def _evaluate(self, X, out, *args, **kwargs):
        sx = np.sum(X[:, :2] ** 2, axis=1)
        theta = _theta_abs(X)
        t = (.64 - sx) ** 2 + self._tail_distance(X)
        out["F"] = np.column_stack((1 - theta, theta)) * (1 + t[:, None])
        G = np.zeros((len(X), 5))
        G[:, 0] = -X[:, 0] * X[:, 1] + np.finfo(float).eps
        for i, (x, y, s, th) in enumerate(zip(X[:, 0], X[:, 1], sx, theta)):
            if x <= 0 and y <= 0:
                if th <= .5:
                    G[i, 1:] = th - .25, -th, s - .81, -s + .49
                else:
                    G[i, 1:] = th - 1, -th + .75, s - .81, -s + .49
            elif x >= 0 and y >= 0:
                if th <= .5:
                    G[i, 1:] = -th, th - .25, -x - y + .5, x + y - 1.5
                else:
                    G[i, 1:] = -th + .75, th - 1, -x - y + .5, x + y - 1.5
        out["G"] = G


CMMF_PROBLEMS = {i: cls for i, cls in enumerate((CMMF1, CMMF2, CMMF3, CMMF4, CMMF5), 1)}
