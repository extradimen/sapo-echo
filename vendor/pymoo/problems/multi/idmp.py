"""Independent NumPy implementation of the published IDMP benchmark suite."""

import numpy as np

from pymoo.core.problem import Problem


class IDMP(Problem):
    """Imbalanced Distance Minimization Problem ``IDMPM{M}T{T}``."""

    def __init__(self, n_obj=2, problem_type=1):
        self.problem_type = int(problem_type)
        if n_obj not in (2, 3, 4) or self.problem_type not in (1, 2, 3, 4):
            raise ValueError("IDMP supports M in {2,3,4} and T in {1,2,3,4}.")
        super().__init__(n_var=n_obj, n_obj=n_obj,
                         xl=-np.ones(n_obj), xu=np.ones(n_obj))
        self.n_basins = 2 if n_obj == 2 else 4
        if n_obj == 2:
            self.vertices = np.array([[-0.6, 0.4], [-0.4, 0.6]])
        else:
            if n_obj == 3:
                polygon = np.array([[-np.sqrt(3) / 2, -.5], [0., 1.],
                                    [np.sqrt(3) / 2, -.5]]) * .1
            else:
                q = np.sqrt(.5) * .1
                polygon = np.array([[-q, -q], [-q, q], [q, q], [q, -q]])
            centers = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
            self.vertices = centers[:, None, :] + polygon[None, :, :]

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = self._evaluate_two(X) if self.n_obj == 2 else self._evaluate_polygon(X)

    def _evaluate_two(self, X):
        x, z = X[:, 0], X[:, 1]
        base = np.abs(x[:, None, None] - self.vertices[None, :, :])
        if self.problem_type == 1:
            penalty = np.column_stack((np.abs(z + .5), 3 * np.abs(z - .5)))
        elif self.problem_type == 2:
            penalty = np.column_stack((100 * np.abs(z + .5) ** 2,
                                       100 * np.abs(z - .5) ** 1.6))
        elif self.problem_type == 3:
            penalty = np.column_stack((100 * (z + .5) ** 2,
                                       100 * (z - .5 + .4 * (x - .5)) ** 2))
        else:
            penalty = np.column_stack((
                100 * ((z + .5) ** 2 + 1 - np.cos(2 * np.pi * (z + .5))),
                100 * ((z - .5) ** 2 + 1 - np.cos(8 * np.pi * (z - .5))),
            ))
        return np.min(base + penalty[:, None, :], axis=2)

    def _evaluate_polygon(self, X):
        xy, tail = X[:, :2], X[:, 2:]
        distances = np.linalg.norm(
            xy[:, None, None, :] - self.vertices[None, :, :, :], axis=3
        ).transpose(0, 2, 1)
        targets = np.array([-.6, -.2, .2, .6])
        if self.problem_type == 1:
            penalties = np.column_stack([
                (i + 1) * np.sum(np.abs(tail - target), axis=1)
                for i, target in enumerate(targets)])
        elif self.problem_type == 2:
            penalties = np.column_stack([
                100 * np.sum(np.abs(tail - target) ** exponent, axis=1)
                for target, exponent in zip(targets, [2.0, 1.8, 1.6, 1.4])])
        elif self.problem_type == 3:
            slopes = [0.0, .1, .2, .3] if self.n_obj == 3 else [0.0, .05, .1, .15]
            offsets = np.column_stack((np.zeros(len(X)),
                xy[:, 0] - .5 + xy[:, 1] + .5,
                xy[:, 0] - .5 + xy[:, 1] - .5,
                xy[:, 0] + .5 + xy[:, 1] - .5))
            penalties = np.column_stack([
                100 * np.sum((tail - target + slope * offsets[:, i, None]) ** 2, axis=1)
                for i, (target, slope) in enumerate(zip(targets, slopes))])
        else:
            penalties = np.column_stack([
                100 * np.sum((tail - target) ** 2, axis=1)
                + 100 * (1 - np.cos(2 * np.pi * (i + 1) * (tail[:, 0] - target)))
                for i, target in enumerate(targets)])
        return np.min(distances + penalties[:, None, :], axis=2)
