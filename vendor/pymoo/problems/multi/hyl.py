"""HYL1--5 multimodal multi-objective benchmark problems.

The equations are an independent NumPy translation of the definitions added
to PlatEMO 4.16.  The suite was proposed by Ishibuchi, Peng, and Pang at CEC
2022 to test problems with infinitely many equivalent Pareto-optimal
solutions.  Bounds and reference-set constructions follow the PlatEMO files.
"""

import numpy as np

from pymoo.core.problem import Problem


def _grid_count(n_points, divisor, minimum=2):
    return max(minimum, int(np.floor(n_points / divisor)))


class HYL1(Problem):
    def __init__(self, a=10.0):
        self.a = float(a)
        super().__init__(n_var=2, n_obj=2,
                         xl=np.full(2, -self.a / 2),
                         xu=np.full(2, self.a / 2))

    def _evaluate(self, X, out, *args, **kwargs):
        s = X[:, 0] + X[:, 1]
        out["F"] = np.column_stack((s ** 2, (s - 2) ** 2))

    def _calc_pareto_front(self, n_points=100, *args, **kwargs):
        f1 = np.linspace(0, 4, n_points)
        return np.column_stack((f1, (np.sqrt(f1) - 2) ** 2))

    def _calc_pareto_set(self, n_points=1000, *args, **kwargs):
        x2 = np.linspace(-self.a / 2, self.a / 2,
                         _grid_count(n_points, 20))
        points = []
        for level in np.linspace(0, 2, 20):
            X = np.column_stack((-x2 + level, x2))
            points.append(X[np.all((X >= self.xl) & (X <= self.xu), axis=1)])
        return np.row_stack(points)


class HYL2(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2,
                         xl=np.full(2, -10.0), xu=np.full(2, 10.0))

    def _evaluate(self, X, out, *args, **kwargs):
        x = X[:, 0]
        out["F"] = np.column_stack((
            np.minimum(np.abs(x + 8), np.abs(x - 4)),
            np.minimum(np.abs(x + 4), np.abs(x - 8)),
        ))

    def _calc_pareto_front(self, n_points=100, *args, **kwargs):
        f1 = np.linspace(0, 4, n_points)
        return np.column_stack((f1, 4 - f1))

    def _calc_pareto_set(self, n_points=1000, *args, **kwargs):
        x2 = np.linspace(-10, 10, _grid_count(n_points // 2, 20))
        points = []
        for x1 in np.r_[np.linspace(-8, -4, 20), np.linspace(4, 8, 20)]:
            points.append(np.column_stack((np.full_like(x2, x1), x2)))
        return np.row_stack(points)


class _PolygonHYL(Problem):
    centers = None
    vertices = None
    radius = 2.0

    def __init__(self, n_obj):
        super().__init__(n_var=3, n_obj=n_obj,
                         xl=np.full(3, -10.0), xu=np.full(3, 10.0))
        self.centers = np.asarray(self.centers, dtype=float)
        self.vertices = np.asarray(self.vertices, dtype=float) * self.radius
        self.points = self.centers[:, None, :] + self.vertices[None, :, :]

    def _evaluate(self, X, out, *args, **kwargs):
        xy = X[:, :2]
        distances = np.linalg.norm(
            xy[:, None, None, :] - self.points[None, :, :, :], axis=3)
        out["F"] = distances.min(axis=1)

    @staticmethod
    def _inside_convex_polygon(X, polygon, tolerance=1e-12):
        edges = np.roll(polygon, -1, axis=0) - polygon
        rel = X[:, None, :] - polygon[None, :, :]
        cross = edges[None, :, 0] * rel[:, :, 1] - edges[None, :, 1] * rel[:, :, 0]
        return np.all(cross >= -tolerance, axis=1) | np.all(cross <= tolerance, axis=1)

    def _polygon_grid(self, center, resolution):
        lo = self.points[0].min(axis=0) - self.centers[0] + center
        hi = self.points[0].max(axis=0) - self.centers[0] + center
        x, y = np.meshgrid(np.linspace(lo[0], hi[0], resolution),
                           np.linspace(lo[1], hi[1], resolution), indexing="ij")
        xy = np.column_stack((x.ravel(), y.ravel()))
        polygon = center + self.vertices
        return xy[self._inside_convex_polygon(xy, polygon)]

    def _calc_pareto_front(self, n_points=2500, *args, **kwargs):
        resolution = max(2, int(np.ceil(np.sqrt(n_points))))
        xy = self._polygon_grid(self.centers[0], resolution)
        return self.evaluate(np.column_stack((xy, np.zeros(len(xy)))))

    def _calc_pareto_set(self, n_points=10000, *args, **kwargs):
        n_regions = len(self.centers)
        z_resolution = max(2, int(round((n_points / n_regions) ** (1 / 3))))
        xy = [self._polygon_grid(c, z_resolution) for c in self.centers]
        z = np.linspace(-10, 10, z_resolution)
        blocks = []
        for region in xy:
            blocks.append(np.column_stack((
                np.repeat(region[:, 0], len(z)),
                np.repeat(region[:, 1], len(z)),
                np.tile(z, len(region)),
            )))
        return np.row_stack(blocks)


class HYL3(_PolygonHYL):
    centers = [[-6, 6], [6, -6]]
    vertices = [[0, 1], [np.sqrt(3) / 2, -0.5], [-np.sqrt(3) / 2, -0.5]]

    def __init__(self):
        super().__init__(n_obj=3)


class HYL4(_PolygonHYL):
    centers = [[-7, 7], [7, 7], [7, -7], [-7, -7]]
    _d = np.sqrt(2) / 2
    vertices = [[-_d, _d], [_d, _d], [_d, -_d], [-_d, -_d]]

    def __init__(self):
        super().__init__(n_obj=4)


class HYL5(Problem):
    points = np.array([[[-6, -2], [-2, -6]],
                       [[2, 6], [6, 2]]], dtype=float)

    def __init__(self):
        super().__init__(n_var=2, n_obj=2,
                         xl=np.full(2, -10.0), xu=np.full(2, 10.0))

    def _evaluate(self, X, out, *args, **kwargs):
        distances = np.abs(X[:, None, None, :] - self.points[None, :, :, :]).sum(axis=3)
        out["F"] = distances.min(axis=1)

    def _calc_pareto_front(self, n_points=100, *args, **kwargs):
        f1 = np.linspace(0, 8, n_points)
        return np.column_stack((f1, 8 - f1))

    def _calc_pareto_set(self, n_points=1000, *args, **kwargs):
        n = max(2, int(np.floor(np.sqrt(n_points / 2))))
        blocks = []
        for lo, hi in [(-6, -2), (2, 6)]:
            x, y = np.meshgrid(np.linspace(lo, hi, n), np.linspace(lo, hi, n), indexing="ij")
            blocks.append(np.column_stack((x.ravel(), y.ravel())))
        return np.row_stack(blocks)


HYL_PROBLEMS = {i: cls for i, cls in enumerate((HYL1, HYL2, HYL3, HYL4, HYL5), 1)}
