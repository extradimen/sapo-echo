"""OpenStreetMap-derived real multimodal location-selection problem."""

import json
from pathlib import Path

import numpy as np

from pymoo.core.problem import Problem


class SCNULocationSelection(Problem):
    """Minimize accessibility bands from a location near SCNU."""

    center_lat, center_lon = 23.1409, 113.3470
    category_names = ("school", "shopping", "healthcare", "subway")

    def __init__(self, data_path=None, candidate_radius=3000.0,
                 deduplication_radius=100.0, center_lat=None, center_lon=None):
        if data_path is None:
            data_path = (Path(__file__).parents[3] / "data" / "real_location" /
                         "scnu_osm_2026-08-30.json")
        self.data_path = Path(data_path)
        self.center_lat = self.__class__.center_lat if center_lat is None else float(center_lat)
        self.center_lon = self.__class__.center_lon if center_lon is None else float(center_lon)
        self.candidate_radius = float(candidate_radius)
        self.facilities = self._load_facilities(float(deduplication_radius))
        super().__init__(n_var=2, n_obj=4, vtype=float,
                         xl=np.full(2, -self.candidate_radius),
                         xu=np.full(2, self.candidate_radius))

    def _to_xy(self, lat, lon):
        radius = 6371000.0
        return np.array([
            radius * np.deg2rad(lon - self.center_lon) * np.cos(np.deg2rad(self.center_lat)),
            radius * np.deg2rad(lat - self.center_lat),
        ])

    @staticmethod
    def _category(tags):
        if tags.get("amenity") == "school": return "school"
        if tags.get("shop") in {"mall", "supermarket"}: return "shopping"
        if tags.get("amenity") in {"hospital", "clinic"}: return "healthcare"
        if tags.get("railway") == "subway_entrance": return "subway"
        if tags.get("railway") == "station" and tags.get("station") == "subway": return "subway"
        return None

    def _load_facilities(self, deduplication_radius):
        with self.data_path.open(encoding="utf-8") as stream:
            elements = json.load(stream)["elements"]
        grouped = {name: [] for name in self.category_names}
        for element in elements:
            category = self._category(element.get("tags", {}))
            location = element.get("center", element)
            if category is None or "lat" not in location or "lon" not in location:
                continue
            point = self._to_xy(location["lat"], location["lon"])
            if grouped[category] and np.min(np.linalg.norm(
                    np.asarray(grouped[category]) - point, axis=1)) < deduplication_radius:
                continue
            grouped[category].append(point)
        facilities = {name: np.asarray(grouped[name], dtype=float)
                      for name in self.category_names}
        if any(len(points) == 0 for points in facilities.values()):
            raise ValueError("Every objective requires at least one facility.")
        return facilities

    def raw_distances(self, X, unavailable=None):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        unavailable = unavailable or {}
        columns = []
        for category in self.category_names:
            points = self.facilities[category]
            excluded = set(unavailable.get(category, ()))
            keep = np.array([i not in excluded for i in range(len(points))])
            if not np.any(keep):
                columns.append(np.full(len(X), np.inf))
            else:
                columns.append(np.min(np.linalg.norm(
                    X[:, None, :] - points[keep][None, :, :], axis=2), axis=1))
        return np.column_stack(columns)

    def nearest_facility_indices(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.column_stack([
            np.argmin(np.linalg.norm(X[:, None, :] - self.facilities[name][None, :, :], axis=2), axis=1)
            for name in self.category_names
        ])

    def _evaluate(self, X, out, *args, **kwargs):
        out["F"] = np.clip(np.floor(self.raw_distances(X) / 500.0) + 1.0, 1.0, 12.0)
