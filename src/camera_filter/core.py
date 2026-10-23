"""Core of the camera filter: load the calibration config and filter detections by road region and exclusion set.

The detection model is neither modified nor retrained; this is post-processing on its output
(a hybrid of a learned detector and human-calibrated, camera-specific rules).

Each detection is tested at the **bottom-centre** of its box (where the wheels touch the road).
All coordinates are normalised to 0-1 with the origin at the top-left, matching
`results/detection/detections.csv`. Rules, in order:
1. If the camera has a road region, a point outside every road polygon is removed (`off_road`).
2. Otherwise the exclusion boxes are checked in order. An exclusion applies when its time window
   matches (`all` / `day` / `night`) and either `rule = point` (the point lies in the box) or
   `rule = box` (the whole detection box lies inside it); the detection is removed (`exclude:k`).
3. Everything else is kept. Cameras that are not in the config are left untouched.
Day and night use Singapore time; by default 07:00-19:00 is day, matching the detection module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

KEEP = "keep"
OFF_ROAD = "off_road"


def _inside(v: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Pull points slightly into the frame. A vehicle cut off by the image edge has a box edge of
    exactly 0 or 1, which lies on a polygon edge snapped to the frame border; the ray-casting test
    would otherwise count it as outside."""
    return np.clip(v, eps, 1 - eps)


def points_in_polygon(x: np.ndarray, y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised ray-casting test for many points against one polygon of shape (k, 2)."""
    inside = np.zeros(len(x), dtype=bool)
    xs, ys = poly[:, 0], poly[:, 1]
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi, xj, yj = xs[i], ys[i], xs[j], ys[j]
        crosses = (yi > y) != (yj > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
        inside ^= crosses & (x < x_cross)
        j = i
    return inside


@dataclass
class Exclusion:
    box: tuple[float, float, float, float]
    when: str = "all"  # all / day / night
    rule: str = "point"  # point / box
    note: str = ""


@dataclass
class CameraRules:
    road: list[np.ndarray] = field(default_factory=list)
    zone: list[np.ndarray] = field(default_factory=list)
    exclude: list[Exclusion] = field(default_factory=list)

    def in_any(self, polys: list[np.ndarray], x: np.ndarray, y: np.ndarray) -> np.ndarray:
        hit = np.zeros(len(x), dtype=bool)
        for poly in polys:
            hit |= points_in_polygon(x, y, poly)
        return hit


class CameraFilter:
    def __init__(self, cameras: dict[str, CameraRules], day_hours: tuple[int, int] = (7, 19)):
        self.cameras = cameras
        self.day_hours = day_hours

    @classmethod
    def from_json(cls, path: str | Path) -> "CameraFilter":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        cams = {}
        for cam, c in raw.get("cameras", {}).items():
            cams[str(cam)] = CameraRules(
                road=[np.asarray(p, dtype=float) for p in c.get("road", []) if len(p) >= 3],
                zone=[np.asarray(p, dtype=float) for p in c.get("zone", []) if len(p) >= 3],
                exclude=[Exclusion(tuple(e["box"]), e.get("when", "all"), e.get("rule", "point"), e.get("note", ""))
                         for e in c.get("exclude", [])],
            )
        day = tuple(raw.get("rules", {}).get("day_hours_sgt", (7, 19)))
        return cls(cams, day)

    # ---------- decisions ----------
    def is_day(self, captured_at_utc: pd.Series) -> np.ndarray:
        hour = pd.to_datetime(captured_at_utc, utc=True).dt.tz_convert("Asia/Singapore").dt.hour
        return hour.between(self.day_hours[0], self.day_hours[1] - 1).to_numpy()

    def reasons(self, det: pd.DataFrame) -> np.ndarray:
        """Decision for every detection: `keep`, `off_road` or `exclude:<1-based index>`.

        `det` needs the columns camera_id, captured_at_utc, x1, y1, x2, y2 (normalised).
        """
        out = np.full(len(det), KEEP, dtype=object)
        if len(det) == 0:
            return out
        cam_col = det["camera_id"].astype(str).to_numpy()
        day = self.is_day(det["captured_at_utc"])
        x1, y1, x2, y2 = (det[c].to_numpy(dtype=float) for c in ("x1", "y1", "x2", "y2"))
        px, py = _inside((x1 + x2) / 2), _inside(y2)  # bottom-centre of the box
        for cam, rules in self.cameras.items():
            idx = np.where(cam_col == cam)[0]
            if len(idx) == 0:
                continue
            alive = np.ones(len(idx), dtype=bool)
            if rules.road:
                on_road = rules.in_any(rules.road, px[idx], py[idx])
                out[idx[~on_road]] = OFF_ROAD
                alive &= on_road
            for k, e in enumerate(rules.exclude, start=1):
                a, b, c, d = e.box
                if e.rule == "box":
                    hit = (x1[idx] >= a) & (x2[idx] <= c) & (y1[idx] >= b) & (y2[idx] <= d)
                else:
                    hit = (px[idx] >= a) & (px[idx] <= c) & (py[idx] >= b) & (py[idx] <= d)
                if e.when == "day":
                    hit &= day[idx]
                elif e.when == "night":
                    hit &= ~day[idx]
                hit &= alive
                out[idx[hit]] = f"exclude:{k}"
                alive &= ~hit
        return out

    def filter(self, det: pd.DataFrame) -> pd.DataFrame:
        """Return the kept detections with the same columns."""
        return det[self.reasons(det) == KEEP]

    def zone_index(self, det: pd.DataFrame) -> np.ndarray:
        """1-based index of the detection zone each box falls in, 0 if none. Used for congestion analysis."""
        out = np.zeros(len(det), dtype=int)
        cam_col = det["camera_id"].astype(str).to_numpy()
        px = _inside(((det["x1"] + det["x2"]) / 2).to_numpy(dtype=float))
        py = _inside(det["y2"].to_numpy(dtype=float))
        for cam, rules in self.cameras.items():
            idx = np.where(cam_col == cam)[0]
            for k, poly in enumerate(rules.zone, start=1):
                hit = points_in_polygon(px[idx], py[idx], poly) & (out[idx] == 0)
                out[idx[hit]] = k
        return out
