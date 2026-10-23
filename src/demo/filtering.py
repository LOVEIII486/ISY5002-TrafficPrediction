"""Camera-filter layer for the monitoring page. Reads only the plugin's artifacts.

Inputs (all optional; without them the page looks exactly as before):
  src/camera_filter/camera_filter.json     road region, detection zones and exclusion set per camera
  results/camera_filter/removed.csv        boxes the filter removed, with the reason
  results/camera_filter/apply_summary.csv  per-camera removal counts

Produced by `python -m src.camera_filter.apply`. Like detection.py, this module does not import
the plugin; it only reads what the plugin wrote.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "src" / "camera_filter" / "camera_filter.json"
RESULTS = ROOT / "results" / "camera_filter"
REMOVED = RESULTS / "removed.csv"
SUMMARY = RESULTS / "apply_summary.csv"

# removed.csv and detections.csv share the same box coordinates, so a rounded key matches exactly.
_KEY_DIGITS = 5


def available() -> bool:
    return CONFIG.is_file() and REMOVED.is_file()


@lru_cache(maxsize=1)
def config() -> dict[str, Any]:
    return json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.is_file() else {}


@lru_cache(maxsize=1)
def removed() -> pd.DataFrame:
    if not REMOVED.is_file():
        return pd.DataFrame(columns=["camera_id", "image", "cls", "x1", "y1", "x2", "y2", "reason"])
    return pd.read_csv(REMOVED, dtype={"camera_id": str})


@lru_cache(maxsize=1)
def summary() -> dict[str, dict[str, Any]]:
    if not SUMMARY.is_file():
        return {}
    table = pd.read_csv(SUMMARY, dtype={"camera_id": str})
    return {
        row.camera_id: {
            "boxes": int(row.boxes),
            "off_road": int(row.off_road),
            "exclude": int(row.exclude),
            "removed_share": round(float(row.removed_pct) / 100, 4),
        }
        for row in table.itertuples()
    }


def _key(cls: str, x1: float, y1: float, x2: float, y2: float) -> tuple:
    return (cls, *(round(float(v), _KEY_DIGITS) for v in (x1, y1, x2, y2)))


def overlay(camera_id: str) -> dict[str, Any] | None:
    """Polygons and boxes to draw on the frame, in normalised 0-1 coordinates."""
    camera = config().get("cameras", {}).get(camera_id)
    if not camera:
        return None
    return {
        "road": camera.get("road") or [],
        "zones": camera.get("zone") or [],
        "exclude": [
            {"box": item["box"], "rule": item.get("rule", "point"), "when": item.get("when", "all")}
            for item in camera.get("exclude") or []
        ],
    }


def annotate(detail: dict[str, Any]) -> dict[str, Any]:
    """Mark the boxes the filter removed and attach the camera's overlay. Mutates and returns detail."""
    if not available() or not detail.get("camera"):
        return detail
    rows = removed()
    rows = rows[rows["image"] == detail["image"]]
    reasons = {
        _key(r.cls, r.x1, r.y1, r.x2, r.y2): str(r.reason).split(":")[0] for r in rows.itertuples()
    }
    for box in detail["boxes"]:
        reason = reasons.get(_key(box["cls"], box["x1"], box["y1"], box["x2"], box["y2"]))
        if reason:
            box["removed"] = reason
    detail["filter"] = overlay(detail["camera"])
    return detail
