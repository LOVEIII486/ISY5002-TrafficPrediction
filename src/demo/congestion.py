"""Congestion layer for the monitoring page. Reads only the congestion module's artifacts.

Inputs (all optional; without them the page hides the congestion parts):
  results/congestion/congestion_levels.csv    one row per frame, level 0 free / 1 moderate / 2 congested
                                              (`python -m src.congestion.ensemble_eval apply`)
  results/congestion/eval_test.json           the one-off test-set evaluation (`... ensemble_eval test`)
  results/congestion/eval_test_by_group.csv   the same, per camera
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "congestion"
LEVELS = RESULTS / "congestion_levels.csv"
EVALUATION = RESULTS / "eval_test.json"
BY_GROUP = RESULTS / "eval_test_by_group.csv"

LEVEL_NAMES = ("free", "moderate", "congested")
WEEKDAYS = 7  # 0 = Monday, as in congestion_levels.csv
HOURS = 24    # SGT


def available() -> bool:
    return LEVELS.is_file()


@lru_cache(maxsize=1)
def levels() -> pd.DataFrame:
    if not LEVELS.is_file():
        return pd.DataFrame(columns=["image", "camera_id", "weekday", "hour", "level"])
    return pd.read_csv(LEVELS, dtype={"camera_id": str}, usecols=["image", "camera_id", "weekday", "hour", "level"])


@lru_cache(maxsize=1)
def _by_image() -> dict[str, int]:
    table = levels()
    return dict(zip(table["image"], table["level"].astype(int)))


@lru_cache(maxsize=1)
def _evaluation() -> dict[str, Any]:
    if not EVALUATION.is_file():
        return {}
    result = json.loads(EVALUATION.read_text(encoding="utf-8"))
    primary = result["primary"]
    method = result["methods"][primary]
    out = {
        "method": primary,
        "n": int(result["n_test"]),
        "accuracy": round(float(method["accuracy"]), 3),
        "accuracy_ci95": [round(float(v), 3) for v in method["accuracy_ci95"]],
        "kappa": round(float(method["kappa"]), 3),
        "cameras": {},
    }
    if BY_GROUP.is_file():
        groups = pd.read_csv(BY_GROUP, dtype={"group": str})
        for row in groups[groups["method"] == primary].itertuples():
            out["cameras"][row.group] = {"n": int(row.n), "accuracy": round(float(row.accuracy), 3)}
    return out


def evaluation(camera_id: str) -> dict[str, Any] | None:
    """Test-set accuracy of the primary method, overall and on this camera's test frames."""
    result = _evaluation()
    if not result:
        return None
    return {key: value for key, value in result.items() if key != "cameras"} | {
        "camera": result["cameras"].get(camera_id),
    }


def frame_level(image: str) -> dict[str, Any] | None:
    level = _by_image().get(image)
    if level is None:
        return None
    return {"level": level, "name": LEVEL_NAMES[level]}


def _grid(series: pd.Series) -> list[list[float | None]]:
    """weekday x hour matrix; cells without frames stay None so the chart leaves them blank."""
    grid: list[list[float | None]] = [[None] * HOURS for _ in range(WEEKDAYS)]
    for (weekday, hour), value in series.items():
        grid[int(weekday)][int(hour)] = round(float(value), 3)
    return grid


@lru_cache(maxsize=16)
def heatmap(camera_id: str) -> dict[str, Any] | None:
    table = levels()
    table = table[table["camera_id"] == camera_id]
    if table.empty:
        return None
    table = table.assign(congested=table["level"] == 2, busy=table["level"] >= 1)
    cells = table.groupby(["weekday", "hour"])
    stats = pd.DataFrame({
        "congested": cells["congested"].mean(),
        "busy": cells["busy"].mean(),
        "frames": cells.size(),
    })
    # Busiest slot: highest congested share; ties go to the busier slot, then the one with more frames.
    peak = stats.sort_values(["congested", "busy", "frames"], ascending=False).iloc[0]
    peak_weekday, peak_hour = peak.name
    return {
        "camera": camera_id,
        "frames": int(len(table)),
        "congested": _grid(stats["congested"]),
        "busy": _grid(stats["busy"]),
        "counts": _grid(stats["frames"]),
        "share": {
            "congested": round(float(table["congested"].mean()), 3),
            "busy": round(float(table["busy"].mean()), 3),
        },
        "peak": {
            "weekday": int(peak_weekday),
            "hour": int(peak_hour),
            "share": round(float(peak["congested"]), 3),
        },
        "evaluation": evaluation(camera_id),
    }
