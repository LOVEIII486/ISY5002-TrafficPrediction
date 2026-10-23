"""Shared helpers for the congestion voters: frame list, image lookup, zones."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.camera_filter import CameraFilter

DAY_HOURS = (7, 19)  # 07:00-18:59 Singapore time is day, matching the detection module


def load_frames(frames_csv: Path) -> pd.DataFrame:
    """frames.csv with Singapore-time columns added: weekday (0 = Monday), hour, is_day."""
    fr = pd.read_csv(frames_csv, dtype={"camera_id": str})
    sgt = pd.to_datetime(fr["captured_at_utc"], utc=True).dt.tz_convert("Asia/Singapore")
    fr["captured_at_sgt"] = sgt.dt.strftime("%Y-%m-%d %H:%M")
    fr["weekday"] = sgt.dt.weekday
    fr["hour"] = sgt.dt.hour
    fr["is_day"] = fr["hour"].between(DAY_HOURS[0], DAY_HOURS[1] - 1)
    return fr


def image_index(root: Path) -> dict[str, Path]:
    """File name -> path for every image under datasets/lta (week folders / camera folders).
    macOS resource-fork files (._xxx.jpg) in the dataset are skipped."""
    index = {}
    for path in root.rglob("*.jpg"):
        if not path.name.startswith("._"):
            index[path.name] = path
    if not index:
        raise SystemExit(f"No images under {root}. Download them first: python scripts/fetch_data.py lta")
    return index


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """cv2.imread fails on non-ASCII Windows paths; decoding from bytes works everywhere."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def zone_polygons(config: Path) -> dict[str, list[np.ndarray]]:
    """Detection zones per camera (normalised coordinates) from the camera filter config."""
    flt = CameraFilter.from_json(config)
    return {cam: rules.zone for cam, rules in flt.cameras.items() if rules.zone}


def zone_masks(polys: list[np.ndarray], width: int, height: int) -> list[np.ndarray]:
    masks = []
    for poly in polys:
        m = np.zeros((height, width), np.uint8)
        cv2.fillPoly(m, [np.round(poly * [width, height]).astype(np.int32)], 1)
        masks.append(m.astype(bool))
    return masks

