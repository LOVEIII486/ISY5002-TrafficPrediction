"""YOLO model and camera filter wrapped together, used like a model:

    det = FilteredDetector("results/detection/finetune/m736_joint_v4/weights/best.pt",
                           "src/camera_filter/camera_filter.json")
    boxes = det.predict("20260924T040558Z_2701_0405_5bc30fa4fa4e.jpg", camera_id="2701")

Returns a DataFrame with the same columns as detections.csv plus a `reason` column
(only `keep` rows unless keep_removed=True). Only car / motorcycle / bus / truck are kept,
following the class mapping convention of the detection module (classes.yaml).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .core import KEEP, CameraFilter

VEHICLES = ("car", "motorcycle", "bus", "truck")
NAME = re.compile(r"^(\d{8}T\d{6})Z_(\d+)_")


class FilteredDetector:
    def __init__(self, weights: str | Path, config: str | Path, imgsz: int = 1280, conf: float = 0.25,
                 class_map: dict[str, str] | None = None):
        from ultralytics import YOLO  # imported only when inference is needed

        self.model = YOLO(str(weights))
        self.filter = CameraFilter.from_json(config)
        self.imgsz, self.conf = imgsz, conf
        self.class_map = {k.lower(): v for k, v in (class_map or {}).items()}

    def _parse_name(self, path: Path):
        """LTA file names encode the capture time (UTC) and camera id."""
        m = NAME.match(path.name)
        if not m:
            return None, None
        ts = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        return m.group(2), ts.isoformat()

    def predict(self, image: str | Path, camera_id: str | None = None, captured_at_utc: str | None = None,
                keep_removed: bool = False) -> pd.DataFrame:
        image = Path(image)
        cam_from_name, ts_from_name = self._parse_name(image)
        camera_id = str(camera_id or cam_from_name or "")
        captured_at_utc = captured_at_utc or ts_from_name or datetime.now(timezone.utc).isoformat()
        r = self.model.predict(str(image), imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        h, w = r.orig_shape
        rows = []
        for box in r.boxes:
            name = str(r.names[int(box.cls)]).lower()
            name = self.class_map.get(name, name)
            if name not in VEHICLES:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            rows.append((camera_id, captured_at_utc, image.name, name, round(float(box.conf), 4),
                         x1 / w, y1 / h, x2 / w, y2 / h))
        df = pd.DataFrame(rows, columns=["camera_id", "captured_at_utc", "image", "cls", "conf", "x1", "y1", "x2", "y2"])
        df["reason"] = self.filter.reasons(df)
        return df if keep_removed else df[df.reason == KEEP].reset_index(drop=True)
