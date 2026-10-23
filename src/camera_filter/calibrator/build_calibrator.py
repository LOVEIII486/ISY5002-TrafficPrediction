"""Build the camera calibration page: pack two weeks of detection positions as points into one standalone HTML file.

    python src/camera_filter/calibrator/build_calibrator.py --detections results/detection \
        --images <folder with the 80 test images> --out camera_calibrator.html

- The points come from the detector output (detections.csv), not from human labels.
- The 80 annotation-package images (the human-labelled test set) are left out of the points, so the
  exclusion set is not drawn by looking at test errors. They are only used as each camera's background.
- Automatic candidates: positions where the same box (IoU >= 0.5) recurs in many frames, for a person
  to confirm whether they are fixed false positives.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from suggest import suggest  # noqa: E402

CLASSES = ["car", "motorcycle", "bus", "truck"]
CAMERA_NAMES = {
    "2701": "Causeway · Towards Johor",
    "2702": "Causeway · Towards BKE",
    "2704": "Causeway · Towards Checkpoint",
    "4703": "Second Link · Bridge view",
    "4712": "Second Link · Towards Tuas",
    "4713": "Second Link · Checkpoint view",
    "4798": "Sentosa Gateway · Towards Telok Blangah",
    "4799": "Sentosa Gateway · Towards Sentosa",
}
DAY_HOURS = range(7, 19)  # 07:00-18:59 Singapore time is day, matching the detection module


def sgt_hour(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.tz_convert("Asia/Singapore").dt.hour


def iou_one(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0]), 0, None)
    iy = np.clip(np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]), 0, None)
    inter = ix * iy
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-12)


def find_candidates(d: pd.DataFrame, n_frames: int, n_day: int, n_night: int, top: int = 12) -> list[dict]:
    """Find boxes that recur at the same position: coarse bucketing by coordinates, then merging by IoU."""
    boxes = d[["x1", "y1", "x2", "y2"]].to_numpy()
    frame = d["frame"].to_numpy()
    night = ~d["hour"].isin(DAY_HOURS).to_numpy()
    key = pd.Series(list(zip(*(np.round(boxes[:, k] / 0.01).astype(int) for k in range(4)))))
    buckets = pd.DataFrame({"key": key, "frame": frame}).groupby("key")["frame"].nunique()
    seeds = buckets[buckets >= max(5, 0.01 * n_frames)].sort_values(ascending=False)
    used = np.zeros(len(d), bool)
    out = []
    for k in seeds.index:
        idx = np.where((key == k).to_numpy() & ~used)[0]
        if len(idx) == 0:
            continue
        rep = np.median(boxes[idx], axis=0)
        member = (iou_one(rep, boxes) >= 0.5) & ~used
        if not member.any():
            continue
        used |= member
        fr = np.unique(frame[member])
        fr_day = np.unique(frame[member & ~night])
        fr_night = np.unique(frame[member & night])
        if len(fr) < max(5, 0.02 * n_frames):
            continue
        cls_counts = d.loc[member, "cls"].value_counts()
        out.append(
            {
                "box": [round(float(v), 4) for v in rep],
                "frames": int(len(fr)),
                "day": int(len(fr_day)),
                "night": int(len(fr_night)),
                "day_pct": round(len(fr_day) / max(n_day, 1) * 100, 1),
                "night_pct": round(len(fr_night) / max(n_night, 1) * 100, 1),
                "cls": {c: int(v) for c, v in cls_counts.items()},
                "conf": round(float(d.loc[member, "conf"].median()), 2),
            }
        )
    out.sort(key=lambda c: -max(c["day_pct"], c["night_pct"]))
    return out[:top]


def pack_points(d: pd.DataFrame) -> str:
    """12 bytes per box: bottom-centre x, y, width, height, frame index (uint16 each); confidence, class + hour (uint8 each)."""
    cx = ((d.x1 + d.x2) / 2).clip(0, 1)
    q = lambda s: np.round(s.clip(0, 1).to_numpy() * 65535).astype("<u2")
    flags = (d["cls"].map(CLASSES.index).to_numpy() | (d["hour"].to_numpy() << 2)).astype("u1")
    blob = b"".join(
        [
            q(cx).tobytes(),
            q(d.y2).tobytes(),
            q(d.x2 - d.x1).tobytes(),
            q(d.y2 - d.y1).tobytes(),
            d["frame"].to_numpy().astype("<u2").tobytes(),
            np.round(d.conf.to_numpy() * 100).astype("u1").tobytes(),
            flags.tobytes(),
        ]
    )
    return base64.b64encode(gzip.compress(blob, 9)).decode()


def encode_image(path: Path) -> str:
    im = Image.open(path).convert("RGB").resize((1280, 720), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def pick_reference(images: list[str], counts: pd.Series, night: bool) -> str | None:
    def hour(name: str) -> int:
        return (int(name[9:11]) + 8) % 24

    if night:
        pool = [n for n in images if hour(n) >= 20 or hour(n) <= 5] or [n for n in images if hour(n) not in DAY_HOURS]
    else:
        pool = [n for n in images if 8 <= hour(n) <= 17] or [n for n in images if hour(n) in DAY_HOURS]
    if not pool:
        return None
    return min(pool, key=lambda n: counts.get(n, 0))  # the frame with the fewest vehicles shows the road most clearly


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections", type=Path, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--template", type=Path, default=Path(__file__).with_name("calibrator_template.html"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    det = pd.read_csv(args.detections / "detections.csv", dtype={"camera_id": str})
    frames = pd.read_csv(args.detections / "frames.csv", dtype={"camera_id": str})
    run_file = args.detections / "run.json"  # optional: only used to label which model produced the points
    run = json.loads(run_file.read_text(encoding="utf-8")) if run_file.exists() else {}
    held_out = sorted(p.name for p in args.images.glob("*.jpg"))
    counts = det.groupby("image").size()

    det = det[~det.image.isin(held_out)].copy()
    frames = frames[~frames.image.isin(held_out)].copy()
    det["hour"] = sgt_hour(det.captured_at_utc)
    frames["hour"] = sgt_hour(frames.captured_at_utc)

    cameras = {}
    for cam in sorted(CAMERA_NAMES):
        fr = frames[frames.camera_id == cam].sort_values("captured_at_utc").reset_index(drop=True)
        frame_index = {name: i for i, name in enumerate(fr.image)}
        d = det[det.camera_id == cam].copy()
        d["frame"] = d.image.map(frame_index)
        d = d.dropna(subset=["frame"]).astype({"frame": int})
        n_day = int(fr.hour.isin(DAY_HOURS).sum())
        n_night = len(fr) - n_day
        cam_images = [n for n in held_out if n.split("_")[1] == cam]
        refs = {}
        for tag, is_night in (("day", False), ("night", True)):
            name = pick_reference(cam_images, counts, is_night)
            if name:
                refs[tag] = {"src": encode_image(args.images / name), "image": name}
        cameras[cam] = {
            "name": CAMERA_NAMES[cam],
            "n_frames": len(fr),
            "n_day": n_day,
            "n_night": n_night,
            "frame_hours": base64.b64encode(fr.hour.to_numpy().astype("u1").tobytes()).decode(),
            "n_points": len(d),
            "points": pack_points(d),
            "refs": refs,
            "candidates": find_candidates(d, len(fr), n_day, n_night),
            "suggest": suggest(d),
        }
        print(f"{cam}: {len(fr)} frames, {len(d)} boxes, {len(cameras[cam]['candidates'])} candidates")

    payload = {
        "source": {
            "detections": "results/detection/detections.csv",
            "model": run.get("model_path"),
            "imgsz": run.get("imgsz"),
            "conf": run.get("confidence"),
            "held_out_images": len(held_out),
        },
        "classes": CLASSES,
        "day_hours": [DAY_HOURS.start, DAY_HOURS.stop],
        "cameras": cameras,
    }
    html = args.template.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(payload, ensure_ascii=False))
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
