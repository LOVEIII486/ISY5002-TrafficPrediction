"""Apply the camera filter to existing detection results, without re-running detection.

    python -m src.camera_filter.apply
    python -m src.camera_filter.apply --detections results/detection/detections.csv \
        --config src/camera_filter/camera_filter.json --out results/camera_filter

Outputs:
- detections_filtered.csv: kept boxes, same columns as detections.csv, so downstream code only needs a new path;
- removed.csv: removed boxes with an extra `reason` column (off_road / exclude:k);
- apply_summary.csv: per camera, how many boxes were removed and why, and the change in
  vehicles per frame by day and night.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .core import KEEP, CameraFilter


def main() -> None:
    ap = argparse.ArgumentParser(description="Camera filter: road region + exclusion set")
    ap.add_argument("--detections", type=Path, default=Path("results/detection/detections.csv"))
    ap.add_argument("--frames", type=Path, default=None,
                    help="frames.csv, used for vehicles per frame; defaults to the detections folder")
    ap.add_argument("--config", type=Path, default=Path("src/camera_filter/camera_filter.json"))
    ap.add_argument("--out", type=Path, default=Path("results/camera_filter"))
    args = ap.parse_args()

    det = pd.read_csv(args.detections, dtype={"camera_id": str})
    flt = CameraFilter.from_json(args.config)
    reason = flt.reasons(det)
    args.out.mkdir(parents=True, exist_ok=True)
    det[reason == KEEP].to_csv(args.out / "detections_filtered.csv", index=False)
    removed = det[reason != KEEP].copy()
    removed["reason"] = reason[reason != KEEP]
    removed.to_csv(args.out / "removed.csv", index=False)

    frames_path = args.frames or args.detections.with_name("frames.csv")
    frames = pd.read_csv(frames_path, dtype={"camera_id": str}) if frames_path.exists() else None
    det["reason"] = reason
    det["day"] = flt.is_day(det["captured_at_utc"])
    rows = []
    for cam, g in det.groupby("camera_id"):
        row = {"camera_id": cam, "boxes": len(g), "off_road": int((g.reason == "off_road").sum()),
               "exclude": int(g.reason.str.startswith("exclude").sum()), "kept": int((g.reason == KEEP).sum())}
        row["removed_pct"] = round(100 * (1 - row["kept"] / row["boxes"]), 2)
        if frames is not None:
            fr = frames[frames.camera_id == cam]
            fday = flt.is_day(fr["captured_at_utc"])
            for tag, mask_f, mask_d in (("day", fday, g.day.to_numpy()), ("night", ~fday, ~g.day.to_numpy())):
                n = max(int(mask_f.sum()), 1)
                row[f"per_frame_{tag}_before"] = round(mask_d.sum() / n, 2)
                row[f"per_frame_{tag}_after"] = round(((g.reason == KEEP).to_numpy() & mask_d).sum() / n, 2)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "apply_summary.csv", index=False)
    total = summary[["boxes", "off_road", "exclude", "kept"]].sum()
    print(summary.to_string(index=False))
    print(f"\nTotal: {total.boxes} boxes, off road {total.off_road}, excluded {total.exclude}, kept {total.kept} "
          f"({100 * (1 - total.kept / total.boxes):.1f}% removed) -> {args.out}")


if __name__ == "__main__":
    main()
