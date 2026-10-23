"""Voter B for congestion: background difference inside each camera's detection zones.

    python -m src.congestion.background
    python -m src.congestion.background --images datasets/lta --out results/congestion

Classical image processing, independent of the detector:
1. Every frame is read as greyscale at 1/4 size (480 x 270).
2. For each camera and each hour of the day (Singapore time), the background is the per-pixel
   median of the frames from that hour and its neighbours over the two weeks (the window widens
   until at least --min-bg frames are available). Hourly backgrounds absorb street lights and the
   day/night change in lighting.
3. Frame and background are compared as edge maps (Canny), which are less sensitive to overall
   brightness than raw intensities: an edge in the frame that is not near an edge of the background
   is foreground. Foreground edges are dilated into blobs.
4. Over-exposed pixels (headlights, glare) are ignored in both numerator and denominator.
5. The score is the foreground share of the valid zone pixels, for all zones together and per zone.

Known weaknesses: shadows and rain reflections create edges; a road that is busy for
most of an hour gets vehicles baked into that hour's background.

Outputs (results/congestion/):
- voter_background.csv: image, camera_id, bg_score, bg_z{k}, overexposed (share of zone pixels ignored);
- backgrounds/<camera>_h<hour>.jpg: the hourly backgrounds, for inspection;
- background_examples/<camera>_*.jpg (with --examples): frame | background | foreground, for the report.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .common import image_index, load_frames, read_image, zone_masks, zone_polygons

W, H = 480, 270
CANNY = (50, 150)
OVEREXPOSED = 245


def load_gray(path: Path) -> np.ndarray | None:
    img = read_image(path, cv2.IMREAD_REDUCED_GRAYSCALE_4)  # JPEG decoded directly at 1/4 size
    if img is None:
        return None
    if img.shape != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return img


def hourly_backgrounds(stack: np.ndarray, hours: np.ndarray, min_bg: int, max_bg: int, seed: int = 0) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for h in range(24):
        for width in range(1, 13):
            window = {(h + d) % 24 for d in range(-width, width + 1)}
            idx = np.where(np.isin(hours, list(window)))[0]
            if len(idx) >= min_bg or width == 12:
                break
        if len(idx) == 0:
            continue
        if len(idx) > max_bg:
            idx = rng.choice(idx, max_bg, replace=False)
        out[h] = np.median(stack[idx], axis=0).astype(np.uint8)
    return out


def edges(gray: np.ndarray) -> np.ndarray:
    return cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), *CANNY) > 0


def foreground(gray: np.ndarray, bg: np.ndarray, bg_edges_dilated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Foreground blobs and the mask of valid (not over-exposed) pixels."""
    new_edges = edges(gray) & ~bg_edges_dilated
    blobs = cv2.dilate(new_edges.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    glare = cv2.dilate((gray >= OVEREXPOSED).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    return blobs, ~glare


def save_example(path: Path, gray: np.ndarray, bg: np.ndarray, blobs: np.ndarray, masks: list[np.ndarray], score: float) -> None:
    fg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    fg[blobs] = (0.4 * fg[blobs] + 0.6 * np.array([255, 43, 200])).astype(np.uint8)
    for m in masks:
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(fg, contours, -1, (11, 158, 245), 2)
    panel = np.hstack([cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR), fg])
    cv2.putText(panel, f"frame | hourly background | foreground  score {score:.3f}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imencode(".jpg", panel)[1].tofile(str(path))


def main() -> None:
    ap = argparse.ArgumentParser(description="Congestion voter B: background difference in the detection zones")
    ap.add_argument("--images", type=Path, default=Path("datasets/lta"))
    ap.add_argument("--frames", type=Path, default=Path("results/detection/frames.csv"))
    ap.add_argument("--config", type=Path, default=Path("src/camera_filter/camera_filter.json"))
    ap.add_argument("--out", type=Path, default=Path("results/congestion"))
    ap.add_argument("--min-bg", type=int, default=30, help="minimum frames per hourly background")
    ap.add_argument("--max-bg", type=int, default=200, help="frames sampled per hourly background")
    ap.add_argument("--examples", type=int, default=3, help="example panels saved per camera (0 = none)")
    args = ap.parse_args()

    frames = load_frames(args.frames)
    index = image_index(args.images)
    zones = zone_polygons(args.config)
    (args.out / "backgrounds").mkdir(parents=True, exist_ok=True)
    if args.examples:
        (args.out / "background_examples").mkdir(parents=True, exist_ok=True)

    rows = []
    for cam in sorted(frames["camera_id"].unique()):
        if cam not in zones:
            print(f"{cam}: no detection zone in the config, skipped")
            continue
        fr = frames[(frames["camera_id"] == cam) & frames["image"].isin(index)].reset_index(drop=True)
        missing = int(((frames["camera_id"] == cam) & ~frames["image"].isin(index)).sum())
        stack = np.zeros((len(fr), H, W), np.uint8)
        ok = np.zeros(len(fr), bool)
        for i, name in enumerate(fr["image"]):
            img = load_gray(index[name])
            if img is not None:
                stack[i], ok[i] = img, True
        fr, stack = fr[ok].reset_index(drop=True), stack[ok]
        hours = fr["hour"].to_numpy()

        bgs = hourly_backgrounds(stack, hours, args.min_bg, args.max_bg)
        for h, bg in bgs.items():
            cv2.imencode(".jpg", bg)[1].tofile(str(args.out / "backgrounds" / f"{cam}_h{h:02d}.jpg"))
        kernel = np.ones((3, 3), np.uint8)
        bg_edges = {h: cv2.dilate(edges(bg).astype(np.uint8), kernel) > 0 for h, bg in bgs.items()}

        masks = zone_masks(zones[cam], W, H)
        union = np.logical_or.reduce(masks)
        scores = np.zeros(len(fr))
        for i in range(len(fr)):
            h = int(hours[i])
            blobs, valid = foreground(stack[i], bgs[h], bg_edges[h])
            u = union & valid
            row = {"image": fr.at[i, "image"], "camera_id": cam,
                   "bg_score": round(float((blobs & u).sum() / max(u.sum(), 1)), 4),
                   "overexposed": round(float(1 - u.sum() / max(union.sum(), 1)), 4)}
            for k, m in enumerate(masks, start=1):
                mv = m & valid
                row[f"bg_z{k}"] = round(float((blobs & mv).sum() / max(mv.sum(), 1)), 4)
            rows.append(row)
            scores[i] = row["bg_score"]

        if args.examples and len(fr):
            for q in np.linspace(0.1, 0.99, args.examples):
                i = int(np.argsort(scores)[int(q * (len(fr) - 1))])
                h = int(hours[i])
                blobs, _ = foreground(stack[i], bgs[h], bg_edges[h])
                save_example(args.out / "background_examples" / f"{cam}_q{int(q * 100):02d}.jpg",
                             stack[i], bgs[h], blobs, masks, scores[i])
        print(f"{cam}: {len(fr)} frames ({missing} missing), median score {np.median(scores):.3f}, "
              f"max {scores.max():.3f}")

    out = pd.DataFrame(rows)
    zone_cols = sorted([c for c in out.columns if c.startswith("bg_z")], key=lambda c: int(c[4:]))
    out = out[["image", "camera_id", "bg_score", "overexposed"] + zone_cols]
    out.to_csv(args.out / "voter_background.csv", index=False)
    print(f"Wrote {len(out)} frames -> {args.out / 'voter_background.csv'}")


if __name__ == "__main__":
    main()
