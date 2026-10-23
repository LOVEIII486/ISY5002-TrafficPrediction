"""Voter A for congestion: box occupancy inside each camera's detection zones.

    python -m src.congestion.occupancy
    python -m src.congestion.occupancy --detections results/detection/detections.csv \
        --config src/camera_filter/camera_filter.json --out results/congestion

For every frame in frames.csv (frames with no detections score 0):
1. drop fixed false positives with the camera filter (road region + exclusion set);
2. assign each remaining box to a detection zone by its bottom-centre;
3. per zone, count vehicles and compute the occupancy: the area of the union of the zone's boxes,
   clipped to the zone, divided by the zone area (rasterised on a 384 x 216 grid, 5 px per cell
   at 1920 x 1080).

Boxes are image-plane rectangles, so occupancy is a relative measure within a camera.

Output: results/congestion/voter_occupancy.csv, one row per frame:
image, camera_id, captured_at_utc, captured_at_sgt, weekday (0 = Monday), hour, is_day,
n_vehicles, n_heavy (bus + truck), pcu, occupancy (all zones together), and per zone k:
n_z{k}, pcu_z{k}, occ_z{k}.

`pcu` weights vehicles by Transport for London passenger car units (motorcycle 0.4, car 1.0, bus 2.0,
heavy goods vehicle 2.3), fixed in advance; override with --pcu motorcycle=0.5,truck=2.0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.camera_filter import CameraFilter
from src.camera_filter.core import KEEP

GW, GH = 384, 216  # raster grid for occupancy
HEAVY = {"bus", "truck"}
# Passenger car units (Transport for London urban values). Our "truck" class mixes medium and heavy
# goods vehicles; the HGV value is used.
PCU = {"motorcycle": 0.4, "car": 1.0, "bus": 2.0, "truck": 2.3}


def zone_masks(polys: list[np.ndarray]) -> list[np.ndarray]:
    masks = []
    for poly in polys:
        m = np.zeros((GH, GW), np.uint8)
        pts = np.round(poly * [GW, GH]).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
        masks.append(m.astype(bool))
    return masks


def box_mask(boxes: np.ndarray) -> np.ndarray:
    m = np.zeros((GH, GW), bool)
    for x1, y1, x2, y2 in boxes:
        c1, r1 = int(np.floor(x1 * GW)), int(np.floor(y1 * GH))
        c2, r2 = int(np.ceil(x2 * GW)), int(np.ceil(y2 * GH))
        m[max(r1, 0):min(r2, GH), max(c1, 0):min(c2, GW)] = True
    return m


def score_camera(frames: pd.DataFrame, det: pd.DataFrame, zones: list[np.ndarray], pcu: dict[str, float]) -> pd.DataFrame:
    masks = zone_masks(zones)
    union = np.logical_or.reduce(masks) if masks else np.zeros((GH, GW), bool)
    areas = [m.sum() for m in masks]
    by_image = {k: g for k, g in det.groupby("image")}
    rows = []
    for image in frames["image"]:
        g = by_image.get(image)
        row = {"image": image}
        if g is None or len(g) == 0:
            row.update(n_vehicles=0, n_heavy=0, pcu=0.0, occupancy=0.0)
            for k in range(1, len(masks) + 1):
                row[f"n_z{k}"], row[f"pcu_z{k}"], row[f"occ_z{k}"] = 0, 0.0, 0.0
            rows.append(row)
            continue
        boxes = g[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        zone = g["zone"].to_numpy()
        weight = g["cls"].map(pcu).fillna(1.0).to_numpy()
        row["n_vehicles"] = int(len(g))
        row["n_heavy"] = int(g["cls"].isin(HEAVY).sum())
        row["pcu"] = round(float(weight.sum()), 2)
        row["occupancy"] = round(float((box_mask(boxes) & union).sum() / max(union.sum(), 1)), 4)
        for k, (m, area) in enumerate(zip(masks, areas), start=1):
            sel = zone == k
            row[f"n_z{k}"] = int(sel.sum())
            row[f"pcu_z{k}"] = round(float(weight[sel].sum()), 2)
            row[f"occ_z{k}"] = round(float((box_mask(boxes[sel]) & m).sum() / max(area, 1)), 4) if sel.any() else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Congestion voter A: box occupancy in the detection zones")
    ap.add_argument("--detections", type=Path, default=Path("results/detection/detections.csv"))
    ap.add_argument("--frames", type=Path, default=None, help="frames.csv; defaults to the detections folder")
    ap.add_argument("--config", type=Path, default=Path("src/camera_filter/camera_filter.json"))
    ap.add_argument("--out", type=Path, default=Path("results/congestion"))
    ap.add_argument("--pcu", default="", help="override PCU weights, e.g. motorcycle=0.5,truck=2.0")
    args = ap.parse_args()
    pcu = dict(PCU)
    for item in filter(None, args.pcu.split(",")):
        name, value = item.split("=")
        pcu[name.strip()] = float(value)

    flt = CameraFilter.from_json(args.config)
    det = pd.read_csv(args.detections, dtype={"camera_id": str})
    frames = pd.read_csv(args.frames or args.detections.with_name("frames.csv"), dtype={"camera_id": str})

    det = det[flt.reasons(det) == KEEP].copy()
    det["zone"] = flt.zone_index(det)
    det = det[det["zone"] > 0]

    parts = []
    for cam in sorted(frames["camera_id"].unique()):
        rules = flt.cameras.get(cam)
        if rules is None or not rules.zone:
            print(f"{cam}: no detection zone in the config, skipped")
            continue
        fr = frames[frames["camera_id"] == cam]
        scored = score_camera(fr, det[det["camera_id"] == cam], rules.zone, pcu)
        parts.append(fr.merge(scored, on="image", how="left"))
        print(f"{cam}: {len(fr)} frames, {len(rules.zone)} zone(s), "
              f"median occupancy {scored['occupancy'].median():.3f}, max {scored['occupancy'].max():.3f}")

    out = pd.concat(parts, ignore_index=True)
    sgt = pd.to_datetime(out["captured_at_utc"], utc=True).dt.tz_convert("Asia/Singapore")
    out.insert(3, "captured_at_sgt", sgt.dt.strftime("%Y-%m-%d %H:%M"))
    out.insert(4, "weekday", sgt.dt.weekday)
    out.insert(5, "hour", sgt.dt.hour)
    out.insert(6, "is_day", flt.is_day(out["captured_at_utc"]))
    order = {"n_z": 0, "pcu_z": 1, "occ_z": 2}
    zone_cols = sorted([c for c in out.columns if c.startswith(tuple(order))],
                       key=lambda c: (int(c.split("_z")[-1]), order[c.split("_z")[0] + "_z"]))
    out = out[[c for c in out.columns if c not in zone_cols] + zone_cols]
    args.out.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out / "voter_occupancy.csv", index=False)
    print(f"Wrote {len(out)} frames -> {args.out / 'voter_occupancy.csv'}")


if __name__ == "__main__":
    main()
