"""Draft road regions and detection zones for each camera from two weeks of detection positions.

These are only drafts; a person adjusts the vertices in the calibration page. Approach:
- Road region: density map of box bottom-centres -> smooth and threshold -> drop small blobs
  (isolated fixed false positives) -> morphological closing to fill gaps between lanes
  -> dilate slightly to cover the kerb -> simplify the contour into a polygon.
- Detection zone: within the road region, the area that is "near" (tall boxes) and often occupied
  during busy frames; three variants are offered (near full width / queue area / near-mid-far segments),
  simplified to 4-6 vertices and kept away from the image border.
"""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

GW, GH = 192, 108  # grid: each cell is 10 x 10 pixels at 1920 x 1080


def _grid(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    gx = np.clip((x * GW).astype(int), 0, GW - 1)
    gy = np.clip((y * GH).astype(int), 0, GH - 1)
    g = np.zeros((GH, GW), np.float32)
    np.add.at(g, (gy, gx), 1 if weights is None else weights)
    return g


def _polys(mask: np.ndarray, max_polys: int, min_area_frac: float, eps_frac: float, max_pts: int) -> list[list[list[float]]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    out = []
    for c in contours[:max_polys]:
        if cv2.contourArea(c) < min_area_frac * GW * GH:
            continue
        eps = eps_frac * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        while len(approx) > max_pts:
            eps *= 1.3
            approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) < 3:
            continue
        pts = []
        for px, py in approx[:, 0, :]:
            x, y = (px + 0.5) / GW, (py + 0.5) / GH
            x = 0.0 if x < 0.012 else 1.0 if x > 0.988 else round(float(x), 4)
            y = 0.0 if y < 0.012 else 1.0 if y > 0.988 else round(float(y), 4)
            pts.append([x, y])
        out.append(pts)
    return out


def road_mask(d: pd.DataFrame) -> np.ndarray:
    x = ((d.x1 + d.x2) / 2).to_numpy()
    y = d.y2.to_numpy()
    g = _grid(x, y)
    g = np.log1p(cv2.GaussianBlur(g, (0, 0), 1.5))
    occupied = g[g > 0]
    thr = np.quantile(occupied, 0.35) if len(occupied) else 1
    m = (g >= thr).astype(np.uint8)
    # drop isolated small blobs (fixed false positives on signs and lights), then fill the gaps between lanes
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= 0.004 * GW * GH:
            keep[lab == i] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))  # grow by about 20 px to cover the kerb
    # fill holes: keep only the outer contours and fill them
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    filled = np.zeros_like(keep)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled


def _hull_on_road(mask: np.ndarray, road: np.ndarray) -> np.ndarray:
    """Convex hull of a region intersected with the road, giving a regular shape; keep clear of the image border."""
    pts = cv2.findNonZero(mask.astype(np.uint8))
    out = np.zeros_like(road)
    if pts is None or len(pts) < 3:
        return out
    cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    out &= road
    out[:, :5] = 0; out[:, -5:] = 0; out[-4:, :] = 0  # vehicles at the border are only half visible
    return out


def _largest(mask: np.ndarray) -> np.ndarray:
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    return (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def _fields(d: pd.DataFrame, road: np.ndarray):
    """Mean box height map (near vehicles are taller) and vehicle density during busy frames."""
    x = ((d.x1 + d.x2) / 2).to_numpy()
    y = d.y2.to_numpy()
    cnt = _grid(x, y)
    hmean = cv2.GaussianBlur(_grid(x, y, ((d.y2 - d.y1) * 1080).to_numpy()), (0, 0), 3) / np.maximum(
        cv2.GaussianBlur(cnt, (0, 0), 3), 1e-6)
    on_road = (road > 0) & (cv2.GaussianBlur(cnt, (0, 0), 3) > 0.05)
    per_frame = d.groupby("frame").size()
    b = d[d.frame.isin(per_frame[per_frame >= per_frame.quantile(0.8)].index)]  # the 20% of frames with the most vehicles
    busy = cv2.GaussianBlur(_grid(((b.x1 + b.x2) / 2).to_numpy(), b.y2.to_numpy()), (0, 0), 3.0)
    return hmean, on_road, busy


def zone_options(d: pd.DataFrame, road: np.ndarray) -> list[dict]:
    hmean, on_road, busy = _fields(d, road)
    if not on_road.any():
        return []
    near_thr = max(30.0, float(np.quantile(hmean[on_road], 0.6)))
    near = on_road & (hmean >= near_thr)
    close = lambda m, k: cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    opts = []

    # A near, full width: every lane in the near section that vehicles regularly use
    a = _largest(close(near & (busy >= np.quantile(busy[near], 0.2)), 15)) if near.any() else None
    if a is not None and a.any():
        opts.append(("near", "A Near, full width", "All lanes in the near section. Vehicles are large and clear, so the detector is most reliable here, but both directions may be included.", [_hull_on_road(a, road)]))

    # B queue area: the near section where vehicles often stand during busy periods
    q = _largest(close(near & (busy >= np.quantile(busy[near], 0.5)), 15)) if near.any() else None
    if q is not None and q.any():
        opts.append(("queue", "B Queue area", "The near section where vehicles often stand during busy periods. Closest to \"congested or not\"; recommended as the default.", [_hull_on_road(q, road)]))

    # C three segments: cut the road containing B into near / mid / far parts and see how far the queue reaches
    base = q if (q is not None and q.any()) else (a if a is not None else None)
    if base is not None and base.any():
        busy_road = close(on_road & (busy >= np.quantile(busy[on_road], 0.5)) & (hmean >= 18), 9)
        n, lab, st, _ = cv2.connectedComponentsWithStats(busy_road, 8)
        overlap = [((lab == i) & (base > 0)).sum() for i in range(n)]
        lane = (lab == int(np.argmax(overlap[1:])) + 1).astype(np.uint8) if n > 1 else busy_road
        ys, xs = np.nonzero(lane)
        if len(xs) > 30:
            P = np.stack([xs, ys], 1).astype(float)
            axis = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]
            t = (P - P.mean(0)) @ axis
            if np.corrcoef(t, hmean[ys, xs])[0, 1] < 0:  # orient t so that larger means nearer (taller boxes)
                t = -t
            lo, hi = np.quantile(t, [0.03, 0.97])
            cuts = np.linspace(hi, lo, 4)  # near -> mid -> far
            segs = []
            for k in range(3):
                sel = (t <= cuts[k]) & (t >= cuts[k + 1])
                m = np.zeros_like(lane)
                m[ys[sel], xs[sel]] = 1
                segs.append(_hull_on_road(close(m, 7), road))
            if all(sg.any() for sg in segs):
                opts.append(("segments", "C Three segments (near / mid / far)", "Three segments along the queue direction (zone 1 = nearest). How far the queue reaches indicates how congested the road is, a layout commonly used for queue-length detection in traffic engineering. The far segment has small vehicles that the detector often misses.", segs))

    out = []
    for key, name, desc, masks in opts:
        polys = []
        for m in masks:
            polys += _polys(m, max_polys=1, min_area_frac=0.001, eps_frac=0.02, max_pts=6)
        if polys:
            out.append({"key": key, "name": name, "desc": desc, "zones": polys})
    return out


def suggest(d: pd.DataFrame) -> dict:
    road = road_mask(d)
    options = zone_options(d, road)
    default = next((o for o in options if o["key"] == "queue"), options[0] if options else {"zones": []})
    return {
        "road": _polys(road, max_polys=4, min_area_frac=0.01, eps_frac=0.008, max_pts=18),
        "zone": default["zones"],
        "zone_options": options,
    }
