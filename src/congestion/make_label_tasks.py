"""Human congestion labels: pick the frames to label and build the labelling page.

    python -m src.congestion.make_label_tasks
    python -m src.congestion.make_label_tasks --practice 25 --test 50 --seed 20261023

Then open results/congestion/labeling/label_tool.html in Chrome or Edge.

Sampling (per camera, fixed seed, reproducible):
- frames are grouped into five time strata (Singapore time) and drawn at random inside each stratum;
  daytime and peak hours are over-sampled so that congested frames actually appear in the sample;
- the frames are never chosen by any voter's score, so the labels can judge every voter fairly;
- each camera gets 25 practice frames (used to choose thresholds, weights and the CLIP crop) and
  50 test frames (used once, at the end); the two sets do not overlap;
- every camera x stratum also gets spare frames: when a frame is skipped as impossible to judge
  (rain, fog, glare), the page replaces it with the next spare from the same camera and stratum,
  so the planned number of labels per camera and stratum is kept.

Strata (Singapore time; weekday = Monday-Friday):
  weekday_peak     weekday 07:00-09:59 and 16:00-18:59
  weekday_offpeak  weekday 10:00-15:59
  weekend_day      Saturday / Sunday 07:00-18:59
  evening          every day 19:00-23:59
  late_night       every day 00:00-06:59

Outputs (results/congestion/labeling/):
- tasks.json: the slots to label, the candidate frames, the strata sizes and the task hash;
- label_tool.html: the labelling page (images are loaded from datasets/lta by relative path,
  so keep the page inside the repository).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.camera_filter import CameraFilter

from .common import image_index, load_frames

STRATA = ["weekday_peak", "weekday_offpeak", "weekend_day", "evening", "late_night"]
STRATA_DEF = {"weekday_peak": "Mon-Fri 07:00-09:59 and 16:00-18:59", "weekday_offpeak": "Mon-Fri 10:00-15:59",
              "weekend_day": "Sat-Sun 07:00-18:59", "evening": "every day 19:00-23:59", "late_night": "every day 00:00-06:59"}
SHARE = {"weekday_peak": 0.28, "weekday_offpeak": 0.20, "weekend_day": 0.24, "evening": 0.16, "late_night": 0.12}
TEMPLATE = Path(__file__).with_name("label_tool_template.html")


def stratum(weekday: pd.Series, hour: pd.Series) -> pd.Series:
    out = np.where(hour < 7, "late_night",
          np.where(hour >= 19, "evening",
          np.where(weekday >= 5, "weekend_day",
          np.where((hour < 10) | (hour >= 16), "weekday_peak", "weekday_offpeak"))))
    return pd.Series(out, index=weekday.index)


def quotas(total: int) -> dict[str, int]:
    """Split `total` over the strata by SHARE, rounding by largest remainder."""
    raw = {s: total * SHARE[s] for s in STRATA}
    q = {s: int(math.floor(v)) for s, v in raw.items()}
    for s in sorted(STRATA, key=lambda s: raw[s] - q[s], reverse=True)[: total - sum(q.values())]:
        q[s] += 1
    return q


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick frames for human congestion labels and build the labelling page")
    ap.add_argument("--images", type=Path, default=Path("datasets/lta"))
    ap.add_argument("--frames", type=Path, default=Path("results/detection/frames.csv"))
    ap.add_argument("--config", type=Path, default=Path("src/camera_filter/camera_filter.json"))
    ap.add_argument("--out", type=Path, default=Path("results/congestion/labeling"))
    ap.add_argument("--practice", type=int, default=25, help="practice frames per camera")
    ap.add_argument("--test", type=int, default=50, help="test frames per camera")
    ap.add_argument("--spare", type=float, default=0.5, help="spare frames per stratum, as a share of its quota (at least 5)")
    ap.add_argument("--seed", type=int, default=20261023)
    args = ap.parse_args()

    frames = load_frames(args.frames)
    index = image_index(args.images)
    frames = frames[frames["image"].isin(index)].copy()
    frames["stratum"] = stratum(frames["weekday"], frames["hour"])
    flt = CameraFilter.from_json(args.config)
    raw_cfg = json.loads(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    qp, qt = quotas(args.practice), quotas(args.test)
    cameras, pools, slots, sizes = {}, {}, {"practice": [], "test": []}, {}
    for cam in sorted(frames["camera_id"].unique()):
        rules = flt.cameras.get(cam)
        if rules is None or not rules.zone:
            print(f"{cam}: no detection zone in the config, skipped")
            continue
        cameras[cam] = {"name": raw_cfg["cameras"][cam].get("name", cam),
                        "zones": [np.round(z, 4).tolist() for z in rules.zone]}
        rng = np.random.default_rng([args.seed, int(cam)])
        fc = frames[frames["camera_id"] == cam]
        sizes[cam] = fc["stratum"].value_counts().reindex(STRATA, fill_value=0).astype(int).to_dict()
        for s in STRATA:
            names = sorted(fc.loc[fc["stratum"] == s, "image"])
            need = qp[s] + qt[s]
            spare = max(5, math.ceil(args.spare * need))
            if len(names) < need:
                raise SystemExit(f"{cam} {s}: only {len(names)} frames for {need} slots")
            pick = [names[i] for i in rng.permutation(len(names))[: need + spare]]
            key = f"{cam}|{s}"
            pools[key] = [{"image": n, "path": Path(os.path.relpath(index[n], args.out)).as_posix()} for n in pick]
            for i in range(qp[s]):
                slots["practice"].append({"split": "practice", "camera": cam, "stratum": s, "pool": key, "first": i})
            for i in range(qt[s]):
                slots["test"].append({"split": "test", "camera": cam, "stratum": s, "pool": key, "first": qp[s] + i})

    rng = np.random.default_rng(args.seed)
    order = []
    for split in ("practice", "test"):  # practice first; cameras and strata mixed within each set
        order += [slots[split][i] for i in rng.permutation(len(slots[split]))]
    for k, slot in enumerate(order):
        slot["id"] = k

    body = {"slots": order, "pools": pools}
    task_hash = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    data = {"format": "congestion_label_tasks/v1", "task_hash": task_hash, "seed": args.seed,
            "levels": ["free", "moderate", "congested"],
            "strata": {"definition": STRATA_DEF, "timezone": "Asia/Singapore",
                       "share": SHARE, "quota_practice": qp, "quota_test": qt, "frames_per_camera": sizes},
            "config": str(args.config.as_posix()), "cameras": cameras, **body}
    (args.out / "tasks.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (args.out / "label_tool.html").write_text(html, encoding="utf-8")

    n_spare = sum(len(p) for p in pools.values()) - len(order)
    print(f"Per camera: practice {qp}, test {qt}")
    print(f"{len(cameras)} cameras, {sum(s['split'] == 'practice' for s in order)} practice + "
          f"{sum(s['split'] == 'test' for s in order)} test slots, {n_spare} spare frames; task hash {task_hash}")
    print(f"Open {args.out / 'label_tool.html'} in Chrome or Edge")


if __name__ == "__main__":
    main()
