"""Compare precision of the raw detector output and the camera-filtered output on the
human-labelled Singapore test images.

    python -m src.camera_filter.evaluate --labels lta_annotation_export_final.json

- Ground truth: the Label Studio export of task A (63 images). Raw and filtered output are scored
  against **the same** ground truth.
- Predictions: the boxes for these images in detections.csv (joint model, confidence >= 0.25).
- Matching: class-agnostic; a prediction matches a ground-truth box at IoU >= 0.5, greedily in
  descending confidence order, and each ground-truth box can be matched once.
- Metrics: precision, recall, F1, false positives and the per-image count error. Removed boxes are
  split, using the raw matching, into "correctly removed (was a false positive)" and
  "removed a true vehicle".
- Intervals: image-level bootstrap with 2,000 resamples gives 95% intervals for the change.

The test images were not used for calibration (the calibration page's points exclude them),
so this is an independent check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import KEEP, CameraFilter


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def match(pred: np.ndarray, conf: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """For each prediction, whether it matches a ground-truth box (True = correct)."""
    ok = np.zeros(len(pred), dtype=bool)
    if len(pred) == 0 or len(gt) == 0:
        return ok
    m = iou_matrix(pred, gt)
    used = np.zeros(len(gt), dtype=bool)
    for i in np.argsort(-conf):
        cand = np.where(~used & (m[i] >= thr))[0]
        if len(cand):
            j = cand[np.argmax(m[i, cand])]
            used[j] = True
            ok[i] = True
    return ok


def load_labels(path: Path) -> list[dict]:
    tasks = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for t in tasks:
        data = t["data"]
        ann = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        boxes = []
        for r in (ann[-1]["result"] if ann else []):
            if r.get("type") != "rectanglelabels":
                continue
            v = r["value"]
            boxes.append([v["x"] / 100, v["y"] / 100, (v["x"] + v["width"]) / 100, (v["y"] + v["height"]) / 100])
        out.append({"image": data["image_name"], "camera_id": str(data["camera_id"]), "period": data.get("period"),
                    "batch": data.get("batch"), "gt": np.asarray(boxes, dtype=float).reshape(-1, 4)})
    return out


def summarize(df: pd.DataFrame) -> dict:
    s = df[["gt", "pred_raw", "tp_raw", "pred_flt", "tp_flt", "removed_fp", "removed_tp"]].sum()
    p_raw, p_flt = s.tp_raw / max(s.pred_raw, 1), s.tp_flt / max(s.pred_flt, 1)
    r_raw, r_flt = s.tp_raw / max(s["gt"], 1), s.tp_flt / max(s["gt"], 1)
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0
    return {
        "images": len(df), "gt": int(s["gt"]),
        "pred_raw": int(s.pred_raw), "pred_filtered": int(s.pred_flt),
        "fp_raw": int(s.pred_raw - s.tp_raw), "fp_filtered": int(s.pred_flt - s.tp_flt),
        "removed_fp": int(s.removed_fp), "removed_tp": int(s.removed_tp),
        "precision_raw": round(p_raw, 4), "precision_filtered": round(p_flt, 4),
        "recall_raw": round(r_raw, 4), "recall_filtered": round(r_flt, 4),
        "f1_raw": round(f1(p_raw, r_raw), 4), "f1_filtered": round(f1(p_flt, r_flt), 4),
        "count_mae_raw": round(float((df.pred_raw - df["gt"]).abs().mean()), 3),
        "count_mae_filtered": round(float((df.pred_flt - df["gt"]).abs().mean()), 3),
    }


def bootstrap(df: pd.DataFrame, n: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    a = df[["gt", "pred_raw", "tp_raw", "pred_flt", "tp_flt"]].to_numpy(dtype=float)
    dp, dr = [], []
    for _ in range(n):
        s = a[rng.integers(0, len(a), len(a))].sum(0)
        dp.append(s[4] / max(s[3], 1) - s[2] / max(s[1], 1))
        dr.append(s[4] / max(s[0], 1) - s[2] / max(s[0], 1))
    q = lambda v: [round(float(np.quantile(v, 0.025)), 4), round(float(np.quantile(v, 0.975)), 4)]
    return {"precision_delta_95ci": q(dp), "recall_delta_95ci": q(dr)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Camera filter: before / after comparison on the human-labelled test set")
    ap.add_argument("--labels", type=Path, required=True, help="Label Studio export JSON (ground truth)")
    ap.add_argument("--detections", type=Path, default=Path("results/detection/detections.csv"))
    ap.add_argument("--config", type=Path, default=Path("src/camera_filter/camera_filter.json"))
    ap.add_argument("--out", type=Path, default=Path("results/camera_filter"))
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    flt = CameraFilter.from_json(args.config)
    det = pd.read_csv(args.detections, dtype={"camera_id": str})
    tasks = load_labels(args.labels)
    det = det[det.image.isin({t["image"] for t in tasks})].copy()
    det["reason"] = flt.reasons(det)

    rows, boxes = [], []
    for t in tasks:
        p = det[det.image == t["image"]]
        pb, pc = p[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float), p.conf.to_numpy(dtype=float)
        ok_raw = match(pb, pc, t["gt"], args.iou)
        keep = (p.reason == KEEP).to_numpy()
        ok_flt = match(pb[keep], pc[keep], t["gt"], args.iou)
        rows.append({"image": t["image"], "camera_id": t["camera_id"], "period": t["period"], "batch": t["batch"],
                     "gt": len(t["gt"]), "pred_raw": len(pb), "tp_raw": int(ok_raw.sum()),
                     "pred_flt": int(keep.sum()), "tp_flt": int(ok_flt.sum()),
                     "removed_fp": int((~keep & ~ok_raw).sum()), "removed_tp": int((~keep & ok_raw).sum())})
        for (_, r), ok, kp in zip(p.iterrows(), ok_raw, keep):
            if not kp:
                boxes.append({**r.to_dict(), "was_correct": bool(ok)})
    per_image = pd.DataFrame(rows)

    report = {"overall": {**summarize(per_image), **bootstrap(per_image)},
              "by_period": {k: summarize(g) for k, g in per_image.groupby("period")},
              "by_batch": {k: summarize(g) for k, g in per_image.groupby("batch")},
              "settings": {"iou": args.iou, "labels": str(args.labels), "config": str(args.config),
                           "class_agnostic": True}}
    by_cam = pd.DataFrame([{"camera_id": k, **summarize(g)} for k, g in per_image.groupby("camera_id")])

    args.out.mkdir(parents=True, exist_ok=True)
    per_image.to_csv(args.out / "eval_per_image.csv", index=False)
    by_cam.to_csv(args.out / "eval_by_camera.csv", index=False)
    pd.DataFrame(boxes).to_csv(args.out / "eval_removed_boxes.csv", index=False)
    (args.out / "eval_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    o = report["overall"]
    print(f"{o['images']} test images, {o['gt']} ground-truth boxes")
    print(f"Precision {o['precision_raw']:.3f} -> {o['precision_filtered']:.3f}  (95% CI of change {o['precision_delta_95ci']})")
    print(f"Recall    {o['recall_raw']:.3f} -> {o['recall_filtered']:.3f}  (95% CI of change {o['recall_delta_95ci']})")
    print(f"False positives {o['fp_raw']} -> {o['fp_filtered']}; of the removed boxes, {o['removed_fp']} were false positives "
          f"and {o['removed_tp']} matched a ground-truth vehicle")
    print(f"Per-image count error (MAE) {o['count_mae_raw']} -> {o['count_mae_filtered']}")
    print(by_cam[["camera_id", "images", "pred_raw", "removed_fp", "removed_tp", "precision_raw", "precision_filtered",
                  "recall_raw", "recall_filtered"]].to_string(index=False))


if __name__ == "__main__":
    main()
