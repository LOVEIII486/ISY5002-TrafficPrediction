"""在标注集的 val 划分上评估一个或多个权重，口径统一、可直接比较。

    python -m src.detection.evaluate --weights yolo/yolo26m.pt results/detection/finetune/m960_pseudo/weights/best.pt

为什么不直接用 ultralytics 的 `model.val()`：COCO 权重有 80 类、类别编号与本标注集
（4 类）对不上，val() 没法把两者放在同一把尺子上。这里统一按类名映射到
car / motorcycle / bus / truck 后自己算，COCO 权重与微调权重走完全相同的代码。

报告三组数：

- mAP@0.5、mAP@0.5:0.95（COCO 的 101 点插值）与各类 AP@0.5；
- 检测管线默认阈值 0.25 下的精确率 / 召回率；
- 每张图的计数误差 |预测车辆数 − 标注车辆数|，计数正是下游关心的量。

每组都按白天 / 夜间拆开，夜间是检测器最弱的地方。
产物 `results/detection/finetune/eval/<名字>.json` 与汇总表 `comparison.csv`。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .run_detect import VEHICLE_CLASSES, load_class_map, pick_device

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
OPERATING_CONF = 0.25  # 与 run_detect 的默认阈值一致
EVAL_DIR = paths.FINETUNE_RESULTS / "eval"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两组 xyxy 框的两两 IoU，形状 (len(a), len(b))。"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def load_ground_truth(split: pd.DataFrame) -> dict[str, dict]:
    """读 val 的标签：{图片名: {"cls": (n,), "xywhn": (n, 4)}}。"""
    truth = {}
    for image in split["image"]:
        label = (paths.LABELED / "labels" / "val" / image).with_suffix(".txt")
        rows = np.loadtxt(label, ndmin=2) if label.is_file() and label.stat().st_size else np.zeros((0, 5))
        truth[image] = {"cls": rows[:, 0].astype(int), "xywhn": rows[:, 1:5]}
    return truth


def to_xyxy(xywhn: np.ndarray, width: int, height: int) -> np.ndarray:
    cx, cy, w, h = (xywhn[:, i] for i in range(4))
    return np.stack([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height], axis=1)


def predict(weights: Path, split: pd.DataFrame, imgsz: int, examples: int) -> tuple[dict, dict]:
    """逐张推理，返回 {图片名: {"cls", "conf", "xyxy"}} 与 {图片名: (宽, 高)}。"""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    class_map = load_class_map(model.names)
    to_index = {name: index for index, name in enumerate(VEHICLE_CLASSES)}
    device = pick_device()
    example_dir = EVAL_DIR / f"{weights_label(weights)}@{imgsz}" / "examples"
    example_images: set[str] = set()
    if examples:
        example_dir.mkdir(parents=True, exist_ok=True)
        for _period, images in split.groupby("period")["image"]:  # 每个时段等距取 examples 张
            example_images.update(images.iloc[:: max(1, len(images) // examples)].head(examples))

    predictions, sizes = {}, {}
    for image in split["image"]:
        result = model.predict(
            str(paths.LABELED / "images" / "val" / image),
            imgsz=imgsz,
            conf=0.001,  # 低阈值才能画出完整的 PR 曲线
            max_det=300,
            classes=sorted(class_map),
            device=device,
            verbose=False,
        )[0]
        height, width = result.orig_shape
        sizes[image] = (width, height)
        boxes = result.boxes
        predictions[image] = {
            "cls": np.array([to_index[class_map[int(c)]] for c in boxes.cls.tolist()], dtype=int),
            "conf": boxes.conf.cpu().numpy(),
            "xyxy": boxes.xyxy.cpu().numpy(),
        }
        if image in example_images:
            result.update(boxes=boxes.data[boxes.conf >= OPERATING_CONF])
            result.save(filename=str(example_dir / image))
    return predictions, sizes


def weights_label(weights: Path) -> str:
    """权重的显示名：微调产物用实验目录名，其余用文件名。"""
    return weights.parent.parent.name if weights.parent.name == "weights" else weights.stem


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO 的 101 点插值 AP。"""
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0, 1, 101)
    index = np.searchsorted(recall, points, side="left")
    return float(np.mean([envelope[i] if i < len(envelope) else 0.0 for i in index]))


def detection_metrics(images: list[str], truth: dict, predictions: dict, sizes: dict) -> dict:
    """在给定图片子集上算 AP 与阈值 0.25 下的 P / R / 计数误差。"""
    ap = np.full((len(VEHICLE_CLASSES), len(IOU_THRESHOLDS)), np.nan)
    gt_per_class = np.zeros(len(VEHICLE_CLASSES), dtype=int)
    for cls in range(len(VEHICLE_CLASSES)):
        records = []  # (conf, 每个 IoU 阈值下是否 TP)
        n_gt = 0
        for image in images:
            gt_boxes = to_xyxy(truth[image]["xywhn"][truth[image]["cls"] == cls], *sizes[image])
            pred = predictions[image]
            mask = pred["cls"] == cls
            conf, boxes = pred["conf"][mask], pred["xyxy"][mask]
            order = np.argsort(-conf)
            conf, boxes = conf[order], boxes[order]
            n_gt += len(gt_boxes)
            ious = iou_matrix(boxes, gt_boxes)
            hits = np.zeros((len(boxes), len(IOU_THRESHOLDS)), dtype=bool)
            for t, threshold in enumerate(IOU_THRESHOLDS):
                taken = np.zeros(len(gt_boxes), dtype=bool)
                for p in range(len(boxes)):  # 置信度从高到低贪心匹配
                    if not len(gt_boxes):
                        break
                    candidates = np.where(~taken & (ious[p] >= threshold))[0]
                    if len(candidates):
                        best = candidates[np.argmax(ious[p, candidates])]
                        taken[best] = True
                        hits[p, t] = True
            records.extend(zip(conf, hits))
        gt_per_class[cls] = n_gt
        if n_gt == 0:
            continue
        if not records:
            ap[cls] = 0.0
            continue
        records.sort(key=lambda item: -item[0])
        hits = np.array([hit for _conf, hit in records])
        tp = np.cumsum(hits, axis=0)
        fp = np.cumsum(~hits, axis=0)
        for t in range(len(IOU_THRESHOLDS)):
            recall = tp[:, t] / n_gt
            precision = tp[:, t] / np.maximum(tp[:, t] + fp[:, t], 1)
            ap[cls, t] = average_precision(recall, precision)

    # 阈值 0.25 下的 P / R（IoU 0.5，类别必须一致）与计数误差
    tp = fp = fn = 0
    count_errors = []
    for image in images:
        pred = predictions[image]
        keep = pred["conf"] >= OPERATING_CONF
        p_cls, p_box = pred["cls"][keep], pred["xyxy"][keep]
        g_cls, g_box = truth[image]["cls"], to_xyxy(truth[image]["xywhn"], *sizes[image])
        count_errors.append(abs(len(p_cls) - len(g_cls)))
        for cls in range(len(VEHICLE_CLASSES)):
            pb, gb = p_box[p_cls == cls], g_box[g_cls == cls]
            ious = iou_matrix(pb, gb)
            taken = np.zeros(len(gb), dtype=bool)
            matched = 0
            for p in range(len(pb)):
                candidates = np.where(~taken & (ious[p] >= 0.5))[0] if len(gb) else []
                if len(candidates):
                    taken[candidates[np.argmax(ious[p, candidates])]] = True
                    matched += 1
            tp += matched
            fp += len(pb) - matched
            fn += len(gb) - matched

    valid = ~np.isnan(ap[:, 0])
    return {
        "images": len(images),
        "gt_boxes": {name: int(n) for name, n in zip(VEHICLE_CLASSES, gt_per_class)},
        "mAP50": float(np.nanmean(ap[valid, 0])) if valid.any() else None,
        "mAP50_95": float(np.nanmean(ap[valid])) if valid.any() else None,
        "AP50": {name: (None if np.isnan(ap[i, 0]) else float(ap[i, 0])) for i, name in enumerate(VEHICLE_CLASSES)},
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "count_MAE": float(np.mean(count_errors)) if count_errors else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="在 val 划分上评估权重")
    parser.add_argument("--weights", nargs="+", type=lambda value: paths.ROOT / value, required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--examples", type=int, default=4, help="每个时段存几张带框的示例图，0 不存")
    args = parser.parse_args()

    split = pd.read_csv(paths.SPLIT_FILE, dtype={"camera_id": str})
    split = split[split["split"] == "val"].reset_index(drop=True)
    truth = load_ground_truth(split)
    # prelabel.json 只在标签来自教师模型时存在；人工标注或外部数据集没有它，按真值对待。
    note_file = paths.LABELED / "prelabel.json"
    labels_note = json.loads(note_file.read_text(encoding="utf-8")).get("labels_are", "") if note_file.is_file() else ""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for weights in args.weights:
        # 同一权重换推理尺寸结果不同，尺寸进名字，免得互相覆盖。
        label = f"{weights_label(weights)}@{args.imgsz}"
        print(f"评估 {label}（{weights.name}）· {len(split)} 张 val", flush=True)
        predictions, sizes = predict(weights, split, args.imgsz, args.examples)
        shown = weights.relative_to(paths.ROOT) if weights.is_relative_to(paths.ROOT) else weights
        report = {"weights": str(shown), "imgsz": args.imgsz, "ground_truth": labels_note}
        for period, subset in [("all", split), *split.groupby("period")]:
            report[period] = detection_metrics(subset["image"].tolist(), truth, predictions, sizes)
            m = report[period]
            rows.append(
                {
                    "model": label,
                    "period": period,
                    "mAP50": m["mAP50"],
                    "mAP50_95": m["mAP50_95"],
                    **{f"AP50_{name}": value for name, value in m["AP50"].items()},
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "count_MAE": m["count_MAE"],
                }
            )
        (EVAL_DIR / f"{label}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总表按模型名增量合并：分几次评估（如不同 imgsz）也能落进同一张表。
    table = pd.DataFrame(rows)
    summary = EVAL_DIR / "comparison.csv"
    if summary.is_file():
        previous = pd.read_csv(summary)
        table = pd.concat([previous[~previous["model"].isin(table["model"])], table], ignore_index=True)
    table.to_csv(summary, index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    if "pseudo" in labels_note:
        print("\n注意：val 标签是教师模型的伪标签，以上数字表示「学得像不像教师」，不是准确率。")
    print(f"产物 → {EVAL_DIR}")


if __name__ == "__main__":
    main()
