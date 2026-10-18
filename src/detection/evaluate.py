"""在标注集的某个划分上评估一个或多个权重，口径统一、可直接比较。

    python -m src.detection.evaluate --weights yolo/yolo26m.pt results/detection/finetune/m960_pseudo/weights/best.pt
    python -m src.detection.evaluate --data datasets/miotcd_yolo_v2 --split test --weights ... --imgsz 736

为什么不直接用 ultralytics 的 `model.val()`：COCO 权重有 80 类、类别编号与本标注集（4 类）对不上，
直接 val() 不会自动重映射真值编号。这里统一按类名映射到 car / motorcycle / bus / truck 后自己算，
COCO 权重与微调权重走完全相同的代码。AP 采用 COCO 的 101 点插值、按置信度贪心匹配，
但没有 crowd / ignore 区域处理，也不保证与 ultralytics 或官方 COCO 评估器的数值逐位相同。

报告的指标：

- mAP@0.5、mAP@0.5:0.95 与各类 AP@0.5；
- 阈值 0.25（与 run_detect 一致）、IoU 0.5 下的微平均精确率 / 召回率（四类合并计数，不是 ultralytics 的汇总 P/R）；
- 计数：每张图 |预测数 − 标注数| 的均值（总数口径，漏检与误检会互相抵消）、带符号的平均偏差，
  以及各类各自的计数误差。

split.csv 带 period 列（LTA 的白天 / 夜间）时每组再按它拆开。
标签来自教师模型（prelabel.json 标明 pseudo）时，指标只表示与教师预测的一致程度，不是准确率。
产物 `results/detection/finetune/eval/<数据集>/<划分>/<名字>.json` 与汇总表 `comparison.csv`。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .frozen import VERSIONED_PREFIXES, check_eval_split
from .run_detect import VEHICLE_CLASSES, load_class_map, pick_device

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
OPERATING_CONF = 0.25  # 与 run_detect 的默认阈值一致


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
    union = area_a[:, None] + area_b[None, :] - inter
    # 只对零面积并集单独处理；正常框用真实并集，IoU 恰为 0.5 时不会被 epsilon 压到阈值以下
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)


def canonical_order(boxes: np.ndarray, conf: np.ndarray | None = None) -> np.ndarray:
    """确定的排序：先置信度降序（给了 conf 时），并列再按坐标 x1、y1、x2、y2 升序。"""
    keys = [boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0]]
    if conf is not None:
        keys.append(-conf)
    return np.lexsort(keys) if len(boxes) else np.zeros(0, dtype=int)


def greedy_match(conf: np.ndarray, pred_boxes: np.ndarray, gt_boxes: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """单图单类：按置信度从高到低，每个预测匹配 IoU 最大且未被占用、IoU ≥ 阈值的真值。

    返回 (排序后的置信度, 每个预测是否 TP)。预测按 (置信度降序, 坐标) 排序，真值按坐标排序，
    并列置信度、并列 IoU 都有确定的次序，结果与预测、真值的输入顺序无关。
    """
    order = canonical_order(pred_boxes, conf)
    conf, pred_boxes = conf[order], pred_boxes[order]
    gt_boxes = gt_boxes[canonical_order(gt_boxes)]
    hits = np.zeros(len(pred_boxes), dtype=bool)
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return conf, hits
    ious = iou_matrix(pred_boxes, gt_boxes)
    taken = np.zeros(len(gt_boxes), dtype=bool)
    for p in range(len(pred_boxes)):
        candidates = np.flatnonzero(~taken & (ious[p] >= threshold))
        if len(candidates):
            taken[candidates[np.argmax(ious[p, candidates])]] = True
            hits[p] = True
    return conf, hits


def load_ground_truth(data_dir: Path, split_name: str, images: list[str]) -> dict[str, dict]:
    """读标签：{图片名: {"cls": (n,), "xywhn": (n, 4)}}。缺文件直接报错；存在的空文件才代表纯背景。"""
    truth, missing = {}, []
    for image in images:
        label = (data_dir / "labels" / split_name / image).with_suffix(".txt")
        if not label.is_file():
            missing.append(label)
            continue
        rows = np.loadtxt(label, ndmin=2) if label.stat().st_size else np.zeros((0, 5))
        truth[image] = {"cls": rows[:, 0].astype(int), "xywhn": rows[:, 1:5]}
    if missing:
        raise SystemExit(f"{len(missing)} 张图缺标签文件（不会当成背景），例如 {missing[0]}")
    return truth


def to_xyxy(xywhn: np.ndarray, width: int, height: int) -> np.ndarray:
    cx, cy, w, h = (xywhn[:, i] for i in range(4))
    return np.stack([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height], axis=1)


def predict(weights: Path, data_dir: Path, split_name: str, eval_dir: Path, split: pd.DataFrame, imgsz: int, examples: int) -> tuple[dict, dict]:
    """逐张推理，返回 {图片名: {"cls", "conf", "xyxy"}} 与 {图片名: (宽, 高)}。"""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    class_map = load_class_map(model.names)
    to_index = {name: index for index, name in enumerate(VEHICLE_CLASSES)}
    device = pick_device()
    example_dir = eval_dir / f"{weights_label(weights)}@{imgsz}" / "examples"
    example_images: set[str] = set()
    if examples:
        example_dir.mkdir(parents=True, exist_ok=True)
        for _group, images in groups_of(split):  # 每个分组等距取 examples 张
            example_images.update(images["image"].iloc[:: max(1, len(images) // examples)].head(examples))

    predictions, sizes = {}, {}
    total = len(split)
    for index, image in enumerate(split["image"], start=1):
        result = model.predict(
            str(data_dir / "images" / split_name / image),
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
        if index % 500 == 0 or index == total:
            print(f"    {index}/{total}", flush=True)
    return predictions, sizes


def groups_of(split: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """评估分组：整体一组；split.csv 有 period 列（白天 / 夜间）时再各加一组。"""
    groups = [("all", split)]
    if "period" in split.columns:
        groups += list(split.groupby("period"))
    return groups


def weights_label(weights: Path) -> str:
    """权重的显示名：微调产物用「实验目录名」，last.pt 另加后缀；其余用文件名。"""
    if weights.parent.name == "weights":
        return weights.parent.parent.name + ("" if weights.stem == "best" else f"-{weights.stem}")
    return weights.stem


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO 的 101 点插值 AP。"""
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0, 1, 101)
    index = np.searchsorted(recall, points, side="left")
    return float(np.mean([envelope[i] if i < len(envelope) else 0.0 for i in index]))


def detection_metrics(images: list[str], truth: dict, predictions: dict, sizes: dict) -> dict:
    """在给定图片子集上算 AP、阈值 0.25 下的 P / R 与计数误差。

    跨图片汇总时按图片名排序处理，再对置信度做稳定排序：并列分数的先后由（图片名, 图内坐标次序）决定，
    与调用方传入的图片顺序无关。
    """
    images = sorted(images)
    n_classes = len(VEHICLE_CLASSES)
    ap = np.full((n_classes, len(IOU_THRESHOLDS)), np.nan)
    gt_per_class = np.zeros(n_classes, dtype=int)
    for cls in range(n_classes):
        confs, hits_by_threshold = [], [[] for _ in IOU_THRESHOLDS]
        for image in images:
            gt_boxes = to_xyxy(truth[image]["xywhn"][truth[image]["cls"] == cls], *sizes[image])
            pred = predictions[image]
            mask = pred["cls"] == cls
            gt_per_class[cls] += len(gt_boxes)
            for t, threshold in enumerate(IOU_THRESHOLDS):
                conf, hits = greedy_match(pred["conf"][mask], pred["xyxy"][mask], gt_boxes, threshold)
                hits_by_threshold[t].append(hits)
            confs.append(conf)
        n_gt = gt_per_class[cls]
        if n_gt == 0:
            continue
        conf = np.concatenate(confs)
        if len(conf) == 0:
            ap[cls] = 0.0
            continue
        order = np.argsort(-conf, kind="stable")
        for t in range(len(IOU_THRESHOLDS)):
            hits = np.concatenate(hits_by_threshold[t])[order]
            tp, fp = np.cumsum(hits), np.cumsum(~hits)
            ap[cls, t] = average_precision(tp / n_gt, tp / np.maximum(tp + fp, 1))

    # 阈值 0.25、IoU 0.5 下的 P / R（类别必须一致），复用同一个匹配函数
    tp = fp = fn = 0
    count_errors, count_bias = [], []
    per_class_errors = np.zeros((len(images), n_classes))
    for row, image in enumerate(images):
        pred = predictions[image]
        keep = pred["conf"] >= OPERATING_CONF
        p_cls, p_conf, p_box = pred["cls"][keep], pred["conf"][keep], pred["xyxy"][keep]
        g_cls, g_box = truth[image]["cls"], to_xyxy(truth[image]["xywhn"], *sizes[image])
        count_errors.append(abs(len(p_cls) - len(g_cls)))
        count_bias.append(len(p_cls) - len(g_cls))
        for cls in range(n_classes):
            in_pred, in_gt = p_cls == cls, g_cls == cls
            _conf, hits = greedy_match(p_conf[in_pred], p_box[in_pred], g_box[in_gt], 0.5)
            matched = int(hits.sum())
            tp += matched
            fp += int(in_pred.sum()) - matched
            fn += int(in_gt.sum()) - matched
            per_class_errors[row, cls] = abs(int(in_pred.sum()) - int(in_gt.sum()))

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
        "count_bias": float(np.mean(count_bias)) if count_bias else None,
        "count_MAE_per_class": {name: float(per_class_errors[:, i].mean()) for i, name in enumerate(VEHICLE_CLASSES)} if images else None,
    }


def check_frozen(data_dir: Path, split_name: str, split: pd.DataFrame) -> None:
    """评估前核对：split.csv 与清单 <划分>.txt 一致；版本化目录再核对冻结记录。"""
    listing = data_dir / f"{split_name}.txt"
    if not listing.is_file():
        if data_dir.name.startswith(VERSIONED_PREFIXES):
            raise SystemExit(f"{data_dir.name} 是版本化数据目录，却没有清单 {listing.name}，拒绝评估")
        return
    listed = sorted(Path(line).name for line in listing.read_text(encoding="utf-8").splitlines())
    if listed != sorted(split["image"]):
        raise SystemExit(f"{listing.name} 与 split.csv 的 {split_name} 行不一致，数据可能被改过")
    check_eval_split(data_dir, split_name)  # 版本化目录：清单与标签摘要必须与冻结记录一致


def main() -> None:
    parser = argparse.ArgumentParser(description="在标注集的某个划分上评估权重")
    parser.add_argument("--weights", nargs="+", type=lambda value: paths.ROOT / value, required=True)
    parser.add_argument("--data", type=lambda value: paths.ROOT / value, default=paths.LABELED, help="YOLO 数据目录（含 split.csv、labels/<划分>）")
    parser.add_argument("--split", default="val", help="评估哪个划分：LTA 用 val；MIO-TCD v2 用 dev / test / test_moto")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--examples", type=int, default=4, help="每个分组存几张带框的示例图，0 不存")
    args = parser.parse_args()

    data_dir: Path = args.data
    split = pd.read_csv(data_dir / "split.csv", dtype={"camera_id": str, "image": str})
    split = split[split["split"] == args.split].reset_index(drop=True)
    if split.empty:
        raise SystemExit(f"{data_dir / 'split.csv'} 里没有 split == {args.split!r} 的行")
    check_frozen(data_dir, args.split, split)
    truth = load_ground_truth(data_dir, args.split, split["image"].tolist())
    # 标签来源说明：prelabel.json（教师伪标签）或 dataset.json（外部真值）；都没有就留空。
    labels_note = ""
    for note_name in ("prelabel.json", "dataset.json"):
        note_file = data_dir / note_name
        if note_file.is_file():
            labels_note = json.loads(note_file.read_text(encoding="utf-8")).get("labels_are", "")
            break
    # 每个数据集、每个划分各自一个评估目录，dev 与 test 的结果不会混在一起。
    eval_dir = paths.FINETUNE_RESULTS / "eval" / data_dir.name / args.split
    eval_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for weights in args.weights:
        # 同一权重换推理尺寸结果不同，尺寸进名字，免得互相覆盖。
        label = f"{weights_label(weights)}@{args.imgsz}"
        print(f"评估 {label}（{weights.name}）· {data_dir.name}/{args.split} · {len(split)} 张", flush=True)
        predictions, sizes = predict(weights, data_dir, args.split, eval_dir, split, args.imgsz, args.examples)
        shown = weights.relative_to(paths.ROOT) if weights.is_relative_to(paths.ROOT) else weights
        report = {"weights": str(shown), "data": data_dir.name, "split": args.split, "imgsz": args.imgsz, "ground_truth": labels_note}
        for period, subset in groups_of(split):
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
                    "count_bias": m["count_bias"],
                }
            )
        (eval_dir / f"{label}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 汇总表按模型名增量合并：分几次评估（如不同 imgsz）也能落进同一张表。
    table = pd.DataFrame(rows)
    summary = eval_dir / "comparison.csv"
    if summary.is_file():
        previous = pd.read_csv(summary)
        table = pd.concat([previous[~previous["model"].isin(table["model"])], table], ignore_index=True)
    table.to_csv(summary, index=False)
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    if "pseudo" in labels_note:
        print("\n注意：标签是教师模型的伪标签，以上数字只表示与教师预测的一致程度，不是准确率。")
    print(f"产物 → {eval_dir}")


if __name__ == "__main__":
    main()
